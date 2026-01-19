#!/usr/bin/env python3
"""
Spectramotus Interactive Portrait Player - Fixed Version
Plays LivePortrait clips based on MediaPipe gesture detection

Improvements:
- Dynamic aspect ratio display
- Fixed gesture mapping (only uses gestures that actually work)
- Webcam preview off by default
- Better visual feedback
- Cleaner code structure
"""

import cv2
import mediapipe as mp
import pygame
import sys
import os
from pathlib import Path
import time

# Configuration
PHOTO_NAME = "karol-photo"
CLIPS_DIR = f"generated_clips/{PHOTO_NAME}"

# Display settings (will be calculated dynamically)
DISPLAY_WIDTH = None
DISPLAY_HEIGHT = None
WEBCAM_PREVIEW_SIZE = 200

# Gesture to clip mapping - ONLY gestures that actually work
GESTURE_MAP = {
    "thumbs_up": "smile",
    "wave": "sup_dude",
    "pointing_left": "look_left",
    "pointing_right": "look_right",
    "none": "idle"
}

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (100, 100, 255)

class GestureDetector:
    """Detects hand gestures using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture state tracking for debouncing
        self.current_gesture = "none"
        self.gesture_frames = 0
        self.required_frames = 3  # Must see gesture for 3 frames
        
    def detect(self, frame):
        """Detect gesture in frame, returns gesture name"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected = "none"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame (for debugging)
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            # Detect specific gestures based on landmark positions
            detected = self._classify_gesture(hand_landmarks)
        
        # Debouncing: require gesture to be stable for multiple frames
        if detected == self.current_gesture:
            self.gesture_frames += 1
        else:
            self.current_gesture = detected
            self.gesture_frames = 1
        
        # Only return gesture if it's been stable
        if self.gesture_frames >= self.required_frames:
            return self.current_gesture
        else:
            return "none"
    
    def _classify_gesture(self, landmarks):
        """Classify hand pose into gesture"""
        # Get landmark positions
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        wrist = landmarks.landmark[0]
        
        # Get finger base positions (MCP joints)
        index_mcp = landmarks.landmark[5]
        middle_mcp = landmarks.landmark[9]
        ring_mcp = landmarks.landmark[13]
        pinky_mcp = landmarks.landmark[17]
        
        # Check if fingers are extended
        fingers_extended = (
            index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y < ring_mcp.y and
            pinky_tip.y < pinky_mcp.y
        )
        
        # Thumbs up: thumb extended upward, other fingers curled
        if (thumb_tip.y < index_mcp.y and 
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y):
            return "thumbs_up"
        
        # Pointing gestures: only index finger extended
        index_extended_alone = (
            index_tip.y < index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
        )
        
        if index_extended_alone:
            # Determine direction based on where index finger is pointing
            # Compare index tip position relative to wrist
            horizontal_diff = index_tip.x - wrist.x
            
            # Pointing left: index finger is to the LEFT of wrist (from camera view)
            # Note: In camera view, user's left is image right (mirrored)
            if horizontal_diff > 0.1:  # Significant distance to the right in image = user pointing left
                return "pointing_left"
            # Pointing right: index finger is to the RIGHT of wrist (from camera view)
            elif horizontal_diff < -0.1:  # Significant distance to the left in image = user pointing right
                return "pointing_right"
            else:
                # Pointing straight up or ambiguous
                return "pointing_up"
        
        # Now check open hand gestures
        if fingers_extended:
            # Calculate hand orientation - is it raised up?
            # If wrist is significantly below middle fingertip, hand is raised
            hand_raised = (wrist.y - middle_tip.y) > 0.1
            
            if hand_raised:
                # Wave: open hand raised up
                return "wave"
            else:
                # Open palm: open hand flat/forward (not raised)
                return "open_palm"
        
        return "none"

class VideoPlayer:
    """Plays video clips with seamless looping"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30
        self.last_frame_time = 0
        
        # Track video completion
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.has_completed_cycle = False
        self.started_playing = False
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
    
    def get_frame(self):
        """Get current frame, loop if at end"""
        current_time = time.time()
        
        # Respect frame timing
        if current_time - self.last_frame_time < self.frame_delay:
            return None
        
        self.last_frame_time = current_time
        
        ret, frame = self.cap.read()
        
        if not ret:
            # Video has completed one full cycle
            self.has_completed_cycle = True
            self.current_frame = 0
            # Loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        else:
            self.current_frame += 1
        
        return frame if ret else None
    
    def reset(self):
        """Reset to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.has_completed_cycle = False
        self.started_playing = True
        self.current_frame = 0
        self.last_frame_time = 0
    
    def has_finished(self):
        """Check if video has completed at least one full cycle"""
        return self.has_completed_cycle
    
    def get_progress(self):
        """Get playback progress as percentage"""
        if self.total_frames == 0:
            return 0
        return (self.current_frame / self.total_frames) * 100
    
    def release(self):
        self.cap.release()

class InteractivePortrait:
    """Main application"""
    
    def __init__(self, clips_dir, photo_name):
        self.clips_dir = clips_dir
        self.photo_name = photo_name
        
        # Load all video clips first to get dimensions
        self.clips = self._load_clips()
        
        # Get video dimensions from idle clip
        idle_clip = self.clips["idle"]
        video_width = int(idle_clip.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(idle_clip.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate display size to maintain aspect ratio
        max_dimension = 1080  # Max screen dimension
        aspect_ratio = video_width / video_height
        
        if aspect_ratio > 1:  # Wider than tall
            self.display_width = min(video_width, max_dimension)
            self.display_height = int(self.display_width / aspect_ratio)
        else:  # Taller than wide or square
            self.display_height = min(video_height, max_dimension)
            self.display_width = int(self.display_height * aspect_ratio)
        
        print(f"Video resolution: {video_width}x{video_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption(f"Interactive Portrait - {photo_name}")
        
        self.current_clip_name = "idle"
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize gesture detector
        self.gesture_detector = GestureDetector()
        
        # State
        self.running = True
        self.show_webcam_preview = False  # OFF by default
        self.show_ui = True  # Show gesture/clip info
        self.clip_locked = False
        self.fullscreen = False
        
        # Font for UI
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def _load_clips(self):
        """Load all video clips"""
        clips = {}
        
        print(f"\nLoading clips from: {self.clips_dir}")
        print("="*60)
        
        for gesture, clip_name in GESTURE_MAP.items():
            video_path = os.path.join(
                self.clips_dir, 
                f"{self.photo_name}_{clip_name}.mp4"
            )
            
            if os.path.exists(video_path):
                clips[clip_name] = VideoPlayer(video_path)
                print(f"✓ Loaded: {clip_name}")
            else:
                print(f"✗ Missing: {video_path}")
        
        if "idle" not in clips:
            raise ValueError("idle clip is required but not found!")
        
        print("="*60 + "\n")
        return clips
    
    def switch_clip(self, new_clip_name):
        """Switch to a different clip"""
        if new_clip_name not in self.clips:
            return
        
        # Don't switch if it's the same clip
        if new_clip_name == self.current_clip_name:
            return
        
        # Always allow switching to idle
        if new_clip_name == "idle":
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = False
            print(f"→ Switched to: {new_clip_name}")
            return
        
        # For non-idle clips, check if we can switch
        current_clip = self.clips[self.current_clip_name]
        
        # Allow switching if:
        # 1. Current clip is idle
        # 2. Current clip has finished one complete cycle
        can_switch = (
            self.current_clip_name == "idle" or
            current_clip.has_finished()
        )
        
        if can_switch:
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = True
            print(f"→ Switched to: {new_clip_name}")
    
    def draw_ui(self, gesture, webcam_frame):
        """Draw UI overlay with gesture and clip info"""
        if not self.show_ui:
            return
        
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.display_width, 100))
        overlay.set_alpha(180)
        overlay.fill(COLOR_BLACK)
        
        # Draw current clip name
        clip_text = self.font_medium.render(
            f"Clip: {self.current_clip_name}", 
            True, 
            COLOR_GREEN if self.current_clip_name != "idle" else COLOR_WHITE
        )
        overlay.blit(clip_text, (10, 10))
        
        # Draw detected gesture
        gesture_color = COLOR_YELLOW if gesture != "none" else COLOR_WHITE
        gesture_text = self.font_medium.render(
            f"Gesture: {gesture}", 
            True, 
            gesture_color
        )
        overlay.blit(gesture_text, (10, 50))
        
        # Draw progress bar if not idle
        if self.current_clip_name != "idle":
            progress = self.clips[self.current_clip_name].get_progress()
            bar_width = 200
            bar_height = 10
            bar_x = self.display_width - bar_width - 10
            bar_y = 45
            
            # Background
            pygame.draw.rect(overlay, COLOR_WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
            # Progress
            pygame.draw.rect(overlay, COLOR_BLUE, (bar_x, bar_y, int(bar_width * progress / 100), bar_height))
        
        self.screen.blit(overlay, (0, 0))
        
        # Draw webcam preview if enabled
        if self.show_webcam_preview and webcam_frame is not None:
            webcam_small = cv2.resize(
                webcam_frame, 
                (WEBCAM_PREVIEW_SIZE, int(WEBCAM_PREVIEW_SIZE * 0.75))
            )
            webcam_small = cv2.cvtColor(webcam_small, cv2.COLOR_BGR2RGB)
            webcam_surface = pygame.surfarray.make_surface(
                webcam_small.swapaxes(0, 1)
            )
            
            # Position in bottom-right corner
            self.screen.blit(
                webcam_surface, 
                (self.display_width - WEBCAM_PREVIEW_SIZE - 10, 
                 self.display_height - int(WEBCAM_PREVIEW_SIZE * 0.75) - 10)
            )
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        last_triggered_gesture = "none"
        
        print("\n" + "="*60)
        print("INTERACTIVE PORTRAIT RUNNING")
        print("="*60)
        print("Gestures (trigger once, animation plays to completion):")
        print("  👍 Thumbs up → Smile")
        print("  👋 Wave (raised hand) → Sup dude")
        print("  👈 Point left → Look left")
        print("  👉 Point right → Look right")
        print("\nControls:")
        print("  W → Toggle webcam preview")
        print("  U → Toggle UI overlay")
        print("  F → Toggle fullscreen")
        print("  Q/ESC → Quit")
        print("="*60 + "\n")
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_w:
                        self.show_webcam_preview = not self.show_webcam_preview
                        print(f"Webcam preview: {'ON' if self.show_webcam_preview else 'OFF'}")
                    elif event.key == pygame.K_u:
                        self.show_ui = not self.show_ui
                        print(f"UI overlay: {'ON' if self.show_ui else 'OFF'}")
                    elif event.key == pygame.K_f:
                        self.toggle_fullscreen()
                        print(f"Fullscreen: {'ON' if self.fullscreen else 'OFF'}")
            
            # Capture webcam frame
            ret, webcam_frame = self.webcam.read()
            if not ret:
                continue
            
            # Detect gesture
            gesture = self.gesture_detector.detect(webcam_frame)
            
            # Only trigger new animation if:
            # 1. A gesture is detected (not "none")
            # 2. It's different from the last triggered gesture (prevents re-triggering)
            # 3. We're not currently locked in an animation OR we're in idle
            if gesture != "none" and gesture != last_triggered_gesture:
                if not self.clip_locked or self.current_clip_name == "idle":
                    target_clip = GESTURE_MAP.get(gesture, "idle")
                    self.switch_clip(target_clip)
                    last_triggered_gesture = gesture
            
            # Reset last triggered gesture when hand is removed
            if gesture == "none":
                last_triggered_gesture = "none"
            
            # Check if current clip has finished
            if self.clip_locked and self.current_clip_name != "idle":
                if self.clips[self.current_clip_name].has_finished():
                    # Animation finished, return to idle
                    self.clip_locked = False
                    self.switch_clip("idle")
            
            # Get current portrait frame
            portrait_frame = self.clips[self.current_clip_name].get_frame()
            
            if portrait_frame is not None:
                # Resize to display size maintaining aspect ratio
                portrait_frame = cv2.resize(
                    portrait_frame, 
                    (self.display_width, self.display_height)
                )
                
                # Convert BGR to RGB for pygame
                portrait_frame = cv2.cvtColor(portrait_frame, cv2.COLOR_BGR2RGB)
                
                # Rotate for pygame (OpenCV uses different coordinate system)
                portrait_frame = pygame.surfarray.make_surface(
                    portrait_frame.swapaxes(0, 1)
                )
                
                # Display portrait
                self.screen.blit(portrait_frame, (0, 0))
                
                # Draw UI overlay
                self.draw_ui(gesture, webcam_frame)
                
                pygame.display.flip()
            
            clock.tick(30)  # 30 FPS
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        for clip in self.clips.values():
            clip.release()
        self.webcam.release()
        pygame.quit()
        print("Goodbye!")

def main():
    if not os.path.exists(CLIPS_DIR):
        print(f"ERROR: Clips directory not found: {CLIPS_DIR}")
        print("Run batch_generate.py first to create clips!")
        sys.exit(1)
    
    app = InteractivePortrait(CLIPS_DIR, PHOTO_NAME)
    app.run()

if __name__ == "__main__":
    main()