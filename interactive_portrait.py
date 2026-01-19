#!/usr/bin/env python3
"""
Spectramotus Interactive Portrait Player
Plays LivePortrait clips based on MediaPipe gesture detection
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
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 720
WEBCAM_PREVIEW_SIZE = 20  # Size of webcam preview in corner

# Gesture to clip mapping
GESTURE_MAP = {
    "thumbs_up": "smile",
    "wave": "sup_dude",
    "pointing_left": "look_left",
    "pointing_right": "look_right",
    "none": "idle"  # Default when no gesture detected
}

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
        
        # Thumbs up: thumb extended, other fingers curled
        if (thumb_tip.y < index_mcp.y and 
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y):
            return "thumbs_up"
        
        # Wave: hand raised with fingers extended (similar to open palm but hand is raised)
        # Check if most fingers are extended and hand is raised (wrist below fingertips)
        fingers_extended = (
            index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y < ring_mcp.y and
            pinky_tip.y < pinky_mcp.y
        )
        hand_raised = wrist.y > middle_tip.y  # Wrist is below fingertips when hand is raised
        
        if fingers_extended and hand_raised:
            return "wave"
        
        # Pointing up: only index extended
        if (index_tip.y < index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y):
            return "pointing_up"
        
        # Open palm: all fingers extended (but hand not necessarily raised)
        if (index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y < ring_mcp.y and
            pinky_tip.y < pinky_mcp.y):
            return "open_palm"
        
        return "none"

class VideoPlayer:
    """Plays video clips with seamless looping"""
    
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30
        self.last_frame_time = 0
        
        # Track video completion
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            # Loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def reset(self):
        """Reset to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.has_completed_cycle = False
        self.started_playing = True
        self.last_frame_time = 0  # Reset timing
    
    def has_finished(self):
        """Check if video has completed at least one full cycle"""
        return self.has_completed_cycle
    
    def release(self):
        self.cap.release()

class InteractivePortrait:
    """Main application"""
    
    def __init__(self, clips_dir, photo_name):
        self.clips_dir = clips_dir
        self.photo_name = photo_name
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption(f"Interactive Portrait - {photo_name}")
        
        # Load all video clips
        self.clips = self._load_clips()
        self.current_clip_name = "idle"
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize gesture detector
        self.gesture_detector = GestureDetector()
        
        # State
        self.running = True
        self.show_webcam_preview = True
        self.clip_locked = False  # Lock to prevent switching until clip finishes
        
    def _load_clips(self):
        """Load all video clips"""
        clips = {}
        
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
        
        return clips
    
    def switch_clip(self, new_clip_name):
        """Switch to a different clip - only if current clip has finished or switching to idle"""
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
        # 1. Current clip is idle (always allow switching from idle)
        # 2. Current clip has finished one complete cycle
        can_switch = (
            self.current_clip_name == "idle" or
            current_clip.has_finished()
        )
        
        if can_switch:
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = True  # Lock the new clip until it finishes
            print(f"→ Switched to: {new_clip_name}")
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        
        print("\n" + "="*60)
        print("INTERACTIVE PORTRAIT RUNNING")
        print("="*60)
        print("Controls:")
        print("  Thumbs up → Smile")
        print("  Wave → Sup dude")
        print("  Pointing up → Look left")
        print("  Open palm → Look right")
        print("  Press 'W' to toggle webcam preview")
        print("  Press 'Q' or ESC to quit")
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
            
            # Capture webcam frame
            ret, webcam_frame = self.webcam.read()
            if not ret:
                continue
            
            # Detect gesture
            gesture = self.gesture_detector.detect(webcam_frame)
            
            # Map gesture to clip
            target_clip = GESTURE_MAP.get(gesture, "idle")
            
            # Try to switch clip (switch_clip will check if switching is allowed)
            self.switch_clip(target_clip)
            
            # Get current portrait frame
            portrait_frame = self.clips[self.current_clip_name].get_frame()
            
            # Check if current clip has finished (for non-idle clips)
            # This unlocks the clip so a new gesture can trigger a switch
            if self.clip_locked and self.current_clip_name != "idle":
                if self.clips[self.current_clip_name].has_finished():
                    self.clip_locked = False  # Unlock to allow next switch
            
            if portrait_frame is not None:
                # Resize to display size
                portrait_frame = cv2.resize(
                    portrait_frame, 
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                )
                
                # Convert BGR to RGB for pygame
                portrait_frame = cv2.cvtColor(portrait_frame, cv2.COLOR_BGR2RGB)
                
                # Rotate for pygame (OpenCV uses different coordinate system)
                portrait_frame = pygame.surfarray.make_surface(
                    portrait_frame.swapaxes(0, 1)
                )
                
                # Display portrait
                self.screen.blit(portrait_frame, (0, 0))
                
                # Overlay webcam preview in corner
                if self.show_webcam_preview:
                    webcam_small = cv2.resize(
                        webcam_frame, 
                        (WEBCAM_PREVIEW_SIZE, int(WEBCAM_PREVIEW_SIZE * 0.75))
                    )
                    webcam_small = cv2.cvtColor(webcam_small, cv2.COLOR_BGR2RGB)
                    webcam_surface = pygame.surfarray.make_surface(
                        webcam_small.swapaxes(0, 1)
                    )
                    self.screen.blit(
                        webcam_surface, 
                        (DISPLAY_WIDTH - WEBCAM_PREVIEW_SIZE - 10, 10)
                    )
                    
                    # Show current gesture text
                    font = pygame.font.Font(None, 36)
                    text = font.render(
                        f"Gesture: {gesture}", 
                        True, 
                        (255, 255, 255)
                    )
                    self.screen.blit(text, (10, 10))
                
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