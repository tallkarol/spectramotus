#!/usr/bin/env python3
"""
Spectramotus Interactive Portrait Player - Layered Version
Composites person animations over background with dynamic effects

Usage:
    python interactive_portrait_layered.py prepared-photos/karol-photo
"""

import cv2
import mediapipe as mp
import pygame
import sys
import os
import json
import math
from pathlib import Path
import time
import numpy as np
import argparse

# Display settings (calculated dynamically)
DISPLAY_WIDTH = None
DISPLAY_HEIGHT = None
WEBCAM_PREVIEW_SIZE = 200

# Gesture to clip mapping
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
        
        # Gesture state tracking
        self.current_gesture = "none"
        self.gesture_frames = 0
        self.required_frames = 3
        
    def detect(self, frame):
        """Detect gesture in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected = "none"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            detected = self._classify_gesture(hand_landmarks)
        
        # Debouncing
        if detected == self.current_gesture:
            self.gesture_frames += 1
        else:
            self.current_gesture = detected
            self.gesture_frames = 1
        
        if self.gesture_frames >= self.required_frames:
            return self.current_gesture
        else:
            return "none"
    
    def _classify_gesture(self, landmarks):
        """Classify hand pose into gesture"""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        wrist = landmarks.landmark[0]
        
        index_mcp = landmarks.landmark[5]
        middle_mcp = landmarks.landmark[9]
        ring_mcp = landmarks.landmark[13]
        pinky_mcp = landmarks.landmark[17]
        
        fingers_extended = (
            index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y < ring_mcp.y and
            pinky_tip.y < pinky_mcp.y
        )
        
        # Thumbs up
        if (thumb_tip.y < index_mcp.y and 
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y):
            return "thumbs_up"
        
        # Pointing gestures
        index_extended_alone = (
            index_tip.y < index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
        )
        
        if index_extended_alone:
            horizontal_diff = index_tip.x - wrist.x
            if horizontal_diff > 0.1:
                return "pointing_left"
            elif horizontal_diff < -0.1:
                return "pointing_right"
            else:
                return "pointing_up"
        
        # Wave vs open palm
        if fingers_extended:
            hand_raised = (wrist.y - middle_tip.y) > 0.1
            if hand_raised:
                return "wave"
            else:
                return "open_palm"
        
        return "none"

class VideoPlayer:
    """Plays video clips with alpha channel support"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30
        self.last_frame_time = 0
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.has_completed_cycle = False
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
    
    def get_frame(self):
        """Get current frame"""
        current_time = time.time()
        
        if current_time - self.last_frame_time < self.frame_delay:
            return None
        
        self.last_frame_time = current_time
        ret, frame = self.cap.read()
        
        if not ret:
            self.has_completed_cycle = True
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        else:
            self.current_frame += 1
        
        return frame if ret else None
    
    def reset(self):
        """Reset to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.has_completed_cycle = False
        self.current_frame = 0
        self.last_frame_time = 0
    
    def has_finished(self):
        """Check if completed one cycle"""
        return self.has_completed_cycle
    
    def get_progress(self):
        """Get progress percentage"""
        if self.total_frames == 0:
            return 0
        return (self.current_frame / self.total_frames) * 100
    
    def release(self):
        self.cap.release()

class BackgroundEffects:
    """Handles background animation effects"""
    
    def __init__(self, background_image):
        self.original_bg = background_image.copy()
        self.height, self.width = background_image.shape[:2]
        
        # Effect parameters
        self.time = 0
        self.breathing_speed = 0.5  # Cycles per second
        self.breathing_amplitude = 0.02  # 2% zoom
        self.parallax_amplitude = 5  # Pixels
        
        # Time of day effect
        self.time_of_day_cycle = 60.0  # Seconds for full day cycle
        
    def apply_effects(self, effect_list):
        """Apply selected effects to background"""
        result = self.original_bg.copy()
        self.time += 1/30  # Assuming 30 FPS
        
        if "breathing" in effect_list:
            result = self._apply_breathing(result)
        
        if "parallax" in effect_list:
            result = self._apply_parallax(result)
        
        if "time_of_day" in effect_list:
            result = self._apply_time_of_day(result)
        
        return result
    
    def _apply_breathing(self, image):
        """Subtle zoom in/out effect"""
        # Calculate scale factor (oscillates between 1.0 - amplitude and 1.0 + amplitude)
        scale = 1.0 + self.breathing_amplitude * math.sin(2 * math.pi * self.breathing_speed * self.time)
        
        # Calculate new dimensions
        new_width = int(self.width * scale)
        new_height = int(self.height * scale)
        
        # Resize
        resized = cv2.resize(image, (new_width, new_height))
        
        # Crop to original size (center crop)
        start_x = (new_width - self.width) // 2
        start_y = (new_height - self.height) // 2
        
        if scale > 1.0:
            # Zoomed in - crop
            result = resized[start_y:start_y+self.height, start_x:start_x+self.width]
        else:
            # Zoomed out - would need padding, just return original for now
            result = image
        
        return result
    
    def _apply_parallax(self, image):
        """Subtle horizontal/vertical shift"""
        # Calculate offset (oscillates)
        offset_x = int(self.parallax_amplitude * math.sin(2 * math.pi * 0.3 * self.time))
        offset_y = int(self.parallax_amplitude * 0.5 * math.cos(2 * math.pi * 0.2 * self.time))
        
        # Create translation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply translation
        result = cv2.warpAffine(image, M, (self.width, self.height), borderMode=cv2.BORDER_REPLICATE)
        
        return result
    
    def _apply_time_of_day(self, image):
        """Simulate time of day lighting changes"""
        # Calculate time of day (0 to 1, where 0.5 is noon)
        time_of_day = (self.time % self.time_of_day_cycle) / self.time_of_day_cycle
        
        # Calculate brightness factor (brightest at 0.5, darkest at 0 and 1)
        brightness = 0.8 + 0.2 * math.sin(2 * math.pi * time_of_day)
        
        # Calculate color temperature (warmer at dawn/dusk, cooler at noon)
        # More blue at noon, more orange at dawn/dusk
        temp = abs(time_of_day - 0.5) * 2  # 0 at noon, 1 at midnight
        
        # Apply brightness
        result = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Apply subtle color temperature shift
        if temp > 0.5:  # Dawn/dusk - warmer
            result[:, :, 2] = cv2.convertScaleAbs(result[:, :, 2], alpha=1.1, beta=0)  # More red
            result[:, :, 0] = cv2.convertScaleAbs(result[:, :, 0], alpha=0.9, beta=0)  # Less blue
        
        return result

class LayeredPortrait:
    """Main application with layered compositing"""
    
    def __init__(self, prepared_photo_dir):
        self.prepared_photo_dir = prepared_photo_dir
        self.photo_name = Path(prepared_photo_dir).name
        
        # Load metadata
        metadata_path = os.path.join(prepared_photo_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"animations": {"background_effects": []}}
        
        # Load background
        background_path = os.path.join(prepared_photo_dir, "background.png")
        if not os.path.exists(background_path):
            raise ValueError(f"Background not found: {background_path}")
        
        self.background_image = cv2.imread(background_path)
        self.bg_height, self.bg_width = self.background_image.shape[:2]
        
        # Load mask for compositing
        mask_path = os.path.join(prepared_photo_dir, "mask.png")
        if os.path.exists(mask_path):
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            # Store original mask dimensions
            self.mask_height, self.mask_width = mask_image.shape[:2]
            self.original_mask = mask_image
            print(f"Loaded mask: {self.mask_width}x{self.mask_height}")
        else:
            print(f"Warning: Mask not found at {mask_path}, using fallback threshold method")
            self.original_mask = None
        
        # Initialize background effects
        self.background_effects = BackgroundEffects(self.background_image)
        
        # Load person clips
        clips_dir = f"generated_clips/{self.photo_name}"
        if not os.path.exists(clips_dir):
            raise ValueError(f"Clips directory not found: {clips_dir}\nRun batch_generate_v2.py first!")
        
        self.clips = self._load_clips(clips_dir)
        
        # Calculate display size
        max_dimension = 1080
        aspect_ratio = self.bg_width / self.bg_height
        
        if aspect_ratio > 1:
            self.display_width = min(self.bg_width, max_dimension)
            self.display_height = int(self.display_width / aspect_ratio)
        else:
            self.display_height = min(self.bg_height, max_dimension)
            self.display_width = int(self.display_height * aspect_ratio)
        
        print(f"Background resolution: {self.bg_width}x{self.bg_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        
        # Pre-resize mask to display size for efficient compositing
        if self.original_mask is not None:
            self.display_mask = cv2.resize(
                self.original_mask,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_LINEAR
            )
            # Apply slight Gaussian blur to smooth edges and reduce meshing artifacts
            # This creates a soft feather at the edges for better blending
            self.display_mask = cv2.GaussianBlur(self.display_mask, (5, 5), 1.0)
            # Normalize mask to 0.0-1.0 range (255 = person, 0 = background)
            self.display_mask_float = (self.display_mask.astype(float) / 255.0)
            # Expand to 3 channels for BGR compositing
            self.display_mask_3d = np.stack([self.display_mask_float] * 3, axis=2)
            print(f"Pre-computed mask for display size (with edge smoothing)")
        else:
            self.display_mask_3d = None
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption(f"Layered Portrait - {self.photo_name}")
        
        self.current_clip_name = "idle"
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize gesture detector
        self.gesture_detector = GestureDetector()
        
        # State
        self.running = True
        self.show_webcam_preview = False
        self.show_ui = True
        self.clip_locked = False
        self.fullscreen = False
        
        # Background effects
        self.effects_enabled = self.metadata.get("animations", {}).get("background_effects", [])
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def _load_clips(self, clips_dir):
        """Load all video clips"""
        clips = {}
        
        print(f"\nLoading clips from: {clips_dir}")
        print("="*60)
        
        for gesture, clip_name in GESTURE_MAP.items():
            video_path = os.path.join(
                clips_dir,
                f"{self.photo_name}_{clip_name}_person.mp4"
            )
            
            if os.path.exists(video_path):
                clips[clip_name] = VideoPlayer(video_path)
                print(f"✓ Loaded: {clip_name}")
            else:
                print(f"✗ Missing: {video_path}")
        
        if "idle" not in clips:
            raise ValueError("idle clip is required!")
        
        print("="*60 + "\n")
        return clips
    
    def composite_frame(self, background, person_frame):
        """Composite person frame over background using chroma key detection"""
        
        # Resize both to display size
        bg_resized = cv2.resize(background, (self.display_width, self.display_height))
        person_resized = cv2.resize(person_frame, (self.display_width, self.display_height))
        
        # CHROMA KEY APPROACH: Videos have green (0, 255, 0) backgrounds.
        # Green is never present in person pixels, so we can reliably detect it.
        # This solves the black background vs dark person pixels problem.
        
        # Split into channels (BGR format)
        b, g, r = cv2.split(person_resized)
        
        # Detect chroma key green: high green, low red and blue
        # Allow some tolerance for video compression artifacts
        is_chroma_key = (g > 200) & (b < 50) & (r < 50)
        
        # Create alpha mask: 1.0 = person (keep), 0.0 = background (transparent)
        alpha = (~is_chroma_key).astype(float)
        
        # Apply slight blur to smooth edges and reduce harsh transitions
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)
        
        # Expand to 3 channels
        alpha_3d = np.stack([alpha] * 3, axis=2)
        
        # Composite: person where alpha=1.0, background where alpha=0.0
        result = (person_resized * alpha_3d + bg_resized * (1 - alpha_3d)).astype(np.uint8)
        
        return result
    
    def switch_clip(self, new_clip_name):
        """Switch to different clip"""
        if new_clip_name not in self.clips or new_clip_name == self.current_clip_name:
            return
        
        if new_clip_name == "idle":
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = False
            print(f"→ Switched to: {new_clip_name}")
            return
        
        current_clip = self.clips[self.current_clip_name]
        can_switch = (self.current_clip_name == "idle" or current_clip.has_finished())
        
        if can_switch:
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = True
            print(f"→ Switched to: {new_clip_name}")
    
    def draw_ui(self, gesture, webcam_frame):
        """Draw UI overlay"""
        if not self.show_ui:
            return
        
        overlay = pygame.Surface((self.display_width, 120))
        overlay.set_alpha(180)
        overlay.fill(COLOR_BLACK)
        
        clip_text = self.font_medium.render(
            f"Clip: {self.current_clip_name}",
            True,
            COLOR_GREEN if self.current_clip_name != "idle" else COLOR_WHITE
        )
        overlay.blit(clip_text, (10, 10))
        
        gesture_color = COLOR_YELLOW if gesture != "none" else COLOR_WHITE
        gesture_text = self.font_medium.render(
            f"Gesture: {gesture}",
            True,
            gesture_color
        )
        overlay.blit(gesture_text, (10, 50))
        
        # Show active effects
        effects_text = self.font_small.render(
            f"Effects: {', '.join(self.effects_enabled)}",
            True,
            COLOR_BLUE
        )
        overlay.blit(effects_text, (10, 90))
        
        if self.current_clip_name != "idle":
            progress = self.clips[self.current_clip_name].get_progress()
            bar_width = 200
            bar_height = 10
            bar_x = self.display_width - bar_width - 10
            bar_y = 55
            
            pygame.draw.rect(overlay, COLOR_WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
            pygame.draw.rect(overlay, COLOR_BLUE, (bar_x, bar_y, int(bar_width * progress / 100), bar_height))
        
        self.screen.blit(overlay, (0, 0))
        
        if self.show_webcam_preview and webcam_frame is not None:
            webcam_small = cv2.resize(
                webcam_frame,
                (WEBCAM_PREVIEW_SIZE, int(WEBCAM_PREVIEW_SIZE * 0.75))
            )
            webcam_small = cv2.cvtColor(webcam_small, cv2.COLOR_BGR2RGB)
            webcam_surface = pygame.surfarray.make_surface(webcam_small.swapaxes(0, 1))
            
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
        print("LAYERED INTERACTIVE PORTRAIT RUNNING")
        print("="*60)
        print("Gestures (trigger once, animation plays to completion):")
        print("  👍 Thumbs up → Smile")
        print("  👋 Wave → Sup dude")
        print("  👈 Point left → Look left")
        print("  👉 Point right → Look right")
        print(f"\nBackground effects: {', '.join(self.effects_enabled)}")
        print("\nControls:")
        print("  W → Toggle webcam preview")
        print("  U → Toggle UI overlay")
        print("  F → Toggle fullscreen")
        print("  Q/ESC → Quit")
        print("="*60 + "\n")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_w:
                        self.show_webcam_preview = not self.show_webcam_preview
                    elif event.key == pygame.K_u:
                        self.show_ui = not self.show_ui
                    elif event.key == pygame.K_f:
                        self.toggle_fullscreen()
            
            ret, webcam_frame = self.webcam.read()
            if not ret:
                continue
            
            gesture = self.gesture_detector.detect(webcam_frame)
            
            if gesture != "none" and gesture != last_triggered_gesture:
                if not self.clip_locked or self.current_clip_name == "idle":
                    target_clip = GESTURE_MAP.get(gesture, "idle")
                    self.switch_clip(target_clip)
                    last_triggered_gesture = gesture
            
            if gesture == "none":
                last_triggered_gesture = "none"
            
            if self.clip_locked and self.current_clip_name != "idle":
                if self.clips[self.current_clip_name].has_finished():
                    self.clip_locked = False
                    self.switch_clip("idle")
            
            # Get animated background
            background = self.background_effects.apply_effects(self.effects_enabled)
            
            # Get person frame
            person_frame = self.clips[self.current_clip_name].get_frame()
            
            if person_frame is not None:
                # Composite layers
                composited = self.composite_frame(background, person_frame)
                
                # Convert to pygame
                composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
                composited_surface = pygame.surfarray.make_surface(composited_rgb.swapaxes(0, 1))
                
                self.screen.blit(composited_surface, (0, 0))
                self.draw_ui(gesture, webcam_frame)
                pygame.display.flip()
            
            clock.tick(30)
        
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
    parser = argparse.ArgumentParser(
        description="Run interactive layered portrait"
    )
    parser.add_argument(
        "prepared_photo_dir",
        help="Path to prepared photo directory (e.g., prepared-photos/karol-photo)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prepared_photo_dir):
        print(f"ERROR: Prepared photo directory not found: {args.prepared_photo_dir}")
        print("Run prepare_photo.py first!")
        sys.exit(1)
    
    app = LayeredPortrait(args.prepared_photo_dir)
    app.run()

if __name__ == "__main__":
    main()