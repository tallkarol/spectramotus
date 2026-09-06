#!/usr/bin/env python3
"""
Spectramotus Interactive Portrait V2 — RGBA PNG Sequence Player

Loads pre-computed RGBA PNG sequences and composites them over dynamic
backgrounds using clean alpha blending. No chroma keying at runtime.

Usage:
    python interactive_portrait_v2.py prepared-photos/karol-photo

Expects:
    generated_clips/<photo_name>/
        idle/frame_0000.png, frame_0001.png, ...
        smile/frame_0000.png, ...
        ...
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import pygame

# Optional: mediapipe for gesture detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Gesture detection disabled.")
    print("Install: pip install mediapipe")

# Gesture → clip mapping
GESTURE_MAP = {
    "thumbs_up": "smile",
    "wave": "sup_dude",
    "point_left": "look_left",
    "point_right": "look_right",
}


class GestureDetector:
    """Hand gesture detection using MediaPipe"""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.hands = None
            return
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        if self.hands is None:
            return "none"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return "none"

        hand = results.multi_hand_landmarks[0]
        landmarks = hand.landmark

        # Simple gesture detection
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        index_mcp = landmarks[5]

        # Thumbs up: thumb high, other fingers curled
        if (thumb_tip.y < index_tip.y and
            thumb_tip.y < middle_tip.y and
            thumb_tip.y < wrist.y - 0.15):
            return "thumbs_up"

        # Wave: all fingers extended and spread
        if (index_tip.y < index_mcp.y and
            middle_tip.y < landmarks[9].y and
            ring_tip.y < landmarks[13].y and
            pinky_tip.y < landmarks[17].y):
            return "wave"

        # Point left/right: index extended, others curled
        if (index_tip.y < index_mcp.y and
            middle_tip.y > landmarks[9].y):
            if index_tip.x < wrist.x - 0.1:
                return "point_left"
            elif index_tip.x > wrist.x + 0.1:
                return "point_right"

        return "none"


class PNGClipPlayer:
    """
    Plays a clip from pre-loaded RGBA PNG frames.
    
    Frames are loaded into memory on init for smooth playback.
    """

    def __init__(self, clip_dir):
        self.clip_dir = clip_dir
        self.frames = []
        self.current_frame = 0
        self.has_completed_cycle = False
        self.last_frame_time = 0

        # Load clip metadata
        meta_path = os.path.join(clip_dir, "clip_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.fps = meta.get("fps", 25)
        else:
            self.fps = 25

        self.frame_delay = 1.0 / self.fps

        # Load all frames into memory
        frame_files = sorted(Path(clip_dir).glob("frame_*.png"))
        if not frame_files:
            raise ValueError(f"No frame_*.png files found in {clip_dir}")

        print(f"    Loading {len(frame_files)} frames from {clip_dir}...")
        for fp in frame_files:
            # Load as BGRA (unchanged, preserving alpha)
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    Warning: could not load {fp}")
                continue
            if img.shape[2] == 3:
                # No alpha — add fully opaque
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img[:, :, 3] = 255
            self.frames.append(img)

        self.total_frames = len(self.frames)
        if self.total_frames == 0:
            raise ValueError(f"No valid frames loaded from {clip_dir}")

    def get_frame(self):
        """Get current BGRA frame (or None if not time yet)"""
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_delay:
            return None

        self.last_frame_time = current_time
        frame = self.frames[self.current_frame]
        self.current_frame += 1

        if self.current_frame >= self.total_frames:
            self.has_completed_cycle = True
            self.current_frame = 0

        return frame

    def reset(self):
        self.current_frame = 0
        self.has_completed_cycle = False
        self.last_frame_time = 0

    def has_finished(self):
        return self.has_completed_cycle

    def get_progress(self):
        if self.total_frames == 0:
            return 0
        return (self.current_frame / self.total_frames) * 100

    def release(self):
        self.frames = []


class BackgroundEffects:
    """Animated background effects"""

    def __init__(self, background_image):
        self.original_bg = background_image.copy()
        self.height, self.width = background_image.shape[:2]
        self.time = 0
        self.breathing_speed = 0.5
        self.breathing_amplitude = 0.02
        self.parallax_amplitude = 5
        self.time_of_day_cycle = 60.0

    def apply_effects(self, effect_list):
        result = self.original_bg.copy()
        self.time += 1 / 30

        if "breathing" in effect_list:
            result = self._apply_breathing(result)
        if "parallax" in effect_list:
            result = self._apply_parallax(result)
        if "time_of_day" in effect_list:
            result = self._apply_time_of_day(result)

        return result

    def _apply_breathing(self, image):
        scale = 1.0 + self.breathing_amplitude * math.sin(
            2 * math.pi * self.breathing_speed * self.time
        )
        new_w = int(self.width * scale)
        new_h = int(self.height * scale)
        resized = cv2.resize(image, (new_w, new_h))
        sx = (new_w - self.width) // 2
        sy = (new_h - self.height) // 2
        if scale > 1.0:
            return resized[sy : sy + self.height, sx : sx + self.width]
        return image

    def _apply_parallax(self, image):
        ox = int(self.parallax_amplitude * math.sin(2 * math.pi * 0.3 * self.time))
        oy = int(self.parallax_amplitude * 0.5 * math.cos(2 * math.pi * 0.2 * self.time))
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        return cv2.warpAffine(image, M, (self.width, self.height), borderMode=cv2.BORDER_REPLICATE)

    def _apply_time_of_day(self, image):
        tod = (self.time % self.time_of_day_cycle) / self.time_of_day_cycle
        brightness = 0.8 + 0.2 * math.sin(2 * math.pi * tod)
        result = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        temp = abs(tod - 0.5) * 2
        if temp > 0.5:
            result[:, :, 2] = cv2.convertScaleAbs(result[:, :, 2], alpha=1.1, beta=0)
            result[:, :, 0] = cv2.convertScaleAbs(result[:, :, 0], alpha=0.9, beta=0)
        return result


class LayeredPortraitV2:
    """Main application — RGBA alpha compositing, no chroma key"""

    def __init__(self, prepared_photo_dir):
        self.prepared_photo_dir = prepared_photo_dir
        self.photo_name = Path(prepared_photo_dir).name

        # Load metadata
        metadata_path = os.path.join(prepared_photo_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"animations": {"background_effects": []}}

        # Load background
        bg_path = os.path.join(prepared_photo_dir, "background.png")
        if not os.path.exists(bg_path):
            raise ValueError(f"Background not found: {bg_path}")
        self.background_image = cv2.imread(bg_path)
        self.bg_height, self.bg_width = self.background_image.shape[:2]

        # Initialize background effects
        self.background_effects = BackgroundEffects(self.background_image)

        # Load RGBA clips
        clips_dir = os.path.join("generated_clips", self.photo_name)
        if not os.path.exists(clips_dir):
            raise ValueError(f"Clips directory not found: {clips_dir}")
        self.clips = self._load_clips(clips_dir)

        # Calculate display size
        max_dim = 1080
        aspect = self.bg_width / self.bg_height
        if aspect > 1:
            self.display_width = min(self.bg_width, max_dim)
            self.display_height = int(self.display_width / aspect)
        else:
            self.display_height = min(self.bg_height, max_dim)
            self.display_width = int(self.display_height * aspect)

        print(f"Background: {self.bg_width}x{self.bg_height}")
        print(f"Display: {self.display_width}x{self.display_height}")

        # Init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption(f"Portrait V2 — {self.photo_name}")

        self.current_clip_name = "idle"

        # Webcam
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Gesture detector
        self.gesture_detector = GestureDetector()

        # State
        self.running = True
        self.show_webcam_preview = False
        self.show_ui = True
        self.clip_locked = False
        self.fullscreen = False
        self.effects_enabled = self.metadata.get("animations", {}).get(
            "background_effects", []
        )

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

    def _load_clips(self, clips_dir):
        """Load RGBA PNG sequence clips"""
        clips = {}
        print(f"\nLoading RGBA clips from: {clips_dir}")
        print("=" * 60)

        for gesture, clip_name in GESTURE_MAP.items():
            clip_path = os.path.join(clips_dir, clip_name)
            if os.path.isdir(clip_path) and any(Path(clip_path).glob("frame_*.png")):
                clips[clip_name] = PNGClipPlayer(clip_path)
                print(f"  ✓ {clip_name}: {clips[clip_name].total_frames} frames")
            else:
                print(f"  ✗ Missing: {clip_path}/")

        # Always need idle
        if "idle" not in clips:
            # Check for idle directory
            idle_path = os.path.join(clips_dir, "idle")
            if os.path.isdir(idle_path):
                clips["idle"] = PNGClipPlayer(idle_path)
                print(f"  ✓ idle: {clips['idle'].total_frames} frames")
            else:
                raise ValueError("idle clip is required!")

        print("=" * 60 + "\n")
        return clips

    def composite_frame(self, background, person_bgra):
        """
        Composite BGRA person frame over BGR background using alpha blending.
        
        No chroma keying! The alpha channel does all the work.
        """
        bg = cv2.resize(background, (self.display_width, self.display_height))
        fg = cv2.resize(person_bgra, (self.display_width, self.display_height),
                        interpolation=cv2.INTER_LINEAR)

        # Split person into BGR + Alpha
        fg_bgr = fg[:, :, :3]
        alpha = fg[:, :, 3].astype(np.float32) / 255.0

        # Expand alpha to 3 channels
        alpha_3d = np.stack([alpha] * 3, axis=2)

        # Alpha composite: result = fg * alpha + bg * (1 - alpha)
        result = (fg_bgr.astype(np.float32) * alpha_3d +
                  bg.astype(np.float32) * (1.0 - alpha_3d))

        return result.astype(np.uint8)

    def switch_clip(self, new_clip_name):
        if new_clip_name not in self.clips or new_clip_name == self.current_clip_name:
            return

        if new_clip_name == "idle":
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = False
            print(f"→ Switched to: {new_clip_name}")
            return

        current_clip = self.clips[self.current_clip_name]
        can_switch = self.current_clip_name == "idle" or current_clip.has_finished()

        if can_switch:
            self.current_clip_name = new_clip_name
            self.clips[new_clip_name].reset()
            self.clip_locked = True
            print(f"→ Switched to: {new_clip_name}")

    def draw_ui(self, gesture, webcam_frame):
        if not self.show_ui:
            return

        overlay = pygame.Surface((self.display_width, 120))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, self.display_height - 120))

        # Current state
        state_text = self.font_medium.render(
            f"Clip: {self.current_clip_name} | Gesture: {gesture}", True, (255, 255, 255)
        )
        self.screen.blit(state_text, (10, self.display_height - 110))

        # Progress bar
        clip = self.clips[self.current_clip_name]
        progress = clip.get_progress()
        bar_w = self.display_width - 20
        pygame.draw.rect(
            self.screen, (60, 60, 60), (10, self.display_height - 70, bar_w, 20)
        )
        pygame.draw.rect(
            self.screen,
            (0, 200, 100),
            (10, self.display_height - 70, int(bar_w * progress / 100), 20),
        )

        # Controls
        controls = self.font_small.render(
            "W=webcam  U=UI  F=fullscreen  Q=quit", True, (180, 180, 180)
        )
        self.screen.blit(controls, (10, self.display_height - 40))

        # V2 badge
        badge = self.font_small.render("V2 — RGBA Alpha", True, (100, 200, 100))
        self.screen.blit(badge, (self.display_width - 160, self.display_height - 40))

        # Webcam preview
        if self.show_webcam_preview and webcam_frame is not None:
            preview = cv2.resize(webcam_frame, (160, 120))
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            preview_surface = pygame.surfarray.make_surface(preview_rgb.swapaxes(0, 1))
            self.screen.blit(
                preview_surface,
                (self.display_width - 170, int(self.display_height * 0.75) - 10),
            )

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(
                (self.display_width, self.display_height)
            )

    def run(self):
        clock = pygame.time.Clock()
        last_triggered_gesture = "none"

        print("\n" + "=" * 60)
        print("INTERACTIVE PORTRAIT V2 — RGBA ALPHA COMPOSITING")
        print("=" * 60)
        print("Gestures:")
        print("  👍 Thumbs up → Smile")
        print("  👋 Wave → Sup dude")
        print("  👈 Point left → Look left")
        print("  👉 Point right → Look right")
        print(f"\nBackground effects: {', '.join(self.effects_enabled)}")
        print("\nControls: W=webcam  U=UI  F=fullscreen  Q/ESC=quit")
        print("=" * 60 + "\n")

        # Cache the last composited frame for when get_frame returns None
        last_composited = None

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
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

            # Get person BGRA frame
            person_frame = self.clips[self.current_clip_name].get_frame()

            if person_frame is not None:
                # Alpha composite — clean, no chroma keying
                composited = self.composite_frame(background, person_frame)
                last_composited = composited
            elif last_composited is not None:
                # Re-composite with updated background (for effects)
                # but reuse the last person frame
                composited = last_composited
            else:
                continue

            # Render
            composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(composited_rgb.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))
            self.draw_ui(gesture, webcam_frame)
            pygame.display.flip()

            clock.tick(30)

        self.cleanup()

    def cleanup(self):
        print("\nCleaning up...")
        for clip in self.clips.values():
            clip.release()
        self.webcam.release()
        pygame.quit()
        print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive portrait with RGBA alpha compositing (V2)"
    )
    parser.add_argument(
        "prepared_photo_dir",
        help="Path to prepared photo directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.prepared_photo_dir):
        print(f"ERROR: Not found: {args.prepared_photo_dir}")
        sys.exit(1)

    app = LayeredPortraitV2(args.prepared_photo_dir)
    app.run()


if __name__ == "__main__":
    main()