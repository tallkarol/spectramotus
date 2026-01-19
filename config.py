"""Configuration for Interactive Portrait"""

# Paths
PHOTO_NAME = "karol-photo"
SOURCE_PHOTOS_DIR = "source-photos"
DRIVING_VIDEOS_DIR = "driving-videos"
OUTPUT_DIR = "generated_clips"

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WEBCAM_PREVIEW_SIZE = 240
TARGET_FPS = 30

# Gesture detection
GESTURE_CONFIDENCE = 0.7
GESTURE_TRACKING_CONFIDENCE = 0.5
GESTURE_DEBOUNCE_FRAMES = 3

# Gesture to clip mapping
GESTURE_MAP = {
    "thumbs_up": "smile",
    "wave": "sup_dude",
    "pointing_up": "look_left",
    "open_palm": "look_right",
    "none": "idle"
}

# Driving videos for batch generation
DRIVING_VIDEOS = {
    "idle": "base-position.mp4",
    "look_left": "look-left.mp4",
    "look_right": "look-right.mp4",
    "smile": "smile.mp4",
    "sup_dude": "sup-dude.mp4",
}