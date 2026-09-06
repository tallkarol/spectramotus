"""MediaPipe Tasks model files used by intake (pipeline) and the sensing runtime.

`spectramotus models download` fetches them into `models/mediapipe/` (git-ignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from spectramotus.config import settings

_BASE = "https://storage.googleapis.com/mediapipe-models"

MEDIAPIPE_MODELS: dict[str, str] = {
    "face_landmarker.task": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_full.task": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "gesture_recognizer.task": f"{_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "blaze_face_short_range.tflite": f"{_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def mediapipe_dir() -> Path:
    return settings.models_dir / "mediapipe"


def model_path(name: str) -> Path:
    return mediapipe_dir() / name


def missing_models() -> list[str]:
    return [name for name in MEDIAPIPE_MODELS if not model_path(name).exists()]


def download_models(force: bool = False, log=print) -> list[Path]:
    """Download every MediaPipe model that is missing (or all, with force). Returns the written paths."""
    mediapipe_dir().mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url in MEDIAPIPE_MODELS.items():
        dest = model_path(name)
        if dest.exists() and not force:
            log(f"  = {name} (present, {dest.stat().st_size // 1024} KB)")
            continue
        log(f"  > {name} <- {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        written.append(dest)
        log(f"  + {name} ({dest.stat().st_size // 1024} KB)")
    return written
