from spectramotus import __version__
from spectramotus.sense.models import MEDIAPIPE_MODELS


def test_version():
    assert __version__.startswith("4.")


def test_model_table():
    assert {"face_landmarker.task", "pose_landmarker_full.task", "gesture_recognizer.task"} <= set(MEDIAPIPE_MODELS)
