"""Environment checks: `spectramotus doctor`."""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata

from spectramotus.config import REPO_ROOT, settings
from spectramotus.sense.models import MEDIAPIPE_MODELS, model_path

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _installed(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001 - report anything, this is a diagnostic
        return None, f"{type(e).__name__}: {e}"


def check_python() -> Check:
    v = sys.version.split()[0]
    ok = sys.version_info[:2] == (3, 11)
    return Check("python", OK if ok else WARN, f"{v} at {sys.executable} ({platform.machine()})")


def check_cv2() -> Check:
    dists = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless") if _installed(d)]
    mod, err = _try_import("cv2")
    if err:
        return Check("opencv", FAIL, err)
    has_ximgproc = hasattr(mod, "ximgproc")
    if len(dists) != 1:
        return Check("opencv", FAIL, f"expected exactly one OpenCV distribution, found {dists or 'none'}")
    status = OK if has_ximgproc else WARN
    return Check("opencv", status, f"{dists[0]} {mod.__version__}, ximgproc={'yes' if has_ximgproc else 'no (need contrib for guidedFilter)'}")


def check_torch() -> Check:
    mod, err = _try_import("torch")
    if err:
        return Check("torch", FAIL, err)
    mps = bool(getattr(mod.backends, "mps", None) and mod.backends.mps.is_available())
    return Check("torch", OK if mps else WARN, f"{mod.__version__}, mps_available={mps}")


def check_onnxruntime() -> Check:
    dists = [d for d in ("onnxruntime", "onnxruntime-silicon", "onnxruntime-gpu") if _installed(d)]
    mod, err = _try_import("onnxruntime")
    if err:
        return Check("onnxruntime", FAIL, err)
    providers = mod.get_available_providers()
    coreml = "CoreMLExecutionProvider" in providers
    status = OK if (coreml and len(dists) == 1) else WARN
    return Check("onnxruntime", status, f"{dists} {mod.__version__}, providers={providers}")


def check_mediapipe() -> Check:
    mod, err = _try_import("mediapipe")
    if err:
        return Check("mediapipe", FAIL, err)
    _, err2 = _try_import("mediapipe.tasks.python.vision")
    if err2:
        return Check("mediapipe", FAIL, f"{mod.__version__}: tasks API missing ({err2})")
    return Check("mediapipe", OK, f"{mod.__version__} (Tasks API)")


def check_ffmpeg() -> Check:
    exe = shutil.which("ffmpeg")
    if not exe:
        return Check("ffmpeg", FAIL, "ffmpeg not on PATH (brew install ffmpeg)")
    try:
        ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
        dec = subprocess.run([exe, "-hide_banner", "-decoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return Check("ffmpeg", FAIL, f"{exe}: {e}")
    need = {"libvpx-vp9 encoder": "libvpx-vp9" in enc, "libx264": "libx264" in enc, "libvpx-vp9 decoder": "libvpx-vp9" in dec}
    missing = [k for k, v in need.items() if not v]
    return Check("ffmpeg", OK if not missing else FAIL, f"{ver.split(' Copyright')[0]}; missing={missing or 'none'}")


def check_weights() -> Check:
    base = settings.weights_dir / "liveportrait"
    required = [
        base / "base_models" / "appearance_feature_extractor.pth",
        base / "base_models" / "motion_extractor.pth",
        base / "base_models" / "spade_generator.pth",
        base / "base_models" / "warping_module.pth",
        base / "retargeting_models" / "stitching_retargeting_module.pth",
        base / "landmark.onnx",
        settings.weights_dir / "insightface" / "models" / "buffalo_l" / "det_10g.onnx",
        settings.weights_dir / "insightface" / "models" / "buffalo_l" / "2d106det.onnx",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.exists()]
    return Check("liveportrait weights", OK if not missing else FAIL, "all present" if not missing else f"missing: {missing}")


def check_mediapipe_models() -> Check:
    missing = [n for n in MEDIAPIPE_MODELS if not model_path(n).exists()]
    return Check("mediapipe models", OK if not missing else WARN, "all present" if not missing else f"missing {missing} -> spectramotus models download")


def check_keys() -> Check:
    ks = settings.key_status()
    missing = [k for k, v in ks.items() if not v]
    required_missing = [k for k in ("FAL_KEY", "ANTHROPIC_API_KEY") if k in missing]
    status = FAIL if required_missing else (WARN if missing else OK)
    return Check("api keys", status, "all set" if not missing else f"missing: {missing} (see .env.example)")


def check_dirs() -> Check:
    drivers = settings.drivers_dir
    n = len(list(drivers.rglob("*.mp4"))) if drivers.exists() else 0
    status = OK if n else WARN
    return Check("driving performances", status, f"{n} mp4 under {drivers.relative_to(REPO_ROOT)}" if drivers.exists() else "driving-performances/ not created yet")


def check_disk() -> Check:
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1e9
    return Check("disk", OK if free_gb > 20 else WARN, f"{free_gb:.0f} GB free")


def run_all() -> list[Check]:
    return [
        check_python(),
        check_cv2(),
        check_torch(),
        check_onnxruntime(),
        check_mediapipe(),
        check_ffmpeg(),
        check_weights(),
        check_mediapipe_models(),
        check_keys(),
        check_dirs(),
        check_disk(),
    ]
