#!/usr/bin/env python3
"""
Spectramotus Runway API Test
Generates a few idle-behavior clips from one photo and measures how well each
clip returns to the starting frame (the "rest frame") so it can loop.

Needs RUNWAYML_API_SECRET in the environment (or .env) - the same key tkcrm uses.

Usage:
    python runway_test.py source-photos/karol-linkedin.jpg
    python runway_test.py source-photos/karol-linkedin.jpg --model gen4.5 --behaviors breathe
    python runway_test.py source-photos/karol-linkedin.jpg --last-frame   # also pin the end frame

Output: runway-tests/<photo>/<timestamp>/  (clips, report.json)
"""

import os
import sys
import json
import time
import base64
import argparse
import mimetypes
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = "https://api.dev.runwayml.com/v1"
API_VERSION = "2024-11-06"
OUTPUT_BASE_DIR = "runway-tests"

LOCKED_CAMERA = (
    "Static locked-off tripod shot, no camera movement, no zoom. "
    "The person stays in the same position and framing throughout."
)

# Each behavior must start and end on the original photo pose
BEHAVIORS = {
    "breathe": (
        "The person breathes slowly and naturally, shoulders rising and falling very slightly, "
        "blinks once, then is still in exactly the starting pose."
    ),
    "adjust_shirt": (
        "The person briefly adjusts their shirt collar with one hand, then lowers the hand "
        "and settles back into exactly the starting pose."
    ),
    "glance": (
        "The person glances slowly to the side as if something caught their attention, "
        "then looks back at the camera and returns to exactly the starting pose."
    ),
}


class RunwayClient:
    """Minimal Runway API client (REST, no SDK)"""

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-Runway-Version": API_VERSION,
        })

    def _request(self, method, path, **kwargs):
        response = self.session.request(method, f"{API_BASE}{path}", timeout=60, **kwargs)
        if not response.ok:
            # Print the API's own error - it usually names the valid models/ratios
            raise RuntimeError(f"{method} {path} -> HTTP {response.status_code}: {response.text[:500]}")
        return response.json() if response.content else {}

    def organization(self):
        return self._request("GET", "/organization")

    def image_to_video(self, payload):
        return self._request("POST", "/image_to_video", json=payload)["id"]

    def wait_for_task(self, task_id, poll_seconds=10, timeout_seconds=1800):
        start = time.time()
        while time.time() - start < timeout_seconds:
            task = self._request("GET", f"/tasks/{task_id}")
            status = task.get("status")
            print(f"    {status} ({int(time.time() - start)}s)")
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Task {task_id} {status}: {task.get('failure') or task}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Task {task_id} did not finish in {timeout_seconds}s")


def image_to_data_uri(image_path):
    """Encode image as data URI for the promptImage field"""
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_frame(video_path, output_path, last=False):
    """Extract first or last frame with ffmpeg"""
    if last:
        cmd = ["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(video_path),
               "-update", "1", str(output_path)]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vframes", "1", str(output_path)]
    subprocess.run(cmd, check=True)
    return np.asarray(Image.open(output_path).convert("RGB")).astype(np.int16)


def frame_diff(a, b):
    """Mean absolute pixel difference (0-255) and share of pixels that changed visibly"""
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))).astype(np.int16)
    diff = np.abs(a - b).max(axis=2)
    return round(float(diff.mean()), 2), round(float((diff > 30).mean() * 100), 2)


def score_seam(video_path, frames_dir):
    """How close the last frame gets back to the first (lower = better loop)"""
    first = extract_frame(video_path, frames_dir / f"{video_path.stem}_first.png")
    last = extract_frame(video_path, frames_dir / f"{video_path.stem}_last.png", last=True)
    mean_diff, changed_pct = frame_diff(first, last)

    if mean_diff < 3 and changed_pct < 1:
        verdict = "loops"
    elif mean_diff < 8:
        verdict = "needs crossfade"
    else:
        verdict = "reject"

    return {"seam_mean_diff": mean_diff, "seam_changed_pct": changed_pct, "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description="Test Runway image-to-video for looping idle clips")
    parser.add_argument("image_path", help="Source photo (e.g., source-photos/karol-linkedin.jpg)")
    parser.add_argument("--model", default="gen4.5", help="Runway API model id (default: gen4.5)")
    parser.add_argument("--ratio", default="720:1280", help="Output ratio, e.g. 720:1280 or 1280:720")
    parser.add_argument("--duration", type=int, default=5, help="Clip length in seconds")
    parser.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    parser.add_argument("--last-frame", action="store_true",
                        help="Also pin the source photo as the last frame (only some models support this)")
    parser.add_argument("--seed", type=int, help="Fixed seed for comparable runs")
    args = parser.parse_args()

    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        print("ERROR: RUNWAYML_API_SECRET not set (env or .env)")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"ERROR: Source photo not found: {args.image_path}")
        sys.exit(1)

    photo_name = Path(args.image_path).stem
    run_dir = Path(OUTPUT_BASE_DIR) / photo_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    client = RunwayClient(api_key)

    print("=" * 60)
    print("SPECTRAMOTUS RUNWAY TEST")
    print("=" * 60)
    print(f"Source: {args.image_path}")
    print(f"Model: {args.model}  Ratio: {args.ratio}  Duration: {args.duration}s")
    print(f"Pin last frame: {args.last_frame}")
    print(f"Output: {run_dir}/")

    credits_before = None
    try:
        credits_before = client.organization().get("creditBalance")
        print(f"API credit balance: {credits_before}")
    except Exception as e:
        print(f"⚠ Could not read credit balance: {e}")
    print("=" * 60)

    image_uri = image_to_data_uri(args.image_path)
    prompt_image = [{"uri": image_uri, "position": "first"}]
    if args.last_frame:
        prompt_image.append({"uri": image_uri, "position": "last"})

    results = []
    for behavior in args.behaviors:
        print(f"\n→ {behavior}")
        payload = {
            "model": args.model,
            "promptImage": prompt_image,
            "promptText": f"{BEHAVIORS[behavior]} {LOCKED_CAMERA}",
            "ratio": args.ratio,
            "duration": args.duration,
        }
        if args.seed is not None:
            payload["seed"] = args.seed

        result = {"behavior": behavior, "prompt": payload["promptText"]}
        started = time.time()
        try:
            task_id = client.image_to_video(payload)
            task = client.wait_for_task(task_id)
            video_path = run_dir / f"{photo_name}_{behavior}.mp4"
            with requests.get(task["output"][0], stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

            result.update({"file": video_path.name, "task_id": task_id, "seconds": int(time.time() - started)})
            result.update(score_seam(video_path, frames_dir))
            print(f"✓ {video_path.name}: seam diff {result['seam_mean_diff']} → {result['verdict']}")
        except Exception as e:
            result["error"] = str(e)
            print(f"✗ {behavior} failed: {e}")
        results.append(result)

    credits_after = None
    try:
        credits_after = client.organization().get("creditBalance")
    except Exception:
        pass

    report = {
        "source": args.image_path,
        "model": args.model,
        "ratio": args.ratio,
        "duration": args.duration,
        "last_frame_pinned": args.last_frame,
        "credits_before": credits_before,
        "credits_after": credits_after,
        "results": results,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    for r in results:
        status = r.get("verdict") or f"ERROR: {r.get('error', '')[:120]}"
        print(f"  {r['behavior']:<14} {status}")
    if credits_before is not None and credits_after is not None:
        print(f"\nCredits used: {credits_before - credits_after}")
    print(f"\nReport: {run_dir / 'report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
