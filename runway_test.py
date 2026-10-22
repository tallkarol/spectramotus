#!/usr/bin/env python3
"""
Spectramotus Runway API Test
Generates behavior clips from one photo and measures how well each clip returns
to the photo (the "rest frame") so the player can loop and chain them.

Needs RUNWAYML_API_SECRET in the environment (or .env) - the same key tkcrm uses.

Usage:
    python runway_test.py ~/Downloads/plitvice.jpg --dry-run          # show prompts, spend nothing
    python runway_test.py ~/Downloads/plitvice.jpg                    # round 1: wave, thumbs_up, away_and_back
    python runway_test.py ~/Downloads/plitvice.jpg --behaviors wave --model gen4.5
    python runway_test.py ~/Downloads/plitvice.jpg --last-frame       # also pin the end frame (some models only)

Output: runway-tests/<photo>/<timestamp>/  (clips, frames, report.json)
"""

import os
import io
import sys
import json
import time
import base64
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
from PIL import Image, ImageOps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = "https://api.dev.runwayml.com/v1"
API_VERSION = "2024-11-06"
OUTPUT_BASE_DIR = "runway-tests"
UPLOAD_MAX_DIM = 2048  # Downscale before upload - output is 720p anyway and data URIs have a size cap

LOCKED_CAMERA = (
    "Static locked-off tripod shot, no camera movement, no zoom, no cuts. "
    "Realistic natural motion. The background stays the same; only leaves and clouds move gently."
)

# Written for the Plitvice railing photo: man on the left, one hand on the wooden railing,
# the other hand at his side, valley behind him. Every clip must end in the starting pose.
BEHAVIORS = {
    # Round 1 - the riskiest clips
    "wave": {
        "round": 1,
        "duration": 5,
        "prompt": (
            "The man lifts his hand off the wooden railing and gives a friendly wave toward the camera "
            "with a warm smile, then puts his hand back on the railing and returns to exactly his "
            "starting pose, looking at the camera."
        ),
    },
    "thumbs_up": {
        "round": 1,
        "duration": 5,
        "prompt": (
            "The man raises the hand at his side and gives a thumbs up toward the camera with a big smile, "
            "holds it for a moment, then lowers his hand and returns to exactly his starting pose, "
            "looking at the camera."
        ),
    },
    "away_and_back": {
        "round": 1,
        "duration": 10,
        "prompt": (
            "The man turns and walks out of the frame to the left, leaving the viewpoint empty with only "
            "the railing and the valley visible for a moment. Then he walks back into the frame from the "
            "left, rests one hand on the wooden railing and returns to exactly his starting pose and smile, "
            "looking at the camera."
        ),
    },
    # Round 2 - idles the player shuffles through
    "breathe": {
        "round": 2,
        "duration": 5,
        "prompt": (
            "The man stands relaxed and breathes slowly, shoulders rising and falling very slightly, "
            "blinks once, and stays in exactly his starting pose, looking at the camera."
        ),
    },
    "adjust_strap": {
        "round": 2,
        "duration": 5,
        "prompt": (
            "The man briefly adjusts the strap of his shoulder bag with the hand at his side, then lowers "
            "his hand and settles back into exactly his starting pose, looking at the camera."
        ),
    },
    "look_at_view": {
        "round": 2,
        "duration": 5,
        "prompt": (
            "The man turns his head to look out over the valley behind him for a moment, then turns back "
            "to the camera with a small smile and returns to exactly his starting pose."
        ),
    },
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


def prepare_upload(image_path):
    """Load photo (respecting EXIF rotation), downscale, return (PIL image, JPEG data URI)"""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail((UPLOAD_MAX_DIM, UPLOAD_MAX_DIM), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(f"Upload image: {image.size[0]}x{image.size[1]}, {len(buffer.getvalue()) // 1024} KB")
    return image, f"data:image/jpeg;base64,{encoded}"


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
    """Mean absolute pixel difference (0-255) and % of pixels that changed visibly"""
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
    parser = argparse.ArgumentParser(description="Test Runway image-to-video for looping behavior clips")
    parser.add_argument("image_path", help="Source photo (e.g., ~/Downloads/plitvice.jpg)")
    parser.add_argument("--model", default="gen4_turbo",
                        help="Runway API model id (default: gen4_turbo - cheapest for dialing in prompts)")
    parser.add_argument("--ratio", default="832:1104", help="Output ratio (default 832:1104 = 3:4 portrait)")
    parser.add_argument("--behaviors", nargs="+", choices=list(BEHAVIORS),
                        help="Behaviors to generate (default: all round 1 behaviors)")
    parser.add_argument("--last-frame", action="store_true",
                        help="Also pin the source photo as the last frame (only some models support this)")
    parser.add_argument("--seed", type=int, help="Fixed seed for comparable runs")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent, spend nothing")
    args = parser.parse_args()

    behaviors = args.behaviors or [name for name, b in BEHAVIORS.items() if b["round"] == 1]

    if not os.path.exists(args.image_path):
        print(f"ERROR: Source photo not found: {args.image_path}")
        sys.exit(1)

    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key and not args.dry_run:
        print("ERROR: RUNWAYML_API_SECRET not set (env or .env)")
        sys.exit(1)

    photo_name = Path(args.image_path).stem
    run_dir = Path(OUTPUT_BASE_DIR) / photo_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    frames_dir = run_dir / "frames"

    print("=" * 60)
    print("SPECTRAMOTUS RUNWAY TEST" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)
    print(f"Source: {args.image_path}")
    print(f"Model: {args.model}  Ratio: {args.ratio}  Pin last frame: {args.last_frame}")
    print(f"Behaviors: {', '.join(behaviors)} ({sum(BEHAVIORS[b]['duration'] for b in behaviors)}s of video total)")

    _, image_uri = prepare_upload(args.image_path)
    prompt_image = [{"uri": image_uri, "position": "first"}]
    if args.last_frame:
        prompt_image.append({"uri": image_uri, "position": "last"})

    if args.dry_run:
        for behavior in behaviors:
            spec = BEHAVIORS[behavior]
            print(f"\n→ {behavior} ({spec['duration']}s)\n  {spec['prompt']} {LOCKED_CAMERA}")
        print("\nDry run - nothing sent.")
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    client = RunwayClient(api_key)

    credits_before = None
    try:
        credits_before = client.organization().get("creditBalance")
        print(f"API credit balance: {credits_before}")
    except Exception as e:
        print(f"⚠ Could not read credit balance: {e}")
    print(f"Output: {run_dir}/")
    print("=" * 60)

    results = []
    for behavior in behaviors:
        spec = BEHAVIORS[behavior]
        print(f"\n→ {behavior} ({spec['duration']}s)")
        payload = {
            "model": args.model,
            "promptImage": prompt_image,
            "promptText": f"{spec['prompt']} {LOCKED_CAMERA}",
            "ratio": args.ratio,
            "duration": spec["duration"],
        }
        if args.seed is not None:
            payload["seed"] = args.seed

        result = {"behavior": behavior, "duration": spec["duration"], "prompt": payload["promptText"]}
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
            if "model" in str(e).lower() or "ratio" in str(e).lower():
                print("  Stopping - fix --model/--ratio before spending more")
                results.append(result)
                break
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
