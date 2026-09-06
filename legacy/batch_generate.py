#!/usr/bin/env python3
"""
Spectramotus Batch Generator V3 — RGBA PNG Sequence Output

Eliminates the green screen approach entirely. Instead:
1. Feeds the ORIGINAL source photo to LivePortrait (with pasteback enabled)
2. LivePortrait animates the face and pastes it back onto the original image
3. We extract RGBA frames using the RMBG-2.0 person mask from prepare_photo.py
4. Saves as PNG sequences with clean alpha — no chroma keying needed at runtime

Usage:
    python batch_generate_v3.py prepared-photos/karol-photo

Output structure:
    generated_clips/<photo_name>/
        idle/
            frame_0000.png  (RGBA)
            frame_0001.png
            ...
            clip_meta.json  (frame count, fps, etc.)
        smile/
            frame_0000.png
            ...
        ...
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image

# Configuration
DRIVING_VIDEOS_DIR = "driving-videos"
OUTPUT_BASE_DIR = "generated_clips"

# Define your driving videos and output clip names
DRIVING_VIDEOS = {
    "idle": "base-position.mp4",
    "look_left": "look-left.mp4",
    "look_right": "look-right.mp4",
    "smile": "smile.mp4",
    "sup_dude": "sup-dude.mp4",
}


def load_metadata(prepared_photo_dir):
    """Load metadata.json from prepared photo directory"""
    metadata_path = os.path.join(prepared_photo_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: No metadata.json found in {prepared_photo_dir}")
        return None
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_metadata(prepared_photo_dir, metadata):
    """Save updated metadata.json"""
    metadata_path = os.path.join(prepared_photo_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_photo_name(prepared_photo_dir):
    return Path(prepared_photo_dir).name


def get_python_executable():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(conda_python):
            return conda_python
    return sys.executable


def load_person_mask(prepared_photo_dir):
    """
    Load the RMBG-2.0 person mask from prepare_photo.py output.
    
    The mask.png from prepare_photo has: 0 = background (keep), 255 = person (remove/inpaint).
    We invert this so: 255 = person (visible), 0 = background (transparent).
    
    We also load person.png alpha channel as a higher-quality alternative if available.
    """
    # Prefer person.png alpha channel — it has soft edges from RMBG-2.0
    person_path = os.path.join(prepared_photo_dir, "person.png")
    if os.path.exists(person_path):
        person_img = Image.open(person_path).convert("RGBA")
        alpha = np.array(person_img)[:, :, 3]  # Extract alpha channel
        print(f"  ✓ Loaded person alpha from person.png ({alpha.shape[1]}x{alpha.shape[0]})")
        return alpha

    # Fallback: use mask.png (binary, less quality at edges)
    mask_path = os.path.join(prepared_photo_dir, "mask.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask.png convention: 255 = person area, 0 = background
        # (prepare_photo.py: alpha > 10 → 255)
        print(f"  ✓ Loaded mask from mask.png ({mask.shape[1]}x{mask.shape[0]})")
        return mask

    raise FileNotFoundError(
        f"No person.png or mask.png found in {prepared_photo_dir}. Run prepare_photo.py first!"
    )


def extract_rgba_frames(video_path, person_mask, output_dir, feather_px=3):
    """
    Extract frames from a LivePortrait pasteback video and combine with
    the person mask to create RGBA PNG frames.
    
    Args:
        video_path: Path to the LivePortrait output MP4 (pasteback onto original image)
        person_mask: Numpy array (H, W) with person alpha (255=person, 0=bg)
        output_dir: Directory to save PNG frames
        feather_px: Pixels to feather/soften mask edges (0 = no feather)
    
    Returns:
        dict with frame_count, fps, dimensions
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize mask to match video dimensions
    mask_resized = cv2.resize(
        person_mask,
        (frame_width, frame_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Optional: feather the mask edges for softer blending
    if feather_px > 0:
        # Erode slightly to eat any edge artifacts, then blur for soft transition
        kernel = np.ones((feather_px, feather_px), np.uint8)
        mask_eroded = cv2.erode(mask_resized, kernel, iterations=1)
        blur_size = feather_px * 2 + 1
        mask_resized = cv2.GaussianBlur(mask_eroded, (blur_size, blur_size), 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame is BGR from OpenCV, convert to BGRA
        # Add the person mask as the alpha channel
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_resized

        # Save as PNG (lossless, preserves alpha)
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, bgra)
        frame_idx += 1

    cap.release()

    # Save clip metadata
    clip_meta = {
        "frame_count": frame_idx,
        "fps": fps,
        "width": frame_width,
        "height": frame_height,
        "format": "BGRA_PNG",
        "feather_px": feather_px,
    }
    meta_path = os.path.join(output_dir, "clip_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(clip_meta, f, indent=2)

    return clip_meta


def generate_clip_with_rgba(source_image, driving_video, output_dir, 
                             clip_name, photo_name, person_mask, feather_px=3):
    """
    Generate one clip:
    1. Run LivePortrait with the original source image (pasteback enabled)
    2. Extract RGBA PNG frames using the person mask
    """
    print(f"\n{'='*60}")
    print(f"Generating: {clip_name}")
    print(f"{'='*60}")

    # Step 1: Run LivePortrait → MP4 with pasteback onto original
    temp_output = os.path.join("temp_output", clip_name)
    Path(temp_output).mkdir(parents=True, exist_ok=True)

    python_exe = get_python_executable()
    cmd = [
        python_exe, "inference.py",
        "--source", source_image,
        "--driving", driving_video,
        "--output-dir", temp_output,
        # pasteback is True by default — animated face composited onto source image
    ]

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ LivePortrait failed: {e}")
        return False, None

    # Find the output video (not the _concat version)
    temp_files = list(Path(temp_output).glob("*.mp4"))
    non_concat_files = [f for f in temp_files if '_concat' not in f.name]

    if not non_concat_files:
        print(f"  ✗ No output video found in {temp_output}")
        return False, None

    pasteback_video = str(non_concat_files[0])
    print(f"  ✓ LivePortrait output: {pasteback_video}")

    # Step 2: Extract RGBA PNG sequence
    clip_output_dir = os.path.join(output_dir, clip_name)
    print(f"  Extracting RGBA frames → {clip_output_dir}/")

    clip_meta = extract_rgba_frames(
        pasteback_video,
        person_mask,
        clip_output_dir,
        feather_px=feather_px,
    )

    print(f"  ✓ Saved {clip_meta['frame_count']} RGBA frames @ {clip_meta['fps']:.1f}fps")

    # Cleanup temp
    shutil.rmtree(temp_output, ignore_errors=True)

    return True, clip_meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate RGBA PNG sequence clips from prepared photos (no green screen)"
    )
    parser.add_argument(
        "prepared_photo_dir",
        help="Path to prepared photo directory (e.g., prepared-photos/karol-photo)"
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=3,
        help="Edge feather amount in pixels (default: 3, 0 = no feather)"
    )
    args = parser.parse_args()
    prepared_photo_dir = args.prepared_photo_dir

    if not os.path.exists(prepared_photo_dir):
        print(f"ERROR: Directory not found: {prepared_photo_dir}")
        sys.exit(1)

    metadata = load_metadata(prepared_photo_dir)
    photo_name = get_photo_name(prepared_photo_dir)

    # Determine source image
    # V3 pipeline: prepare_photo.py saves source.png (the original photo)
    source_image = os.path.join(prepared_photo_dir, "source.png")
    
    if not os.path.exists(source_image):
        # Fallback: try to reconstruct from person + background (for older prepared photos)
        person_path = os.path.join(prepared_photo_dir, "person.png")
        background_path = os.path.join(prepared_photo_dir, "background.png")

        if os.path.exists(person_path) and os.path.exists(background_path):
            print("source.png not found — reconstructing from person + background...")
            person = Image.open(person_path).convert("RGBA")
            background = Image.open(background_path).convert("RGBA")
            if background.size != person.size:
                background = background.resize(person.size, Image.LANCZOS)
            composite = Image.alpha_composite(background, person).convert("RGB")
            composite.save(source_image)
            print(f"  ✓ Saved reconstructed source: {source_image}")
        else:
            print(f"ERROR: source.png not found in {prepared_photo_dir}")
            print("Re-run prepare_photo.py to generate it.")
            sys.exit(1)

    print(f"Source: {source_image}")

    # Load person mask
    print("\nLoading person mask...")
    person_mask = load_person_mask(prepared_photo_dir)

    # Create output directory
    output_folder = os.path.join(OUTPUT_BASE_DIR, photo_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("SPECTRAMOTUS BATCH GENERATOR V3 — RGBA PNG SEQUENCES")
    print("="*60)
    print(f"Source image: {source_image}")
    print(f"Person mask: {person_mask.shape[1]}x{person_mask.shape[0]}")
    print(f"Edge feather: {args.feather}px")
    print(f"Clips to generate: {len(DRIVING_VIDEOS)}")
    print(f"Output: {output_folder}/")
    print("="*60)

    # Generate all clips
    results = {}
    generated_clips = []

    for clip_name, driving_filename in DRIVING_VIDEOS.items():
        driving_path = os.path.join(DRIVING_VIDEOS_DIR, driving_filename)

        if not os.path.exists(driving_path):
            print(f"WARNING: Driving video not found: {driving_path}")
            results[clip_name] = False
            continue

        success, clip_meta = generate_clip_with_rgba(
            source_image,
            driving_path,
            output_folder,
            clip_name,
            photo_name,
            person_mask,
            feather_px=args.feather,
        )
        results[clip_name] = success

        if success:
            generated_clips.append({
                "name": clip_name,
                "type": "rgba_png_sequence",
                "directory": clip_name,
                "frame_count": clip_meta["frame_count"],
                "fps": clip_meta["fps"],
            })

    # Update metadata
    if metadata:
        metadata["clips"] = generated_clips
        metadata["clip_format"] = "rgba_png_sequence"
        metadata["animations"]["person_clips"] = list(results.keys())
        save_metadata(prepared_photo_dir, metadata)
        print(f"\n✓ Updated metadata.json")

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    successful = sum(results.values())
    total = len(results)
    print(f"Successfully generated: {successful}/{total} clips")

    total_frames = 0
    for clip in generated_clips:
        total_frames += clip["frame_count"]
        print(f"  ✓ {clip['name']}: {clip['frame_count']} frames @ {clip['fps']:.0f}fps")

    # Estimate disk usage
    # Rough estimate: ~200KB per RGBA PNG frame at 512x512ish
    est_size_mb = total_frames * 0.2
    print(f"\nTotal frames: {total_frames} (~{est_size_mb:.0f}MB estimated)")
    print(f"Output: {os.path.abspath(output_folder)}/")
    print(f"\nNext step: python interactive_portrait_v2.py {prepared_photo_dir}")
    print("="*60)


if __name__ == "__main__":
    main()