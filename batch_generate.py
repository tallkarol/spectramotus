#!/usr/bin/env python3
"""
SpectramotusBatch LivePortrait Generator
Processes one source photo with multiple driving videos
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
SOURCE_PHOTO = "source-photos/karol-photo.png"
OUTPUT_BASE_DIR = "generated_clips"

# Define your driving videos and output clip names
DRIVING_VIDEOS = {
    "idle": "driving-videos/base-position.mp4",
    "look_left": "driving-videos/look-left.mp4",
    "look_right": "driving-videos/look-right.mp4",
    "smile": "driving-videos/smile.mp4",
    "sup_dude": "driving-videos/sup-dude.mp4",
}

def get_photo_name(photo_path):
    """Extract base name without extension"""
    return Path(photo_path).stem

def generate_clip(source, driving, output_folder, clip_name):
    """Run LivePortrait for one source+driving combination"""
    photo_name = get_photo_name(source)
    final_filename = f"{photo_name}_{clip_name}.mp4"
    
    print(f"\n{'='*60}")
    print(f"Generating: {final_filename}")
    print(f"{'='*60}")
    
    # LivePortrait outputs to a temp directory first
    temp_output = os.path.join("temp_output", clip_name)
    
    cmd = [
        "python", "inference.py",
        "--source", source,
        "--driving", driving,
        "--output-dir", temp_output
    ]
    
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    try:
        subprocess.run(cmd, env=env, check=True)
        
        # Find the generated video (LivePortrait creates it with its own naming)
        temp_files = list(Path(temp_output).glob("*.mp4"))
        if not temp_files:
            print(f"✗ No output file found in {temp_output}")
            return False
        
        # Move and rename to our convention
        source_file = temp_files[0]
        destination = os.path.join(output_folder, final_filename)
        shutil.move(str(source_file), destination)
        
        # Clean up temp directory
        shutil.rmtree(temp_output, ignore_errors=True)
        
        print(f"✓ Successfully generated: {final_filename}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate: {final_filename}")
        print(f"Error: {e}")
        return False

def main():
    photo_name = get_photo_name(SOURCE_PHOTO)
    output_folder = os.path.join(OUTPUT_BASE_DIR, photo_name)
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Spectramotus Batch Generator")
    print("="*60)
    print(f"Source photo: {SOURCE_PHOTO}")
    print(f"Photo name: {photo_name}")
    print(f"Number of clips to generate: {len(DRIVING_VIDEOS)}")
    print(f"Output folder: {output_folder}/")
    print("="*60)
    
    # Check source photo exists
    if not os.path.exists(SOURCE_PHOTO):
        print(f"ERROR: Source photo not found: {SOURCE_PHOTO}")
        sys.exit(1)
    
    # Generate all clips
    results = {}
    for clip_name, driving_path in DRIVING_VIDEOS.items():
        if not os.path.exists(driving_path):
            print(f"WARNING: Driving video not found: {driving_path}")
            results[clip_name] = False
            continue
            
        success = generate_clip(SOURCE_PHOTO, driving_path, output_folder, clip_name)
        results[clip_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    successful = sum(results.values())
    total = len(results)
    print(f"Successfully generated: {successful}/{total} clips")
    
    print("\nGenerated clips:")
    for clip_name, success in results.items():
        status = "✓" if success else "✗"
        filename = f"{photo_name}_{clip_name}.mp4"
        print(f"  {status} {filename}")
    
    print(f"\nAll clips saved to:")
    print(f"  {os.path.abspath(output_folder)}/")
    print("\nNaming convention: {photo_name}_{gesture}.mp4")
    print("="*60)

if __name__ == "__main__":
    main()