#!/usr/bin/env python3
"""
Spectramotus Batch LivePortrait Generator V2
Processes person layers from prepared photos with multiple driving videos

This version expects photos to be prepared first with prepare_photo.py

Usage:
    python batch_generate_v2.py prepared-photos/karol-photo
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import argparse

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
    """Extract photo name from directory"""
    return Path(prepared_photo_dir).name


def get_python_executable():
    """Get the Python executable to use, preferring conda Python if available"""
    # Check if we're in a conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(conda_python):
            return conda_python
    # Fall back to sys.executable
    return sys.executable

def generate_clip(source, driving, output_folder, clip_name, photo_name):
    """Run LivePortrait for one source+driving combination"""
    final_filename = f"{photo_name}_{clip_name}_person.mp4"
    
    print(f"\n{'='*60}")
    print(f"Generating: {final_filename}")
    print(f"{'='*60}")
    
    # LivePortrait outputs to a temp directory first
    temp_output = os.path.join("temp_output", clip_name)
    
    python_exe = get_python_executable()
    cmd = [
        python_exe, "inference.py",
        "--source", source,
        "--driving", driving,
        "--output-dir", temp_output
    ]
    
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    try:
        subprocess.run(cmd, env=env, check=True)
        
        # LivePortrait creates multiple files:
        # - {name}_concat.mp4 (3-panel view) - DON'T USE THIS
        # - {name}.mp4 (animated result only) - USE THIS
        temp_files = list(Path(temp_output).glob("*.mp4"))
        
        # Filter out concat files
        non_concat_files = [f for f in temp_files if '_concat' not in f.name]
        
        if not non_concat_files:
            print(f"✗ No non-concat output file found in {temp_output}")
            print(f"Available files: {[f.name for f in temp_files]}")
            return False, None
        
        # Move and rename to our convention
        source_file = non_concat_files[0]
        destination = os.path.join(output_folder, final_filename)
        shutil.move(str(source_file), destination)
        
        # Clean up temp directory
        shutil.rmtree(temp_output, ignore_errors=True)
        
        print(f"✓ Successfully generated: {final_filename}")
        return True, final_filename
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate: {final_filename}")
        print(f"Error: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(
        description="Generate animation clips from prepared photos"
    )
    parser.add_argument(
        "prepared_photo_dir",
        help="Path to prepared photo directory (e.g., prepared-photos/karol-photo)"
    )
    
    args = parser.parse_args()
    prepared_photo_dir = args.prepared_photo_dir
    
    # Validate input
    if not os.path.exists(prepared_photo_dir):
        print(f"ERROR: Prepared photo directory not found: {prepared_photo_dir}")
        print("Run prepare_photo.py first!")
        sys.exit(1)
    
    # Load metadata
    metadata = load_metadata(prepared_photo_dir)
    
    # Get paths
    photo_name = get_photo_name(prepared_photo_dir)
    person_layer_path = os.path.join(prepared_photo_dir, "person_greenscreen.png")
    
    if not os.path.exists(person_layer_path):
        print(f"ERROR: Greenscreen source not found: {person_layer_path}")
        print("Run prepare_photo.py first to create person_greenscreen.png!")
        print("Or run: python convert_to_chroma_key.py {prepared_photo_dir}")
        sys.exit(1)
    
    # Create output directory
    output_folder = os.path.join(OUTPUT_BASE_DIR, photo_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SPECTRAMOTUS BATCH GENERATOR V2")
    print("="*60)
    print(f"Source greenscreen: {person_layer_path}")
    print(f"Photo name: {photo_name}")
    print(f"Number of clips to generate: {len(DRIVING_VIDEOS)}")
    print(f"Output folder: {output_folder}/")
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
        
        success, filename = generate_clip(
            person_layer_path,
            driving_path,
            output_folder,
            clip_name,
            photo_name
        )
        results[clip_name] = success
        
        if success:
            generated_clips.append({
                "name": clip_name,
                "type": "person_layer",
                "file": filename,
                "has_background": False
            })
    
    # Update metadata with generated clips
    if metadata:
        metadata["clips"] = generated_clips
        metadata["animations"]["person_clips"] = list(results.keys())
        save_metadata(prepared_photo_dir, metadata)
        print(f"\n✓ Updated metadata.json with clip information")
    
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
        filename = f"{photo_name}_{clip_name}_person.mp4"
        print(f"  {status} {filename}")
    
    print(f"\nAll clips saved to:")
    print(f"  {os.path.abspath(output_folder)}/")
    print("\nNaming convention: {photo_name}_{gesture}_person.mp4")
    print("\nNext step: Run interactive_portrait_layered.py to see the result!")
    print("="*60)

if __name__ == "__main__":
    main()