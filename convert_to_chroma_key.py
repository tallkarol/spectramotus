#!/usr/bin/env python3
"""
Create greenscreen source images from existing person.png files.

This generates person_greenscreen.png by compositing person.png (with alpha)
onto a bright green background. This allows regenerating videos with proper
chroma key backgrounds without re-running the full photo preparation.

Usage:
    python convert_to_chroma_key.py prepared-photos/karol-linkedin/
    python convert_to_chroma_key.py prepared-photos/
"""

import os
import sys
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

# Chroma key color (bright green in RGB)
CHROMA_KEY_COLOR = (0, 255, 0)


def create_greenscreen_source(prepared_photo_dir):
    """Create person_greenscreen.png from person.png in a prepared photo directory."""
    person_path = os.path.join(prepared_photo_dir, "person.png")
    greenscreen_path = os.path.join(prepared_photo_dir, "person_greenscreen.png")
    
    if not os.path.exists(person_path):
        print(f"  ⚠ person.png not found in {prepared_photo_dir}")
        return False
    
    print(f"Creating greenscreen source: {greenscreen_path}")
    
    # Load person image with alpha channel
    person_rgba = Image.open(person_path).convert("RGBA")
    width, height = person_rgba.size
    
    # Create green background
    green_bg = Image.new("RGB", (width, height), CHROMA_KEY_COLOR)
    
    # Composite person onto green background using alpha channel
    greenscreen = Image.alpha_composite(
        green_bg.convert("RGBA"),
        person_rgba
    ).convert("RGB")
    
    # Save as RGB (no alpha needed since background is green)
    greenscreen.save(greenscreen_path)
    print(f"  ✓ Created {greenscreen_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create greenscreen source images from person.png files"
    )
    parser.add_argument(
        "path",
        help="Path to prepared photo directory or parent directory"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_dir():
        # Check if this is a prepared photo directory (has person.png)
        person_png = path / "person.png"
        if person_png.exists():
            # Single prepared photo directory
            create_greenscreen_source(str(path))
        else:
            # Parent directory - find all prepared photo directories
            prepared_dirs = [d for d in path.iterdir() if d.is_dir() and (d / "person.png").exists()]
            if not prepared_dirs:
                print(f"No prepared photo directories found in {path}")
                sys.exit(1)
            
            print(f"Found {len(prepared_dirs)} prepared photo directories")
            print("=" * 60)
            
            for prep_dir in prepared_dirs:
                create_greenscreen_source(str(prep_dir))
            
            print("=" * 60)
            print(f"✓ Created greenscreen sources for {len(prepared_dirs)} directories")
    else:
        print(f"Path not found or not a directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
