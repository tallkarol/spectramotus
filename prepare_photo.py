#!/usr/bin/env python3
"""
Spectramotus Photo Preparation Pipeline

OPTION C4 (Best Quality - Mac Compatible):
1. ClipDrop API inpainting (quality mode)
2. Feathered blending (eliminates hard seams)
3. OpenCV unsharp mask sharpening (fixes blur)

Result: Professional, seamless background - all tools work on Mac
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
import requests

# Load .env file for CLIPDROP_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

import cv2
import numpy as np
from PIL import Image
from rembg import remove

# Depth estimation (optional)
try:
    from transformers import pipeline
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False

OUTPUT_BASE_DIR = "prepared-photos"

def get_photo_name(photo_path):
    """Extract base name without extension"""
    return Path(photo_path).stem

def remove_background(image_path):
    """
    Remove background using RMBG
    Returns: PIL Image with alpha channel
    """
    print("Removing background...")
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    print(f"✓ Background removed (output size: {output_image.size})")
    return output_image

def create_binary_mask(person_rgba):
    """
    Create pure binary mask from alpha channel
    Returns: PIL Image (grayscale), numpy array
    """
    print("Creating binary mask...")
    
    # Extract alpha channel
    person_array = np.array(person_rgba)
    alpha = person_array[:, :, 3]
    
    # Create binary mask: 
    # - Where person exists (alpha > 10) → 255 (remove/inpaint this area)
    # - Where background is (alpha ≈ 0) → 0 (keep this area)
    binary_mask = ((alpha > 10) * 255).astype(np.uint8)
    
    # Verify it's pure binary
    unique_values = np.unique(binary_mask)
    print(f"  Mask unique values: {unique_values} (should be [0, 255] only)")
    
    if len(unique_values) > 2:
        print(f"  ⚠ Warning: Mask has {len(unique_values)} unique values, expected 2")
    
    mask_pil = Image.fromarray(binary_mask, mode='L')
    
    print(f"✓ Binary mask created")
    return mask_pil, binary_mask

def create_feathered_mask(binary_mask_array, feather_amount=20):
    """
    Create soft/feathered mask for seamless blending
    
    Args:
        binary_mask_array: Hard binary mask (0 or 255)
        feather_amount: Pixels to feather/blur at edges
    
    Returns:
        Soft mask (0.0 to 1.0 float values)
    """
    print(f"  Creating feathered mask (feather: {feather_amount}px)...")
    
    # Dilate mask slightly to expand the blend zone
    kernel = np.ones((feather_amount, feather_amount), np.uint8)
    dilated = cv2.dilate(binary_mask_array, kernel, iterations=1)
    
    # Apply Gaussian blur to create soft transition
    feathered = cv2.GaussianBlur(dilated.astype(float), (feather_amount*2+1, feather_amount*2+1), 0)
    
    # Normalize to 0.0-1.0
    feathered = feathered / 255.0
    
    print(f"✓ Feathered mask created")
    return feathered

def blend_with_feathering(original_image, inpainted_image, feathered_mask):
    """
    Blend inpainted result with original using soft mask
    Eliminates hard seams/edges
    """
    print("  Blending inpainted area with feathered edges...")
    
    original_array = np.array(original_image).astype(float)
    inpainted_array = np.array(inpainted_image).astype(float)
    
    # Expand mask to 3 channels (RGB)
    mask_3d = np.stack([feathered_mask] * 3, axis=2)
    
    # Blend: inpainted where mask=1, original where mask=0, smooth transition in between
    blended = (inpainted_array * mask_3d + original_array * (1 - mask_3d))
    
    print(f"✓ Blended seamlessly")
    return Image.fromarray(blended.astype(np.uint8))

def sharpen_with_opencv(image, mask_array):
    """
    Apply OpenCV unsharp mask sharpening to fix blur from feathering
    Works perfectly on Mac, no special dependencies needed
    """
    print("  Sharpening with OpenCV unsharp mask...")
    
    img_array = np.array(image)
    
    # Create blurred version using Gaussian blur
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    
    # Unsharp mask formula: sharpened = original + amount * (original - blurred)
    # Using addWeighted for efficiency: original * 1.5 + blurred * -0.5
    sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
    
    # Apply sharpening primarily to the inpainted area
    # Dilate mask to include the feathered transition zone
    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)
    
    # Blend sharpened and original based on mask
    mask_3d = np.stack([dilated_mask / 255.0] * 3, axis=2)
    result = (sharpened * mask_3d + img_array * (1 - mask_3d)).astype(np.uint8)
    
    print(f"✓ Sharpened with OpenCV")
    return Image.fromarray(result)

def inpaint_with_clipdrop(original_image, mask_pil, api_key):
    """
    Use ClipDrop cleanup API - professional object removal
    
    API Doc: https://clipdrop.co/apis/docs/cleanup
    - mask: 0 = keep, 255 = remove/inpaint
    - mode: 'quality' for best results (slower)
    """
    if not api_key:
        print("  ⚠ No ClipDrop API key provided")
        return None
    
    print("  Using ClipDrop cleanup API (quality mode, ~10-15 sec)...")
    
    try:
        import io
        
        # Convert image to PNG bytes
        img_byte_arr = io.BytesIO()
        original_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Convert mask to PNG bytes
        mask_byte_arr = io.BytesIO()
        mask_pil.save(mask_byte_arr, format='PNG')
        mask_bytes = mask_byte_arr.getvalue()
        
        # API request
        url = "https://clipdrop-api.co/cleanup/v1"
        
        files = {
            'image_file': ('image.png', img_bytes, 'image/png'),
            'mask_file': ('mask.png', mask_bytes, 'image/png')
        }
        
        data = {
            'mode': 'quality'  # Best quality, slower
        }
        
        headers = {
            'x-api-key': api_key
        }
        
        print(f"    Request details:")
        print(f"      - Image size: {original_image.size}")
        print(f"      - Mode: quality")
        print(f"      - Timeout: 90s")
        
        response = requests.post(url, files=files, data=data, headers=headers, timeout=90)
        
        if response.status_code == 200:
            result = Image.open(io.BytesIO(response.content))
            print(f"✓ Background inpainted using ClipDrop API")
            return result
        else:
            print(f"⚠ ClipDrop API failed: HTTP {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"⚠ ClipDrop API error: {e}")
        import traceback
        traceback.print_exc()
        return None

def inpaint_with_opencv(original_cv, mask_array):
    """
    OpenCV inpainting - basic fallback
    """
    print("  Using OpenCV inpainting (basic quality)...")
    try:
        inpainted = cv2.inpaint(
            original_cv,
            mask_array,
            inpaintRadius=10,
            flags=cv2.INPAINT_TELEA
        )
        print(f"✓ Background inpainted using OpenCV")
        return Image.fromarray(inpainted)
    except Exception as e:
        print(f"⚠ OpenCV inpainting failed: {e}")
        return None

def extract_background(original_image_path, person_rgba, mask_pil, mask_array, api_key=None):
    """
    Inpaint background with C4 Pipeline (Mac compatible):
    1. ClipDrop API inpainting
    2. Feathered blending (smooth seams)
    3. OpenCV sharpening (fix blur)
    
    Fallback: OpenCV if no API key
    """
    print("Extracting and inpainting background...")
    
    # Load original image
    original_image = Image.open(original_image_path).convert("RGB")
    original_cv = np.array(original_image)
    
    # Try ClipDrop first (if API key available)
    if api_key:
        inpainted = inpaint_with_clipdrop(original_image, mask_pil, api_key)
        
        if inpainted:
            # OPTION C4 POST-PROCESSING PIPELINE
            print("\nPost-processing for seamless result...")
            
            # Step 1: Create feathered mask
            feathered_mask = create_feathered_mask(mask_array, feather_amount=20)
            
            # Step 2: Blend with feathering (eliminates seams)
            blended = blend_with_feathering(original_image, inpainted, feathered_mask)
            
            # Step 3: Sharpen with OpenCV (fixes blur from feathering)
            final = sharpen_with_opencv(blended, mask_array)
            
            print("✓ Post-processing complete\n")
            return final
        
        print("  ClipDrop failed, falling back to OpenCV...")
    
    # Fallback to OpenCV
    result = inpaint_with_opencv(original_cv, mask_array)
    if result:
        return result
    
    # Last resort: return background with person area blacked out
    print("⚠ All inpainting methods failed")
    print("  Returning background with masked area")
    person_array = np.array(person_rgba)
    alpha = person_array[:, :, 3]
    background_mask = 1.0 - (alpha / 255.0)
    mask_3d = np.stack([background_mask] * 3, axis=2)
    fallback = (original_cv * mask_3d).astype(np.uint8)
    return Image.fromarray(fallback)

def generate_depth_map(image_path):
    """Generate depth map using Depth-Anything-V2"""
    if not DEPTH_AVAILABLE:
        return None
    
    print("Generating depth map (10-30 seconds)...")
    try:
        depth_estimator = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        image = Image.open(image_path)
        depth_result = depth_estimator(image)
        depth_map = depth_result["depth"]
        print(f"✓ Depth map generated")
        return depth_map
    except Exception as e:
        print(f"⚠ Depth map generation failed: {e}")
        return None

def create_metadata(photo_name, source_path, dimensions, output_dir):
    """Create metadata.json"""
    metadata = {
        "source": source_path,
        "photo_name": photo_name,
        "dimensions": {"width": dimensions[0], "height": dimensions[1]},
        "aspect_ratio": dimensions[0] / dimensions[1],
        "prepared_at": datetime.now().isoformat(),
        "layers": {
            "person": "person.png",
            "background": "background.png",
            "mask": "mask.png",
            "depth": "depth.png" if os.path.exists(os.path.join(output_dir, "depth.png")) else None
        },
        "animations": {
            "background_effects": ["parallax", "time_of_day"],
            "person_clips": []
        },
        "clips": []
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved")
    return metadata

def prepare_photo(image_path, skip_depth=False, api_key=None):
    """Main preparation pipeline"""
    
    print("\n" + "="*60)
    print("SPECTRAMOTUS PHOTO PREPARATION")
    print("="*60)
    print(f"Source: {image_path}")
    if api_key:
        print(f"C4 Pipeline: ClipDrop + Feathered Blending + OpenCV Sharpening")
    else:
        print(f"Inpainting: OpenCV fallback only (no API key)")
    print()
    
    # Validate input
    if not os.path.exists(image_path):
        print(f"ERROR: Source photo not found: {image_path}")
        return False
    
    # Create output directory
    photo_name = get_photo_name(image_path)
    output_dir = os.path.join(OUTPUT_BASE_DIR, photo_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    try:
        # Step 1: Remove background
        person_rgba = remove_background(image_path)
        person_path = os.path.join(output_dir, "person.png")
        person_rgba.save(person_path)
        print(f"  → Saved: {person_path}")
        
        # Step 1b: Create greenscreen source (person composited onto green background)
        # This is used for video generation with chroma key backgrounds
        green_bg = Image.new("RGB", person_rgba.size, (0, 255, 0))  # Bright green
        greenscreen = Image.alpha_composite(
            green_bg.convert("RGBA"),
            person_rgba
        ).convert("RGB")
        greenscreen_path = os.path.join(output_dir, "person_greenscreen.png")
        greenscreen.save(greenscreen_path)
        print(f"  → Saved: {greenscreen_path}")
        
        # Step 2: Create binary mask
        mask_pil, mask_array = create_binary_mask(person_rgba)
        mask_path = os.path.join(output_dir, "mask.png")
        mask_pil.save(mask_path)
        print(f"  → Saved: {mask_path}")
        
        # Step 3: Inpaint background
        background_image = extract_background(
            image_path,
            person_rgba,
            mask_pil,
            mask_array,
            api_key
        )
        background_path = os.path.join(output_dir, "background.png")
        background_image.save(background_path)
        print(f"  → Saved: {background_path}")
        
        # Step 4: Generate depth map (optional)
        if not skip_depth:
            depth_map = generate_depth_map(image_path)
            if depth_map:
                depth_path = os.path.join(output_dir, "depth.png")
                depth_map.save(depth_path)
                print(f"  → Saved: {depth_path}")
        
        # Step 5: Create metadata
        dimensions = person_rgba.size
        create_metadata(photo_name, image_path, dimensions, output_dir)
        
        # Summary
        print("\n" + "="*60)
        print("PREPARATION COMPLETE")
        print("="*60)
        print(f"Photo name: {photo_name}")
        print(f"Files created:")
        print(f"  ✓ person.png - Person with alpha channel")
        print(f"  ✓ person_greenscreen.png - Person on green chroma key background")
        print(f"  ✓ mask.png - Binary mask (0=keep, 255=remove)")
        print(f"  ✓ background.png - Inpainted background")
        if api_key:
            print(f"      (C4: ClipDrop + Feathered Blending + OpenCV Sharpening)")
            print(f"      → Eliminates seams, smooths transitions, fixes blur")
        if not skip_depth and os.path.exists(os.path.join(output_dir, "depth.png")):
            print(f"  ✓ depth.png - Depth map")
        print(f"  ✓ metadata.json - Configuration")
        print(f"\nNext step:")
        print(f"  python batch_generate_v2.py {output_dir}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Prepare photos for animation - RMBG → ClipDrop API → OpenCV fallback"
    )
    parser.add_argument(
        "image_path",
        help="Path to source photo (e.g., source-photos/photo.png)"
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth map generation (faster)"
    )
    parser.add_argument(
        "--clipdrop-api-key",
        help="ClipDrop API key (or set CLIPDROP_API_KEY in .env)",
        default=os.environ.get("CLIPDROP_API_KEY")
    )
    
    args = parser.parse_args()
    
    success = prepare_photo(
        args.image_path,
        skip_depth=args.skip_depth,
        api_key=args.clipdrop_api_key
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()