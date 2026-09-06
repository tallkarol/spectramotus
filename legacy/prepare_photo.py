#!/usr/bin/env python3
"""
Spectramotus Photo Preparation Pipeline

OPTION C4 (Best Quality - Mac Compatible):
1. ClipDrop API inpainting (quality mode)
2. Feathered blending (eliminates hard seams)
3. OpenCV unsharp mask sharpening (fixes blur)

Result: Professional, seamless background - all tools work on Mac

V3 Pipeline Changes:
- Saves original source image as source.png (for LivePortrait pasteback)
- No longer creates person_greenscreen.png (green screen approach eliminated)
- Alpha channel from person.png used directly for compositing
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import requests

# Load .env file for CLIPDROP_API_KEY and HUGGINGFACE_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set HF_TOKEN for Hugging Face authentication (required for RMBG-2.0)
    # rembg uses huggingface_hub which expects HF_TOKEN or HUGGINGFACE_HUB_TOKEN
    hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if hf_api_key and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_api_key
        print("✓ Hugging Face API key loaded from .env (for RMBG-2.0)")
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

# Try to import torch and transformers for direct RMBG-2.0 model access
try:
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation, pipeline
    TORCH_AVAILABLE = True
    DEPTH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    try:
        from transformers import pipeline
        DEPTH_AVAILABLE = True
    except ImportError:
        DEPTH_AVAILABLE = False

OUTPUT_BASE_DIR = "prepared-photos"

def get_photo_name(image_path):
    """Extract clean photo name from image path"""
    return Path(image_path).stem

def refine_mask(mask_pil, original_image, threshold=0.5, morph_kernel_size=5, edge_smooth=3):
    """
    Refine mask to improve person detection quality
    
    Args:
        mask_pil: PIL Image mask (grayscale, 0-255)
        original_image: Original PIL Image for reference
        threshold: Threshold for binarization (0.0-1.0, lower = more inclusive)
        morph_kernel_size: Size of morphological operations kernel
        edge_smooth: Gaussian blur radius for edge smoothing
    
    Returns: Refined PIL Image mask
    """
    mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
    
    # 1. Threshold adjustment (lower threshold = include more pixels)
    mask_array = np.where(mask_array > threshold, mask_array, 0)
    
    # 2. Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_array = cv2.morphologyEx((mask_array * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
    mask_array = mask_array.astype(np.float32) / 255.0
    
    # 3. Edge smoothing
    if edge_smooth > 0:
        blur_size = edge_smooth * 2 + 1
        mask_array = cv2.GaussianBlur(mask_array, (blur_size, blur_size), 0)
    
    result = (mask_array * 255).astype(np.uint8)
    return Image.fromarray(result, mode='L')

def remove_background(image_path, model_name="rmbg-2.0", mask_threshold=0.5,
                      enhance_contrast=1.0, enhance_brightness=1.0):
    """Remove background using rembg with RMBG-2.0 model"""
    print(f"Step 1: Removing background with {model_name}...")
    
    image = Image.open(image_path).convert("RGB")
    
    # Optional: enhance image before background removal
    if enhance_contrast != 1.0 or enhance_brightness != 1.0:
        from PIL import ImageEnhance
        if enhance_contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(enhance_contrast)
        if enhance_brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(enhance_brightness)
        print(f"  Applied enhancement: contrast={enhance_contrast}, brightness={enhance_brightness}")
    
    # Map model names
    model_map = {
        "rmbg-2.0": "briaai/RMBG-2.0",
        "briaai": "briaai/RMBG-2.0",
    }
    actual_model = model_map.get(model_name, model_name)
    
    print(f"  Using model: {actual_model}")
    session = new_session(actual_model)
    result = remove(image, session=session)
    
    # Refine mask if threshold is not default
    if mask_threshold != 0.5:
        print(f"  Refining mask with threshold={mask_threshold}")
        alpha = result.split()[-1]
        refined_alpha = refine_mask(alpha, image, threshold=mask_threshold)
        result.putalpha(refined_alpha)
    
    print(f"✓ Background removed ({result.size[0]}x{result.size[1]})")
    return result

def create_binary_mask(person_rgba):
    """
    Create pure binary mask from alpha channel
    Returns: PIL Image (grayscale), numpy array
    """
    print("Creating binary mask...")
    
    person_array = np.array(person_rgba)
    alpha = person_array[:, :, 3]
    
    # Create binary mask: 
    # - Where person exists (alpha > 10) → 255 (remove/inpaint this area)
    # - Where background is (alpha ≈ 0) → 0 (keep this area)
    binary_mask = ((alpha > 10) * 255).astype(np.uint8)
    
    unique_values = np.unique(binary_mask)
    print(f"  Mask unique values: {unique_values} (should be [0, 255] only)")
    
    mask_pil = Image.fromarray(binary_mask, mode='L')
    
    print(f"✓ Binary mask created")
    return mask_pil, binary_mask

def create_feathered_mask(binary_mask_array, feather_amount=20):
    """
    Create soft/feathered mask for seamless blending
    """
    print(f"  Creating feathered mask (feather: {feather_amount}px)...")
    
    kernel = np.ones((feather_amount, feather_amount), np.uint8)
    dilated = cv2.dilate(binary_mask_array, kernel, iterations=1)
    
    feathered = cv2.GaussianBlur(dilated.astype(float), (feather_amount*2+1, feather_amount*2+1), 0)
    feathered = feathered / 255.0
    
    print(f"✓ Feathered mask created")
    return feathered

def blend_with_feathering(original_image, inpainted_image, feathered_mask):
    """Blend inpainted result with original using soft mask"""
    print("  Blending inpainted area with feathered edges...")
    
    original_array = np.array(original_image).astype(float)
    inpainted_array = np.array(inpainted_image).astype(float)
    
    mask_3d = np.stack([feathered_mask] * 3, axis=2)
    blended = (inpainted_array * mask_3d + original_array * (1 - mask_3d))
    
    print(f"✓ Blended seamlessly")
    return Image.fromarray(blended.astype(np.uint8))

def sharpen_with_opencv(image, mask_array):
    """Apply OpenCV unsharp mask sharpening to fix blur from feathering"""
    print("  Sharpening with OpenCV unsharp mask...")
    
    img_array = np.array(image)
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
    
    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)
    
    mask_3d = np.stack([dilated_mask / 255.0] * 3, axis=2)
    result = (sharpened * mask_3d + img_array * (1 - mask_3d)).astype(np.uint8)
    
    print(f"✓ Sharpened with OpenCV")
    return Image.fromarray(result)

def inpaint_with_clipdrop(original_image, mask_pil, api_key):
    """Use ClipDrop cleanup API for background inpainting"""
    if not api_key:
        print("  ⚠ No ClipDrop API key provided")
        return None
    
    print("  Using ClipDrop cleanup API (quality mode, ~10-15 sec)...")
    
    try:
        import io
        
        img_byte_arr = io.BytesIO()
        original_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        mask_byte_arr = io.BytesIO()
        mask_pil.save(mask_byte_arr, format='PNG')
        mask_bytes = mask_byte_arr.getvalue()
        
        response = requests.post(
            'https://clipdrop-api.co/cleanup/v1',
            headers={'x-api-key': api_key},
            files={
                'image_file': ('image.png', img_bytes, 'image/png'),
                'mask_file': ('mask.png', mask_bytes, 'image/png'),
            },
            data={'mode': 'quality'}
        )
        
        if response.status_code == 200:
            result = Image.open(io.BytesIO(response.content))
            print(f"  ✓ ClipDrop inpainting complete")
            return result
        else:
            print(f"  ✗ ClipDrop API error: {response.status_code} - {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  ✗ ClipDrop error: {e}")
        return None

def inpaint_with_opencv(original_cv, mask_array):
    """Fallback: OpenCV inpainting"""
    print("  Using OpenCV inpainting (Navier-Stokes)...")
    try:
        result = cv2.inpaint(original_cv, mask_array, 3, cv2.INPAINT_NS)
        print(f"✓ OpenCV inpainting complete")
        return Image.fromarray(result)
    except Exception as e:
        print(f"  ✗ OpenCV inpainting failed: {e}")
        return None

def extract_background(original_image_path, person_rgba, mask_pil, mask_array, api_key):
    """Extract and inpaint background"""
    print("Extracting and inpainting background...")
    
    original_image = Image.open(original_image_path).convert("RGB")
    original_cv = np.array(original_image)
    
    if api_key:
        inpainted = inpaint_with_clipdrop(original_image, mask_pil, api_key)
        
        if inpainted:
            print("\nPost-processing for seamless result...")
            feathered_mask = create_feathered_mask(mask_array, feather_amount=20)
            blended = blend_with_feathering(original_image, inpainted, feathered_mask)
            final = sharpen_with_opencv(blended, mask_array)
            print("✓ Post-processing complete\n")
            return final
        
        print("  ClipDrop failed, falling back to OpenCV...")
    
    result = inpaint_with_opencv(original_cv, mask_array)
    if result:
        return result
    
    print("⚠ All inpainting methods failed")
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
        "pipeline_version": "v3",
        "layers": {
            "source": "source.png",
            "person": "person.png",
            "background": "background.png",
            "mask": "mask.png",
            "depth": "depth.png" if os.path.exists(os.path.join(output_dir, "depth.png")) else None
        },
        "animations": {
            "background_effects": ["parallax", "time_of_day"],
            "person_clips": []
        },
        "clip_format": "rgba_png_sequence",
        "clips": []
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved")
    return metadata

def prepare_photo(image_path, skip_depth=False, api_key=None, bg_model="u2net_human_seg", 
                  mask_threshold=0.5, enhance_contrast=1.0, enhance_brightness=1.0):
    """
    Main preparation pipeline
    
    Args:
        image_path: Path to source image
        skip_depth: Skip depth map generation
        api_key: ClipDrop API key for background inpainting
        bg_model: Background removal model name (default: "rmbg-2.0" for RMBG-2.0)
        mask_threshold: Mask threshold (0.0-1.0). Lower = more inclusive, higher = more conservative
        enhance_contrast: Contrast enhancement factor (1.0 = no change, >1.0 = more contrast)
        enhance_brightness: Brightness enhancement factor (1.0 = no change, >1.0 = brighter)
    """
    
    print("\n" + "="*60)
    print("SPECTRAMOTUS PHOTO PREPARATION (V3)")
    print("="*60)
    print(f"Source: {image_path}")
    print(f"Background removal: {bg_model}")
    if bg_model == "rmbg-2.0":
        print("  → Using RMBG-2.0 (BRIA AI) - State-of-the-art quality")
    if mask_threshold != 0.5:
        print(f"  → Mask threshold: {mask_threshold:.2f} (lower = more inclusive)")
    if enhance_contrast != 1.0 or enhance_brightness != 1.0:
        print(f"  → Image enhancement: contrast={enhance_contrast:.2f}, brightness={enhance_brightness:.2f}")
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
        # Step 1a: Remove background using RMBG-2.0
        person_rgba = remove_background(
            image_path, 
            model_name=bg_model,
            mask_threshold=mask_threshold,
            enhance_contrast=enhance_contrast,
            enhance_brightness=enhance_brightness
        )
        person_path = os.path.join(output_dir, "person.png")
        person_rgba.save(person_path)
        print(f"  → Saved: {person_path}")
        
        # Step 1b: Save original source image
        # LivePortrait needs the full original image for pasteback compositing.
        # We copy it into the prepared directory so the pipeline is self-contained.
        source_dest = os.path.join(output_dir, "source.png")
        source_image = Image.open(image_path).convert("RGB")
        source_image.save(source_dest)
        print(f"  → Saved: {source_dest}")
        
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
        print(f"  ✓ source.png - Original source image (for LivePortrait)")
        print(f"  ✓ person.png - Person with alpha channel (RMBG-2.0)")
        print(f"  ✓ mask.png - Binary mask (0=keep, 255=remove)")
        print(f"  ✓ background.png - Inpainted background")
        if api_key:
            print(f"      (C4: ClipDrop + Feathered Blending + OpenCV Sharpening)")
        if not skip_depth and os.path.exists(os.path.join(output_dir, "depth.png")):
            print(f"  ✓ depth.png - Depth map")
        print(f"  ✓ metadata.json - Configuration")
        print(f"\nNext step:")
        print(f"  python batch_generate_v3.py {output_dir}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Prepare photos for animation - RMBG-2.0 → ClipDrop API → OpenCV fallback"
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
    parser.add_argument(
        "--bg-model",
        default="u2net_human_seg",
        help="Background removal model (default: u2net_human_seg). Options: u2net_human_seg, rmbg-2.0, briaai, u2net, u2netp, u2net_cloth_seg, silueta, isnet-general-use"
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold (0.0-1.0). Lower = more inclusive (captures more of person), higher = more conservative. Default: 0.5"
    )
    parser.add_argument(
        "--enhance-contrast",
        type=float,
        default=1.0,
        help="Contrast enhancement factor (1.0 = no change, >1.0 = more contrast). Can help with detection. Default: 1.0"
    )
    parser.add_argument(
        "--enhance-brightness",
        type=float,
        default=1.0,
        help="Brightness enhancement factor (1.0 = no change, >1.0 = brighter). Can help with detection. Default: 1.0"
    )
    
    args = parser.parse_args()
    
    success = prepare_photo(
        args.image_path,
        skip_depth=args.skip_depth,
        api_key=args.clipdrop_api_key,
        bg_model=args.bg_model,
        mask_threshold=args.mask_threshold,
        enhance_contrast=args.enhance_contrast,
        enhance_brightness=args.enhance_brightness
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()