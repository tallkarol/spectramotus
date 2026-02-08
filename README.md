# Spectramotus

An interactive portrait animation system that creates animated portraits from photos using LivePortrait, with gesture-controlled animations and dynamic background compositing.

## Features

- **Photo Preparation**: Remove backgrounds, create masks, and generate depth maps
- **Video Generation**: Create animated portrait clips from prepared photos
- **Interactive Playback**: Control animations with hand gestures via webcam
- **Chroma Key Compositing**: Clean transparency using green screen backgrounds
- **Dynamic Backgrounds**: Animated backgrounds with parallax and time-of-day effects

## Prerequisites

- **Python 3.9+**
- **FFmpeg** (for video processing)
- **Conda** (recommended) or pip
- **ClipDrop API Key** (optional, for high-quality background inpainting)

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg
```

## Installation

### Option 1: Using Conda (Recommended)

1. **Create and activate conda environment:**
```bash
conda create -n spectramotus python=3.9
conda activate spectramotus
```

2. **Install dependencies:**

**macOS (Apple Silicon):**
```bash
pip install -r requirements_macOS.txt
```

**macOS (Intel) / Windows:**
```bash
pip install -r requirements.txt
```

3. **Install base requirements:**
```bash
pip install -r requirements_base.txt
```

4. **Install additional dependencies:**
```bash
pip install -r requirements_additional.txt
```

### Option 2: Using pip only

**macOS (Apple Silicon):**
```bash
python3 -m pip install -r requirements_macOS.txt
python3 -m pip install -r requirements_base.txt
python3 -m pip install -r requirements_additional.txt
```

**macOS (Intel) / Windows:**
```bash
pip install -r requirements.txt
pip install -r requirements_base.txt
pip install -r requirements_additional.txt
```

### Environment Variables

Create a `.env` file in the project root (optional, for ClipDrop API):

```bash
CLIPDROP_API_KEY=your_api_key_here
```

## Complete Installation Checklist

### macOS

- [ ] Install FFmpeg: `brew install ffmpeg`
- [ ] Create conda environment: `conda create -n spectramotus python=3.9`
- [ ] Activate environment: `conda activate spectramotus`
- [ ] Install macOS requirements: `pip install -r requirements_macOS.txt`
- [ ] Install base requirements: `pip install -r requirements_base.txt`
- [ ] Install additional requirements: `pip install -r requirements_additional.txt`
- [ ] (Optional) Create `.env` file with `CLIPDROP_API_KEY`

### Windows

- [ ] Install Python 3.9+ from python.org
- [ ] Install FFmpeg and add to PATH
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate environment: `venv\Scripts\activate`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Install base requirements: `pip install -r requirements_base.txt`
- [ ] Install additional requirements: `pip install -r requirements_additional.txt`
- [ ] (Optional) Create `.env` file with `CLIPDROP_API_KEY`

## Quick Start

### 1. Prepare a Photo

```bash
python prepare_photo.py source-photos/your-photo.jpg
```

**Note:** By default, this uses **RMBG-2.0** (BRIA AI) for state-of-the-art background removal quality.

This creates:
- `prepared-photos/your-photo/person.png` - Person with alpha channel
- `prepared-photos/your-photo/person_greenscreen.png` - Person on green background
- `prepared-photos/your-photo/background.png` - Inpainted background
- `prepared-photos/your-photo/mask.png` - Binary mask
- `prepared-photos/your-photo/depth.png` - Depth map (optional)

**With ClipDrop API (better quality):**
```bash
python prepare_photo.py source-photos/your-photo.jpg --clipdrop-api-key YOUR_KEY
```

**Skip depth map generation (faster):**
```bash
python prepare_photo.py source-photos/your-photo.jpg --skip-depth
```

### 2. Generate Animation Clips

```bash
python batch_generate.py prepared-photos/your-photo
```

This generates animated clips for:
- `idle` - Base position
- `look_left` - Looking left
- `look_right` - Looking right
- `smile` - Smiling
- `sup_dude` - Wave gesture

Output: `generated_clips/your-photo/*_person.mp4`

### 3. Run Interactive Portrait

```bash
python interactive_portrait.py prepared-photos/your-photo
```

## Usage Guide

### Photo Preparation

**Basic usage:**
```bash
python prepare_photo.py <source_image_path>
```

**Options:**
- `--skip-depth` - Skip depth map generation (faster)
- `--clipdrop-api-key KEY` - Use ClipDrop API for better background inpainting
- `--bg-model MODEL` - Background removal model (default: `rmbg-2.0`)

**Background Removal Models:**
- `rmbg-2.0` (default) - BRIA AI's RMBG v2.0, state-of-the-art quality
- `briaai` - Alternative name for RMBG-2.0
- `u2net` - Original U2Net model
- `u2netp` - Lightweight U2Net model
- `u2net_human_seg` - Human segmentation model
- `u2net_cloth_seg` - Clothing segmentation model
- `silueta` - Silueta model
- `isnet-general-use` - ISNet general use model

**Examples:**
```bash
# Using default RMBG-2.0 (recommended)
python prepare_photo.py source-photos/portrait.jpg

# With ClipDrop API for better background inpainting
python prepare_photo.py source-photos/portrait.jpg --clipdrop-api-key sk-xxx

# Using different background removal model
python prepare_photo.py source-photos/portrait.jpg --bg-model u2net_human_seg

# Skip depth map for faster processing
python prepare_photo.py source-photos/portrait.jpg --skip-depth
```

### Batch Video Generation

**Basic usage:**
```bash
python batch_generate.py <prepared_photo_directory>
```

**Example:**
```bash
python batch_generate.py prepared-photos/karol-linkedin
```

**What it does:**
- Uses `person_greenscreen.png` as source
- Generates videos with green chroma key backgrounds
- Saves to `generated_clips/<photo_name>/`

### Interactive Portrait Player

**Basic usage:**
```bash
python interactive_portrait.py <prepared_photo_directory>
```

**Controls:**
- **👍 Thumbs up** → Smile animation
- **👋 Wave** → Sup dude animation
- **👈 Point left** → Look left animation
- **👉 Point right** → Look right animation
- **W** → Toggle webcam preview
- **U** → Toggle UI overlay
- **F** → Toggle fullscreen
- **Q/ESC** → Quit

### Utility Scripts

**Create greenscreen sources for existing prepared photos:**
```bash
python convert_to_chroma_key.py prepared-photos/your-photo
```

Or for all prepared photos:
```bash
python convert_to_chroma_key.py prepared-photos/
```

## Project Structure

```
spectramotus/
├── source-photos/          # Input photos
├── prepared-photos/        # Processed photos with layers
│   └── <photo-name>/
│       ├── person.png
│       ├── person_greenscreen.png
│       ├── background.png
│       ├── mask.png
│       ├── depth.png
│       └── metadata.json
├── generated_clips/        # Generated animation videos
│   └── <photo-name>/
│       └── *_person.mp4
├── driving-videos/         # Animation driving videos
├── prepare_photo.py        # Photo preparation script
├── batch_generate.py       # Batch video generation
├── interactive_portrait.py # Interactive player
├── convert_to_chroma_key.py # Utility for greenscreen conversion
└── requirements*.txt       # Dependencies
```

## Dependencies

### Core Libraries

- **numpy** (1.26.4) - Numerical computing
- **opencv-python** (4.10.0.84) - Image/video processing
- **Pillow** (≥10.2.0) - Image manipulation
- **scipy** (1.13.1) - Scientific computing
- **scikit-image** (0.24.0) - Image processing
- **albumentations** (1.4.10) - Image augmentation

### ML/AI Libraries

- **torch** (2.3.0) - PyTorch (CPU/GPU)
- **torchvision** (0.18.0) - Computer vision
- **onnx** (1.16.1) - ONNX model format
- **onnxruntime** - ONNX runtime (platform-specific)
- **transformers** (4.38.0) - Hugging Face transformers

### Video/Media

- **ffmpeg-python** (0.2.0) - FFmpeg wrapper
- **imageio** (2.34.2) - Image I/O
- **imageio-ffmpeg** (0.5.1) - FFmpeg plugin

### UI/CLI

- **gradio** (5.1.0) - Web UI framework
- **rich** (13.7.1) - Terminal formatting
- **tyro** (0.8.5) - CLI argument parsing
- **pygame** - Interactive display (installed separately)

### Other

- **pyyaml** (6.0.1) - YAML parsing
- **tqdm** (4.66.4) - Progress bars
- **matplotlib** (3.9.0) - Plotting
- **pykalman** (0.9.7) - Kalman filtering
- **lmdb** (1.4.1) - Database
- **rembg** - Background removal (supports RMBG-2.0 model)
- **mediapipe** - Hand gesture detection (installed separately)
- **python-dotenv** - Environment variables (installed separately)

## Platform-Specific Notes

### macOS (Apple Silicon)

- Uses `onnxruntime-silicon` for M1/M2/M3 optimization
- PyTorch CPU version recommended (GPU support via MPS)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for compatibility

### macOS (Intel)

- Use standard `requirements.txt`
- May need to install PyTorch with CUDA support if using GPU

### Windows

**Installation steps:**

1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/)

2. **Install FFmpeg:**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract and add `bin` folder to PATH
   - Or use Chocolatey: `choco install ffmpeg`

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements_base.txt
pip install -r requirements_additional.txt
```

**Notes:**
- May need Visual C++ Redistributables for some packages
- Use `onnxruntime-gpu` if you have CUDA GPU, otherwise use CPU version
- For GPU support, install CUDA-compatible PyTorch from [pytorch.org](https://pytorch.org/)

## Troubleshooting

### ModuleNotFoundError: No module named 'tyro'

**Solution:** Install dependencies:
```bash
pip install -r requirements_base.txt
pip install -r requirements_additional.txt
```

### ModuleNotFoundError: No module named 'pygame' / 'mediapipe' / 'rembg'

**Solution:** Install additional dependencies:
```bash
pip install -r requirements_additional.txt
```

Or install individually:
```bash
pip install pygame mediapipe rembg python-dotenv
```

### ModuleNotFoundError: No module named 'onnx'

**Solution:** Install ONNX:
```bash
pip install onnx==1.16.1
```

### FFmpeg not found

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### Conda environment issues

If using conda, ensure you're in the correct environment:
```bash
conda activate spectramotus
```

The script automatically detects conda Python via `CONDA_PREFIX`.

### Video generation fails

1. Check that `person_greenscreen.png` exists:
```bash
ls prepared-photos/your-photo/person_greenscreen.png
```

2. If missing, create it:
```bash
python convert_to_chroma_key.py prepared-photos/your-photo
```

3. Verify driving videos exist:
```bash
ls driving-videos/*.mp4
```

### Chroma key compositing issues

- Ensure videos were generated with `person_greenscreen.png` (green backgrounds)
- Check that `interactive_portrait.py` is detecting green correctly
- Regenerate videos if needed: `python batch_generate.py prepared-photos/your-photo`

## Development

### Running Tests

```bash
# Test photo preparation
python prepare_photo.py source-photos/test.jpg --skip-depth

# Test batch generation
python batch_generate.py prepared-photos/test

# Test interactive player
python interactive_portrait.py prepared-photos/test
```

### Adding New Gestures

Edit `interactive_portrait.py`:
1. Add gesture mapping in `GESTURE_MAP`
2. Add driving video in `batch_generate.py` → `DRIVING_VIDEOS`
3. Place driving video in `driving-videos/`

## License

[Add your license here]

## Credits

- Built on [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- Uses [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) (BRIA AI) for state-of-the-art background removal
- Uses [rembg](https://github.com/danielgatis/rembg) Python library
- Uses [MediaPipe](https://mediapipe.dev/) for gesture detection

## Background Removal Models

This project uses **RMBG-2.0** by default, which is BRIA AI's state-of-the-art background removal model. It provides superior quality compared to previous models and handles:
- General stock images
- E-commerce content
- Gaming images
- Advertising content
- Human portraits

The model is available for non-commercial use. For commercial use, see [BRIA AI's licensing](https://huggingface.co/briaai/RMBG-2.0).
