# Enhancement Roadmap - Interactive Portrait Frame

This document outlines AI models and techniques to enhance the interactive portrait frame beyond the basic POC.

## Table of Contents
- [Must-Have Models (High Impact)](#must-have-models-high-impact)
- [Game-Changing Models](#game-changing-models)
- [Creative Enhancement Models](#creative-enhancement-models)
- [Atmospheric Effects](#atmospheric-effects)
- [Practical Pipeline Ideas](#practical-pipeline-ideas)
- [Recommended Implementation Order](#recommended-implementation-order)
- [Advanced Features (Post-POC)](#advanced-features-post-poc)

---

## Must-Have Models (High Impact)

### 1. RMBG-2.0 - Background Removal ⭐⭐⭐

**Purpose:** Separate person from background with pixel-perfect accuracy

**Installation:**
```bash
pip install rembg
```

**Usage:**
```python
from rembg import remove
from PIL import Image

input_image = Image.open('portrait.jpg')
output_image = remove(input_image)
output_image.save('portrait_nobg.png')
```

**Benefits:**
- Works locally on M4 Pro
- Faster and more accurate than SAM2 for portraits
- Creates clean alpha channel mattes
- Enables independent person/background animation

**Use Cases:**
- Animate person and background separately with different motion
- Add parallax depth effects
- Swap backgrounds dynamically
- Apply different effects to foreground vs background

---

### 2. Depth-Anything-V2 - Scene Depth Estimation ⭐⭐⭐

**Purpose:** Generate depth maps for 3D parallax effects

**Installation:**
```bash
pip install transformers torch
```

**Usage:**
```python
from transformers import pipeline

depth_estimator = pipeline(
    "depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Large"
)

depth_map = depth_estimator(image)
```

**Benefits:**
- Creates convincing 2.5D parallax motion
- Runs efficiently on M4 Pro
- Works with any image, no training needed
- Enables "living window" effect

**Use Cases:**
- Background layers shift based on depth when user moves
- Create the "window into another world" illusion
- Trees/objects at different depths move at different speeds
- Add depth-aware blur effects

---

### 3. AnimateDiff + ControlNet - Background Animation ⭐⭐⭐

**Purpose:** Animate background elements while keeping person static

**Installation:**
```bash
pip install diffusers transformers accelerate
```

**Usage:**
```python
from diffusers import AnimateDiffPipeline, MotionAdapter
import torch

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter
).to("mps")

# Animate just the background
video = pipe(
    prompt="gentle wind, leaves rustling, clouds drifting",
    negative_prompt="person moving, face changing",
    num_frames=16
).frames
```

**Benefits:**
- Animate environmental elements independently
- Cloud models can be expensive, local is free
- Highly controllable with ControlNet

**Use Cases:**
- Clouds drift behind person
- Water ripples, leaves rustle
- Fire flickers, candles glow
- Snow falls, rain streaks
- Person remains perfectly still (LivePortrait), world moves around them

---

## Game-Changing Models

### 4. GroundingDINO - Zero-Shot Object Detection ⭐⭐

**Purpose:** Detect objects in photos using natural language prompts

**Installation:**
```bash
pip install transformers
```

**Usage:**
```python
from transformers import pipeline

detector = pipeline(
    "zero-shot-object-detection", 
    model="IDEA-Research/grounding-dino-base"
)

# Detect with natural language!
results = detector(
    image, 
    candidate_labels=["coffee cup", "newspaper", "hat", "book", "flower"]
)

for detection in results:
    print(f"Found {detection['label']} at {detection['box']}")
```

**Benefits:**
- No training or manual labeling required
- Understands context from text descriptions
- Can detect anything you can describe
- Runs locally on M4 Pro

**Use Cases:**
- Automatically detect props: "holding coffee", "wearing glasses", "sitting on chair"
- Generate contextual animations based on detected objects
  - Coffee cup → steam rising
  - Newspaper → pages turning
  - Hat → brim tipping
  - Flowers → petals falling
- Create "smart" animations that understand scene context
- Build gesture library based on detected objects

---

### 5. CodeFormer - Face Enhancement/Restoration ⭐⭐

**Purpose:** Enhance low-quality or damaged photos

**Installation:**
```bash
pip install codeformer
# Or clone: git clone https://github.com/sczhou/CodeFormer
```

**Usage:**
```python
from codeformer import CodeFormer

model = CodeFormer(model_path='weights/CodeFormer/codeformer.pth')
enhanced_face = model.restore(degraded_image, fidelity_weight=0.7)
```

**Benefits:**
- Dramatically improves quality of old/damaged photos
- Preserves identity while enhancing details
- Works on heavily compressed images
- Local processing

**Use Cases:**
- Old family photos → crystal clear animations
- Fix grainy, blurry, or pixelated source photos
- Restore damaged/scratched historical photographs
- Enhance low-resolution images before animation
- Critical for vintage photo use cases

---

### 6. Real-ESRGAN - 4x Image Upscaling ⭐⭐

**Purpose:** AI upscaling for 4K displays

**Installation:**
```bash
pip install realesrgan
```

**Usage:**
```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
upsampler = RealESRGANer(
    scale=4,
    model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth',
    model=model
)

output, _ = upsampler.enhance(image, outscale=4)
```

**Benefits:**
- 4x upscaling with minimal quality loss
- Maintains detail during upscale
- Works on M4 Pro (slower but functional)
- Better than traditional interpolation

**Use Cases:**
- Small/old photos → 4K frame quality
- Maintain sharpness during animation
- Enable high-DPI display support
- Prepare images for large format frames

---

## Creative Enhancement Models

### 7. IP-Adapter - Style Transfer ⭐

**Purpose:** Apply artistic styles while preserving identity

**Installation:**
```bash
pip install diffusers
```

**Usage:**
```python
from diffusers import StableDiffusionXLPipeline, IPAdapterPlusXL

pipe = StableDiffusionXLPipeline.from_pretrained(...)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")

# Apply style while keeping identity
image = pipe(
    prompt="oil painting style",
    ip_adapter_image=portrait,
    strength=0.7
).images[0]
```

**Use Cases:**
- Oil painting mode for classical portraits
- Watercolor mode for soft, dreamy effect
- Old photograph sepia tone
- Vintage film look
- Comic book style
- Switch styles based on time of day or user preference
- Create themed collections (Victorian, Modern, Futuristic)

---

### 8. FILM (Frame Interpolation for Large Motion) ⭐⭐

**Purpose:** Create smooth slow-motion from existing videos

**Installation:**
```bash
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
pip install -r requirements.txt
```

**Usage:**
```bash
python -m eval.interpolator_cli \
    --pattern "input_frames" \
    --model_path pretrained_models/film_net/Style/saved_model \
    --times_to_interpolate 1  # 30fps -> 60fps
```

**Benefits:**
- Buttery smooth animations
- Reduces jerkiness in LivePortrait outputs
- Creates premium viewing experience
- Local processing

**Use Cases:**
- Smooth LivePortrait animations from 30fps → 60fps
- Perfect transitions between gesture clips
- Slow-motion effects for dramatic moments
- Reduce artifacts in fast movements

---

### 9. Wav2Lip - Lip Sync Animation ⭐

**Purpose:** Make portraits speak with realistic lip sync

**Installation:**
```bash
git clone https://github.com/Rudrabha/Wav2Lip
cd Wav2Lip
pip install -r requirements.txt
```

**Usage:**
```bash
python inference.py \
    --checkpoint_path checkpoints/wav2lip.pth \
    --face path/to/portrait.mp4 \
    --audio path/to/speech.wav
```

**Benefits:**
- Perfect lip sync to any audio
- Works with LivePortrait animations
- Can use AI voice cloning for deceased relatives
- Local processing available

**Use Cases:**
- Portrait greets you: "Good morning!"
- Responds with voice: "How was your day?"
- Tells stories from the past (using recorded audio)
- Reads quotes or letters
- **Combine with ElevenLabs voice cloning for ultimate realism**
- Speaking portraits of historical figures
- Audiobook narration with animated portrait

---

### 10. InstantID - Consistent Character Generation ⭐⭐

**Purpose:** Generate multiple poses/expressions while maintaining identity

**Installation:**
```bash
pip install diffusers insightface onnxruntime
```

**Usage:**
```python
from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis

# Extract face embedding
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(reference_image)
face_emb = faces[0].normed_embedding

# Generate new pose with same identity
pipe = StableDiffusionXLPipeline.from_pretrained(...)
image = pipe(
    prompt="person waving, full body, happy",
    ip_adapter_image_embeds=face_emb
).images[0]
```

**Benefits:**
- Create gestures you don't have photos for
- Maintains identity consistency
- Generate missing poses/expressions
- Local processing possible

**Use Cases:**
- Generate full-body wave when only have headshot
- Create gestures you never photographed
- "What would grandma look like waving?"
- Fill in missing animations
- Generate contextual poses (holding objects, different actions)

---

## Atmospheric Effects

### 11. Stable Video Diffusion - Full Scene Animation

**Purpose:** Animate entire scenes, not just faces

**Installation:**
```bash
pip install diffusers transformers accelerate
```

**Usage:**
```python
from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt"
).to("mps")

frames = pipe(
    image=scene_image,
    num_frames=25,
    motion_bucket_id=127
).frames
```

**Use Cases:**
- Rain on window glass
- Fireplace crackling
- Candles flickering
- Curtains swaying
- Steam rising from tea
- Leaves falling
- Water flowing
- Especially good for portraits in specific atmospheric settings

---

### 12. GFPGAN - Old Photo Restoration ⭐⭐⭐

**Purpose:** Restore severely damaged or degraded photos

**Installation:**
```bash
pip install gfpgan
```

**Usage:**
```python
from gfpgan import GFPGANer

restorer = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2
)

_, _, restored_img = restorer.enhance(
    degraded_image,
    has_aligned=False,
    only_center_face=False,
    paste_back=True
)
```

**Benefits:**
- Restores heavily damaged photos
- Better for old photos than CodeFormer
- Removes scratches, stains, damage
- Can colorize black & white photos

**Use Cases:**
- Restore great-grandparents from 1920s → animate
- Fix scratched/faded/damaged photos
- Recover nearly-destroyed family heirlooms
- **Killer feature for historical family photos**
- Museum/archive digitization projects
- Colorize old black & white portraits

---

## Practical Pipeline Ideas

### Pipeline 1: Premium Quality Portrait (Old Photo)
```
Old/Damaged Photo 
  → GFPGAN (restore damage)
  → Real-ESRGAN (upscale to 4K)
  → CodeFormer (enhance face details)
  → RMBG (isolate person from background)
  → LivePortrait (animate face)
  → Depth-Anything (create background depth map)
  → AnimateDiff (animate background elements)
  → FILM (smooth to 60fps)
  → Final composite
```

**Timeline:** 30-45 minutes per photo
**Quality:** Cinema-grade
**Best for:** Premium family heirlooms, special occasions

---

### Pipeline 2: Contextual Smart Animation
```
Modern Photo
  → GroundingDINO (detect objects: "coffee cup", "book", "hat")
  → Generate object-specific micro-animations
  → RMBG (separate person from background)
  → LivePortrait (animate face with appropriate expression)
  → Depth-Anything (background depth)
  → AnimateDiff (contextual background: steam from coffee, pages flutter)
  → Composite all layers
```

**Timeline:** 15-20 minutes per photo
**Quality:** High, context-aware
**Best for:** Daily photos, casual portraits with props

---

### Pipeline 3: Speaking Portrait (Advanced)
```
Photo + Audio Recording
  → GFPGAN (if old photo)
  → LivePortrait (base facial animation)
  → ElevenLabs API (voice clone from old recordings)
  → Wav2Lip (perfect lip sync to speech)
  → RMBG + Depth (layered background)
  → Final composite
```

**Timeline:** 20-30 minutes per photo
**Quality:** Mind-blowing
**Best for:** Storytelling, memorial tributes, "grandma tells bedtime stories"

---

### Pipeline 4: Fast Interactive (POC Current)
```
Photo
  → LivePortrait (5-10 gesture clips)
  → MediaPipe (real-time gesture detection)
  → State machine switches between clips
```

**Timeline:** 10-15 minutes per photo
**Quality:** Good, interactive
**Best for:** POC, demos, fast turnaround

---

## Recommended Implementation Order

### Phase 1: Current POC ✅
- [x] LivePortrait for face animation
- [x] MediaPipe for gesture detection
- [x] Basic clip switching

### Phase 2: Layered Animation (1-2 weeks)
1. **RMBG-2.0** - Background removal
   - Easy install, immediate impact
   - Enables all future enhancements
   - 30 min to integrate

2. **Depth-Anything-V2** - Depth maps
   - Creates parallax effect
   - Makes backgrounds "alive"
   - 1 hour to integrate

3. **Basic compositing** - Combine person + animated background
   - Use depth for parallax shift
   - Add subtle background motion
   - 2-3 hours to build compositor

### Phase 3: Intelligence (1-2 weeks)
4. **GroundingDINO** - Object detection
   - Detect props automatically
   - Enable context-aware animations
   - 3-4 hours to integrate

5. **Context-based animation library**
   - Build animations for common objects
   - Coffee → steam, Book → pages turn, etc.
   - 1 week to build library

### Phase 4: Quality Enhancement (1 week)
6. **GFPGAN/CodeFormer** - Photo restoration
   - Critical for old photos
   - Pre-processing step
   - 2 hours to integrate

7. **Real-ESRGAN** - Upscaling
   - Enable 4K output
   - 1 hour to integrate

8. **FILM** - Frame interpolation
   - 60fps smoothness
   - 3-4 hours to integrate

### Phase 5: Advanced Features (2-3 weeks)
9. **Wav2Lip** - Speaking portraits
   - Voice integration
   - Lip sync
   - 1 week to integrate properly

10. **InstantID** - Generate missing poses
    - Expand gesture library
    - 3-4 days to integrate

### Phase 6: Polish (ongoing)
11. **AnimateDiff** - Scene-wide animation
12. **IP-Adapter** - Style modes
13. **Custom model fine-tuning**

---

## Advanced Features (Post-POC)

### Multi-Portrait Interactions
- Multiple frames "see" each other via network
- Family photos interact across frames
- Grandparents wave at grandkids in different frame
- Portraits have "conversations"

**Tech Stack:**
- WebSocket server for frame communication
- Synchronized gesture triggering
- Shared state machine

---

### Environmental Awareness
- **Time of day** → Lighting changes on portrait
- **Weather outside** → Portrait reacts (rainy day = different mood)
- **Music playing** → Portrait "listens" and reacts
- **Room occupancy** → Portrait "wakes up" when people enter

**Tech Stack:**
- Home Assistant integration
- Spotify API for music
- Weather API
- Motion sensors

---

### Voice Interaction
- "Show me dad's smile" → Switches to smile clip
- "Make grandma wave" → Triggers wave animation
- "Tell me a story" → Speaking portrait mode
- Whisper AI for speech recognition

**Tech Stack:**
- Whisper (OpenAI) for speech-to-text
- Local LLM for response generation
- ElevenLabs for voice synthesis

---

### Shared Memory Across Frames
- Portrait remembers conversations
- "Yesterday you told me about..."
- Builds personality over time
- Can reference past interactions

**Tech Stack:**
- Vector database (ChromaDB)
- Local LLM (Llama 3)
- Memory management system

---

## Performance Optimization Notes

### M4 Pro (24GB) Capabilities
- **LivePortrait:** ~250ms per frame (4 FPS) - acceptable for pre-rendering
- **RMBG:** ~2-3 seconds per image - fast
- **Depth-Anything-V2:** ~5 seconds per image - acceptable
- **GroundingDINO:** ~3-4 seconds per image - acceptable
- **GFPGAN:** ~10-15 seconds per image - acceptable
- **Real-ESRGAN:** ~30-60 seconds per image - slow but one-time
- **AnimateDiff:** ~2-5 minutes for 16 frames - slow, consider cloud
- **FILM:** ~1-2 minutes for 60fps conversion - acceptable

### Optimization Strategies
1. **Pre-render everything** - No real-time inference needed
2. **Batch processing** - Process multiple photos overnight
3. **Cloud hybrid** - Use Replicate API for heavy models (AnimateDiff, SVD)
4. **Cache aggressively** - Store all intermediate results
5. **Resolution management** - Process at 720p, upscale final output

---

## Cost Analysis

### Local-Only (Current)
- **Hardware:** $0 (existing M4 Pro)
- **Software:** $0 (all open source)
- **Per photo:** $0 (just electricity + time)

### Hybrid (Cloud Enhancement)
- **Per photo (5-10 clips):**
  - LivePortrait (local): $0
  - AnimateDiff (Replicate): ~$0.10-0.30
  - SVD (Replicate): ~$0.20-0.40
  - **Total: $0.30-0.70 per photo**

### Premium (All Features)
- **Per photo:**
  - Base animations (local): $0
  - Cloud enhancements: $0.30-0.70
  - Voice cloning (ElevenLabs): ~$0.10-0.30
  - **Total: $0.40-1.00 per photo**

---

## Next Steps

1. **Quick Win (This Week):** Add RMBG + Depth-Anything for layered parallax
2. **Medium Term (Next 2 weeks):** Integrate GroundingDINO for smart animations
3. **Long Term (Next month):** Add speaking portraits with Wav2Lip
4. **Future:** Multi-frame interactions, environmental awareness

---

## Resources & Links

### Model Repositories
- LivePortrait: https://github.com/KwaiVGI/LivePortrait
- RMBG: https://github.com/danielgatis/rembg
- Depth-Anything-V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- CodeFormer: https://github.com/sczhou/CodeFormer
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip
- FILM: https://github.com/google-research/frame-interpolation
- GFPGAN: https://github.com/TencentARC/GFPGAN

### Community & Tutorials
- Hugging Face: https://huggingface.co/models
- Replicate: https://replicate.com (for cloud inference)
- ComfyUI: https://github.com/comfyanonymous/ComfyUI (workflow builder)

---

*Last Updated: January 2026*
*Version: 1.0*