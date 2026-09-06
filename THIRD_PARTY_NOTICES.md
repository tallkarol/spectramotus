# Third-party notices and licensing status

Spectramotus is a proprietary product of Sondry. It builds on the components below. "Commercial OK"
reflects our reading as of 2026-09-06; re-verify before launch.

| Component | Use | Licence | Commercial OK? | Action |
|---|---|---|---|---|
| LivePortrait code (`src/`) + Kwai weights + `landmark.onnx` | Face set (local) | MIT | yes | keep; see `src/VENDORED.md` |
| insightface `buffalo_l` `det_10g.onnx` / `2d106det.onnx` (LivePortrait cropper) | Face detection/landmarks | research, non-commercial | **no** | replace cropper with MediaPipe FaceLandmarker before launch |
| insightface vendored code | cropper | MIT | yes | unused after the swap |
| BiRefNet (`ZhengPeng7/BiRefNet`, portrait variant) | image + video matting | MIT | yes | default |
| MediaPipe Tasks (FaceDetector, FaceLandmarker, PoseLandmarker, GestureRecognizer) | intake, registration, sensing | Apache-2.0 | yes | default |
| RMBG-2.0 (BRIA) | - | CC-BY-NC-4.0 | no | removed from the pipeline |
| RobustVideoMatting | optional preview matting | GPL-3.0 | internal tool only | never redistribute |
| MatAnyone / MatAnyone 2 | - | S-Lab (non-commercial) | no | excluded |
| Depth-Anything-V2 Small | optional depth | Apache-2.0 (Base/Large are CC-BY-NC) | Small only | off in v4 |
| LaMa (`simple-lama-inpainting`) | clean-plate fallback | Apache-2.0 | yes | optional |
| RIFE / rife-ncnn-vulkan | optional interpolation | MIT | yes | optional |
| PyTorch, transformers, timm, kornia, einops, onnxruntime, OpenCV | runtime | BSD/Apache/MIT | yes | - |
| pygame, PyAV, sounddevice, openWakeWord | player + sensing | LGPL/BSD/MIT/Apache | yes (dynamic linking) | - |
| ffmpeg (Homebrew build with libx264 = GPL) | offline tooling | GPL-3.0 | internal tooling yes | do not ship the Homebrew binary; Pi uses distro ffmpeg |
| Wan 2.2 Animate via fal.ai | body motion | Apache-2.0 model; fal ToS | yes | check fal data-retention terms for customer photos |
| Kling 3.0 Motion Control (fal), Runway Act-Two, Nano Banana Pro (Google), Claude (Anthropic) | body motion / plate / QA | commercial API ToS | yes | customer privacy + retention notice; Gemini images carry SynthID |
| ClipDrop | legacy fallback | commercial API | yes | legacy only |
