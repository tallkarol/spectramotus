# Vendored: LivePortrait

- Upstream: https://github.com/KwaiVGI/LivePortrait (KwaiVGI / Kling team), MIT licence.
- Snapshot: v1.1-era tree, vendored in the initial Spectramotus commit (2026-01-19). No local modifications
  (`git diff 98a2972 HEAD -- src/` is empty). TODO: record the exact upstream commit hash here when known.
- Used by Spectramotus for the local face set only (`spectramotus/face/liveportrait.py` drives
  `src/live_portrait_pipeline.py`'s image-source path in-process). `inference.py` at the repo root is the
  unchanged upstream CLI, kept for parity debugging.
- Not used: `src/gradio_pipeline.py`, the animal pipeline (`src/live_portrait_pipeline_animal.py`,
  `src/utils/dependencies/XPose/`), whose weights were removed from `pretrained_weights/`.
- Licensing note: the vendored insightface *code* is MIT, but the `buffalo_l` detector weights the cropper
  downloads (`det_10g.onnx`, `2d106det.onnx`) are research/non-commercial. They must be replaced by a
  MediaPipe FaceLandmarker-based cropper before commercial launch (see THIRD_PARTY_NOTICES.md).
