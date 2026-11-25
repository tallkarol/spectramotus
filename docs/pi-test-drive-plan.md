# Spectramotus · current hardware milestone

## Success criterion

Dave can install a self-contained kit on Raspberry Pi 5 4 GB with Raspberry Pi OS Desktop and a ribbon-connected camera, play the existing Karol wave/thumbs-up/idle videos, operate offline and export diagnostics. Improve real observed failures before choosing the next Motus and generating clips for three new images.

## Phases

1. **Admin reliability:** searchable/filterable Motus, distinguish draft revision from live revision, explicit save/publish/disable language, unsaved-change feedback. Keep generation and billing disabled.
2. **Hardware kit:** trusted shared player and bundled MediaPipe, current locally trimmed Karol clips, Picamera2 local adapter, installation script, hardware report, test checklist, checksums and release metadata. No public source-photo upload or generation.
3. **On-device baseline:** camera/display setup, manual playback, five waves and five thumbs-ups, continuous gesture queue, offline/reboot test, 30-minute soak. See `pi/TEST-DRIVE.md`.
4. **Measured optimization:** compare balanced/light capture in identical conditions, separate inference delays from queue wait/video startup/clip motion, check dropped frames and thermals. Do not promise Pi performance before measurement.
5. **Motus and three-photo trial:** retain successful reaction and idle types, define rest-frame/camera/identity/hand criteria, select three photos and confirm a separate generation budget. Generate one representative clip first and review before batching.

## Decisions

- Same player and gesture gate on desktop, Railway and Pi; only camera acquisition differs for CSI.
- Pi camera frames stay in RAM and travel only over loopback to the same local player. No camera frames in JSON diagnostics.
- The current production app remains useful independently of hardware; no new generation or billing is enabled.
- A kit is not an OS image. Keep the Pi 3 card and previous kit as fallbacks.
- Hardware acceptance requires Dave/Karol to run the actual Pi tests; desktop tests cannot certify it.
