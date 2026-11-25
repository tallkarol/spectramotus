# Camera preview

The Railway preview, downloaded pack, desktop localhost player and intended Pi Chromium kiosk share `preview.html` and `static/runtime/`. There is no separate Pi gesture implementation. Pi throughput has not yet been measured.

Enable camera in the player. Show one hand, wave an open palm side to side, or hold a thumbs-up for about 0.6 seconds. Lower the hand between gestures. A detected gesture fades idle to the rest photo in 220ms, then starts without the former extra 600ms cooldown. During a reaction, one next gesture is queued; further gestures are ignored until that slot clears. Buttons remain available. Camera stops when disabled, on page exit or when the tab becomes hidden; re-enable it when returning.

Frames are processed locally, never recorded or uploaded. The browser requests video only, not audio. A classic web worker runs MediaPipe Tasks Vision 0.10.21 with the bundled float16 gesture recognizer model v1, CPU delegate, one hand, at most fifteen samples/second and one frame in flight. Camera capture requests 640×480. A 10-second inference watchdog closes the camera on stalls. MediaPipe is pinned and copied into each pack; no CDN or model download is needed at playback time. Vendor provenance and SHA256 hashes are in `runtime/vendor/PROVENANCE.json`; Apache license is bundled.

`gesture-gate.js` accepts an open hand from at least three extended-finger landmarks or a palm classification >=0.60. It measures both palm travel and wrist angle, requiring an out-and-back movement over at least 180ms and three samples within 2.2s. Translation thresholds scale with hand size (0.018–0.06 image widths); wrist rotation uses 0.30 radians. It tolerates up to 350ms of tracking/classification dropout during waves. Thumbs-up requires confidence >=0.70 held for 350ms, or >=0.85 held for 250ms. A trigger latches until 500ms of neutral/no hand; cooldown is one second. These are initial thresholds, not a measured accuracy claim.

The player prioritizes the portrait and camera button. Manual reactions and idle/repeat controls are collapsed under “Try without a camera.” “Detection details” shows current hand tracking, classification, inference timing and a bounded 12-entry gesture event log. The log is transient DOM text, with no camera images or server reporting. Feedback distinguishes no hand from an unrecognized hand pose and persists after a successful gesture.

For an offline pack, unzip and serve the folder on the device:

```sh
python3 -m http.server 8767 --bind 127.0.0.1 --directory /path/to/unzipped-pack
```

Open `http://127.0.0.1:8767/preview.html` in a browser. Use that same localhost URL in Chromium kiosk on the Pi. HTTPS is required when accessing a remote host; a plain LAN IP is not sufficient for camera permissions. Opening the HTML directly still supports buttons, but camera mode directs the user to localhost. Existing packs can be safely imported with `python -m spectramotus.studio.preview portrait.zip`; imported executable runtime is replaced with the installed trusted runtime.

Validation: seven Python pack/backend tests; four temporal gesture-filter tests; real browser worker initialization and blank-frame inference without console errors. Physical hand accuracy, end-to-end webcam triggering, and Pi hardware performance still require hands-on verification. No generation API calls or credits were used.

API reference: https://developers.google.com/edge/api/mediapipe/js/tasks-vision.gesturerecognizer

## Live webcam verification follow-up
The first live camera attempt exposed a deployment-only failure: the worker inherited the server's CSP, which blocked WebAssembly compilation. The localhost smoke test served without that policy, so it did not catch the issue. The fix permits `wasm-unsafe-eval` only on the bundled `gesture-worker.js` response; document policy, same-origin asset restrictions, and JavaScript eval restrictions remain in place. A backend regression check verifies that scope. Worker errors now appear in the browser console without camera frames or recordings.


## Wave debugging and player revision
The earlier implementation required every moving frame to classify as Open_Palm with >=0.75 confidence, reset motion on any dropout, and ignored wrist rotation. Revised detector tests cover those failure modes. A real webcam wave produced “Wave → accepted” and “Waving back”; Karol confirmed it matched his intentional wave. The live log also exposed a thumbs-up being discarded during the wave, leading to the one-slot queue fix. Local observation was approximately 9.9 inference samples/second and 24ms inference time on this Mac; no Pi measurement. Ten JavaScript tests cover gesture filtering and player queue/idle transitions; seven backend/pack tests pass. No generation credits spent.

Karol confirmed the refreshed test sequence: “Both work, then idle” (wave, lower hand, thumbs-up, lower hand). This is user-reported end-to-end webcam validation; the earlier wave trigger was independently observed in the local player's event log. The later inspected local tab showed only its camera-ready event, so no claim is made of an independently captured log for the entire sequence. The same runtime is deployed on Railway; Pi remains untested.


## Faster gesture timing
Thirteen JavaScript tests pass, including 250ms high-confidence thumbs-up, 350ms moderate-confidence thumbs-up, 200ms clear wave, held-gesture suppression, jitter rejection, queued playback, and idle fade without extra cooldown. Queued reactions also start without the extra cooldown at clip completion. Detection details log the elapsed candidate-to-recognition duration for each gesture; this excludes time before the first qualifying hand observation and excludes video playback/queued wait. Webcam feel and accidental-trigger rate need user validation. No generation credits used.

## Ten-attempt diagnostic test
Detection details now includes a live, mirrored 21-landmark hand overlay and a guided test (five waves followed by five thumbs-ups). The prompt appears below the portrait. Each attempt has two seconds of preparation, then a six-second recognition window; successful attempts wait for the response to finish before advancing. Misses, wrong gestures, rejected requests, playback failures, cancelled attempts and missing return-to-idle are explicit outcomes. The final attempt waits for an idle video frame. Enable the camera and press Start 10-gesture test when ready. Stop test cancels the run; hiding the page stops the camera and cancels the test.

Download timing report produces a local JSON file with bounded event history (1500 entries) and per-attempt classification/tracking statistics. No images, video, or landmark coordinates are saved. Candidate-to-recognition is measured from the detector's qualifying motion/pose window; prompt-to-recognition also includes the person's reaction time. Request-to-first-frame uses requestVideoFrameCallback where supported, otherwise it is labeled a play-promise fallback. Queue events identify what blocked playback. Clip motion onset is an estimate from the existing files, logged when playback passes that media timestamp; it is not a live visual-motion measurement.

Existing clip audit: both contain roughly 0.5s of opening stillness. Low-resolution person-region mean absolute difference crosses 3/255 at 0.667s for wave and 0.875s for thumbs-up. Contact sheets show clear arm movement by approximately 0.75–1s for wave, 1–1.5s for thumbs-up. Files were not trimmed for this baseline run. Audit and contact sheets: work/responsiveness-audit/. Fifteen JavaScript tests and seven backend/pack tests pass. No generation credits spent.


## Isolated diagnostic retest (schema 2)
The test now suppresses gesture dispatch during preparation, countdown, playback, and inter-trial periods, and blocks manual reactions during the run. Pending requests are cleared when the test begins; any already-playing clip finishes. Every attempt waits for the player to be fully at rest and at least 700ms of neutral/no-hand observations. The gesture gate is reset outside the listening window and again on entry, so confidence/hold/release state cannot carry across attempts. Idle is disabled during attempts and restored for the final return-to-idle check. This isolates recognition and response timing from idle interruptions and duplicate queues.

Schema 2 adds a bounded per-sample trace for each listening window: relative timestamp, label, confidence, hand size/edge warning, inference time, frame age, hold duration/required duration, latch/cooldown, decision reason, and hold-reset reason. Aggregate samples now cover only listening, not the ensuing clip. Raw landmark coordinates and camera frames are never exported. The visible results show aggregates; Download timing report includes the full trace.

Hand acquisition v2 requests 960×720 (actual dimensions logged at camera start), lowers detection/presence thresholds from .6/.6 to .45/.5, and uses .5 tracking confidence to permit more lenient tracking. Gesture classification thresholds remain unchanged for this retest. The UI explicitly warns about a hand near the image edge or too small, and asks for full-hand visibility and lighting when no hand is found. These are acquisition adjustments to validate, not an accuracy guarantee. Eighteen JavaScript tests cover isolation, filter diagnostics, existing gesture safety and playback; the seven backend/pack tests also run before deployment. No generation credits used.

### Retest revision 3

Thumbs-up now accepts sustained Thumb_Up scores >=0.60 (350ms; 250ms at >=0.85), with a maximum 120ms grace for brief missing/low-confidence samples. A different pose or longer dropout resets the hold. Normal release and duplicate suppression remain in place. This needs human false-positive testing; replaying five previous positive attempts is not an accuracy benchmark.

Schema 3 records wave window duration/sample count plus translation and wrist-rotation range, reversal, threshold and pass/fail. These are derived movement statistics, not images or landmark coordinates. Isolated trial boundaries are unchanged.

Existing demo reactions were locally shortened with the original rest frame and a 125ms entrance blend. Wave removed 13 frames (0.542s); thumbs-up removed 18 frames (0.750s), at 24fps. Return sequences were retained. Original files are in ignored work/responsiveness-audit/original-clips. work/studio-demo/clip-timing.json supplies motion estimates to online and offline manifests. Changed media is migrated on server startup and requires review before export. No generation was used.

### Continuous interaction

Outside the isolated diagnostic test, the camera recognizes throughout reaction playback and queues one next reaction. Switching gesture types no longer requires a neutral hand pose; each new type must still pass its normal motion/hold rule. Repeating the same gesture requires a 250ms neutral release. A 250ms minimum separation prevents immediate duplicate emissions. Reaction completion has no extra rest cooldown; queued reactions start at the next event-loop turn. Current reactions still finish to preserve rest-frame continuity. The isolated test intentionally retains its between-attempt reset/countdown and is not representative of continuous interaction.

### Pi 5 CSI test-drive preparation

The local Pi launcher now accepts `--camera csi`: a Picamera2 JPEG adapter feeds the same browser worker/gate through loopback-only snapshot requests. Camera start/stop requires a same-origin custom header; camera frames remain in memory, and capture stops on disconnect/inactivity. The kit includes OS-package setup and hardware checks. Balanced (960×720) and light (640×480) acquisition are explicit comparison settings, not different gesture thresholds. Recognition now schedules its next sample from completion, rather than waiting for a separate polling tick; one frame remains in flight, with a roughly 15 Hz cap. Browser diagnostics add periodic throughput and video dropped-frame counters. Local synthetic-frame smoke testing exercises the CSI HTTP/browser/MediaPipe path; actual Picamera2 sensor acquisition and Pi throughput still require Dave's hardware test. See `pi/TEST-DRIVE.md`.
