# Dave + Karol · first hardware session

**Goal:** the current Karol spectrum reacts reliably to waves and thumbs-ups, returns to idle, and works offline on the Pi 5. Do not generate new videos during this test.

## 1. Establish a baseline (5 minutes)

- Run setup and save `hardware-report.json`.
- With camera off, manually play wave, thumbs-up and idle. Look for black flashes, stutter, abrupt joins or incorrect screen fit.
- Start the camera. Confirm the preview follows your hand, overlay aligns, and the entire hand is visible. Note distance, light and screen orientation.

## 2. Isolated recognition (5 minutes)

Open Detection details; run **Start 10-gesture test**. Follow each prompt: five waves, five thumbs-ups. This test deliberately waits for rest/neutral between attempts to isolate recognition; normal playback does not have that test cooldown.

Download the timing report and rename it `pi-balanced-isolated.json`. Record which gestures you intentionally made and any obvious misses. Candidate timing excludes time before tracking acquires the hand; prompt timing includes your reaction time.

## 3. Continuous interaction (5 minutes)

Outside the guided test, wave, then give a thumbs-up while the wave is playing. Reverse the sequence. Repeat five times. Confirm the next reaction is recognized and queued; it starts when the current clip returns to rest. Holding one gesture should not repeat it endlessly. Only one next reaction is queued by design.

Leave hands relaxed for two minutes; count accidental triggers. Download another report as `pi-balanced-continuous.json`.

## 4. Compare only when needed

If it stutters or tracking is slow, run `--camera-profile light` and repeat the same distance/light/test. Save `pi-light-isolated.json` and `pi-light-continuous.json`. Do not change lighting, distance and profile all at once. Re-run doctor after load to capture heat/throttling. Prefer repeatable behavior over one unusually good attempt.

## 5. Acceptance and soak

Provisional targets for this first hardware session:

- All three existing clips play; no black flashes or frozen playback.
- At least 4/5 intentional attempts of **each gesture** match, then a clean repeat run. Log all misses; this small sample does not prove broad accuracy.
- Switching gesture types during a reaction queues the next response reliably.
- Return to idle is observed after reactions.
- No false triggers during the two-minute relaxed-hands check.
- Run 30 minutes; note heat, dropped frames, disconnects or slowdown. Restart twice and test with Wi-Fi disconnected.

If any target fails, capture the report and the exact symptom. Fix one cause, rerun the affected test, then repeat the complete acceptance sequence. Hardware success remains pending until these steps are performed on Dave's Pi.

## Bring back

Hardware report, gesture reports, camera/screen model names, chosen profile, lighting/distance, false-trigger count, and notes about visible playback joins. A phone recording of the screen/hand is optional if you want to show a subjective delay; it is not collected by the app.
