# Spectramotus studio: current architecture and release workflow

Updated 25 November 2026. This describes the implemented system; older outlines are design references, not current implementation facts.

## What exists

- Repository: https://github.com/tallkarol/spectramotus
- Studio: https://portrait-workshop-production.up.railway.app/
- Owner admin: https://portrait-workshop-production.up.railway.app/admin
- Python FastAPI serves HTML/JavaScript, authenticated APIs, and media. One process / one replica, with a small background preparation executor. No React build or separate frontend service.
- Railway Docker build: Python 3.11, ffmpeg, pinned requirements. Persistent volume mounted at `/data`. Demo footage is copied into `/app/demo` in the deployment bundle.
- `/data/projects/<id>` stores source photo, project/recipe snapshots, pack, and ZIP. `/data/motus` stores versioned recipe records and content-addressed driving-video uploads. Data currently uses JSON files with atomic replacements and process locks, not a SQL database or object bucket.
- One password protects the entire owner studio, including admin, project APIs, and media. This is not a public multi-user app. Anyone with that password has owner capabilities.
- Camera recognition and playback run in-browser with bundled MediaPipe assets. The same player is shipped in downloaded packs and the Pi starter kit. Camera frames are not uploaded.
- No paid generation, token debits, checkout, or customer accounts are implemented in the studio.

## Motus flow

Admin recipe → enabled catalog → selected movements at upload → immutable recipe snapshots in the project → existing matching footage or labelled still-photo placeholders → review → downloadable offline pack.

Recipe fields: name, short description, event/idle/manual trigger, prompt, Markdown text, optional H.264 driving video, enabled flag, proposed token price, revision. Markdown is stored as plain text. Driving videos are not automatically sent to any provider. Old hashed assets are retained for earlier project snapshots.

Only one selected recipe may occupy each gesture/idle slot; multiple manual movements are allowed. New clips do not imply new gesture classifiers: wave and thumbs-up are the current recognized gestures. The source photo must exactly match the demo photo before existing Plitvice footage is reused.

Prices are configuration previews. The studio displays planned totals and explicitly records zero tokens charged. A paid-looking badge is not an enforced paywall.

## How deployment works today

Deployments are manual Railway CLI uploads of an explicit staging directory. GitHub auto-deploy is not configured by this workflow. Never run `railway up` against the repository root: that risks uploading local photos, credentials, and work folders.

1. Run `node --test tests/gesture-gate.test.cjs tests/player-queue.test.cjs tests/diagnostics.test.cjs` and `.venv/bin/python -m pytest tests/test_studio.py -q`.
2. Commit only reviewed application source, tests, scripts, and documentation. Keep private photo/media/credentials in ignored folders.
3. Make a fresh staging folder:
   ```sh
   python3 scripts/stage-studio.py --photo work/studio-demo/photo.jpg --wave work/studio-demo/wave.mp4 --thumbs-up work/studio-demo/thumbs_up.mp4 --idle work/studio-demo/idle_blink.mp4 --output work/studio-deploy-RELEASE
   ```
4. Upload that folder using `railway up work/studio-deploy-RELEASE --path-as-root --project 5cc644a4-894c-488b-9034-2f3b0267d1db --service 21e53877-d9e3-4b3e-b6aa-5ca01026f8f9 --environment 3ce5ca55-7d61-4f71-98fd-32172f4fcec9 --detach`.
5. Verify deployment success, `/health`, login, existing portrait playback, and `/admin`. Record the deployment ID alongside the source commit.

Secrets live in Railway environment variables and ignored local `.env` files. Source photos, generated clips, driving assets, reports, packs, and releases must never be committed to the public repo. The explicit staging bundle includes only the required demo photo/clips and application runtime, not the rest of the local workspace.

## Pi delivery

`pi/launch.py` is a standard-library localhost server and pack validator. `scripts/build-pi-kit.py` bundles it, the current trusted player/model, a portrait ZIP, checksums, and setup instructions. It is a starter kit for Raspberry Pi OS Desktop, not a pre-flashed OS image. Camera permission is granted normally in a dedicated Chromium profile. Kiosk mode fills the screen and attempts camera startup; desktop startup can be enabled after the hardware test. See `pi/README.md`.

The setup follows Raspberry Pi's documented labwc/Chromium approach: https://www.raspberrypi.com/tutorials/how-to-use-a-raspberry-pi-in-kiosk-mode/

## Next steps, in order

1. Validate the kit on the purchased Pi/display/camera: offline operation, cold boot, recognition rate, response latency, long-running stability.
2. Use the Motus admin to settle recipe structure and driving-video workflow. Choose provider/local-worker adapters and explicit per-job spending limits before enabling generation.
3. Make releases reproducible from GitHub source. Move private demo assets out of the image build before enabling GitHub auto-deploy; use a private asset store or explicit provisioning step. Do not commit media to make CI easier.
4. Before opening to customers, add individual accounts and server-enforced owner/admin roles, durable jobs, a token ledger and verified payment webhooks. Reserve tokens atomically at job submission, settle once, and reconcile failed provider submissions. Add backups for the persistent volume; use a database/private object storage as needed for that multi-user stage.

Do not add payment infrastructure, a worker fleet, or a second frontend merely to support this owner-only prototype.
