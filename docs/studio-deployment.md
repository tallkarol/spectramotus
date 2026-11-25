# Spectramotus studio deployment

Updated 25 November 2026.

## App and storage

- Public repository: https://github.com/tallkarol/spectramotus
- Studio: https://portrait-workshop-production.up.railway.app
- FastAPI serves the studio, APIs, authenticated media and the shared browser/Pi player. One replica runs the background placeholder preparation executor.
- PostgreSQL stores accounts, aliases, hashed sign-in links, revocable sessions, spectrum ownership, Motus recipes and immutable revisions. SQLAlchemy schema version 001 is additive; version 002 imports legacy JSON metadata once.
- Railway's `/data` volume holds source images, clips, packs, driving videos and `/data/demo`. Media remains private and is served through account permissions.
- The import preserves original metadata and also copies it to `/data/backups/before-postgres`. New changes live in Postgres; reverting to old JSON code would lose visibility of those changes. For an email outage, keep the new database-backed build and temporarily set `STUDIO_AUTH_MODE=password` rather than reverting the application.
- Generation, token debits and payments remain disabled. Existing footage and clearly labelled placeholders are used. No camera frames leave the browser.

## Accounts

`STUDIO_OWNER_EMAILS` lists the primary owner email followed by aliases for the same account. `STUDIO_INITIAL_ADMINS` prepares admin access on first startup without sending email. Do not leave removed admins in this bootstrap list if accounts are later deleted.

Owner and admin can access all spectra and manage Motus. Only the owner can grant or revoke admin access. Both can prepare beta invitations; beta users see only their own spectra and published Motus, without private prompts or driving assets. Disabling an account immediately revokes sessions and pending links.

Sign-in links expire after 15 minutes and can be consumed once, atomically. The token travels in a URL fragment, is removed from browser history, and requires an explicit confirmation POST so link scanners cannot log in. Sessions expire after seven days. Invitation access expires after seven days unless activated; the People page can renew it. Request endpoints apply per-email and per-IP rate limits and do not reveal whether an address exists.

Resend uses the existing CRM sender configuration. Required variables: `RESEND_API_KEY`, `RESEND_FROM_EMAIL`, `STUDIO_APP_URL` (canonical HTTPS origin), `STUDIO_AUTH_MODE=email`, `STUDIO_SECURE_COOKIE=1`, `STUDIO_OWNER_EMAILS`, and `DATABASE_URL=${{Postgres.DATABASE_URL}}`. Keep credentials in Railway variables only. Preparing access never sends an email; the explicit Send sign-in invitation action does.

## Motus publishing

Admin → draft/published/disabled recipe → published catalog → selected immutable recipe snapshots → review → offline pack. Saving a draft preserves the previous published revision. Publishing exposes the new revision for future spectra. Disabling removes it from the catalog. Existing spectrum snapshots do not change.

Prompt, plain-text Markdown, optional H.264 driving video, trigger and proposed token price are configuration only. One recipe per gesture/idle slot; multiple manual actions are allowed. Wave and thumbs-up are the current gesture classifiers. Private demo footage is available only to the team, for the matching source photo.

## Builds and releases

`.github/workflows/studio.yml` checks the player, studio, accounts and Pi launcher. Account tests use PostgreSQL 18 with a unique schema per test. `railway.json` selects `Dockerfile.studio` and the database-aware `/health` check. `.dockerignore` allowlists only runtime source and requirements; private assets and secrets must never enter Git.

The current release branch is `codex/motus-admin-pi-kit`. Push reviewed changes and wait for Studio checks before a production release. Railway's GitHub source connection and any CI deployment gate must be verified in the service settings; the presence of this workflow alone does not enforce a Railway gate.

A one-time bootstrap CLI bundle copies the existing private demo assets into `/data/demo`:

```sh
python3 scripts/stage-studio.py --bootstrap-demo --photo work/studio-demo/photo.jpg --wave work/studio-demo/wave.mp4 --thumbs-up work/studio-demo/thumbs_up.mp4 --idle work/studio-demo/idle_blink.mp4 --output work/studio-deploy-RELEASE
```

Upload only that explicit staging directory with `railway up --path-as-root`. Subsequent GitHub Docker builds use `/data/demo` and include no private media. Verify `/health`, existing spectra/media, the Motus catalog and People after deployment. Keep the app at one replica while media and preparation jobs use a single volume. Restarted preparation jobs become retryable failures.

## Pi and next work

The existing `pi/launch.py` and `scripts/build-pi-kit.py` ship the same player, clips and MediaPipe assets for offline operation. Account changes do not require a new player. Hardware verification remains outstanding; see `pi/README.md`.

Before wider beta: configure and test database/volume backups, settle retention and upload quotas, and measure preparation capacity. Generation needs a durable job queue and explicit spending limits; payments need a transaction ledger. Neither is required for the current invite-only placeholder studio.

Product language: singular **spectrum**, plural **spectra**, creation **Craft your Spectrum**.
