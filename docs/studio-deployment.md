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

The current release branch is `codex/motus-admin-pi-kit`. The current application release is `f980909`, deployed from GitHub as `36291eeb-9b8b-401a-a9c8-57b36cda6331`. All 46 checks passed. Railway's GitHub App now has repository access. Automatic deployments follow `codex/motus-admin-pi-kit` and wait for successful GitHub checks (`checkSuites=true`, trigger `baa75d9d-7a68-4d28-8b6f-f0f82f806a02`). Push reviewed changes to that branch; other branches do not deploy production. `main` is not yet the production branch.

A one-time bootstrap CLI bundle copies the existing private demo assets into `/data/demo`:

```sh
python3 scripts/stage-studio.py --bootstrap-demo --photo work/studio-demo/photo.jpg --wave work/studio-demo/wave.mp4 --thumbs-up work/studio-demo/thumbs_up.mp4 --idle work/studio-demo/idle_blink.mp4 --output work/studio-deploy-RELEASE
```

Upload only that explicit staging directory with `railway up --path-as-root`. Subsequent GitHub Docker builds use `/data/demo` and include no private media. Verify `/health`, existing spectra/media, the Motus catalog and People after deployment. Keep the app at one replica while media and preparation jobs use a single volume. Restarted preparation jobs become retryable failures.

## Pi and next work

The existing `pi/launch.py` and `scripts/build-pi-kit.py` ship the same player, clips and MediaPipe assets for offline operation. Account changes do not require a new player. Hardware verification remains outstanding; see `pi/README.md`.

Before wider beta: configure and test database/volume backups, settle retention and upload quotas, and measure preparation capacity. Generation needs a durable job queue and explicit spending limits; payments need a transaction ledger. Neither is required for the current invite-only placeholder studio.

Product language: singular **spectrum**, plural **spectra**, creation **Craft your Spectrum**.

## Release verification

The live health check reports email authentication and PostgreSQL. Anonymous project, admin and media requests return 401. The owner confirmed email sign-in. The production gallery showed both existing spectra and loaded the existing player. Browser logout returned to the public landing page, and revisiting `/studio` required sign-in. Dave's invitation has not been sent. The login page was visually checked. Local PostgreSQL integration tests, SQLite tests and GitHub PostgreSQL checks passed.

## Public page and studio refresh

`/` shows the public product page to visitors and redirects signed-in members to `/studio`. `/welcome` always shows the public page, including for team review. `/studio`, media and admin remain authenticated. The public illustration contains no member photo or footage.

The studio opens on a spectrum gallery. Selecting a card opens the existing player/review/export flow; returning to the collection unloads the player. Creation and account controls work on narrow screens.

Logout uses a POST, revokes the server-side session and expires the Secure/HttpOnly host-prefixed cookie with matching attributes. Ordinary pages now use `Referrer-Policy: same-origin`: `no-referrer` made browser form POSTs send `Origin: null`, which the CSRF check correctly rejected. Token confirmation pages retain `no-referrer`. Cross-origin POSTs remain rejected. GET `/logout` is a confirmation screen only.

Owner confirmed email delivery and sign-in on 25 November. Local browser checks covered the landing interaction, gallery, demo preparation, Motus dialog, mobile layout and logout/private-route behavior. A production-cookie regression covers revocation and cookie flags. GitHub access is now authorized and the CI-gated deployment trigger is enabled.
