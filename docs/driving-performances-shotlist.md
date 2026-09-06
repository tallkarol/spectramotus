# Driving performances — recording shot list (v4)

These clips are recorded **once** and reused for every portrait. Body clips go through a cloud
motion-transfer model (the person in the photo performs your motion); face clips drive LivePortrait
locally. Consistency matters more than acting: identical timing across portraits is what makes
screen-to-screen handoffs work.

## Setup (both sessions)

- Phone on a tripod, **vertical (portrait) orientation** for body clips; 1080p or 4K at **30 fps**
  (iPhone: Settings → Camera → Record Video → 4K 30; turn off HDR video and "Auto FPS"). We convert to
  constant 30 fps afterwards anyway.
- Lock exposure / white balance / focus (tap-and-hold AE/AF lock). Even, frontal light from ~45° both
  sides so there is no shadow on the wall. Plain matte wall, light grey or beige — **not green**.
- Camera at chest height, ~3 m away. Whole body **including feet**, with ≥10 % headroom.
- Tape mark on the floor = the neutral spot. Fitted plain clothes that contrast the wall; no watch,
  no props, empty pockets. Silent (audio is stripped).
- One clip per file, one take per file: `<role>_t01.mp4`, `<role>_t02.mp4`. Say the role name
  before the start hold if you like — it gets trimmed.

## The neutral pose (every body clip starts and ends here)

Feet shoulder-width on the mark, arms relaxed at the sides, facing the lens, neutral expression, eyes to
the lens. **Hold it for 1 s at the start and 1 s at the end of every clip.**

## Body clips (`driving-performances/body/portrait/`)

| File | Length | Performance |
|---|---|---|
| `neutral_hold` | 3 s | Just the neutral pose. |
| `idle_sway_a` | 8 s | Tiny weight shifts, breathing, micro head moves. Nothing theatrical. |
| `idle_sway_b` | 8 s | Look around the room slowly, adjust a sleeve, settle back. |
| `wave_back` | 4 s | Raise right hand, 2–3 friendly waves, lower. |
| `wave_big` | 4 s | Bigger, arm-over-head wave. |
| `walk_off_left` | 4 s | Turn, walk out of frame to the LEFT, then **1 s of empty frame**. Be fully out. |
| `peek_in_left` | 4 s | From the left edge: lean in, look at the lens, hold ~1 s, retreat out. Starts and ends empty. |
| `walk_in_left` | 5 s | Start out of frame (1 s empty), walk in from the LEFT, stop on the mark, turn to lens, hold. |
| `nod_greet` | 2 s | The "sup" upward nod. |
| `shrug` (optional) | 3 s | |
| `thumbs_up` (optional) | 3 s | |

Right-side variants are mirrored in software. Record the right-side versions too only if time allows.

### Settle clips (photo pose → neutral)

Motion transfer starts from *your* pose, not the photo's. For photos that are not neutral we play a
one-time "settle" clip. Start **in the photo's pose**, hold 1 s, relax to neutral over ~1.5 s, hold 1 s.

| File | Pose to start in |
|---|---|
| `settle_from_arms_raised` | Both arms up and wide, chin up (mimic `karol_rome_1.jpg` on a reference screen). |
| `settle_from_hands_in_pockets` | Hands in pockets (Einstein). |
| `settle_from_arms_crossed` | Arms crossed. |
| `settle_from_hands_on_hips` | Hands on hips. |
| `settle_from_leaning` (optional) | Leaning on something with one hand. |

## Face clips (`driving-performances/face/`)

Head and shoulders, phone at eye level, face ≥ 400 px wide in frame, same lighting rules. Lips closed,
no talking, head rotations ≤ 25°, face never leaves the frame. **0.5 s neutral hold at both ends.**

| File | Length | Performance |
|---|---|---|
| `blink_single` | 1.5 s | One natural blink. |
| `blink_double` | 2 s | Two quick blinks. |
| `breathe_idle_a` | 8 s | Breathing, tiny natural head movement, eyes on lens. |
| `breathe_idle_b` | 8 s | Same with a slow sway. |
| `glance_left` | 3 s | Eyes then head glance left, return. (Right is mirrored.) |
| `glance_up` | 3 s | Glance up, return. |
| `smile_soft` | 3 s | Closed-mouth smile, hold, release. |
| `smile_big` | 3 s | Full smile, hold, release. |
| `nod_yes` | 2 s | Two small nods. |
| `nod_greet` | 1.5 s | Upward "sup" nod. |
| `eyebrow_raise` | 2 s | |
| `head_tilt_listen` | 3 s | Tilt, hold, return. |

## After recording

Drop the files into the folders above (they are git-ignored; they will move to the asset bucket).
`spectramotus drivers prepare` converts them to constant 30 fps, builds the mirrored variants and the
LivePortrait templates, and writes `driving-performances/manifest.yaml`, where the trim/loop points are
authored once per clip.
