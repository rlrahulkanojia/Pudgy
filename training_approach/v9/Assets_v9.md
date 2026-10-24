# v9 assets: named props and set backgrounds, in training and inference

**In short:** every prop, set and character is listed once in an **asset registry**. The
registry gives each one name and one look. Training captions and inference prompts are built
from it, never typed by hand. A new prop is a new registry entry plus one short training stage
on the same growing LoRA. The new background plates are used in training as composites, and
at inference as the backdrop of the first frame.

| | |
|---|---|
| **Extends** | [Training_Approach_v9.md](Training_Approach_v9.md): §0 (phone stage first), §3.2 backgrounds, §3.3 props, §3.4 captions, §4 incremental stages, §7 keyframes |
| **Data** | `raw/iteration_6` (2026-10-24): the first prop (the iFin phone) and the first empty set plates (igloo living room / TV room). Changelog (`Data/raw/iteration_6/CHANGELOG.md`, outside the repo; Azure `pudgy/raw/iteration_6/CHANGELOG.md`) |
| **Status** | 2026-10-24. **Built:** the registry, `prep_v9.py`, the phone-stage training and eval scripts, the [runbook](RUNBOOK_v9c_phone.md). **Still to write** (v9b and inference): `scenes/composite_plates.py` with plate floor annotations, `longform/compose_keyframe.py` |
| **Code** | `finetune/wan/assets/registry.{toml,py}` · `finetune/wan/prep_v9.py` · `finetune/wan/train_pudgy_v9.sh` · `finetune/wan/eval_v9/` · `finetune/wan/v9_box.sh` |

---

## 1. What iteration_6 adds

The delivery is two zips, extracted to `Samples/`; the extraction can be rebuilt from them.
387 files: 323 `.mov` and 64 `.png`.

| | contents | state | use |
|---|---|---|---|
| **Phone, in use** | `PRP_PHONE_TEXTING_{SITTING,STANDING}`, Pax + Polly, 9 angles: **36 clips × 74 frames**, of which **frames 0–63 are animation** (64–73 repeat frame 63) | Pax 18/18 clean. **Polly: 11 of 18 damaged** (frames lost, some at frames 1–5); every damaged clip keeps ≥ 26 clean consecutive animated frames | v9c-phone, as clean windows (§4.3) |
| **Phone, views** | `PRP_PHONE_01…27`, 1080×1080 RGBA | **24 unique** (25–27 repeat 01–03): back, front with a black screen, both edges, turns | inference keyframes; eval reference. **Not trained** (§4.1) |
| **Set plates** | `BGS/BG_LIVINGROOM`, 1080×1920 PNG | **33 unique** of 37 (§5.1) | v9b composites; inference keyframes |
| **Sitting V2** | 18 clips | a **new performance** (PSNR 13–29 dB against V1), all clean, loops cleanly (seam 0.11–0.35) | second sitting take (replay pool) |
| **Confused V2** | re-export of the same performance | fixes the 4 damaged Confused clips | replaces every earlier confused source |
| **Interactions** | hugging (3 angles, 21 f), waving (9, 26 f), walking (9, 17 f) | all 21 clean; all seen before (walking first in `test_data`). **Never trained** | v9b scenes |

**Duplication.** 257 of the 323 clips are byte-identical to `iteration_3`–`5` or `test_data`.
Only **66 are new**: 36 phone clips, 18 Sitting V2 and 12 Confused V2 re-exports. Prep
de-duplicates by MD5 and checks that every other `iteration_6` clip hashes to a known one.

**Not fixed in this delivery:**
- `PAX_MOTION_WALKING_QF1_R` and `PAX_EXPRESSION_BLUSHING_QF2_L` are still damaged.
- **No fast-motion footage arrived.** Running, jumping and walking are unchanged, so v9a is
  still blocked on Round 5 Tier A.

**Confirmed by the client 2026-10-24:** `PRP_PHONE` is the iFin (`Prp_iFin`). The phone
clips are used as delivered; the client is not asked for more active phone clips.

### 1.1 Filename overrides (raw is never renamed)

| file (as delivered) | read as | evidence |
|---|---|---|
| `INTERACTION_CLIPS/WALKING/INTERACTION_WAVING_SIDE_R.mov` | interaction **walking**, SIDE_R | **Rename approved 2026-10-24.** 17 frames like the other walking clips (waving clips are 26), and walking has no other SIDE_R |
| `INTERACTION_CLIPS/HUGGING/INTERACTION_HUGGING__QF3_L.mov` | hugging, QF3_L | doubled underscore |
| `MOTION_CLIPS/MOTION_WALKING/PAX/PAX_MOTION_WAVING_{FRONT,QF1_L,SIDE_R}.mov` | **not used** | 16-frame walking-length renders named WAVING inside WALKING (QF1_L and SIDE_R are the same bytes). The true angle is unknown, so prep_v8 skips them, and prep_v9 keeps that rule |

The first two are keyed by **MD5** in `prep_v9.py` (`OVERRIDES`), so a later re-send of the
same bytes is corrected too. They are applied when the interactions are discovered; the
interactions themselves are built in v9b, not in the phone stage.

---

## 2. The rule: one name, one look

Each asset's identity travels on two channels, and they must always agree.

1. **Text.** One canonical phrase, used word for word in every training caption that shows
   the asset and in every prompt that asks for it. This is what lets a prompt *call* the
   asset.
2. **Pixels.** The client's reference art, placed into the first (or last) frame at
   inference. This is what *guarantees* the look.

Text alone drifts: the model draws *a* phone, not *this* phone. Pixels alone cannot be
called by name, and cannot appear mid-shot. Both come from the registry.

---

## 3. The asset registry

### 3.1 The file

The registry is one file, [`finetune/wan/assets/registry.toml`](../../finetune/wan/assets/registry.toml),
with one entry per asset. [`registry.py`](../../finetune/wan/assets/registry.py) loads it,
lints it and builds caption fragments. prep_v9 and eval_v9 already import it; the prompt
compiler and keyframe composer will. TOML, not YAML: Python reads it with no extra package.

```toml
[assets.phone]
kind = "prop"
client_ids = ["PRP_PHONE", "Prp_iFin"]          # confirmed 2026-10-24
name = "the iFin phone"
phrase = "the iFin phone, a small pale grey phone with a black screen, three camera lenses on the back and a little tuft on top"
refs_dir = "raw/iteration_6/Samples/PROPS/PHONE/PRP_PHONE"
holdout_views = ["PRP_PHONE_05", "PRP_PHONE_14", "PRP_PHONE_20"]   # one back, one front, one edge
[assets.phone.clips]
clips_dir = "raw/iteration_6/Samples/PROPS/PHONE"
animated_frames = 64                             # frames 64-73 are a hold
[[assets.phone.actions]]
folder = "PRP_PHONE_TEXTING_STANDING"
label = "phone_texting_standing"
action = "standing and holding {phrase}, up in front of its chest in one flipper, tapping at it with the other, the screen turned toward itself, head still and eyes on the phone"
label_text = "texting on a phone, standing"
```

The action clauses were written from the footage frame by frame. One flipper holds the
phone at chest height and the other taps it; the back of the phone faces the camera.

Characters (`pax`, `polly`) carry v8's anchors verbatim, and prep refuses to run if they
differ from `prep_v8.ANCHORS`. Sets (`living_room`) list every plate with its family, time
of day and framing (§5.1); floor polygons are added when the plates are annotated for v9b.
`kind` is open: `outfit` is next (the skits show Pax in a headband, a scarf and a jersey),
then `effect`.

### 3.2 Lint rules (`registry.py lint`, also run by prep)

1. **Unique names, no containment.** No asset's name may appear inside another's name or
   phrase.
2. **No character colour words in asset wording.** "blue" and "pink" are the character
   anchors; v4 rendered Polly blue when her captions lacked "pink" (FINDINGS_v6 §3). The
   phone is pale blue-grey, so it is written "pale grey". This is a precaution: v8's
   "plain pastel blue background" did not turn Polly blue.
3. **References exist and are MD5-unique.** This catches `PRP_PHONE_25–27` = `01–03` and
   duplicate plates, so a held-out view cannot leak in under another number.
4. **Every action names its asset** (`{phrase}`), and every plate's time of day has a
   caption word.

The check that named means visible is structural rather than a lint rule. Every manifest
row lists the assets in it, and the caption is compiled from that list, so a caption cannot
name an absent asset.

### 3.3 Why "the iFin phone"

It is the client's own name, so prompts read like their storyboards. It is also a rare word,
and a rare word attaches to one look more easily than "phone", which Wan already knows as a
generic object. The description covers what makes this phone different from a generic one:
the three lenses, the tuft and the pale grey. The full phrase goes into every caption.

---

## 4. Training

### 4.1 What teaches the phone, and what does not

| sample | teaches | phone stage |
|---|---|---|
| **in-use clips**: the client's texting clips | the name ↔ this object; stays the same object while it is handled; how it is held | **all of the new data**: 32 clips (4 QF2_R held out) × 4 grounds × up to 3 lengths = 380 clips |
| **scene shots**: the phone in a set, untouched | sits still in a room at scene scale | v9b: plates + mined skit shots (~50 of the 249 v4 skit prompts mention a phone) |
| **stills**: the 24 views | **nothing useful in training.** In image-to-video, a 1-frame sample is conditioned on itself (the first frame is given), so the loss barely depends on the caption | not trained: they are the **inference keyframe pixels** and the **eval reference** |

The stills row corrects v9 §3.3, which planned the stills as musubi image samples. In an
image-to-video model the first frame carries the look; the clips carry the name.

### 4.2 Captions are compiled, not written

The template is v8's, with the prop's action clause from the registry
(`prep_v9.prop_caption`):

> A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and
> flat pastel colors, showing Pax, a short round blue penguin, standing and holding the
> iFin phone, a small pale grey phone with a black screen, three camera lenses on the back
> and a little tuft on top, up in front of its chest in one flipper, tapping at it with the
> other, the screen turned toward itself, head still and eyes on the phone; texting on a
> phone, standing; static close-up shot, eye level, facing the camera directly, front
> view; plain white studio background.

v8's "static … shot" wording is kept for this stage, so the phone is the only new variable.
The camera-clause fix (v9 §1.3 #5) moves to v9a.

### 4.3 The phone data, specifically

- **Clean windows, by timestamp.** Each phone clip trains on its longest run of real,
  animated frames (inside 0–63), found from frame timestamps. Two Polly clips start after a
  gap at frames 1–5 (`window 6+58`), and one after a gap at 35–37 (`38+26`). Decoding uses
  passthrough timing: ffmpeg's constant-rate output fills a gap with duplicates and can shift
  the next frame by one, which was seen on `STANDING_SIDE_R`. The decoder also emits a torn
  frame on some runs and not others, so the torn-frame check and retry stay on.
- **Lengths f21 / f41 / f61.** This is prep_v8's ladder from the f21 floor to each window's
  longest legal length; damaged clips get shorter rungs. f21 is also every expression's
  common bucket, so length never predicts "phone".
- **Holdout:** QF2_R (4 clips), as in v8, plus 3 held-out views for evaluation.
- **Size is fine at 1024².** At zoom 1.0 the phone is about 150 px tall, about 19 latent
  rows. The 1.4× crop proposed earlier is dropped, and v8's zoom ladder is kept. The
  few-tokens problem is real only at **scene scale** in v9b (480×848, characters a third of
  the frame). That is where the VAE check of the phone matters.
- **Near-static footage.** The texting clips move less than the neutral expression clips.
  The decision (2026-10-24) is to train on them as delivered. The suite's `frozen` flag and
  v8-label motion energy are what would show "phone ⇒ freeze".

### 4.4 One LoRA that grows, stage by stage

```
pudgy-v8-merged ──(v9c-phone)──► merge ──(v9b scenes)──► merge ──(next props)──► …
                 fresh low-noise LoRA, rank 16, ≤ 200 steps
                 data = new asset clips + 20% stratified replay of everything before
```

- **No per-prop LoRAs loaded side by side.** Stacking two files that share a base counted
  that base twice in v7: a **76% error** (Training_Approach_v8 §1 #1). Each stage is merged
  exactly into the base (`lora_tools_v8.py combine`), so inference always loads one base
  plus at most one new file.
- **Replay** is sampled per (character, label). Each group first gets one clip at its kind's
  common bucket, then the rest at random across angles, grounds and lengths. The phone
  stage draws 95 replay clips across all 26 groups.
- **Gate:** held-out v8-label loss ≤ 1.05× base, and held-out phone loss < 1.0× base. A
  failed stage is re-run with more replay, never kept ([runbook](RUNBOOK_v9c_phone.md) §5).
- The registry's `trained_in` field records which stage and checkpoint learned each asset.

### 4.5 Adding a future prop

1. The client delivers PNG views and in-use clips: one folder per action, the same
   `<CHAR>_<PRP_…>_<ANGLE>.mov` naming as the phone.
2. Add a registry entry: name, phrase, `refs_dir`, `holdout_views`, `clips` and one
   `actions` row per folder. Then run `registry.py lint --data …`.
3. prep picks up every prop with clips. Today `prep_v9.py` builds one stage (`STAGE =
   v9c_phone`); the second prop group needs a small extension: build only props whose
   `trained_in` is empty, and add earlier prop stages to the replay pool.
4. Train the stage with the same scripts on the latest merged base, gate it, merge it and
   set `trained_in`.
5. At inference the prop is usable as pixels from step 2, and callable by name after step 4.

---

## 5. Backgrounds in training (v9b)

### 5.1 The plates

These are first-pass families from the contact sheets, recorded per plate in the registry.

| family | plates | framing | best use |
|---|---|---|---|
| `door_a`: couch, glass table, door closed → open | `AFrame_001`, `Frame_002–004` | wide | composites; **entering through the door** |
| `door_b`: round window, side table, door closed → open | `Frame_01–05` (`Frame_06` = `Frame_04`) | wide, open floor | composites; entrances |
| `tvroom`: couch, glass table, round window; **day → morning → storm → sunset → evening → night** | `PP_BG_Ep089Sc04_TVRoom_001 (2)`, `_002–_007`, `BG_Ep081Sc06_TVRoom_001`, `Frame_001_` | wide | composites; time of day as a prompt attribute |
| `tv_fg`: TV + armchair in the foreground | `PP_BG_Ep089Sc04_TVRoom_001`, `PP_Scene02_v01`, `PP_Scene02_v01 (2)` | wide / medium | keyframes for "watching TV"; few composites (nothing can stand behind the TV) |
| `dim_couch`: Ep080, dim light, TV foreground, blanket | `BG_Ep080Sc01_LivingRoom_*` (6) | medium | evening keyframes |
| `closeup`: couch, popcorn, cushion | `Frame_001 (3)`, `Frame_002 (2)`, `PP_Scene02_v02`, `PP_BG_Ep038Sc01_Livingroom_A001` | close | seated keyframes only |
| `dressed`: props on the coffee table | `Frame_001`, `Frame_001 (2)` | medium | scene shots with props on a surface |

**Same rooms as our skits, different shots.** Posters is drawn in the `door_b` room and Tone
in the `tv_fg` room. Checked 2026-10-24, however: no plate lines up pixel-for-pixel with any
frame of those skits; the shots are framed differently. The plates are in-distribution
backgrounds, not the exact backgrounds of shots we have.

### 5.2 Composites (the main use)

- **The shadow is built in.** The client clips' alpha contains a semi-transparent black
  contact shadow (mean alpha ≈ 57–60, RGB 0), which shrinks mid-hop on jumps. Compositing
  onto a plate carries the right shadow, so **v9 §3.2's synthetic shadow is not needed**.
- **Placement.** Each plate gets a floor polygon and a scale line (character height against
  y), annotated once and stored in the registry. Feet go on the floor, and characters are
  a quarter to a third of the frame tall, as in the sampled skits.
- **Only in front of furniture.** A plain composite cannot put a character behind the couch.
- **Format.** 9:16 buckets (480×848) at f33 / f49 / f81, as v9b. Captions take the set's
  phrase, plus the time-of-day word when it is not day.
- **Start with day plates.** In the one dim-lit skit sampled (Rollover, a bedroom) the
  characters are not darkened, which suggests plain composites stay faithful at night, but
  that is one sample.

### 5.3 Travel and entrances on real sets

v9 §3.1 slides in-place cycles across the frame. Doing it on a plate turns it into a scene:
a character enters, crosses the room or leaves. The door sequences add real entrances. The
foot-slide guard (≤ 2 px per frame) and the ≤ 30% synthetic cap still apply. This is the
travel data the high-noise LoRA needs.

### 5.4 Holdout and limits

- One plate per main family (`Frame_05`, `PP_BG_Ep089Sc04_TVRoom_003`,
  `BG_Ep080Sc01_LivingRoom_003`, `PP_Scene02_v02`) is never composited or trained. These
  are the unseen-set test, like v8's lavender and sky-blue grounds.
- Composites stay at **≤ 25% of `v9_scenes`** (v9 §3.2), and the pastel grounds stay in
  training.

### 5.5 What a plate ↔ shot pairing would buy (optional client ask)

If the client exported the **character layer with transparency for skit shots** (the same
export they already do for the motion clips), every shot would become training footage of a
real performance. That footage has travel, turns, props and two characters together, which
is the "more distinct performances" every run has lacked (v9 F1). The empty plate of that
exact shot would let us composite it back, or onto other sets.

We cannot do this ourselves from the files we have: the plates do not match the skit
framing (§5.1). It is a request for an extra export, not something hidden in this delivery.

---

## 6. Inference

```
scene spec            set + plate + time of day; characters (action, position); props (id, state,
(shots.py, per beat)  held_by / on_surface / enters_at)
   │
   ├─► keyframe composer (longform/compose_keyframe.py, to write)
   │     plate → character first frame from the alpha library (shadow included), on the floor
   │     at registry scale → props from their reference PNGs at registry scale
   │     → last frame too, when a prop enters or the layout must hold
   ├─► prompt compiler (Prompt_Compiler_Spec, reading the registry)
   ├─► Wan2.2 + the merged v9 base (+ the latest stage): I2V, or first+last frame (FLF)
   ├─► QA: qa_take (pops, freezes, drift), dupcheck, colormatch → regenerate a failed take
   └─► chain: last frame → next shot's first frame; the scene sheet keeps the wording fixed
```

- **New sets without client frames.** The composer builds the first frame from any
  registered plate.
- **A prop is usable before it is trained.** Its PNG goes straight into the first frame;
  training adds staying the same object through motion, and being named.
- **Held poses** come only from poses the client animated: today, texting while sitting or
  standing.
- **FLF is the default when a prop enters or several props share a table** (v8's long-form
  lost such a layout about 0.8 s in).

---

## 7. Per-asset checks

Implemented for the phone in `finetune/wan/eval_v9/`, and run on v8merged first as the
"before".

| check | how | bar |
|---|---|---|
| **regression** | v8 diffusion classifier on QF2_R clips of every v8 label | held-out loss ≤ 1.05× base |
| **learned** | held-out (QF2_R) phone clips, loss under the training caption | < 1.0× base |
| **name binding** | the same classifier: the registry phrase against "a phone", "a small red book", "a white cup", "a black TV remote control" | reported, breaks ties |
| **looks right** | Tier-2 `phone` suite: start from held-out phone frames; contact sheets, pop and freeze flags | reviewed by eye |
| **callable by text** | Tier-2 `textcall`: start with no phone; prompt asks for it; CLIP phone-presence | reported; expected weak in I2V |
| **no leak** | Tier-2 `leak`: v8 prompts; a phone must not appear | reviewed |

---

## 8. Changes to the v9 plan

| v9 said | now | why |
|---|---|---|
| order v9a → v9b → v9c | **v9c-phone first**, then v9b; v9a when Tier A arrives | the phone is the only complete new data; v9a's data has not arrived |
| §3.3: props as 10–30 still images (musubi image samples) | **stills not trained**; in-use clips carry the name | a 1-frame I2V sample is conditioned on itself |
| §3.2: add a soft synthetic contact shadow | not needed | the shadow is in the clip alpha |
| §3.2: plates cleaned from skit frames | 33 client plates | iteration_6 |
| §1.3 #5: camera clause in captions | deferred to v9a | keep one new variable per stage |

---

## 9. Status and next steps

| step | status |
|---|---|
| iteration_6 mirrored to Azure | ✅ 2026-10-24, 389 blobs, size+MD5 verified |
| `raw/iteration_6/CHANGELOG.md`, `raw/README.md` | ✅ |
| PRP_PHONE = iFin | ✅ confirmed by the client |
| registry, `prep_v9.py`, train + eval scripts, `v9_box.sh`, runbook | ✅ on branch `v9/phone-stage`; prep built locally end to end |
| **v9c-phone run** | **ready.** Needs a GPU box; [runbook](RUNBOOK_v9c_phone.md) |
| client: re-export the 11 Polly phone clips + 2 old files | to send (not blocking: windows are used meanwhile) |
| client (optional): character-layer exports of skit shots (§5.5) | to decide |
| v9b: plate floor annotations, `composite_plates.py`, skit mining | after the phone stage |
| `compose_keyframe.py`, prompt compiler reading the registry | after v9b |

---

## 10. Risks and open questions

- **Small props at scene scale.** At 480×848 a phone is a few tokens. Check the VAE round
  trip before v9b relies on it.
- **Near-static phone clips** could teach "phone ⇒ freeze". Watch the `frozen` flag and
  v8-label motion energy.
- **"Callable by text" may stay low** in I2V because the first frame dominates. FLF with a
  composed last frame is the dependable path for a prop that enters.
- **Night plates** are unverified beyond one skit.
- **Replay at 20%** is thin: 3–4 clips per (character, label). The gate decides, and the
  runbook re-runs at 35% if it fails.
