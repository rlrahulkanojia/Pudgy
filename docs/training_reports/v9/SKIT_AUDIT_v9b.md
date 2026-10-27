# Skit audit for v9b (long-form): are the client skits fit to train on?

**Date** 2026-10-26 · **Data** `raw/iteration_1/30_videos` (29) + `raw/iteration_2/72_videos` (72),
pulled from Azure to `/workspace/Data/raw/` · **Plan** [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md) §0.1 ·
**Scripts + per-shot verdicts** `finetune/wan/scenes/audit/` (`probe.py`, `shots_ocr.py`, `pair.py`, the re-check scripts, `shot_review.json`).

## Verdict

**Usable, with no client request needed.** The `_Base` exports have **no burned-in meme
captions**. The text in them belongs to the scene: signs, packaging, phones, clocks. The
work left is ours: a stricter per-shot curation than v4 had.

## What was checked

| check | result |
|---|---|
| Inventory | 29 + 72 files, as the iteration READMEs say. iteration_2 holds **2 animatics**, not 1: `PP_Feed_Animatic_v4.mov` (540×960) and `PP_ChillGF_Animatic_v3.mp4`. Both excluded → **99 skits** |
| Format | 98 skits are H.264 1080×1920 at 24 fps. `PP_Customize_9x16_BASE_1.mp4` is 23.976 fps (resample) |
| Integrity | `nb_frames == nb_read_frames` and zero decoder errors on all 101 files |
| Storyboards | **every skit has its PDF** (aliases: Arm→NumbArm, Bicept→Biceps, Comfy→GF_Comfy, Sleep→MovieSleep, Thinking of Her→Thinking_Of_Herd, Fixer→Fixer2, Customizing→Customize). Single-page, text-extractable, numbered beats → usable for ordered captions. `PP_Unhelpful_v2.pdf` has no video |
| Burned-in captions | OCR (RapidOCR, every 6th frame, 4,822 frames). Each storyboard's beat-1 caption line is found on screen in **2 of 99** skits, and on inspection neither is an overlay caption (BF: a title card shot; VoiceMessage: a phone UI shot), both dropped below |
| Duration | 20.3 min total; 409 detected shots, 366 ≥ 1 s (19.4 min); 230 ≥ f49, **112 ≥ f81** (11.0 min) |

## Per-shot review (all 366 shots ≥ 1 s, by eye)

| verdict | shots | min | what |
|---|---|---|---|
| **keep** | 311 | 16.4 | the leads acting in sets: bedroom, living room, kitchen, gym, shops, car, outdoors, and plain pale-blue grounds with furniture. The first pass kept 321 (16.9 min); the re-check below moved 10 to *check* |
| drop: graphic / overlay | 10 | | title card, battery and OK/NOT-OK meters, chat / voice-message / speech bubbles (100%, BF, Thinking_Of_Herd, VoiceMessage) |
| drop: ad / merch | 2 | | IB_Picnic "Available now", Customize "Perfect Pair Collection" (a photo) |
| drop: insert, no lead | 21 | | phone/map UI, dashboard, empty bed/couch, establishing plates, extreme flipper close-ups |
| drop: duplicate leads | 7 | | plush claw machine, dream/thought bubbles, framed photo, poster, Mother's Day card |
| drop: off-style | 4 | | MovingIn isometric cut-away, Universe underwater |
| check | 11 | | LittleSpoon#0 (lead hidden under a blanket), plus the 10 from the re-check |

Burned-in text that stays (in-world): shop signs, product packaging (Polar Tarts, Freetos,
Snowckers), the speed-limit sign, alarm clocks, the AC display. Captions must not
transcribe it.

## The detector is not enough: cuts and camera moves inside shots

Frame-difference scan of every shot ≥ 4 s, then strips checked by eye. Of 24 flagged shots:

- **12 contain a missed cut or transition**: hard cuts (BF ×2, Chair, KeepUsSafe, Sherlock,
  Temp, Without, Yapper), a cross-dissolve (BF), a dissolve (TheresYou), a fade from black
  (IB_Picnic), a blanket wipe (IB_Picnic).
- **11 contain a camera move**: zoom out / pull-back (LifeWithHer, Vibing, BackTogether, BF),
  push-in (BF), pans (Economic ×2, Handful, MyOffer, MissedHim, Temp).

Shorter shots were not scanned this way, so expect more. v9 captions say "fixed camera,
locked-off shot", so a clip with a hidden zoom teaches exactly the zoom-out regression v9
is meant to fix (REPORT_v8 §8).

## Re-check (2026-10-26, evening): what the first pass missed

Every first-pass verdict was looked at again on the contact sheets, and the doubtful shots
were pulled at four time points each. The corrections are in `shot_review.json` (`verdict`,
`reason`, and a new `notes` field).

**Wrongly kept (10 shots, now `check`):**
- **MothersDay2 #1, #3, #5**: Polly's mum, drawn as Polly plus a pearl necklace, alone or
  next to Polly. dupcheck sees two Pollys.
- **Aging #0–#3**: a time-lapse. Day and night flash past, and snow and trees build up
  around a still Polly.
- **Jump #0**: a fly-in. Polly grows from a dot through a light burst, then a dark extreme
  close-up.
- **BF #1**: a red overlay arrow appears mid-shot, besides the missed cut.
- **Tasks #1**: a missed cut or pull-back to a wide shot near the end. The shot is 3.4 s, so
  the ≥ 4 s missed-cut scan never looked at it.

**Kept, but the captions must account for them (`notes`):**
- **Side characters:** crowds of silhouettes (TheresYou), gym-goers (ActingTough), a
  spectator crowd (Award) and a third penguin (BubbleWrap). v4 also records grey penguins
  (ActingTough, KeepUsSafe) and a flower-crab (Scared).
- **Outfits on the leads:** a hard hat, scarf, headband, tie, backpack, shower cap, hood and
  a long white beard (Yapper #4).
- **Effects:** dark vignettes (WithHer, Without).

**Measured across the kept shots:**

| | result | consequence |
|---|---|---|
| Two characters | 59% of v4's kept scene time has both leads | two-character captions are v9b's core (plan §3.4) |
| Motion | max drift from frame 0: median 9.7 (p10 3.0, p90 31.8), on the measure where v8 motions score 2–16 and the phone clips 0.83. Only 7 shots are as still as the phone clips | no "freeze" risk like v9c's F3 |
| Held drawings | 72% of frames are on ones; 16% sit in holds of ≥ 6 frames (acting holds) | the skits are not animated on twos |
| Settings | two kinds: full sets, and plain pale-blue grounds with a few pieces of furniture | captions name the kind (plan §3.4) |
| Reuse / leakage | no identical files; no cross-skit pair of shots closer than 48 of 480 dHash bits. The nearest pairs are the same flat-ground furniture with different action | holding out whole skits is clean |
| Lengths | 41 kept shots (0.8 min) are under f33, the plan's old smallest bucket | the plan adds f21 |

Scripts in `finetune/wan/scenes/audit/`: `holds.py`, `shot_motion.py`, `xdup.py`,
`sheet.py` (contact sheets for any shot list). A modal-colour test for flat grounds was
inconclusive, because the walls of full sets are as flat as the plain grounds, so settings are
labelled by the curator.

## Existing curation to reuse: v4 (`processed/v4_ltx_249clip/_work/curator/`)

70 iteration_2 skits, each adjudicated cut-by-cut at 0.1 s by a curator pass, with
keep/drop, `camera`, `environment`, `action_beats` and an object catalog: 289 scenes, 277
kept with a lead (13.8 min). Camera labels: 83% static, the rest push-in / pull-back / pan.

It does not transfer as-is, because v4's rules differ from v9b's:

| v4 rule | v9b needs |
|---|---|
| no split on pan / zoom / push | trim to static spans, or keep moving spans only with a camera caption, in a separate tagged group |
| overlay scenes kept (9) | drop |
| 26 short scenes merged (carry cuts) | never merge across a cut |
| tokens `pxngn0` / `plngn0`, appearance words banned | v8/v9 wording: names + colour anchors, registry phrases for props and sets |

`action_beats`, `environment`, cut times and the object catalog carry over. iteration_1 (29
skits) has no equivalent pass and needs one.

## Next steps (all ours)

The full list, with the decisions behind it (base, resolution, static spans only, side
characters, outfits, buckets), is in the plan: [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md)
§0.1. In short:

1. **Curator pass, v9b rules:** iteration_1 fresh; iteration_2 re-adjudicated from the v4
   JSONs, seeded with `shot_review.json`. It records:
   - every cut and fade, at frame accuracy;
   - static spans only;
   - characters with their screen side, outfits and side characters;
   - prop events;
   - overlays anywhere in a span.

   It drops lookalikes, time-lapses and fly-ins.
2. **Registry phrases** for side characters, outfits and settings.
3. **Captions** from the storyboard beats + v4 `action_beats`, in v9 wording, with a scene
   sheet per skit.
4. `prep_v9b.py` (f21 · f33 · f49 · f81 at 480×848, `dupcheck.py` on every window) and the
   scene eval suite.
5. **Holdout:** 6 whole skits (3 per iteration).

**Nothing is needed from the client for this.** Optional, as before: character-layer alpha
exports of skit shots (Assets_v9 §5.5).
