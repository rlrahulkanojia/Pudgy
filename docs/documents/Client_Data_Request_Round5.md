# Pudgy Penguins — Data Request, Round 5

**What this is:** the request that decides how far the next model can go. The latest model
handles **expressions** well and can now play out short, calm movements — but **fast, dynamic
movement (running, chasing, quick jumps) and full scenes are held back by the amount of
footage**, not by the training. This round asks for more of exactly that, sized by how
dynamic each action is.

Thank you for the last delivery — **blushing** worked on the first try (the model now
recognises it for both characters), and the first **interaction clips** gave us the starting
frames for our first two-character tests: waving together and hugging already render without
the characters blending into each other. Those clips go into the next training run.

---

## 1. Why we're asking for more movement footage

A simple rule has emerged from the results: **the faster and bigger a movement, the more
footage the model needs to learn it.** A smile changes slowly and a few seconds of footage
is enough; a run changes a lot every fraction of a second, and the model only sees each
stride briefly.

Here is how much *unique* movement footage the model has today — across all characters and
camera angles combined:

| Action | Total unique footage | Longest single clip |
|---|---|---|
| walking | 7.6 s | 0.67 s |
| running | **11.3 s** | **0.71 s** — less than two strides |
| sitting | 11.8 s | 1.2 s |
| waving | 16.7 s | 1.1 s |
| jumping | 22.0 s | 1.4 s |

Two things limit the model beyond the totals:

- **Every clip is one performance filmed from nine angles.** The angles are useful, but the
  model sees the *same run* nine times rather than nine different runs. Variety of
  performance is what teaches it how a run can look.
- **Every action happens on the spot.** Nothing enters, crosses or leaves the frame, so the
  model has never seen a character actually *travel* — which almost every real scene needs.

---

## 2. The main ask: dynamic movement, sized by how dynamic it is

Please prioritise **more different takes and longer clips** over more camera angles.

| Tier | Actions | Per action, per character | Clip length |
|---|---|---|---|
| **A — fast / travelling** | running, running across the frame (left→right and right→left), chasing, jumping sequences (several hops), falling down and getting up, dancing | **≥ 5 different takes**, about **90 s** in total | 3–5 s each, **at least 3 full cycles** of the action |
| **B — body gestures** | waving, pointing, turning around, sitting down, standing up, bouncing, stretching | **≥ 5 different takes**, about 45 s in total | 3–5 s each |
| **C — subtle** | standing idle (breathing, weight shift, blink), head turn, expressions | **≥ 3 different takes**, about 20 s in total | 3–5 s each |

**What "different takes" means:** the same action performed differently — a slightly faster
run, a bouncier run, a tired run, a happy skip. Not the same animation re-rendered.

**Camera angles:** four are enough this round — `FRONT`, `QF2_L`, `QF2_R`, `SIDE_L` —
instead of nine. We'd rather have five runs from four angles than one run from nine.

**For the travelling actions**, please include some clips where the character **enters,
crosses and leaves the frame**, as well as the on-the-spot version.

**Speed variants help a lot:** if you can, give walking and running at two speeds (normal
and fast), and label which is which.

---

## 3. Starts, stops and changes between actions

Everything delivered so far is an action already in progress. For scenes, the model needs
to see characters **change** what they are doing:

- [ ] idle → walk → idle (start and stop)
- [ ] walk → run → walk
- [ ] standing → sitting down → standing up
- [ ] idle → jump → land → idle
- [ ] an **expression changing during a movement** — e.g. walking along happily, then
      surprised; running while laughing

The last one matters because when we currently ask for a movement *and* an expression at
the same time, the expression tends to take over half-way through. A few clips that show
both at once will fix that faster than anything we can do on our side.

---

## 4. Two characters together — building on what you sent

The first interaction clips (hugging, waving together) were exactly the right idea. To make
this usable in scenes we need a little more:

- [ ] **Hugging — the missing angles.** It arrived with 3 of 9 angles (`QF1_L`, `QF2_L`,
      `QF3_L` only). `FRONT` and the right-hand angles would complete it.
- [ ] **More interactions, 3–5 s each, ≥ 3 takes:** high-five, holding flippers and walking
      together, handing something over, one chasing the other, a kiss on the cheek,
      back-to-back, dancing together, sitting side by side.
- [ ] **Please keep both characters in the same clip but name the file by the action**, as
      you did (e.g. `INTERACTION_HUGGING_FRONT.mov`) — that works well for us.

---

## 5. Finished scenes — more of the skits, and the pieces behind them

The finished skits (like *Life With Her* and *Eating Stages*) are the richest footage we
have: real sets, props, camera moves, and characters actually moving through a room. We
used two of them to produce the long-form demo, and we'll cut clips from all of them for
training. More would help directly:

- [ ] **More finished skits**, in the same 1080×1920 format — as many as you can share.
- [ ] **Clean versions without text, captions, end cards or product shots** where they exist.
- [ ] **The individual shots before editing** (one file per shot), if your pipeline has them —
      this saves us cutting them apart and loses nothing at the joins.
- [ ] **Empty backgrounds** for the recurring sets (dining table, couch, bedroom, park bench,
      kitchen) as still images — useful for building new scenes.
- [ ] **The storyboard / prompt PDF for each skit**, as you did for the first batch.

---

## 6. Still outstanding from earlier rounds

- [ ] **Re-export the damaged files — second request.** The 20 files from our re-export list
      came back in the last delivery **byte-for-byte identical to the originals**, so the
      fault is still there. One new file has the same problem:
      `EXPRESSIONS/BLUSHING/PAX/PAX_EXPRESSION_BLUSHING_QF2_L.mov` plays as 41 frames but only
      33 are real. **21 files in total — listed at the end of this document.** The artwork is
      fine; a straight re-export from the project should fix them.
- [ ] **Standing idle** (Round 4) — still the most useful single clip type, now under Tier C.
- [ ] **Turning, head turn, bouncing** (Round 4) — now under Tiers B/C.
- [ ] **Missing expressions:** sad, scared, affectionate, exasperated.
- [ ] **Polly's design sheets** (turnaround + expression sheet) and **Pax's expression sheet.**
- [ ] **`happy` and `confused` at full length** — they arrived at 0.88 s each, against
      1.7–2.5 s for the other expressions, so the model sees about half as much of them.

### Small filing notes from the last delivery

- `EXPRESSIONS/BLUSHING/PAX/` contains `POLLY_EXPRESSION_BLUSHING_QF1_L.mov` — a **Polly**
  clip in Pax's folder (we've handled it; please confirm it is Polly).
- Blushing has 7 angles (no `SIDE_L` / `SIDE_R`). Are those coming?
- `INTERACTION_HUGGING__QF3_L.mov` has a double underscore.
- **Third request:** `EXPRESSIONS/HAPPY/POLLY/` still contains `PAX_HAPPY_SIDE_R.mov` (is there a
  real `POLLY_HAPPY_SIDE_R`?), and `EXPRESSIONS/HAPPY/Pax/` still lacks the `PAX_` prefix and
  uses `QF_L`/`QF_R` instead of `QF1_L`/`QF1_R`.

---

## Format — same as before

| | Single-character clips | Interactions | Finished skits |
|---|---|---|---|
| Resolution | 1080×1080 | 1080×1080 | 1080×1920 |
| Frame rate | 24 fps | 24 fps | 24 fps |
| Background | transparent (alpha) | transparent (alpha) | as finished |
| Camera | still | still | as finished |
| File | `.mov` ProRes 4444 | `.mov` ProRes 4444 | `.mp4` or `.mov` |
| Name | `<CHARACTER>_<ACTION>_<TAKE>_<ANGLE>.mov` | `INTERACTION_<ACTION>_<TAKE>_<ANGLE>.mov` | `<Skit name>.mp4` |

The only new part of the naming is **`<TAKE>`** — `T1`, `T2`, `T3`… — so we can tell
different performances apart. For example `PAX_RUNNING_T3_QF2_L.mov`.

**Please check one thing before sending:** open a couple of files end to end in a player that
shows every frame (not just scrubbing). Last time's damaged files only showed up that way.

---

## Priority order

If you can only do some of this, in this order:

1. **Tier A fast movement** (§2) — running and travelling across the frame first. This is the
   biggest limit on the next model.
2. **Re-export the 21 damaged files** (§6) — small effort, second request.
3. **More finished skits + clean/per-shot versions** (§5) — the fastest route to full scenes.
4. **Starts, stops, transitions and expression-during-movement** (§3).
5. **More interactions + the missing hugging angles** (§4).
6. **Tier B/C** — gestures, idle, head turn (§2), and the missing expressions (§6).

**Delivery:** any shared drive or link, one folder per section. Best-quality files, no
watermarks or text. We handle the technical formatting on our side.

---

## Appendix — the 21 files to re-export

Each is missing real frames partway through (players hide this by holding the previous frame).
Paths are as in the latest delivery (`iteration_5`).

| Folder | Files |
|---|---|
| `MOTION_CLIPS/MOTION_SITTING/PAX/` | `QF1_L`, `QF1_R`, `QF2_L`, `QF2_R`, `QF3_L`, `SIDE_L`, `SIDE_R` (7) |
| `MOTION_CLIPS/MOTION_SITTING/POLLY/` | `FRONT`, `QF1_L`, `QF1_R`, `QF2_L`, `QF2_R`, `QF3_L`, `QF3_R`, `SIDE_L` (8) |
| `MOTION_CLIPS/MOTION_WALKING/PAX/` | `PAX_MOTION_WALKING_QF1_R.mov` — missing from the very first frame |
| `EXPRESSIONS/CONFUSED/PAX/` | `PAX_MOTION_CONFUSED_QF1_R.mov`, `PAX_MOTION_CONFUSED_QF3_L.mov` |
| `EXPRESSIONS/CONFUSED/POLLY/` | `POLLY_MOTION_CONFUSED_QF1_R.mov`, `POLLY_MOTION_CONFUSED_SIDE_L.mov` |
| `EXPRESSIONS/BLUSHING/PAX/` | `PAX_EXPRESSION_BLUSHING_QF2_L.mov` (new) |

Sitting files are named `<CHARACTER>_MOTION_SITTING_<ANGLE>.mov`.
