# Pudgy Penguins — Data Request, Round 4

**What this is:** a short, specific follow-up to the motion/expression delivery. Two things
are new — **16 files that need re-exporting** because of a technical fault, and **four
motion clips we're still missing** — plus a reminder of the Round 3 items that are still
open.

The delivery itself was good. The angle coverage (nine per action) is genuinely useful, the
alpha channels are clean, and `laughing` filled a real gap. Everything below is small and
specific.

---

## 1. Sixteen files need re-exporting (technical fault, not artwork)

Sixteen of the `.mov` files in `LSLTTT-Project` have a damaged video stream. The artwork is
fine — this is an export or file-write problem, so a straight re-export from the same
project should fix it.

**How it shows up:** the file says it has one number of frames, but only some of them
actually open. On several, a few frames in the middle of the clip come out visibly broken —
the penguin's body missing, just a sliver of the head. It is not something you'd catch by
scrubbing quickly, because most players quietly paper over the bad frames.

**All 16 are in `MOTION_CLIPS`. The `EXPRESSIONS` folders are all clean**, as is everything
from the previous delivery.

### `MOTION_CLIPS/MOTION_SITTING/PAX/` — 7 of 9

```
PAX_MOTION_SITTING_QF1_L.mov      PAX_MOTION_SITTING_QF2_R.mov
PAX_MOTION_SITTING_QF1_R.mov      PAX_MOTION_SITTING_QF3_L.mov
PAX_MOTION_SITTING_QF2_L.mov      PAX_MOTION_SITTING_SIDE_L.mov
                                  PAX_MOTION_SITTING_SIDE_R.mov
```

### `MOTION_CLIPS/MOTION_SITTING/POLLY/` — 8 of 9

```
POLLY_MOTION_SITTING_FRONT.mov    POLLY_MOTION_SITTING_QF2_R.mov
POLLY_MOTION_SITTING_QF1_L.mov    POLLY_MOTION_SITTING_QF3_L.mov
POLLY_MOTION_SITTING_QF1_R.mov    POLLY_MOTION_SITTING_QF3_R.mov
POLLY_MOTION_SITTING_QF2_L.mov    POLLY_MOTION_SITTING_SIDE_L.mov
```

### `MOTION_CLIPS/MOTION_WALKING/PAX/` — 1

```
PAX_MOTION_WALKING_QF1_R.mov      ← the worst one: says 16 frames, only 11 exist
```

**Priority:** `PAX_MOTION_WALKING_QF1_R.mov` is the only one we cannot use at all — a third
of the clip is simply absent. We recovered the 15 sitting clips, but they're unreliable and
we'd rather train on clean files.

---

## 2. Filenames — four small fixes

Nothing here is urgent; we handle it on our side. But fixing it at source keeps your copies
and ours matching.

| Where | What's wrong |
|---|---|
| `MOTION_CLIPS/MOTION_WALKING/PAX/` | Three files are named `PAX_MOTION_WAVING_*` (`FRONT`, `QF1_L`, `SIDE_R`) but contain **walking**. Two of them are also byte-identical to each other. **We can't tell which angles they are, so we've had to drop them** — please confirm the angles, and whether `WALKING_SIDE_R` is missing. |
| `EXPRESSIONS/HAPPY/POLLY/` | Contains `PAX_HAPPY_SIDE_R.mov` — a **Pax** clip in Polly's folder. We flagged this on 20 Aug and it came back unchanged. Is there a real `POLLY_HAPPY_SIDE_R`? |
| `EXPRESSIONS/HAPPY/PAX/` | Missing the `PAX_` prefix, and uses `QF_L`/`QF_R` where everything else uses `QF1_L`/`QF1_R`. Also only 7 angles, not 9. |
| `EXPRESSIONS/NEUTRAL/` | Uses `FR` instead of `FRONT`, and a different quarter-angle scheme again (`QF_L`, `QF_L2`, `QF_L3`). |

**Going forward:** `<CHARACTER>_<ACTION>_<ANGLE>.mov` with angles `FRONT`, `QF1_L/R`,
`QF2_L/R`, `QF3_L/R`, `SIDE_L/R` — the convention most of the delivery already uses.

---

## 3. Four motion clips we're still missing

From the nine actions in Round 3, you've delivered five (walking, running, sitting, waving,
jumping). **Four are still outstanding**, and the first is the most valuable:

- [ ] **Standing idle** — the character simply standing, not performing an action, with the
      subtle breathing and small weight shifts a normal idle animation has

> **Why standing idle matters more than it sounds.** Every motion clip we have shows the
> character *doing* something. With no example of it deliberately doing nothing, the model
> has no counter-example for "stop" — so asking it to stand still may just produce another
> action. It is the motion equivalent of the `neutral` expression, which is what makes
> "don't emote" work on the expression side.
>
> Please shoot it as a real idle rather than a held pose: gentle breathing, a little weight
> shift, an occasional blink. A frozen character reads as a broken render.

- [ ] **Turning** — the character rotating on the spot, left and right
- [ ] **Head turn** — body still, head turning to look left/right and back
- [ ] **Bouncing** — a soft repeated bob, lighter than the jump

**Same format as before:** one action per clip, static camera, nine angles, both characters,
transparent background.

### One new thing worth adding: start/stop transitions

Everything delivered so far is a clip of an action already in progress. Nothing shows a
character **beginning** or **ending** an action. That's what makes "start walking" or "come
to a stop" hard for the model to produce.

- [ ] **Start** — from standing still into walking (a second or two)
- [ ] **Stop** — from walking back to standing still

Just for walking is enough; we don't need it for every action.

---

## 4. Clip length — please aim longer

This is the one spec we'd most like changed. Our original request asked for **3–5 seconds**
per clip. What's arrived is considerably shorter:

| | Shortest | Longest |
|---|---|---|
| Motion clips | 0.67 s (walking) | 1.38 s (jumping) |
| Expression clips | 0.88 s (happy) | 2.50 s (neutral) |

Short clips limit what the model can learn about *sustaining* an action or an expression —
it sees the beginning of a laugh but rarely a laugh that holds. **For anything new, please
aim for 3–5 seconds**, even if that means the character simply repeats the cycle or holds
the pose for longer. Where an action loops (walking, bouncing), letting it run for several
full cycles is ideal.

---

## 5. Still open from Round 3

These were requested in Round 3 and haven't arrived yet. Listed in the order they'd help us
most:

- [ ] **Interaction clips — Pax & Polly together.** Nothing has been delivered. This is the
      single biggest gap: two-character shots are the hardest thing for the model, and it's
      the one area where more footage helps most directly. Hugging, holding flippers,
      high-five, back-to-back, piggyback, doing the same action in sync, plus one clear
      side-by-side size comparison.
- [ ] **Polly character design sheets.** Turnaround (8 angles) and expression sheet, as
      transparent PNGs with colour codes. **Pax has a turnaround; Polly still has none** —
      and Polly is the character that comes out less consistent, so she's the one we most
      need reference art for.
- [ ] **Pax expression sheet** — the turnaround arrived, the expression sheet didn't.
- [ ] **Remaining expressions:** sad, scared, confused, crying, affectionate, exasperated.
- [ ] **Camera variety** (Round 3 §5) — everything so far is the same square framing at the
      same distance. A mix of close-up / medium / wide would help. *We currently synthesise
      this ourselves, which works but is not the same as real footage.*

---

## Priority order

If you can only do some of this:

1. **Re-export the 16 damaged files** (§1) — small effort, and one clip is unusable without it.
2. **Interaction clips** (§5) — the biggest quality gap in the whole programme.
3. **Polly's design sheets** (§5) — she is the weaker character and this is why.
4. **Standing idle** (§3) — the missing counter-class for "stop".
5. **Turning / head turn / bouncing + start-stop transitions** (§3).
6. **Longer clips going forward** (§4).

**Delivery:** same as before — any shared drive or link, one folder per section. Best-quality
files, no watermarks or text. We'll handle the technical formatting.
