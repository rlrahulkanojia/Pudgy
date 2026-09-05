# 16 clips that need re-exporting

**Short version:** 16 of the 180 clips in `LSLTTT-Project` have a few damaged frames inside
them. **The artwork is fine** — this is a file/export problem, so re-exporting the same
shots from the same project should fix it. Nothing needs to be re-animated.

**These will look fine when you play them.** That is the tricky part, and why we're sending
a list rather than saying "you'll see it". See [Why they look fine](#why-they-look-fine).

---

## The list

All 16 are in `MOTION_CLIPS`. **Every clip in `EXPRESSIONS` is fine**, and so is everything
from the previous delivery.

### `MOTION_CLIPS/MOTION_SITTING/PAX/` — 7 of 9 clips

| File | Damaged frames | Where in the clip |
|---|---|---|
| `PAX_MOTION_SITTING_QF1_L.mov` | 2 | 0.71 – 0.79 s |
| `PAX_MOTION_SITTING_QF1_R.mov` | 3 | 0.62 – 0.75 s |
| `PAX_MOTION_SITTING_QF2_L.mov` | 2 | 0.54 – 0.62 s |
| `PAX_MOTION_SITTING_QF2_R.mov` | 2 | 0.71 – 0.79 s |
| `PAX_MOTION_SITTING_QF3_L.mov` | 3 | 0.54 – 0.67 s |
| `PAX_MOTION_SITTING_SIDE_L.mov` | 1 | 0.67 – 0.71 s |
| `PAX_MOTION_SITTING_SIDE_R.mov` | 1 | 0.75 – 0.79 s |

*(`PAX_MOTION_SITTING_FRONT` and `PAX_MOTION_SITTING_QF3_R` are fine.)*

### `MOTION_CLIPS/MOTION_SITTING/POLLY/` — 8 of 9 clips

| File | Damaged frames | Where in the clip |
|---|---|---|
| `POLLY_MOTION_SITTING_FRONT.mov` | 1 | 0.79 – 0.83 s |
| `POLLY_MOTION_SITTING_QF1_L.mov` | 1 | 0.83 – 0.88 s |
| `POLLY_MOTION_SITTING_QF1_R.mov` | 2 | 0.88 – 0.96 s |
| `POLLY_MOTION_SITTING_QF2_L.mov` | 2 | 0.83 – 0.92 s |
| `POLLY_MOTION_SITTING_QF2_R.mov` | 2 | 0.88 – 0.96 s |
| `POLLY_MOTION_SITTING_QF3_L.mov` | 2 | 0.83 – 0.92 s |
| `POLLY_MOTION_SITTING_QF3_R.mov` | 2 | 0.96 – 1.04 s |
| `POLLY_MOTION_SITTING_SIDE_L.mov` | 2 | 1.04 – 1.12 s |

*(`POLLY_MOTION_SITTING_SIDE_R` is fine.)*

All the sitting clips are 1.17 s long, so the damage sits roughly halfway through.

### `MOTION_CLIPS/MOTION_WALKING/PAX/` — 1 clip ⚠️ the worst one

| File | Damaged frames | Where in the clip |
|---|---|---|
| `PAX_MOTION_WALKING_QF1_R.mov` | **5** | **0.00 – 0.21 s** — the very start |

This one is different and worse. The damage is at the **beginning**, and it's 5 frames out
of only 16 — **almost a third of the clip is gone.** This is the only clip of the 16 we
cannot use at all; we've had to drop it entirely. If you only re-export one file, make it
this one.

---

## Why they look fine

When a video player hits a damaged frame, it doesn't show an error — it quietly repeats the
**previous** frame instead and keeps going. So a damaged clip plays as a tiny freeze rather
than anything obviously broken.

In these clips that freeze is **1 to 3 frames**, which at 24 fps is **0.04 to 0.13 seconds**
inside a clip that only runs 1.17 seconds. You would have to know exactly where to look, and
even then it reads as a slightly held pose rather than a fault. Scrubbing the timeline in
QuickTime or Premiere will not show it.

Our training pipeline can't repeat a frame the way a player does — it has to read every
frame as it actually is — which is where the damaged frames turn into visible garbage. Below
is one clip decoded strictly. Frames 15–17 are the damage; every other frame is perfect.

*(Attach `damage_full.png` — the four green frames in the third row.)*

## How to check a file yourself

Every clip stores a count of how many frames it should have. On these 16, that count doesn't
match how many frames can actually be read:

```
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_frames,nb_read_frames \
  -of default=nw=1 PAX_MOTION_SITTING_QF1_R.mov
```

```
nb_frames=28        ← the clip says it has 28 frames
nb_read_frames=25   ← only 25 can actually be read
```

**If those two numbers differ, the file is damaged.** On a healthy clip they match exactly.
We ran this across all 249 clips you've sent us across both deliveries — 233 match, and
these 16 don't.

---

## What we need

- **Re-export the 16 files above** from the same project, same settings (ProRes 4444 with
  alpha, 1080×1080, 24 fps). No re-animation needed.
- **Priority:** `PAX_MOTION_WALKING_QF1_R.mov` first — it's the only unusable one.
- Worth a quick check of the export/copy step that produced these, since 15 of the 16 come
  from the same `MOTION_SITTING` batch. That clustering suggests something went wrong in one
  export run rather than 16 separate accidents.
