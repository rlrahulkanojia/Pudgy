# 20 clips need re-exporting

**The problem:** in each clip, a few frames near the middle are **blank** — the picture for
those frames was never written into the file. **The artwork is fine**; nothing needs
re-animating. It's a fault in the export/copy step, so a re-export should fix it.

**Why they look fine when you play them:** a player that hits a blank frame doesn't show an
error — it holds the previous frame a moment longer and carries on. The clips run ~1 second
and the blank stretch is typically 0.08s, so it reads as a tiny stutter you'd have to hunt
for. Our training process can't do that: it reads every frame literally, so a blank frame
stays blank.

---

## The list

All in the `iteration_4` delivery. Everything in `CRYING` and the whole previous delivery
are fine.

| File | Folder | Blank frames | Where | Clip |
|---|---|---|---|---|
| `PAX_MOTION_WALKING_QF1_R.mov` ⚠️ | `MOTION_WALKING/PAX` | **5** | 0.00–0.21s | 0.67s |
| `PAX_MOTION_CONFUSED_QF1_R.mov` ⚠️ | `CONFUSED/PAX` | 3 | 0.29–0.42s | 0.88s |
| `PAX_MOTION_CONFUSED_QF3_L.mov` ⚠️ | `CONFUSED/PAX` | 2 | 0.21–0.29s | 0.88s |
| `POLLY_MOTION_CONFUSED_QF1_R.mov` | `CONFUSED/POLLY` | 3 | 0.71–0.83s | 0.88s |
| `POLLY_MOTION_CONFUSED_SIDE_L.mov` | `CONFUSED/POLLY` | 4 | 0.67–0.83s | 0.88s |
| `PAX_MOTION_SITTING_QF1_L.mov` | `SITTING/PAX` | 2 | 0.71–0.79s | 1.17s |
| `PAX_MOTION_SITTING_QF1_R.mov` | `SITTING/PAX` | 4 | 0.58–0.75s | 1.17s |
| `PAX_MOTION_SITTING_QF2_L.mov` | `SITTING/PAX` | 3 | 0.50–0.62s | 1.17s |
| `PAX_MOTION_SITTING_QF2_R.mov` | `SITTING/PAX` | 2 | 0.71–0.79s | 1.17s |
| `PAX_MOTION_SITTING_QF3_L.mov` | `SITTING/PAX` | 3 | 0.54–0.67s | 1.17s |
| `PAX_MOTION_SITTING_SIDE_L.mov` | `SITTING/PAX` | 2 | 0.62–0.71s | 1.17s |
| `PAX_MOTION_SITTING_SIDE_R.mov` | `SITTING/PAX` | 1 | 0.75–0.79s | 1.17s |
| `POLLY_MOTION_SITTING_FRONT.mov` | `SITTING/POLLY` | 2 | 0.75–0.83s | 1.17s |
| `POLLY_MOTION_SITTING_QF1_L.mov` | `SITTING/POLLY` | 1 | 0.83–0.88s | 1.17s |
| `POLLY_MOTION_SITTING_QF1_R.mov` | `SITTING/POLLY` | 3 | 0.83–0.96s | 1.17s |
| `POLLY_MOTION_SITTING_QF2_L.mov` | `SITTING/POLLY` | 2 | 0.83–0.92s | 1.17s |
| `POLLY_MOTION_SITTING_QF2_R.mov` | `SITTING/POLLY` | 2 | 0.88–0.96s | 1.17s |
| `POLLY_MOTION_SITTING_QF3_L.mov` | `SITTING/POLLY` | 3 | 0.79–0.92s | 1.17s |
| `POLLY_MOTION_SITTING_QF3_R.mov` | `SITTING/POLLY` | 3 | 0.92–1.04s | 1.17s |
| `POLLY_MOTION_SITTING_SIDE_L.mov` | `SITTING/POLLY` | 3 | 1.00–1.12s | 1.17s |

**20 files, 53 blank frames.**

⚠️ **= unusable.** We rescued the other 17 by re-reading them until a clean read came
through. For these three the frames are genuinely empty, so re-reading can't help and
they're excluded from training. `PAX_MOTION_WALKING_QF1_R.mov` is the worst — 5 blank
frames of 16, at the very start, so **a third of the clip doesn't exist**. Re-export that
one first.

---

## Checking a file yourself

Every file records how many frames it *should* have. On these 20, that doesn't match how
many can actually be read:

```
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_frames,nb_read_frames \
  -of default=nw=1 PAX_MOTION_SITTING_QF1_R.mov

nb_frames=28        ← the file says 28 frames
nb_read_frames=25   ← only 25 can be read
```

**Different numbers = damaged.** On a healthy clip they match. Of the 284 clips delivered
so far, 264 match and these 20 don't.

> ⚠️ Some damaged frames have a valid label but empty contents, so a check that only counts
> frames will pass them. Use the two-number check above.

---

## What we need

1. **Re-export the 20 files** — same project, same settings (ProRes 4444 with alpha,
   1080×1080, 24 fps).
2. **`PAX_MOTION_WALKING_QF1_R.mov` first.**
3. **Please check the export step itself** — see below.

**This is recurring, not a one-off.** The first 16 were reported on 3 Sep. The files
received on 5 Sep were byte-for-byte identical — not re-exported — and that delivery added
4 new damaged files. Whatever writes these files is still faulty, so a re-export alone will
likely reproduce it in the next batch. The damage is always the same shape: a short run of
consecutive blank frames mid-clip, which points at the export/copy step rather than the
animation.
