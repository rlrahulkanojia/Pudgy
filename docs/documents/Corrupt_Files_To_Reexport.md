# 21 clips need re-exporting

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

20 in the `iteration_4` delivery, 1 new in `iteration_5`. Everything in `CRYING` and the
whole of `iteration_3` are fine.

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
| `PAX_EXPRESSION_BLUSHING_QF2_L.mov` † | `BLUSHING/PAX` | 8 † | 1.33–1.63s | 1.71s |

**21 files, 61 damaged frames.**

† **The new one is damaged a different way — please read this separately.** On the other 20
the frames are present but blank. On `PAX_EXPRESSION_BLUSHING_QF2_L.mov` eight frames are
**not in the file at all**: the timestamps jump straight from 1.292s to 1.667s, and every
frame that *is* there looks correct. The file says 41 frames and 33 can be read. Because
the gap sits just before the final frame, the first 32 frames are a clean run we can still
use — so this one is not a write-off, but it is still 8 frames short of the performance.

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

**Different numbers = damaged.** On a healthy clip they match. Of the 310 clips delivered
so far, 289 match and these 21 don't.

> ⚠️ Some damaged frames have a valid label but empty contents, so a check that only counts
> frames will pass them. Use the two-number check above.

---

## What we need

1. **Re-export the 21 files** — same project, same settings (ProRes 4444 with alpha,
   1080×1080, 24 fps).
2. **`PAX_MOTION_WALKING_QF1_R.mov` first.**
3. **Please check the export step itself** — see below.

**This is recurring, and the request has now been missed twice.** The first 16 were
reported on 3 Sep. The files received on 5 Sep were byte-for-byte identical — not
re-exported — and that delivery added 4 new damaged files. **The 26 Sep delivery re-sent
the entire project folder and again none of the 20 was re-exported** — all 20 verified
byte-for-byte identical by MD5 — while adding a 21st damaged file.

So three deliveries have now carried the same 20 broken clips untouched. Whatever writes
these files is still faulty, so a re-export alone will likely reproduce it in the next
batch. Two damage shapes have now appeared — a short run of blank frames mid-clip (20
files) and a run of frames missing outright (1 file) — and both point at the export/copy
step rather than the animation.

**Please confirm receipt of this list before the next delivery.** Twice now it has gone
out and come back unactioned with no reply, and we cannot train on these clips.
