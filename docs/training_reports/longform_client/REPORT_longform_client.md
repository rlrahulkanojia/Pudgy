# Long-form client videos — report (28 Oct – 1 Nov 2026)

**Delivered:** 16 single-action videos + 5 loop versions (1080x1920, ~10 s), 2 story edits (15 s), 7 skit
remakes (480x848, with the original soundtrack) and a write-up of 4 of them, all in Azure
`pudgy/2026-10-28_longform/`. Every video was checked by eye; ~20 takes were rejected and re-made on the way.
Code and pipeline: [`finetune/wan/longform_client/`](../../../finetune/wan/longform_client/README.md).

## Deliverables (Azure `pudgy/2026-10-28_longform/`)

| Folder | What | Files |
|---|---|---|
| `/` | single-action videos, v8 model, 720x1280 generated, upscaled to 1080x1920 | 16 `<item>_1080x1920.mp4` + 5 `<item>_loop_1080x1920.mp4` |
| `stories/` | Surprise Visit, Cheer Up, Polly - 6 shots each from take 1 of the batch clips | 2 |
| `skits/` | Coming Home, Reunion, Not Mad, Little Spoon, 4AM Show, Morning Fart, Wake Up Early: silent, with the skit's soundtrack, side by side with the original | 21 + the write-up PDF |
| `samples/` | the approved sample (walking, full settings) and the 4-step distill sample | 2 |
| `speedtest/` | side by side: current settings vs SageAttention + time embedding + compile | 1 |
| `workdir/` | code, keyframes, cut-outs, every take (incl. `rejected/` with `.manual` reasons), QA sheets and json, logs, docs, speed test | 496 |

## The 16 videos

Set: S sofa, D door, W window, L lounge. Size = character height vs frame 0 (min-max); seams = change at
the two take joins / the median frame change; wrap = last->first frame (loop versions ~1-2).

| Video | Char / set | Made with | Notes |
|---|---|---|---|
| walking | Pax / S | chain s42-44, + loop | the approved sample; loop wrap 1.2 |
| running | Polly / W | chain s1042-1044 | seed 42 hopped off toward the door; t3 flagged x0.81 height = background change (check fixed) |
| waving | Pax / D | chain s1042-1044, + loop | re-made from a re-centred keyframe (x .27 -> .35): Pax kept sliding to the door and shrinking |
| sitting | Polly / S | chain s42-44, + loop | calm; loop seam at 6.7 s 3.8x |
| jumping | Pax / W | **pinned** s1042-1044 (= loop) | chain seeds shrank Pax early (s42) or grew him 1.3x over takes 2-3 (all t3 seeds) |
| happy | Polly / W | chain s42-44 | turns slightly to the side in the last second |
| surprised | Pax / S | chain s1042-1044 | seed 42 shot Pax out of frame to the right at 3.3 s (sideways check added) |
| angry | Polly / W | chain s42-43, t3 s1044 | t3 seed 44 faded Polly pale and see-through |
| confused | Pax / L | chain s42-44 | head-scratch, puzzled side-glances |
| crying | Polly / W | chain s1042-1043, t3 s2044 | seed 42 stopped crying and smiled; t3 s1044 moved back into the room; ends with cartoon puddles |
| neutral | Pax / S | chain s42-44 | |
| laughing | Polly / W | chain s42-44 | smoothest joins with wave_together |
| blushing | Pax / D | chain s1042, s2043, s2044 | **9.71 s**: visitors at the open door at the end of three takes (bear, figure, purple penguin); last 8 frames trimmed |
| wave_together | both / S | chain s42-44 | |
| walk_together | both / W | **pinned** s1042-1044 (= loop) | chain seeds 42 and 1042 walked the pair off into the room |
| hug | both / S | chain s42-44 | hug, a happy pause, a second hug (the prompt's "stepping toward each other" separates them) |

Expressions say "medium shot": "medium close-up" (not in the v8 shot vocabulary) cut a crying take to a
close-up mid-take.

## Story edits (stories/)

Cut from take 1 of the batch clips (the first 2.75 s, before any end-of-take drift), hard cuts, no GPU:
**Surprise Visit** (sitting, waving, happy, blushing, hug, wave together) and **Cheer Up, Polly** (crying,
confused, walking, hug, laughing, walk together). The characters face the camera in every shot; the story
is in the order of the shots.

## Skit remakes (skits/)

Seven client skits from `raw/iteration_2`, chosen for few props; none uses only trained actions. Each scene
is re-generated between the skit's own frames (first + last frame, parts <= 3.4 s), colour-matched, cut at
the skit's cut points so the soundtrack fits. 480x848, not upscaled (the user's call: ~3.5x faster).

| Skit | Length | Parts | Notes |
|---|---|---|---|
| Coming Home | 11.5 s | 6 | the skit's two lifts replaced by standing hugs (`remake.FREE`, no end anchor), at the user's request |
| Reunion | 10.4 s | 5 | the 0.5 s whip pan where Polly runs off is not matched (3 seeds) |
| Not Mad | 9.2 s | 6 | flipper lift and drink hand-over both read well |
| Little Spoon | 8.2 s | 5 | closest of all; written up |
| 4AM Show | 9.0 s | 4 | last part re-made once (jump 18.7x); written up |
| Morning Fart | 7.8 s | 4 | yawn later and quicker (part 1 took 3 seeds); written up |
| Wake Up Early | 6.6 s | 4 | opening push-in kept as best of 3: a faint patch on the wall for 0.6 s; written up |

Write-up (prompt, details, observation, links per skit): [`SKIT_REMAKES_bedroom.md`](SKIT_REMAKES_bedroom.md),
also as PDF in `skits/pudgy_skit_remakes_bedroom.pdf` and as a Claude Doc.

## Speed

| Option | Per take | Result |
|---|---|---|
| Current settings (720x1280, 25 steps) | ~37 min (82 s/step + ~3 min load/decode) | used for the batch |
| One 241-frame generation | - | OOM on A100 80GB (fp32 modulation tensor ~25 GiB), 3 attempts |
| lightx2v 4-step distill LoRA, baked into the bases | ~7 min | 17% less motion, feet planted; rejected |
| 480x848 | ~4-12 min per skit part | used for the skit remakes |
| SageAttention 1.0.6 + `--force_v2_1_time_embedding` + `--compile` | ~30 min (65 s/step) | **1.26x**, 41 vs 63 GB VRAM, PSNR 36.1 dB vs sample A, same motion; not adopted yet |

## Open

- Adopt the speed settings for future generation (install `sageattention==1.0.6` in `.venv-wan`).
- Optional: blushing at the full 10 s (pin its last take to the keyframe), a continuous 10 s hug (pinned).
- The read-only links handed out expire 5-7 Nov 2026; new ones: `python upload.py <date>` / `upload.link`.
