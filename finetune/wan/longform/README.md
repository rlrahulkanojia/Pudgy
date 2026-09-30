# Long-form scenes (v8)

Re-makes a client skit (iteration_1: finished 1080×1920 skit + storyboard PDF) shot by
shot with the v8 model, then assembles a client-ready 10–15 s video.

**Delivered (2026-09-30):** LifeWithHer (14.5 s) and EatingStages (13.9 s) —
Azure `pudgy/v8/longform/final/` (with and without the original soundtrack), and the
dashboard's *Long-form scenes* group.

## Pipeline

1. **Beats → prompts** (`shots.py`). One entry per storyboard beat: the start keyframe (the
   beat's first frame in the client's reference, 720×1280), the frame count, and a prompt in the
   v2 skit-caption style (`STYLE + character anchors + action + "the camera stays still"`).
   Beats whose shared table has several props also get an **end keyframe** (`end=`) — see below.
2. **Generate** (`run_production.py <seeds> [ids]`): v8@550 (`--lora_weight` v7-combined base +
   step-550 τ), 1280×720 portrait, 25 steps, Wan's default negative + anti-duplicate terms.
   One batch per beat so each take lands, and is reviewed, early. `GEN_EXTRA="--video_size 1280 720"`.
3. **QA every take**
   - `qa_take.py` — contact sheet + border drift (zoom/camera creep), frozen fraction, largest
     single-frame jump (pops). Inspect the frames around any jump > ~3× median.
   - `dupcheck.py` — never more than one Pax or one Polly in a frame. Flags need eyes: a raised
     flipper split off by its outline, or a pink bench, reads as a second body.
4. **Fix what fails**
   - **Prop pops / morphs** (two characters + props on one table lose the layout ~0.8 s in):
     first+last-frame generation, `end=` → `--ei` (anchors the final state).
   - **Palette jump** that first+last-frame introduces mid-shot: `colormatch.py` — offset measured
     on the background border (not the whole frame; a close-up's character skews it), plus a
     flat-fill that snaps near-background pixels to the keyframe's exact background colour
     (the client's backgrounds are one flat colour; ours carried ~1 level of grain).
   - **One bad tail**: trim (`trim_end` in the takes JSON).
5. **Assemble** (`assemble.py <name>` with `<name>.json`): trim, lanczos upscale to 1080×1920 +
   luma unsharp (restores line weight lost in the upscale), hard cuts, H.264 CRF 16 faststart;
   a second copy with the reference soundtrack. Prints duration / frames / black-frame QA.

`run_test.py` is the golden-vs-v8 comparison on beat 1 (near-identical per seed; v8 used).

## What to tell the client

- These are **remakes**: every beat starts from the client's own frame. New scenes need new
  keyframes (or images generated first).
- The soundtrack is the original skit's, not re-edited to the new cuts.
- Complex prop interactions needed end-frame anchoring + grading; single-character beats
  worked directly.

Paths assume the box layout (`/workspace/longform`, `/workspace/Data/raw/iteration_1`).
