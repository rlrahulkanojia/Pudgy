# Long-form client videos (v8 model, Oct 2026)

Production code for the client's ~10 s videos and story edits on their own iteration_6 living-room sets,
plus the skit remakes (client skits re-generated scene by scene). Results, deliverables and what went
wrong: [`docs/training_reports/longform_client/REPORT_longform_client.md`](../../../docs/training_reports/longform_client/REPORT_longform_client.md).

**Model:** Wan2.2-I2V-A14B, v8 merged LoRA on the low-noise expert, v2 golden on the high-noise expert
(`eval_v8/gen_v8.py` settings): 25 steps unipc, guidance 5.0 / 3.5, flow shift 5, fp8.
**Working dir on the box:** `/workspace/longform_client` (paths in the code assume it; the disk is not a
persistent volume, so everything is mirrored to Azure `pudgy/2026-10-28_longform/workdir/` by `backup_azure.py`).

## Files

| File | Does |
|---|---|
| `keyframes.py` | start pictures: the client's character cut-outs composited on 4 set plates (sofa, door, window, lounge) at a fixed size and spot per item -> `keyframes/<item>_<char>.png` |
| `produce.py` | the 16 videos (13 motion/expression items, alternating Pax/Polly, + 3 interactions), their prompts, the take jobs (`jobs_for`, `loop_job`), `SEED_SHIFT` per re-rolled video |
| `stream.py` | the batch runner: one model load (`--interactive`), takes generated in order, each checked for drift and re-rolled (seed +1000, up to 3 tries), then assembled and uploaded as it finishes |
| `assemble_client.py` | joins a video's takes, colour-matches pinned takes, upscales 720x1280 -> 1080x1920 (lanczos + unsharp, H.264 CRF 16), QA json + contact sheet; also the drift check (`end_drift`, `drift_ok`) |
| `stories.py` | the two 15 s story edits, cut from take 1 of the batch clips |
| `remake.py` | skit remakes: first+last frame on the skit's own frames, cut at the skit's cuts (soundtrack fits), QA against the original scene, 480x848 |
| `skit_report.py`, `make_doc_pdf.py` | data and PDF for the skit write-up |
| `upload.py`, `backup_azure.py` | Azure upload (MD5-skip, verify, 7-day read links); full backup of the job |
| `single.py` | one 241-frame (10 s) generation - OOMs on an A100 80GB, kept for the record |
| `merge_lightning.py` | bakes the lightx2v 4-step distill LoRA into base copies (speed test 1, rejected) |
| `speedtest/speedtest.py` | SageAttention + Wan2.1 time embedding + torch.compile, same take as sample A, isolated venv |

## How a video is made

1. **Chain (default):** 3 takes of 81 frames; take 1 starts on the keyframe, take k on take k-1's last
   frame -> 241 frames = 10.04 s at 24 fps.
2. **Check every take** (`stream.make_take` -> `assemble_client.end_drift`), from the character mask
   (pixels away from the clean plate) and the Pax-blue / Polly-pink centroid:
   - ground line (lowest point, shadow included) start vs end <= 0.03 of frame height,
   - feet line 2.1-2.5 s vs last 0.3 s <= 0.02 (not for jumping - airborne),
   - sideways centroid start vs end <= 0.15 of frame width.
   Height is reported only (background changes gave false alarms). A failed take moves to
   `takes/rejected/<tag>_s<seed>.mp4` and is re-made with seed +1000; if 3 fail, the least drifted is kept.
3. **Look at every take.** The checks miss figures that are not Pax/Polly (a visitor at the door), fading,
   and a pair walking off with little ground change. Reject by eye with `takes/rejected/<tag>_s<seed>.mp4`
   **plus** `<same>.manual` (one-line reason): the same seed and start frame give an identical video, and
   the `.manual` marker keeps the runner from restoring it.
4. **Pinned mode** (`<id>:loop`): every take is first+last frame on the keyframe. Used when a motion keeps
   drifting over the chain (jumping grew Pax 1.3x, walk_together walked off). Pinning shifts the palette;
   `assemble_client.match_colour` undoes it with per-channel tone curves fitted to the keyframe (a quadratic
   RGB map was tried and turned the black outlines navy). A pinned video also loops seamlessly.
5. **Loop versions** (`--loops`, `produce.LOOPS`): takes 1-2 of the chain + `t3loop` aimed back at the
   keyframe, colour-matched across the last join; wrap jump ~1-2x a normal frame change.

```bash
cd /workspace/longform_client; set -a; . /workspace/Pudgy/.env; set +a
GEN_EXTRA="--video_size 1280 720" /workspace/Pudgy/.venv-wan/bin/python -u stream.py 2026-10-28 --loops   # all
GEN_EXTRA="--video_size 1280 720" /workspace/Pudgy/.venv-wan/bin/python -u stream.py 2026-10-28 jumping:loop
python stories.py                                   # story edits (no GPU)
GEN_EXTRA="--video_size 848 480" python remake.py run little_spoon four_am_show   # skit remakes
python backup_azure.py                               # mirror everything to Azure
```

## Lessons

- **Seed 42 drifted at the end of a take** in 5 of the first 11 take-1s (walk off, shrink, lurch). The
  last ~0.6 s of an 81-frame take is where it goes wrong; story edits use only the first 2.75 s of take 1.
- **The door set pulls Pax toward the door** (waving, blushing at x .27 slid to ~.35): keyframes moved to
  x .35. **The open door invites visitors** - blushing got a bear, a figure and a purple penguin at the
  door at the end of three takes; delivered trimmed to 9.7 s.
- **Expression prompts say "medium shot"**: "medium close-up" (not in the v8 shot vocabulary) made a take
  cut to a close-up mid-take.
- **Skit remakes:** none of the 99 client skits uses only trained actions; list the untrained moments per
  skit before generating. A part can be re-made from its first frame only (`remake.FREE`, no end anchor)
  to replace an untrained action (Coming Home's lift -> standing hug). Duplicate checks must count only
  the characters in the scene (pink scenery reads as Polly).
- **Speed:** a single 241-frame generation OOMs (the fp32 modulation tensor alone is ~25 GiB); the 4-step
  distill LoRA is ~5x faster but loses 17% of the motion; SageAttention + Wan2.1 time embedding + compile
  is 1.26x faster per step with matching output (not yet adopted - needs `sageattention==1.0.6` in the
  production venv and `--attn_mode sageattn --force_v2_1_time_embedding --compile`).
