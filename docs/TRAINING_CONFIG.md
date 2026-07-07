# Training config — how to point the trainer at this dataset

Trainer: **Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train** (`finetune/train_cogvideox_image_to_video_lora.py`).
Dataset format is the CogVideoX-Fun convention — this folder already matches it.

## Point the script at the data

In `scripts/train.sh` (relative-path mode):

```sh
export DATASET_NAME="/path/to/training_dataset/"          # train_data_dir (folder that CONTAINS train/)
export DATASET_META_NAME="/path/to/training_dataset/metadata.json"
```

`file_path` fields in `metadata.json` are relative (`train/00000001.mp4`), so they resolve against `DATASET_NAME`.

## ⚠️ Required overrides (defaults will NOT work as-is)

The stock script defaults are `--video_sample_n_frames=49 --video_sample_stride=3`, which expect ~147-frame source clips. **These clips are 33 frames**, so you must override:

```sh
  --video_sample_n_frames=33 \   # our clips are exactly 33 frames (8N+1)
  --video_sample_stride=1 \      # read consecutive frames (no skipping)
  --image_sample_size=592 \      # MAX on-grid for portrait; 512 for less VRAM
  --video_sample_size=592 \
  --token_sample_size=592 \
  --enable_bucket \              # REQUIRED — dataloader only exists under this
  # do NOT pass --random_hw_adapt or --training_with_video_token_length
```

> **⚠️ `--enable_bucket` is required, the two adaptive flags are not.** The
> trainer builds its dataloader only under `if args.enable_bucket`, so removing
> it → `UnboundLocalError: train_dataloader`. But `--random_hw_adapt` /
> `--training_with_video_token_length` pick an off-grid downsample that crashes
> CogVideoX1.5's rotary embedding (`RuntimeError: ... Expected size 94 but got
> size 48`). Keep bucket ON, those two OFF.

> **⚠️ Resolution ceiling — why 592, not 768.** The clips are PORTRAIT 768×1360
> (ratio 1.77). CogVideoX1.5-5B-I2V's rotary grid is capped at **48 × 85 patches**
> (= `sample_height/2` × `sample_width/2` = 96/2 × 170/2), i.e. a native canvas of
> **768 (H) × 1360 (W)** — so a *portrait* clip can be at most **768 px tall**. The
> ratio-1.77 bucket is `[672,384]` (H,W) scaled by `sample_size/512` and floored /16:
> - `size 512` → **672×384** (H,W), grid 42×24 — safe, low VRAM
> - `size 592` → **768×432** (H,W), grid 48×27 — **the maximum on-grid resolution**
> - `size 768` → 1008×576, grid **63** > 48 → **crashes** (`Expected size 63 but got 48`)
> - `size 1024` → 1344×768, grid **84** > 48 → **crashes**
>
> So `592` (any value 586–597 lands on 768×432) is the max; `train_pudgy_lora.sh`
> uses it. Do not raise it for these portrait clips.

Everything else from the repo's `CogVideoX-5B-I2V-v1.5` example is fine. Suggested tweaks for a **small dataset (75 clips)** — the repo defaults are tuned for large sets:

| Param | Repo default | Suggested here | Why |
|---|---|---|---|
| `--rank` | 128 | **64** | 128 overfits on ~75 clips |
| `--lora_alpha` | 64 | 32 or 64 | keep = rank or rank/2 |
| `--learning_rate` | 1e-4 | **3e-5** | 1e-4 burns/melts CogVideoX by ~step 1.5k |
| `--lr_scheduler` | constant_with_warmup | cosine (optional) | smoother refinement |
| `--num_train_epochs` | 30 | tune to ~2–4k steps | watch checkpoints for overfit |

Model: `--pretrained_model_name_or_path` → `THUDM/CogVideoX1.5-5B-I2V` (or a local copy). `--mixed_precision bf16`. 1× A100 80GB.

## Clip spec recap (all 75 conform)

768×1360 · 16 fps · 33 frames (2.06 s) · H.264 yuv420p · silent · frame 0 = I2V conditioning frame.

## Note on frame count (33 vs 49/81)

33 satisfies the model's **8N+1** rule and `(frames−1) % 4 == 0`; verified to train on this stack (A100 80GB, fixed-square 768). Note: the rotary-embedding crash we hit was a **resolution** problem (adaptive flags), not a frame-count one — see the warning above. If you ever want longer motion per clip, regenerate at **49 or 81 frames** (`FRAMES=49` in `split6.py`) and set `--video_sample_n_frames` to match (yields fewer clips).

## Files in this folder

| File | Used by trainer? | Purpose |
|---|---|---|
| `train/*.mp4` | ✅ | the 75 clips |
| `metadata.json` | ✅ | the manifest (`file_path`/`text`/`type`) |
| `captions/*.txt` | optional | per-clip caption mirror (diffusers txt-mode) |
| `videos.txt` / `prompts.txt` | optional | line-aligned lists (alt trainers) |
| `metadata_provenance.json` | ❌ | source skit/shot/window per clip (reference only) |
| `README.md` | ❌ | dataset build docs |
