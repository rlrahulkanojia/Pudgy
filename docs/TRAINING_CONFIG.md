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
  --video_sample_size=768 \      # our native short edge; or leave 512 to train smaller
  --token_sample_size=768 \
  --enable_bucket \              # keep on: no center-crop, buckets by resolution (portrait 768x1360)
```

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

33 satisfies the model's **8N+1** rule and `(frames−1) % 4 == 0`, and the 1.5-I2V pipeline pads latent frames internally, so it trains. If you hit a `patch_size_t`/OFS shape error on your diffusers build, the safe fallback is to regenerate at **49 frames** (one-line change in `split6.py`: `FRAMES=49`) — that yields ~32 clips but uses the more battle-tested length for this model.

## Files in this folder

| File | Used by trainer? | Purpose |
|---|---|---|
| `train/*.mp4` | ✅ | the 75 clips |
| `metadata.json` | ✅ | the manifest (`file_path`/`text`/`type`) |
| `captions/*.txt` | optional | per-clip caption mirror (diffusers txt-mode) |
| `videos.txt` / `prompts.txt` | optional | line-aligned lists (alt trainers) |
| `metadata_provenance.json` | ❌ | source skit/shot/window per clip (reference only) |
| `README.md` | ❌ | dataset build docs |
