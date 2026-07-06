# Training the Pudgy LoRA — GPU setup & run guide

The trainer is **CogVideoX1.5-5B-I2V LoRA** (Passenger12138 fork of the diffusers CogVideoX finetune). This folder has the repo code plus three helper files:

| File | What it does |
|---|---|
| `setup_gpu_env.sh` | one-time environment install (diffusers-from-source, deps, model download) |
| `check_dataset.py` | pre-flight validation of `../training_dataset` |
| `finetune/scripts/train_pudgy_lora.sh` | the launch script, wired to this dataset with the correct overrides |

## ⚠️ You cannot train on the Mac
CogVideoX LoRA training needs an **NVIDIA CUDA GPU** — an **A100 80GB** is the reference (also fits on H100; ~40–48 GB min with 8-bit Adam). Apple-silicon/MPS is not supported by this trainer. Develop/inspect on the Mac, but run training on the GPU box.

## Expected folder layout on the GPU machine
Keep `trainer_code/` and `training_dataset/` under the same parent (as they are now) so the launch script finds the data automatically:
```
Pudgy/Data/
├── training_dataset/     # the 75 clips + metadata.json
└── trainer_code/         # this folder
    ├── setup_gpu_env.sh
    ├── check_dataset.py
    └── finetune/scripts/train_pudgy_lora.sh
```
Copy the whole `Data/` folder to the GPU machine (or at least these two subfolders side by side). If you put them elsewhere, override `DATASET_NAME` / `DATASET_META_NAME` when launching.

## Steps

```bash
# 0. copy Data/ to the GPU box, then:
cd trainer_code

# 1. install everything (~20 GB model download; takes a while)
bash setup_gpu_env.sh
source .venv/bin/activate            # if the script created a venv

# 2. sanity-check the dataset (fast)
python check_dataset.py
#    -> should print "Dataset looks good" and confirm 33-frame / 768x1360 clips

# 3. train
bash finetune/scripts/train_pudgy_lora.sh
```

Checkpoints land in `finetune/output_dir/pudgy-lora-v1/checkpoint-*/` every 250 steps.

## Key settings already baked into the launch script
- `--video_sample_n_frames=33 --video_sample_stride=1 --fps=16` — **mandatory** for this dataset (the repo defaults of 49/stride-3 would sample only 11 non-8N+1 frames and break).
- `--rank 64 --lora_alpha 32 --learning_rate 3e-5 --lr_scheduler cosine` — tuned for a small (~75-clip) set; the repo defaults (rank 128, LR 1e-4) overfit/burn on this size.
- `--max_train_steps 2500 --checkpointing_steps 250` — gives ~10 checkpoints to pick the golden one from (character fidelity usually peaks ~1000–2000 steps).
- Low-VRAM resolution preset (`video_sample_size=512` + token-length + bucket). To train at native **768×1360**, see the NOTES block at the bottom of the launch script.

## After training
Pick the best checkpoint by eye (don't assume the last is best), then use the repo's `inference/` scripts (or ComfyUI) to run I2V generation with the LoRA. The trained weight is a `.safetensors` in the chosen `checkpoint-*/` folder.

## Common gotchas
- **`diffusers` release installed instead of source** → CogVideoX1.5 I2V class not found. Re-run `pip install -e ./diffusers`.
- **OOM** → add `--use_8bit_adam` (needs `bitsandbytes`), keep resolution at 512, or lower `--gradient_accumulation_steps`.
- **`patch_size_t` / shape error at 33 frames** → your diffusers build has a stricter check; regenerate the dataset at 49 frames (`FRAMES=49` in `split6.py`) and set `--video_sample_n_frames=49`.
- **decord import error** → `pip install decord`.
