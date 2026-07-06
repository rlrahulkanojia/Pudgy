# Pudgy Penguins — CogVideoX1.5-5B-I2V Character LoRA

Training setup for a Pax & Polly (Pudgy Penguins) style/character LoRA on
**CogVideoX1.5-5B-I2V**, using the Passenger12138 diffusers fine-tune.

> **Data lives separately.** This repo is code + scripts + docs only. The
> training clips (`training_dataset/`) are **not** committed here — see
> [Getting the data](#getting-the-data).

## Repo contents

```
.
├── README.md                     ← you are here
├── SETUP_GPU.md                  ← full GPU setup + run guide
├── setup_gpu_env.sh              ← one-time env install (diffusers-from-source, deps, model)
├── check_dataset.py              ← pre-flight dataset validator
├── finetune/                     ← trainer code (CogVideoX1.5-5B-I2V LoRA)
│   └── scripts/train_pudgy_lora.sh   ← the launch script (wired for this dataset)
├── inference/  scripts/  tools/  ← upstream inference / helper code
├── requirements.txt
├── UPSTREAM_README.md            ← original trainer README (reference)
└── docs/
    ├── DATASET_BUILD.md              ← how the dataset was built + its spec
    ├── TRAINING_CONFIG.md            ← exact trainer settings for this data
    ├── Data_Readiness_Gap_Analysis.md
    └── Data_Discrepancies.md
```

## Requirements

- **NVIDIA CUDA GPU** — A100 80GB reference (H100 fine; ~40–48 GB min with 8-bit Adam). **Not runnable on macOS/MPS.**
- diffusers installed **from source** (the 1.5 I2V code isn't in any pip release) — handled by `setup_gpu_env.sh`.

## Quickstart (on the GPU box)

```bash
# 1. install environment (~20 GB base-model download)
bash setup_gpu_env.sh
source .venv/bin/activate

# 2. point at the data (see below) and validate
export DATASET_DIR=/path/to/training_dataset
python check_dataset.py "$DATASET_DIR"

# 3. train
DATASET_DIR=/path/to/training_dataset bash finetune/scripts/train_pudgy_lora.sh
```

Checkpoints → `finetune/output_dir/pudgy-lora-v1/checkpoint-*/` every 250 steps. See `SETUP_GPU.md` for details, VRAM options, and gotchas.

## Getting the data

The dataset (`training_dataset/`: 75 clips + `metadata.json`) is kept out of git (large binary video). Transfer it to the GPU box separately — e.g. `scp -r training_dataset user@gpu-box:/data/`, a cloud bucket, or an out-of-band drive. Then set `DATASET_DIR` to wherever it lands.

Dataset spec (all clips): **768×1360, 16 fps, 33 frames (8N+1), H.264, silent.** Full build details and provenance in `docs/DATASET_BUILD.md`.

## Key config baked into the launch script

- `--video_sample_n_frames=33 --video_sample_stride=1 --fps=16` — **mandatory** for this dataset (repo defaults of 49/stride-3 would sample 11 non-8N+1 frames and break).
- `--rank 64 --lora_alpha 32 --learning_rate 3e-5 --lr_scheduler cosine` — tuned for a small (~75-clip) set.
- `--max_train_steps 2500 --checkpointing_steps 250` — ~10 checkpoints; pick the golden one (fidelity usually peaks ~1000–2000 steps).

## Credits

Trainer: [Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train) · Base model: `THUDM/CogVideoX1.5-5B-I2V`.
