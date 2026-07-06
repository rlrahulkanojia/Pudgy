# Training the Pudgy LoRA — GPU setup & run guide

Trainer: **CogVideoX1.5-5B-I2V LoRA** (Passenger12138 fork of the diffusers CogVideoX finetune). This repo adds three helpers:

| File | What it does |
|---|---|
| `setup_gpu_env.sh` | one-time env install (system ffmpeg, CUDA torch, diffusers-from-source, deps, model) |
| `check_dataset.py` | pre-flight validation of the dataset |
| `finetune/scripts/train_pudgy_lora.sh` | launch script, wired to this dataset with the correct overrides |

## ⚠️ GPU required — not the Mac
Needs an **NVIDIA CUDA GPU**. Verified on an **RTX 5090 (32 GB, Blackwell/sm_120)**; also runs on A100/H100 80 GB (more headroom). Apple-silicon/MPS is not supported. On 32 GB cards you'll likely need the OOM tweaks below.

## Folder layout on the GPU box
Keep the repo folder and `training_dataset/` **side by side** so the launch script auto-finds the data:
```
<parent>/
├── <repo>/               # this repo (cloned; e.g. Pudgy/)
│   ├── setup_gpu_env.sh
│   ├── check_dataset.py
│   └── finetune/scripts/train_pudgy_lora.sh
└── training_dataset/     # the 75 clips + metadata.json (transferred separately)
```
If they're not side by side, set `export DATASET_DIR=/path/to/training_dataset` before launching.

## Steps

```bash
cd <repo>                       # e.g. /workspace/Pudgy

# 1. install everything (installs ffmpeg, cu128 torch, diffusers-from-source,
#    all deps, and downloads the ~20 GB base model). Needs ~25 GB free.
bash setup_gpu_env.sh
source .venv/bin/activate

# 2. verify GPU + validate dataset
python -c "import torch; print(torch.cuda.get_device_name(0))"   # no sm_120 warning = good
export DATASET_DIR=/path/to/training_dataset
python check_dataset.py "$DATASET_DIR"      # -> "Dataset looks good — ready to train."

# 3. train (use tmux so it survives disconnects)
tmux new -s train
DATASET_DIR=/path/to/training_dataset bash finetune/scripts/train_pudgy_lora.sh
#   detach: Ctrl-b then d   |   reattach: tmux attach -t train
```
Checkpoints → `finetune/output_dir/pudgy-lora-v1/checkpoint-*/` every 250 steps.

## Key settings baked into the launch script
- `--video_sample_n_frames=33 --video_sample_stride=1 --fps=16` — **mandatory** for this dataset (repo defaults of 49/stride-3 sample only 11 non-8N+1 frames and break).
- `--rank 64 --lora_alpha 32 --learning_rate 3e-5 --lr_scheduler cosine` — tuned for a small (~75-clip) set; repo defaults (rank 128, LR 1e-4) overfit/burn.
- `--max_train_steps 2500 --checkpointing_steps 250` — ~10 checkpoints; character fidelity usually peaks ~1000–2000 steps.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set by the script to reduce fragmentation.
- **Fixed-square resolution** (`image/video/token_sample_size=768`, NO `enable_bucket`/`random_hw_adapt`/`training_with_video_token_length`). The adaptive path buckets to an off-grid size that breaks CogVideoX1.5's rotary embedding (see gotchas). Square center-crops 9:16 clips to their center; drop to 512 for less VRAM/faster.

## Out of memory (common on 32 GB cards like the 5090)
The 5B model + rank-64 LoRA + AdamW optimizer states is heavy. In rough order of impact, add to the launch script:
1. `--use_8bit_adam` — biggest win (halves optimizer memory; needs `bitsandbytes`).
2. `--enable_slicing --enable_tiling` — cuts VAE peak memory.
3. Lower resolution: `--video_sample_size=384 --token_sample_size=384`.
4. (Already on: `--gradient_checkpointing`, `expandable_segments`.)

On an 80 GB A100/H100 the stock preset fits without these.

## After training
Pick the best checkpoint by eye (don't assume the last is best). The LoRA weight is a `.safetensors` in the chosen `checkpoint-*/`. Run I2V generation with it via the repo's `inference/` scripts or ComfyUI (install the inference extras listed at the bottom of `requirements.txt`).

## Gotchas we actually hit (all now handled by the scripts)
- **RTX 50-series / Blackwell** → needs CUDA 12.8 torch (`cu128`); the `cu121` build errors with `sm_120 not compatible`. Setup uses cu128 by default.
- **`ModuleNotFoundError` (cv2 / func_timeout / peft / …)** → all folded into `requirements.txt` now.
- **`huggingface-cli` deprecated** → use `hf download` (setup uses it).
- **`ffprobe: not found`** in check_dataset → install system `ffmpeg` (setup does this).
- **`diffusers` release instead of source** → CogVideoX1.5 I2V class missing; re-run `pip install -e ./diffusers`.
- **Rotary embedding error** — `RuntimeError: Sizes of tensors must match ... Expected size 94 but got size 48` in `_prepare_rotary_positional_embeddings`. Caused by the **adaptive resolution** path (`enable_bucket`/`random_hw_adapt`/`training_with_video_token_length`) picking an off-grid size. Fix = fixed-square resolution (already the default here). Not a frame-count problem.
- **`ValueError: invalid literal for int(): '16/1'`** in `check_dataset.py` → ffprobe field-order difference across builds; fixed in the current script (parses by field name).
