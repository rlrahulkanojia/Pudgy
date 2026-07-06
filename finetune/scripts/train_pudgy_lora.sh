#!/bin/bash
# =============================================================================
# Pudgy Penguins — CogVideoX1.5-5B-I2V LoRA training launcher
# Reads the dataset from $DATASET_DIR (75 clips, 768x1360, 16fps, 33 frames).
#
# RUN ON A CUDA GPU (A100 80GB recommended). This will NOT run on macOS/MPS.
# Usage:  bash finetune/scripts/train_pudgy_lora.sh
# =============================================================================
set -euo pipefail

# --- resolve paths ---
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
cd "$FINETUNE_DIR"
REPO_ROOT="$(cd "$FINETUNE_DIR/.." && pwd)"

# --- DATA lives SEPARATELY from this repo. ---
# Set DATASET_DIR to your training_dataset folder (the one containing train/ + metadata.json).
#   export DATASET_DIR=/path/to/training_dataset
# Default assumes it sits next to the repo folder: <repo>/../training_dataset
DATASET_DIR="${DATASET_DIR:-$(cd "$REPO_ROOT/.." && pwd)/training_dataset}"

export MODEL_PATH="${MODEL_PATH:-$FINETUNE_DIR/models/CogVideoX1.5-5B-I2V/}"
export CACHE_PATH="${CACHE_PATH:-$FINETUNE_DIR/cache/}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FINETUNE_DIR/output_dir/pudgy-lora-v1}"
export DATASET_NAME="$DATASET_DIR/"                    # train_data_dir (contains train/)
export DATASET_META_NAME="$DATASET_DIR/metadata.json"

if [ ! -f "$DATASET_META_NAME" ]; then
  echo "ERROR: dataset not found at $DATASET_DIR"
  echo "Set it explicitly:  export DATASET_DIR=/path/to/training_dataset"
  exit 1
fi

echo "MODEL_PATH   = $MODEL_PATH"
echo "DATASET_NAME = $DATASET_NAME"
echo "META         = $DATASET_META_NAME"
echo "OUTPUT       = $OUTPUT_PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation / OOM
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT_MS=1800000

accelerate launch --num_processes=1 --mixed_precision=bf16 \
  train_cogvideox_image_to_video_lora.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --cache_dir "$CACHE_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  \
  `# ---- resolution: BUCKETED. --enable_bucket is REQUIRED (the trainer's` \
  `# dataloader only exists under 'if enable_bucket'). Keep the two adaptive` \
  `# flags OFF (--random_hw_adapt, --training_with_video_token_length): those` \
  `# pick an off-grid downsample that breaks CogVideoX1.5 rotary (grid 94 vs 48).` \
  `# enable_bucket alone buckets to a clean, portrait-preserving /16 resolution` \
  `# (~576x1008 at size 768). Drop to 512 for less VRAM / faster.` \
  --image_sample_size=768 \
  --video_sample_size=768 \
  --token_sample_size=768 \
  --enable_bucket \
  \
  `# ---- REQUIRED for THIS dataset (clips are 33 consecutive frames @16fps) ----` \
  --video_sample_n_frames=33 \
  --video_sample_stride=1 \
  --fps=16 \
  --video_repeat 1 \
  \
  `# ---- LoRA hyperparameters (tuned for a small ~75-clip set) ----` \
  --rank 64 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --learning_rate 3e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  \
  `# ---- schedule / batching ----` \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps 2500 \
  --checkpointing_steps 250 \
  --checkpoints_total_limit 12 \
  --dataloader_num_workers 4 \
  \
  `# ---- misc ----` \
  --mixed_precision bf16 \
  --seed 42 \
  --allow_tf32 \
  --output_dir "$OUTPUT_PATH" \
  --nccl_timeout "$NCCL_TIMEOUT_MS"

# =============================================================================
# NOTES
# - Golden checkpoint: evaluate each checkpoint-*/ (every 250 steps); character
#   fidelity usually peaks around 1000-2000 steps before overfitting. Pick the
#   best, don't assume the last is best.
# - Resolution: --enable_bucket is REQUIRED and buckets to a clean portrait
#   /16 size. Do NOT add --random_hw_adapt or --training_with_video_token_length
#   (they trigger the "grid 94 vs 48" rotary crash). Lower to 512 for less VRAM:
#   set --image/video/token_sample_size=512.
# - VRAM tight? add: --use_8bit_adam  (needs `pip install bitsandbytes`).
# - Multi-GPU / DeepSpeed: use finetune/scripts/train_cogvideox_i2v_lora_single_rank.sh
#   as a reference and the zero_stage2_config.json in this folder.
# - If you regenerate the dataset at 49 frames, change --video_sample_n_frames=49.
# =============================================================================
