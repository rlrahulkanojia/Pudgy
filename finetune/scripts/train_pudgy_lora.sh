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
  `# ---- resolution: BUCKETED at the MAXIMUM the model supports for THIS data. ----` \
  `# The clips are PORTRAIT 768x1360 (ratio 1.77). CogVideoX1.5-5B-I2V's rotary grid is` \
  `# capped at 48 x 85 patches (== sample_height/2=96/2 x sample_width/2=170/2), i.e. a` \
  `# native canvas of 768 (H) x 1360 (W). So a portrait clip can be at most 768px TALL.` \
  `# The bucket for ratio 1.77 is [672,384] (H,W) scaled by sample_size/512, floored /16.` \
  `# sample_size=592 -> 768x432 (H,W), grid 48x27  <- the max on-grid portrait resolution.` \
  `# DO NOT raise this: 768 -> 1008x576 (grid 63) and 1024 -> 1344x768 (grid 84) both blow` \
  `# past the 48-row grid and crash with "Expected size 63/84 but got 48". Drop to 512` \
  `# (-> 672x384, grid 42x24) for less VRAM. Keep --enable_bucket ON and the two adaptive` \
  `# flags (--random_hw_adapt, --training_with_video_token_length) OFF (they go off-grid).` \
  --image_sample_size=592 \
  --video_sample_size=592 \
  --token_sample_size=592 \
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
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
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
  --nccl_timeout "$NCCL_TIMEOUT_MS" \
  \
  `# ---- training log / loss curve (TensorBoard) ----` \
  `# Without --report_to the trainer's accelerator.log(loss, lr) calls are silent no-ops,` \
  `# so NO loss curve is captured (the v1 run had to be reverse-engineered from optimizer` \
  `# state). This writes tfevents to $OUTPUT_PATH/logs. Needs \`pip install tensorboard\`.` \
  `# Watch live:  tensorboard --logdir "$OUTPUT_PATH/logs" --host 0.0.0.0 --port 6006` \
  --report_to tensorboard \
  --logging_dir logs \
  --tracker_name pudgy-lora-v1

# =============================================================================
# NOTES
# - Golden checkpoint: evaluate each checkpoint-*/ (every 250 steps); character
#   fidelity usually peaks around 1000-2000 steps before overfitting. Pick the
#   best, don't assume the last is best.
# - Resolution: 592 is the MAXIMUM on-grid size for these portrait clips -> 768x432
#   (H,W), grid 48x27. The model caps portrait height at 768px (rotary grid 48 rows).
#   Do NOT raise above 597 (768 or 1024 crash: "Expected size 63/84 but got 48"), and
#   do NOT add --random_hw_adapt / --training_with_video_token_length (same crash).
#   Lower to 512 (-> 672x384) for less VRAM: set --image/video/token_sample_size=512.
# - VRAM tight? add: --use_8bit_adam  (needs `pip install bitsandbytes`).
# - Multi-GPU / DeepSpeed: use finetune/scripts/train_cogvideox_i2v_lora_single_rank.sh
#   as a reference and the zero_stage2_config.json in this folder.
# - If you regenerate the dataset at 49 frames, change --video_sample_n_frames=49.
# =============================================================================
