#!/bin/bash
# =============================================================================
# v5 pilot — Pax "happy" expression LoRA, continue-trained from the v2 golden.
# Adapted from train_pudgy_wan_a14b.sh; adds NETWORK_WEIGHTS / RANK / ALPHA and
# points at the happy-only dataset config.
#
#   EXPERT=high bash train_pudgy_happy_expr.sh    # plan-of-record (motion spec.)
#   EXPERT=low  bash train_pudgy_happy_expr.sh    # ablation (fine facial detail)
#
# RANK/ALPHA must match the checkpoint named by NETWORK_WEIGHTS — the v2 golden
# LoRAs are rank 16 / alpha 32 (400 modules), so rank 8 cannot load them.
# =============================================================================
set -euo pipefail

EXPERT="${EXPERT:-high}"
EPOCHS="${EPOCHS:-18}"
SAVE_EVERY="${SAVE_EVERY:-1}"        # tiny dataset -> fine-grained golden pick
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"    # 0 = off. In-process sampling loads T5+VAE
                                     # on top of a ~60GB DiT and can OOM-kill the
                                     # run (finetune/wan/README.md). Eval after.
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
LOG_WITH="${LOG_WITH:-tensorboard}"  # no WANDB_API_KEY on this box
WANDB_PROJECT="${WANDB_PROJECT:-pudgy}"
FP8_BASE="${FP8_BASE:-0}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-0}"

REPO=/workspace/musubi-tuner
ACC=/workspace/Pudgy/.venv-wan/bin/accelerate
DIT_DIR=/workspace/wan_models/comfy22/split_files/diffusion_models
GOLDEN=/workspace/wan_output/v2_golden
DATASET="${DATASET:-/workspace/Pudgy/finetune/wan/dataset_config_happy.toml}"

case "$EXPERT" in
  low)
    DIT="$DIT_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    MIN_TS=0;   MAX_TS=900
    LR="${LR:-3e-5}"
    NETWORK_WEIGHTS="${NETWORK_WEIGHTS:-$GOLDEN/lora_lownoise_GOLDEN_ep40.safetensors}"
    NAME=pudgy-happy-expr-lownoise-v1 ;;
  high)
    DIT="$DIT_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    MIN_TS=900; MAX_TS=1000
    LR="${LR:-3e-5}"                 # gentle: 1e-4 (v2) would wreck the prior on 28 clips
    NETWORK_WEIGHTS="${NETWORK_WEIGHTS:-$GOLDEN/lora_highnoise_GOLDEN_ep40.safetensors}"
    NAME=pudgy-happy-expr-highnoise-v1 ;;
  *) echo "EXPERT must be 'low' or 'high'"; exit 1 ;;
esac

OUT=/workspace/wan_output/$NAME
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA=()
[ "$FP8_BASE" = "1" ] && EXTRA+=(--fp8_base)
[ "$BLOCKS_TO_SWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
[ -n "$NETWORK_WEIGHTS" ] && EXTRA+=(--network_weights "$NETWORK_WEIGHTS")
if [ "$LOG_WITH" != "none" ]; then
  EXTRA+=(--log_with "$LOG_WITH" --logging_dir "$OUT/logs" --log_config)
  [ "$LOG_WITH" != "tensorboard" ] && EXTRA+=(--log_tracker_name "$WANDB_PROJECT" --wandb_run_name "$NAME")
fi

echo "== v5 happy-expression pilot: $EXPERT-noise expert =="
echo "   dit    : $(basename "$DIT")"
echo "   init   : $(basename "$NETWORK_WEIGHTS")  (rank $RANK / alpha $ALPHA)"
echo "   ts     : [$MIN_TS,$MAX_TS]   lr=$LR   epochs=$EPOCHS   -> $OUT"

cd "$REPO"
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit "$DIT" \
  --dataset_config "$DATASET" \
  --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan \
  --network_dim "$RANK" --network_alpha "$ALPHA" \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep "$MIN_TS" --max_timestep "$MAX_TS" --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_epochs "$EPOCHS" --save_every_n_epochs "$SAVE_EVERY" \
  --save_state \
  --seed 42 \
  --output_dir "$OUT" --output_name "$NAME" \
  "${EXTRA[@]}"
echo "== done: $OUT =="
