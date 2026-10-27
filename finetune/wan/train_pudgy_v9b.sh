#!/bin/bash
# =============================================================================
# v9b-1 - long-form scenes: a fresh low-noise LoRA on the exactly-merged v8 base.
# Plan: training_approach/v9/Training_Approach_v9.md sections 0.1 and 4; runbook:
# training_approach/v9/RUNBOOK_v9b.md. Everything not listed here is the v9c/v8 recipe
# unchanged (train_pudgy_v9.sh), because v8 showed it works:
#
#   1. Base = pudgy-v8-merged (v7fixed + v8@550, one exact LoRA) merged into the frozen DiT
#      with --base_weights (D1: not the v9c-phone stage). The saved file is a pure task
#      vector for v9b; inference loads base + it, both x1.0.
#   2. Rank 32 / alpha 32 (v9 section 4: rank 32 for the scene stage; alpha/rank = 1 as v8).
#   3. 400 optimizer steps (not the plan's 800), checkpoint every 50. The curated set is
#      ~480 clips (~310 scene windows + ~170 replay), so 400 x 8 = 3,200 samples is ~6.7
#      epochs; 800 would be ~13, and v9c-phone plateaued at ~2 epochs. Tier 1 at 100-400
#      picks the checkpoint (RUNBOOK_v9b section 6).
#   4. Data = curated skit windows at 480x848 (f21/f33/f49/f81) + 35% v8 replay at 1024^2
#      with label anchors (prep_v9b.py, D6/D10). Plain MSE loss (D8: the motion-weighted
#      loss is not used in v9b-1).
#   5. High-noise expert frozen (D7: a separate high-noise run only if E1a / the travel bar
#      call for it).
#   6. The heaviest sample is a replay f57 at 1024^2 (15 latent frames x 4096 tokens), not
#      a scene f81 (21 x 1590): BLOCKS_TO_SWAP 24 as v9c measured - CONFIRM with
#      `v9b_box.sh smoke` (it runs the heaviest bucket).
#
# Steps are OPTIMIZER steps (GRAD_ACCUM clips each).
#
#   bash finetune/wan/train_pudgy_v9b.sh
# =============================================================================
set -euo pipefail

NAME="${NAME:-pudgy-v9b-lownoise}"
MAX_STEPS="${MAX_STEPS:-400}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-20}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANK="${RANK:-32}"
ALPHA="${ALPHA:-32}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-24}"
DATASET="${DATASET:-/workspace/data_v9b/dataset_config_v9b.workspace.toml}"
CACHE="${CACHE:-/workspace/wan_cache/latents_v9b}"
SEED="${SEED:-42}"

REPO=/workspace/musubi-tuner
ACC=/workspace/Pudgy/.venv-wan/bin/accelerate
DIT=/workspace/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
BASE_WEIGHTS="${BASE_WEIGHTS:-/workspace/wan_output/v8_merged/pudgy-v8-merged.safetensors}"
OUT=/workspace/wan_output/$NAME

set -a; [ -f /workspace/Pudgy/.env ] && . /workspace/Pudgy/.env; set +a

for f in "$DIT" "$BASE_WEIGHTS" "$DATASET"; do
  [ -s "$f" ] || { echo "missing: $f"; exit 1; }
done
n_cache=$(find "$CACHE" -name "*_te.safetensors" 2>/dev/null | wc -l)
[ "$n_cache" -gt 0 ] || { echo "no text-encoder cache under $CACHE - run: bash finetune/wan/v9b_box.sh cache"; exit 1; }

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LOG=(--log_with tensorboard --logging_dir "$OUT/logs" --log_config --log_prefix "$NAME-")
if [ -n "${WANDB_API_KEY:-}" ]; then
  LOG=(--log_with all --logging_dir "$OUT/logs" --log_config
       --log_tracker_name "${WANDB_PROJECT:-pudgy}" --wandb_run_name "$NAME")
fi

EXTRA=()
[ "$BLOCKS_TO_SWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
[ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")

echo "== v9b-1: long-form scenes, low-noise expert  ($NAME) =="
echo "   base   : $(basename "$DIT") + MERGED $(basename "$BASE_WEIGHTS")"
echo "   lora   : fresh, rank $RANK / alpha $ALPHA"
echo "   optim  : adamw8bit lr=$LR cosine->x$MIN_LR_RATIO warmup=$WARMUP  accum=$GRAD_ACCUM"
echo "   steps  : <= $MAX_STEPS optimizer steps, save every $SAVE_EVERY_STEPS"
echo "   swap   : $BLOCKS_TO_SWAP blocks   log: ${LOG[1]}"
echo "   data   : $DATASET"
echo "   out    : $OUT"

cd "$REPO"
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit "$DIT" \
  --base_weights "$BASE_WEIGHTS" --base_weights_multiplier 1.0 \
  --dataset_config "$DATASET" \
  --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan \
  --network_dim "$RANK" --network_alpha "$ALPHA" \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep 0 --max_timestep 900 --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --lr_scheduler cosine_with_min_lr --lr_warmup_steps "$WARMUP" \
  --lr_scheduler_min_lr_ratio "$MIN_LR_RATIO" \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1.0 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_steps "$MAX_STEPS" --save_every_n_steps "$SAVE_EVERY_STEPS" \
  --save_state \
  --seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" \
  "${LOG[@]}" "${EXTRA[@]}"
echo "== done: $OUT =="
