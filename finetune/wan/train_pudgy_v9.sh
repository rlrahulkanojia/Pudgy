#!/bin/bash
# =============================================================================
# v9c-phone - the first PROP stage: a fresh low-noise LoRA on the exactly-merged v8 base.
# Plan: training_approach/v9/Training_Approach_v9.md section 4 and RUNBOOK_v9c_phone.md.
# Everything not listed here is v8's recipe unchanged (train_pudgy_v8.sh), because v8
# showed it works:
#
#   1. Base = pudgy-v8-merged = v7fixed + v8@550 as ONE exact LoRA (lora_tools_v8.py
#      combine), merged into the frozen DiT with --base_weights. The file this run saves
#      is a pure task vector for the phone stage; inference loads base + it, both x1.0.
#      Loading v7fixed and v8@550 as two --base_weights would give the same sum, but one
#      file is what every later stage builds on (and what eval_v9 loads).
#   2. Rank 16 / alpha 16 (v9 section 4: rank 16 per prop stage; alpha/rank = 1 as v8).
#   3. <= 200 optimizer steps (v9 section 4 budget), checkpoint every 25 so the gate can
#      pick the step. Data = phone clips + 20% stratified replay (prep_v9.py).
#   4. High-noise expert frozen (v9 section 4: frozen for prop stages).
#   5. f61 is the longest bucket (v8's was f57): BLOCKS_TO_SWAP 22 instead of 20 - CONFIRM
#      with the f61-only smoke test (v9_box.sh smoke) before the real run.
#
# Steps are OPTIMIZER steps (GRAD_ACCUM clips each).
#
#   bash finetune/wan/train_pudgy_v9.sh
#   MAX_STEPS=2 SAVE_EVERY_STEPS=1 NAME=pudgy-v9c-smoke DATASET=<f61-only toml> bash finetune/wan/train_pudgy_v9.sh
# =============================================================================
set -euo pipefail

NAME="${NAME:-pudgy-v9c-phone-lownoise}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-25}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-20}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-22}"
DATASET="${DATASET:-/workspace/data_v9c/dataset_config_v9c_phone.workspace.toml}"
CACHE="${CACHE:-/workspace/wan_cache/latents_v9c}"
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
[ "$n_cache" -gt 0 ] || { echo "no text-encoder cache under $CACHE - run: bash finetune/wan/v9_box.sh cache"; exit 1; }

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

echo "== v9c: phone stage, low-noise expert  ($NAME) =="
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
