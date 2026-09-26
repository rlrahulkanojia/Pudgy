#!/bin/bash
# =============================================================================
# v8 — ONE joint (motion + expression) LoRA on the Wan2.2 LOW-noise expert.
# Plan: training_approach/v8/Training_Approach_v8.md. Each change from v7 answers a
# measured v7 finding (docs/training_reports/v7/ANALYSIS_v7.md):
#
#   1. --base_weights, not --network_weights. The starting point is MERGED into the frozen
#      DiT and a FRESH LoRA is trained on top, so the file this run saves is a pure task
#      vector tau. Default start (2026-09-26 decision): **v7, correctly combined** -
#      pudgy-v7-fixed-combined.safetensors = v7 expression + v7 motion - v2 golden
#      = G + tau_e + tau_m, built exactly as one rank-48 LoRA (lora_tools_v8.py combine;
#      rel. error 2.5e-8). Stacking the two v7 files as-is would count G twice (a 76%
#      error, ANALYSIS_v7 section 1). BASE_WEIGHTS=<v2 golden> restores the golden-only start.
#   2. One joint LoRA. v7's motion and expression LoRAs shared a common direction (cos
#      0.33 overall, 0.54-0.58 on self-attn q/k) and 0.67 input-subspace overlap; one
#      LoRA learns that shared component once instead of twice.
#   3. Gradient accumulation 8 + cosine decay to 10%. v7's batch-1 constant-LR updates
#      were noise-dominated: ||tau(t)|| grew as t^0.55, consecutive 500-step updates had
#      cosine 0.10-0.24, and the systematic drift was only 5% of the squared displacement
#      at step 500 (19% at 2500). Averaging 8 samples per update and annealing the LR
#      turns more of each step into drift. LR 5e-5 at alpha/rank = 1 keeps the per-step
#      update scale close to v7's 3e-5 at alpha/rank = 2.
#   4. Rank 32 / alpha 32. v7's updates had effective rank ~9 of 32 PER KIND; a joint
#      LoRA carries both kinds, so rank 16 would be the first binding constraint.
#
# Steps are OPTIMIZER steps (8 clips each). 2,096 clips -> 262 steps/epoch.
#
#   bash finetune/wan/train_pudgy_v8.sh                     # the run
#   MAX_STEPS=4 SAVE_EVERY_STEPS=2 NAME=pudgy-v8-smoke bash finetune/wan/train_pudgy_v8.sh
# =============================================================================
set -euo pipefail

NAME="${NAME:-pudgy-v8-joint-lownoise}"
MAX_STEPS="${MAX_STEPS:-600}"          # ~2.3 epochs; a CEILING - stop where the gates plateau
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-20}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANK="${RANK:-32}"
ALPHA="${ALPHA:-32}"
# v7 defect 1: f57 OOMs at 0. Measured on this box (A100 80GB, rank 32, f57): swap 16
# peaks at 81.2 GB with 31 allocator retry warnings and 77 s/clip (thrashing at the
# limit); swap 20 has 0 retries and runs 58.5 s/clip - more swap is FASTER here.
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-20}"
DATASET="${DATASET:-/workspace/data_v8/dataset_config_joint_v8.workspace.toml}"
SEED="${SEED:-42}"

REPO=/workspace/musubi-tuner
ACC=/workspace/Pudgy/.venv-wan/bin/accelerate
DIT=/workspace/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
BASE_WEIGHTS="${BASE_WEIGHTS:-/workspace/wan_output/v7_baseline/pudgy-v7-fixed-combined.safetensors}"
OUT=/workspace/wan_output/$NAME

# Credentials (Azure for the mirror, optional WANDB_API_KEY) live in the repo's .env.
set -a; [ -f /workspace/Pudgy/.env ] && . /workspace/Pudgy/.env; set +a

# Fail before the 28.6 GB DiT load, not after it (v7 defect 2 died 2 min in).
for f in "$DIT" "$BASE_WEIGHTS" "$DATASET"; do
  [ -s "$f" ] || { echo "missing: $f"; exit 1; }
done
n_cache=$(find /workspace/wan_cache/latents_v8/joint -name "*_te.safetensors" 2>/dev/null | wc -l)
[ "$n_cache" -gt 0 ] || { echo "no text-encoder cache - run /workspace/cache_v8.sh first"; exit 1; }

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# musubi re-fetches the T5 tokenizer on every start; one process per job earned a 429 in v7.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LOG=(--log_with tensorboard --logging_dir "$OUT/logs" --log_config --log_prefix "$NAME-")
if [ -n "${WANDB_API_KEY:-}" ]; then
  LOG=(--log_with all --logging_dir "$OUT/logs" --log_config
       --log_tracker_name "${WANDB_PROJECT:-pudgy}" --wandb_run_name "$NAME")
fi

EXTRA=()
[ "$BLOCKS_TO_SWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
[ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")

echo "== v8: joint LoRA, low-noise expert  ($NAME) =="
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
