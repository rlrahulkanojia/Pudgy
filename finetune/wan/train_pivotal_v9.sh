#!/bin/bash
# =============================================================================
# v9c2 pivotal stage 2 (RESEARCH_v9c_fixes section 6, step 3): a rank-16 LoRA on the cross-attention
# k/v projections only (80 modules), trained through BOTH Wan2.2 experts (musubi two-expert mode; one
# LoRA shared by both, as musubi trains it), with the E5 phone embedding spliced in FROZEN, and a
# preservation loss on replay samples (target = the base's own prediction, weight PRESERVE_W).
# Data: phone + replay ~36.5% + sitting anchors (build_pivotal_set_v9.py).
#
#   E_FILE=<E5 .asset.safetensors> SMOKE=1 bash finetune/wan/train_pivotal_v9.sh   # 16 steps
#   E_FILE=<E5 .asset.safetensors> bash finetune/wan/train_pivotal_v9.sh
# Load at inference: low  = v8merged + <lora>, high = golden + <lora>, E spliced (gen_asset_v9.py).
# =============================================================================
set -euo pipefail
NAME="${NAME:-pudgy-v9c2-pivotal}"
MAX_STEPS="${MAX_STEPS:-150}"; SAVE_EVERY="${SAVE_EVERY:-25}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"; RANK="${RANK:-16}"; PRESERVE_W="${PRESERVE_W:-1.0}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-32}"
DATASET="${DATASET:-/workspace/data_v9c2/dataset_pivotal.toml}"
E_FILE="${E_FILE:?set E_FILE to the frozen phone embedding (.asset.safetensors, offset mode)}"
SEED="${SEED:-42}"
W=/workspace; ACC=$W/Pudgy/.venv-wan/bin/accelerate; FW=$W/Pudgy/finetune/wan
DIT_LOW=$W/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
DIT_HIGH=$W/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
BASE_LOW="${BASE_LOW:-$W/wan_output/v8_merged/pudgy-v8-merged.safetensors}"
BASE_HIGH="${BASE_HIGH:-$W/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors}"
if [ -n "${SMOKE:-}" ]; then NAME="$NAME-smoke"; MAX_STEPS=16; SAVE_EVERY=8; GRAD_ACCUM=1; fi
OUT=$W/wan_output/$NAME
for f in "$DIT_LOW" "$DIT_HIGH" $BASE_LOW "$BASE_HIGH" "$DATASET" "$E_FILE"; do [ -s "$f" ] || { echo "missing: $f"; exit 1; }; done
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$W/.hf_home
echo "== pivotal LoRA ($NAME): rank $RANK cross-attn k/v, lr $LR, preserve x$PRESERVE_W, $MAX_STEPS steps x $GRAD_ACCUM, E frozen: $E_FILE"
cd $W/musubi-tuner
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$FW/asset_embed_v9.py" train --init "$E_FILE" --offset --freeze-asset --preserve --preserve-weight "$PRESERVE_W" \
  --base-weights-high "$BASE_HIGH" -- \
  --task i2v-A14B --dit "$DIT_LOW" --dit_high_noise "$DIT_HIGH" --timestep_boundary 900 \
  --base_weights $BASE_LOW \
  --dataset_config "$DATASET" --sdpa --mixed_precision fp16 \
  --network_module musubi_tuner.networks.lora_wan --network_dim "$RANK" --network_alpha "$RANK" \
  --network_args "exclude_patterns=['.*']" "include_patterns=[r'.*cross_attn\.(k|v)']" \
  --timestep_sampling shift --discrete_flow_shift 5.0 --min_timestep 0 --max_timestep 1000 --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate "$LR" --lr_scheduler cosine_with_min_lr --lr_warmup_steps 10 \
  --lr_scheduler_min_lr_ratio 0.1 --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1.0 \
  --gradient_checkpointing --blocks_to_swap "$BLOCKS_TO_SWAP" \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_steps "$MAX_STEPS" --save_every_n_steps "$SAVE_EVERY" --seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" --log_with tensorboard --logging_dir "$OUT/logs"
echo "== done: $OUT =="
