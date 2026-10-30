#!/bin/bash
# =============================================================================
# Asset-embedding training (E5, RESEARCH_asset_memory section 3) through BOTH Wan2.2 experts.
# The DiT is frozen; only the phone embedding E (the registry phrase's rows of the cached T5
# context) is trained. Low-noise expert = DiT + v8merged (musubi --base_weights); high-noise
# expert = DiT + the golden LoRA (asset_embed_v9 --base-weights-high; musubi alone would leave it
# without). Steps sample t over [0, 1000]; each step routes to the expert for its t (boundary 0.9).
#
#   SMOKE=1 bash finetune/wan/train_asset_v9.sh       # 12 steps: memory, speed, both experts used
#                                                       (P(no high-noise sample in 12) = 0.72^12 ~ 2%)
#   BASE_LOW="<v8merged> <v9c@200>" ...                # several low-noise base LoRAs, each merged x1.0
#   bash finetune/wan/train_asset_v9.sh
# Needs a span-annotated cache built from captions in the CURRENT registry wording
# (asset_embed_v9.py cache), and E's init from the same build (asset_init*.safetensors).
# =============================================================================
set -euo pipefail
NAME="${NAME:-pudgy-v9c2-asset-dual}"
MAX_STEPS="${MAX_STEPS:-150}"; SAVE_EVERY="${SAVE_EVERY:-25}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"; ASSET_LR="${ASSET_LR:-5e-3}"
MODE="${MODE:-offset}"                                   # offset (init 0, exact base at step 0) | replace
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-32}"                   # two experts keep (40 - swap) blocks each on GPU
DATASET="${DATASET:-/workspace/data_v9c2/dataset_asset_short.toml}"
CACHE="${CACHE:-/workspace/wan_cache/latents_v9c2_emb_props}"   # phone clips only: musubi trains on every latent in a cache dir
SEED="${SEED:-42}"
W=/workspace; ACC=$W/Pudgy/.venv-wan/bin/accelerate; FW=$W/Pudgy/finetune/wan
DIT_LOW=$W/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
DIT_HIGH=$W/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
BASE_LOW="${BASE_LOW:-$W/wan_output/v8_merged/pudgy-v8-merged.safetensors}"
BASE_HIGH="${BASE_HIGH:-$W/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors}"
INIT="$CACHE/asset_init$([ "$MODE" = offset ] && echo _offset).safetensors"
if [ -n "${SMOKE:-}" ]; then NAME="$NAME-smoke"; MAX_STEPS=12; SAVE_EVERY=6; GRAD_ACCUM=1; fi
OUT=$W/wan_output/$NAME
for f in "$DIT_LOW" "$DIT_HIGH" $BASE_LOW "$BASE_HIGH" "$DATASET" "$INIT"; do [ -s "$f" ] || { echo "missing: $f"; exit 1; }; done
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$W/.hf_home
echo "== asset embedding, both experts ($NAME): mode $MODE, lr $ASSET_LR, $MAX_STEPS steps x $GRAD_ACCUM, swap $BLOCKS_TO_SWAP"
cd $W/musubi-tuner
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  "$FW/asset_embed_v9.py" train --init "$INIT" --asset-lr "$ASSET_LR" --embed-only \
  $([ "$MODE" = offset ] && echo --offset) --base-weights-high "$BASE_HIGH" -- \
  --task i2v-A14B --dit "$DIT_LOW" --dit_high_noise "$DIT_HIGH" --timestep_boundary 900 \
  --base_weights $BASE_LOW \
  --dataset_config "$DATASET" --sdpa --mixed_precision fp16 \
  --network_module musubi_tuner.networks.lora_wan --network_dim 4 --network_alpha 4 \
  --timestep_sampling shift --discrete_flow_shift 5.0 --min_timestep 0 --max_timestep 1000 --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate 5e-5 --lr_scheduler cosine_with_min_lr --lr_warmup_steps 10 \
  --lr_scheduler_min_lr_ratio 0.1 --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1.0 \
  --gradient_checkpointing --blocks_to_swap "$BLOCKS_TO_SWAP" \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_steps "$MAX_STEPS" --save_every_n_steps "$SAVE_EVERY" --seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" --log_with tensorboard --logging_dir "$OUT/logs"
echo "== done: $OUT =="
