#!/bin/bash
# =============================================================================
# Pudgy Wan2.2-I2V-A14B LoRA — two-expert training (musubi-tuner)
# Implements Training_Approach_v2 §1.3 / §4: decoupled two-LoRA recipe on the
# A14B MoE — low-noise expert = identity/texture, high-noise expert = motion.
#
# Trains ONE expert per invocation (separate LoRAs, per the wan.md recommended
# min/max-timestep split + --preserve_distribution_shape). Run it twice:
#     EXPERT=low  bash train_pudgy_wan_a14b.sh      # identity/texture LoRA
#     EXPERT=high bash train_pudgy_wan_a14b.sh      # motion LoRA
#
# Prereqs (already done in stand-up): .venv-wan built, weights in
# /workspace/wan_models, latents+text-encoder outputs pre-cached.
# =============================================================================
set -euo pipefail

EXPERT="${EXPERT:-low}"                 # low | high
EPOCHS="${EPOCHS:-40}"                  # 75 clips/epoch -> ~3000 steps at 40
SAVE_EVERY="${SAVE_EVERY:-2}"           # checkpoint cadence (pick golden by eye)
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"       # in-training preview cadence (epochs); 0 = off
RESUME="${RESUME:-}"                    # path to a saved-state dir to resume exactly
LOG_WITH="${LOG_WITH:-all}"            # tracker: wandb | tensorboard | all | none
WANDB_PROJECT="${WANDB_PROJECT:-pudgy}" # wandb project (musubi: --log_tracker_name)
FP8_BASE="${FP8_BASE:-0}"              # 1 = --fp8_base (smaller/faster, slight quality cost)

REPO=/workspace/musubi-tuner
PY=/workspace/Pudgy/.venv-wan/bin/python
ACC=/workspace/Pudgy/.venv-wan/bin/accelerate
DIT_DIR=/workspace/wan_models/comfy22/split_files/diffusion_models
DATASET=/workspace/Pudgy/finetune/wan/dataset_config.toml
VAE=/workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors
T5=/workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth
SAMPLE_PROMPTS=/workspace/Pudgy/finetune/wan/sample_prompts.txt

case "$EXPERT" in
  low)
    DIT="$DIT_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    MIN_TS=0;   MAX_TS=900          # I2V low-noise range (wan.md table)
    LR="${LR:-5e-5}"               # identity: slightly lower LR
    NAME=pudgy-wan22-a14b-lownoise ;;
  high)
    DIT="$DIT_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    MIN_TS=900; MAX_TS=1000         # I2V high-noise range (wan.md table)
    LR="${LR:-1e-4}"               # motion/composition: standard LR
    NAME=pudgy-wan22-a14b-highnoise ;;
  *) echo "EXPERT must be 'low' or 'high'"; exit 1 ;;
esac

OUT=/workspace/wan_output/$NAME
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional sampling (in-process preview — no separate process, no OOM risk).
# Needs VAE+T5 loaded; set SAMPLE_EVERY=0 to disable. Sampling config lives in
# sample_prompts.txt (plain I2V from the start keyframe, 480x832 for speed).
SAMPLE_ARGS=()
if [ "$SAMPLE_EVERY" -gt 0 ] && [ -f "$SAMPLE_PROMPTS" ]; then
  SAMPLE_ARGS=(--vae "$VAE" --t5 "$T5" --sample_prompts "$SAMPLE_PROMPTS" \
               --sample_every_n_epochs "$SAMPLE_EVERY" --sample_at_first)
fi
# Optional exact resume from a saved-state dir (requires the prior run used --save_state).
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume "$RESUME")

# Optional fp8 base weights (fallback if fp16 OOMs). fp16 is default (full quality).
PREC_ARGS=()
[ "$FP8_BASE" = "1" ] && PREC_ARGS=(--fp8_base)

# Experiment tracking. wandb project = "$WANDB_PROJECT" (default: pudgy), one run per
# expert. Auth: export WANDB_API_KEY (or run `wandb login`). Set LOG_WITH=none to disable.
LOG_ARGS=()
if [ "$LOG_WITH" != "none" ]; then
  LOG_ARGS=(--log_with "$LOG_WITH" --logging_dir "$OUT/logs" \
            --log_tracker_name "$WANDB_PROJECT" --wandb_run_name "$NAME" --log_config)
  [ -n "${WANDB_API_KEY:-}" ] && LOG_ARGS+=(--wandb_api_key "$WANDB_API_KEY")
fi

echo "== Training $EXPERT-noise expert =="
echo "   dit=$DIT"
echo "   timesteps=[$MIN_TS,$MAX_TS]  lr=$LR  epochs=$EPOCHS  sample_every=$SAMPLE_EVERY  -> $OUT"
[ -n "$RESUME" ] && echo "   resuming from: $RESUME"

cd "$REPO"
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit "$DIT" \
  --dataset_config "$DATASET" \
  --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan \
  --network_dim 16 --network_alpha 32 \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep "$MIN_TS" --max_timestep "$MAX_TS" --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_epochs "$EPOCHS" --save_every_n_epochs "$SAVE_EVERY" \
  --save_state \
  --seed 42 \
  --output_dir "$OUT" --output_name "$NAME" \
  "${PREC_ARGS[@]}" "${SAMPLE_ARGS[@]}" "${RESUME_ARGS[@]}" "${LOG_ARGS[@]}"
# --- If OOM on the 14B fp16 DiT: add `--blocks_to_swap 16` (raise toward 39). ---
# --- Faster/smaller (slight quality cost): add `--fp8_base`. ---
echo "== $EXPERT-noise expert done: $OUT =="
