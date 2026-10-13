#!/bin/bash
# expv1 arm A (PLAN §6.1): one fresh LoRA on Kandinsky 5.0 Pro I2V (SD, 512^2), input layout A
# (G0.3), frame 0 out of the loss. Resumes automatically from the newest *-stepN-state in the
# output dir (musubi patched to continue the step count: patches/musubi-v0.3.4-resume-step.patch),
# so the supervisor service can be stopped for a gate break and simply started again.
#
#   configs/train_expv1.sh                      # normally run by the expv1-train supervisor service
#   ARM=b RANK=16 ALPHA=16 configs/train_expv1.sh   # pre-declared fallback arm B (PLAN §6.4)
set -euo pipefail

ARM="${ARM:-a}"
NAME="${NAME:-expv1-k5pro-$ARM}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-20}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANK="${RANK:-32}"
ALPHA="${ALPHA:-32}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-0}"
SEED="${SEED:-42}"

WORK=/workspace/expv1
REPO=/workspace/musubi-tuner
PY=/workspace/venv-expv1/bin
M=/workspace/models
DIT=$M/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s/model/kandinsky5pro_i2v_sft_5s.safetensors
VAE=$M/hunyuanvideo-community/HunyuanVideo          # the directory (MODEL_NOTES §2.9)
QWEN=$M/Qwen/Qwen2.5-VL-7B-Instruct
CLIP=$M/openai/clip-vit-large-patch14
DATASET="${DATASET:-/workspace/Data/processed/expv1_k5pro/dataset.toml}"
OUT=$WORK/output

for f in "$DIT" "$DATASET"; do [ -s "$f" ] || { echo "missing: $f"; exit 1; }; done
n_te=$(find $WORK/cache -name "*_te.safetensors" | wc -l)
n_lat=$(find $WORK/cache -name "*_k5.safetensors" | wc -l)
[ "$n_te" -gt 0 ] && [ "$n_te" = "$n_lat" ] || { echo "caches incomplete: $n_lat latents / $n_te text"; exit 1; }
git -C "$REPO" diff --quiet -- src/musubi_tuner/kandinsky5_train_network.py && { echo "musubi is not patched"; exit 1; }

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXTRA=()
[ "$BLOCKS_TO_SWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
RESUME="${RESUME:-$(ls -d "$OUT"/"$NAME"-step*-state 2>/dev/null | sort | tail -1)}"
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")

echo "== expv1 arm $ARM ($NAME) $(date -u +%FT%TZ) =="
echo "   base   : $(basename "$DIT")   layout A, frame 0 out of the loss"
echo "   lora   : rank $RANK / alpha $ALPHA   lr $LR cosine->x$MIN_LR_RATIO warmup $WARMUP accum $GRAD_ACCUM"
echo "   steps  : <= $MAX_STEPS, save every $SAVE_EVERY_STEPS   swap $BLOCKS_TO_SWAP   resume: ${RESUME:-none}"
echo "   data   : $DATASET ($n_lat cached clips)"

cd "$REPO"
"$PY/accelerate" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/kandinsky5_train_network.py \
  --task k5-pro-i2v-5s-sd --dit "$DIT" --vae "$VAE" \
  --text_encoder_qwen "$QWEN" --text_encoder_clip "$CLIP" \
  --dataset_config "$DATASET" --mixed_precision bf16 --sdpa --gradient_checkpointing \
  --i2v_layout A \
  --network_module networks.lora_kandinsky --network_dim "$RANK" --network_alpha "$ALPHA" \
  --optimizer_type AdamW8Bit --learning_rate "$LR" \
  --optimizer_args "weight_decay=0.001" "betas=(0.9,0.95)" --max_grad_norm 1.0 \
  --lr_scheduler cosine_with_min_lr --lr_warmup_steps "$WARMUP" --lr_scheduler_min_lr_ratio "$MIN_LR_RATIO" \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_train_steps "$MAX_STEPS" \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --save_every_n_steps "$SAVE_EVERY_STEPS" --save_state --seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" \
  --log_with tensorboard --logging_dir "$OUT/logs" --log_prefix "$NAME-" \
  "${EXTRA[@]}"
echo "== done: $OUT $(date -u +%FT%TZ) =="
