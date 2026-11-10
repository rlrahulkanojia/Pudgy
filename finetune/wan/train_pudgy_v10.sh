#!/bin/bash
# =============================================================================
# v10 Stage 1 - a fresh LoRA on the HIGH-noise expert only (Training_Approach_v10.md section 0
# Q2/Q10/Q16/Q17 and section 4). The low-noise expert (pudgy-v8-merged) is not loaded or trained.
#
#   1. musubi single-expert mode: --dit = Wan2.2 I2V HIGH-noise DiT, t in [900, 1000], no
#      --dit_high_noise. --base_weights lora_highnoise_GOLDEN_ep40 x1: musubi merges it into the
#      DiT's Linear weights before the new LoRA is applied, so the saved file is a pure task vector
#      on top of golden. Inference: high = golden + this LoRA, low = pudgy-v8-merged.
#   2. Rank 16 / alpha 16, all linear layers of the attention blocks (musubi's lora_wan default,
#      as golden). lr 3e-5, warmup 20, cosine -> x0.1, grad-accum 8, <= 300 steps, save every 25.
#   3. Preservation lambda = 1 on REPLAY samples only: target = the golden-merged expert's own
#      prediction with the LoRA at multiplier 0 (asset_embed_v9.py train --preserve --preserve-marker
#      key; block-swap-safe base pass). Replay = `varlen_preserve` = 1 in the sample's _te cache,
#      written by cache_mark_v10.py into the replay_f<n> cache dirs. Scene windows get plain MSE.
#   4. E5 (the frozen phone embedding, offset mode) is added to the phone phrase's T5 rows on every
#      window that names the phone, as at inference (spans written by cache_mark_v10.py).
#   5. Refuses to start unless `cache_mark_v10.py check` passes on the dataset (every _te file
#      marked, replay samples present, captions unchanged since marking).
#
#   bash finetune/wan/train_pudgy_v10.sh                 # the run (auto-resumes from the latest state)
#   SMOKE=1 bash finetune/wan/train_pudgy_v10.sh         # 3 steps x 1 sample: heaviest replay sample
#                                                        # (preservation), a phone window, a plain window
#   DRY_RUN=1 bash finetune/wan/train_pudgy_v10.sh       # checks + print the command, no training
#
# Env (defaults below): NAME MAX_STEPS SAVE_EVERY_STEPS GRAD_ACCUM LR WARMUP MIN_LR_RATIO RANK ALPHA
# BLOCKS_TO_SWAP DATASET CACHE SEED RESUME(auto|none|<state dir>) PRESERVE_WEIGHT LOG_EVERY
# E5 GOLDEN DIT CUDA_VISIBLE_DEVICES(0). Steps are OPTIMIZER steps (GRAD_ACCUM samples each).
# Pause/resume: the trainer's processes match
#   pgrep -f "asset_embed_v9\.py train .*--output_name pudgy-v10-highnoise( |$)"
# and a re-run of this script resumes from the newest complete <NAME>-stepNNNNNNNN-state.
# =============================================================================
set -euo pipefail

W="${W:-/workspace}"
NAME="${NAME:-pudgy-v10-highnoise}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-25}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-3e-5}"
WARMUP="${WARMUP:-20}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-24}"
DATASET="${DATASET:-$W/data_v10/dataset_config_v10.workspace.toml}"
CACHE="${CACHE:-$W/wan_cache/latents_v10}"
SEED="${SEED:-42}"
RESUME="${RESUME:-auto}"
PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-5}"

FW="${FW:-$W/Pudgy/finetune/wan}"
MUSUBI_DIR="${MUSUBI_DIR:-$W/musubi-tuner}"
ACC="${ACC:-$W/Pudgy/.venv-wan/bin/accelerate}"
PY="${PY:-$W/Pudgy/.venv-wan/bin/python}"
DIT="${DIT:-$W/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors}"
GOLDEN="${GOLDEN:-$W/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors}"
E5="${E5:-$W/wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors}"
SMOKE_CACHE="${SMOKE_CACHE:-$W/wan_cache/latents_v10_smoke}"

if [ -n "${SMOKE:-}" ]; then
  # One epoch of the 3-sample smoke set, one sample per step, a checkpoint (+ state) at step 3.
  NAME="$NAME-smoke"; MAX_STEPS=3; SAVE_EVERY_STEPS=3; GRAD_ACCUM=1; LOG_EVERY=1; RESUME=none
fi
OUT="$W/wan_output/$NAME"

die() { echo "REFUSING TO TRAIN: $*" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$CUDA_VISIBLE_DEVICES" in *,*|"") die "CUDA_VISIBLE_DEVICES must name exactly one GPU (got '$CUDA_VISIBLE_DEVICES')";; esac
awk -v w="$PRESERVE_WEIGHT" 'BEGIN { exit !(w + 0 > 0) }' || die "PRESERVE_WEIGHT must be > 0 (got $PRESERVE_WEIGHT)"

set -a; [ -f "$W/Pudgy/.env" ] && . "$W/Pudgy/.env"; set +a

for f in "$DIT" "$GOLDEN" "$E5" "$DATASET"; do
  [ -s "$f" ] || die "missing: $f"
done

# The cache must be marked for preservation + E5 before anything trains (cache_mark_v10.py).
"$PY" "$FW/cache_mark_v10.py" check --config "$DATASET" --asset "$E5" --cache-root "$CACHE" \
  || die "the cache is not marked or not consistent - run: $PY $FW/cache_mark_v10.py mark --config $DATASET --asset $E5"

if [ -n "${SMOKE:-}" ]; then
  "$PY" "$FW/cache_mark_v10.py" smoke --config "$DATASET" --asset "$E5" --cache-root "$CACHE" --out "$SMOKE_CACHE" \
    || die "could not build the smoke set"
  DATASET="$SMOKE_CACHE/smoke.toml"
  "$PY" "$FW/cache_mark_v10.py" check --config "$DATASET" --asset "$E5" --cache-root "$SMOKE_CACHE" \
    || die "the smoke set fails the check"
  case "$OUT" in *-smoke) rm -rf "$OUT";; esac        # a smoke run never resumes
fi

# Resume: RESUME=auto picks the newest COMPLETE state dir of this run (accelerate writes
# random_states_0.pkl last), RESUME=none starts fresh, anything else is a state dir.
if [ "$RESUME" = auto ]; then
  RESUME=""
  shopt -s nullglob; STATES=("$OUT/$NAME"-step*-state); shopt -u nullglob
  for ((i = ${#STATES[@]} - 1; i >= 0; i--)); do
    if [ -s "${STATES[$i]}/random_states_0.pkl" ]; then RESUME="${STATES[$i]}"; break; fi
    echo "   skipping incomplete state ${STATES[$i]}"
  done
elif [ "$RESUME" = none ]; then
  RESUME=""
fi
if [ -n "$RESUME" ]; then
  [ -s "$RESUME/random_states_0.pkl" ] || die "resume state is incomplete or missing: $RESUME"
  RS=$(basename "$RESUME" | sed -nE 's/.*-step0*([0-9]+)-state$/\1/p')
  if [ -n "$RS" ] && [ "$RS" -ge "$MAX_STEPS" ]; then
    echo "== state at step $RS >= MAX_STEPS $MAX_STEPS: nothing left to train =="
    echo "== done: $OUT (already complete) =="
    exit 0
  fi
fi

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
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")

echo "== v10 Stage 1: high-noise expert ($NAME) =="
echo "   base     : $(basename "$DIT") + MERGED $(basename "$GOLDEN") x1.0 ; t in [900, 1000], shift 5.0"
echo "   lora     : fresh, rank $RANK / alpha $ALPHA, all linear (lora_wan default)"
echo "   optim    : adamw8bit lr=$LR cosine->x$MIN_LR_RATIO warmup=$WARMUP accum=$GRAD_ACCUM seed=$SEED"
echo "   steps    : <= $MAX_STEPS optimizer steps, save every $SAVE_EVERY_STEPS (+ state)"
echo "   preserve : x$PRESERVE_WEIGHT on replay-marked samples (varlen_preserve) ; PRESERVE_STATS every $LOG_EVERY steps"
echo "   phone    : E5 frozen, offset: $E5"
echo "   swap     : $BLOCKS_TO_SWAP blocks ; GPU $CUDA_VISIBLE_DEVICES"
echo "   data     : $DATASET"
echo "   resume   : ${RESUME:-no}"
echo "   out      : $OUT"

CMD=("$ACC" launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision fp16
  "$FW/asset_embed_v9.py" train --init "$E5" --offset --freeze-asset
  --preserve --preserve-weight "$PRESERVE_WEIGHT" --preserve-marker key
  --log-every "$LOG_EVERY" --strict-base-weights --
  --task i2v-A14B
  --dit "$DIT"
  --base_weights "$GOLDEN" --base_weights_multiplier 1.0
  --dataset_config "$DATASET"
  --sdpa --mixed_precision fp16
  --network_module musubi_tuner.networks.lora_wan
  --network_dim "$RANK" --network_alpha "$ALPHA"
  --timestep_sampling shift --discrete_flow_shift 5.0
  --min_timestep 900 --max_timestep 1000 --preserve_distribution_shape
  --optimizer_type adamw8bit --learning_rate "$LR"
  --lr_scheduler cosine_with_min_lr --lr_warmup_steps "$WARMUP"
  --lr_scheduler_min_lr_ratio "$MIN_LR_RATIO"
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1.0
  --gradient_checkpointing
  --max_data_loader_n_workers 2 --persistent_data_loader_workers
  --max_train_steps "$MAX_STEPS" --save_every_n_steps "$SAVE_EVERY_STEPS"
  --save_state
  --seed "$SEED"
  --output_dir "$OUT" --output_name "$NAME"
  "${LOG[@]}" ${EXTRA[@]+"${EXTRA[@]}"})

if [ -n "${DRY_RUN:-}" ]; then
  echo "== DRY_RUN: would run in $MUSUBI_DIR:"; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
cd "$MUSUBI_DIR"
"${CMD[@]}"
echo "== done: $OUT =="
