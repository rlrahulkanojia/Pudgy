#!/usr/bin/env bash
# =========================================================================================
# Stage 2 (DiffSynth `sft:train`) for the four expv2 runs (PLAN 6.2-6.6). Every run starts
# from zero: DiffSynth's --resume_from_checkpoint restores weights only, so nothing depends
# on continuing a run (MODEL_NOTES 2.15).
#
#   bash expv2-minimax-h3/configs/train.sh expv2-h3-s12          # pilot, shift 12 / audio 3
#   bash expv2-minimax-h3/configs/train.sh expv2-h3-s222         # pilot, DiffSynth's 2.22 / 2.22
#   bash expv2-minimax-h3/configs/train.sh expv2-h3-main         # SHIFT read from eval/tier1/calibration.json
#   bash expv2-minimax-h3/configs/train.sh expv2-h3-d            # arm D, only if PLAN 6.6 fires
#
# Overrides: SHIFT, AUDIO_SHIFT, REPEAT, LR, MAX_PREVIEW=1 (print the command and exit),
#            FORCE=1 (allow an output dir that already holds checkpoints).
#
# Checkpoints: output/<run>/step-<samples>.safetensors. DiffSynth's --save_steps counts
# SAMPLES (ModelLogger.on_step_end runs once per dataloader item), so "every 50 optimizer
# steps" at accumulation 8 is --save_steps 400, and step-2000 is optimizer step 250
# (MODEL_NOTES 2.18). The last checkpoint is written at the end whatever the remainder.
# =========================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
RUN="${1:?usage: train.sh <expv2-h3-s12|expv2-h3-s222|expv2-h3-main|expv2-h3-d>}"

CFG=4; ADAPTER_FLAGS=()
case "$RUN" in
  # Budgets (PLAN 6.3-6.4) on the 672 f22 clips: pilots 2 x 672 = 1,344 samples = 168
  # optimizer steps; main 6 x 672 = 4,032 samples = 504 steps (~168 exposures/performance).
  expv2-h3-s12)  SHIFT="${SHIFT:-12}";   AUDIO_SHIFT="${AUDIO_SHIFT:-3}";    REPEAT="${REPEAT:-2}" ;;
  expv2-h3-s222) SHIFT="${SHIFT:-2.22}"; AUDIO_SHIFT="${AUDIO_SHIFT:-2.22}"; REPEAT="${REPEAT:-2}" ;;
  expv2-h3-main|expv2-h3-d)
    if [ -z "${SHIFT:-}" ]; then   # the pilot winner, written by eval/dcls.py --calibrate (PLAN 6.3)
      SHIFT="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["shift"]["winner_train_flow_shift"])' \
               "$EXPV2_WORK/eval/tier1/calibration.json" 2>/dev/null || true)"
    fi
    : "${SHIFT:?no SHIFT given and no shift.winner_train_flow_shift in \$EXPV2_WORK/eval/tier1/calibration.json - run dcls.py --calibrate}"
    # The pilots pair video shift 12 with audio 3 (the inference pairing) and 2.22 with 2.22.
    if [ -z "${AUDIO_SHIFT:-}" ]; then
      case "$SHIFT" in 12|12.0) AUDIO_SHIFT=3 ;; *) AUDIO_SHIFT="$SHIFT" ;; esac
    fi
    REPEAT="${REPEAT:-6}"
    if [ "$RUN" = expv2-h3-d ]; then
      CFG=1     # the adapter replaces the guidance loss; never both (MODEL_NOTES 2.16)
      [ -f "$EXPV2_ADAPTER" ] || { echo "missing $EXPV2_ADAPTER - fetch_models.py --set adapter"; exit 1; }
      ADAPTER_FLAGS=(--preset_lora_path "$EXPV2_ADAPTER" --preset_lora_model dit)
    fi ;;
  *) echo "unknown run $RUN"; exit 1 ;;
esac
LR="${LR:-1e-4}"

CACHE="$EXPV2_WORK/cache/train"
OUT="$EXPV2_WORK/output/$RUN"
N="$(find "$CACHE" -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
[ "$N" -gt 0 ] || { echo "no training cache under $CACHE - run configs/cache.sh"; exit 1; }
SAMPLES=$((N * REPEAT))
SAVE_STEPS=$((EXPV2_SAVE_EVERY_OPT * EXPV2_GRAD_ACCUM))
OPT_STEPS=$(( (SAMPLES + EXPV2_GRAD_ACCUM - 1) / EXPV2_GRAD_ACCUM ))
if ls "$OUT"/step-*.safetensors >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then
  echo "$OUT already holds checkpoints; move it away or FORCE=1"; exit 1
fi
mkdir -p "$OUT" "$EXPV2_WORK/logs"

CMD=(accelerate launch --num_processes 1 examples/minimax_h3/model_training/train.py
  --dataset_base_path "$CACHE" --dataset_repeat "$REPEAT" --num_epochs 1
  --data_file_keys "video,input_audio" --extra_inputs "input_audio,input_image"
  --height 768 --width 768 --num_frames 22    # unused with a cache, but train.py rejects non-17n+5 values
  --model_id_with_origin_paths "$EXPV2_DIT" --processor_path "$EXPV2_H3_REPO:FL2VA/processor/"
  --remove_prefix_in_ckpt "pipe.dit."
  --learning_rate "$LR" --weight_decay 1e-3 --gradient_accumulation_steps "$EXPV2_GRAD_ACCUM"
  --lr_scheduler cosine --lr_warmup_steps 20 --lr_min_ratio 0.1
  --training_cfg_scale "$CFG" --audio_loss_weight 0
  --train_flow_shift "$SHIFT" --train_audio_flow_shift "$AUDIO_SHIFT" --seed 42
  --lora_base_model dit --lora_target_modules "$EXPV2_LORA_TARGETS" --lora_rank "$EXPV2_LORA_RANK"
  --use_gradient_checkpointing --find_unused_parameters
  --save_steps "$SAVE_STEPS" --enable_csv_log
  --output_path "$OUT" --task "sft:train" ${ADAPTER_FLAGS[@]+"${ADAPTER_FLAGS[@]}"})

cat > "$OUT/expv2_run.json" <<EOF
{"run": "$RUN", "shift": "$SHIFT", "audio_shift": "$AUDIO_SHIFT", "training_cfg_scale": $CFG,
 "repeat": $REPEAT, "cache_files": $N, "samples": $SAMPLES, "optimizer_steps": $OPT_STEPS,
 "save_steps_samples": $SAVE_STEPS, "lr": "$LR", "adapter": "${ADAPTER_FLAGS[*]:-}",
 "diffsynth_sha": "$(git -C "$DIFFSYNTH_DIR" rev-parse HEAD)",
 "repo_sha": "$(git -C "$REPO_ROOT" rev-parse HEAD)", "started": "$(date -u +%FT%TZ)"}
EOF
echo "$RUN: $N cache files x $REPEAT = $SAMPLES samples = $OPT_STEPS optimizer steps; save every $SAVE_STEPS samples"
if [ "${MAX_PREVIEW:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
cd "$DIFFSYNTH_DIR"
"${CMD[@]}" 2>&1 | tee "$OUT/train.log"
ls -1 "$OUT"/step-*.safetensors
