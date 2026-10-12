#!/usr/bin/env bash
# =========================================================================================
# Stage 1 (DiffSynth `sft:data_process`): Qwen3-VL-32B + video/audio VAE encodings of the
# TRAINING corpus, one cache per length bucket. No DiT is loaded (as in DiffSynth's
# lora/MiniMax-H3-Pruned-FL2VA.sh). Stage 2 reads all buckets at once from cache/train/.
#
#   bash expv2-minimax-h3/configs/cache.sh            # f22 only - the trained bucket (PLAN 5.2)
#   bash expv2-minimax-h3/configs/cache.sh 22 39 56   # also the neutral-only long rungs (NOT trained by default)
#   CACHE_OFFLOAD=1 bash .../cache.sh                 # if Qwen3-VL + VAE OOM on 80 GB
#
# Why each flag (MODEL_NOTES section 2):
#   --training_cfg_scale 4   REQUIRED HERE TOO (2.19). At 1.0 the data-process stage strips
#                            the unconditional prompt embeddings from the cache, and stage 2's
#                            guidance loss then stops with "requires unconditional inputs".
#                            A cache built at 4 also serves arm D (cfg 1 ignores them).
#   --extra_inputs "input_audio,input_image"   first frame only - never end_image (2.4)
#   --silent_on_missing_audio                  corpus mp4s have no audio stream (2.9)
#   --num_frames F           exact bucket length; the mp4 carries F+1 frames (2.17)
# A partial bucket is rebuilt from scratch: DiffSynth numbers cache files 0..N-1 per run.
#
# Why f22 only: the corpus also holds f39/f56 clips, but after the active-length rule (A5)
# only `neutral` reaches them, so training them would make "length > 22" mean "neutral"
# (B11) and triple neutral's weight. They are built for the record and never cached by
# default. Stage 2 trains whatever sits under cache/train/, so caching them IS the switch.
# =========================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
BUCKETS="${*:-${EXPV2_BUCKETS:-22}}"
CORPUS="$PUDGY_DATA/processed/expv2_h3"
mkdir -p "$EXPV2_WORK/logs" "$EXPV2_WORK/cache/train"
OFFLOAD=""; [ "${CACHE_OFFLOAD:-0}" = 1 ] && OFFLOAD="--enable_model_cpu_offload"

rows() {  # data rows of a metadata CSV (prompts contain newlines, so not wc -l)
  python -c "import pandas, sys; print(len(pandas.read_csv(sys.argv[1])))" "$1"
}

cd "$DIFFSYNTH_DIR"
for F in $BUCKETS; do
  META="$CORPUS/metadata_f${F}.csv"
  OUT="$EXPV2_WORK/cache/train/f${F}"
  [ -f "$META" ] || { echo "missing $META - build or pull the corpus first (RUNBOOK step 4)"; exit 1; }
  N="$(rows "$META")"
  if [ "$N" -eq 0 ]; then echo "f$F: empty bucket, skipped"; continue; fi
  HAVE="$(find "$OUT" -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$HAVE" -eq "$N" ]; then echo "f$F: cached ($N clips)"; continue; fi
  [ "$HAVE" -gt 0 ] && { echo "f$F: partial cache ($HAVE/$N) - rebuilding"; rm -rf "$OUT"; }
  echo "f$F: caching $N clips -> $OUT"
  accelerate launch --num_processes 1 examples/minimax_h3/model_training/train.py \
    --dataset_base_path "$CORPUS" \
    --dataset_metadata_path "$META" \
    --data_file_keys "video,input_audio" \
    --extra_inputs "input_audio,input_image" \
    --silent_on_missing_audio \
    --height 768 --width 768 --num_frames "$F" \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "$EXPV2_CACHE_MODELS" \
    --processor_path "$EXPV2_H3_REPO:FL2VA/processor/" \
    --training_cfg_scale 4 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --lora_base_model "dit" --lora_target_modules "$EXPV2_LORA_TARGETS" --lora_rank "$EXPV2_LORA_RANK" \
    --use_gradient_checkpointing $OFFLOAD \
    --output_path "$OUT" \
    --task "sft:data_process" 2>&1 | tee "$EXPV2_WORK/logs/cache_f${F}.log"
  HAVE="$(find "$OUT" -name '*.pth' | wc -l | tr -d ' ')"
  [ "$HAVE" -eq "$N" ] || { echo "f$F: expected $N cache files, found $HAVE"; exit 1; }
  echo "f$F: ok ($HAVE files, $(du -sh "$OUT" | cut -f1))"
done
find "$EXPV2_WORK/cache/train" -name '*.pth' | wc -l | xargs echo "training cache files:"
