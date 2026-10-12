#!/usr/bin/env bash
# Apply the expv2 DiffSynth patches to $DIFFSYNTH_DIR (pinned SHA). Idempotent: a patch that
# is already applied is skipped, and one that applies neither way stops the setup.
#
#   diffsynth-h3-train-shift.patch  train.py: --train_flow_shift, --train_audio_flow_shift
#                                   (training sigma schedule, MODEL_NOTES 2.5) and --seed
#                                   (DiffSynth has no seed flag, MODEL_NOTES 2.20)
#   diffsynth-cosine-lr.patch       runner/parsers/logger: --lr_scheduler cosine,
#                                   --lr_warmup_steps, --lr_min_ratio over OPTIMIZER steps,
#                                   and the LR logged to loss.csv (MODEL_NOTES 2.15)
#
# Both were written against acf2ad2 and checked with `git apply --check` there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="${DIFFSYNTH_DIR:-/workspace/DiffSynth-Studio}"
for p in diffsynth-h3-train-shift.patch diffsynth-cosine-lr.patch; do
  if git -C "$DS" apply --reverse --check "$HERE/$p" 2>/dev/null; then
    echo "  $p: already applied"
  elif git -C "$DS" apply --check "$HERE/$p" 2>/dev/null; then
    git -C "$DS" apply "$HERE/$p"
    echo "  $p: applied"
  else
    echo "  $p: does NOT apply to $(git -C "$DS" rev-parse --short HEAD) - stop (wrong SHA or local edits)"
    exit 1
  fi
done
# Prove the flags exist (G0.2h re-checks this through train.py --help).
grep -q -- "--train_flow_shift" "$DS/examples/minimax_h3/model_training/train.py"
grep -q -- "--lr_scheduler" "$DS/diffsynth/diffusion/parsers.py"
echo "  patches ok"
