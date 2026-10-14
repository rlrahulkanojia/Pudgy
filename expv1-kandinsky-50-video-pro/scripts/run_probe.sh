#!/bin/bash
# PLAN §6.3-6.4: motion probe on base + every 100-step checkpoint (+ any extra configs), then score.
#   nohup scripts/run_probe.sh [extra configs...] > /workspace/expv1/logs/probe.log 2>&1 &
set -u
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
PY=/workspace/venv-expv1/bin/python
export HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PFX=${PFX:-expv1}                              # expv1 (arm A) | expv1b (arm B)
CFGS="base $PFX@100 $PFX@200 $PFX@300 $PFX@400 $PFX@500 $PFX@600 $*"
cd "$E"
for c in $CFGS; do
  echo "== [$(date -u +%FT%TZ)] probe $c"
  "$PY" eval/gen_core.py --config "$c" --suite probe || { echo "!! generation failed for $c"; exit 1; }
done
"$PY" eval/motion_probe.py --prefix "$PFX" --configs $CFGS
echo "== [$(date -u +%FT%TZ)] probe done"
