#!/bin/bash
# PLAN §6.3-6.4: motion probe on base + every 100-step checkpoint (+ any extra configs), then score.
#   nohup scripts/run_probe.sh [extra configs...] > /workspace/expv1/logs/probe.log 2>&1 &
set -u
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
PY=/workspace/venv-expv1/bin/python
export HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CFGS="base expv1@100 expv1@200 expv1@300 expv1@400 expv1@500 expv1@600 $*"
cd "$E"
for c in $CFGS; do
  echo "== [$(date -u +%FT%TZ)] probe $c"
  "$PY" eval/gen_core.py --config "$c" --suite probe || { echo "!! generation failed for $c"; exit 1; }
done
"$PY" eval/motion_probe.py --configs $CFGS
echo "== [$(date -u +%FT%TZ)] probe done"
