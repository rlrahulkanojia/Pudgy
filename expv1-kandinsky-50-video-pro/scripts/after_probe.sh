#!/bin/bash
# After the motion probe: winner rule (PLAN §7) -> core suite for the winner (seed 42) -> §8b vs v8@550.
#   nohup scripts/after_probe.sh > /workspace/expv1/logs/after_probe.log 2>&1 &
set -u
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
PY=/workspace/venv-expv1/bin/python
export HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$E"
until grep -qE "probe done|!!|Traceback" /workspace/expv1/logs/probe.log; do sleep 60; done
grep -q "probe done" /workspace/expv1/logs/probe.log || { echo "!! probe did not finish"; exit 1; }
"$PY" eval/winner.py || { echo "!! winner rule failed"; exit 1; }
W=$("$PY" -c "import json;print(json.load(open('/workspace/expv1/eval/reports/winner.json'))['winner'])")
echo "== [$(date -u +%FT%TZ)] core suite for $W"
"$PY" eval/gen_core.py --config "$W" --seeds 42 || { echo "!! gen_core failed"; exit 1; }
G=/workspace/expv1/eval/gen/$W/core
D=/workspace/expv1/eval/dcls/${W}__A_k2.json
"$PY" eval/compare_v8.py --a-gen "$G" --a-dcls "$D" --a-name "$W" --rubric-template
"$PY" eval/compare_v8.py --a-gen "$G" --a-dcls "$D" --a-name "$W"
echo "== [$(date -u +%FT%TZ)] after_probe done"
