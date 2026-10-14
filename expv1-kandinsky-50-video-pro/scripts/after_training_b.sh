#!/bin/bash
# Arm B (PLAN §6.4), after gate_breaks.sh has finished Tier 1 on every checkpoint:
# motion probe (base + every 100 steps) -> winner rule (probing any extra candidate it asks for)
# -> core suite for the winner (seed 42) -> §8b vs v8@550 -> arm A vs arm B (same scorer).
#   nohup scripts/after_training_b.sh > /workspace/expv1/logs/after_training_b.log 2>&1 &
set -u
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
PY=/workspace/venv-expv1/bin/python
export HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PFX=expv1b
GB=/workspace/expv1/logs/gate_breaks.log
cd "$E"
until grep -qE "all done|^!!" "$GB"; do sleep 120; done
grep -q "all done" "$GB" || { echo "!! gate_breaks did not finish"; exit 1; }
EXTRA=""
for i in 1 2 3; do
  scripts/run_probe.sh $EXTRA || { echo "!! probe failed"; exit 1; }
  out=$("$PY" eval/winner.py --prefix $PFX); rc=$?; echo "$out"
  [ $rc = 0 ] && break
  [ $rc = 3 ] || { echo "!! winner rule failed"; exit 1; }
  EXTRA="$EXTRA $(echo "$out" | sed -n 's/^NEED_PROBE: //p')"
done
[ $rc = 0 ] || { echo "!! winner unresolved"; exit 1; }
W=$("$PY" -c "import json;print(json.load(open('/workspace/expv1/eval/reports/winner_expv1b.json'))['winner'])")
echo "== [$(date -u +%FT%TZ)] core suite for $W"
"$PY" eval/gen_core.py --config "$W" --seeds 42 || { echo "!! gen_core failed"; exit 1; }
G=/workspace/expv1/eval/gen/$W/core
D=/workspace/expv1/eval/dcls/${W}__A_k2.json
"$PY" eval/compare_v8.py --a-gen "$G" --a-dcls "$D" --a-name "$W" --rubric-template
"$PY" eval/compare_v8.py --a-gen "$G" --a-dcls "$D" --a-name "$W"
echo "== arm B vs arm A (expv1@550), same scorer"
"$PY" eval/compare_v8.py --a-gen "$G" --a-dcls "$D" --a-name "$W" \
  --b-gen /workspace/expv1/eval/gen/expv1@550/core --b-dcls /workspace/expv1/eval/dcls/expv1@550__A_k2.json --b-name expv1@550
echo "== [$(date -u +%FT%TZ)] after_training_b done"
