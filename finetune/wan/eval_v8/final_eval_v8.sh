#!/bin/bash
# End-of-run evaluation for v8, in priority order (stopping early keeps what matters most):
#   0. wait for training to finish cleanly (abort if it stops without "== done")
#   1. Tier 0: trajectory over every checkpoint
#   2. Tier 1: dcls on steps 350..600 + an SWA of the low-LR plateau (400..600)
#   3. pick the winner by a rule fixed BEFORE the data (select_winner_v8.py)
#   4. Tier 2: all suites on the winner; core+compose on v7fixed; core on golden
# Azure is mirrored after every stage (the monitor exits when training finishes).
#
#   nohup finetune/wan/eval_v8/final_eval_v8.sh > /workspace/eval_v8/final_eval.log 2>&1 &
set -u
RUN=/workspace/wan_output/pudgy-v8-joint-lownoise
NAME=$(basename "$RUN")
TLOG=/var/log/portal/pudgy-v8-train.log
EV=/workspace/Pudgy/finetune/wan/eval_v8
PY=/workspace/Pudgy/.venv-wan/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ts() { date -u +%FT%TZ; }
mirror() { "$PY" /workspace/Pudgy/finetune/wan/monitor_v8.py --once > /dev/null 2>&1 \
             && echo "   [$(ts)] mirrored to Azure" || echo "!! [$(ts)] mirror failed"; }
cd "$EV"

echo "== [$(ts)] waiting for training to finish"
while supervisorctl status pudgy-v8-train | grep -q RUNNING; do sleep 120; done
if ! grep -aq "== done:" "$TLOG" || [ ! -s "$RUN/$NAME-step00000600.safetensors" ]; then
  echo "!! [$(ts)] training stopped WITHOUT finishing (no '== done' / no step 600) - not evaluating"
  exit 1
fi
echo "== [$(ts)] training finished"
mirror

echo "== [$(ts)] Tier 0: trajectory"
"$PY" lora_tools_v8.py traj "$RUN" --tb | tail -4

echo "== [$(ts)] Tier 0: SWA of the low-LR plateau, steps 400..600"
"$PY" lora_tools_v8.py swa "$RUN" --from 400 --to 600 --rank 64 | tail -2

echo "== [$(ts)] Tier 1: dcls"
"$PY" dcls_v8.py --configs v8@350 v8@400 v8@450 v8@500 v8@550 v8@600 v8@swa400-600 --k 2 \
  2>&1 | grep -aE "^== |holdout" || echo "!! [$(ts)] dcls FAILED"
mirror

echo "== [$(ts)] selecting the winner"
WIN=$("$PY" select_winner_v8.py) || { echo "!! [$(ts)] no checkpoint beats v7fixed - stopping before Tier 2"; mirror; exit 1; }
echo "== [$(ts)] winner: $WIN   ($(cat /workspace/eval_v8/winner.json | tr -d '\n' | cut -c1-200))"

echo "== [$(ts)] Tier 2: all suites on $WIN"
"$PY" suite_v8.py --suite all --configs "$WIN" 2>&1 | grep -aE "^\s+\[|Error|Traceback" ; mirror
echo "== [$(ts)] Tier 2: core + compose on v7fixed"
for s in core compose; do "$PY" suite_v8.py --suite $s --configs v7fixed "$WIN" 2>&1 | grep -aE "^\s+\[|Error|Traceback"; done; mirror
echo "== [$(ts)] Tier 2: core on golden"
"$PY" suite_v8.py --suite core --configs golden v7fixed "$WIN" 2>&1 | grep -aE "^\s+\[|Error|Traceback"; mirror
echo "== [$(ts)] final evaluation done - scores: /workspace/eval_v8/suites/*/scores.json"
