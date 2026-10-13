#!/bin/bash
# PLAN §6.3 gate breaks, automated (the v8 pattern, finetune/wan/eval_v8/gate_break_v8.sh):
# for each step S: wait for the step-S checkpoint + state -> stop the expv1-train service ->
# Tier 1 on expv1@S (eval/dcls.py, layout A, holdout + seen) -> gate_rule.py -> restart
# training (it resumes from the newest state) and VERIFY the step count continued.
# Training and the classifier cannot share the 80 GB card, hence the stop. After the last
# break, Tier 1 runs on every 50-step checkpoint once training finishes (the winner rule, §7).
#
#   nohup scripts/gate_breaks.sh 100 300 > /workspace/expv1/logs/gate_breaks.log 2>&1 &
set -u
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
OUT=/workspace/expv1/output
NAME=${NAME:-expv1-k5pro-a}
TLOG=/workspace/expv1/logs/train.log
PY=/workspace/venv-expv1/bin/python
export HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%FT%TZ; }
running() { supervisorctl status expv1-train | grep -q RUNNING; }

for S in "$@"; do
  STATE=$OUT/$NAME-step$(printf %08d "$S")-state
  CKPT=$OUT/$NAME-step$(printf %08d "$S").safetensors
  echo "== [$(ts)] waiting for step $S: $CKPT"
  until [ -f "$STATE/random_states_0.pkl" ] && [ -s "$CKPT" ]; do
    running || { echo "!! [$(ts)] training not running while waiting for step $S"; exit 1; }
    sleep 60
  done
  sleep 45
  echo "== [$(ts)] step $S saved - stopping training"
  supervisorctl stop expv1-train
  cp "$TLOG" "$OUT/train_segment_to_step$S.log" 2>/dev/null
  ( cd "$E" && "$PY" eval/dcls.py score --configs "expv1@$S" --split holdout seen ) || echo "!! dcls failed"
  if [ "$S" = 300 ]; then                    # PLAN §6.3: the step-300 break also runs the motion probe
    ( cd "$E" && "$PY" eval/gen_core.py --config "expv1@$S" --suite probe ) || echo "!! motion probe failed"
  fi
  ( cd "$E" && "$PY" eval/gate_rule.py --step "$S" ); rc=$?
  if [ $rc = 2 ]; then
    echo "!! [$(ts)] gate at step $S FAILED - training stays stopped (PLAN §6.3). See eval/reports/gate_$S.json"
    exit 2
  fi
  echo "== [$(ts)] restarting training"
  : > "$TLOG"
  supervisorctl start expv1-train
  ok=0
  for i in $(seq 1 45); do
    sleep 60
    if grep -aq "continuing at global step $S " "$TLOG" &&
       tr '\r' '\n' < "$TLOG" | grep -aqE "\| *($((S+1))|$((S+2))|$((S+3)))/[0-9]+ \["; then ok=1; break; fi
    grep -aq "Traceback" "$TLOG" && break
  done
  if [ $ok = 1 ]; then echo "== [$(ts)] resume verified at step $S"
  else
    echo "!! [$(ts)] RESUME CHECK FAILED after step $S - stopping training"; supervisorctl stop expv1-train
    tail -c 3000 "$TLOG" | tr '\r' '\n' | tail -15; exit 1
  fi
done

echo "== [$(ts)] gate breaks done; waiting for training to finish"
while running; do sleep 120; done
CK=$(ls "$OUT"/"$NAME"-step*.safetensors | sed -E 's/.*-step0*([0-9]+)\.safetensors/expv1@\1/' | sort -t@ -k2 -n)
echo "== [$(ts)] Tier 1 on every checkpoint: $CK"
( cd "$E" && "$PY" eval/dcls.py score --configs $CK --split holdout seen )
echo "== [$(ts)] all done"
