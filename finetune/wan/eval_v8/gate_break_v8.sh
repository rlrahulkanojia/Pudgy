#!/bin/bash
# Automated gate breaks for the v8 run (plan section 5): for each step S given,
#   wait for the step-S resume state -> stop training -> Tier 1 (dcls v8@S) + Tier 0
#   (trajectory) -> restart training -> VERIFY the resume continued at S and advanced.
#
# Training (~67 GB) and the classifier (~38 GB) cannot share the 80 GB card, hence the
# stop. The supervisor wrapper resumes from the newest *-stepN-state; musubi is patched
# (patches/musubi-v0.3.4-resume-step.patch) to continue the step count from N. If the
# post-restart check fails, training is STOPPED and flagged rather than left running on a
# resume that would overwrite checkpoints.
#
#   nohup finetune/wan/eval_v8/gate_break_v8.sh 100 300 > /workspace/eval_v8/gate_break.log 2>&1 &
set -u
RUN=/workspace/wan_output/pudgy-v8-joint-lownoise
NAME=$(basename "$RUN")
TLOG=/var/log/portal/pudgy-v8-train.log
EV=/workspace/Pudgy/finetune/wan/eval_v8
PY=/workspace/Pudgy/.venv-wan/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ts() { date -u +%FT%TZ; }

for S in "$@"; do
  STATE=$RUN/$NAME-step$(printf %08d "$S")-state
  CKPT=$RUN/$NAME-step$(printf %08d "$S").safetensors
  echo "== [$(ts)] waiting for step $S state: $STATE"
  # accelerate writes random_states_0.pkl last ("Random states saved" is its final line)
  until [ -f "$STATE/random_states_0.pkl" ] && [ -s "$CKPT" ]; do
    if ! supervisorctl status pudgy-v8-train | grep -q RUNNING; then
      echo "!! [$(ts)] training is not running while waiting for step $S - aborting gate breaks"; exit 1
    fi
    sleep 60
  done
  sleep 45                                   # let the save flush fully
  echo "== [$(ts)] step $S saved - stopping training for the gate break"
  supervisorctl stop pudgy-v8-train
  cp "$TLOG" "$RUN/train_segment_to_step$S.log" 2>/dev/null   # the wrapper rotates the log on restart

  echo "== [$(ts)] Tier 1: dcls v8@$S"
  ( cd "$EV" && "$PY" dcls_v8.py --configs "v8@$S" --k 2 ) || echo "!! [$(ts)] dcls v8@$S FAILED - continuing"
  echo "== [$(ts)] Tier 0: trajectory"
  "$PY" "$EV/lora_tools_v8.py" traj "$RUN" --tb || echo "   (trajectory needs >= 2 step checkpoints)"

  echo "== [$(ts)] restarting training"
  supervisorctl start pudgy-v8-train
  # Verify: the patched counter logs "continuing at global step S", and the bar passes S.
  ok=0
  for i in $(seq 1 60); do                    # up to 60 min: DiT load + merge + >= 1 step
    sleep 60
    if grep -aq "continuing at global step $S " "$TLOG" 2>/dev/null &&
       tr '\r' '\n' < "$TLOG" | grep -aqE "steps: +[0-9]+% *($((S+1))|$((S+2))|$((S+3)))/"; then ok=1; break; fi
    if grep -aq "Traceback" "$TLOG" 2>/dev/null; then break; fi
  done
  if [ $ok = 1 ]; then
    echo "== [$(ts)] resume verified: continued from step $S and advanced"
  else
    echo "!! [$(ts)] RESUME CHECK FAILED after step $S - stopping training to protect checkpoints"
    supervisorctl stop pudgy-v8-train
    tail -c 3000 "$TLOG" | tr '\r' '\n' | tail -15
    exit 1
  fi
done
echo "== [$(ts)] all gate breaks done"
