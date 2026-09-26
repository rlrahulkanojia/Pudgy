#!/bin/bash
# v8 joint LoRA training, run under supervisor. Resumes from the newest saved state.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
OUT=/workspace/wan_output/${NAME:-pudgy-v8-joint-lownoise}
LATEST=$(ls -d "$OUT"/*-step*-state 2>/dev/null | sort | tail -1)
if [ -n "$LATEST" ]; then
  echo "== resuming from $LATEST"
  export RESUME="$LATEST"
  # Append to the SAME W&B run instead of opening a second one: wandb.init honours these.
  WID=$(python3 -c "import json,sys;u=json.load(open('$OUT/status.json')).get('wandb') or '';print(u.rstrip('/').split('/')[-1])" 2>/dev/null)
  if [ -n "$WID" ]; then
    export WANDB_RUN_ID="$WID" WANDB_RESUME=allow
    echo "== W&B: resuming run $WID"
  fi
fi
cd /workspace/Pudgy
pty bash finetune/wan/train_pudgy_v8.sh 2>&1
