#!/bin/bash
# GPU job queue: keeps the GPU from sitting idle. Runs job scripts from $Q/pending/ one at a time, in
# name order, but only after the GPU has been idle (no compute processes and < 2 GB in use) for
# IDLE_CHECKS polls in a row, so it never collides with GPU work started by hand.
#   add a job : cp job.sh /workspace/gpu_queue/pending/030_name.sh
#   status    : tail /workspace/gpu_queue/queue.log; ls /workspace/gpu_queue/*/
#   pause     : touch /workspace/gpu_queue/PAUSE   (running jobs continue; nothing new starts)
Q="${GPU_QUEUE:-/workspace/gpu_queue}"; POLL="${POLL:-60}"; IDLE_CHECKS="${IDLE_CHECKS:-3}"
mkdir -p "$Q"/{pending,running,done,failed,logs}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$Q/queue.log"; }
gpu_busy() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "${used:-0}" -ge 2000 ] || [ "${apps:-0}" -gt 0 ]
}
log "queue runner up (poll ${POLL}s, idle checks $IDLE_CHECKS)"
idle=0; last_note=0
while true; do
  for r in "$Q"/running/*.sh; do                     # reap finished jobs
    [ -e "$r" ] || continue
    if ! kill -0 "$(cat "$r.pid" 2>/dev/null)" 2>/dev/null; then
      rc=$(cat "$r.rc" 2>/dev/null || echo "lost"); dest=failed; [ "$rc" = 0 ] && dest=done
      mv "$r" "$Q/$dest/"; rm -f "$r.pid" "$r.rc"; log "job $(basename "$r") -> $dest (exit $rc)"
    fi
  done
  if compgen -G "$Q/running/*.sh" >/dev/null || [ -e "$Q/PAUSE" ] || gpu_busy; then idle=0; sleep "$POLL"; continue; fi
  idle=$((idle + 1)); [ "$idle" -lt "$IDLE_CHECKS" ] && { sleep "$POLL"; continue; }
  next=$(compgen -G "$Q/pending/*.sh" | sort | head -1)
  if [ -z "$next" ]; then
    now=$(date +%s); [ $((now - last_note)) -ge 3600 ] && { log "GPU idle and queue empty"; last_note=$now; }
    sleep "$POLL"; continue
  fi
  job="$Q/running/$(basename "$next")"; mv "$next" "$job"; name=$(basename "$job" .sh)
  nohup setsid bash -c "bash '$job' > '$Q/logs/$name.log' 2>&1; echo \$? > '$job.rc'" > /dev/null 2>&1 < /dev/null &
  echo $! > "$job.pid"; log "started $name (pid $!, log $Q/logs/$name.log)"; idle=0; sleep "$POLL"
done
