#!/bin/bash
# =============================================================================
# The Pudgy inference API on a GPU box (server.py; guide: README.md in this folder).
#
#   bash finetune/wan/infer_api/serve.sh start    # first time: the API venv + preset start frames; then the
#                                                  # server + an HTTPS tunnel in tmux 'infer'. The dashboard
#                                                  # finds it by itself within ~20 s.
#   bash finetune/wan/infer_api/serve.sh status   # health, public address, last log lines
#   bash finetune/wan/infer_api/serve.sh stop     # clean stop: the dashboard says "no server" straight away
#   bash finetune/wan/infer_api/serve.sh logs     # follow the server log (generator output: $WORK/logs/generator.log)
#
# Needs the box set up as for v10 (v10_box.sh setup + pull: Wan2.2 base, v8 merged, golden, E5) and
# two lines in /workspace/Pudgy/.env: INFERENCE_API_KEY=<key> (the dashboard has the same key) and
# AZURE_STORAGE_CONNECTION_STRING=... (registration + saved videos).
# Env: MODELS (v8merged; e.g. MODELS=v8merged,v10@250), GPU (0), PORT (8765),
#      FORCE=1 (start even though a training run holds the GPU - jobs would then be refused anyway).
# Switching the box off without `stop` is fine: the heartbeat stops and the dashboard notices in 2 min.
# =============================================================================
set -euo pipefail
W="${W:-/workspace}"
REPO=$W/Pudgy
API=$REPO/finetune/wan/infer_api
VENV=$W/infer_api_venv
WORK=$W/infer_api
LOG=$WORK/server.log
PORT="${PORT:-8765}"
MODELS="${MODELS:-v8merged}"
GPU="${GPU:-0}"
mkdir -p "$WORK"

health() { curl -s -m 5 "http://127.0.0.1:$PORT/v1/health"; }
server_pids() { pgrep -f "[s]erver.py --backend wan" || true; }

case "${1:-start}" in
start)
  if [ -n "$(server_pids)" ]; then echo "already running (bash $0 status)"; exit 0; fi
  if pgrep -f "[a]sset_embed_v9\.py train" >/dev/null && [ -z "${FORCE:-}" ]; then
    echo "A training run is using the GPU; generation would run out of memory. Start this when it has finished."
    exit 1
  fi
  if [ -z "${INFERENCE_API_KEY:-}" ] && ! grep -q "^INFERENCE_API_KEY=." "$REPO/.env" 2>/dev/null; then
    echo "Add INFERENCE_API_KEY=<key> to $REPO/.env first (and give the dashboard the same key). Make one with:"
    echo "  python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
    exit 1
  fi
  for f in ffmpeg ffprobe curl tmux; do command -v $f >/dev/null || { echo "$f is not installed"; exit 1; }; done
  if [ ! -x "$VENV/bin/uvicorn" ]; then
    echo "== the API venv ($VENV): FastAPI + uvicorn, separate from the trainer venv"
    "$REPO/.venv-wan/bin/python" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$API/requirements.txt"
  fi
  if [ ! -s "$WORK/keyframes/keyframes.json" ]; then
    echo "== preset start frames (the client long-form keyframes) from Azure"
    "$REPO/.venv-wan/bin/python" "$REPO/finetune/wan/azure_sync.py" pull --dest "$WORK/kf_pull" \
      2026-10-28_longform/workdir/keyframes/
    mkdir -p "$WORK/keyframes"
    cp "$WORK/kf_pull/2026-10-28_longform/workdir/keyframes/"* "$WORK/keyframes/"
  fi
  rm -f "$WORK/public_url"
  echo "== $(date -u +%FT%TZ) start (models $MODELS, GPU $GPU, port $PORT)" >> "$LOG"
  # A crash restarts the server (up to 3 times, 10 s apart); a clean stop (exit 0) ends the loop.
  tmux kill-session -t infer 2>/dev/null || true
  tmux new-session -d -s infer "cd $API && for i in 1 2 3 4; do \
    $VENV/bin/python server.py --backend wan --tunnel --register --upload --port $PORT --gpu $GPU --models $MODELS \
      >> $LOG 2>&1 && break; echo \"== \$(date -u +%FT%TZ) server exited \$? (try \$i of 4)\" >> $LOG; sleep 10; done; \
    echo SERVER_EXIT; exec bash"
  echo "== starting: checking the model files and opening the tunnel (about a minute)"
  for _ in $(seq 1 90); do
    if health >/dev/null && [ -s "$WORK/public_url" ]; then break; fi
    sleep 2
  done
  if ! health >/dev/null; then echo "the server did not come up - see: tail -n 40 $LOG"; exit 1; fi
  health | "$VENV/bin/python" -c "import json, sys; h = json.load(sys.stdin); print(f\"== up: {h['state']}\" + (f\" ({h['reason']})\" if h['reason'] else ''))"
  if [ -s "$WORK/public_url" ]; then
    echo "== public address: $(cat "$WORK/public_url")   (docs: $(cat "$WORK/public_url")/docs)"
    echo "   registered for the dashboard; it shows the server as online within ~20 s"
  else
    echo "== no tunnel address yet (cloudflared is slow or blocked) - the server works locally; see $LOG"
  fi
  ;;
status)
  if [ -n "$(server_pids)" ]; then echo "server: running (pid $(server_pids | tr '\n' ' '))"; else echo "server: NOT running"; fi
  health | "$VENV/bin/python" -m json.tool 2>/dev/null || echo "(no answer on port $PORT)"
  echo "public address: $(cat "$WORK/public_url" 2>/dev/null || echo none)"
  echo "--- last log lines"; tail -n 8 "$LOG" 2>/dev/null || true
  ;;
stop)
  if [ -z "$(server_pids)" ]; then echo "not running"; tmux kill-session -t infer 2>/dev/null || true; exit 0; fi
  # SIGINT = uvicorn's clean shutdown: running jobs are cancelled, the registration is marked stopped
  kill -INT $(server_pids)
  for _ in $(seq 1 30); do [ -z "$(server_pids)" ] && break; sleep 1; done
  [ -n "$(server_pids)" ] && { echo "still running after 30 s - killing"; kill -9 $(server_pids) || true; }
  tmux kill-session -t infer 2>/dev/null || true
  echo "stopped"
  ;;
logs)
  tail -n 40 -f "$LOG"
  ;;
*)
  echo "usage: bash $0 start|status|stop|logs"; exit 2 ;;
esac
