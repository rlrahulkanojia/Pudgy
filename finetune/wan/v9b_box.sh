#!/bin/bash
# =============================================================================
# v9b-1 (long-form scenes) on the GPU box, end to end. One entry point, one step at a time,
# every step idempotent. Runbook: training_approach/v9/RUNBOOK_v9b.md. Assumes the v9c box
# state: venv, musubi, base weights, pudgy-v8-merged (v9_box.sh setup / restore / merge).
#
#   bash finetune/wan/v9b_box.sh sync       # curation + code -> Azure v9b/work (any time; the box is not persistent)
#   bash finetune/wan/v9b_box.sh check      # registry lint + every curation file valid + holdout chosen
#   bash finetune/wan/v9b_box.sh prep       # prep_v9b -> /workspace/data_v9b
#   bash finetune/wan/v9b_box.sh cache      # latents + T5 for the training set, verified
#   bash finetune/wan/v9b_box.sh evalprep   # held-out scene windows + start frames, latents cached
#   bash finetune/wan/v9b_box.sh baseline   # Tier 1 on v8merged: regression set (copied from v9c) + scene set
#   bash finetune/wan/v9b_box.sh smoke      # 2 steps on the heaviest bucket: memory, s/clip
#   bash finetune/wan/v9b_box.sh train      # the run (background) + Azure monitor
#   bash finetune/wan/v9b_box.sh status     # where the run is
#   bash finetune/wan/v9b_box.sh pause S    # stop training once step S's state is saved (mid-run Tier 1)
#   bash finetune/wan/v9b_box.sh tier1 S..  # Tier 1 (regression + scene) on checkpoints S..
#   bash finetune/wan/v9b_box.sh eval       # after training: Tier 1 on all, gate, Tier 2 scene suite
#   bash finetune/wan/v9b_box.sh archive    # everything that must survive the box -> Azure v9b/
# =============================================================================
set -euo pipefail
W=/workspace
REPO=$W/Pudgy
FW=$REPO/finetune/wan
PY=$REPO/.venv-wan/bin/python
M=$W/wan_models
RUN_NAME="${RUN_NAME:-pudgy-v9b-lownoise}"
export RUN_NAME
RUN=$W/wan_output/$RUN_NAME
MERGED=$W/wan_output/v8_merged/pudgy-v8-merged.safetensors
DATA_LINK=$W/data_v9b
CFG=$DATA_LINK/dataset_config_v9b.workspace.toml
EVAL=$W/eval_v9b
LOGS=$W/logs_v9b
STEPS_EVAL="${STEPS_EVAL:-100 200 300 400}"
export PUDGY_DATA=$W/Data
mkdir -p "$LOGS"
ts() { date -u +%FT%TZ; }
log() { echo "== [$(ts)] $*" | tee -a "$LOGS/v9b_box.log"; }

musubi() { ( source "$REPO/.venv-wan/bin/activate" && cd "$W/musubi-tuner" && "$@" ); }

cache_cfg() {   # $1 = dataset toml; latents (+ T5 unless $2 = latents-only)
  musubi python src/musubi_tuner/wan_cache_latents.py --dataset_config "$1" \
    --vae "$M/comfy21/split_files/vae/wan_2.1_vae.safetensors" --i2v --skip_existing --num_workers 16
  if [ "${2:-}" != "latents-only" ]; then
    musubi python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "$1" \
      --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" --batch_size 16 --skip_existing --num_workers 16
  fi
}

step=${1:-}
case "$step" in
sync)
  "$PY" "$FW/azure_sync.py" push --prefix v9b/work/scenes --tree "$FW/scenes" --exclude __pycache__
  "$PY" "$FW/azure_sync.py" push --prefix v9b/work/code "$FW/prep_v9b.py" "$FW/train_pudgy_v9b.sh" "$FW/v9b_box.sh"
  [ -d "$FW/eval_v9b" ] && "$PY" "$FW/azure_sync.py" push --prefix v9b/work/eval_v9b --tree "$FW/eval_v9b" --exclude __pycache__
  "$PY" "$FW/azure_sync.py" push --prefix v9b/work/docs --tree "$REPO/training_approach/v9"
  "$PY" "$FW/azure_sync.py" push --prefix v9b/work/reports --tree "$REPO/docs/training_reports/v9"
  [ -d "$LOGS" ] && "$PY" "$FW/azure_sync.py" push --prefix v9b/logs/box --tree "$LOGS"
  log "synced curation, code and docs to Azure v9b/work"
  ;;

check)
  "$PY" "$FW/assets/registry.py" lint --data "$W/Data"
  ( cd "$FW/scenes" && "$PY" curate_check.py curation/ )
  n=$(ls "$FW/scenes/curation/"*.json | wc -l)
  [ "$n" -eq 99 ] || { echo "only $n of 99 skits curated"; exit 1; }
  [ -s "$FW/scenes/holdout_v9b.json" ] || { echo "no holdout_v9b.json"; exit 1; }
  log "check OK: 99 curation files valid, holdout chosen"
  ;;

prep)
  if [ -s "$DATA_LINK/manifest.json" ]; then log "already built: $(readlink -f "$DATA_LINK")"; exit 0; fi
  # 16 workers: ~128 concurrent ffmpeg 6.1 decodes deadlocked on this box (v9c prep).
  "$PY" "$FW/prep_v9b.py" --workers "${PREP_WORKERS:-16}" --out "$W/Data/processed" 2>&1 | tee "$LOGS/prep.log"
  SET=$(ls -d "$W/Data/processed"/v9b_[0-9]* | tail -1)
  ln -sfn "$SET" "$DATA_LINK"
  log "training set: $SET -> $DATA_LINK"
  ;;

cache)
  [ -s "$CFG" ] || { echo "no $CFG - run prep first"; exit 1; }
  cache_cfg "$CFG" 2>&1 | tee "$LOGS/cache.log"
  "$PY" "$FW/verify_cache_v8.py" --config "$CFG" 2>&1 | tee -a "$LOGS/cache.log"
  # Spatial size too: scene latents must be 480x848 / 8 = 60 x 106 (no silent re-bucketing),
  # replay 1024 / 8 = 128 x 128.
  "$PY" - "$CFG" <<'PYEOF' 2>&1 | tee -a "$LOGS/cache.log"
import sys, tomllib
from pathlib import Path
from safetensors import safe_open
cfg = tomllib.loads(Path(sys.argv[1]).read_text())
bad = 0
for ds in cfg["datasets"]:
    want = (128, 128) if "replay" in ds["cache_directory"] else (106, 60)
    files = sorted(Path(ds["cache_directory"]).glob("*_wan.safetensors"))
    shapes = set()
    for f in files:
        with safe_open(str(f), "pt") as s:
            k = next(k for k in s.keys() if k.startswith("latents_") and "image" not in k)
            shapes.add(tuple(s.get_slice(k).get_shape()[2:]))
    ok = shapes == {want}
    bad += not ok
    print(f"  {Path(ds['cache_directory']).name:<12} {len(files):4d} latents  H x W {sorted(shapes)}  {'ok' if ok else 'WRONG, want ' + str(want)}")
sys.exit(1 if bad else 0)
PYEOF
  ;;

evalprep)
  "$PY" "$FW/eval_v9b/build_eval_v9b.py" 2>&1 | tee "$LOGS/evalprep.log"
  cache_cfg "$EVAL/scenes/scenes.toml" latents-only 2>&1 | tee -a "$LOGS/evalprep.log"
  ;;

baseline)
  cd "$FW/eval_v9b"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$PY" tier1_v9b.py --configs v8merged 2>&1 | tee "$LOGS/baseline.log"
  ;;

smoke)
  [ -s "$CFG" ] || { echo "no $CFG - run prep + cache first"; exit 1; }
  "$PY" - "$CFG" "$DATA_LINK/smoke.workspace.toml" <<'PYEOF'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
head, *blocks = re.split(r"(?m)^\[\[datasets\]\]", text)
def tokens(b):   # latent frames x tokens per frame
    n = int(re.search(r"target_frames = \[(\d+)\]", b).group(1))
    m = re.search(r"resolution = \[(\d+), (\d+)\]", b) or re.search(r"resolution = \[(\d+), (\d+)\]", head)
    return ((n - 1) // 4 + 1) * (int(m.group(1)) // 16) * (int(m.group(2)) // 16)
top = max(blocks, key=tokens)
open(dst, "w").write(head + "[[datasets]]" + top)
print("smoke bucket:", re.search(r'cache_directory = "([^"]+)"', top).group(1), "tokens", tokens(top))
PYEOF
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$LOGS/smoke_mem.csv" &
  SMI=$!
  MAX_STEPS=2 SAVE_EVERY_STEPS=2 GRAD_ACCUM=1 NAME=pudgy-v9b-smoke \
    DATASET="$DATA_LINK/smoke.workspace.toml" bash "$FW/train_pudgy_v9b.sh" 2>&1 | tee "$LOGS/smoke.log" || true
  kill $SMI 2>/dev/null || true
  echo "peak GPU memory: $(sort -n "$LOGS/smoke_mem.csv" | tail -1) MiB"
  grep -aiE "out of memory|retr" "$LOGS/smoke.log" | tail -3 || true
  grep -aE "s/it|it/s" "$LOGS/smoke.log" | tail -2 || true
  grep -aq "== done:" "$LOGS/smoke.log" && echo "SMOKE OK" || { echo "SMOKE FAILED - see $LOGS/smoke.log"; exit 1; }
  ;;

train)
  [ -s "$MERGED" ] || { echo "no merged base - run v9_box.sh merge"; exit 1; }
  LATEST=$(ls -d "$RUN"/*-step*-state 2>/dev/null | sort | tail -1 || true)
  if [ -n "$LATEST" ]; then log "resuming from $LATEST"; export RESUME="$LATEST"; fi
  NAME="$RUN_NAME" nohup bash "$FW/train_pudgy_v9b.sh" >> "$LOGS/train.log" 2>&1 &
  echo $! > "$LOGS/train.pid"
  if ! { [ -s "$LOGS/monitor.pid" ] && kill -0 "$(cat "$LOGS/monitor.pid")" 2>/dev/null; }; then
    V8_RUN="$RUN" V8_LOG="$LOGS/train.log" MONITOR_PREFIX=v9b MONITOR_EVAL="$EVAL" \
      nohup "$PY" "$FW/monitor_v8.py" >> "$LOGS/monitor.log" 2>&1 &
    echo $! > "$LOGS/monitor.pid"
  fi
  log "training pid $(cat "$LOGS/train.pid"), monitor pid $(cat "$LOGS/monitor.pid"); tail -f $LOGS/train.log"
  ;;

status)
  cat "$RUN/status.json" 2>/dev/null || echo "no status yet"
  tail -c 600 "$LOGS/train.log" | tr '\r' '\n' | tail -3
  ls "$RUN"/*.safetensors 2>/dev/null | tail -3 || true
  ;;

pause)
  # pause S: wait until step S's training state is fully written, then stop the trainer, so
  # nothing is lost (it resumes from that state). For a mid-run Tier 1 (RUNBOOK_v9b section 6).
  S=${2:?usage: v9b_box.sh pause STEP}
  ST=$(printf "%s/%s-step%08d-state" "$RUN" "$RUN_NAME" "$S")
  log "waiting for $ST"
  until [ -d "$ST" ]; do sleep 60; done
  prev=-1; cur=$(du -sb "$ST" | cut -f1)
  while [ "$cur" != "$prev" ]; do sleep 30; prev=$cur; cur=$(du -sb "$ST" | cut -f1); done
  # The pattern names the trainer's script, never this shell's own command line.
  pgrep -f "src/musubi_tuner/wan_train_network.py" | xargs -r kill -INT
  sleep 30; pgrep -f "src/musubi_tuner/wan_train_network.py" | xargs -r kill -9 || true
  log "trainer stopped after step $S (state $ST); resume with: bash finetune/wan/v9b_box.sh train"
  ;;

tier1)
  shift
  cd "$FW/eval_v9b"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  CONFIGS=""; for s in "$@"; do CONFIGS="$CONFIGS v9b@$s"; done
  "$PY" tier1_v9b.py --configs $CONFIGS 2>&1 | tee -a "$LOGS/tier1.log"
  "$PY" gate_v9b.py 2>&1 | tee -a "$LOGS/tier1.log" || true
  ;;

eval)
  grep -aq "== done:" "$LOGS/train.log" || { echo "training has not finished"; exit 1; }
  cd "$FW/eval_v9b"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  CONFIGS=""; for s in $STEPS_EVAL; do CONFIGS="$CONFIGS v9b@$s"; done
  "$PY" tier1_v9b.py --configs v8merged $CONFIGS 2>&1 | tee "$LOGS/eval_tier1.log"
  if ! "$PY" gate_v9b.py 2>&1 | tee "$LOGS/eval_gate.log"; then
    log "NO CHECKPOINT PASSES THE TIER-1 GATE - stage not kept; see RUNBOOK_v9b section 7"; exit 1
  fi
  WIN=$(tail -1 "$LOGS/eval_gate.log")
  log "Tier 2 scene suite on v8merged and $WIN"
  "$PY" suite_v9b.py --configs v8merged "$WIN" 2>&1 | tee "$LOGS/eval_suite.log"
  log "eval done: $EVAL/runs/$RUN_NAME"
  ;;

archive)
  "$PY" "$FW/azure_sync.py" push --prefix "v9b/weights/$RUN_NAME" "$RUN"/*.safetensors
  "$PY" "$FW/azure_sync.py" push --prefix "v9b/logs/$RUN_NAME" --tree "$RUN/logs" || true
  "$PY" "$FW/azure_sync.py" push --prefix v9b/logs/box --tree "$LOGS"
  "$PY" "$FW/azure_sync.py" push --prefix v9b/eval --tree "$EVAL" --exclude /cache/ /.raw_
  SET=$(readlink -f "$DATA_LINK")
  "$PY" "$FW/azure_sync.py" push --prefix "processed/$(basename "$SET")" --tree "$SET" --exclude /cache/
  bash "$0" sync
  log "archived to Azure pudgy/v9b/ and processed/$(basename "$SET")"
  ;;

*)
  sed -n '2,24p' "$0"; exit 1 ;;
esac
