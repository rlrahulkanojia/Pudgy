#!/bin/bash
# =============================================================================
# v9c-phone on a fresh GPU box (A100 80 GB), end to end. One entry point, one step at a
# time, every step idempotent. Runbook: training_approach/v9/RUNBOOK_v9c_phone.md.
#
#   bash finetune/wan/v9_box.sh setup      # venv + musubi 0.3.4 + 65 GB base weights
#   bash finetune/wan/v9_box.sh restore    # raw data + v8 weights from Azure
#   bash finetune/wan/v9_box.sh merge      # pudgy-v8-merged = v7fixed + v8@550 (one file)
#   bash finetune/wan/v9_box.sh prep       # registry lint + prep_v9 -> /workspace/data_v9c
#   bash finetune/wan/v9_box.sh cache      # latents + T5 for the training set, verified
#   bash finetune/wan/v9_box.sh evalprep   # eval clip set + keyframes, latents cached
#   bash finetune/wan/v9_box.sh baseline   # Tier-1 on v8merged and v8@550 (merge check)
#   bash finetune/wan/v9_box.sh smoke      # 2 steps on the f61 bucket only: memory, s/clip
#   bash finetune/wan/v9_box.sh train      # the run (background) + Azure monitor
#   bash finetune/wan/v9_box.sh status     # where the run is
#   bash finetune/wan/v9_box.sh eval       # after training: Tier-1 per checkpoint, gate, Tier-2
#   bash finetune/wan/v9_box.sh archive    # everything that must survive the box -> Azure v9c/
# =============================================================================
set -euo pipefail
W=/workspace
REPO=$W/Pudgy
FW=$REPO/finetune/wan
PY=$REPO/.venv-wan/bin/python
M=$W/wan_models
RUN_NAME="${RUN_NAME:-pudgy-v9c-phone-lownoise}"   # override for a re-run, e.g. pudgy-v9c-phone-r35
export RUN_NAME
RUN=$W/wan_output/$RUN_NAME
MERGED=$W/wan_output/v8_merged/pudgy-v8-merged.safetensors
DATA_LINK=$W/data_v9c
CFG=$DATA_LINK/dataset_config_v9c_phone.workspace.toml
EVAL=$W/eval_v9
LOGS=$W/logs_v9c
STEPS_EVAL="${STEPS_EVAL:-100 125 150 175 200}"
export PUDGY_DATA=$W/Data
mkdir -p "$LOGS"
ts() { date -u +%FT%TZ; }
log() { echo "== [$(ts)] $*" | tee -a "$LOGS/v9_box.log"; }

musubi() {   # musubi's own cache scripts, in the trainer venv
  ( source "$REPO/.venv-wan/bin/activate" && cd "$W/musubi-tuner" && "$@" )
}

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
setup)
  [ -f "$REPO/.env" ] || { echo "copy the .env (Azure + optional W&B keys) to $REPO/.env first"; exit 1; }
  bash "$REPO/setup_wan_env.sh" 2>&1 | tee "$LOGS/setup.log"
  ;;

restore)
  log "raw iteration_3-6 (iteration_6: the extracted Samples/, not the zips)"
  "$PY" "$FW/azure_sync.py" pull --dest "$W/Data" raw/iteration_3/ raw/iteration_4/ \
    raw/iteration_5/ raw/iteration_6/Samples/ raw/iteration_6/CHANGELOG.md
  log "v8 base, v8@550, v2 goldens"
  "$PY" "$FW/azure_sync.py" pull --dest "$W/restore" v8/base/pudgy-v7-fixed-combined.safetensors \
    v8/weights/pudgy-v8-joint-lownoise/pudgy-v8-joint-lownoise-step00000550.safetensors \
    v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors \
    v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors
  # The paths eval_v8/common.py (and so eval_v9) expect.
  mkdir -p "$W/wan_output/v7_baseline" "$W/wan_output/pudgy-v8-joint-lownoise" "$W/wan_output/v2_golden"
  ln -sf "$W/restore/v8/base/pudgy-v7-fixed-combined.safetensors" "$W/wan_output/v7_baseline/"
  ln -sf "$W/restore/v8/weights/pudgy-v8-joint-lownoise/pudgy-v8-joint-lownoise-step00000550.safetensors" \
    "$W/wan_output/pudgy-v8-joint-lownoise/"
  ln -sf "$W/restore/v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors" \
    "$W/restore/v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors" "$W/wan_output/v2_golden/"
  ls -la "$W/wan_output"/*/ | tee -a "$LOGS/v9_box.log"
  ;;

merge)
  if [ -s "$MERGED" ]; then log "already merged: $MERGED"; exit 0; fi
  mkdir -p "$(dirname "$MERGED")"
  "$PY" "$FW/eval_v8/lora_tools_v8.py" combine "$MERGED" \
    --part "$W/wan_output/v7_baseline/pudgy-v7-fixed-combined.safetensors" 1 \
    --part "$W/wan_output/pudgy-v8-joint-lownoise/pudgy-v8-joint-lownoise-step00000550.safetensors" 1 \
    2>&1 | tee "$LOGS/merge.log"
  ls -la "$MERGED"
  ;;

prep)
  if [ -s "$DATA_LINK/manifest.json" ]; then log "already built: $(readlink -f "$DATA_LINK")"; exit 0; fi
  "$PY" "$FW/assets/registry.py" lint --data "$W/Data"
  "$PY" "$FW/prep_v9.py" --workers "$(nproc)" --out "$W/Data/processed" 2>&1 | tee "$LOGS/prep.log"
  SET=$(ls -d "$W/Data/processed"/v9c_phone_[0-9]* | tail -1)
  ln -sfn "$SET" "$DATA_LINK"
  log "training set: $SET -> $DATA_LINK"
  ;;

cache)
  [ -s "$CFG" ] || { echo "no $CFG - run prep first"; exit 1; }
  cache_cfg "$CFG" 2>&1 | tee "$LOGS/cache.log"
  "$PY" "$FW/verify_cache_v8.py" --config "$CFG" 2>&1 | tee -a "$LOGS/cache.log"
  ;;

evalprep)
  "$PY" "$FW/eval_v9/build_assets_v9.py" 2>&1 | tee "$LOGS/evalprep.log"
  cache_cfg "$EVAL/holdout/holdout.toml" latents-only 2>&1 | tee -a "$LOGS/evalprep.log"
  ;;

baseline)
  # Tier-1 on the base twice: as ONE merged file and as the two files v8 shipped. Losses
  # must agree to ~1e-3 relative, else the merge is wrong and nothing downstream is valid.
  cd "$FW/eval_v9"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$PY" dcls_v9.py --configs v8merged v8@550 --k 2 2>&1 | tee "$LOGS/baseline.log"
  "$PY" - <<PYEOF 2>&1 | tee -a "$LOGS/baseline.log"
import json, numpy as np
a = json.load(open("$EVAL/runs/$RUN_NAME/dcls/v8merged_k2.json")); b = json.load(open("$EVAL/runs/$RUN_NAME/dcls/v8@550_k2.json"))
la = {r["clip"]: np.mean(r["loss_true"]) for r in a["rows"]}
rel = [abs(np.mean(r["loss_true"]) - la[r["clip"]]) / la[r["clip"]] for r in b["rows"]]
print(f"merge check: max relative loss difference {max(rel):.2e} over {len(rel)} clips ->",
      "OK" if max(rel) < 1e-3 else "MISMATCH - do not train on this base")
PYEOF
  ;;

smoke)
  # The longest bucket only, 2 optimizer steps of 1 clip: proves memory and speed at f61
  # before committing ~20 h. Watch for allocator retries (v8: swap 16 thrashed at f57).
  [ -s "$CFG" ] || { echo "no $CFG - run prep + cache first"; exit 1; }
  "$PY" - "$CFG" "$DATA_LINK/smoke_f61.workspace.toml" <<'PYEOF'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
head, *blocks = text.split("[[datasets]]")
top = max(blocks, key=lambda b: int(re.search(r"target_frames = \[(\d+)\]", b).group(1)))
open(dst, "w").write(head + "[[datasets]]" + top)
print("smoke bucket:", re.search(r"target_frames = \[(\d+)\]", top).group(1))
PYEOF
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$LOGS/smoke_mem.csv" &
  SMI=$!
  MAX_STEPS=2 SAVE_EVERY_STEPS=2 GRAD_ACCUM=1 NAME=pudgy-v9c-smoke \
    DATASET="$DATA_LINK/smoke_f61.workspace.toml" bash "$FW/train_pudgy_v9.sh" 2>&1 | tee "$LOGS/smoke.log" || true
  kill $SMI 2>/dev/null || true
  echo "peak GPU memory: $(sort -n "$LOGS/smoke_mem.csv" | tail -1) MiB"
  echo "allocator retry lines: $(grep -aci 'retr' "$LOGS/smoke.log" || true)"
  grep -aE "s/it|it/s" "$LOGS/smoke.log" | tail -2 || true
  grep -aq "== done:" "$LOGS/smoke.log" && echo "SMOKE OK" || { echo "SMOKE FAILED - see $LOGS/smoke.log"; exit 1; }
  ;;

train)
  [ -s "$MERGED" ] || { echo "no merged base - run merge"; exit 1; }
  LATEST=$(ls -d "$RUN"/*-step*-state 2>/dev/null | sort | tail -1 || true)
  if [ -n "$LATEST" ]; then log "resuming from $LATEST"; export RESUME="$LATEST"; fi
  NAME="$RUN_NAME" nohup bash "$FW/train_pudgy_v9.sh" >> "$LOGS/train.log" 2>&1 &
  echo $! > "$LOGS/train.pid"
  # Status + Azure mirror every 30 min (weights, logs) - the box is not persistent.
  V8_RUN="$RUN" V8_LOG="$LOGS/train.log" MONITOR_PREFIX=v9c MONITOR_EVAL="$EVAL" \
    nohup "$PY" "$FW/monitor_v8.py" >> "$LOGS/monitor.log" 2>&1 &
  echo $! > "$LOGS/monitor.pid"
  log "training pid $(cat "$LOGS/train.pid"), monitor pid $(cat "$LOGS/monitor.pid"); tail -f $LOGS/train.log"
  ;;

status)
  cat "$RUN/status.json" 2>/dev/null || echo "no status yet"
  tail -c 600 "$LOGS/train.log" | tr '\r' '\n' | tail -3
  ls "$RUN"/*.safetensors 2>/dev/null | tail -3 || true
  ;;

eval)
  grep -aq "== done:" "$LOGS/train.log" || { echo "training has not finished (no '== done:' in $LOGS/train.log)"; exit 1; }
  cd "$FW/eval_v9"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  CONFIGS=""; for s in $STEPS_EVAL; do CONFIGS="$CONFIGS v9c@$s"; done
  log "Tier 1: dcls on$CONFIGS"
  "$PY" dcls_v9.py --configs v8merged $CONFIGS --k 2 2>&1 | tee "$LOGS/eval_dcls.log"
  log "gate"
  if ! "$PY" gate_v9.py --steps $STEPS_EVAL 2>&1 | tee "$LOGS/eval_gate.log"; then
    log "NO CHECKPOINT PASSES - stage not kept; see RUNBOOK (re-run with --replay-frac 0.35)"; exit 1
  fi
  WIN=$(tail -1 "$LOGS/eval_gate.log")
  log "Tier 2 suites on v8merged and $WIN"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE         # CLIP is fetched once for scoring
  "$PY" suite_v9.py --suite all --configs v8merged "$WIN" 2>&1 | tee "$LOGS/eval_suites.log"
  log "eval done: $EVAL/runs/$RUN_NAME (winner_v9c.json, dcls/, suites/*/scores.json + sheets)"
  ;;

archive)
  "$PY" "$FW/azure_sync.py" push --prefix "v9c/weights/$RUN_NAME" "$RUN"/*.safetensors
  "$PY" "$FW/azure_sync.py" push --prefix v9c/base "$MERGED"
  "$PY" "$FW/azure_sync.py" push --prefix "v9c/logs/$RUN_NAME" --tree "$RUN/logs" || true
  "$PY" "$FW/azure_sync.py" push --prefix v9c/logs/box --tree "$LOGS"
  "$PY" "$FW/azure_sync.py" push --prefix v9c/eval --tree "$EVAL" --exclude /cache/ /.raw_
  SET=$(readlink -f "$DATA_LINK")
  "$PY" "$FW/azure_sync.py" push --prefix "processed/$(basename "$SET")" --tree "$SET" --exclude /cache/
  log "archived to Azure pudgy/v9c/ and processed/$(basename "$SET")"
  ;;

*)
  sed -n '2,20p' "$0"; exit 1 ;;
esac
