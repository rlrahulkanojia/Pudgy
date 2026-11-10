#!/bin/bash
# =============================================================================
# v10 Stage 1 (high-noise LoRA) on a 2-GPU box, end to end. One entry point, one step at a time,
# every step idempotent. Guide: training_approach/v10/HANDOVER_v10.md (read it first).
# GPU 0 trains; GPU 1 evaluates. Decisions: Training_Approach_v10.md section 0 (Q1-Q20).
#
#   bash finetune/wan/v10_box.sh setup      # venv, musubi v0.3.4 + patches, base weights (v9_box.sh setup) + eval deps
#   bash finetune/wan/v10_box.sh pull       # Azure: merged base, golden, E5, training set, eval sets, keyframes; links + checks
#   bash finetune/wan/v10_box.sh pull_v9b   # Azure: v9b@100/200 (Phase 0 #1 only; big, run in the background)
#   bash finetune/wan/v10_box.sh cache      # latents + T5 for the training set, verified, then the replay/phone marks
#   bash finetune/wan/v10_box.sh evalprep   # latents for the held-out Tier-1 windows
#   bash finetune/wan/v10_box.sh smoke      # 2 steps on the heaviest bucket, preservation on (GPU 0)
#   bash finetune/wan/v10_box.sh train      # Stage 1 in the background + Azure monitor (GPU 0)
#   bash finetune/wan/v10_box.sh gpu1       # GPU 1 scheduler: Tier-1 base, Tier 1 per checkpoint (+ early stop),
#                                           #   baseline generation, Phase 0 #1; run in the background
#   bash finetune/wan/v10_box.sh solo_watch # ONE-GPU box: solo + retries of failed stages (run THIS in tmux)
#   bash finetune/wan/v10_box.sh solo       # ONE-GPU box: the whole sequence on GPU 0, resumable
#   bash finetune/wan/v10_box.sh status     # training + GPU 1 state
#   bash finetune/wan/v10_box.sh pause S    # stop the trainer once step S's state is saved
#   bash finetune/wan/v10_box.sh candidate  # after training + gpu1: pick the candidate, run G2 / G4 / G5 on it
#   bash finetune/wan/v10_box.sh phase0     # finish Phase 0 #1 (EVAL_GPU=0 once G5 has freed GPU 0)
#   bash finetune/wan/v10_box.sh g1         # build the blind G1 pair page -> Azure v10/eval/g1 for the user
#   bash finetune/wan/v10_box.sh sync       # code, docs, logs -> Azure v10/work, v10/logs/box (any time)
#   bash finetune/wan/v10_box.sh archive    # weights, logs, eval -> Azure v10/
#   bash finetune/wan/v10_box.sh git        # commit, push to branch v10/box-run, bundle to Azure v10/git (Q7)
# =============================================================================
set -euo pipefail
W="${V10_WORKSPACE:-/workspace}"     # overridden only by tests
REPO=$W/Pudgy
FW=$REPO/finetune/wan
PY=$REPO/.venv-wan/bin/python
M=$W/wan_models
RUN_NAME="${RUN_NAME:-pudgy-v10-highnoise}"
export RUN_NAME
RUN=$W/wan_output/$RUN_NAME
SET_NAME="${SET_NAME:-v10_1525}"
MERGED=$W/wan_output/v8_merged/pudgy-v8-merged.safetensors
GOLDEN=$W/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors
E5=$W/wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors
DATA_LINK=$W/data_v10
CFG=$DATA_LINK/dataset_config_v10.workspace.toml
EVAL=$W/eval_v10
R=$EVAL/runs/$RUN_NAME          # every v10 eval output
E10=$FW/eval_v10
GEN_CHUNK="${GEN_CHUNK:-10}"     # clips per generation slice on GPU 1 (each slice reloads the model, ~5-15 min)
LOGS=$W/logs_v10
TRAIN_GPU="${TRAIN_GPU:-0}"
# GPU 1 evaluates on a 2-GPU box; on a one-GPU box everything is on GPU 0 (use the `solo` step there).
EVAL_GPU="${EVAL_GPU:-$( [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 2 ] && echo 1 || echo 0 )}"
TIER1_EVERY=50
MAX_STEPS="${MAX_STEPS:-300}"
export PUDGY_DATA=$W/Data
mkdir -p "$LOGS"
ts() { date -u +%FT%TZ; }
log() { echo "== [$(ts)] $*" | tee -a "$LOGS/v10_box.log"; }
musubi() { ( source "$REPO/.venv-wan/bin/activate" && cd "$W/musubi-tuner" && "$@" ); }
az() { "$PY" "$FW/azure_sync.py" "$@"; }
evalenv() { export CUDA_VISIBLE_DEVICES=$EVAL_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; }
# Matches the launcher, the trainer and its data-loader workers of THIS run, never a smoke run or this shell.
trainer_pids() { pgrep -f "asset_embed_v9\.py train .*--output_name ${RUN_NAME}( |\$)" || true; }

cache_cfg() {   # $1 = dataset toml; latents (+ T5 unless $2 = latents-only)
  musubi python src/musubi_tuner/wan_cache_latents.py --dataset_config "$1" \
    --vae "$M/comfy21/split_files/vae/wan_2.1_vae.safetensors" --i2v --skip_existing --num_workers 16
  if [ "${2:-}" != "latents-only" ]; then
    musubi python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "$1" \
      --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" --batch_size 16 --skip_existing --num_workers 16
  fi
}

smoke_gate() {   # $1 = smoke log, $2 = exit code. Gate lines from the training-side build (2026-11-10).
  local L=$1 RC=$2 OOM ST PRES ASSET
  OOM=$(grep -ac "memory mapping failed with OOM" "$L" || true)
  ST=$(tr '\r' '\n' < "$L" | grep -a "PRESERVE_STATS" | tail -1)
  PRES=$(echo "$ST" | sed -nE 's/.* preserve=([0-9]+) .*/\1/p'); ASSET=$(echo "$ST" | sed -nE 's/.* asset=([0-9]+) .*/\1/p')
  echo "exit $RC; mapping-OOM lines $OOM (must be 0); $ST"
  if [ "$RC" -eq 0 ] && grep -aq "== done:" "$L" && [ "${OOM:-1}" -eq 0 ] && [ "${PRES:-0}" -gt 0 ] && [ "${ASSET:-0}" -gt 0 ] \
     && grep -aqE "\[asset\] base weights: ([0-9]+)/\1 LoRA modules match" "$L"; then
    echo "SMOKE OK"
  else
    echo "SMOKE FAILED - see $L. If the mapping-OOM count is > 0: BLOCKS_TO_SWAP=26 bash $0 smoke (then 28), and train with that value"
    return 1
  fi
}

# ---- eval hooks (finetune/wan/eval_v10; exit codes in each script's docstring) ----
ev() { ( cd "$E10" && RUN_NAME="$RUN_NAME" V10_EVAL_GPU="$EVAL_GPU" "$PY" "$@" ); }
eval_tier1_base_done() { [ -s "$R/tier1/v8merged_k2.json" ]; }
eval_tier1_base() { log "Tier 1 base rows (v8merged + golden, high-noise)"; ev tier1_v10.py score --configs v8merged; }
eval_tier1_done() { [ -s "$R/tier1/gate_step$1.json" ]; }
eval_tier1() {   # Tier 1 on v10@S: 0 holds, 10 does not hold (training goes on), 20 early stop, 2 missing input
  log "Tier 1 on v10@$1"
  set +e; ev tier1_v10.py run --step "$1"; T1RC=$?; set -e
  case $T1RC in
    0)  log "Tier 1 step $1: holds" ;;
    10) log "Tier 1 step $1: does not hold (training continues; see $R/tier1/gate_step$1.json)" ;;
    20) echo "Tier 1 early stop at step $1 ($R/tier1/gate_step$1.json)" > "$LOGS/EARLY_STOP" ;;
    *)  log "Tier 1 step $1: missing input (exit $T1RC) - stopping the scheduler"; exit 1 ;;
  esac
}
eval_early_stop() { [ "${T1RC:-0}" -ne 20 ]; }      # false (1) when the early-stop rule fired
gen_chunk() {   # $1 = phase name; rest = command. 0 = did work (or just finished), 1 = already complete
  local flag="$LOGS/$1.done" out; shift
  [ -f "$flag" ] && return 1
  out=$(ev "$@" --max-new "$GEN_CHUNK" 2>&1 | tee -a "$LOGS/gpu1_gen.log" | tail -1)
  case "$out" in
    "REMAINING 0") touch "$flag"; log "$* complete" ;;
    REMAINING*)    : ;;
    *)             log "generation failed: $* (see $LOGS/gpu1_gen.log)"; exit 1 ;;
  esac
  return 0
}
eval_baseline_chunk() { gen_chunk phase0_p2 phase0_v10.py p2; }   # Phase 0 #2: v8merged side of G1/G2 (+FLF) and G4
eval_phase0_chunk() { gen_chunk phase0_p1 phase0_v10.py p1; }     # Phase 0 #1: v9b@100/200 vs v8merged

eval_candidate() {   # plan section 5.1: the latest Tier-1 checkpoint; full G2/G4/G5; on a FAIL, the previous one
  local before="${CANDIDATE_BEFORE:-}" C S RC G2 G4 G5 G5P   # CANDIDATE_BEFORE=S: skip S and later (an eye-check rejection)
  [ -f "$LOGS/phase0_p2.done" ] || { echo "the baseline (Phase 0 #2) is not complete - the gpu1 scheduler must finish first"; return 2; }
  while true; do
    set +e
    if [ -z "$before" ]; then ev tier1_v10.py summary > "$LOGS/summary.out" 2>&1; else ev tier1_v10.py summary --before "$before" > "$LOGS/summary.out" 2>&1; fi
    RC=$?; set -e
    cat "$LOGS/summary.out"
    if [ $RC -eq 30 ]; then log "NO CHECKPOINT HOLDS TIER 1 - Stage 1 is not kept; write the report (HANDOVER_v10 Step 12)"; return 30; fi
    C=$(tail -1 "$LOGS/summary.out"); S=${C#v10@}
    if [ "$TRAIN_GPU" = "$EVAL_GPU" ]; then        # one GPU: one job at a time
      log "candidate $C: G5, G2, G4 in turn on GPU $EVAL_GPU"
      ( cd "$E10" && CUDA_VISIBLE_DEVICES=$TRAIN_GPU RUN_NAME="$RUN_NAME" "$PY" g5_v10.py gen --config "$C" ) > "$LOGS/g5_$S.log" 2>&1 \
        || { log "G5 generation failed - see $LOGS/g5_$S.log"; return 2; }
      ev suite_v10.py gen --config "$C"
      ev g4_v10.py gen --config "$C"
    else
      log "candidate $C: G5 on GPU $TRAIN_GPU, G2 + G4 on GPU $EVAL_GPU"
      ( cd "$E10" && CUDA_VISIBLE_DEVICES=$TRAIN_GPU RUN_NAME="$RUN_NAME" "$PY" g5_v10.py gen --config "$C" ) > "$LOGS/g5_$S.log" 2>&1 &
      G5P=$!
      ev suite_v10.py gen --config "$C"
      ev g4_v10.py gen --config "$C"
      wait $G5P || { log "G5 generation failed - see $LOGS/g5_$S.log"; return 2; }
    fi
    ev suite_v10.py score --configs v8merged "$C"
    ev suite_v10.py sheets --configs v8merged "$C"
    ev g4_v10.py score --configs v8merged "$C"
    set +e
    ev suite_v10.py gate --cand "$C"; G2=$?
    ev g4_v10.py gate --cand "$C"; G4=$?
    ( cd "$E10" && CUDA_VISIBLE_DEVICES=$TRAIN_GPU RUN_NAME="$RUN_NAME" "$PY" g5_v10.py gate --config "$C" ); G5=$?
    set -e
    echo "$(ts) $C G2=$G2 G4=$G4 G5=$G5   (0 pass, 10 fail, 11 review, 2 missing)" | tee -a "$LOGS/candidate_verdicts.txt"
    if [ $G2 -eq 2 ] || [ $G4 -eq 2 ] || [ $G5 -eq 2 ]; then log "$C: missing input - fix and re-run candidate"; return 2; fi
    if [ $G2 -eq 10 ] || [ $G4 -eq 10 ] || [ $G5 -eq 10 ]; then log "$C fails a guardrail; trying the previous Tier-1 checkpoint"; before=$S; continue; fi
    echo "$C" > "$LOGS/CANDIDATE"
    if [ $G2 -eq 11 ]; then log "$C passes G4 and G5; G2 needs an eye check of the listed clips (HANDOVER_v10 Step 10), then: v10_box.sh g1"; return 11; fi
    log "$C passes G2, G4 and G5; next: v10_box.sh g1"; return 0
  done
}

eval_g1() {
  local C N
  C=$(cat "$LOGS/CANDIDATE" 2>/dev/null) || { echo "no candidate yet - run: v10_box.sh candidate"; return 1; }
  ev g1_pairs_v10.py build --cand "$C"
  N="${C/@/_}_vs_v8merged"
  az push --prefix "v10/eval/g1/pages/$N" --tree "$R/g1/pages/$N"
  az push --prefix v10/eval/g1/keys "$R/g1/pages/$N.key.json"
  log "G1 page for $C -> Azure v10/eval/g1/pages/$N (the key is separate: v10/eval/g1/keys). Tell the user it is ready to judge."
}

step=${1:-}
case "$step" in
setup)
  bash "$FW/v9_box.sh" setup
  # eval extras: OCR for G2 (no text absent from frame 0); DINOv2 for G2 identity comes through transformers
  "$REPO/.venv-wan/bin/pip" install -q rapidocr_onnxruntime
  ev common_v10.py prefetch            # umT5 tokenizer, DINOv2-base, CLIP-L/14 into the HF cache (needs network once)
  log "setup done"
  ;;

pull)
  # Azure -> box is slow (0.2-0.4 MB/s measured 2026-10-27). These are the small, needed-first items (~2.5 GB).
  az pull --dest "$W/restore" v9b/base/pudgy-v8-merged.safetensors \
     v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors \
     v9c2/weights/pudgy-v9c2-asset-dual/pudgy-v9c2-asset-dual-step00000150.asset.safetensors
  az pull --dest "$W/Data" "processed/$SET_NAME/"
  az pull --dest "$W/Data" processed/v10_eval/
  az pull --dest "$W" v9b/handover/eval_for_v9b.tar        # the v9 regression set with latents -> /workspace/eval_v9/holdout
  mkdir -p "$W/wan_output/v8_merged" "$W/wan_output/v2_golden" "$W/wan_output/e5"
  ln -sf "$W/restore/v9b/base/pudgy-v8-merged.safetensors" "$MERGED"
  ln -sf "$W/restore/v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors" "$GOLDEN"
  ln -sf "$W/restore/v9c2/weights/pudgy-v9c2-asset-dual/pudgy-v9c2-asset-dual-step00000150.asset.safetensors" "$E5"
  ln -sfn "$W/Data/processed/$SET_NAME" "$DATA_LINK"
  ln -sfn "$W/Data/processed/v10_eval" "$EVAL"
  [ -d "$W/eval_v9/holdout" ] || tar -xf "$W/v9b/handover/eval_for_v9b.tar" -C "$W"
  echo "merged base md5 (want ec333748284410bb341f431b88f5026f): $(md5sum "$MERGED" | cut -d' ' -f1)"
  ( cd "$DATA_LINK" && sha256sum -c --quiet SHA256SUMS ) && echo "training set: SHA256SUMS all match"
  ( cd "$EVAL" && sha256sum -c --quiet SHA256SUMS ) && echo "eval set: SHA256SUMS all match"
  echo "clips: $(ls "$DATA_LINK/clips" | wc -l) (want 1525)"
  ev common_v10.py configs v8merged                          # every weight file for the base config resolves
  log "pull done"
  ;;

pull_v9b)
  az pull --dest "$W/restore" v9b/weights/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000100.safetensors \
     v9b/weights/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000200.safetensors
  mkdir -p "$W/wan_output/pudgy-v9b-lownoise"
  ln -sf "$W/restore/v9b/weights/pudgy-v9b-lownoise/"*.safetensors "$W/wan_output/pudgy-v9b-lownoise/"
  log "v9b checkpoints for Phase 0 #1 in place"
  ;;

cache)
  [ -s "$CFG" ] || { echo "no $CFG - run pull first"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$TRAIN_GPU
  cache_cfg "$CFG" 2>&1 | tee -a "$LOGS/cache.log"
  # Self-healing after an interrupted run: delete any file cut off mid-write, re-encode it, then verify strictly.
  "$PY" "$FW/verify_cache_v8.py" --config "$CFG" --fix 2>&1 | tee -a "$LOGS/cache.log" || true
  cache_cfg "$CFG" 2>&1 | tee -a "$LOGS/cache.log"
  "$PY" "$FW/verify_cache_v8.py" --config "$CFG" 2>&1 | tee -a "$LOGS/cache.log"
  # Spatial size: scene latents 480x848 / 8 = 60 x 106 (no silent re-bucketing), replay 1024 / 8 = 128 x 128.
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
  # Preservation marker on replay text-encoder files + the phone-phrase span for E5 (Q10, Q17).
  # (needs the umT5 tokenizer from the T5 step's HF cache; uses GPU 0 only for a moment)
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU "$PY" "$FW/cache_mark_v10.py" mark --config "$CFG" --asset "$E5" \
    --cache-root "$W/wan_cache/latents_v10" 2>&1 | tee -a "$LOGS/cache.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { log "cache_mark_v10 FAILED - see $LOGS/cache.log"; exit 1; }
  grep -a "CACHE_MARK_CHECK OK" "$LOGS/cache.log" | tail -1
  log "cache done"
  ;;

evalprep)
  export CUDA_VISIBLE_DEVICES=$EVAL_GPU
  cache_cfg "$EVAL/scenes/scenes_f33.toml" latents-only 2>&1 | tee "$LOGS/evalprep.log"
  log "evalprep done"
  ;;

smoke)
  # SMOKE=1 builds a 3-sample set (the heaviest replay clip, the heaviest phone window, the heaviest plain
  # window), runs 3 steps x 1 sample at the real settings, never resumes, and saves at step 3.
  nvidia-smi -i "$TRAIN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$LOGS/smoke_mem.csv" &
  SMI=$!
  set +e
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU SMOKE=1 BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-24}" bash "$FW/train_pudgy_v10.sh" > "$LOGS/smoke.log" 2>&1
  RC=$?
  set -e
  kill $SMI 2>/dev/null || true
  echo "peak GPU memory (the allocator's reservation; not the gate): $(sort -n "$LOGS/smoke_mem.csv" | tail -1) MiB"
  grep -aE "s/it|it/s" "$LOGS/smoke.log" | tail -2 || true
  smoke_gate "$LOGS/smoke.log" "$RC"
  ;;

train)
  [ -s "$MERGED" ] && [ -s "$GOLDEN" ] && [ -s "$E5" ] || { echo "missing base, golden or E5 - run pull"; exit 1; }
  [ -n "$(trainer_pids)" ] && { echo "a trainer is already running: $(trainer_pids)"; exit 1; }
  # train_pudgy_v10.sh resumes by itself from the newest COMPLETE *-step*-state (never a half-written one).
  unset RESUME
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU NAME="$RUN_NAME" MAX_STEPS="$MAX_STEPS" nohup bash "$FW/train_pudgy_v10.sh" >> "$LOGS/train.log" 2>&1 &
  echo $! > "$LOGS/train.pid"
  if ! { [ -s "$LOGS/monitor.pid" ] && kill -0 "$(cat "$LOGS/monitor.pid")" 2>/dev/null; }; then
    V8_RUN="$RUN" V8_LOG="$LOGS/train.log" MONITOR_PREFIX=v10 MONITOR_EVAL="$EVAL" \
      nohup "$PY" "$FW/monitor_v8.py" >> "$LOGS/monitor.log" 2>&1 &
    echo $! > "$LOGS/monitor.pid"
  fi
  log "training pid $(cat "$LOGS/train.pid"), monitor pid $(cat "$LOGS/monitor.pid"); tail -f $LOGS/train.log"
  ;;

status)
  cat "$RUN/status.json" 2>/dev/null || echo "no training status yet"
  tail -c 600 "$LOGS/train.log" 2>/dev/null | tr '\r' '\n' | tail -3 || true
  ls "$RUN"/*.safetensors 2>/dev/null | tail -3 || true
  grep -a "preserve" "$LOGS/train.log" 2>/dev/null | tail -1 || true
  echo "--- GPU 1"; tail -5 "$LOGS/gpu1.log" 2>/dev/null || echo "gpu1 scheduler not started"
  ls "$R/tier1"/gate_step*.json 2>/dev/null | sed 's#.*/##' | tr '\n' ' '; echo
  [ -f "$LOGS/EARLY_STOP" ] && echo "EARLY STOP: $(cat "$LOGS/EARLY_STOP")"
  for f in phase0_p2 phase0_p1; do [ -f "$LOGS/$f.done" ] && echo "$f complete"; done
  [ -f "$LOGS/CANDIDATE" ] && echo "candidate: $(cat "$LOGS/CANDIDATE")"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ;;

pause)
  S=${2:?usage: v10_box.sh pause STEP}
  ST=$(printf "%s/%s-step%08d-state" "$RUN" "$RUN_NAME" "$S")
  log "waiting for $ST"
  # Give up if the trainer is gone and the state never appeared (a crash before step S), instead of waiting forever.
  until [ -d "$ST" ]; do
    sleep 60
    [ -d "$ST" ] && break
    [ -n "$(trainer_pids)" ] || { log "pause: the trainer is not running and $ST never appeared - see $LOGS/train.log"; exit 1; }
  done
  prev=-1; cur=$(du -sb "$ST" | cut -f1)
  while [ "$cur" != "$prev" ]; do sleep 30; prev=$cur; cur=$(du -sb "$ST" | cut -f1); done
  P=$(trainer_pids); [ -n "$P" ] && kill -INT $P
  sleep 30; P=$(trainer_pids); [ -n "$P" ] && kill -9 $P || true
  log "trainer stopped after step $S (state $ST); resume with: bash finetune/wan/v10_box.sh train"
  ;;

gpu1)
  # The GPU 1 scheduler (Q6, Q8, Q18). Priority each round: (1) Tier-1 rows for the base, (2) Tier 1 for any new
  # checkpoint at a multiple of $TIER1_EVERY, with the early-stop rule, (3) baseline generation (Phase 0 #2) in
  # small chunks, (4) Phase 0 #1 in small chunks. Generation chunks stop at clip boundaries (--max-new), so a new
  # checkpoint waits at most one chunk. Ends when training is done, every checkpoint is scored and the queue is empty.
  evalenv
  exec >> "$LOGS/gpu1.log" 2>&1
  echo $$ > "$LOGS/gpu1.pid"
  log "gpu1 scheduler started on GPU $EVAL_GPU"
  while true; do
    if ! eval_tier1_base_done; then eval_tier1_base; continue; fi
    NEXT=""
    for f in $(ls "$RUN"/"$RUN_NAME"-step*.safetensors 2>/dev/null | sort); do
      S=$((10#$(basename "$f" .safetensors | sed 's/.*-step//')))
      [ $((S % TIER1_EVERY)) -eq 0 ] || continue
      eval_tier1_done "$S" || { NEXT=$S; break; }
    done
    if [ -n "$NEXT" ]; then
      eval_tier1 "$NEXT"
      if ! eval_early_stop "$NEXT"; then
        echo "step $NEXT: $(cat "$LOGS/EARLY_STOP")"
        bash "$0" pause "$NEXT" || true
        log "EARLY STOP at step $NEXT - training stopped; report to the user (HANDOVER_v10 section 4, Step 9)"
      fi
      continue
    fi
    if eval_baseline_chunk; then continue; fi
    V9B_OK=0; [ -s "$W/wan_output/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000100.safetensors" ] \
      && [ -s "$W/wan_output/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000200.safetensors" ] && V9B_OK=1
    TRAINING_OVER=0; { grep -aq "== done:" "$LOGS/train.log" 2>/dev/null || [ -f "$LOGS/EARLY_STOP" ]; } && TRAINING_OVER=1
    # Phase 0 #1 only fills GPU 1 while training runs; once training is over GPU 1 belongs to the candidate gates.
    if [ $TRAINING_OVER -eq 0 ] && [ $V9B_OK -eq 1 ] && eval_phase0_chunk; then continue; fi
    if [ $TRAINING_OVER -eq 1 ]; then
      [ -f "$LOGS/phase0_p1.done" ] || log "gpu1: Phase 0 #1 is not complete$([ $V9B_OK -eq 0 ] && echo ' (the v9b@100/200 checkpoints are missing: v10_box.sh pull_v9b)'); finish it with: EVAL_GPU=0 v10_box.sh phase0 once GPU 0 is free (HANDOVER_v10 Step 10)"
      log "gpu1: training is over, every checkpoint is scored and the baseline is complete; GPU 1 is free"
      rm -f "$LOGS/gpu1.pid"; break
    fi
    sleep 300
  done
  ;;

solo)
  # ONE-GPU SEQUENCE (decided 2026-11-10: the box has a single A100 80 GB). Everything runs on GPU $TRAIN_GPU, one job
  # at a time. Resumable: re-running `solo` skips finished stages (flag files, Tier-1 results, existing clips) and the
  # trainer resumes from its newest complete state. Run it in tmux. Exit 20 = early stop, 11 = G2 eye check needed,
  # 30 = no checkpoint holds Tier 1; each is a stop to report (HANDOVER_v10 section 4b).
  export EVAL_GPU=$TRAIN_GPU
  POLL="${SOLO_POLL:-30}"
  # One GPU: no interleaving needed, so each generation phase runs in ONE model load instead of 10-clip slices.
  GEN_CHUNK="${GEN_CHUNK_SOLO:-1000000}"
  stage() { echo "== [$(ts)] SOLO $*" | tee -a "$LOGS/solo.log" "$LOGS/v10_box.log"; }
  wait_trainer_gone() { while [ -n "$(trainer_pids)" ]; do sleep "$POLL"; done; }
  state_of() { printf "%s/%s-step%08d-state" "$RUN" "$RUN_NAME" "$1"; }
  start_train() { [ -n "$(trainer_pids)" ] && { stage "trainer already running: $(trainer_pids | tr '\n' ' ')"; return 0; }; bash "$0" train; }
  phase0_p1() {   # Phase 0 #1 needs no decision: run it whenever the sequence would otherwise leave the GPU idle
    [ -f "$LOGS/phase0_p1.done" ] && return 0
    if [ -s "$W/wan_output/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000100.safetensors" ] && [ -s "$W/wan_output/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000200.safetensors" ]; then
      stage "Phase 0 #1"; while gen_chunk phase0_p1 phase0_v10.py p1; do :; done
      ev phase0_v10.py report 2>&1 | tee "$LOGS/phase0_p1_report.txt"
    else
      stage "Phase 0 #1 skipped: v9b@100/200 missing (v10_box.sh pull_v9b, then re-run solo)"
    fi
  }
  [ -f "$LOGS/cache.done" ]    || { stage "cache"; bash "$0" cache && touch "$LOGS/cache.done"; }
  [ -f "$LOGS/evalprep.done" ] || { stage "evalprep"; bash "$0" evalprep && touch "$LOGS/evalprep.done"; }
  [ -f "$LOGS/smoke.done" ]    || { stage "smoke"; bash "$0" smoke && touch "$LOGS/smoke.done"; }
  evalenv
  eval_tier1_base_done || { stage "Tier-1 base rows"; eval_tier1_base; }
  FINAL=$(printf "%s/%s-step%08d.safetensors" "$RUN" "$RUN_NAME" "$MAX_STEPS")
  # 1. train to step 100 (MAX_STEPS stays $MAX_STEPS, so the cosine schedule is the planned one), then the early stop
  if ! eval_tier1_done 100; then
    if [ ! -d "$(state_of 100)" ]; then stage "train to step 100"; start_train; bash "$0" pause 100; fi
    wait_trainer_gone
    for S in 50 100; do eval_tier1_done $S || { stage "Tier 1 step $S"; eval_tier1 $S; }; done
  fi
  if [ -f "$LOGS/EARLY_STOP" ]; then
    stage "EARLY STOP: $(cat "$LOGS/EARLY_STOP") - Stage 1 is not kept; Phase 0 #1 runs first (it informs the next decision)"
    bash "$0" archive; MSG="v10 box: early stop at step 100" bash "$0" git
    phase0_p1; bash "$0" archive; MSG="v10 box: early stop; Phase 0 #1 done" bash "$0" git; exit 20
  fi
  # 2. finish training, then Tier 1 on the remaining checkpoints
  if [ ! -s "$FINAL" ]; then
    stage "train to step $MAX_STEPS"; start_train; sleep 60; wait_trainer_gone
    [ -s "$FINAL" ] || { stage "training stopped before step $MAX_STEPS - see $LOGS/train.log; re-run solo to resume"; exit 1; }
  fi
  for S in $(seq 150 "$TIER1_EVERY" "$MAX_STEPS"); do eval_tier1_done $S || { stage "Tier 1 step $S"; eval_tier1 $S; }; done
  # 3. baseline (Phase 0 #2), then the candidate's G5 / G2 / G4
  [ -f "$LOGS/phase0_p2.done" ] || { stage "baseline (Phase 0 #2)"; while gen_chunk phase0_p2 phase0_v10.py p2; do :; done; }
  if [ ! -f "$LOGS/CANDIDATE" ]; then
    stage "candidate gates"
    set +e; eval_candidate 2>&1 | tee -a "$LOGS/candidate.log"; RC=${PIPESTATUS[0]}; set -e
    if [ $RC -eq 30 ]; then
      stage "no checkpoint holds Tier 1 - Stage 1 is not kept; Phase 0 #1 runs first"
      bash "$0" archive; MSG="v10 box: no checkpoint holds Tier 1" bash "$0" git; phase0_p1; bash "$0" archive; exit 30
    fi
    # 11 = G2 needs a human eye check: keep going (G1 page, Phase 0 #1) so the GPU is not idle while it waits
    if [ $RC -eq 11 ]; then stage "G2 needs an eye check (HANDOVER_v10 Step 10); continuing meanwhile"; touch "$LOGS/G2_EYE_CHECK_PENDING"
    elif [ $RC -ne 0 ]; then stage "candidate exit $RC"; exit $RC; fi
  fi
  # 4. the G1 page for the user, then Phase 0 #1 while they judge
  [ -f "$LOGS/g1.done" ] || { stage "G1 page"; eval_g1 && touch "$LOGS/g1.done"; }
  bash "$0" archive; MSG="v10 box: Stage 1 done, candidate $(cat "$LOGS/CANDIDATE"), G1 page ready" bash "$0" git
  phase0_p1
  bash "$0" archive; MSG="v10 box: solo sequence complete" bash "$0" git
  if [ -f "$LOGS/G2_EYE_CHECK_PENDING" ]; then
    stage "WAITING FOR THE G2 EYE CHECK of $(cat "$LOGS/CANDIDATE") (HANDOVER_v10 section 4b); the G1 page is built but must not be judged until it passes"
    exit 11
  fi
  stage "SOLO DONE: candidate $(cat "$LOGS/CANDIDATE"); G1 page on Azure v10/eval/g1/pages/; write the report (Step 12)"
  ;;

solo_watch)
  # solo + a watchdog: a failed stage (exit 1: a crash, an OOM, a network blip) is retried after 2 min, up to
  # SOLO_RETRIES times (solo is resumable). Decision stops (0, 20, 11, 30) are never retried. Run THIS in tmux.
  tries=0; max="${SOLO_RETRIES:-3}"
  while true; do
    set +e; bash "$0" solo; RC=$?; set -e
    case $RC in
      0|11|20|30) log "solo_watch: solo exited $RC - a stop for a person (HANDOVER_v10 section 4b)"; exit $RC ;;
    esac
    tries=$((tries + 1))
    if [ $tries -gt "$max" ]; then log "solo_watch: solo failed $tries times (last exit $RC) - giving up; see $LOGS/solo.log"; exit $RC; fi
    log "solo_watch: solo exited $RC (attempt $tries of $max) - retrying in 2 min"
    sleep "${SOLO_RETRY_WAIT:-120}"
  done
  ;;

candidate)
  grep -aq "== done:" "$LOGS/train.log" || [ -f "$LOGS/EARLY_STOP" ] || { echo "training has not finished"; exit 1; }
  if [ -s "$LOGS/gpu1.pid" ] && kill -0 "$(cat "$LOGS/gpu1.pid")" 2>/dev/null; then
    echo "the gpu1 scheduler is still running (pid $(cat "$LOGS/gpu1.pid")); wait for 'GPU 1 is free' in $LOGS/gpu1.log"; exit 1
  fi
  evalenv
  set +e; eval_candidate 2>&1 | tee -a "$LOGS/candidate.log"; RC=${PIPESTATUS[0]}; set -e; exit $RC
  ;;

phase0)
  # Finish Phase 0 #1 (v9b@100/200 vs v8merged; informs Stage 2 only). EVAL_GPU=0 once G5 has freed GPU 0.
  evalenv
  until ! gen_chunk phase0_p1 phase0_v10.py p1; do :; done
  ev phase0_v10.py report 2>&1 | tee "$LOGS/phase0_p1_report.txt"
  ;;

g1)
  evalenv
  set +e; eval_g1 2>&1 | tee -a "$LOGS/g1.log"; RC=${PIPESTATUS[0]}; set -e; exit $RC
  ;;

sync)
  az push --prefix v10/work/code "$FW/prep_v10.py" "$FW/train_pudgy_v10.sh" "$FW/v10_box.sh" "$FW/cache_mark_v10.py" "$FW/asset_embed_v9.py"
  [ -d "$FW/eval_v10" ] && az push --prefix v10/work/eval_v10 --tree "$FW/eval_v10" --exclude __pycache__
  az push --prefix v10/work/docs --tree "$REPO/training_approach/v10"
  [ -d "$REPO/docs/training_reports/v10" ] && az push --prefix v10/work/reports --tree "$REPO/docs/training_reports/v10"
  az push --prefix v10/logs/box --tree "$LOGS"
  log "synced code, docs and logs to Azure v10/work, v10/logs/box"
  ;;

archive)
  az push --prefix "v10/weights/$RUN_NAME" "$RUN"/*.safetensors
  for st in "$RUN"/*-step*-state; do [ -d "$st" ] && az push --prefix "v10/weights/$RUN_NAME/$(basename "$st")" --tree "$st"; done
  az push --prefix "v10/logs/$RUN_NAME" --tree "$RUN/logs" || true
  az push --prefix v10/eval/runs --tree "$EVAL/runs" --exclude /cache/ /.raw_ .key.json
  bash "$0" sync
  log "archived to Azure pudgy/v10/"
  ;;

git)
  # Q7: commit on the box, push to the box's own branch (never main), and keep a bundle on Azure as a backup.
  cd "$REPO"
  git config user.name >/dev/null || git config user.name "pudgy-gpu-box"
  git config user.email >/dev/null || git config user.email "gpu-box@invalid"
  git checkout -B v10/box-run
  git add -A training_approach/v10 docs/training_reports/v10 finetune/wan 2>/dev/null || true
  git diff --cached --quiet || git commit -q -m "${MSG:-v10 box: run logs and report ($(ts))}"
  git push -u origin v10/box-run || log "push to GitHub failed - the bundle below is the record"
  B="$LOGS/v10_box-run_$(git rev-parse --short HEAD).bundle"
  git bundle create "$B" "origin/main..v10/box-run" 2>/dev/null || git bundle create "$B" v10/box-run
  az push --prefix v10/git "$B"
  log "git: v10/box-run at $(git rev-parse --short HEAD); bundle $(basename "$B") -> Azure v10/git"
  ;;

*)
  sed -n '2,24p' "$0"; exit 1 ;;
esac
