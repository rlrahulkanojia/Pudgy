#!/usr/bin/env bash
# =========================================================================================
# expv2 stage runner. Chains the RUNBOOK's box steps so the GPU never idles between them.
# Every stage is exactly the RUNBOOK command. A stage that succeeds leaves a marker in
# $EXPV2_WORK/stages/, so re-running skips it. The runner STOPS at every failure and at
# every decision a person must make: the G0.1 eye check, a calibration STOP, the rubrics.
#
#   bash expv2-minimax-h3/run_expv2.sh status          # which stages are done
#   bash expv2-minimax-h3/run_expv2.sh auto            # run every pending stage in order, up to the rubrics
#   bash expv2-minimax-h3/run_expv2.sh <stage>         # run one stage (even if marked done)
#
# Stages, in order (RUNBOOK section in brackets):
#   corpus   pull or build the corpus, then check_loader                       [4]
#   g01      VAE gate metrics -> then STOP until `vae_gate.py --eye PASS`      [6]
#   cache    stage-1 training cache f22/f39/f56                                [6]
#   g02      parity DiT fetch, pipeline checks a-h, parity DiT delete          [6]
#   g05      source baseline (CPU)                                             [6]
#   g06      memory probe                                                      [6]
#   encode   pass-1 encodings for Tier 1 and the suites                        [6]
#   base     G0.3 + G0.4: the untrained control (Tier 1, core+native, scores)  [6]
#   pilots   S12 + S2.22, Tier 1 on both, --calibrate (STOP on its STOP)        [7]
#   main     the main run at the calibrated shift, then Tier 0                 [7]
#   select   Tier 1 + probe on every main checkpoint, the winner rule          [8]
#   tier2    core/native/gb on the winner, scored                              [8]
# After tier2 the runner prints the manual steps (two rubrics, section 8, the verdict).
# Start the Azure mirror (RUNBOOK 5) BEFORE `auto`: a lost box loses only what it had not pushed.
# =========================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/configs/env.sh"
E="$EXPV2_ROOT"
ST="$EXPV2_WORK/stages"; mkdir -p "$ST" "$EXPV2_WORK/logs"
LOG="$EXPV2_WORK/logs/run_expv2.log"
STAGES="corpus g01 cache g02 g05 g06 encode base pilots main select tier2"

say()  { printf '\n[%s] == %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }
run()  { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
stop() { say "STOP: $*"; exit 3; }
jget() { python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d.get(k) if isinstance(d, dict) else None
print("" if d is None else d)' "$1" "$2" 2>/dev/null || true; }

stage_corpus() {
  if [ ! -f "$PUDGY_DATA/processed/expv2_h3/manifest.json" ]; then
    run python "$E/scripts/azure_sync.py" pull corpus || true
  fi
  if [ ! -f "$PUDGY_DATA/processed/expv2_h3/manifest.json" ]; then
    run python "$E/scripts/azure_sync.py" pull raw
    run python "$E/prep/prep_expv2.py" --dry-run
    run python "$E/prep/prep_expv2.py" --workers "$(nproc)" --exact-tokens
    run python "$E/scripts/azure_sync.py" push corpus
  fi
  run python "$E/prep/check_loader.py"
}

stage_g01() {
  local rep="$EXPV2_WORK/eval/phase0/vae_gate/report.json"
  if [ ! -f "$rep" ] || [ "$(jget "$rep" gate)" = "" ]; then
    run python "$E/eval/vae_gate.py" || stop "G0.1 metric bars FAILED - stop expv2 and report (PLAN 4)"
  fi
  case "$(jget "$rep" gate)" in
    PASS) return 0 ;;
    FAIL) stop "G0.1 FAILED - stop expv2 and report (PLAN 4)" ;;
    *)    stop "G0.1 metrics passed; the 4x eye check is PENDING. View $EXPV2_WORK/eval/phase0/vae_gate/sheets/
       then: python $E/eval/vae_gate.py --eye PASS --eye-notes '...'   (or --eye FAIL, which ends expv2)
       and re-run: bash $E/run_expv2.sh auto" ;;
  esac
}

stage_cache() { run bash "$E/configs/cache.sh"; }

stage_g02() {
  run python "$E/setup/fetch_models.py" --set parity
  # f (mixed f22/f39/f56 caches) only matters if the long rungs are trained (PLAN 5.2):
  # EXPV2_BUCKETS="22 39 56" adds it back.
  local checks="a,b,c,d,e,g,h"; [ "${EXPV2_BUCKETS:-22}" != 22 ] && checks="a,b,c,d,e,f,g,h"
  if ! run python "$E/eval/pipeline_checks.py" --checks "$checks"; then
    stop "G0.2 hard failure - see $EXPV2_WORK/eval/phase0/pipeline_checks/report.json (PLAN 4)"
  fi
  run python "$E/setup/fetch_models.py" --delete parity
}

stage_g05() { run python "$E/eval/source_baseline.py"; }
stage_g06() { run python "$E/eval/mem_probe.py" --frames 22,39,56; }

stage_encode() {
  run python "$E/eval/dcls.py" --encode --set holdout,calib
  run python "$E/eval/generate.py" --encode --suites core,native,gb,probe
}

stage_base() {
  run python "$E/eval/dcls.py" --ckpt base --set holdout,calib --sigmas grid
  run python "$E/eval/generate.py" --ckpt base --suites core,native
  run python "$E/eval/score.py" --ckpt base --suites core,native
  # Early read, not a gate: the untrained base vs v8@550 on the same 48 PLAN-8 pairs.
  # The final PLAN-8 label is only ever computed on the trained winner (stage tier2).
  run python "$E/scripts/azure_sync.py" pull v8
  run python "$E/eval/h2h_v8.py" --ckpt base --v8-dir "$EXPV2_WORK/v8_550" \
    --out "$EXPV2_WORK/eval/reports/h2h_v8_base.json"
}

stage_pilots() {
  [ "$(jget "$EXPV2_WORK/eval/phase0/vae_gate/report.json" gate)" = PASS ] || stop "G0.1 has not passed"
  for r in expv2-h3-s12 expv2-h3-s222; do
    ls "$EXPV2_WORK/output/$r"/step-*.safetensors >/dev/null 2>&1 && [ -f "$ST/train_$r.done" ] || {
      run bash "$E/configs/train.sh" "$r"; touch "$ST/train_$r.done"; }
  done
  run python "$E/eval/dcls.py" --ckpt expv2-h3-s12@last,expv2-h3-s222@last --set holdout,calib --sigmas grid
  run python "$E/eval/dcls.py" --calibrate --pilots expv2-h3-s12@last,expv2-h3-s222@last
  local cal="$EXPV2_WORK/eval/tier1/calibration.json" action
  action="$(jget "$cal" action)"
  case "$action" in
    STOP*|*re-run*) stop "calibration says: $action" ;;
  esac
  [ -n "$(jget "$cal" shift.winner_train_flow_shift)" ] || stop "calibration.json has no shift winner"
  say "shift winner: $(jget "$cal" shift.winner) (train_flow_shift $(jget "$cal" shift.winner_train_flow_shift)); sigmas $(jget "$cal" sigmas)"
}

stage_main() {
  run bash "$E/configs/train.sh" expv2-h3-main
  run python "$E/eval/tier0.py" --run expv2-h3-main
}

stage_select() {
  run python "$E/eval/dcls.py" --ckpt base,expv2-h3-main@all --set holdout --sigmas calibrated
  run python "$E/eval/generate.py" --ckpt expv2-h3-main@all --suites probe
  run python "$E/eval/score.py" --ckpt expv2-h3-main@all --suites probe
  run python "$E/eval/select_winner.py" --run expv2-h3-main
  [ -n "$(jget "$EXPV2_WORK/eval/reports/winner.json" expv2-h3-main.winner)" ] \
    || stop "no checkpoint is eligible under PLAN 7.3 - see eval/reports/winner.json (that is a result, report it)"
}

stage_tier2() {
  local W; W="$(jget "$EXPV2_WORK/eval/reports/winner.json" expv2-h3-main.winner)"
  [ -n "$W" ] || stop "no winner"
  run python "$E/eval/generate.py" --ckpt "$W" --suites core,native,gb
  run python "$E/eval/score.py" --ckpt "$W" --suites core,native,gb
  run python "$E/eval/verdict.py" --rubric-template --ckpt "$W"
  run python "$E/scripts/azure_sync.py" pull v8
  run python "$E/eval/h2h_v8.py" --ckpt "$W" --v8-dir "$EXPV2_WORK/v8_550" --rubric-template
  say "winner $W generated and scored. Manual steps now (RUNBOOK 8):
   1. fill $EXPV2_WORK/eval/reports/rubric_$W.json   (V2, by eye)
   2. fill $EXPV2_WORK/eval/reports/h2h_rubric.json   (section 8, blind - do not open the key)
   3. python $E/eval/h2h_v8.py --ckpt '$W' --v8-dir $EXPV2_WORK/v8_550
   4. python $E/eval/verdict.py --arm A     (it prints whether the arm-D trigger fired)"
}

status() {
  for s in $STAGES; do
    if [ -f "$ST/$s.done" ]; then echo "  done     $s  ($(cat "$ST/$s.done"))"; else echo "  pending  $s"; fi
  done
}

do_stage() {
  local s="$1"
  say "stage $s"
  "stage_$s"
  date -u +%FT%TZ > "$ST/$s.done"
  say "stage $s done"
}

cmd="${1:-status}"
case "$cmd" in
  status) status ;;
  auto)
    for s in $STAGES; do
      [ -f "$ST/$s.done" ] && { echo "  skip $s (done)"; continue; }
      do_stage "$s"
    done
    say "all automatic stages done" ;;
  *)
    case " $STAGES " in *" $cmd "*) do_stage "$cmd" ;; *) echo "unknown stage $cmd; stages: $STAGES"; exit 1 ;; esac ;;
esac
