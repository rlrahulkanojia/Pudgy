#!/bin/bash
# Run every v10 training-side test (CPU only, no GPU). Build the environment first:
#   bash finetune/wan/tests/v10_train/setup_env.sh
#   bash finetune/wan/tests/v10_train/run_all.sh
# Work dir: $V10_TEST_WORK (default ~/.cache/pudgy_v10_tests). Nothing is written to the repo
# (no bytecode either); temp files go to <work>/tmp/run.XXXX, deleted when everything passes.
set -u
TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO; REPO=$(cd "$TESTS/../../../.." && pwd)
export V10_TEST_WORK="${V10_TEST_WORK:-$HOME/.cache/pudgy_v10_tests}"
WK=$V10_TEST_WORK
VPY=$WK/venv/bin/python
MUSUBI_SHA=30c658c4f4b0bf05038b3346eff9670259b10fc7      # kohya-ss/musubi-tuner tag v0.3.4
if [ ! -x "$VPY" ] || [ ! -d "$WK/musubi/src" ] || [ ! -d "$WK/hf" ]; then
  echo "no test environment in $WK - run: bash $TESTS/setup_env.sh"; exit 2
fi
[ "$(git -C "$WK/musubi" rev-parse HEAD 2>/dev/null)" = "$MUSUBI_SHA" ] || { echo "$WK/musubi is not musubi-tuner v0.3.4"; exit 2; }
export MUSUBI=$WK/musubi HF_HOME=$WK/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ASSET_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WK/tmp"
export TEST_TMP; TEST_TMP=$(mktemp -d "$WK/tmp/run.XXXX")
export TMPDIR=$TEST_TMP
FW=$REPO/finetune/wan
F=0
echo "repo $REPO | work $WK | python $("$VPY" -V 2>&1 | cut -d' ' -f2), torch $("$VPY" -B -c 'import torch; print(torch.__version__)')"

"$VPY" -B -c 'import sys; [compile(open(f).read(), f, "exec") for f in sys.argv[1:]]' \
  "$FW/asset_embed_v9.py" "$FW/cache_mark_v10.py" "$TESTS"/*.py && echo "syntax (python): OK" || { echo "syntax (python): FAIL"; F=1; }
bash -n "$FW/train_pudgy_v10.sh" && bash -n "$TESTS/test_train_sh.sh" && echo "syntax (bash -n): OK" || { echo "syntax (bash -n): FAIL"; F=1; }

report() {   # name rc output
  local n_ok n_fail
  n_ok=$(printf "%s\n" "$3" | grep -cE '^  ok ')
  n_fail=$(printf "%s\n" "$3" | grep -cE '^  FAIL')
  if [ "$2" -eq 0 ]; then echo "$1: PASS ($n_ok ok)"; else echo "$1: FAIL (rc $2, $n_ok ok, $n_fail failed)"; F=1; fi
  printf "%s\n" "$3" | grep -E '^  (FAIL|SKIP)' || true
}
for t in test_call_dit.py test_cache_mark.py test_spans_real.py test_parser.py; do
  out=$("$VPY" -B "$TESTS/$t" 2>&1); rc=$?
  report "$t" $rc "$out"
  [ -n "${VERBOSE:-}" ] && printf "%s\n" "$out" | grep -E '^  (ok|FAIL|SKIP)|^=='
done
out=$(bash "$TESTS/test_train_sh.sh" 2>&1); rc=$?
report test_train_sh.sh $rc "$out"
[ -n "${VERBOSE:-}" ] && printf "%s\n" "$out" | grep -E '^  (ok|FAIL)|^=='

if [ $F -eq 0 ]; then rm -rf "$TEST_TMP"; echo "ALL TESTS PASS"; else echo "SOME TESTS FAILED (temp files kept in $TEST_TMP)"; fi
exit $F
