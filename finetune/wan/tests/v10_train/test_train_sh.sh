#!/bin/bash
# finetune/wan/train_pudgy_v10.sh with a fake /workspace (build_ws.py) and a fake accelerate (no GPU):
# refusals, the exact Stage 1 command, auto-resume, smoke mode, the box script's smoke gate lines and
# the pgrep pause pattern. Works with macOS's bash 3.2. Run through run_all.sh (needs its env).
set -u
TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-$(cd "$TESTS/../../../.." && pwd)}"
WORK="${V10_TEST_WORK:-$HOME/.cache/pudgy_v10_tests}"
VPY=$WORK/venv/bin/python
mkdir -p "${TEST_TMP:-$WORK/tmp}"
T=$(mktemp -d "${TEST_TMP:-$WORK/tmp}/trainsh.XXXX"); WS=$T/ws
FAILS=0
ok()   { echo "  ok   $*"; }
bad()  { echo "  FAIL $*"; FAILS=$((FAILS + 1)); }
expect() { if eval "$1"; then ok "$2"; else bad "$2"; fi; }
mkdir -p "$T/bin"
cat > "$T/bin/py" <<PY
#!/bin/bash
exec "$VPY" -B "$TESTS/run_py.py" "\$@"
PY
cat > "$T/bin/accelerate" <<'ACC'
#!/bin/bash
# fake trainer: log argv, print what the real wrapper prints, exit with $FAKE_RC
echo "ACC_ARGS $*" >> "$FAKE_ARGS"
printf 'steps:  33%%|###  | 1/3 [00:10<00:20]\r'
echo; echo "[asset] base weights: 400/400 LoRA modules match the DiT (x1.0); merged into its weights, so LoRA multiplier 0 keeps them"
printf 'steps: 100%%|#####| 3/3\r'; echo; echo "PRESERVE_STATS step=3 total=3 preserve=1 data=2 asset=1 high=0 low=0 marker=key | since last line: preserve_mse 1.0e-06 (n=1) data_mse 1.2e-01 (n=2)"
exit "${FAKE_RC:-0}"
ACC
chmod +x "$T/bin/py" "$T/bin/accelerate"
CFG=$("$VPY" -B "$TESTS/build_ws.py" "$WS" 2>/dev/null | tail -1)
E5F=$WS/wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors
export W=$WS FW=$REPO/finetune/wan PY=$T/bin/py ACC=$T/bin/accelerate MUSUBI_DIR=$T DATASET=$CFG FAKE_ARGS=$T/args.txt
unset CUDA_VISIBLE_DEVICES RESUME SMOKE DRY_RUN
TR="bash $REPO/finetune/wan/train_pudgy_v10.sh"
expect '[ -s "$CFG" ]' "fake workspace built ($WS)"

echo "== refuses an unmarked cache"
$TR > "$T/o1.txt" 2>&1; rc=$?
expect '[ $rc -ne 0 ] && grep -q "REFUSING TO TRAIN: the cache is not marked" "$T/o1.txt" && [ ! -s "$FAKE_ARGS" ]' "unmarked cache: exit $rc, accelerate never called"

"$PY" "$FW/cache_mark_v10.py" mark --config "$CFG" --asset "$E5F" > "$T/mark.txt" 2>&1
expect 'grep -q "CACHE_MARK_CHECK OK" "$T/mark.txt"' "mark the fake cache"

echo "== dry run: the Stage 1 command"
DRY_RUN=1 $TR > "$T/o2.txt" 2>&1
C=$(grep -A1 "== DRY_RUN" "$T/o2.txt" | tail -1)
for f in "launch --num_processes 1" "asset_embed_v9.py train --init $E5F --offset --freeze-asset" \
         "--preserve --preserve-weight 1.0 --preserve-marker key --log-every 5 --strict-base-weights --" \
         "--dit $WS/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
         "--base_weights $WS/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors --base_weights_multiplier 1.0" \
         "--network_module musubi_tuner.networks.lora_wan --network_dim 16 --network_alpha 16" \
         "--timestep_sampling shift --discrete_flow_shift 5.0 --min_timestep 900 --max_timestep 1000 --preserve_distribution_shape" \
         "--optimizer_type adamw8bit --learning_rate 3e-5 --lr_scheduler cosine_with_min_lr --lr_warmup_steps 20 --lr_scheduler_min_lr_ratio 0.1" \
         "--gradient_accumulation_steps 8" "--max_train_steps 300 --save_every_n_steps 25 --save_state --seed 42" \
         "--output_dir $WS/wan_output/pudgy-v10-highnoise --output_name pudgy-v10-highnoise" "--blocks_to_swap 24" "--mixed_precision fp16"; do
  case "$C" in *"$f"*) ok "has: $(echo "$f" | sed "s#$WS#<W>#g" | cut -c1-90)";; *) bad "missing: $f";; esac
done
case "$C" in *dit_high_noise*|*--resume*|*network_args*) bad "unexpected --dit_high_noise / --resume / --network_args";; *) ok "no --dit_high_noise, no --resume, no --network_args";; esac
expect 'grep -q "GPU 0" "$T/o2.txt"' "CUDA_VISIBLE_DEVICES defaults to 0"

echo "== resume"
O=$WS/wan_output/pudgy-v10-highnoise; mkdir -p "$O/pudgy-v10-highnoise-step00000025-state" "$O/pudgy-v10-highnoise-step00000050-state"
echo x > "$O/pudgy-v10-highnoise-step00000025-state/random_states_0.pkl"
DRY_RUN=1 $TR > "$T/o3.txt" 2>&1
expect 'grep -q "skipping incomplete state .*step00000050-state" "$T/o3.txt" && grep -A1 "== DRY_RUN" "$T/o3.txt" | tail -1 | grep -q -- "--resume $O/pudgy-v10-highnoise-step00000025-state"' "auto-resume: newest COMPLETE state (25), incomplete 50 skipped"
RESUME=none DRY_RUN=1 $TR > "$T/o4.txt" 2>&1
expect '! grep -A1 "== DRY_RUN" "$T/o4.txt" | tail -1 | grep -q -- "--resume"' "RESUME=none starts fresh"
MAX_STEPS=25 DRY_RUN=1 $TR > "$T/o5.txt" 2>&1; rc=$?
expect '[ $rc -eq 0 ] && grep -q "nothing left to train" "$T/o5.txt" && grep -q "== done:" "$T/o5.txt"' "state at MAX_STEPS: nothing trained, exit 0"
RESUME=$O/pudgy-v10-highnoise-step00000050-state DRY_RUN=1 $TR > "$T/o6.txt" 2>&1; rc=$?
expect '[ $rc -ne 0 ] && grep -q "resume state is incomplete" "$T/o6.txt"' "explicit incomplete RESUME refused"
rm -rf "$O"

echo "== refusals"
PRESERVE_WEIGHT=0 DRY_RUN=1 $TR > "$T/o7.txt" 2>&1; rc=$?
expect '[ $rc -ne 0 ] && grep -q "PRESERVE_WEIGHT must be > 0" "$T/o7.txt"' "PRESERVE_WEIGHT=0 refused"
CUDA_VISIBLE_DEVICES=0,1 DRY_RUN=1 $TR > "$T/o8.txt" 2>&1; rc=$?
expect '[ $rc -ne 0 ] && grep -q "exactly one GPU" "$T/o8.txt"' "two visible GPUs refused"
CACHE=$WS/elsewhere DRY_RUN=1 $TR > "$T/o9.txt" 2>&1; rc=$?
expect '[ $rc -ne 0 ] && grep -q "is not under" "$T/o9.txt"' "toml cache dirs outside CACHE refused"

echo "== smoke (fake trainer) and the gate lines for the box script"
: > "$FAKE_ARGS"
LOG=$T/smoke.log
SMOKE=1 $TR > "$LOG" 2>&1; RC=$?
A=$(cat "$FAKE_ARGS")
for f in "--output_name pudgy-v10-highnoise-smoke" "--max_train_steps 3 --save_every_n_steps 3" "--gradient_accumulation_steps 1" \
         "--log-every 1" "--dataset_config $WS/wan_cache/latents_v10_smoke/smoke.toml"; do
  case "$A" in *"$f"*) ok "smoke has: $(echo "$f" | sed "s#$WS#<W>#g")";; *) bad "smoke missing: $f";; esac
done
case "$A" in *--resume*) bad "smoke must not resume";; *) ok "smoke never resumes";; esac
# --- the gate the box script should use ---
OOM=$(grep -ac "memory mapping failed with OOM" "$LOG")
ST=$(tr '\r' '\n' < "$LOG" | grep -a "PRESERVE_STATS" | tail -1)
PRES=$(echo "$ST" | sed -nE 's/.* preserve=([0-9]+) .*/\1/p'); ASSET=$(echo "$ST" | sed -nE 's/.* asset=([0-9]+) .*/\1/p')
if [ "$RC" -eq 0 ] && grep -aq "== done:" "$LOG" && [ "$OOM" -eq 0 ] && [ "${PRES:-0}" -gt 0 ] && [ "${ASSET:-0}" -gt 0 ] \
   && grep -aqE "\[asset\] base weights: ([0-9]+)/\1 LoRA modules match" "$LOG"; then GATE=OK; else GATE=FAIL; fi
expect '[ $GATE = OK ] && [ "$PRES" = 1 ] && [ "$ASSET" = 1 ]' "gate passes on a good smoke log (rc $RC, OOM $OOM, preserve $PRES, asset $ASSET)"
FAKE_RC=1 SMOKE=1 $TR > "$T/smoke_bad.log" 2>&1; RC=$?
expect '[ $RC -ne 0 ] && ! grep -aq "== done:" "$T/smoke_bad.log"' "failed trainer: non-zero exit, no '== done:'"
sed 's/ preserve=1 / preserve=0 /' "$LOG" > "$T/smoke_nopres.log"
P0=$(tr '\r' '\n' < "$T/smoke_nopres.log" | grep -a "PRESERVE_STATS" | tail -1 | sed -nE 's/.* preserve=([0-9]+) .*/\1/p')
expect '[ "${P0:-0}" -eq 0 ]' "gate parser reads preserve=0 as a failure"

echo "== pgrep pattern for pause"
PAT='asset_embed_v9\.py train .*--output_name pudgy-v10-highnoise( |$)'
SLEEP=(-c "import time; time.sleep(30)")
"$VPY" "${SLEEP[@]}" /workspace/Pudgy/.venv-wan/bin/accelerate launch /workspace/Pudgy/finetune/wan/asset_embed_v9.py train --init e5 -- --dit x --output_name pudgy-v10-highnoise --log_with tensorboard & P1=$!
"$VPY" "${SLEEP[@]}" /workspace/Pudgy/finetune/wan/asset_embed_v9.py train --init e5 -- --output_name pudgy-v10-highnoise & P2=$!
"$VPY" "${SLEEP[@]}" /workspace/Pudgy/finetune/wan/asset_embed_v9.py train --init e5 -- --output_name pudgy-v10-highnoise-smoke --log_with tensorboard & P3=$!
"$VPY" "${SLEEP[@]}" bash /workspace/Pudgy/finetune/wan/v10_box.sh pause 50 & P4=$!
sleep 1
M=$(pgrep -f "$PAT" | sort | tr '\n' ' ')
W2=$(printf "%s\n" $P1 $P2 | sort | tr '\n' ' ')
expect '[ "$M" = "$W2" ]' "pgrep matches the launcher + trainer, not the smoke run or the box shell"
{ kill $P1 $P2 $P3 $P4; wait $P1 $P2 $P3 $P4; } 2>/dev/null

echo; [ $FAILS -eq 0 ] && echo "ALL PASS" || echo "$FAILS FAILED"
[ $FAILS -eq 0 ] && rm -rf "$T"
exit $FAILS
