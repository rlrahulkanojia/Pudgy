#!/bin/bash
# E5 (queued 2026-10-28 by the GPU queue; first attempt 04:03 UTC failed: replay clips in the cache dir gave a step
# with no embedding gradient -> fp16 GradScaler abort. Fixed: phone-only cache dirs, latents_v9c2_emb_props).
# 1. base decision (pre-registered): v9c@200 stays in the low-noise base only if, on the new wording,
#    it beats v8merged by >= 2 of 24 ID1 defects (R vs Rb); otherwise E5 trains on v8merged alone.
# 2. smoke (12 steps): must exit 0, route >= 1 sample to EACH expert, keep E finite, peak GPU < 79.5 GB.
# 3. full run (150 steps x 4), 4. Tier 1 on phone clips, 5. ID1 confirm tier (24 clips) vs its control.
set -uo pipefail
W=/workspace; PY=$W/Pudgy/.venv-wan/bin/python; D=$W/research_v9/v2; FW=$W/Pudgy/finetune/wan
A=$W/eval_v9/runs/pudgy-v9c-phone-lownoise/arms_v9c2/confirm
export HF_HOME=$W/.hf_home RUN_NAME=pudgy-v9c-phone-lownoise PUDGY_DATA=$W/Data
log() { echo "== [$(date -u +%FT%TZ)] E5: $*" | tee -a "$D/v2.log"; }

R=$(python3 -c "import json;print(json.load(open('$A/R/id1.json'))['defects'])" 2>/dev/null || echo NA)
RB=$(python3 -c "import json;print(json.load(open('$A/Rb/id1.json'))['defects'])" 2>/dev/null || echo NA)
V8=$W/wan_output/v8_merged/pudgy-v8-merged.safetensors
V9=$W/wan_output/pudgy-v9c-phone-lownoise/pudgy-v9c-phone-lownoise-step00000200.safetensors
if [ "$R" != NA ] && [ "$RB" != NA ] && [ $((RB - R)) -ge 2 ]; then
  export BASE_LOW="$V8 $V9" E5_CONFIG=v9c@200; CTRL=R
else
  export BASE_LOW="$V8" E5_CONFIG=v8merged; CTRL=Rb
fi
log "base decision: R=$R/24, Rb=$RB/24 -> low-noise base = $E5_CONFIG (control $CTRL)"

# smoke
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$D/e5_smoke_mem.csv" & MEM=$!
SMOKE=1 bash $FW/train_asset_v9.sh > "$D/e5_smoke.log" 2>&1; RC=$?
kill $MEM 2>/dev/null
PEAK=$(sort -n "$D/e5_smoke_mem.csv" | tail -1)
SIDE=$(ls $W/wan_output/pudgy-v9c2-asset-dual-smoke/*.asset.safetensors 2>/dev/null | sort | tail -1)
CHK=$("$PY" -c "
import sys,torch
from safetensors import safe_open
f=safe_open('$SIDE','pt'); m=f.metadata(); E=f.get_tensor('E')
ok=int(m['samples_high'])>=1 and int(m['samples_low'])>=1 and bool(torch.isfinite(E).all())
print(('OK' if ok else 'BAD'), 'high', m['samples_high'], 'low', m['samples_low'], 'delta', m['delta_norm'])" 2>&1 | tail -1)
STEP=$(tr '\r' '\n' < "$D/e5_smoke.log" | grep -aoE "[0-9.]+s/it" | tail -1)
log "smoke: exit $RC, peak GPU ${PEAK} MiB, $CHK, $STEP"
if [ "$RC" -ne 0 ] || [ "${PEAK:-99999}" -ge 79500 ] || [[ "$CHK" != OK* ]]; then log "smoke FAILED - not starting the full run"; exit 1; fi

# full run
log "training: 150 steps x 4 clips, both experts, base $E5_CONFIG"
bash $FW/train_asset_v9.sh > "$D/e5_train.log" 2>&1 || { log "training FAILED - see $D/e5_train.log"; exit 1; }
E=$(ls $W/wan_output/pudgy-v9c2-asset-dual/*.asset.safetensors | sort | tail -1)
log "trained: $E ($("$PY" -c "from safetensors import safe_open;m=safe_open('$E','pt').metadata();print('samples high',m['samples_high'],'low',m['samples_low'],'delta',m['delta_norm'])"))"

# Tier 1 (phone clips, low-noise expert, t in {0.65, 0.85})
cd $FW/eval_v9
"$PY" dcls_asset_v9.py --config "$E5_CONFIG" --asset "$E" --offset --tag e5_final > "$D/e5_tier1.log" 2>&1 \
  && log "Tier 1: $(grep -aE 'holdout/prop' "$D/e5_tier1.log" | tail -1 | tr -s ' ')" || log "Tier 1 FAILED"
# ID1 confirm tier (8 starts x 3 seeds, 41 frames) vs control $CTRL
"$PY" arms_v9c2.py --tier confirm --arms E5 --asset "$E" > "$D/e5_confirm.log" 2>&1 \
  && log "confirm: $(tail -1 $W/eval_v9/runs/pudgy-v9c-phone-lownoise/arms_v9c2/arms.log) | control $CTRL=$( [ $CTRL = R ] && echo $R || echo $RB)/24" \
  || log "confirm FAILED - see $D/e5_confirm.log"
log "chain done - GPU free"
