#!/bin/bash
# Pivotal LoRA stage (user request 2026-10-28, queued after job 020). Pre-registered:
# 1. embedding: the E5 checkpoint selected by job 015 on held-out phone loss (used only if its held-out
#    ratio < 1.00), unless E5's ID1 confirm is worse than Rb by >= 2/24 (or missing): then E = 0.
# 2. smoke (16 steps): exit 0, peak GPU < 79.5 GB, >= 1 sample on each expert AND >= 1 preservation sample,
#    E unchanged (frozen), LoRA = 80 cross-attn k/v modules with nonzero updates.
# 3. full run 150 x 4
# 4. Tier 1, low-noise expert: gate R1/R2/P1/P2. Tier 1, HIGH-noise expert (t 0.95/0.92, v8 labels): gate R2
#    (the LoRA also changes the high-noise expert, where layout and motion are decided; R1 reported only)
# 5. ID1 confirm tier (24 clips) vs Rb and E5.
set -uo pipefail
W=/workspace; PY=$W/Pudgy/.venv-wan/bin/python; FW=$W/Pudgy/finetune/wan; D=$W/research_v9/v2
A=$W/eval_v9/runs/pudgy-v9c-phone-lownoise/arms_v9c2/confirm
export HF_HOME=$W/.hf_home RUN_NAME=pudgy-v9c-phone-lownoise PUDGY_DATA=$W/Data
log() { echo "== [$(date -u +%FT%TZ)] PV: $*" | tee -a "$D/v2.log"; }
defects() { python3 -c "import json;print(json.load(open('$1'))['defects'])" 2>/dev/null || echo NA; }

E5=$(cat $D/e5_selected.txt 2>/dev/null || echo "")
ZERO=$W/wan_cache/latents_v9c2_emb/asset_init_offset.safetensors
E5D=$(defects $A/E5/id1.json); RBD=$(defects $A/Rb/id1.json)
if [ -s "$E5" ] && [ "$E5" != "$ZERO" ] && [ "$E5D" != NA ] && [ "$RBD" != NA ] && [ $((E5D - RBD)) -lt 2 ]; then
  E=$E5; WHY="selected E5 $(basename $E5) (ID1 final $E5D/24 vs Rb $RBD/24)"
else E=$ZERO; WHY="zero embedding (selection: ${E5:-none}; ID1 E5=$E5D Rb=$RBD)"; fi
log "embedding: $WHY"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$D/pv_smoke_mem.csv" & MEM=$!
E_FILE=$E SMOKE=1 bash $FW/train_pivotal_v9.sh > "$D/pv_smoke.log" 2>&1; RC=$?
kill $MEM 2>/dev/null; PEAK=$(sort -n "$D/pv_smoke_mem.csv" | tail -1)
S=$W/wan_output/pudgy-v9c2-pivotal-smoke
CHK=$("$PY" - <<PYEOF 2>&1 | tail -1
import glob, torch
from safetensors import safe_open
m = safe_open("$S/pudgy-v9c2-pivotal-smoke.asset.safetensors", "pt").metadata()   # final save of the smoke run
lf = safe_open("$S/pudgy-v9c2-pivotal-smoke.safetensors", "pt")
keys = list(lf.keys()); mods = {k.split(".")[0] for k in keys if k.endswith("lora_up.weight")}
up = sum(float(lf.get_tensor(k).float().norm()) for k in keys if k.endswith("lora_up.weight"))
ok = (int(m["samples_high"]) >= 1 and int(m["samples_low"]) >= 1 and int(m["samples_preserve"]) >= 1
      and m["frozen"] == "True" and float(m["delta_norm"]) == 0.0 and len(mods) == 80
      and all("cross_attn_k" in x or "cross_attn_v" in x for x in mods) and up > 0)
print(("OK" if ok else "BAD"), "high", m["samples_high"], "low", m["samples_low"], "preserve", m["samples_preserve"],
      "E delta", m["delta_norm"], "modules", len(mods), "lora_up norm %.3f" % up)
PYEOF
)
STEP=$(tr '\r' '\n' < "$D/pv_smoke.log" | grep -aoE "[0-9.]+s/it" | tail -1)
log "smoke: exit $RC, peak GPU ${PEAK} MiB, $CHK, $STEP"
if [ "$RC" -ne 0 ] || [ "${PEAK:-99999}" -ge 79500 ] || [[ "$CHK" != OK* ]]; then log "smoke FAILED - not starting the full run"; exit 1; fi

log "training: rank 16 cross-attn k/v, both experts, preserve x1.0, 150 x 4"
E_FILE=$E bash $FW/train_pivotal_v9.sh > "$D/pv_train.log" 2>&1 || { log "training FAILED - see $D/pv_train.log"; exit 1; }
LORA=$W/wan_output/pudgy-v9c2-pivotal/pudgy-v9c2-pivotal.safetensors
log "trained: $LORA ($("$PY" -c "from safetensors import safe_open;m=safe_open('${LORA%.safetensors}.asset.safetensors','pt').metadata();print('samples high',m['samples_high'],'low',m['samples_low'],'preserve',m['samples_preserve'])"))"

cd $FW/eval_v9
[ -s $W/eval_v9/runs/pudgy-v9c-phone-lownoise/dcls/asset_base_newwording_k2.json ] || \
  "$PY" dcls_asset_v9.py --config v8merged --asset "$ZERO" --offset --tag base_newwording > "$D/pv_tier1_base.log" 2>&1
"$PY" dcls_asset_v9.py --config "pivotal:$LORA" --asset "$E" --offset --kinds expression motion prop --split holdout seen \
  --tag pivotal > "$D/pv_tier1.log" 2>&1 || { log "Tier 1 FAILED"; exit 1; }
"$PY" gate_pivotal_v9.py --cand asset_pivotal_k2.json --base-v8 v8merged_k2.json --base-prop asset_base_newwording_k2.json \
  > "$D/pv_gate.txt" 2>&1
log "gate (low-noise expert): $(grep -v Warn "$D/pv_gate.txt" | tr '\n' ';' | sed 's/;*$//')"
[ -s $W/eval_v9/runs/pudgy-v9c-phone-lownoise/dcls/asset_base_high_k2.json ] || \
  "$PY" dcls_asset_v9.py --config v8merged --asset "$ZERO" --offset --expert high --kinds expression motion --split holdout \
    --tag base_high > "$D/pv_tier1_base_high.log" 2>&1
"$PY" dcls_asset_v9.py --config "pivotal:$LORA" --asset "$E" --offset --expert high --kinds expression motion --split holdout \
  --tag pivotal_high > "$D/pv_tier1_high.log" 2>&1 || log "Tier 1 high-noise FAILED"
"$PY" gate_pivotal_v9.py --cand asset_pivotal_high_k2.json --base-v8 asset_base_high_k2.json --gating R2 > "$D/pv_gate_high.txt" 2>&1
log "gate (high-noise expert, R2 gating): $(grep -v Warn "$D/pv_gate_high.txt" | tr '\n' ';' | sed 's/;*$//')"
PIVOTAL_LORA=$LORA "$PY" arms_v9c2.py --tier confirm --arms PV --asset "$E" > "$D/pv_confirm.log" 2>&1 \
  && log "confirm: $(tail -1 $W/eval_v9/runs/pudgy-v9c-phone-lownoise/arms_v9c2/arms.log) | Rb $RBD/24, E5 $E5D/24" \
  || log "confirm FAILED - see $D/pv_confirm.log"
log "done"
