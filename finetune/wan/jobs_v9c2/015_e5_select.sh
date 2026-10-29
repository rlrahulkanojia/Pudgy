#!/bin/bash
# E5 checkpoint selection (introspection 2026-10-28). V1c showed the embedding fitting its training clips
# (seen margin +0.136) while barely moving on the held-out angle, so the final checkpoint is not assumed best.
# Pre-registered: score every saved E5 checkpoint with the classifier on phone clips (held-out + seen,
# k=2, low-noise expert), paired against the base with the SAME wording (E = 0). Select the checkpoint with
# the lowest held-out phone-loss ratio; it is used downstream only if that ratio is < 1.00 (else E = 0).
# The seen/held-out curve is logged as the overfitting check.
set -uo pipefail
W=/workspace; PY=$W/Pudgy/.venv-wan/bin/python; D=$W/research_v9/v2; R=$W/eval_v9/runs/pudgy-v9c-phone-lownoise/dcls
export HF_HOME=$W/.hf_home RUN_NAME=pudgy-v9c-phone-lownoise PUDGY_DATA=$W/Data
log() { echo "== [$(date -u +%FT%TZ)] E5sel: $*" | tee -a "$D/v2.log"; }
OUTD=$W/wan_output/pudgy-v9c2-asset-dual; ZERO=$W/wan_cache/latents_v9c2_emb/asset_init_offset.safetensors
ls $OUTD/*.asset.safetensors >/dev/null 2>&1 || { log "no E5 checkpoints - nothing to select"; echo "$ZERO" > $D/e5_selected.txt; exit 1; }
cd $W/Pudgy/finetune/wan/eval_v9
[ -s $R/asset_base_newwording_k2.json ] || "$PY" dcls_asset_v9.py --config v8merged --asset "$ZERO" --offset --tag base_newwording > $D/e5sel_base.log 2>&1
for E in $(ls $OUTD/*.asset.safetensors | sort); do
  s=$(basename $E .asset.safetensors | sed -E 's/.*-step0*([0-9]+)$/\1/; s/^pudgy-v9c2-asset-dual$/final/')
  [ -s $R/asset_e5_s${s}_k2.json ] || "$PY" dcls_asset_v9.py --config v8merged --asset "$E" --offset --tag e5_s$s > $D/e5sel_$s.log 2>&1
done
"$PY" - "$OUTD" "$R" "$ZERO" > $D/e5_select.txt <<'PYEOF'
import json, sys, glob, os, numpy as np
outd, r, zero = sys.argv[1:]
base = {x["clip"]: x for x in json.load(open(f"{r}/asset_base_newwording_k2.json"))["rows"]}
best = None
print("ckpt   held-out loss ratio   seen loss ratio   held-out margin   text-attrib gain")
for f in sorted(glob.glob(f"{r}/asset_e5_s*_k2.json"), key=lambda p: (p.endswith("sfinal_k2.json"), p)):
    tag = f.split("asset_e5_s")[1].split("_k2")[0]
    rows = {x["clip"]: x for x in json.load(open(f))["rows"]}
    ratio = lambda sp: sum(sum(rows[c]["loss_true"]) for c in rows if rows[c]["split"] == sp) / \
                       sum(sum(base[c]["loss_true"]) for c in rows if rows[c]["split"] == sp)
    ho, se = ratio("holdout"), ratio("seen")
    mg = np.mean([rows[c]["margin"] for c in rows if rows[c]["split"] == "holdout"])
    g = np.mean([np.log(rows[c]["E"]["phone"] / base[c]["E"]["phone"]) -
                 np.mean([np.log(rows[c]["E"][k] / base[c]["E"][k]) for k in rows[c]["E"] if k != "phone"]) for c in rows])
    print(f"{tag:6s} x{ho:.4f}               x{se:.4f}           {mg:+.4f}           {g:+.4f}")
    path = f"{outd}/pudgy-v9c2-asset-dual.asset.safetensors" if tag == "final" else \
           f"{outd}/pudgy-v9c2-asset-dual-step{int(tag):08d}.asset.safetensors"
    if best is None or ho < best[0]:
        best = (ho, tag, path)
chosen = best[2] if best and best[0] < 1.0 else zero
print(f"selected: {best[1] if best else None} (held-out ratio x{best[0]:.4f}) -> {chosen}")
print("PATH", chosen)
PYEOF
grep "^PATH" $D/e5_select.txt | cut -d' ' -f2 > $D/e5_selected.txt
log "curve: $(grep -v '^PATH' $D/e5_select.txt | tr '\n' ';' | sed 's/;*$//')"
log "selected E: $(cat $D/e5_selected.txt)"
