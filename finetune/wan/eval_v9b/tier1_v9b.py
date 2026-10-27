#!/usr/bin/env python3
"""Tier 1 for v9b: the v8 diffusion classifier (eval_v8/dcls_v8.py, unchanged scoring:
same t-grid, same common random numbers, so configurations are PAIRED) on two clip sets,
with one model load per configuration:

  regression  v8 labels (expression + motion) on QF2_R (holdout) / QF2_L (seen):
              /workspace/eval_v9/holdout, the v9c set. Gate bars R1 / R2.
  scene       held-out skit windows (build_eval_v9b.py), true caption vs distractors.
              Gate bars L1 (held-out loss) / P2 (text-attributable gain).

The base's (v8merged) regression rows were scored for v9c with the same clips, noise draws,
weights and code. They are reused after re-scoring the first --spot clips and checking
they match to 1e-3 relative; otherwise everything is re-scored.

    python tier1_v9b.py --configs v8merged v9b@200 v9b@400
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9b as C  # noqa: E402
import dcls_v8 as D  # noqa: E402
import torch  # noqa: E402

D.lora_config = C.lora_config


def score_set(model, index, emb, k, device, holdout):
    D.HOLDOUT = holdout                 # load_latents() reads <holdout>/cache/f<n>/
    return [D.score_clip(model, e, emb, k, device) for e in index]


def summarise(rows_reg, rows_scene):
    out = {}
    for split in ("holdout", "seen"):
        for kind in ("expression", "motion"):
            rs = [r for r in rows_reg if r["split"] == split and r["kind"] == kind]
            if rs:
                out[f"{split}/{kind}"] = {
                    "n": len(rs), "acc": round(float(np.mean([r["pred"] == r["label"] for r in rs])), 3),
                    "margin": round(float(np.mean([r["margin"] for r in rs])), 4),
                    "loss": round(float(np.mean([np.mean(r["loss_true"]) for r in rs])), 5)}
    if rows_scene:
        out["holdout/scene"] = {
            "n": len(rows_scene),
            "acc": round(float(np.mean([r["pred"] == "true" for r in rows_scene])), 3),
            "margin": round(float(np.mean([r["margin"] for r in rows_scene])), 4),
            "loss": round(float(np.mean([np.mean(r["loss_true"]) for r in rows_scene])), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--spot", type=int, default=6, help="base rows re-scored before reusing v9c's")
    a = ap.parse_args()
    device = torch.device("cuda")
    reg = [e for e in json.loads((C.REGRESSION / "index.json").read_text()) if e["kind"] in ("expression", "motion")]
    scene = json.loads((C.SCENES / "index.json").read_text())
    emb = D.encode_captions([c for e in reg + scene for c in e["candidates"].values()], device)
    out = C.RESULTS / "dcls"; out.mkdir(parents=True, exist_ok=True)
    for name in a.configs:
        dest = out / f"{name.replace('/', '_').replace(':', '_')}_k{a.k}.json"
        if dest.exists():
            print(f"== {name}: cached -> {dest}"); continue
        t0 = time.time()
        model = D.load_model(name, device)
        rows_reg, reused = None, False
        if name == "v8merged" and a.k == 2 and C.V9C_BASE_DCLS.exists():
            old = {r["clip"]: r for r in json.loads(C.V9C_BASE_DCLS.read_text())["rows"]}
            spot = score_set(model, reg[:a.spot], emb, a.k, device, C.REGRESSION)
            rel = max(abs(np.mean(r["loss_true"]) - np.mean(old[r["clip"]]["loss_true"])) / np.mean(old[r["clip"]]["loss_true"])
                      for r in spot)
            print(f"   base reuse check: max relative loss difference {rel:.2e} over {len(spot)} clips")
            if rel < 1e-3 and all(e["clip"] in old for e in reg):
                rows_reg, reused = [old[e["clip"]] for e in reg], True
        if rows_reg is None:
            rows_reg = score_set(model, reg, emb, a.k, device, C.REGRESSION)
        rows_scene = score_set(model, scene, emb, a.k, device, C.SCENES)
        res = {"config": name, "k": a.k, "t_grid": D.T_GRID, "regression_reused_from_v9c": reused,
               "summary": summarise(rows_reg, rows_scene), "rows": rows_reg, "rows_scene": rows_scene,
               "seconds": round(time.time() - t0)}
        dest.write_text(json.dumps(res, indent=1))
        print(f"== {name} ({res['seconds']} s) -> {dest}")
        for key, v in res["summary"].items():
            print(f"   {key:<20} acc {v['acc']:.3f}  margin {v['margin']:+.4f}  loss {v['loss']:.5f}  n={v['n']}")
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
