#!/usr/bin/env python3
"""Tier-1 for v9c-phone: the v8 diffusion classifier on the v9 clip set.

The scoring is eval_v8/dcls_v8.py unchanged (same t-grid, same common random numbers, so
two configurations are a PAIRED comparison); only the clip set (build_assets_v9.py) and
the model configurations (common_v9.lora_config) are swapped in. Reported per split and
kind - expression / motion are the regression gate, prop is the phone:

  acc / rank / margin   does the true caption explain the clip best (prop: the registry
                        phrase against a generic phone and other objects = name binding)
  loss                  held-out denoising loss under the true caption

    python dcls_v9.py --configs v8merged v8@550            # merge check: must match
    python dcls_v9.py --configs v8merged v9c@100 v9c@200 --split holdout seen
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402  (puts eval_v8 on sys.path)
import dcls_v8 as D  # noqa: E402
import torch  # noqa: E402

D.HOLDOUT = C.HOLDOUT              # load_latents() reads the cache under this
D.lora_config = C.lora_config      # load_model() resolves configurations through this


def summarise(rows):
    out = {}
    for split in sorted({r["split"] for r in rows}):
        for kind in ("expression", "motion", "prop"):
            rs = [r for r in rows if r["split"] == split and r["kind"] == kind]
            if not rs:
                continue
            out[f"{split}/{kind}"] = {
                "n": len(rs), "chance": round(1 / len(rs[0]["E"]), 3),
                "acc": round(float(np.mean([r["pred"] == r["label"] for r in rs])), 3),
                "rank": round(float(np.mean([r["rank"] for r in rs])), 2),
                "margin": round(float(np.mean([r["margin"] for r in rs])), 4),
                "loss": round(float(np.mean([np.mean(r["loss_true"]) for r in rs])), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=2, help="(t, eps) draws per clip")
    ap.add_argument("--split", nargs="*", default=["holdout", "seen"])
    ap.add_argument("--kind", choices=("expression", "motion", "prop"))
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    ap.add_argument("--out", default=str(C.RESULTS / "dcls"))
    a = ap.parse_args()
    device = torch.device("cuda")
    index = [e for e in json.loads((C.HOLDOUT / "index.json").read_text())
             if e["split"] in a.split and (not a.kind or e["kind"] == a.kind)]
    if a.limit:
        index = index[:a.limit]
    emb = D.encode_captions([c for e in index for c in e["candidates"].values()], device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in a.configs:
        dest = out / f"{name.replace('/', '_').replace(':', '_')}_k{a.k}.json"
        if a.kind or a.limit or set(a.split) != {"holdout", "seen"}:
            dest = dest.with_name(f"{dest.stem}_{a.kind or 'all'}_{'-'.join(a.split)}_n{len(index)}.json")
        if dest.exists():
            print(f"== {name}: cached -> {dest}"); continue
        t0 = time.time()
        model = D.load_model(name, device)
        rows = [D.score_clip(model, e, emb, a.k, device) for e in index]
        res = {"config": name, "k": a.k, "t_grid": D.T_GRID, "summary": summarise(rows),
               "rows": rows, "seconds": round(time.time() - t0)}
        dest.write_text(json.dumps(res, indent=1))
        print(f"== {name} ({res['seconds']} s) -> {dest}")
        for key, v in res["summary"].items():
            print(f"   {key:<20} acc {v['acc']:.3f} (chance {v['chance']})  rank {v['rank']}  "
                  f"margin {v['margin']:+.4f}  loss {v['loss']:.5f}  n={v['n']}")
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
