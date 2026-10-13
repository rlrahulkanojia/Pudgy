#!/usr/bin/env python3
"""PLAN §6.3 gate-break rules, applied to Tier-1 rows (eval/dcls.py, layout A).

  step 100: both kinds' held-out true-caption losses below the base's on >= 75% of clips
            (paired per clip, same noise). FAIL -> stop and re-check the input layout before
            touching LR; if the loss is ABOVE the base on most clips -> restart at LR 5e-5.
  step 300: margins (accuracy, mean margin) above the base's; reported, not a stop rule.

    python eval/gate_rule.py --step 100 [--config expv1@100]
Exit 0 = continue training, 2 = stop. Writes EVAL/reports/gate_<step>.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv1_common as C  # noqa: E402
import stats as S  # noqa: E402

DCLS = C.EVAL / "dcls"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=str(DCLS / "base__A_k2.json"))
    a = ap.parse_args()
    cfg = a.config or f"expv1@{a.step}"
    ck = C.load_json(DCLS / f"{cfg}__A_k2.json")
    base = C.load_json(a.base)
    if not ck or not base:
        sys.exit(f"missing Tier-1 rows: {cfg} or base")
    b = {r["clip"]: r for r in base["rows"] if r["split"] == "holdout"}
    out = {"step": a.step, "config": cfg, "kinds": {}}
    ok_all, above = True, False
    for kind in ("expression", "motion"):
        rs = [r for r in ck["rows"] if r["split"] == "holdout" and r["kind"] == kind and r["clip"] in b]
        d = np.array([np.mean(b[r["clip"]]["loss_true"]) - np.mean(r["loss_true"]) for r in rs])  # > 0: below base
        frac = float((d > 0).mean())
        acc = float(np.mean([r["pred"] == r["label"] for r in rs]))
        acc_b = float(np.mean([b[r["clip"]]["pred"] == r["label"] for r in rs]))
        out["kinds"][kind] = {"n": len(rs), "frac_below_base": frac, "mean_loss_drop": float(d.mean()),
                              "rel_loss_drop": float(d.mean() / np.mean([np.mean(b[r["clip"]]["loss_true"]) for r in rs])),
                              "wilcoxon_p_below": S.wilcoxon_p(d, "greater"), "ci95_drop": S.bootstrap_ci(d),
                              "acc": acc, "acc_base": acc_b,
                              "margin": float(np.mean([r["margin"] for r in rs])),
                              "margin_base": float(np.mean([b[r["clip"]]["margin"] for r in rs]))}
        if frac < 0.75:
            ok_all = False
        if frac < 0.5:
            above = True
    if a.step <= 100:
        out["rule"] = "both kinds below base on >= 75% of held-out clips"
        out["pass"] = ok_all
        out["action"] = ("continue" if ok_all else
                         "STOP: loss above base on most clips -> restart at LR 5e-5" if above else
                         "STOP: re-check the input layout before touching LR")
    else:
        out["rule"] = "margins above base (reported)"
        out["pass"] = all(v["margin"] > v["margin_base"] for v in out["kinds"].values())
        out["action"] = "continue"
    C.write_json(C.EVAL / "reports" / f"gate_{a.step}.json", out)
    for k, v in out["kinds"].items():
        print(f"step {a.step} {k:<10} below base on {v['frac_below_base']:.0%}  loss -{v['rel_loss_drop']:.1%}  "
              f"acc {v['acc']:.3f} (base {v['acc_base']:.3f})  margin {v['margin']:+.4f} (base {v['margin_base']:+.4f})")
    print(f"-> {'PASS' if out['pass'] else 'FAIL'}: {out['action']}")
    sys.exit(0 if out["action"] == "continue" else 2)


if __name__ == "__main__":
    main()
