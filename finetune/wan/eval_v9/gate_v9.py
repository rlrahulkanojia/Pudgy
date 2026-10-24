#!/usr/bin/env python3
"""Pass/fail and checkpoint choice for v9c-phone, by a rule fixed BEFORE the data.

Inputs: dcls_v9 results (k=2, holdout+seen) for `v8merged` (the base, = v8@550) and for
each `v9c@<step>`. Per checkpoint, on the HOLDOUT split (QF2_R, never trained):

  regression  loss on the v8 labels (expression + motion) / v8merged's      must be <= 1.05
              (v9 section 4: earlier data may not get more than 5% worse)
  phone       loss on the held-out phone clips / v8merged's                 must be <  1.00
              (the stage must have learned something about this phone)
  binding     accuracy of the registry phrase against a generic phone and other objects
              (reported, and it breaks ties; not gated on its own - n = 4 clips)

Winner = the passing checkpoint with the lowest phone ratio (ties -> lower regression).
Writes /workspace/eval_v9/runs/<RUN_NAME>/winner_v9c.json; exits 1 if no checkpoint passes, in which case
the stage is re-run with more replay (prep_v9.py --replay-frac 0.35), never kept.

    python gate_v9.py                    # every v9c@* result found
    python gate_v9.py --steps 100 150 200
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402

REG_MAX = 1.05


def load(name, k):
    p = C.RESULTS / "dcls" / f"{name.replace('/', '_').replace(':', '_')}_k{k}.json"
    return json.loads(p.read_text()) if p.exists() else None


def mean_loss(res, split, kinds):
    v = [np.mean(r["loss_true"]) for r in res["rows"] if r["split"] == split and r["kind"] in kinds]
    return float(np.mean(v)) if v else float("nan"), len(v)


def paired_ratio(res, base, split, kinds):
    """Mean over clips of loss(config) / loss(base) - paired, since the noise draws are
    identical across configurations (dcls common random numbers)."""
    b = {r["clip"]: np.mean(r["loss_true"]) for r in base["rows"] if r["split"] == split and r["kind"] in kinds}
    v = [np.mean(r["loss_true"]) / b[r["clip"]] for r in res["rows"]
         if r["split"] == split and r["kind"] in kinds and r["clip"] in b]
    return float(np.mean(v)) if v else float("nan"), len(v)


def acc(res, split, kinds):
    v = [r["pred"] == r["label"] for r in res["rows"] if r["split"] == split and r["kind"] in kinds]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*")
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    base = load("v8merged", a.k)
    if base is None:
        sys.exit("no dcls result for v8merged - run: python dcls_v9.py --configs v8merged")
    if a.steps:
        names = [f"v9c@{s}" for s in a.steps]
    else:
        found = sorted((C.RESULTS / "dcls").glob(f"v9c@*_k{a.k}.json"))
        names = [re.sub(rf"_k{a.k}$", "", p.stem) for p in found]
    rows = []
    print(f"{'config':<14} {'regress':>8} {'phone':>7} {'bind':>5} {'bind_b':>6} {'v8 acc':>6}  verdict")
    b_bind = acc(base, "holdout", {"prop"})
    for name in names:
        res = load(name, a.k)
        if res is None:
            print(f"{name:<14} (no dcls result)"); continue
        reg, n_reg = paired_ratio(res, base, "holdout", {"expression", "motion"})
        ph, n_ph = paired_ratio(res, base, "holdout", {"prop"})
        bind = acc(res, "holdout", {"prop"})
        v8acc = acc(res, "holdout", {"expression", "motion"})
        ok = reg <= REG_MAX and ph < 1.0
        rows.append({"config": name, "regression_ratio": reg, "n_regression": n_reg,
                     "phone_ratio": ph, "n_phone": n_ph, "binding_acc": bind,
                     "binding_acc_base": b_bind, "v8_label_acc": v8acc,
                     "v8_label_acc_base": acc(base, "holdout", {"expression", "motion"}),
                     "seen_phone_ratio": paired_ratio(res, base, "seen", {"prop"})[0],
                     "pass": ok})
        print(f"{name:<14} {reg:8.3f} {ph:7.3f} {bind:5.2f} {b_bind:6.2f} {v8acc:6.2f}  "
              f"{'PASS' if ok else 'fail'}")
    passing = [r for r in rows if r["pass"]]
    out = {"rule": f"regression <= {REG_MAX} and phone < 1.0 on holdout; min phone ratio",
           "base": "v8merged", "candidates": rows}
    if passing:
        win = min(passing, key=lambda r: (round(r["phone_ratio"], 3), -r["binding_acc"],
                                          r["regression_ratio"]))
        out["winner"] = win["config"]
        print(f"\nwinner: {win['config']}")
    (C.RESULTS / "winner_v9c.json").write_text(json.dumps(out, indent=1))
    if not passing:
        print("\nno checkpoint passes - re-run the stage with more replay "
              "(prep_v9.py --replay-frac 0.35); do not keep this one")
        sys.exit(1)
    print(out["winner"])


if __name__ == "__main__":
    main()
