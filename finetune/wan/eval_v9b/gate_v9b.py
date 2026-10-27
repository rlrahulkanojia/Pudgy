#!/usr/bin/env python3
"""Pass/fail and checkpoint choice for v9b, by a rule fixed BEFORE the data
(Training_Approach_v9.md section 5.4; bars from RESEARCH_v9c_fixes section 5, which the
v9c-phone run showed an averaged loss ratio cannot replace).

Inputs: tier1_v9b results (k = 2) for `v8merged` and each `v9b@<step>`. Per checkpoint:

  R1  regression, per kind (expression, motion), HOLDOUT split: per-clip paired change in
      margin vs base, 5,000-sample bootstrap. 95% CI lower bound >= -0.003 AND top-1
      accuracy down by at most 1 clip. Per-label margins are reported, not gated (n = 4-7
      clips each): a v8 clause that a scene word absorbed shows up by name (D10).
  R2  regression, v8 labels on HOLDOUT: paired loss ratio vs base at EACH t of the grid
      <= 1.05 (v9c-phone was 1.039 at t=0.65 but 1.079 at t=0.85).
  L1  learned: held-out scene windows, loss under the true caption / base < 1.00.
  P2  uses the caption: per window, the change in log loss under the true caption minus the
      mean change under the distractor captions (beats reversed, leads swapped, another
      window's action). 95% bootstrap CI upper bound < 0.

Winner = the passing checkpoint with the lowest L1 (ties -> higher R1 lower bound).
Writes <RESULTS>/winner_v9b.json and prints the winner's config name as the LAST line;
exits 1 if no checkpoint passes (then the stage is not kept: RUNBOOK_v9b section 7).

    python gate_v9b.py [--steps 200 400 600 800]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9b as C  # noqa: E402

B = 5000
R1_LO, R1_ACC_DROP, R2_MAX, L1_MAX = -0.003, 1, 1.05, 1.00


def load(name, k=2):
    p = C.RESULTS / "dcls" / f"{name.replace('/', '_').replace(':', '_')}_k{k}.json"
    return json.loads(p.read_text()) if p.exists() else None


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def r1(res, base, kind):
    b = {r["clip"]: r for r in base["rows"] if r["split"] == "holdout" and r["kind"] == kind}
    rs = [r for r in res["rows"] if r["split"] == "holdout" and r["kind"] == kind and r["clip"] in b]
    d = [r["margin"] - b[r["clip"]]["margin"] for r in rs]
    mean, lo, hi = boot_ci(d)
    acc_b = sum(b[r["clip"]]["pred"] == r["label"] for r in rs)
    acc_r = sum(r["pred"] == r["label"] for r in rs)
    per_label = {}
    for lab in sorted({r["label"] for r in rs}):
        per_label[lab] = round(float(np.mean([r["margin"] - b[r["clip"]]["margin"] for r in rs if r["label"] == lab])), 4)
    return {"n": len(rs), "dmargin": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)],
            "acc_base": acc_b, "acc": acc_r, "per_label_dmargin": per_label,
            "pass": lo >= R1_LO and acc_b - acc_r <= R1_ACC_DROP}


def r2(res, base, tgrid):
    b = {r["clip"]: r for r in base["rows"] if r["split"] == "holdout"}
    rs = [r for r in res["rows"] if r["split"] == "holdout" and r["clip"] in b]
    ratios = {}
    for k in range(len(rs[0]["loss_true"])):
        ratios[str(tgrid[k % len(tgrid)])] = round(float(np.mean([r["loss_true"][k] / b[r["clip"]]["loss_true"][k] for r in rs])), 4)
    return {"per_t": ratios, "pass": all(v <= R2_MAX for v in ratios.values())}


def scene_bars(res, base):
    b = {r["clip"]: r for r in base["rows_scene"]}
    rs = [r for r in res["rows_scene"] if r["clip"] in b]
    ratio = float(np.mean([np.mean(r["loss_true"]) / np.mean(b[r["clip"]]["loss_true"]) for r in rs]))
    g = []
    for r in rs:
        bb = b[r["clip"]]
        dt = np.log(r["E"]["true"]) - np.log(bb["E"]["true"])
        dd = [np.log(r["E"][c]) - np.log(bb["E"][c]) for c in r["E"] if c != "true"]
        if dd:
            g.append(dt - float(np.mean(dd)))
    mean, lo, hi = boot_ci(g) if g else (float("nan"),) * 3
    return ({"ratio": round(ratio, 4), "n": len(rs), "pass": ratio < L1_MAX},
            {"gain": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(g), "pass": bool(g) and hi < 0},
            {"acc_base": round(float(np.mean([b[r["clip"]]["pred"] == "true" for r in rs])), 3),
             "acc": round(float(np.mean([r["pred"] == "true" for r in rs])), 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*")
    a = ap.parse_args()
    base = load("v8merged")
    if base is None:
        sys.exit("no v8merged Tier-1 result - run: v9b_box.sh baseline")
    if a.steps:
        names = [f"v9b@{s}" for s in a.steps]
    else:
        found = sorted((C.RESULTS / "dcls").glob("v9b@*_k2.json"),
                       key=lambda p: int(re.search(r"@(\d+)", p.name).group(1)) if re.search(r"@(\d+)", p.name) else 1e9)
        names = [p.name[:-len("_k2.json")] for p in found]
    report = {}
    print(f"{'config':<12} {'R1 expr':>16} {'R1 motion':>16} {'R2 per-t':>22} {'L1':>7} {'P2 gain [CI]':>26}  verdict")
    for n in names:
        res = load(n)
        if res is None:
            print(f"{n:<12} (no result)"); continue
        e, m = r1(res, base, "expression"), r1(res, base, "motion")
        t2 = r2(res, base, res["t_grid"])
        l1, p2, sacc = scene_bars(res, base)
        ok = e["pass"] and m["pass"] and t2["pass"] and l1["pass"] and p2["pass"]
        report[n] = {"R1_expression": e, "R1_motion": m, "R2": t2, "L1": l1, "P2": p2, "scene_acc": sacc, "pass": ok}
        fmt = lambda x: f"{x['dmargin']:+.4f}[{x['ci'][0]:+.3f}]{'✓' if x['pass'] else '✗'}"   # noqa: E731
        print(f"{n:<12} {fmt(e):>16} {fmt(m):>16} {str(list(t2['per_t'].values())) + ('✓' if t2['pass'] else '✗'):>22} "
              f"{l1['ratio']:.3f}{'✓' if l1['pass'] else '✗'} {p2['gain']:+.4f} [{p2['ci'][0]:+.3f},{p2['ci'][1]:+.3f}]{'✓' if p2['pass'] else '✗'}  "
              f"{'PASS' if ok else 'fail'}")
        print(f"{'':<12} per-label Δmargin motion {m['per_label_dmargin']}  expr {e['per_label_dmargin']}  "
              f"scene acc {sacc['acc_base']}→{sacc['acc']}")
    passing = [n for n in report if report[n]["pass"]]
    win = min(passing, key=lambda n: (report[n]["L1"]["ratio"], -min(report[n]["R1_expression"]["ci"][0],
                                                                      report[n]["R1_motion"]["ci"][0]))) if passing else None
    (C.RESULTS / "winner_v9b.json").write_text(json.dumps({"winner": win, "report": report,
                                                            "bars": {"R1_lo": R1_LO, "R1_acc_drop": R1_ACC_DROP,
                                                                     "R2_max": R2_MAX, "L1_max": L1_MAX}}, indent=1))
    if not win:
        print("NO CHECKPOINT PASSES")
        sys.exit(1)
    print(win)


if __name__ == "__main__":
    main()
