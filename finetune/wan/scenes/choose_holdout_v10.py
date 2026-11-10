#!/usr/bin/env python3
"""Choose the v10 holdout: whole iteration_7 skits that are never trained (Training_Approach_v10
section 3.6). Writes scenes/holdout_v10.json. Seeded and deterministic.

Works on the rows that may train (rows_v10.load_rows: trainable, minus Q4 and Q5). Per skit:
- G1 shots: one per span, the span's first window of >= 49 frames (an I2V job for G1/G2);
- Tier-1 windows: spans with a window of >= 33 frames (G3 scene windows, cut at f33);
- strata: has a two-shot, has a camera move (U3), and its main setting kind (flat or set).

Eligible: >= 1 G1 shot, >= 1 Tier-1 window, and a window count between the 25th and 75th
percentile, so the holdout neither starves training nor is trivial. The target (default 16,
about 10% of the skits) is allocated to strata in proportion to their size (largest
remainder), and drawn at random within each stratum with the seed. If the held-out G1 shots
plus v9b's 8 Tier-2 jobs fall short of --min-g1 shots (27 x 3 seeds >= 80 G1 pairs), more
skits are added from the largest strata until they don't.

    PUDGY_DATA=<Data> python choose_holdout_v10.py [--seed 10] [--target 16] [--write]
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from rows_v10 import curation, load_rows

HERE = Path(__file__).resolve().parent
V9B_T2_JOBS = 8          # v9b's Tier-2 jobs from its 6 held-out skits (build_eval_v9b.py), reused for G1


def skit_features(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["skit"]].append(r)
    feats = []
    for skit, rs in by.items():
        d, _ = curation(rs[0]["curation"])
        spans = defaultdict(list)
        for r in rs:
            spans[r["span"]].append(r)
        g1 = [min((w for w in ws if w["frames"] >= 49), key=lambda w: w["a"]) for ws in spans.values()
              if any(w["frames"] >= 49 for w in ws)]
        t1 = [sp for sp, ws in spans.items() if any(w["frames"] >= 33 for w in ws)]
        kinds = defaultdict(int)
        for r in rs:
            kinds[d["settings"][d["spans"][r["span"]]["setting"]]["kind"]] += r["frames"]
        feats.append({
            "skit": skit, "windows": len(rs), "frames": sum(r["frames"] for r in rs),
            "g1": len(g1), "t1": len(t1),
            "stratum": ("two_shot" if any(r["two_shot"] for r in rs) else "one_char",
                        "camera" if any(r["camera"] != "static" for r in rs) else "static",
                        max(kinds, key=kinds.get)),
        })
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--target", type=int, default=16)
    ap.add_argument("--min-g1", type=int, default=27)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows, _ = load_rows(os.environ.get("PUDGY_DATA", "/workspace/Data"))
    feats = skit_features(rows)
    nw = np.array([f["windows"] for f in feats])
    lo, hi = np.percentile(nw, 25), np.percentile(nw, 75)
    elig = [f for f in feats if f["g1"] >= 1 and f["t1"] >= 1 and lo <= f["windows"] <= hi]
    strata = defaultdict(list)
    for f in sorted(elig, key=lambda f: f["skit"]):
        strata[f["stratum"]].append(f)
    rng = random.Random(a.seed)
    for v in strata.values():
        rng.shuffle(v)
    # proportional allocation over ALL skits' strata, largest remainder
    pop = defaultdict(int)
    for f in feats:
        pop[f["stratum"]] += 1
    quota = {k: a.target * n / len(feats) for k, n in pop.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - alloc[k], reverse=True)[:a.target - sum(alloc.values())]:
        alloc[k] += 1
    pick = []
    for k, n in sorted(alloc.items()):
        take = strata.get(k, [])[:n]
        if len(take) < n:
            print(f"  stratum {k}: wanted {n}, only {len(take)} eligible")
        pick += take
        strata[k] = strata.get(k, [])[n:]
    while sum(f["g1"] for f in pick) + V9B_T2_JOBS < a.min_g1:
        k = max((k for k in strata if strata[k]), key=lambda k: pop[k] / (1 + sum(f["stratum"] == k for f in pick)))
        pick.append(strata[k].pop(0))
    n_g1 = sum(f["g1"] for f in pick)
    held_w = sum(f["windows"] for f in pick)
    print(f"{len(feats)} skits train-eligible, {len(elig)} eligible for holdout (windows {lo:.0f}-{hi:.0f})")
    for f in sorted(pick, key=lambda f: f["stratum"]):
        print(f"   {f['skit'].split('/')[-1]:<44} windows {f['windows']:>2}  G1 {f['g1']}  T1 {f['t1']}  {'/'.join(f['stratum'])}")
    print(f"holdout: {len(pick)} skits, {held_w} of {len(rows)} windows ({100 * held_w / len(rows):.1f}%), "
          f"G1 shots {n_g1} + v9b {V9B_T2_JOBS} = {n_g1 + V9B_T2_JOBS} (x 3 seeds = {3 * (n_g1 + V9B_T2_JOBS)} pairs)")
    out = {"skits": sorted(f["skit"] for f in pick), "seed": a.seed, "target": a.target,
           "rule": "choose_holdout_v10.py: rows_v10 rows; eligible = >= 1 G1 shot (window >= 49 f), >= 1 Tier-1 "
                   "window (>= 33 f), window count in the 25-75th percentile; proportional allocation over "
                   "(two-shot, camera move, main setting kind) strata; topped up until G1 shots + v9b's 8 >= min_g1",
           "windows_held": held_w, "windows_total": len(rows), "g1_shots": n_g1, "v9b_t2_jobs": V9B_T2_JOBS,
           "plus_v9b_holdout": json.loads((HERE / "holdout_v9b.json").read_text())["skits"]}
    if a.write:
        (HERE / "holdout_v10.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {HERE / 'holdout_v10.json'}")


if __name__ == "__main__":
    main()
