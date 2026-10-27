#!/usr/bin/env python3
"""Choose the v9b holdout: 6 whole skits (3 per iteration) that are never trained
(Training_Approach_v9.md section 3.2 #6, section 0.1 item 6). Run once, after curation;
writes scenes/holdout_v9b.json. Deterministic (seeded), so the choice can be re-derived.

Eligible skits: at least 2 kept spans of >= 33 frames (Tier-1 windows) and at least one of
>= 49 frames (a Tier-2 job), and kept time between the 25th and 75th percentile, so the
holdout neither starves training nor is trivially small. Within each iteration, the
3 skits are chosen greedily to cover, in order of priority: a two-shot, a one-character
span, a flat-ground setting, a full set, a prop event; ties are broken by the seed.

    /workspace/Pudgy/.venv-wan/bin/python choose_holdout.py [--seed 9] [--write]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEADS = {"pax", "polly"}


def features(d):
    kept = [s for s in d["spans"] if s.get("keep")]
    f = {"skit": d["skit"], "kept_frames": sum(s["end"] - s["start"] for s in kept),
         "t1": sum(s["end"] - s["start"] >= 33 for s in kept), "t2": sum(s["end"] - s["start"] >= 49 for s in kept)}
    tags = set()
    for s in kept:
        n = sum(c["id"] in LEADS for c in s["characters"])
        tags.add("two_shot" if n == 2 else "one_char")
        tags.add(f"setting_{d['settings'][s['setting']]['kind']}")
        if s.get("prop_events"):
            tags.add("prop_event")
    f["tags"] = sorted(tags)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = json.loads((HERE / "skits_v9b.json").read_text())
    missing = [r["skit"] for r in rows if not (HERE / r["out"]).exists()]
    if missing and a.write:
        raise SystemExit(f"{len(missing)} skits not curated yet - the holdout is chosen once, from all 99")
    feats = [features(json.loads((HERE / r["out"]).read_text())) for r in rows if (HERE / r["out"]).exists()]
    kf = np.array([f["kept_frames"] for f in feats])
    lo, hi = np.percentile(kf, 25), np.percentile(kf, 75)
    want = ["two_shot", "one_char", "setting_flat", "setting_set", "prop_event"]
    rng = random.Random(a.seed)
    chosen = []
    for it in ("iteration_1", "iteration_2"):
        pool = [f for f in feats if f["skit"].startswith(it) and f["t1"] >= 2 and f["t2"] >= 1 and lo <= f["kept_frames"] <= hi]
        rng.shuffle(pool)
        if len(pool) < 3:
            raise SystemExit(f"{it}: only {len(pool)} eligible skits (need 3)")
        pick, covered = [], set()
        for _ in range(3):
            best = max(pool, key=lambda f: sum((t in f["tags"]) * (len(want) - i) for i, t in enumerate(want) if t not in covered))
            pick.append(best); covered |= set(best["tags"]); pool.remove(best)
        chosen += pick
        print(f"{it}: {len(pool) + 3} eligible (kept {lo:.0f}-{hi:.0f} frames)")
        for f in pick:
            print(f"   {f['skit']:<55} kept {f['kept_frames']:>4} f  t1 {f['t1']} t2 {f['t2']}  {f['tags']}")
    out = {"skits": [f["skit"] for f in chosen], "seed": a.seed,
           "rule": "choose_holdout.py: 3 per iteration, kept time in the 25-75th percentile, >= 2 Tier-1 and >= 1 "
                   "Tier-2 spans, greedy coverage of two-shot / one-character / flat / set / prop event",
           "kept_frames": int(sum(f["kept_frames"] for f in chosen)), "of_total_kept": int(kf.sum())}
    print(f"holdout keeps {out['kept_frames']} of {out['of_total_kept']} kept frames "
          f"({100 * out['kept_frames'] / out['of_total_kept']:.0f}%) out of training")
    if a.write:
        (HERE / "holdout_v9b.json").write_text(json.dumps(out, indent=1))
        print(f"wrote {HERE / 'holdout_v9b.json'}")


if __name__ == "__main__":
    main()
