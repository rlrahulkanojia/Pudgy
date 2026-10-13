#!/usr/bin/env python3
"""PLAN §6.3-6.4 motion probe, scored: per configuration, the motion energy of its 10 probe
clips (eval/gen_core.py --suite probe: 5 motion labels x 2 characters, QF2_R, seed 42, f13),
against the real QF2_R source clip of the same character and label at the same window
(refs/tier1/f13). Then the D1 trigger (§6.4): median energy falls >= 25% from its best probed
checkpoint across >= 3 consecutive probed checkpoints -> train arm B.

    python eval/motion_probe.py --configs base expv1@100 expv1@200 ... expv1@600
Writes EVAL/reports/motion_probe.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv1_common as C  # noqa: E402
import metrics as M  # noqa: E402

W = C.H2H_WINDOW["motion"]
WHITE = C.GROUNDS["white"]


def energy(frames):
    m, _ = M.clip_measures(frames[:W], WHITE)
    return m["motion_energy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    a = ap.parse_args()
    src = {}
    for char in C.CHARS:
        for lab in C.MOTION_LABELS:
            p = C.REFS / "tier1" / f"f{W}" / f"{char.lower()}_{lab}_{C.HOLDOUT_ANGLE}__white_f{W}.mp4"
            src[(char, lab)] = energy(M.read_video(p, size=C.SIZE))
    out = {"window": W, "source": {f"{c}/{l}": e for (c, l), e in src.items()},
           "source_median": float(np.median(list(src.values()))), "configs": {}}
    for cfg in a.configs:
        d = C.EVAL / "gen" / cfg.replace("/", "_").replace(":", "_") / "core"
        rows = []
        for char in C.CHARS:
            for lab in C.MOTION_LABELS:
                p = d / f"{char.lower()}_{lab}_{C.HOLDOUT_ANGLE}_f{W}_s42.mp4"
                if not p.exists():
                    sys.exit(f"missing probe clip {p}")
                e = energy(M.read_video(p, size=C.SIZE))
                rows.append({"char": char, "label": lab, "energy": e, "ratio": e / src[(char, lab)]})
        out["configs"][cfg] = {"median_energy": float(np.median([r["energy"] for r in rows])),
                               "median_ratio": float(np.median([r["ratio"] for r in rows])),
                               "rows": rows}
    # D1 trigger over the trained checkpoints, in step order
    ck = sorted((c for c in out["configs"] if c.startswith("expv1@")), key=lambda c: int(c.split("@")[1]))
    e = [out["configs"][c]["median_energy"] for c in ck]
    run, trig = 0, False
    for i in range(len(e)):
        run = run + 1 if e[i] <= 0.75 * max(e[:i + 1]) else 0
        trig |= run >= 3
    out["d1"] = {"checkpoints": ck, "median_energy": e, "triggered": trig,
                 "rule": "median energy <= 0.75x its best-so-far on >= 3 consecutive probed checkpoints"}
    C.write_json(C.EVAL / "reports" / "motion_probe.json", out)
    print(f"source (real QF2_R, f{W}) median energy {out['source_median']:.3f}")
    for c, v in out["configs"].items():
        print(f"{c:<11} median energy {v['median_energy']:.3f}  median ratio to source {v['median_ratio']:.2f}")
    print(f"D1 trigger (train arm B): {'YES' if trig else 'no'}")


if __name__ == "__main__":
    main()
