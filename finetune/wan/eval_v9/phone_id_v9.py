#!/usr/bin/env python3
"""Phone-identity check for generated clips (gate bar ID1, RESEARCH_v9c_fixes section 5).

In every training clip the iFin phone shows its back: a pale, slightly blue grey (about
200,208,216). The ground shadow is neutral grey and the characters and backgrounds are
saturated or white, so "back-face pixels" are a clean colour class. A clip is DEFECTIVE when
the visible back face falls below `--keep` of its frame-0 area for `--run` consecutive frames:
that catches the phone turning to show its screen, going edge-on, or disappearing, whatever
is drawn on the screen.

    python phone_id_v9.py clips/*.mp4                  # per-clip table + defect rate
    python phone_id_v9.py --json out.json <dir>/*.mp4
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def frames(path, size=512):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={size}:{size}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3).astype(np.int16)


def back_face(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(-1)
    sat = v - f.min(-1)
    return (b >= g) & (g >= r - 2) & (b - r >= 6) & (b - r <= 32) & (sat <= 32) & (v >= 170) & (v <= 240)


def score(path, keep=0.35, run=4):
    fr = frames(path)
    area = np.array([back_face(f).sum() for f in fr], dtype=float)
    a0 = max(area[0], 1.0)
    ratio = area / a0
    low = ratio < keep
    longest = cur = 0
    for x in low:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return {"clip": path, "area0": int(area[0]), "min_ratio": round(float(ratio.min()), 3),
            "final_ratio": round(float(ratio[-1]), 3), "longest_low_run": int(longest),
            "defect": bool(longest >= run)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--keep", type=float, default=0.35)
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [score(p, a.keep, a.run) for p in a.clips]
    for r in rows:
        print(f"{r['clip'].split('/')[-1][:58]:58s} area0 {r['area0']:6d}  min {r['min_ratio']:.2f}  "
              f"final {r['final_ratio']:.2f}  run {r['longest_low_run']:3d}  {'DEFECT' if r['defect'] else 'ok'}")
    n = sum(r["defect"] for r in rows)
    print(f"defect rate {n}/{len(rows)}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
