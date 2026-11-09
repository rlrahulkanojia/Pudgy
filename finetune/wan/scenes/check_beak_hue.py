"""Screen training windows for the yellow-foot Pudgy design (v10 decision U9) by beak hue.

The yellow-foot variant is the current flat drawing with yellow feet and beak, hue >= ~42 deg.
The current design is salmon to amber, below ~40 deg (CURATION_v10.md). The beak is the
reliable place to measure: a small, compact, warm blob ringed by the white face, with an eye
above it. Feet are often hidden or cut off.

    PY=<python with opencv-python-headless + numpy>  PUDGY_DATA=<Data root>
    $PY scenes/check_beak_hue.py calib                     # iteration_7 labelled spans: yellow-foot vs kept
    $PY scenes/check_beak_hue.py scan <manifest.json|captions.jsonl> [--out rows.json]

`scan` reads a prep manifest (its `scene` list) or a captions jsonl (rows with skit, a, b),
samples 6 frames per window at 480x854 and prints the windows with any beak at >= 42 deg, plus
the count with no readable beak. Check both groups by eye: props (sparkles, yellow tools) can
pass as beaks, and night scenes, close-ups and back views give no reading.

Result on v9b_535's 348 windows (2026-11-09): median 26 deg, 5 hits (all props), 41 without a
beak (all current design by eye), so no yellow-foot design. Calibration: 23 of 24 yellow-foot
spans read >= 42 deg (the miss shows the character from behind), 0 of 120 kept spans.
"""
import glob
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path(os.environ.get("PUDGY_DATA", "/workspace/Data"))
CURATION = Path(__file__).resolve().parent / "curation"
YELLOW = 42.0
K2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def frame_at(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    return cv2.resize(f, (480, 854), interpolation=cv2.INTER_AREA) if ok else None


def _ring(lab, k, st, mask):
    """Share of `mask` in a 2-6 px ring around component k, plus the component's local mask."""
    x, y, w, h = st[k, :4]
    x0, y0, x1, y1 = max(0, x - 8), max(0, y - 8), min(480, x + w + 8), min(854, y + h + 8)
    m = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
    ring = (cv2.dilate(m, K6) > 0) & ~(cv2.dilate(m, K2) > 0)
    return (mask[y0:y1, x0:x1][ring].mean() if ring.any() else 0.0), ring, m, (x0, y0, x1, y1)


def beaks(bgr):
    """[(hue_deg, area, x, y)] for every beak-like blob in a 480x854 BGR frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white, dark = (S < 30) & (V > 215), V < 80
    nd, ld, sd, cd = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    eyes = np.array([cd[k] for k in range(1, nd)
                     if 4 <= sd[k, cv2.CC_STAT_AREA] <= 900 and _ring(ld, k, sd, white)[0] >= 0.55]).reshape(-1, 2)
    warm = ((H >= 8) & (H <= 33) & (S >= 110) & (V >= 150)).astype(np.uint8)   # 16-66 deg
    n, lab, st, cen = cv2.connectedComponentsWithStats(warm, 8)
    out = []
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        x, y, w, h = st[k, :4]
        if not (6 <= a <= 600) or a / float(w * h) < 0.4 or not (0.33 <= w / float(h) <= 3.5):
            continue
        wf, ring, m, (x0, y0, x1, y1) = _ring(lab, k, st, white)
        if wf < 0.45 or wf + dark[y0:y1, x0:x1][ring].mean() < 0.75:
            continue
        s, (cx, cy) = max(w, h, 6), cen[k]
        dx, dy = np.abs(eyes[:, 0] - cx), eyes[:, 1] - cy
        if not ((dx <= 7 * s) & (dy <= 0.5 * s) & (dy >= -6 * s) & (np.hypot(dx, dy) > 0.8 * s)).any():
            continue
        out.append((float(np.median(H[y0:y1, x0:x1][m > 0])) * 2, int(a), int(cx), int(cy)))
    return out


def probe(skit, a, b, k=6):
    cap = cv2.VideoCapture(str(DATA / "raw" / skit))
    hues = []
    for i in np.linspace(a, b - 1, k).round().astype(int):
        f = frame_at(cap, int(i))
        if f is not None:
            hues += [h for h, *_ in beaks(f)]
    return hues


def summary(hues):
    return dict(n=len(hues), med=float(np.median(hues)) if hues else None,
                n_yellow=sum(h >= YELLOW for h in hues))


def calib():
    pos, neg = [], []
    for f in sorted(glob.glob(str(CURATION / "iteration_7__*.json"))):
        d = json.load(open(f))
        for s in d["spans"]:
            nt = s.get("notes") or ""
            if s.get("drop_reason") == "off_style" and "OLD_DESIGN" in nt and "yellow" in nt.lower():
                pos.append((d["skit"], s["start"], s["end"]))
            elif s.get("keep"):
                neg.append((d["skit"], s["start"], s["end"]))
    random.seed(0)
    for name, spans in (("yellow-foot", pos), ("kept", random.sample(neg, min(120, len(neg))))):
        meds = [m for m in (summary(probe(*sp, k=5))["med"] for sp in spans) if m is not None]
        print(f"{name}: {len(spans)} spans, {len(meds)} with a beak, median {np.median(meds):.1f} deg, "
              f">= {YELLOW:.0f} deg: {sum(m >= YELLOW for m in meds)}")


def scan(path, out=None):
    p = Path(path)
    rows = ([json.loads(line) for line in open(p)] if p.suffix == ".jsonl" else json.load(open(p))["scene"])
    res = [dict(skit=r["skit"], a=r["a"], b=r["b"], name=r.get("name", ""), **summary(probe(r["skit"], r["a"], r["b"])))
           for r in rows]
    meds = [r["med"] for r in res if r["med"] is not None]
    print(f"{len(res)} windows: {len(meds)} with a beak (median of medians {np.median(meds):.1f} deg), "
          f"{len(res) - len(meds)} without")
    for r in res:
        if r["n_yellow"]:
            print(f"  >= {YELLOW:.0f} deg: {r['skit']} [{r['a']}, {r['b']}) {r['n_yellow']} of {r['n']} beaks, median {r['med']}")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["calib"]:
        calib()
    elif sys.argv[1:2] == ["scan"] and len(sys.argv) >= 3:
        scan(sys.argv[2], sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)
    else:
        sys.exit(__doc__)
