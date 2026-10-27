#!/usr/bin/env python3
"""Long-baseline camera check on every KEPT span (D3), for the verification pass.

keep12 (curate_tools profile) compares frames 12 apart, so slow push-ins, zooms and pans
pass it; curator-01 caught several, including the pilot example (Chair [0,86), +0.3% scale
per 10 frames). Following curator-01's method: fit a similarity transform between the
span's FIRST and LAST frames (and first -> middle) from ORB features + RANSAC, at 540x960,
with the leads masked out (their body colours, as longform/dupcheck.py finds them, dilated)
so a moving character cannot pose as the camera. A locked camera measures scale 1.000 and
~0 px; a drift accumulates over the span.

  moving        |scale - 1| > 0.006 or shift > 4 px (1080-wide pixels), first->last or first->mid
  unverifiable  < 20 background inliers (plain grounds): judged by eye (character size, shadow)

    /workspace/Pudgy/.venv-wan/bin/python span_motion_check.py [curation/*.json] [--calibrate]
"""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curate_tools as T  # noqa: E402

W, H = 540, 960
PAX = ((100, 90, 120), (125, 255, 255))
POLLY = ((150, 40, 150), (178, 255, 255))
SCALE_TOL, SHIFT_TOL, MIN_INL = 0.006, 4.0, 20


def decode_rgb(path):
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path), "-vf", f"scale={W}:{H}:flags=area",
                        *T._passthrough(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, H, W, 3)


def bg_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lead = cv2.inRange(hsv, *PAX) | cv2.inRange(hsv, *POLLY)
    lead = cv2.dilate(lead, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return cv2.bitwise_not(lead)


ORB = None


def fit(a_rgb, b_rgb):
    global ORB
    ORB = ORB or cv2.ORB_create(4000)
    ga, gb = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    ka, da = ORB.detectAndCompute(ga, bg_mask(a_rgb))
    kb, db = ORB.detectAndCompute(gb, bg_mask(b_rgb))
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 8:
        return None
    A = np.float32([ka[x.queryIdx].pt for x in m]); B = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if M is None:
        return None
    s = float(np.hypot(M[0, 0], M[1, 0]))
    # shift of the image centre under M, in 1080-wide pixels
    c = np.array([W / 2, H / 2, 1.0])
    dx, dy = (M @ c) - c[:2]
    return {"scale": round(s, 4), "shift": round(float(np.hypot(dx, dy)) * 1080 / W, 1), "inliers": int(inl.sum())}


def verdict(fits):
    good = [f for f in fits if f and f["inliers"] >= MIN_INL]
    if not good:
        return "unverifiable"
    return "moving" if any(abs(f["scale"] - 1) > SCALE_TOL or f["shift"] > SHIFT_TOL for f in good) else "locked"


def span_fits(v, s, e):
    mid = s + (e - 1 - s) // 2
    return [fit(v[s], v[e - 1]), fit(v[s], v[mid])]


def series(v, s, e, stride=2):
    """first -> every `stride`-th frame: catches a shake or a drift that starts or ends
    inside the span (verifier set 1: ActingTough's impact shake, Germs' late drift).
    Returns the offending frames and the worst fit."""
    bad, worst, n_ok = [], None, 0
    for k in list(range(s + stride, e, stride)) + [e - 1]:
        f = fit(v[s], v[k])
        if not f or f["inliers"] < MIN_INL:
            continue
        n_ok += 1
        off = abs(f["scale"] - 1) > SCALE_TOL or f["shift"] > SHIFT_TOL
        if off:
            bad.append(k)
            if worst is None or f["shift"] + 1000 * abs(f["scale"] - 1) > worst["shift"] + 1000 * abs(worst["scale"] - 1):
                worst = dict(f, frame=k)
    return {"checked": n_ok, "bad_frames": [bad[0], bad[-1]] if bad else None, "n_bad": len(bad), "worst": worst}


def check_file(path):
    d = json.loads(Path(path).read_text())
    v = decode_rgb(T.skit_path(d["skit"]))
    out = []
    for i, sp in enumerate(d["spans"]):
        if sp.get("keep"):
            fl, fm = span_fits(v, sp["start"], sp["end"])
            se = series(v, sp["start"], sp["end"])
            vd = verdict([fl, fm])
            if se["n_bad"] and vd == "locked":
                vd = "moving_inside"
            if not se["checked"] and vd == "locked":
                vd = "unverifiable"
            out.append({"file": Path(path).name, "skit": d["skit"], "span": i, "start": sp["start"], "end": sp["end"],
                        "framing": sp.get("framing"), "first_last": fl, "first_mid": fm, "series": se, "verdict": vd})
    return out


def calibrate():
    known = [("iteration_2/72_videos/PP-ActingTough-Base.mp4", 49, 71, "shake f58-61"),
             ("iteration_2/72_videos/PP-Germs-Base.mp4", 0, 33, "drift from f29"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 117, 178, "static"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 178, 250, "static"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 300, 371, "static"),
             ("iteration_2/72_videos/PP-Chair-Base.mp4", 136, 223, "static"),
             ("iteration_2/72_videos/PP-Chair-Base.mp4", 0, 86, "slow push-in"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 0, 117, "pan"),
             ("iteration_2/72_videos/PP-MyOffer-Base.mp4", 0, 90, "slow push-in"),
             ("iteration_2/72_videos/PP-Bored-Base_audiov5.mp4", 131, 192, "4% zoom"),
             ("iteration_2/72_videos/PP-4amshow-Base.mp4", 0, 60, "push-in (4amshow)")]
    cache = {}
    for skit, s, e, lab in known:
        if skit not in cache:
            cache[skit] = decode_rgb(T.skit_path(skit))
        f = span_fits(cache[skit], s, min(e, len(cache[skit])))
        se = series(cache[skit], s, min(e, len(cache[skit])))
        print(f"{lab:<18} {Path(skit).stem:<22} [{s},{e})  first->last {f[0]}  => {verdict(f)}  "
              f"series: checked {se['checked']} bad {se['bad_frames']} n {se['n_bad']} worst {se['worst']}")


def main():
    if "--calibrate" in sys.argv:
        calibrate(); return
    files = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(str(p) for p in (HERE / "curation").glob("*.json"))
    with ProcessPoolExecutor(12) as ex:
        rows = [r for rs in ex.map(check_file, files) for r in rs]
    (T.WORK / "span_motion_check.json").write_text(json.dumps(rows, indent=1))
    from collections import Counter
    print("verdicts:", dict(Counter(r["verdict"] for r in rows)), "of", len(rows), "kept spans")
    for r in rows:
        if r["verdict"] != "locked":
            print(f"  {r['verdict']:<13} {r['file']:<48} [{r['start']},{r['end']})  {r['framing']:<16} "
                  f"first->last {r['first_last']}  series bad {r['series']['bad_frames']} n {r['series']['n_bad']}/"
                  f"{r['series']['checked']} worst {r['series']['worst']}")


if __name__ == "__main__":
    main()
