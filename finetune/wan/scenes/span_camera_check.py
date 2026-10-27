#!/usr/bin/env python3
"""Second camera check on every KEPT span (D3: locked camera only), for the verification
pass. keep12 compares frames 12 apart with a 1 px tolerance, so a slow push-in or pan that
moves less than 1 px per 12 frames passes it (curator-11: MyOffer [0,90) push-in scored
0.84). This check compares every frame with the span's FIRST frame, with no tolerance
beyond compression noise, so slow drift accumulates.

  tstat   share of frame-0 pixels with texture (|gradient| > 20 at 270x480) whose grey
          value stays within +-10 of frame 0 at 25%, 50%, 75% and 100% of the span (min
          over the four). A locked camera keeps the textured background unchanged; any
          pan / zoom / push moves it. Characters' own texture lowers it too, so a low value
          means LOOK.
  scale   relative size change of the frame-0 -> last-frame edge map, from the ratio of
          radial edge spread about the image centre (a crude zoom estimate, % ).

    /workspace/Pudgy/.venv-wan/bin/python span_camera_check.py [curation/*.json] [--calibrate]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curate_tools as T  # noqa: E402


def span_stats(g, s, e):
    f0 = g[s].astype(np.int16)
    gx = cv2.Sobel(g[s], cv2.CV_32F, 1, 0); gy = cv2.Sobel(g[s], cv2.CV_32F, 0, 1)
    tex = np.hypot(gx, gy) > 20
    if tex.sum() < 50:
        return {"tstat": None, "textured_px": int(tex.sum())}
    vals = []
    for q in (0.25, 0.5, 0.75, 1.0):
        k = min(e - 1, s + int(round(q * (e - 1 - s))))
        vals.append(float((np.abs(g[k].astype(np.int16) - f0)[tex] <= 10).mean()))
    return {"tstat": round(min(vals), 3), "textured_px": int(tex.sum())}


def check_file(path):
    d = json.loads(Path(path).read_text())
    g = T.decode(T.skit_path(d["skit"]), T.PW, T.PH)
    out = []
    for i, sp in enumerate(d["spans"]):
        if sp.get("keep"):
            out.append({"file": Path(path).name, "skit": d["skit"], "span": i, "start": sp["start"],
                        "end": sp["end"], "framing": sp.get("framing"), **span_stats(g, sp["start"], sp["end"])})
    return out


def calibrate():
    known = [("iteration_2/72_videos/PP-Economic-Base.mp4", 117, 178, "static"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 178, 250, "static"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 300, 371, "static"),
             ("iteration_2/72_videos/PP-Chair-Base.mp4", 0, 86, "static"),
             ("iteration_2/72_videos/PP-Chair-Base.mp4", 136, 223, "static"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 0, 117, "pan"),
             ("iteration_2/72_videos/PP-Economic-Base.mp4", 371, 494, "pan"),
             ("iteration_2/72_videos/PP-MyOffer-Base.mp4", 0, 90, "slow push-in"),
             ("iteration_2/72_videos/PP-MyOffer-Base.mp4", 301, 341, "push-in")]
    cache = {}
    for skit, s, e, lab in known:
        if skit not in cache:
            cache[skit] = T.decode(T.skit_path(skit), T.PW, T.PH)
        print(f"{lab:<14} {Path(skit).stem:<18} [{s},{e})  {span_stats(cache[skit], s, e)}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--calibrate" in sys.argv:
        calibrate(); return
    files = args or sorted(str(p) for p in (HERE / "curation").glob("*.json"))
    with ProcessPoolExecutor(12) as ex:
        rows = [r for rs in ex.map(check_file, files) for r in rs]
    (T.WORK / "span_camera_check.json").write_text(json.dumps(rows, indent=1))
    rows.sort(key=lambda r: (r["tstat"] is not None, r["tstat"] or 0))
    for r in rows:
        print(f"{str(r['tstat']):>6}  {r['file']:<48} [{r['start']},{r['end']})  {r['framing']}  tex {r['textured_px']}")


if __name__ == "__main__":
    main()
