"""research_v9/motion.py's measure (128x128 grey: mean |dframe|, max drift from frame 0),
on the first min(81, len) frames of every kept skit shot. Usage: shot_motion.py review.json RAW OUT"""
import json, subprocess, sys, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path
review = json.load(open(sys.argv[1])); raw = Path(sys.argv[2])
by_file = collections.defaultdict(list)
for s in review:
    if s["verdict"] == "keep": by_file[s["file"]].append(s)
def run(item):
    f, shots = item
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw / f), "-vf", "scale=128:128", "-vsync", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    a = np.frombuffer(b, np.uint8).reshape(-1, 128, 128).astype(np.float32)
    out = []
    for s in shots:
        w = a[s["start"]:min(s["end"], s["start"] + 81)]
        out.append({"file": f, "shot": s["shot"], "frames": s["end"] - s["start"],
                    "dframe": float(np.abs(np.diff(w, axis=0)).mean()),
                    "drift": float(np.abs(w - w[0]).mean(axis=(1, 2)).max())})
    return out
with ProcessPoolExecutor(32) as ex:
    rows = [r for rs in ex.map(run, sorted(by_file.items())) for r in rs]
json.dump(rows, open(sys.argv[3], "w"))
d = np.array([r["drift"] for r in rows]); fr = np.array([r["frames"] for r in rows])
print("max drift from f0, kept shots: p10 %.2f  p25 %.2f  median %.2f  p75 %.2f  p90 %.2f" % tuple(np.percentile(d, [10, 25, 50, 75, 90])))
for t in (0.83, 2.0):
    m = d < t; print(f"drift < {t}: {m.sum()} shots, {fr[m].sum()/24/60:.1f} min")
