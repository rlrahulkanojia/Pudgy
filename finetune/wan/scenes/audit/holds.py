"""Held-frame census: per kept shot, fraction of frames identical to the previous one
(animation on twos/threes), and the run-length histogram of held drawings.
Usage: holds.py shot_review.json RAW_DIR OUT.json"""
import json, subprocess, sys, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path

W, H = 108, 192
review = json.load(open(sys.argv[1])); raw = Path(sys.argv[2])
by_file = collections.defaultdict(list)
for s in review:
    if s["verdict"] == "keep":
        by_file[s["file"]].append(s)

def frames(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H}:flags=area,format=gray",
                        "-vsync", "passthrough", "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return a.reshape(-1, H, W).astype(np.float32)

def run(item):
    f, shots = item
    v = frames(raw / f)
    d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))  # d[i] = |v[i+1]-v[i]|
    out = []
    for s in shots:
        seg = d[s["start"]:s["end"] - 1]
        held = seg < 0.25
        runs, n = [], 1
        for h in held:
            if h: n += 1
            else: runs.append(n); n = 1
        runs.append(n)
        out.append({"file": f, "shot": s["shot"], "frames": s["end"] - s["start"],
                    "held_frac": float(held.mean()) if len(seg) else 0.0,
                    "runs": dict(collections.Counter(min(r, 6) for r in runs))})
    return out

with ProcessPoolExecutor(32) as ex:
    rows = [r for rs in ex.map(run, sorted(by_file.items())) for r in rs]
json.dump(rows, open(sys.argv[3], "w"))
hf = np.array([r["held_frac"] for r in rows])
tot = collections.Counter()
for r in rows:
    for k, c in r["runs"].items(): tot[int(k)] += c * int(k)
print("shots", len(rows), "held-frame fraction: median %.2f, p10 %.2f, p90 %.2f" % tuple(np.percentile(hf, [50, 10, 90])))
s = sum(tot.values())
print("share of frames by drawing-hold length (1=on ones, 2=on twos, ...):", {k: round(v / s, 2) for k, v in sorted(tot.items())})
