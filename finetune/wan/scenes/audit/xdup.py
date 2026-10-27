"""Cross-skit duplicate shots: dHash (16x9 on 9:16) of frames at 25/50/75% of each shot ≥ 1 s;
a pair of shots from different skits matches if all three hashes are within HAM bits.
Also exact-duplicate files by MD5. Usage: xdup.py review.json RAW"""
import json, subprocess, sys, collections, hashlib, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path
review = json.load(open(sys.argv[1])); raw = Path(sys.argv[2])
by_file = collections.defaultdict(list)
for s in review: by_file[s["file"]].append(s)
def dh(img):  # img [H, W+1]
    return (img[:, 1:] > img[:, :-1]).flatten()
def run(item):
    f, shots = item
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw / f), "-vf", "scale=17:30:flags=area", "-vsync", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    a = np.frombuffer(b, np.uint8).reshape(-1, 30, 17).astype(np.int16)
    md5 = hashlib.md5((raw / f).read_bytes()).hexdigest()
    out = []
    for s in shots:
        n = s["end"] - s["start"]
        hs = [dh(a[s["start"] + int(n * q)]) for q in (0.25, 0.5, 0.75)]
        out.append((f, s["shot"], s["verdict"], n, np.stack(hs)))
    return md5, f, out
with ProcessPoolExecutor(32) as ex:
    res = list(ex.map(run, sorted(by_file.items())))
md5s = collections.defaultdict(list)
for m, f, _ in res: md5s[m].append(f)
print("identical files:", [v for v in md5s.values() if len(v) > 1])
shots = [o for _, _, out in res for o in out]
H = np.stack([s[4] for s in shots])  # N,3,256
HAM = int(sys.argv[3]) if len(sys.argv) > 3 else 20
pairs = []
for i, j in itertools.combinations(range(len(shots)), 2):
    if shots[i][0] == shots[j][0]: continue
    d = (H[i] != H[j]).sum(axis=1)
    if d.max() <= HAM: pairs.append((int(d.max()), shots[i][:4], shots[j][:4]))
pairs.sort()
print(len(pairs), "cross-skit shot pairs within", HAM, "bits (of 480)")
for p in pairs[:60]: print(p)
