#!/usr/bin/env python3
"""Flag frames where Pax or Polly appears more than once.

  python dupcheck.py <take.mp4> [...]

Per frame: mask Pax-blue and Polly-pink body colour (HSV), close across the black outline
strokes so one character's head/body/flippers merge into one blob, then count blobs whose
area is at least MIN_FRAC of the largest same-colour blob AND at least MIN_ABS of the frame.
A lone flipper reaching in (LifeWithHer: Polly off-frame) is small and is not counted as a body.
Any frame with >= 2 bodies of one colour is flagged, and a strip of those frames is written.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 270, 480
MIN_FRAC, MIN_ABS = 0.35, 0.012


def frames(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def bodies(f, lo, hi):
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    areas = sorted(st[1:, cv2.CC_STAT_AREA], reverse=True)
    if not areas:
        return 0
    return sum(a >= max(MIN_FRAC * areas[0], MIN_ABS * W * H) for a in areas)


PAX = ((100, 90, 120), (125, 255, 255))      # saturated mid blue
POLLY = ((150, 40, 150), (178, 255, 255))    # pink/magenta


def main(paths):
    worst = 0
    for p in paths:
        v = frames(p)
        pax = [bodies(f, *PAX) for f in v]; pol = [bodies(f, *POLLY) for f in v]
        bad = [i for i in range(len(v)) if pax[i] >= 2 or pol[i] >= 2]
        worst = max(worst, len(bad))
        msg = f"{Path(p).name:32s} max Pax bodies {max(pax)}  max Polly bodies {max(pol)}  flagged frames {len(bad)}"
        if bad:
            strip = Path(p).with_suffix(".dup.png")
            sel = "+".join(f"eq(n\\,{i})" for i in bad[:8])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-vf",
                            f"select='{sel}',scale=240:427,tile={min(8, len(bad))}x1", "-frames:v", "1", str(strip)])
            msg += f" e.g. {bad[:8]} -> {strip}"
        print(msg)
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
