#!/usr/bin/env python3
"""QA one generated take: contact sheet + automatic flags.

  python qa_take.py <take.mp4> [<keyframe.png>]

Flags (heuristics; the contact sheet is the verdict):
  drift     mean |frame - keyframe| over the static background border (camera/zoom creep)
  frozen    fraction of consecutive frame pairs with ~no change (stalled animation)
  jump      largest single-frame change vs the median (pops / morph glitches)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np


def frames(p, w=180, h=320):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def main(take, key=None):
    v = frames(take)
    d = np.abs(np.diff(v, axis=0)).mean((1, 2, 3))
    border = np.zeros(v.shape[1:3], bool); border[:24] = border[-24:] = True; border[:, :14] = border[:, -14:] = True
    ref = frames(key)[0] if key else v[0]
    drift = float(np.abs(v[-4:].mean(0) - ref)[border].mean())
    frozen = float((d < 0.15).mean())
    jump = float(d.max() / max(np.median(d), 1e-6))
    sheet = Path(take).with_suffix(".sheet.png")
    n = len(v); step = max(1, n // 9)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(take), "-vf",
                    f"select='not(mod(n\\,{step}))',scale=240:427,tile=9x1", "-frames:v", "1", str(sheet)])
    flags = [f for f, bad in (("DRIFT", drift > 6), ("FROZEN", frozen > 0.35), ("POP", jump > 8)) if bad]
    print(f"{Path(take).name}: frames {n} | border drift {drift:.2f} | frozen {frozen:.0%} | max-jump x{jump:.1f} "
          f"| {'FLAGS: ' + ','.join(flags) if flags else 'ok'} | sheet {sheet}")


if __name__ == "__main__":
    main(*sys.argv[1:])
