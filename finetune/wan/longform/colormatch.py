#!/usr/bin/env python3
"""Per-frame colour grading back to a reference keyframe.

  python colormatch.py <in.mp4> <keyframe.png> <out.mp4>

First+last-frame generation can shift the whole palette mid-shot (seen on EatingStages beat 1:
the flat background jumps from grey-blue to saturated blue). The shift is global, so each frame
is corrected with a per-channel affine map (mean/std in Lab) onto the keyframe's statistics.
Flat pastel art makes this well-posed; content is untouched. Reports the max per-frame shift
before and after, measured on the background border.
"""
import subprocess
import sys

import cv2
import numpy as np

FLAT_TOL = 4.0   # Lab units; the grain is ~1, the nearest real content (shadow) is > 15


def read(p):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), w, h


def stats(lab):
    x = lab.reshape(-1, 3).astype(np.float64)
    return x.mean(0), x.std(0) + 1e-6


def main(inp, key, out):
    v, w, h = read(inp)
    k = cv2.cvtColor(cv2.resize(cv2.imread(key)[..., ::-1], (w, h)), cv2.COLOR_RGB2LAB)
    km, ks = stats(k)
    border = np.zeros((h, w), bool); border[:60] = border[-60:] = True; border[:, :40] = border[:, -40:] = True
    kb = k[border].astype(float).mean(0)
    kbg = np.median(k[border].reshape(-1, 3).astype(float), axis=0)   # the flat background colour
    res, before, after = [], [], []
    for f in v:
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float64)
        m, s = stats(lab)
        before.append(np.abs(lab[border].mean(0) - kb).max())
        # Offset only (mean), not spread: the FLF palette jump is an offset, and matching the std
        # as well stretched the generator's faint background grain ~3x (0.31 -> 0.97 on beat 1).
        # ...and measured on the BACKGROUND border, not the whole frame: in a close-up the character
        # dominates the frame mean, so a whole-frame offset lands off (beat 4: 3.5 Lab units).
        lab = lab - lab[border].mean(0) + kb
        # Flat-fill: the client's backgrounds are one exact colour (measured grain 0.00). Pixels within
        # a small Lab distance of the keyframe's background are snapped to it; outlines, characters
        # and shadows are far outside the tolerance and untouched.
        dist = np.linalg.norm(lab - kbg, axis=-1)
        lab[dist < FLAT_TOL] = kbg
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        after.append(np.abs(lab[border].astype(float).mean(0) - kb).max())
        res.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    p.communicate(np.stack(res).tobytes())
    print(f"{out}: background shift vs keyframe, max over frames: before {max(before):.1f}  after {max(after):.1f} (Lab units)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
