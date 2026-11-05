#!/usr/bin/env python3
"""Tools for the v9b curator pass (Training_Approach_v9.md section 0.1, item 2).

The scene-cut detector misses cuts, dissolves, fades and camera moves (SKIT_AUDIT_v9b.md),
so a curator places every boundary by eye. These tools give the curator the signals and
the pictures to do that at frame accuracy. They only advise: the curator decides.

    PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY curate_tools.py profile SKIT                # signals + candidate boundaries -> text + PNG plot
    $PY curate_tools.py strip SKIT A B [--step 2]   # frames A..B-1, numbered, as a grid PNG
    $PY curate_tools.py compare SKIT F1 F2          # two frames side by side + edge overlay
    $PY curate_tools.py motion SKIT A B [--step 4 --band 0.05 0.35]   # camera scale/shift vs frame A

SKIT is a path relative to the Data root (e.g. iteration_2/72_videos/PP-Chair-Base.mp4) or
absolute. Frame numbers are 0-based decode order with passthrough timing (every real frame,
no duplicates inserted), the same numbering the curation JSON and prep_v9b.py use.
Outputs go to $V9B_WORK/<skit stem>/ (default /workspace/Data/processed/v9b_work).

Signals in `profile`:
  d1      mean |frame k - frame k-1| (grey, 270x480): a hard cut is an isolated spike.
  d8      mean |frame k+8 - frame k-8|: high and smooth across a window = a gradual change
          (dissolve, fade, camera move, or big motion). Look at the strip.
  luma    mean brightness: a ramp to/from very dark or very bright = a fade.
  keep12  share of frame k's edges still present (1 px tolerance) in frame k+12. A locked
          camera keeps the background's edges: static shots measure median >= 0.8. A pan or
          zoom moves every edge (Economic's pan: median 0.5). Large character motion in a
          close-up also lowers it, so a low value means LOOK, not "camera move". Confirm with
          `compare` (static background edges overlap exactly).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = Path(os.environ.get("PUDGY_DATA", "/workspace/Data"))
RAW = DATA / "raw"
WORK = Path(os.environ.get("V9B_WORK", str(DATA / "processed/v9b_work")))
# Off the GPU box (PUDGY_DATA set, e.g. the laptop's Data tree), the default would write work
# images into Data/processed, which holds only training sets. curator-v10-20 hit this once.
if "PUDGY_DATA" in os.environ and "V9B_WORK" not in os.environ:
    sys.exit("set V9B_WORK to a scratch directory: with PUDGY_DATA set, the default would write into Data/processed")
PW, PH = 270, 480          # analysis size (1080x1920 / 4)


def skit_path(s):
    p = Path(s)
    return p if p.is_absolute() else RAW / s


def work_dir(path):
    d = WORK / Path(path).stem.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _passthrough():
    h = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in h else ["-vsync", "passthrough"]


def decode(path, w, h, fmt="gray"):
    ch = {"gray": 1, "rgb24": 3}[fmt]
    # -threads 4: ~128 concurrent default-threaded ffmpeg 6.1 decodes deadlocked on this box
    # (v9c prep); 12 curators share it, so each decode stays small.
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area",
                        *_passthrough(), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                       capture_output=True, check=True).stdout
    a = np.frombuffer(b, np.uint8)
    return a.reshape(-1, h, w) if ch == 1 else a.reshape(-1, h, w, ch)


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",     # macOS (v10 curation runs locally)
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)                          # Pillow >= 10.1: a scalable default
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------------------
def edges(g):
    return [cv2.Canny(x, 60, 140) > 0 for x in g]


def keep_series(E, L=12):
    """keep[k] = |E_k & dilate(E_{k+L})| / |E_k|; NaN where the frame has no edges."""
    ker = np.ones((3, 3), np.uint8)
    out = np.full(len(E), np.nan)
    for k in range(len(E) - L):
        n = E[k].sum()
        if n:
            out[k] = (E[k] & (cv2.dilate(E[k + L].astype(np.uint8), ker) > 0)).sum() / n
    return out


def profile(path):
    g = decode(path, PW, PH)
    n = len(g)
    f = g.astype(np.float32)
    d1 = np.zeros(n); d1[1:] = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    d8 = np.zeros(n)
    for k in range(8, n - 8):
        d8[k] = np.abs(f[k + 8] - f[k - 8]).mean()
    luma = f.mean(axis=(1, 2))
    E = edges(g)
    keep = keep_series(E)

    # Hard-cut candidates: a spike well above its neighbourhood.
    cuts = []
    for k in range(1, n):
        nb = np.concatenate([d1[max(1, k - 6):k], d1[k + 1:k + 7]])
        base = np.median(nb) if len(nb) else 0.0
        if d1[k] > max(8.0, 4.0 * base + 3.0):
            cuts.append({"frame": k, "d1": round(float(d1[k]), 1), "local_median": round(float(base), 2)})

    # Gradual-change candidates: d8 local maxima that are not just a hard cut.
    grads = []
    cut_frames = {c["frame"] for c in cuts}
    for k in range(9, n - 9):
        if d8[k] < 6.0 or d8[k] < d8[k - 1] or d8[k] < d8[k + 1]:
            continue
        if any(abs(k - c) <= 8 for c in cut_frames):
            continue
        win = d1[k - 8:k + 9]
        grads.append({"frame": k, "d8": round(float(d8[k]), 1), "max_d1_in_window": round(float(win.max()), 1),
                      "luma": round(float(luma[k]), 1)})

    # Fades: luma near black/white at the ends, or ramps.
    fades = []
    for k in range(n):
        if luma[k] < 25 or luma[k] > 245:
            fades.append(k)

    # Camera per detected shot (between hard-cut candidates): edge persistence.
    bounds = [0] + sorted(cut_frames) + [n]
    shots = []
    ker = np.ones((3, 3), np.uint8)
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 2:
            continue
        kk = keep[s:max(s + 1, e - 12)]
        kk = kk[~np.isnan(kk)]
        med = float(np.median(kk)) if len(kk) else float("nan")
        p10 = float(np.percentile(kk, 10)) if len(kk) else float("nan")
        n0 = E[s].sum()
        fl = float((E[s] & (cv2.dilate(E[e - 1].astype(np.uint8), ker) > 0)).sum() / n0) if n0 else float("nan")
        flag = not len(kk) or med < 0.75 or p10 < 0.65
        # Stretches whose 12-frame edge persistence stays >= 0.8: candidate locked spans.
        runs, r0 = [], None
        for k in range(s, e + 1):
            ok = k < e - 12 and not np.isnan(keep[k]) and keep[k] >= 0.8
            if ok and r0 is None:
                r0 = k
            elif not ok and r0 is not None:
                if k + 12 - r0 >= 21:
                    runs.append([r0, min(e, k + 12)])
                r0 = None
        shots.append({"start": s, "end": e, "frames": e - s, "keep12_median": round(med, 2),
                      "keep12_p10": round(p10, 2), "first_vs_last": round(fl, 2),
                      "steady_runs": runs,
                      "camera_hint": "LOOK: camera move or big motion" if flag else "static?"})

    wd = work_dir(path)
    res = {"skit": str(path), "frames": n, "hard_cut_candidates": cuts,
           "gradual_candidates": grads, "dark_or_bright_frames": fades, "shots_by_candidates": shots}
    (wd / "profile.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(wd / "profile_series.npz", d1=d1, d8=d8, luma=luma, keep12=keep)
    plot(wd / "profile.png", d1, d8, luma, keep, cuts, path)
    return res, wd


def plot(dest, d1, d8, luma, keep, cuts, path):
    n = len(d1)
    W, rowh = max(900, min(2400, 3 * n)), 140
    img = Image.new("RGB", (W + 60, rowh * 4 + 30), "white")
    dr = ImageDraw.Draw(img)
    ft = font(12)
    dr.text((4, 2), f"{Path(path).name}  {n} frames  (x = frame)", fill="black", font=ft)
    series = [("d1", d1, (200, 0, 0)), ("d8", d8, (0, 0, 200)), ("luma", luma, (0, 120, 0)),
              ("keep12", np.nan_to_num(keep, nan=1.0), (120, 0, 160))]
    for r, (nm, y, col) in enumerate(series):
        top = 20 + r * rowh
        lo, hi = float(np.min(y)), float(np.max(y)) if np.max(y) > np.min(y) else float(np.min(y)) + 1
        dr.text((4, top), f"{nm} [{lo:.1f}, {hi:.1f}]", fill=col, font=ft)
        pts = [(50 + int(i * W / max(1, n - 1)), top + rowh - 10 - int((float(v) - lo) / (hi - lo) * (rowh - 30)))
               for i, v in enumerate(y)]
        dr.line(pts, fill=col, width=1)
        for i in range(0, n, 24):
            x = 50 + int(i * W / max(1, n - 1))
            dr.line([(x, top + rowh - 8), (x, top + rowh - 4)], fill="grey")
            if r == 3:
                dr.text((x - 6, top + rowh - 2), str(i), fill="grey", font=ft)
    for c in cuts:
        x = 50 + int(c["frame"] * W / max(1, n - 1))
        dr.line([(x, 20), (x, 20 + 4 * rowh)], fill=(255, 160, 0), width=1)
    img.save(dest)


# ---------------------------------------------------------------------------------------
# strip / compare
# ---------------------------------------------------------------------------------------
def strip(path, a, b, step, tw, cols, dest=None):
    th = int(round(tw * 16 / 9))
    v = decode(path, tw, th, "rgb24")
    b = min(b, len(v))
    idx = list(range(a, b, step))
    if idx and idx[-1] != b - 1:
        idx.append(b - 1)                     # always show the last frame of the range
    rows = (len(idx) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, rows * (th + 16)), "white")
    dr = ImageDraw.Draw(img)
    ft = font(13)
    for i, k in enumerate(idx):
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        img.paste(Image.fromarray(v[k]), (x, y + 16))
        dr.text((x + 3, y + 1), f"f{k}", fill="black", font=ft)
    dest = dest or work_dir(path) / f"strip_{a}_{b}_s{step}.jpg"
    img.save(dest, quality=88)
    return dest, len(v)


def compare(path, f1, f2, dest=None):
    tw, th = 360, 640
    v = decode(path, tw, th, "rgb24")
    A, B = v[f1], v[f2]
    ea = cv2.Canny(cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    eb = cv2.Canny(cv2.cvtColor(B, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    ov = np.full((th, tw, 3), 255, np.uint8)
    ov[ea & ~eb] = (220, 0, 0)       # only in F1: red
    ov[eb & ~ea] = (0, 170, 0)       # only in F2: green
    ov[ea & eb] = (0, 0, 0)          # in both: black (static)
    diff = np.clip(np.abs(A.astype(int) - B.astype(int)).sum(axis=2) * 2, 0, 255).astype(np.uint8)
    diff = np.stack([255 - diff] * 3, axis=2)
    img = Image.new("RGB", (tw * 4, th + 20), "white")
    dr = ImageDraw.Draw(img)
    ft = font(13)
    for i, (im, lab) in enumerate([(A, f"f{f1}"), (B, f"f{f2}"),
                                   (ov, "edges: black=both red=F1 green=F2"), (diff, "|F1-F2| (dark=changed)")]):
        img.paste(Image.fromarray(im), (i * tw, 20))
        dr.text((i * tw + 3, 2), lab, fill="black", font=ft)
    both = int((ea & eb).sum()); only = int((ea ^ eb).sum())
    dest = dest or work_dir(path) / f"compare_{f1}_{f2}.jpg"
    img.save(dest, quality=88)
    return dest, both, only


def motion(path, a, b, step, y0, y1, x0=0.0, x1=1.0):
    """Camera motion over [a, b): an affine ECC fit of the background band [y0, y1) x [x0, x1)
    (fractions of the picture) between frame a and every `step`-th frame after it, so slow
    drift accumulates instead of hiding under a per-step tolerance (curator-v10-01's method:
    it settled every FastText push-in that keep12 called LOOK). A locked camera stays at scale
    1.000 and ~0 px. Pick a band with background texture and no character in it."""
    F = decode(path, PW, PH)
    b = min(b, len(F))
    mask = np.zeros((PH, PW), np.uint8)
    mask[int(y0 * PH):int(y1 * PH), int(x0 * PW):int(x1 * PW)] = 255
    ref = F[a].astype(np.float32)
    # Texture in the band, in both directions. On a plain or one-directional band ECC can
    # converge to the identity (scale 1.0000, cc 1.000) without matching anything
    # (curator-v10-08, ActsOfService; curator-v10-17, sofa bases), so a "locked" reading
    # there proves nothing.
    band = ref[mask > 0].reshape(-1, int(x1 * PW) - int(x0 * PW))
    gx = np.abs(np.diff(band, axis=1)) > 8
    gy = np.abs(np.diff(band, axis=0)) > 8
    texture = {"x": round(float(gx.mean()), 4), "y": round(float(gy.mean()), 4)}
    out = [{"texture": texture}]
    for k in range(a + step, b, step):
        M = np.eye(2, 3, dtype=np.float32)
        try:
            cc, M = cv2.findTransformECC(ref, F[k].astype(np.float32), M, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
        except cv2.error:
            out.append({"frame": k, "fit": False}); continue
        s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        out.append({"frame": k, "fit": True, "scale": round(s, 4), "tx": round(float(M[0, 2]) * 1080 / PW, 1),
                    "ty": round(float(M[1, 2]) * 1080 / PW, 1), "cc": round(float(cc), 3)})
    return out, len(F)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("profile"); p.add_argument("skit")
    s = sp.add_parser("strip"); s.add_argument("skit"); s.add_argument("start", type=int); s.add_argument("end", type=int)
    s.add_argument("--step", type=int, default=2); s.add_argument("--width", type=int, default=180)
    s.add_argument("--cols", type=int, default=8); s.add_argument("--out", type=Path)
    c = sp.add_parser("compare"); c.add_argument("skit"); c.add_argument("f1", type=int); c.add_argument("f2", type=int)
    c.add_argument("--out", type=Path)
    m = sp.add_parser("motion"); m.add_argument("skit"); m.add_argument("start", type=int); m.add_argument("end", type=int)
    m.add_argument("--step", type=int, default=4)
    m.add_argument("--band", type=float, nargs=2, default=(0.05, 0.35), metavar=("Y0", "Y1"),
                   help="background rows as fractions of the height (default the top 5-35%%)")
    m.add_argument("--cols", type=float, nargs=2, default=(0.0, 1.0), metavar=("X0", "X1"))
    a = ap.parse_args()
    path = skit_path(a.skit)
    if not path.exists():
        sys.exit(f"no such skit: {path}")
    if a.cmd == "profile":
        res, wd = profile(path)
        print(f"{path.name}: {res['frames']} frames  -> {wd}/profile.json, profile.png")
        print("hard-cut candidates (frame = first frame of the new shot):")
        for x in res["hard_cut_candidates"]:
            print(f"  f{x['frame']:<5} d1 {x['d1']:<6} local median {x['local_median']}")
        print("gradual-change candidates (dissolve / fade / camera move / big motion - look):")
        for x in res["gradual_candidates"]:
            print(f"  f{x['frame']:<5} d8 {x['d8']:<6} max d1 nearby {x['max_d1_in_window']:<6} luma {x['luma']}")
        df = res["dark_or_bright_frames"]
        if df:
            print(f"very dark/bright frames (fade?): {df[:12]}{' ...' if len(df) > 12 else ''} ({len(df)})")
        print("shots between hard-cut candidates, camera hints (confirm with compare):")
        for x in res["shots_by_candidates"]:
            print(f"  [{x['start']},{x['end']}) {x['frames']:>4}f  keep12 median {x['keep12_median']:.2f} "
                  f"p10 {x['keep12_p10']:.2f}  first-vs-last {x['first_vs_last']:.2f}  -> {x['camera_hint']}"
                  + (f"   steady runs {x['steady_runs']}" if x['camera_hint'].startswith("LOOK") else ""))
    elif a.cmd == "strip":
        dest, n = strip(path, a.start, a.end, a.step, a.width, a.cols, a.out)
        print(f"{dest}  (skit has {n} frames)")
    elif a.cmd == "motion":
        res, n = motion(path, a.start, a.end, a.step, *a.band, *a.cols)
        print(f"{path.name}: camera vs frame {a.start}, band y {a.band[0]:.2f}-{a.band[1]:.2f} "
              f"x {a.cols[0]:.2f}-{a.cols[1]:.2f}  (locked = scale 1.000, tx/ty ~0 px at 1080 wide; cc = fit quality)\n"
              f"  trust a row only at cc >= 0.95. Low cc = a character is in the band, pick another band; "
              f"'no fit' = a plain ground with no texture: judge by eye (character size, shadow, compare)")
        tx_ = res[0]["texture"]
        weak = min(tx_["x"], tx_["y"]) < 0.01
        print(f"  band texture: edges across {tx_['x']:.2%}, edges down {tx_['y']:.2%}"
              + ("   ** TOO LITTLE TEXTURE (one or both directions): a 1.0000 / cc 1.000 reading here proves "
                 "NOTHING. Pick a textured band or judge with compare **" if weak else ""))
        for r in res[1:]:
            print(f"  f{r['frame']:<5} " + (f"scale {r['scale']:.4f}  tx {r['tx']:+7.1f}  ty {r['ty']:+7.1f}  cc {r['cc']:.3f}"
                                           if r["fit"] else "no fit (too little texture in the band?)"))
    else:
        dest, both, only = compare(path, a.f1, a.f2, a.out)
        print(f"{dest}  static edge px {both}, changed edge px {only}  "
              f"(a locked camera keeps background edges black/overlapping)")


if __name__ == "__main__":
    main()
