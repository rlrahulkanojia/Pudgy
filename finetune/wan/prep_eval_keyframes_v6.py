#!/usr/bin/env python3
"""
Regenerate the eval keyframes the v6 gates need.

The v5 box was destroyed and `/workspace/eval_v5/keyframes/` went with it, so every
gate that drives I2V from a start image had no input. This rebuilds them from sources
that DO survive: the raw ProRes-4444 alpha clips (Azure `raw/iteration_3/`) and the
v1/v2 75-clip set (Azure `processed/v1v2_75clip/`).

Two families, and the distinction is the whole point of gates G-F vs G-M:

  training frames  — composited exactly like the training clips (same grounds, same
                     zoom ladder). Driving from these reproduces the condition that
                     triggered v5's memorisation, so G-F can test for it.
  novel frames     — a v1 skit frame: different pose, scene, framing and palette,
                     nothing in the 272 clips resembles it. G-M needs this to show
                     motion survives away from the training distribution.

Plus unseen backgrounds for G-B, which must be grounds the model never saw. Trained
grounds are white / pastel blue / peach / mint, so we add lavender and sky — sky is
deliberately pulled away from the trained pastel blue (198,222,241) rather than sitting
next to it, or "unseen" would be a technicality rather than a real test.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_expressions_v6 import (  # noqa: E402  — shared so eval matches training exactly
    BACKGROUNDS, SIZE, composite, read_rgba_zoomed, discover, ZOOMS,
)

# Grounds the model has never seen. G-B checks the expression survives on these.
UNSEEN = {
    "lavender": ((222, 212, 240), "plain pastel lavender background"),
    "sky":      ((170, 205, 235), "plain pastel sky blue background"),
}

DEFAULT_RAW = Path("/workspace/data_raw/iteration_3/03_expression_clips")
DEFAULT_V1 = Path("/workspace/data_v1v2/train")
DEFAULT_OUT = Path("/workspace/eval_v6/keyframes")


def save_png(arr, dest):
    """arr: (H,W,3) uint8 -> png via ffmpeg (no extra image dep)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{arr.shape[1]}x{arr.shape[0]}", "-i", "-", str(dest)],
        stdin=subprocess.PIPE)
    p.communicate(arr.tobytes())
    if p.returncode:
        sys.exit(f"ffmpeg failed writing {dest}")


def pick(jobs, char, emotion, angle="FRONT"):
    for j in jobs:
        if j["char"] == char and j["emotion"] == emotion and j["angle"].startswith(angle):
            return j
    sys.exit(f"no source clip for {char}/{emotion}/{angle}")


def v1_novel_frame(v1_dir, clip, frame, out):
    """A v1 skit frame, centre-cropped to square and resized to the eval geometry.

    v1 is 768x1360 portrait and the gates run at 1024x1024, so it must be squared.
    Centre crop keeps the character; letterboxing would introduce black bars that the
    model reads as scene content.
    """
    src = v1_dir / clip
    if not src.exists():
        sys.exit(f"missing v1 clip {src}")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vf",
         f"select=eq(n\\,{frame}),crop='min(iw,ih)':'min(iw,ih)',scale={SIZE}:{SIZE}",
         "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
    save_png(arr, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--v1", type=Path, default=DEFAULT_V1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    # One novel frame per character, so G-M and G-P can both run away from the training
    # distribution. Both were picked by inspection for the property the gate depends on:
    # the character is FULLY inside the square crop with margin on every side. v5 section
    # 4.4 used clip 00000001, but its Pax sits hard against the right edge and the square
    # crop cuts him — a subject clipped by the frame corrupts the x-range measurement that
    # IS the G-M pass/fail criterion, so it is the wrong source here.
    ap.add_argument("--v1-clip-pax", default="00000010.mp4",
                    help="novel Pax frame (default: living room, broom — fully in frame)")
    ap.add_argument("--v1-clip-polly", default="00000016.mp4",
                    help="novel Polly frame (default: at a table — fully in frame)")
    ap.add_argument("--v1-frame", type=int, default=0)
    args = ap.parse_args()

    jobs, skipped, _ = discover(args.raw)
    if not jobs:
        sys.exit(f"no source clips under {args.raw}")
    print(f"discovered {len(jobs)} source clips")

    out = args.out
    made = []

    # --- 1. Training-distribution start/end frames, per character, on white ---------
    # G-C drives every emotion from ONE start frame per character, so the prompt is the
    # only variable. Neutral is the right source for that frame: it is the least
    # expressive delivery, so it does not pre-load any emotion into the conditioning.
    for char in ("Pax", "Polly"):
        j = pick(jobs, char, "neutral")
        frames = read_rgba_zoomed(j["src"], 2, 1.00)
        rgb = BACKGROUNDS["white"][1]
        save_png(composite(frames, rgb)[0], out / f"{char.lower()}_neutral_start.png")
        made.append(f"{char.lower()}_neutral_start.png")

    # --- 2. In-distribution start/end for the sanity check --------------------------
    for char in ("Pax", "Polly"):
        j = pick(jobs, char, "happy")
        n = 21
        frames = read_rgba_zoomed(j["src"], n, 1.00)
        rgb = BACKGROUNDS["white"][1]
        comp = composite(frames, rgb)
        save_png(comp[0], out / f"{char.lower()}_happy_indist_start.png")
        save_png(comp[-1], out / f"{char.lower()}_happy_indist_end.png")
        made += [f"{char.lower()}_happy_indist_start.png",
                 f"{char.lower()}_happy_indist_end.png"]

    # --- 3. G-F: the SAME training frame at each shot size ---------------------------
    # v5's trigger was pixel-level: near-identical start frames across the whole set.
    # The ladder is one of the two mitigations, so G-F must probe all three scales.
    for char in ("Pax", "Polly"):
        j = pick(jobs, char, "neutral")
        for zoom, label in ZOOMS.items():
            tag = label.replace("static ", "").replace(" shot", "").replace("-", "")
            frames = read_rgba_zoomed(j["src"], 2, zoom)
            save_png(composite(frames, BACKGROUNDS["white"][1])[0],
                     out / f"{char.lower()}_neutral_{tag}_start.png")
            made.append(f"{char.lower()}_neutral_{tag}_start.png")

    # --- 3b. Trained grounds, for showcase variants -----------------------------------
    # white already covered above; blue/peach/mint are the other three the model saw, and
    # a showcase that varies the ground needs a start frame on each.
    for char in ("Pax", "Polly"):
        j = pick(jobs, char, "neutral")
        frames = read_rgba_zoomed(j["src"], 2, 1.00)
        for name in ("blue", "peach", "mint"):
            save_png(composite(frames, BACKGROUNDS[name][1])[0],
                     out / f"{char.lower()}_neutral_{name}_start.png")
            made.append(f"{char.lower()}_neutral_{name}_start.png")

    # --- 4. G-B: unseen grounds ------------------------------------------------------
    for char in ("Pax", "Polly"):
        j = pick(jobs, char, "neutral")
        frames = read_rgba_zoomed(j["src"], 2, 1.00)
        for name, (rgb, _desc) in UNSEEN.items():
            save_png(composite(frames, rgb)[0], out / f"{char.lower()}_neutral_{name}_start.png")
            made.append(f"{char.lower()}_neutral_{name}_start.png")

    # --- 5. G-M / G-P: a novel frame per character -----------------------------------
    for char, clip in (("pax", args.v1_clip_pax), ("polly", args.v1_clip_polly)):
        v1_novel_frame(args.v1, clip, args.v1_frame, out / f"novel_v1_{char}_start.png")
        made.append(f"novel_v1_{char}_start.png")

    print(f"\nwrote {len(made)} keyframes -> {out}")
    for m in sorted(made):
        print(f"   {m}")
    print("\nunseen grounds (G-B):", ", ".join(f"{k}={v[0]}" for k, v in UNSEEN.items()))
    print("trained grounds      :", ", ".join(
        f"{k}={v[1]}" for k, v in BACKGROUNDS.items()))


if __name__ == "__main__":
    main()
