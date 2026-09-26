#!/usr/bin/env python3
"""Build QF (three-quarter) eval keyframes — the angle family prep_eval_keyframes_v7.py
never made.

That script hardcodes `FRONT` for its training-frame family, so every gate that has run
(G-X, G-C, G-P, G-D) drove I2V from a front-view start frame. Six of the nine trained
angles are three-quarter views and account for 1,508 of the 2,272 clips, yet none has
ever been used as a conditioning frame at inference.

Frame 0 of a real v7 clip IS the training start frame by construction (same reasoning as
the front-view script), so these are TRAINING frames in the G-F sense — not novel ones.
An angle probe driven from them tests whether the angle clause stays promptable off-front,
not whether the model generalises to an unseen pose.

f21 is the COMMON expression bucket: every label appears there, so a frame taken from it
carries no length->label association.

Filenames mirror the front-view convention:
    qf/<char>_neutral_<ANGLE>_start.png      e.g. qf/pax_neutral_QF1_L_start.png
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CLIPS = Path("/workspace/data_v7/clips")
OUT = Path("/workspace/eval_v7/keyframes/qf")

# Angle slug -> the caption clause prep_v7.py builds for it (QUARTER, verbatim).
QUARTER = {
    1: "turned slightly to its {side}, three-quarter front view",
    2: "turned further to its {side}, wide three-quarter view",
    3: "turned strongly to its {side}, near-profile three-quarter view",
}
ANGLES = [f"QF{d}_{s}" for d in (1, 2, 3) for s in ("L", "R")]


def clause_for(slug):
    deg = int(slug[2])
    side = "left" if slug[-1] == "L" else "right"
    return QUARTER[deg].format(side=side)


def frame0(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", "select=eq(n\\,0)", "-vframes", "1", str(dest)],
        check=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bg", default="white", help="ground for the training frames")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    for char in ("pax", "polly"):
        for slug in ANGLES:
            src = CLIPS / f"{char}_neutral_{slug}__{args.bg}_f21.mp4"
            if not src.exists():
                sys.exit(f"missing training clip {src}")
            name = f"{char}_neutral_{slug}_start.png"
            frame0(src, out / name)
            meta[name] = {
                "background": args.bg,
                "source": src.name,
                "angle": slug,
                "clause": clause_for(slug),
                "family": "training",
            }

    (out / "keyframes_qf.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(meta)} QF keyframes -> {out}")
    for k, v in meta.items():
        print(f"  {k:<38} {v['clause']}")


if __name__ == "__main__":
    main()
