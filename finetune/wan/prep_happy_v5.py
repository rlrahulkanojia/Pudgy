#!/usr/bin/env python3
"""
v5 pilot dataset prep — Pax / happy expression clips (Azure `interation_3`).

The source clips are **ProRes 4444 with a real alpha channel** (yuva444p12le,
61.6% of frame fully transparent). Two consequences drive this script:

  1. A naive decode composites onto BLACK, not white — training on that would
     teach a black void, badly off-distribution vs the v2 identity/motion LoRA
     (trained on real environments).
  2. Alpha is an asset: we can composite the SAME performance onto several
     backgrounds. That is the only lever available against the core risk both v5
     docs flag — 7 clips are one performance from 7 correlated angles, so a LoRA
     can memorise "this smile on this background" instead of learning "happy".
     Compositing turns 7 clips into 7xN and forces background-invariance.

Output: 1024x1024, 21 frames (4*5+1, musubi-legal), silent mp4 + captions.
1024 not 1080: Wan's VAE is 8x and patchify is 2x2, so the latent side must be
even — 1080/8 = 135 (odd) would leave musubi to bucket it somewhere ambiguous.
1024/8 = 128. Clean.
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("/workspace/data_v5/happy_pax_raw")
OUT = Path("/workspace/data_v5/happy_pax_train")
SIZE = 1024
NFRAMES = 21

# Flat pastel grounds — the style is "flat pastel, no gradients", so the
# backgrounds are flat too. White first: that is the canonical design-sheet look.
BACKGROUNDS = {
    # slug -> (caption phrase, rgb).  The slug must be unique per background:
    # deriving it from the phrase collides ("pastel" x3) and silently overwrites
    # clips, leaving captions describing a colour the video does not show.
    "white": ("plain white studio background",  (255, 255, 255)),
    "blue":  ("plain pastel blue background",   (198, 222, 241)),
    "peach": ("plain pastel peach background",  (247, 219, 205)),
    "mint":  ("plain pastel mint background",   (206, 235, 219)),
}

# Camera-angle clause per source file. Varying ONLY this across the 7 files is
# what lets the model learn that "happy" is angle-independent rather than baking
# in one viewpoint.
ANGLES = {
    "FRONT":  "facing the camera directly, front view",
    "QF_L":   "turned slightly to its left, three-quarter front view",
    "QF_R":   "turned slightly to its right, three-quarter front view",
    "QF2_L":  "turned further to its left, wide three-quarter view",
    "QF2_R":  "turned further to its right, wide three-quarter view",
    "SIDE_L": "seen from its left side, profile view",
    "Right":  "seen from its right side, profile view",
}

# v2 identity anchor, verbatim from the 75-clip dataset's caption convention.
# Deliberately NOT v4's rare token (pxngn0): we are continue-training the v2
# golden weights, so captions must match the v2 distribution. The anchor already
# says "blue penguin", which satisfies the v4 §5.3 colour-grounding rule.
ANCHOR = ("A 2D cartoon animation in the Pudgy Penguins style, with thick clean "
          "black outlines and flat pastel colors, showing Pax, a short round blue penguin")
ACTION = ("breaking into a big happy smile, beak opening into a wide joyful grin, "
          "eyebrows lifting, cheeks lifting")


def read_rgba(path: Path) -> np.ndarray:
    """Decode ProRes 4444 -> (T,H,W,4) uint8, alpha preserved."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"scale={SIZE}:{SIZE}", "-pix_fmt", "rgba",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 4)


def composite(frames: np.ndarray, rgb: tuple) -> np.ndarray:
    a = frames[..., 3:4].astype(np.float32) / 255.0
    fg = frames[..., :3].astype(np.float32)
    bg = np.array(rgb, np.float32)
    return (fg * a + bg * (1 - a)).round().astype(np.uint8)


def write_mp4(frames: np.ndarray, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SIZE}x{SIZE}", "-r", "24", "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-pix_fmt", "yuv420p", "-an", str(dest)],
        stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode:
        sys.exit(f"ffmpeg failed on {dest}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for src in sorted(SRC.glob("*.mov")):
        stem = src.stem
        if stem not in ANGLES:
            print(f"  ! skipping unmapped {stem}")
            continue
        frames = read_rgba(src)
        assert frames.shape[0] == NFRAMES, f"{stem}: {frames.shape[0]} frames, expected {NFRAMES}"
        for tag, (bg_desc, rgb) in BACKGROUNDS.items():
            dest = OUT / f"{stem}__{tag}.mp4"
            write_mp4(composite(frames, rgb), dest)
            caption = (f"{ANCHOR}, {ACTION}; happy expression; "
                       f"static close-up shot, eye level, {ANGLES[stem]}; {bg_desc}.")
            records.append({"video_path": str(dest), "caption": caption})
            print(f"  ✓ {dest.name}")

    jsonl = Path("/workspace/Pudgy/finetune/wan/dataset_happy.jsonl")
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} training clips ({len(records)//len(BACKGROUNDS)} angles "
          f"x {len(BACKGROUNDS)} backgrounds) -> {jsonl}")


if __name__ == "__main__":
    main()
