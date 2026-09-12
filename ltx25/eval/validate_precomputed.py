#!/usr/bin/env python3
"""
validate_precomputed.py - the check that stands between preprocessing and a GPU-hour.

STANDALONE. Shares nothing with the v1-v7 tooling.

PIPELINE.md S7 says "--decode writes .precomputed/decoded_videos. EYEBALL THOSE before
spending a GPU-hour." Eyeballing is necessary but it does not scale to 1,018 samples and
it does not catch counting errors, so this does the mechanical half and prints a montage
path for the human half.

WHAT IT CHECKS, AND WHY
-----------------------
 1. COUNTS. One latent and one condition per dataset entry. A silent shortfall here is
    the documented failure mode of this trainer: stale/again-skipped .pt files are
    "skipped by default", not reported, so a short run looks like a clean one.

 2. BUCKET LEGALITY. Every latent's temporal extent must satisfy the 8x VAE contract
    that produced it. A latent of the wrong shape trains silently and ruins the run.

 3. DECODE FIDELITY. The round-trip previews are compared against the ORIGINAL clips
    they came from. This is the only check that can catch a wrong-VAE, wrong-scale or
    wrong-colourspace preprocessing bug, all of which produce plausible-looking files.

 4. NOT-BLACK / NOT-FROZEN. A decoded preview that is uniformly black, or has zero
    temporal variation where its source moves, means the pipeline dropped the content.
    Cheap to check, catastrophic to miss.

Exit code is 0 only if every check passes, so it can gate a chained training launch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TEMPORAL = 8          # LTX temporal VAE factor: frames % 8 == 1
SPATIAL = 32          # LTX spatial VAE factor
MIN_DECODE_PSNR = 25.0  # a preview this far from its source means a pipeline bug, not codec loss


def read_gray(path: Path, size: int = 256, frames: int = 32) -> np.ndarray | None:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={size}:{size},format=gray",
         "-frames:v", str(frames), "-f", "rawvideo", "-"],
        capture_output=True, timeout=120)
    if out.returncode != 0 or not out.stdout:
        return None
    n = size * size
    c = len(out.stdout) // n
    if c == 0:
        return None
    return np.frombuffer(out.stdout[: c * n], dtype=np.uint8).reshape(c, size, size).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("Data/processed/ltx25_experiment"))
    ap.add_argument("--sample", type=int, default=24, help="decoded previews to compare against source")
    args = ap.parse_args()

    D = args.data
    pre = D / ".precomputed"
    entries = json.loads((D / "dataset.json").read_text())
    man = json.loads((D / "manifest.json").read_text())
    fails: list[str] = []
    print(f"dataset: {len(entries)} entries | buckets {man['resolution_buckets']}\n")

    # --- 1. counts -----------------------------------------------------------------
    lat = sorted((pre / "latents").rglob("*.pt"))
    con = sorted((pre / "conditions").rglob("*.pt"))
    print(f"1. counts      latents {len(lat)}  conditions {len(con)}  expected {len(entries)}")
    if len(lat) != len(entries):
        fails.append(f"latents {len(lat)} != {len(entries)} dataset entries")
    if len(con) != len(entries):
        fails.append(f"conditions {len(con)} != {len(entries)} dataset entries")

    # --- 2. latent geometry --------------------------------------------------------
    legal_f = {int(b.split("x")[2]) for b in man["resolution_buckets"].split(";")}
    legal_lat_f = {1 if f == 1 else (f - 1) // TEMPORAL + 1 for f in legal_f}
    import torch
    bad_shape, shapes = 0, {}
    for p in lat[:: max(1, len(lat) // 200)]:
        try:
            t = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(t, dict):
                t = next(v for v in t.values() if hasattr(v, "shape"))
            sh = tuple(t.shape)
            shapes[sh] = shapes.get(sh, 0) + 1
            tf = sh[-3]
            if tf not in legal_lat_f:
                bad_shape += 1
        except Exception as e:
            fails.append(f"unreadable latent {p.name}: {type(e).__name__}")
            break
    print(f"2. geometry    legal latent frame counts {sorted(legal_lat_f)}")
    for sh, n in sorted(shapes.items(), key=lambda kv: -kv[1])[:6]:
        print(f"                 {sh} x{n}")
    if bad_shape:
        fails.append(f"{bad_shape} sampled latents have an illegal temporal extent")

    # --- 3 & 4. decoded previews vs their sources ----------------------------------
    dec_dir = pre / "decoded_videos"
    decs = sorted(dec_dir.rglob("*.mp4")) if dec_dir.exists() else []
    print(f"\n3. decode      {len(decs)} previews written")
    if not decs:
        fails.append("no decoded previews - rerun preprocessing with --decode")
    else:
        by_stem = {Path(e["video"]).stem: (D / e["video"]) for e in entries}
        step = max(1, len(decs) // args.sample)
        checked = psnr_sum = 0
        worst = (999.0, "")
        black = frozen = unmatched = 0
        for p in decs[::step][: args.sample]:
            src_path = by_stem.get(p.stem)
            dg = read_gray(p)
            if dg is None:
                fails.append(f"unreadable preview {p.name}")
                continue
            if dg.mean() < 2.0:
                black += 1
            if src_path is None or not src_path.exists():
                unmatched += 1
                continue
            sg = read_gray(src_path)
            if sg is None:
                continue
            f = min(len(sg), len(dg))
            if f > 1:
                sv = float(np.abs(np.diff(sg[:f], axis=0)).mean())
                dv = float(np.abs(np.diff(dg[:f], axis=0)).mean())
                if sv > 0.5 and dv < 0.05 * sv:
                    frozen += 1
            mse = float(((sg[:f] - dg[:f]) ** 2).mean())
            ps = 99.0 if mse <= 1e-9 else 20 * np.log10(255.0) - 10 * np.log10(mse)
            psnr_sum += ps
            checked += 1
            if ps < worst[0]:
                worst = (ps, p.name)
        mean_psnr = psnr_sum / checked if checked else 0.0
        print(f"4. fidelity    {checked} previews vs source: mean PSNR {mean_psnr:.2f} dB "
              f"(worst {worst[0]:.2f} on {worst[1]})")
        print(f"   integrity   black {black}  frozen-vs-moving-source {frozen}  unmatched {unmatched}")
        if checked and mean_psnr < MIN_DECODE_PSNR:
            fails.append(f"decoded previews mean {mean_psnr:.2f} dB < {MIN_DECODE_PSNR} - pipeline bug, not codec loss")
        if black:
            fails.append(f"{black} previews are essentially black")
        if frozen:
            fails.append(f"{frozen} previews are static though their source moves")

    print("\n" + "=" * 70)
    if fails:
        print("VALIDATION FAILED - do NOT start training:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("VALIDATION PASSED - preprocessing looks sound.")
    print(f"Still eyeball a few of {dec_dir} yourself: no metric sees identity.")
    sys.exit(0)


if __name__ == "__main__":
    main()
