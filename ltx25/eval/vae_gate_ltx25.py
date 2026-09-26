#!/usr/bin/env python3
"""
vae_gate_ltx25.py - GATE ZERO for the standalone LTX-2.5 experiment.

STANDALONE. Shares nothing with the v1-v7 tooling.

WHY THIS FILE EXISTS AT ALL
---------------------------
PIPELINE.md S1 makes this the first thing to run and the cheapest thing to falsify:
everything downstream assumes LTX-2.5 can REPRESENT flat 2D art with thin black
outlines. If a pure encode->decode round-trip already damages the outlines, no amount
of data or training recovers it, and that finding is worth more than a trained LoRA.

It could not be answered by `training_approach/scripts/vae_roundtrip.py`, which loads
VAEs the diffusers way (`from_pretrained(repo, subfolder="vae")`). LTX-2.5 ships a
SPLIT PACK: bare `.safetensors` whose config lives in file metadata, built through
`ltx_core`. There is no diffusers layout to point at, so this is a purpose-built
round-trip rather than a registry entry on the old script.

Run it inside the LTX-2 venv, which is where ltx_core/ltx_trainer live:

    cd /workspace/LTX-2 && uv run python /workspace/Pudgy/ltx25/eval/vae_gate_ltx25.py \
        --vae diffusion:/workspace/models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
        --vae conv:/workspace/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
        --clips <a Pax clip> <a Polly clip> --out ltx25/eval/vae_gate.json

WHAT IT MEASURES, AND WHY EACH ONE
----------------------------------
Global PSNR/SSIM alone would pass a decoder that softens every outline uniformly, so
three art-specific measures sit alongside them, plus one diagnostic.

  psnr / ssim      Whole-frame fidelity. The bar is CogVideoX's 38.9 dB / 0.996 on
                   this same Pudgy art (v1-era measurement, quoted in PIPELINE.md S1).
                   Treated as a REFERENCE POINT, not a pass condition - see below.

  edge_psnr        PSNR restricted to outline pixels (upper-quartile Sobel magnitude
                   among POSITIVE gradients in the SOURCE). This is the number that
                   matters. Pudgy art is thin black outlines over flat fills, so a
                   decoder can score well globally by nailing the fills while smearing
                   every line. Percentiles run over positive gradients only because
                   92.8% of pixels on this art have exactly zero gradient - a plain
                   percentile returns 0.0 and the mask degenerates to the whole frame.

  edge_cov         Fraction of pixels the outline mask selected. Diagnostic, not a
                   quality measure: ~0.01-0.10 is healthy, 1.0 means the mask
                   degenerated and edge_psnr is merely global psnr wearing a hat.

  flat_band        Variation the decode INTRODUCES inside regions that are exactly
                   flat in the source. Banding and the flat-region grid artefact
                   reported against 2.3 both show up here as a positive number where
                   the source has none.

  shimmer_abs      Grey levels (0-255) of frame-to-frame flicker the decode ADDS on
                   pixels the SOURCE holds perfectly still. LTX-2 #223 isolated this
                   flicker to a pure VAE round-trip with no transformer involved, and
                   the maintainer closed it as "a known limitation of the current VAE's
                   temporal compression ... LTX-2.5's DiffVAE is a newer decoder; it
                   does not fully remove this class of artifact." Reported as a
                   SUBTRACTION, not a ratio: the source holds those pixels exactly
                   still, so a ratio divides by ~0 and explodes into meaningless
                   magnitudes. 0.0 means the decode adds no temporal noise. This is the
                   specific prediction PIPELINE.md S10.2 asks gate zero to test, and a
                   single-frame check would miss it entirely.

VERDICT
-------
The gate is comparative, not absolute: it picks the better of the two 2.5 decoders and
says whether that one is fit for this art. Thresholds are DEFAULTS, not law - they are
printed alongside every number so a human can overrule them after eyeballing the
side-by-side PNGs this writes. PIPELINE.md S8.3 is emphatic that no metric sees
identity; the same caution applies here, so --dump-frames is on by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

SIZE = 1024  # must match the training bucket; W,H divisible by 32
FRAMES = 17  # the corpus floor bucket, and (17-1) % 8 == 0

# Reference point from CogVideoX's 8x VAE on this same art (FINDINGS S4, via PIPELINE.md S1).
BAR_PSNR = 38.9
BAR_SSIM = 0.996
# Defaults, overridable. edge_psnr is the one to argue about; shimmer_abs above ~1 grey
# level means the decoder injects visible flicker where the artist drew none.
BAR_EDGE_PSNR = 32.0
BAR_SHIMMER_ABS = 1.0   # grey levels of flicker injected where the source is perfectly still


def read_rgb(path: Path, size: int = SIZE, frames: int = FRAMES) -> np.ndarray | None:
    """Decode the first `frames` frames as [F, H, W, 3] uint8, scaled to size x size."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={size}:{size}", "-frames:v", str(frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    n = buf.size // (size * size * 3)
    if n == 0:
        return None
    return buf[: n * size * size * 3].reshape(n, size, size, 3)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude, numpy-only so this adds no dependency."""
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
    gy[1:-1, :] = gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)
    return np.hypot(gx, gy)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    se = (a - b) ** 2
    if mask is not None:
        if mask.ndim == se.ndim - 1:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, se.shape)
        if mask.sum() == 0:
            return float("nan")
        mse = float(se[mask].mean())
    else:
        mse = float(se.mean())
    if mse <= 1e-12:
        return 99.0
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def measure(src: np.ndarray, dec: np.ndarray) -> dict:
    """All four measures for one clip. src/dec are [F, H, W, 3] uint8, same shape."""
    f = min(len(src), len(dec))
    src, dec = src[:f], dec[:f]

    g_src = src.mean(axis=3)
    g_dec = dec.mean(axis=3)

    # Outline mask: the strongest gradients in the SOURCE, so we always ask "what did the
    # decode do to the lines the artist drew", never "where did the decode put edges".
    #
    # MEASURED, and the reason this is not a plain percentile: on this art 92.8% of pixels
    # have EXACTLY zero gradient (flat fills on a flat synthetic ground). np.percentile(
    # mags, 90) therefore returns 0.0, `mags >= 0` selects 100% of the frame, and
    # edge_psnr silently collapses into global psnr - it reported identical values to two
    # decimal places on the first real run. Percentiles are taken over POSITIVE gradients
    # only, and the resulting coverage is reported so this cannot degenerate unnoticed.
    mags = np.stack([sobel_mag(g) for g in g_src])
    pos = mags[mags > 0]
    if pos.size == 0:
        edge_mask = np.zeros_like(mags, dtype=bool)
        flat_mask = np.ones_like(mags, dtype=bool)
    else:
        edge_mask = mags >= np.percentile(pos, 75)   # the outlines themselves
        flat_mask = mags == 0                        # genuinely flat, not "low"
    edge_cov = float(edge_mask.mean())

    # Banding: variation the decode ADDS where the source is flat. Per-frame local
    # deviation from a 1-px-shifted copy approximates "should be constant here".
    def local_dev(g: np.ndarray, m: np.ndarray) -> float:
        d = np.abs(np.diff(g, axis=1))
        mm = m[:, :-1]
        return float(d[mm].mean()) if mm.sum() else 0.0

    band = max(0.0, local_dev(g_dec, flat_mask) - local_dev(g_src, flat_mask))

    # Shimmer: temporal instability the decode ADDS on pixels the source holds still.
    #
    # Reported in ABSOLUTE grey levels (0-255), not as a ratio. A ratio is useless here for
    # the same reason the percentile was: the source holds those pixels PERFECTLY still, so
    # the denominator is ~0 and the quotient explodes to meaningless magnitudes (the first
    # real run produced 15-41, and a synthetic check produced 1.9e6). The honest question is
    # "how many grey levels of flicker does the decode inject where the artist drew none",
    # and that is a subtraction, not a division. LTX-2 #223 is precisely this defect.
    if f > 1:
        d_src = np.abs(np.diff(g_src, axis=0))
        d_dec = np.abs(np.diff(g_dec, axis=0))
        static = d_src < 1.0                     # source effectively unchanged here
        if static.sum():
            s_src = float(d_src[static].mean())
            s_dec = float(d_dec[static].mean())
            shimmer_abs = max(0.0, s_dec - s_src)
        else:
            shimmer_abs = float("nan")
    else:
        shimmer_abs = float("nan")

    s = float("nan")
    if ssim_fn is not None:
        s = float(np.mean([
            ssim_fn(src[i], dec[i], channel_axis=2, data_range=255) for i in range(f)
        ]))

    return {
        "frames": f,
        "psnr": round(psnr(src, dec), 2),
        "ssim": round(s, 4) if s == s else None,
        "edge_psnr": round(psnr(src, dec, edge_mask), 2),
        "edge_cov": round(edge_cov, 4),   # sanity: ~0.01-0.10 expected; 1.0 means degenerate
        "flat_band": round(band, 3),
        "shimmer_abs": round(shimmer_abs, 3) if shimmer_abs == shimmer_abs else None,
    }


def roundtrip(vae_path: str, clips: list[Path], device: str, dump: Path | None) -> dict:
    """Encode->decode every clip through one VAE checkpoint. Imports are local so that
    --help and the metric code stay usable outside the LTX venv."""
    import torch
    from ltx_core.model.video_vae import TileSizeConfig
    from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder

    dev = torch.device(device)
    enc = load_video_vae_encoder(vae_path, device=dev, dtype=torch.bfloat16)
    dec = load_video_vae_decoder(vae_path, device=dev, dtype=torch.bfloat16)

    per_clip = {}
    for c in clips:
        src = read_rgb(c)
        if src is None:
            print(f"  !! could not decode {c.name}", file=sys.stderr)
            continue
        # [F,H,W,3] uint8 -> [1,3,F,H,W] in [-1,1], the convention the trainer encodes with
        v = torch.from_numpy(src.copy()).float().div_(255.0)
        v = v.permute(3, 0, 1, 2).unsqueeze(0).mul_(2.0).sub_(1.0)

        with torch.no_grad():
            lat = enc.tiled_encode(v.to(device=dev, dtype=torch.bfloat16), TileSizeConfig.default())
            chunks = [ch.float().cpu().numpy() for ch in dec.decode_video(lat.to(dev))]

        # decode_video yields [f, h, w, c] in [0, 1]
        out = np.concatenate(chunks, axis=0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        per_clip[c.name] = measure(src, out)
        print(f"  {c.name:52s} {per_clip[c.name]}")

        if dump is not None:
            from PIL import Image
            dump.mkdir(parents=True, exist_ok=True)
            tag = Path(vae_path).stem.replace("ltx-2.5-video-vae-", "").replace("-bf16", "")
            i = min(len(src), len(out)) // 2  # a mid clip frame, not frame 0
            Image.fromarray(np.concatenate([src[i], out[i]], axis=1)).save(
                dump / f"{c.stem}__{tag}__src_vs_decoded.png"
            )

    agg = {}
    if per_clip:
        for k in ("psnr", "ssim", "edge_psnr", "edge_cov", "flat_band", "shimmer_abs"):
            vals = [m[k] for m in per_clip.values() if m.get(k) is not None]
            agg[k] = round(float(np.mean(vals)), 4) if vals else None
    return {"per_clip": per_clip, "mean": agg}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vae", action="append", required=True, metavar="NAME:PATH",
                    help="repeatable, e.g. diffusion:/path/vae.safetensors")
    ap.add_argument("--clips", type=Path, nargs="+", required=True,
                    help="real Pudgy clips; include BOTH characters")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("ltx25/eval/vae_gate.json"))
    ap.add_argument("--dump-frames", type=Path, default=Path("ltx25/eval/vae_gate_frames"),
                    help="side-by-side PNGs; ALWAYS look at these, no metric sees identity")
    ap.add_argument("--no-dump", action="store_true")
    args = ap.parse_args()

    if ssim_fn is None:
        print("note: scikit-image absent, ssim will be null (other measures unaffected)\n")

    dump = None if args.no_dump else args.dump_frames
    results = {}
    for spec in args.vae:
        name, _, path = spec.partition(":")
        if not path:
            sys.exit(f"--vae wants NAME:PATH, got {spec!r}")
        if not Path(path).exists():
            sys.exit(f"no such checkpoint: {path}")
        print(f"\n=== {name} :: {Path(path).name} ===")
        results[name] = roundtrip(path, args.clips, args.device, dump)

    # Verdict: rank on edge_psnr, the measure this art actually lives or dies by.
    ranked = sorted(
        (n for n in results if results[n]["mean"].get("edge_psnr") is not None),
        key=lambda n: results[n]["mean"]["edge_psnr"],
        reverse=True,
    )
    print("\n" + "=" * 78)
    print(f"{'decoder':12s} {'psnr':>7s} {'ssim':>7s} {'edge_psnr':>10s} {'edge_cov':>9s} {'flat_band':>10s} {'shimmer':>8s}")
    for n in results:
        m = results[n]["mean"]
        print(f"{n:12s} {m.get('psnr', 0):7.2f} {str(m.get('ssim')):>7s} "
              f"{m.get('edge_psnr', 0):10.2f} {m.get('edge_cov', 0):9.3f} "
              f"{m.get('flat_band', 0):10.3f} {str(m.get('shimmer_abs')):>8s}")
    print(f"\nreference (CogVideoX on this art): psnr {BAR_PSNR}  ssim {BAR_SSIM}")
    print(f"defaults: edge_psnr >= {BAR_EDGE_PSNR}, shimmer_abs <= {BAR_SHIMMER_ABS} levels")

    verdict = {"picked": None, "pass": False, "why": "no decoder produced measurements"}
    if ranked:
        best = ranked[0]
        m = results[best]["mean"]
        ok_edge = m["edge_psnr"] >= BAR_EDGE_PSNR
        ok_shim = (m.get("shimmer_abs") or 0) <= BAR_SHIMMER_ABS
        verdict = {
            "picked": best,
            "pass": bool(ok_edge and ok_shim),
            "why": (f"edge_psnr {m['edge_psnr']} vs {BAR_EDGE_PSNR}; "
                    f"shimmer_abs {m.get('shimmer_abs')} vs {BAR_SHIMMER_ABS}"),
        }
        print(f"\nBEST: {best}  ->  {'PASS' if verdict['pass'] else 'FAIL'}  ({verdict['why']})")
        if not verdict["pass"]:
            print("PIPELINE.md S1: if BOTH decoders soften outlines materially, STOP -\n"
                  "no amount of data or training recovers what the VAE discards.")
    print("=" * 78)
    print(f"\n!! Eyeball {dump}/ before acting on any of this. No metric sees identity.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"size": SIZE, "frames": FRAMES, "reference": {"psnr": BAR_PSNR, "ssim": BAR_SSIM},
         "thresholds": {"edge_psnr": BAR_EDGE_PSNR, "shimmer_abs": BAR_SHIMMER_ABS},
         "results": results, "verdict": verdict}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
