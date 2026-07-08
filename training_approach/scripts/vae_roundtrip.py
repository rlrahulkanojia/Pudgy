#!/usr/bin/env python3
"""
VAE round-trip diagnostic  —  Training Approach v2, Phase 0.1 (Gate G0).

WHY: every candidate video base model has a FROZEN VAE. Whatever the VAE cannot
reconstruct, no LoRA or fine-tune can ever recover — it is a hard quality ceiling.
Flat pastel fills + thick black outlines (the Pudgy style) are exactly the content
high-compression VAEs damage (soft/doubled outlines, banding on flat fills).

This script encodes -> decodes real Pudgy clips through one VAE and reports how much
was lost, with numbers AND look-at-it crops focused on outlines and flat regions.

RUN ON THE GPU BOX (CUDA). One VAE per invocation. Compare the CSVs/montages after.

--------------------------------------------------------------------------------
USAGE
  python vae_roundtrip.py --vae wan21     --clips /data/training_dataset/train --out ./vae_out
  python vae_roundtrip.py --vae wan22_5b  --clips /data/training_dataset/train --out ./vae_out
  python vae_roundtrip.py --vae hunyuan   --clips /data/training_dataset/train --out ./vae_out
  python vae_roundtrip.py --vae cogvideox --clips /data/training_dataset/train --out ./vae_out

  # quick smoke test on 3 clips:
  python vae_roundtrip.py --vae wan21 --clips /data/training_dataset/train --n 3

DEPENDENCIES (in the trainer venv; add if missing)
  pip install "diffusers>=0.32" transformers accelerate torch imageio imageio-ffmpeg \
              numpy pillow scikit-image tqdm

NOTES / VERIFY-ON-YOUR-BUILD
  * The 33-frame clips in training_dataset/train are ideal inputs: 33 = 4*8+1 satisfies
    the (frames-1) % temporal_compression == 0 rule for all four VAEs (temporal ratio 4).
  * HF repo IDs below may be GATED (accept the license once) and class names track your
    installed diffusers version. If a class import fails, `pip install -U diffusers` or
    check the model card's "Diffusers usage" snippet and update REGISTRY.
  * Wan 2.2 A14B reuses the Wan 2.1 VAE (8x spatial) -> test `wan21` for it.
    `wan22_5b` is the DIFFERENT high-compression (16x spatial) VAE — expected worst on 2D.
"""

import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None


# --------------------------------------------------------------------------- #
# VAE registry: which diffusers class + HF source to load for each candidate.
# Every entry loads the `vae` subfolder of the model repo.
# --------------------------------------------------------------------------- #
REGISTRY = {
    # Wan 2.1 VAE (8x spatial, 4x temporal). ALSO the VAE used by Wan 2.2 A14B.
    "wan21": dict(
        cls="AutoencoderKLWan",
        repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        note="Wan2.1 / Wan2.2-A14B VAE — 8x spatial",
    ),
    # Wan 2.2 TI2V-5B VAE (16x spatial) — the high-compression one. Expected worst for 2D.
    "wan22_5b": dict(
        cls="AutoencoderKLWan",
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        note="Wan2.2-5B VAE — 16x spatial (high compression)",
    ),
    # HunyuanVideo VAE (8x spatial, 4x temporal).
    "hunyuan": dict(
        cls="AutoencoderKLHunyuanVideo",
        repo="hunyuanvideo-community/HunyuanVideo",
        note="HunyuanVideo VAE — 8x spatial",
    ),
    # CogVideoX1.5 VAE (8x spatial, 4x temporal) — the incumbent baseline.
    "cogvideox": dict(
        cls="AutoencoderKLCogVideoX",
        repo="THUDM/CogVideoX1.5-5B-I2V",
        note="CogVideoX1.5 VAE — 8x spatial (v1 baseline)",
    ),
}


def load_vae(key, dtype, device):
    import importlib
    spec = REGISTRY[key]
    diffusers = importlib.import_module("diffusers")
    cls = getattr(diffusers, spec["cls"])
    print(f"[load] {key}: {spec['cls']} from {spec['repo']}  ({spec['note']})")
    vae = cls.from_pretrained(spec["repo"], subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).eval()
    # Cut VAE peak memory on big frames.
    for m in ("enable_slicing", "enable_tiling"):
        if hasattr(vae, m):
            getattr(vae, m)()
    return vae


def read_clip(path, max_frames):
    """Read an mp4 -> float tensor [C, T, H, W] in [-1, 1]. Clamp T to a valid 4k+1."""
    frames = iio.imread(path, plugin="pyav")          # [T, H, W, C] uint8
    frames = np.asarray(frames)
    if frames.ndim == 3:                               # single frame safeguard
        frames = frames[None]
    T = frames.shape[0]
    if max_frames:
        T = min(T, max_frames)
    # enforce (T-1) % 4 == 0 for the 4x temporal VAEs
    T = ((T - 1) // 4) * 4 + 1
    frames = frames[:T]
    x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0  # [C,T,H,W]
    return x


@torch.no_grad()
def roundtrip(vae, x, device, dtype):
    """x: [C,T,H,W] in [-1,1] -> reconstruction in [-1,1], same shape."""
    inp = x.unsqueeze(0).to(device=device, dtype=dtype)       # [1,C,T,H,W]
    posterior = vae.encode(inp)
    lat = posterior.latent_dist.sample() if hasattr(posterior, "latent_dist") else posterior.latents
    # apply the model's latent scaling if present (round-trip is scale-invariant, but
    # keep it honest so decode() sees what it expects)
    sf = getattr(vae.config, "scaling_factor", None)
    if sf:
        lat = lat * sf
        rec = vae.decode(lat / sf).sample
    else:
        rec = vae.decode(lat).sample
    rec = rec.clamp(-1, 1).squeeze(0).float().cpu()           # [C,T,H,W]
    return rec


# --------------------------- metrics --------------------------- #
def to_uint8_thw_c(t):  # [C,T,H,W] in [-1,1] -> [T,H,W,C] uint8
    a = ((t.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
    return a


def luminance(u8):      # [...,3] uint8 -> float [0,1]
    return (0.299 * u8[..., 0] + 0.587 * u8[..., 1] + 0.114 * u8[..., 2]) / 255.0


def sobel_edges(gray):  # [H,W] float -> edge magnitude [H,W]
    g = torch.from_numpy(gray)[None, None].float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()[None, None]
    ky = kx.transpose(2, 3)
    ex = torch.nn.functional.conv2d(g, kx, padding=1)
    ey = torch.nn.functional.conv2d(g, ky, padding=1)
    return torch.sqrt(ex ** 2 + ey ** 2)[0, 0].numpy()


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse < 1e-8 else 20 * np.log10(255.0 / np.sqrt(mse))


def frame_metrics(src_u8, rec_u8):
    """Per-frame: overall PSNR/SSIM, edge-SSIM (outline fidelity), flat-area MAE (banding)."""
    p = psnr(src_u8, rec_u8)
    gs, gr = luminance(src_u8), luminance(rec_u8)
    s = ssim_fn(gs, gr, data_range=1.0) if ssim_fn else float("nan")
    # edge fidelity: SSIM of Sobel magnitude — how well thick outlines survive
    es, er = sobel_edges(gs), sobel_edges(gr)
    e_ssim = ssim_fn(es, er, data_range=float(max(es.max(), 1e-6))) if ssim_fn else float("nan")
    # banding on flat fills: mean abs luma error where the SOURCE gradient is near-zero
    flat = es < (0.06 * es.max() + 1e-6)
    flat_mae = float(np.mean(np.abs(gs[flat] - gr[flat]))) if flat.any() else float("nan")
    return dict(psnr=p, ssim=s, edge_ssim=e_ssim, flat_mae=flat_mae)


def save_montage(src_u8, rec_u8, out_png):
    """Mid-frame: source | recon | 5x abs-diff, plus a 256px outline crop row."""
    T = src_u8.shape[0]
    i = T // 2
    s, r = src_u8[i], rec_u8[i]
    diff = np.clip(np.abs(s.astype(np.int16) - r.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
    H, W = s.shape[:2]
    # crop a 256x256 patch near center to inspect linework closely
    cy, cx = H // 2, W // 2
    y0, x0 = max(0, cy - 128), max(0, cx - 128)
    crop = lambda im: im[y0:y0 + 256, x0:x0 + 256]
    top = np.concatenate([s, r, diff], axis=1)
    cs, cr, cd = crop(s), crop(r), crop(diff)
    bottom = np.concatenate([cs, cr, cd], axis=1)
    # pad bottom to top width
    if bottom.shape[1] < top.shape[1]:
        pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), np.uint8)
        bottom = np.concatenate([bottom, pad], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    Image.fromarray(grid).save(out_png)


def main():
    ap = argparse.ArgumentParser(description="VAE round-trip diagnostic (Phase 0.1)")
    ap.add_argument("--vae", required=True, choices=list(REGISTRY), help="which VAE to test")
    ap.add_argument("--clips", required=True, help="folder of .mp4 clips (e.g. training_dataset/train)")
    ap.add_argument("--out", default="./vae_out", help="output root")
    ap.add_argument("--n", type=int, default=8, help="number of clips to sample")
    ap.add_argument("--max-frames", type=int, default=33, help="frames per clip (kept to 4k+1)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. The video VAEs are heavy; CPU will be very slow "
              "and may OOM. Run this on the GPU box.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if ssim_fn is None:
        print("NOTE: scikit-image not installed -> SSIM/edge-SSIM will be NaN. "
              "`pip install scikit-image` for the full report.", file=sys.stderr)

    clip_paths = sorted(Path(args.clips).glob("*.mp4"))[: args.n]
    if not clip_paths:
        sys.exit(f"No .mp4 clips found in {args.clips}")

    out_dir = Path(args.out) / args.vae
    (out_dir / "montages").mkdir(parents=True, exist_ok=True)

    vae = load_vae(args.vae, dtype, device)

    rows, agg = [], {k: [] for k in ("psnr", "ssim", "edge_ssim", "flat_mae")}
    for p in tqdm(clip_paths, desc=f"round-trip [{args.vae}]"):
        x = read_clip(str(p), args.max_frames)
        rec = roundtrip(vae, x, device, dtype)
        src_u8, rec_u8 = to_uint8_thw_c(x), to_uint8_thw_c(rec)
        T = min(src_u8.shape[0], rec_u8.shape[0])
        per_frame = [frame_metrics(src_u8[i], rec_u8[i]) for i in range(T)]
        clip_m = {k: float(np.nanmean([f[k] for f in per_frame])) for k in agg}
        for k in agg:
            agg[k].append(clip_m[k])
        rows.append(dict(clip=p.name, **{k: round(clip_m[k], 4) for k in agg}))
        save_montage(src_u8, rec_u8, str(out_dir / "montages" / f"{p.stem}.png"))

    summary = {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}
    summary.update(vae=args.vae, repo=REGISTRY[args.vae]["repo"],
                   note=REGISTRY[args.vae]["note"], n_clips=len(clip_paths))

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "psnr", "ssim", "edge_ssim", "flat_mae"])
        w.writeheader(); w.writerows(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print(f"VAE:        {summary['vae']}  ({summary['note']})")
    print(f"clips:      {summary['n_clips']}")
    print(f"PSNR:       {summary['psnr']:.2f} dB   (higher = better)")
    print(f"SSIM:       {summary['ssim']:.4f}      (higher = better)")
    print(f"edge-SSIM:  {summary['edge_ssim']:.4f}      (OUTLINE fidelity — the 2D-critical one)")
    print(f"flat-MAE:   {summary['flat_mae']:.4f}      (BANDING on flat fills — lower = better)")
    print("=" * 68)
    print(f"artifacts -> {out_dir}/  (summary.json, per_clip.csv, montages/*.png)")
    print("Compare edge-SSIM and flat-MAE ACROSS the 4 VAEs to pick the base (Gate G0).")


if __name__ == "__main__":
    main()
