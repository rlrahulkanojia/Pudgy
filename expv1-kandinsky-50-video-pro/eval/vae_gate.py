#!/usr/bin/env python3
"""expv1 G0.1 — VAE gate (PLAN §4). Stop expv1 if it fails.

Round-trips real Pudgy art through Kandinsky's own VAE code path
(kandinsky.models.vae.AutoencoderKLHunyuanVideo at kandinsky-5 @ 3a74261, fp16 as
build_vae loads it, the pipeline's automatic tiling: encode(opt_tiling=True), decode()),
with Wan2.1's VAE (v8's) on the same clips as the reference.

Clips (built here from Data/raw, 1080^2 ProRes 4444 with alpha, 24 fps; no processed
dataset is carried over):  Pax and Polly x {angry (expression), waving (in-place motion),
running}, composited on white and mint, n = 4k+1 frames, plus one flat-fill clip.
Sizes: 512 (the training size, gated) and 1024 (for the record).

Pass rule, fixed before any data was run (2026-10-12). PLAN §4 asks for "outlines and
highlights intact at 512^2 and no grid or banding on flat fills"; expv2's G0.1 showed the
eye alone is not enough, so the same condition is measured, on every 512^2 clip:
  1. outline centerline survival: of the source's skeletonised outline (lum < 64), the
     share still dark (< 128) after the round trip; min over frames >= 0.95 AND
     >= Wan2.1's on the same clip - 0.03;
  2. broken strokes: gaps of >= 6 connected centerline pixels lost, max per frame <= 2;
  3. highlight cores (white blobs enclosed by black, eroded 1 px) still > 200: min >= 0.95;
  4. ground drift (mean abs per-channel error on ground pixels) <= 2/255 on every clip;
  5. flat fill: spatial std <= 1, mean abs error <= 2 grey levels.
Informational: PSNR/SSIM/edge-PSNR, motion-energy ratio (does the running stride survive
the 4x temporal compression, A13), and the same table at 1024^2.

    python expv1-kandinsky-50-video-pro/eval/vae_gate.py
-> $EXPV1_WORK/eval/phase0/vae_gate/{report.json, sheets/*.png}

Metric code adapted from expv2-minimax-h3/eval/vae_gate.py (ltx25 gate definitions) and the
expv2 G0.1 audit (centerline / broken-stroke / highlight-core checks); copied, not imported.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PUDGY_DATA", "/workspace/Data")) / "raw"
WORK = Path(os.environ.get("EXPV1_WORK", "/workspace/expv1"))
OUT = WORK / "eval" / "phase0" / "vae_gate"
K5_REPO = Path(os.environ.get("K5_REPO", "/workspace/kandinsky-5"))   # @ 3a74261
K5_VAE = Path(os.environ.get("K5_VAE", "/workspace/models/hunyuanvideo-community/HunyuanVideo"))
WAN21 = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

CLIPS = {  # id -> raw path (FRONT view; the newest delivery of each)
    "pax_angry": "iteration_5/EXPRESSIONS/ANGRY/PAX/PAX_ANGRY_FRONT.mov",
    "polly_angry": "iteration_5/EXPRESSIONS/ANGRY/POLLY/POLLY_ANGRY_FRONT.mov",
    "pax_waving": "iteration_4/LSLTTT-Project/MOTION_CLIPS/MOTION_WAVING/PAX/PAX_MOTION_WAVING_FRONT.mov",
    "polly_waving": "iteration_4/LSLTTT-Project/MOTION_CLIPS/MOTION_WAVING/POLLY/POLLY_MOTION_WAVING_FRONT.mov",
    "pax_running": "iteration_5/MOTION_CLIPS/MOTION_RUNNING/PAX/PAX_MOTION_RUNNING_FRONT.mov",
    "polly_running": "iteration_5/MOTION_CLIPS/MOTION_RUNNING/POLLY/POLLY_MOTION_RUNNING_FRONT.mov",
}
GROUNDS = {"white": (255, 255, 255), "mint": (206, 235, 219)}
SIZES = (512, 1024)
GATE_SIZE = 512
MAX_FRAMES = 41          # 4k+1; covers the angry arc, several waving/running cycles

BAR = dict(centerline_min=0.95, centerline_vs_ref=0.03, broken_max=2, highlight_min=0.95,
           ground_drift=2.0, fill_std=1.0, fill_mae=2.0)


# ---------------------------------------------------------------------------- clips
def k5_len(n):
    """Longest length <= n that Kandinsky's VAE returns whole. Its encode tiles time in
    17-frame windows at stride 8 (get_enc_optimal_tiling -> (1, 17, h, w), (8, h, w)), so
    above 17 frames only 8k+1 survives: measured 21->9, 29->25, 37->33 frames; 13, 17, 25,
    33, 41, 49, 57, 121 intact (MODEL_NOTES §2.8)."""
    if n <= 17:
        return ((n - 1) // 4) * 4 + 1
    return ((n - 1) // 8) * 8 + 1


def decode_raw(path, size, ground, tries=5):
    """ProRes+alpha -> composited RGB at size^2. Decoded twice and compared: ProRes decodes
    can tear non-deterministically (A7); a clip must decode identically twice."""
    r, g, b = ground
    flt = (f"color=c=0x{r:02x}{g:02x}{b:02x}:s={size}x{size}:r=24[bg];"
           f"[0:v]scale={size}:{size}:flags=lanczos,format=rgba[fg];"
           f"[bg][fg]overlay=shortest=1,format=rgb24")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-filter_complex", flt,
           "-frames:v", str(MAX_FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    prev = None
    for _ in range(tries):
        buf = subprocess.run(cmd, capture_output=True, check=True).stdout
        if prev is not None and buf == prev:
            a = np.frombuffer(buf, np.uint8).reshape(-1, size, size, 3)
            n = k5_len(len(a))
            return a[:n].copy()
        prev = buf
    raise RuntimeError(f"{path}: decode not stable over {tries} tries (tearing)")


def build_clips():
    clips = {}
    for size in SIZES:
        for cid, rel in CLIPS.items():
            for gname, rgb in GROUNDS.items():
                clips[(size, f"{cid}__{gname}")] = decode_raw(RAW / rel, size, rgb)
        clips[(size, "flatfill__mint")] = np.tile(np.array(GROUNDS["mint"], np.uint8),
                                                 (25, size, size, 1))
    return clips


# ---------------------------------------------------------------------------- VAEs
def load_k5_vae():
    import torch
    import importlib.util
    # the file itself: kandinsky/__init__ imports the whole pipeline (omegaconf, flash-attn, ...)
    spec = importlib.util.spec_from_file_location("k5_vae", K5_REPO / "kandinsky" / "models" / "vae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AutoencoderKLHunyuanVideo = mod.AutoencoderKLHunyuanVideo   # the class build_vae uses
    vae = AutoencoderKLHunyuanVideo.from_pretrained(str(K5_VAE), subfolder="vae", torch_dtype=torch.float16)
    return vae.to("cuda").eval()


def k5_roundtrip(vae, u8):
    """As the pipeline: generation_utils.encode_video -> vae.encode(x) (opt_tiling=True),
    generate_sample -> vae.decode(z).sample, clamp, (x+1)*127.5 -> uint8. mode() instead of
    sample() so the gate is deterministic."""
    import torch
    x = torch.from_numpy(u8).permute(3, 0, 1, 2)[None].to("cuda", torch.float16) / 127.5 - 1.0
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        rec = vae.decode(z).sample
    rec = ((rec.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()
    return rec


def load_wan21():
    import torch
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(WAN21, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    vae.enable_tiling()
    return vae


def wan_roundtrip(vae, u8):
    import torch
    x = torch.from_numpy(u8).float().permute(3, 0, 1, 2)[None].to("cuda") / 127.5 - 1.0
    with torch.no_grad():
        rec = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)[0].float().cpu()
    return ((rec.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()


# ---------------------------------------------------------------------------- metrics
def lum(a):
    return a[..., :3].astype(np.float32) @ np.array([.299, .587, .114], np.float32)


def psnr(a, b, mask=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if not mask.any():
            return None
        d = d[mask]
    return float(10 * np.log10(255 ** 2 / max(float(d.mean()), 1e-9)))


def highlights(L):
    from scipy import ndimage as nd
    w = L > 230
    lab, n = nd.label(w)
    out = np.zeros_like(w)
    for i, s in enumerate(nd.sum(w, lab, range(1, n + 1)), 1):
        if 4 <= s <= 600:
            comp = lab == i
            ring = nd.binary_dilation(comp, iterations=2) & ~comp
            if (L[ring] < 80).mean() > 0.6:
                out |= comp
    return out


def measure(src, dec, ground):
    from scipy import ndimage as nd
    from skimage.metrics import structural_similarity as ssim
    from skimage.morphology import skeletonize
    T = min(len(src), len(dec))
    src, dec = src[:T], dec[:T]
    # the ground as rendered (ffmpeg's colour conversion moves the nominal RGB by a level or
    # two): the most common colour in the source's border rows, matched within 2 levels
    edge = np.concatenate([src[:, :4].reshape(-1, 3), src[:, -4:].reshape(-1, 3)])
    cols, cnt = np.unique(edge, axis=0, return_counts=True)
    g0 = cols[cnt.argmax()].astype(np.int16)
    gmask = np.all(np.abs(src.astype(np.int16) - g0) <= 2, axis=-1)
    cen, brk, hl, ssims, ep = [], [], [], [], []
    for i in range(T):
        Ls, Ld = lum(src[i]), lum(dec[i])
        o = Ls < 64
        if o.any():
            sk = skeletonize(o)
            dark = Ld < 128
            cen.append(float(dark[sk].mean()))
            lost = sk & ~dark
            lab, n = nd.label(lost, structure=np.ones((3, 3)))
            brk.append(int((nd.sum(lost, lab, range(1, n + 1)) >= 6).sum()) if n else 0)
        core = nd.binary_erosion(highlights(Ls), iterations=1)
        if core.any():
            hl.append(float((Ld[core] > 200).mean()))
        g = np.hypot(*np.gradient(Ls))
        e = g > np.percentile(g[g > 0], 75) if (g > 0).any() else None
        if e is not None:
            v = psnr(src[i], dec[i], e)
            if v is not None:
                ep.append(v)
        ssims.append(ssim(src[i], dec[i], channel_axis=2, data_range=255))
    gd = float(np.abs(src.astype(np.float32) - dec.astype(np.float32))[gmask].mean()) if gmask.any() else None
    me_s = float(np.abs(np.diff(lum(src), axis=0)).mean()) if T > 1 else None
    me_d = float(np.abs(np.diff(lum(dec), axis=0)).mean()) if T > 1 else None
    return {
        "frames": T, "psnr": round(psnr(src, dec), 2), "ssim": round(float(np.mean(ssims)), 4),
        "edge_psnr": round(float(np.mean(ep)), 2) if ep else None,
        "centerline_min": round(min(cen), 4) if cen else None,
        "centerline_mean": round(float(np.mean(cen)), 4) if cen else None,
        "broken_max": max(brk) if brk else None,
        "highlight_core_min": round(min(hl), 4) if hl else None,
        "ground_drift": round(gd, 3) if gd is not None else None,
        "fill_std": round(float(dec.reshape(T, -1, 3).std(axis=1).mean()), 3),
        "fill_mae": round(float(np.abs(dec.astype(np.float32) - src.astype(np.float32)).mean()), 3),
        "motion_ratio": round(me_d / me_s, 3) if me_s else None,
    }


def gate(res):
    """Apply the pre-registered rule to the 512^2 K5 results."""
    fails = []
    for cid, r in res[GATE_SIZE].items():
        k, w = r["k5"], r["wan21"]
        if cid.startswith("flatfill"):
            if k["fill_std"] > BAR["fill_std"] or k["fill_mae"] > BAR["fill_mae"]:
                fails.append(f"{cid}: flat fill std {k['fill_std']} / mae {k['fill_mae']}")
            continue
        if k["centerline_min"] < BAR["centerline_min"]:
            fails.append(f"{cid}: centerline_min {k['centerline_min']} < {BAR['centerline_min']}")
        if k["centerline_min"] < w["centerline_min"] - BAR["centerline_vs_ref"]:
            fails.append(f"{cid}: centerline_min {k['centerline_min']} vs wan21 {w['centerline_min']}")
        if k["broken_max"] > BAR["broken_max"]:
            fails.append(f"{cid}: {k['broken_max']} broken strokes in one frame (> {BAR['broken_max']})")
        if k["highlight_core_min"] is not None and k["highlight_core_min"] < BAR["highlight_min"]:
            fails.append(f"{cid}: highlight_core_min {k['highlight_core_min']}")
        if k["ground_drift"] is not None and k["ground_drift"] > BAR["ground_drift"]:
            fails.append(f"{cid}: ground drift {k['ground_drift']}")
    return {"gate": "FAIL" if fails else "PASS", "fails": fails, "bars": BAR, "gate_size": GATE_SIZE}


def contact_sheet(src, decs, dest, frame):
    from PIL import Image
    ims = [src[frame]] + [d[frame] for d in decs]
    s = src.shape[1]
    sheet = Image.new("RGB", (s * len(ims), s), "white")
    for i, im in enumerate(ims):
        sheet.paste(Image.fromarray(im), (i * s, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    clips = build_clips()
    print(f"built {len(clips)} clips ({time.time() - t0:.0f} s)", flush=True)
    import torch
    rts = {}
    for name, load, rt in (("k5", load_k5_vae, k5_roundtrip), ("wan21", load_wan21, wan_roundtrip)):
        vae = load()
        for key, u8 in clips.items():
            rts[(name,) + key] = rt(vae, u8)
            if len(rts[(name,) + key]) != len(u8):
                raise RuntimeError(f"{name} {key}: {len(u8)} frames in, {len(rts[(name,) + key])} out")
            print(f"  {name} {key[0]} {key[1]}: {len(u8)} -> {len(rts[(name,) + key])} frames", flush=True)
        del vae
        torch.cuda.empty_cache()
    res = {}
    for (size, cid), src in clips.items():
        ground = GROUNDS["mint" if "mint" in cid else "white"]
        res.setdefault(size, {})[cid] = {v: measure(src, rts[(v, size, cid)], ground) for v in ("k5", "wan21")}
        if size == GATE_SIZE:
            worst = int(len(src) // 2)
            contact_sheet(src, [rts[("k5", size, cid)], rts[("wan21", size, cid)]],
                          OUT / "sheets" / f"{cid}_{size}.png", worst)
    verdict = gate(res)
    rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "k5_repo": str(K5_REPO), "k5_vae": str(K5_VAE),
           "reference": WAN21, "clips": CLIPS, "results": {str(k): v for k, v in res.items()}, **verdict}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    keys = ("psnr", "ssim", "edge_psnr", "centerline_min", "broken_max", "highlight_core_min", "ground_drift", "motion_ratio")
    print(f"\n{'clip':22s} {'size':>4s} {'vae':6s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for size in SIZES:
        for cid, r in res[size].items():
            for v in ("k5", "wan21"):
                print(f"{cid:22s} {size:4d} {v:6s} " + " ".join(f"{str(r[v][k]):>12s}" for k in keys))
    print(f"\nG0.1 {verdict['gate']}" + "".join(f"\n  - {f}" for f in verdict["fails"]))
    print(f"-> {OUT / 'report.json'}  ({time.time() - t0:.0f} s)")
    return 0 if verdict["gate"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
