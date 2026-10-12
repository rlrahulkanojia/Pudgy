#!/usr/bin/env python3
"""G0.1 - the VAE gate (PLAN section 4). STOP expv2 if it fails: no LoRA recovers what the
VAE discards.

Round-trips the G0.1 clips (CORPUS/vae_gate/png{768,512}/<id>/) through:
  h3        H3's own video VAE via DiffSynth's code path - the exact encode the trainer
            uses (MiniMaxH3Unit_InputVideoEmbedder, pipelines/minimax_h3_audio_video.py:
            325-340) and the exact decode the sampler uses (__call__, :196-198);
  wan21     Wan2.1 VAE (8x; the one Wan2.2-A14B / v1-v9 use) - same-clip reference;
  wan22_5b  Wan2.2-TI2V-5B VAE (16x) - informational: is a failure about 16x or about H3?
The Wan registry and loader are copied from training_approach/scripts/vae_roundtrip.py
(REGISTRY, load_vae; b902bde). Wan needs 4k+1 frames, so it sees the first 21 (of 22) or
37 (of 39) frames; frames used are recorded.

Metrics: `measure()` is copied verbatim from ltx25/eval/vae_gate_ltx25.py (b902bde): psnr,
ssim, edge_psnr over positive-gradient outlines + edge_cov, flat_band, shimmer_abs. Added
for H3 (PLAN G0.1):
  ground_drift   mean |dec - src| per channel on ground pixels, grey levels (bar <= 2/255)
  chunk_dip      largest step in the per-frame mean-luminance ERROR at a VAE chunk start
                 minus the largest step elsewhere (bar <= 1/255). Chunk starts are read
                 from the VAE's structure: 17-frame chunks from frame 0 -> frames 17, 34, 51
                 (minimax_h3_video_vae.py:439-443). PLAN G0.1 says "frames 5, 22 and 39";
                 those are reported too, as `chunk_dip_plan_frames`, for the record.
  flat fill      on flatfill_*: fill_mae, fill_std, flat_band, grid_ratio (FFT energy on
                 the 16-px latent lattice / all AC energy)

All comparisons are on decoded arrays and PNGs; no mp4 anywhere (MODEL_NOTES 2.11).
The eye check at 4x decides (B7): report.json carries "eye_check": "PENDING" until a person
(or the box agent, by viewing sheets/*.png) records PASS/FAIL with reasons.

    python eval/vae_gate.py --dry
    python eval/vae_gate.py                       # h3 + wan21 + wan22_5b, 768 and 512
    python eval/vae_gate.py --vaes h3 --sizes 768
    python eval/vae_gate.py --eye PASS --eye-notes "beak/eye highlights intact at 4x vs wan21"

Exit 0 = metric bars pass (eye check still pending), 2 = a metric bar FAILED -> stop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3lib as L  # noqa: E402
C = L.C

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

OUT = C.PHASE0 / "vae_gate"
SRC = C.CORPUS / "vae_gate"

# Bars (PLAN G0.1). Defaults; the eye check overrules metrics in either direction (B7).
BAR_GROUND_DRIFT = 2.0      # grey levels, every 768 clip
BAR_CHUNK_DIP = 1.0         # grey levels, every 768 clip
BAR_FILL_STD = 1.0          # grey levels of spatial std on the flat-fill clip
BAR_FILL_BAND = 0.5         # flat_band on the flat-fill clip
BAR_FILL_MAE = 2.0          # grey levels away from the fill colour
PLAN_DIP_FRAMES = (5, 22, 39)

# copied from training_approach/scripts/vae_roundtrip.py REGISTRY (b902bde)
REGISTRY = {
    "wan21": dict(cls="AutoencoderKLWan", repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                  note="Wan2.1 / Wan2.2-A14B VAE - 8x spatial"),
    "wan22_5b": dict(cls="AutoencoderKLWan", repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                     note="Wan2.2-5B VAE - 16x spatial (high compression)"),
}


# ---------------------------------------------------------------------------------------
# copied verbatim from ltx25/eval/vae_gate_ltx25.py (b902bde): sobel_mag, psnr, measure
# ---------------------------------------------------------------------------------------
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

    mags = np.stack([sobel_mag(g) for g in g_src])
    pos = mags[mags > 0]
    if pos.size == 0:
        edge_mask = np.zeros_like(mags, dtype=bool)
        flat_mask = np.ones_like(mags, dtype=bool)
    else:
        edge_mask = mags >= np.percentile(pos, 75)   # the outlines themselves
        flat_mask = mags == 0                        # genuinely flat, not "low"
    edge_cov = float(edge_mask.mean())

    def local_dev(g: np.ndarray, m: np.ndarray) -> float:
        d = np.abs(np.diff(g, axis=1))
        mm = m[:, :-1]
        return float(d[mm].mean()) if mm.sum() else 0.0

    band = max(0.0, local_dev(g_dec, flat_mask) - local_dev(g_src, flat_mask))

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
# ------------------------------------------------------------------- end of copied code


def ground_rgb(frame):
    """The flat ground is regenerated by prep, so it is the modal colour of the border."""
    border = np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]])
    vals, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    return vals[counts.argmax()]


def lum(u8):
    return 0.299 * u8[..., 0] + 0.587 * u8[..., 1] + 0.114 * u8[..., 2]


def h3_measures(src, dec, flat=False):
    f = min(len(src), len(dec))
    src, dec = src[:f].astype(np.float64), dec[:f].astype(np.float64)
    g = ground_rgb(src[0].astype(np.uint8))
    gmask = np.all(src == g[None, None, None, :], axis=-1)            # [F,H,W]
    err = np.abs(dec - src).mean(axis=-1)                              # [F,H,W]
    drift = float(err[gmask].mean()) if gmask.any() else float("nan")
    per_frame = [float(err[i][gmask[i]].mean()) if gmask[i].any() else float("nan") for i in range(f)]
    lerr = lum(dec).mean(axis=(1, 2)) - lum(src).mean(axis=(1, 2))   # per-frame luminance error
    steps = np.abs(np.diff(lerr))                                      # steps[i] = |e[i+1]-e[i]|

    def dip(starts):
        at = [steps[s - 1] for s in starts if 1 <= s < f]
        if not at:
            return None
        rest = [steps[i] for i in range(len(steps)) if (i + 1) not in starts]
        return round(float(max(at) - (max(rest) if rest else 0.0)), 3)

    out = {"ground_rgb": [int(v) for v in g], "ground_cov": round(float(gmask.mean()), 4),
           "ground_drift": round(drift, 3), "ground_drift_max_frame": round(float(np.nanmax(per_frame)), 3),
           "chunk_starts": L.chunk_starts(f), "chunk_dip": dip(L.chunk_starts(f)),
           "chunk_dip_plan_frames": dip([s for s in PLAN_DIP_FRAMES if s < f]),
           "lum_err_per_frame": [round(float(v), 3) for v in lerr]}
    if flat:
        fill = src[0, 0, 0]
        out["fill_mae"] = round(float(np.abs(dec - fill).mean()), 3)
        out["fill_std"] = round(float(np.mean([dec[i, ..., c].std() for i in range(f) for c in range(3)])), 3)
        gl = lum(dec)
        spec = np.abs(np.fft.fft2(gl - gl.mean(axis=(1, 2), keepdims=True))) ** 2
        H, W = gl.shape[1:]
        ky = np.arange(0, H, H // 16 if H >= 16 else 1)
        kx = np.arange(0, W, W // 16 if W >= 16 else 1)
        spec[:, 0, 0] = 0.0                                   # no DC
        lattice = spec[:, ky[:, None], kx[None, :]].sum()
        total = spec.sum()
        # a perfectly flat decode has no AC energy at all: no grid, ratio 0
        out["grid_ratio"] = round(float(lattice / total), 4) if total / gl.size > 1e-6 else 0.0
    return out


# ---------------------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------------------

def h3_roundtrip(pipe, frames):
    import torch
    with torch.no_grad():
        pipe.load_models_to_device(["video_vae"])
        x = pipe.preprocess_video(frames, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        z = pipe.video_vae.encode_video(x, dtype=pipe.torch_dtype).to(pipe.torch_dtype)
        rec = pipe.video_vae.decode_video(z, dtype=pipe.torch_dtype, tiled=True, tile_size=256, tile_overlap=64)
        out = pipe.vae_output_to_video(rec, min_value=0, max_value=1)
    return L.frames_to_u8(out), {"latent_shape": list(z.shape)}


def load_wan(key):
    """copied from training_approach/scripts/vae_roundtrip.py load_vae (b902bde)."""
    import importlib
    import torch
    spec = REGISTRY[key]
    diffusers = importlib.import_module("diffusers")
    cls = getattr(diffusers, spec["cls"])
    print(f"[load] {key}: {spec['cls']} from {spec['repo']}  ({spec['note']})")
    vae = cls.from_pretrained(spec["repo"], subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to("cuda").eval()
    # diffusers 0.35's tiled_encode skips patchify, so a patch_size VAE (Wan2.2-5B) crashes
    # tiled ("expected 12 channels, got 3"). Untiled fits on 80 GB at 768^2.
    tiling = getattr(vae.config, "patch_size", None) in (None, 1)
    for m in ("enable_slicing",) + (("enable_tiling",) if tiling else ()):
        if hasattr(vae, m):
            getattr(vae, m)()
    return vae


def wan_roundtrip(vae, u8):
    import torch
    T = ((len(u8) - 1) // 4) * 4 + 1
    x = torch.from_numpy(u8[:T].copy()).float().permute(3, 0, 1, 2)[None] / 127.5 - 1.0
    with torch.no_grad():
        lat = vae.encode(x.to("cuda", torch.float32)).latent_dist.mode()   # deterministic
        rec = vae.decode(lat).sample.clamp(-1, 1)[0].float().cpu()
    rec = ((rec.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
    return rec, {"frames_used": T}


# ---------------------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------------------
REGIONS = {  # fractions of the character bbox (y0, y1, x0, x1); a visual aid, not a metric
    "eyes": (0.18, 0.40, 0.25, 0.75),
    "beak": (0.33, 0.55, 0.35, 0.65),
    "blush": (0.33, 0.55, 0.12, 0.42),
    "flipper": (0.45, 0.78, 0.00, 0.30),
}
CROP = 96
ZOOM = 4


def char_bbox(frame):
    g = ground_rgb(frame)
    m = np.any(np.abs(frame.astype(int) - g[None, None, :].astype(int)) > 30, axis=-1)
    if not m.any():
        H, W = frame.shape[:2]
        return 0, H, 0, W
    ys, xs = np.where(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def crop_at(frame, cy, cx):
    H, W = frame.shape[:2]
    y0 = int(np.clip(cy - CROP // 2, 0, H - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
    c = frame[y0:y0 + CROP, x0:x0 + CROP]
    return np.repeat(np.repeat(c, ZOOM, axis=0), ZOOM, axis=1)


def contact_sheet(src, decoded, out_png):
    """Rows: (frame, region). Columns: source | each VAE | |src - h3| x5."""
    from PIL import Image, ImageDraw
    f = len(src)
    frames = sorted({f // 2} | ({17} if f > 17 else set()))
    cols = ["source"] + list(decoded) + (["|src-h3|x5"] if "h3" in decoded else [])
    tiles = []
    labels = []
    y0, y1, x0, x1 = char_bbox(src[0])
    for fi in frames:
        for rname, (a, b, c, d) in REGIONS.items():
            cy = int(y0 + (a + b) / 2 * (y1 - y0))
            cx = int(x0 + (c + d) / 2 * (x1 - x0))
            row = [crop_at(src[fi], cy, cx)]
            for k in decoded:
                dk = decoded[k]
                row.append(crop_at(dk[min(fi, len(dk) - 1)], cy, cx) if dk is not None else
                           np.zeros((CROP * ZOOM, CROP * ZOOM, 3), np.uint8))
            if "h3" in decoded:
                diff = np.clip(np.abs(src[fi].astype(int) - decoded["h3"][fi].astype(int)) * 5, 0, 255).astype(np.uint8)
                row.append(crop_at(diff, cy, cx))
            tiles.append(np.concatenate(row, axis=1))
            labels.append(f"f{fi} {rname}")
    pad = 22
    sheet = np.full((pad + len(tiles) * (CROP * ZOOM + pad), tiles[0].shape[1], 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        y = pad + i * (CROP * ZOOM + pad)
        sheet[y:y + t.shape[0]] = t
    im = Image.fromarray(sheet)
    dr = ImageDraw.Draw(im)
    for j, c in enumerate(cols):
        dr.text((j * CROP * ZOOM + 6, 4), c, fill=(0, 0, 0))
    for i, lab in enumerate(labels):
        dr.text((6, pad + i * (CROP * ZOOM + pad) - 16), lab, fill=(200, 0, 0))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)


# ---------------------------------------------------------------------------------------

def clips_for(size, only=None):
    d = SRC / f"png{size}"
    if not d.is_dir():
        return []
    out = [p for p in sorted(d.iterdir()) if p.is_dir() and L.png_files(p)]
    if only:
        out = [p for p in out if any(k in p.name for k in only)]
    return out


def npz_path(vae, size, clip):
    return OUT / "decoded" / vae / str(size) / f"{clip}.npz"


def verdict(results, sizes):
    """Metric bars on H3 at 768 (the training size); 512 is for the record."""
    bars = {"ground_drift": [], "chunk_dip": [], "flat_fill": []}
    h3 = results.get("h3", {}).get("768", {})
    for clip, m in h3.items():
        x = m.get("h3", {})
        if clip.startswith("flatfill"):
            ok = (x.get("fill_std", 99) <= BAR_FILL_STD and x.get("fill_mae", 99) <= BAR_FILL_MAE
                  and m.get("flat_band", 99) <= BAR_FILL_BAND)
            bars["flat_fill"].append({"clip": clip, "pass": bool(ok), "fill_std": x.get("fill_std"),
                                      "fill_mae": x.get("fill_mae"), "flat_band": m.get("flat_band"),
                                      "grid_ratio": x.get("grid_ratio")})
            continue
        bars["ground_drift"].append({"clip": clip, "value": x.get("ground_drift"),
                                     "pass": x.get("ground_drift") is not None and x["ground_drift"] <= BAR_GROUND_DRIFT})
        dip = x.get("chunk_dip")
        bars["chunk_dip"].append({"clip": clip, "value": dip, "pass": dip is None or dip <= BAR_CHUNK_DIP})
    summary = {k: (all(r["pass"] for r in v) if v else None) for k, v in bars.items()}
    metric_pass = bool(h3) and all(v is not False for v in summary.values()) and summary["ground_drift"] is not None
    return {"bars": bars, "summary": summary, "metric_pass": metric_pass,
            "thresholds": {"ground_drift": BAR_GROUND_DRIFT, "chunk_dip": BAR_CHUNK_DIP,
                           "fill_std": BAR_FILL_STD, "fill_mae": BAR_FILL_MAE, "fill_band": BAR_FILL_BAND}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vaes", default="h3,wan21,wan22_5b")
    ap.add_argument("--sizes", default="768,512")
    ap.add_argument("--clips", default="", help="substring filter, comma-separated")
    ap.add_argument("--force", action="store_true", help="redo round trips already on disk")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--eye", choices=["PASS", "FAIL"], help="record the 4x eye check and exit")
    ap.add_argument("--eye-notes", default="")
    args = ap.parse_args()

    report_path = OUT / "report.json"
    if args.eye:
        rep = C.load_json(report_path, {}) or {}
        rep["eye_check"] = {"result": args.eye, "notes": args.eye_notes,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rep["gate"] = "PASS" if (args.eye == "PASS" and rep.get("verdict", {}).get("metric_pass")) else "FAIL"
        C.write_json(report_path, rep)
        print(f"eye check {args.eye} recorded -> gate {rep['gate']} ({report_path})")
        return 0 if rep["gate"] == "PASS" else 2

    vaes = [v for v in args.vaes.split(",") if v]
    sizes = [int(s) for s in args.sizes.split(",") if s]
    only = [s for s in args.clips.split(",") if s]
    plan = {s: clips_for(s, only) for s in sizes}

    print(f"G0.1 VAE gate  src={SRC}  out={OUT}")
    problems = []
    for s in sizes:
        print(f"  {s}px: {len(plan[s])} clips")
        for p in plan[s]:
            files = L.png_files(p)
            n = len(files)
            msg = f"    {p.name:40s} {n:3d} frames"
            if n % 17 != 5:
                problems.append(f"{p.name}@{s}: {n} frames, not on the 17n+5 grid")
            if args.dry:
                from PIL import Image
                w, h = Image.open(files[0]).size
                msg += f"  {w}x{h}"
                if (w, h) != (s, s):
                    problems.append(f"{p.name}@{s}: frame is {w}x{h}")
                if s == 768:
                    msg += f"  wan frames used: {((n - 1) // 4) * 4 + 1}  chunk starts: {L.chunk_starts(n)}"
            print(msg)
        if not plan[s]:
            problems.append(f"no clips under {SRC / f'png{s}'}")
    if not any("flatfill" in p.name for s in sizes for p in plan[s]):
        problems.append("no flatfill_* clip (bar 4 cannot be measured)")
    for v in vaes:
        if v != "h3" and v not in REGISTRY:
            problems.append(f"unknown VAE {v!r}")
    if args.dry:
        print("\n".join(["DRY: problems:"] + ["  - " + p for p in problems]) if problems else "DRY: inputs OK")
        print(f"DRY: would write {report_path}, {OUT / 'sheets'}/*.png, {OUT / 'decoded'}/")
        return 1 if problems else 0
    if problems:
        for p in problems:
            print("  !!", p)
        if not any(plan.values()):
            return 1

    srcs = {(s, p.name): L.frames_to_u8(L.read_frames(p)) for s in sizes for p in plan[s]}
    rep = C.load_json(report_path, {}) or {}
    rep.update(L.report_header("G0.1 VAE gate"))
    results = rep.get("results", {})
    status = rep.get("vae_status", {})

    for v in vaes:
        todo = [(s, p.name) for s in sizes for p in plan[s] if args.force or not npz_path(v, s, p.name).exists()]
        if todo:
            t0 = time.time()
            pipe = model = rt = None
            try:
                if v == "h3":
                    from PIL import Image
                    pipe = L.load_pipe(("vae",))
                    rt = lambda u8: h3_roundtrip(pipe, [Image.fromarray(f) for f in u8])  # noqa: E731
                else:
                    model = load_wan(v)
                    rt = lambda u8: wan_roundtrip(model, u8)  # noqa: E731
            except Exception as e:
                status[v] = f"unavailable: {type(e).__name__}: {e}"
                print(f"  !! {v} could not be loaded: {e}")
                continue
            for (s, clip) in todo:
                dec, info = rt(srcs[(s, clip)])
                if v == "h3" and len(dec) != len(srcs[(s, clip)]):
                    raise RuntimeError(f"H3 round trip returned {len(dec)} frames for {len(srcs[(s, clip)])}")
                p = npz_path(v, s, clip)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, dec=dec, info=np.array([L.dumps(info)]))
                if v == "h3" and s == 768:
                    from PIL import Image
                    dd = OUT / "decoded" / "h3_png768" / clip
                    dd.mkdir(parents=True, exist_ok=True)
                    for i, fr in enumerate(dec):
                        Image.fromarray(fr).save(dd / f"{i:04d}.png")
                print(f"  {v} {s} {clip}: {len(dec)} frames")
            status[v] = f"ok ({time.time() - t0:.0f}s)"
            pipe = model = rt = None   # drop every reference before the next VAE loads
            L.free()
        else:
            status.setdefault(v, "ok (cached)")

    # metrics, from the saved arrays
    for v in vaes:
        for s in sizes:
            for p in plan[s]:
                f = npz_path(v, s, p.name)
                if not f.exists():
                    continue
                z = np.load(f)
                dec = z["dec"]
                src = srcs[(s, p.name)][:len(dec)]
                m = measure(src, dec)
                m["info"] = json.loads(str(z["info"][0])) if "info" in z else {}
                m["h3"] = h3_measures(src, dec, flat=p.name.startswith("flatfill"))
                results.setdefault(v, {}).setdefault(str(s), {})[p.name] = m

    # contact sheets
    for s in sizes:
        for p in plan[s]:
            decs = {}
            for v in vaes:
                f = npz_path(v, s, p.name)
                decs[v] = np.load(f)["dec"] if f.exists() else None
            if any(d is not None for d in decs.values()):
                contact_sheet(srcs[(s, p.name)], decs, OUT / "sheets" / f"{p.name}_{s}.png")

    means = {}
    for v, by_size in results.items():
        for s, by_clip in by_size.items():
            keys = ("psnr", "ssim", "edge_psnr", "edge_cov", "flat_band", "shimmer_abs")
            # NaN = metric undefined for that clip (edge_psnr on the flat-fill clip has no edges)
            ok = lambda x: x is not None and x == x  # noqa: E731
            means.setdefault(v, {})[s] = {k: (round(float(np.mean([m[k] for m in by_clip.values() if ok(m.get(k))])), 4)
                                              if any(ok(m.get(k)) for m in by_clip.values()) else None) for k in keys}
            means[v][s]["ground_drift"] = round(float(np.nanmean([m["h3"]["ground_drift"] for c, m in by_clip.items()
                                                                  if not c.startswith("flatfill")] or [np.nan])), 3)
    vd = verdict(results, sizes)
    h3_768 = means.get("h3", {}).get("768", {})
    rep.update({"results": results, "means": means, "vae_status": status, "verdict": vd,
                "h3_mean_psnr_768": h3_768.get("psnr"),   # V9 uses (this - 3 dB)
                "eye_check": rep.get("eye_check", "PENDING"),
                "eye_check_howto": ("View sheets/*_768.png: beak outline, eye highlights, flipper outline, blush "
                                    "shapes at 4x, H3 vs wan21. Record with --eye PASS|FAIL --eye-notes '...'"),
                "gate": "PENDING_EYE" if vd["metric_pass"] else "FAIL"})
    C.write_json(report_path, rep)

    print("\n" + "=" * 86)
    print(f"{'vae':10s} {'size':>5s} {'psnr':>7s} {'ssim':>7s} {'edge_psnr':>10s} {'edge_cov':>9s} {'flat_band':>10s} {'shimmer':>8s} {'drift':>6s}")
    for v in means:
        for s, m in means[v].items():
            print(f"{v:10s} {s:>5s} {m['psnr'] or 0:7.2f} {str(m['ssim']):>7s} {m['edge_psnr'] or 0:10.2f} "
                  f"{m['edge_cov'] or 0:9.3f} {m['flat_band'] or 0:10.3f} {str(m['shimmer_abs']):>8s} {m['ground_drift']:6.2f}")
    print(f"\nH3 @768 metric bars: {vd['summary']}  ->  {'metric PASS' if vd['metric_pass'] else 'FAIL'}")
    print(f"eye check: {rep['eye_check']}   sheets: {OUT / 'sheets'}")
    print("=" * 86)
    return 0 if vd["metric_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
