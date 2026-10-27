#!/usr/bin/env python3
"""Phase 0 #4 (Training_Approach_v9.md section 5.1): does the Wan VAE keep Pax and Polly
intact at scene scale? Decides D2: train at 480x848 unless the faces break, then 576x1024.

For kept skit shots (scenes/audit/shot_review.json, a fixed random sample with the wide
shots over-represented, since they put the leads smallest) and, as the reference, v8
training clips at 1024^2 (what the model already reproduces):
  * take 17 frames, resize to each resolution exactly as prep does, VAE encode + decode;
  * PSNR between the resized input and its reconstruction, (a) full frame and (b) inside
    the leads' bounding box (Pax / Polly body colour, as longform/dupcheck.py finds it);
  * a sheet of lead crops per shot: input vs reconstruction at each resolution, enlarged
    to the same size, for the by-eye check of eyes, beak and outline.

Runs on the CPU by default, so it never competes with GPU jobs (~1-2 min per clip).

    /workspace/Pudgy/.venv-wan/bin/python finetune/wan/eval_v9b/phase0_vae_v9b.py --n 12
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
WAN = HERE.parent
sys.path.insert(0, str(WAN))
sys.path.insert(0, "/workspace/musubi-tuner/src")
from musubi_tuner.wan.modules.vae import WanVAE  # noqa: E402

VAE = "/workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors"
RAW = Path("/workspace/Data/raw")
OUT = Path("/workspace/eval_v9b/phase0_vae")
RES = {"480x848": (480, 848, 854), "576x1024": (576, 1024, 1024)}   # w, h, scaled h before crop
PAX = ((100, 90, 120), (125, 255, 255))
POLLY = ((150, 40, 150), (178, 255, 255))
NF = 17


def decode(path, start, w, h, scaled_h, square=False):
    vf = f"scale={w}:{h}:flags=lanczos" if square else \
        f"scale={w}:{scaled_h}:flags=lanczos,crop={w}:{h}:0:{(scaled_h - h) // 2}"
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path), "-vf",
                        f"select=gte(n\\,{start}),{vf}", "-frames:v", str(NF), "-vsync", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, h, w, 3)[:NF]


def roundtrip(vae, frames):
    x = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float() / 127.5 - 1.0     # C F H W
    with torch.no_grad():
        z = vae.encode([x.to(vae.dtype)])
        y = vae.decode(z)[0]
    return ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()


def lead_box(frame):
    """Box around the largest character-sized blob of a lead's body colour. Blue skies and
    teal props share Pax's hue, so blobs over 20% of the frame are ignored, and only the
    single largest remaining blob (one lead) is boxed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    best = None
    for rng in (PAX, POLLY):
        m = cv2.inRange(hsv, *rng)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 200 or area > 0.2 * m.size:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    pad = max(6, int(0.15 * max(w, h)))
    return (max(0, x - pad), max(0, y - pad), min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def measure(vae, frames):
    rec = roundtrip(vae, frames)
    full = psnr(frames, rec)
    boxes = [lead_box(f) for f in frames]
    vals = [psnr(f[y0:y1, x0:x1], r[y0:y1, x0:x1]) for f, r, bx in zip(frames, rec, boxes) if bx
            for (x0, y0, x1, y1) in [bx]]
    return rec, full, (float(np.mean(vals)) if vals else None), boxes[NF // 2]


def crop_pair(frames, rec, box, size=220):
    i = NF // 2
    if box is None:
        return None
    x0, y0, x1, y1 = box
    tiles = []
    for im in (frames[i], rec[i]):
        c = Image.fromarray(im[y0:y1, x0:x1])
        s = size / max(c.size)
        tiles.append(c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.NEAREST))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="skit shots")
    ap.add_argument("--ref", type=int, default=4, help="v8 1024^2 reference clips")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    vae = WanVAE(vae_path=VAE, device=a.device, dtype=torch.float32 if a.device == "cpu" else torch.bfloat16)

    shots = [s for s in json.loads((WAN / "scenes/audit/shot_review.json").read_text())
             if s["verdict"] == "keep" and s["end"] - s["start"] >= NF + 4]
    rng = random.Random(9)
    rng.shuffle(shots)
    sample = shots[:a.n]
    rows, sheet_rows = [], []
    for s in sample:
        path = RAW / s["file"]
        st = s["start"] + (s["end"] - s["start"] - NF) // 2
        row = {"shot": f"{s['file']}#{s['shot']}", "start": st}
        tiles = []
        for name, (w, h, sh) in RES.items():
            fr = decode(path, st, w, h, sh)
            rec, full, lead, box = measure(vae, fr)
            row[name] = {"psnr_full": round(full, 2), "psnr_lead": None if lead is None else round(lead, 2),
                         "lead_box_h_px": None if box is None else int(box[3] - box[1])}
            tp = crop_pair(fr, rec, box)
            if tp:
                tiles += tp
        rows.append(row); sheet_rows.append((row["shot"], tiles))
        print(json.dumps(row), flush=True)

    # Reference: v8 training clips at 1024^2 (flat grounds), the scale the model already knows.
    ref = sorted(Path("/workspace/data_v9c/clips").glob("*__white_f41.mp4"))
    rng.shuffle(ref)
    refs = []
    for f in ref[:a.ref]:
        fr = decode(f, 0, 1024, 1024, 1024, square=True)
        rec, full, lead, box = measure(vae, fr)
        r = {"clip": f.name, "psnr_full": round(full, 2), "psnr_lead": None if lead is None else round(lead, 2),
             "lead_box_h_px": None if box is None else int(box[3] - box[1])}
        refs.append(r); print(json.dumps(r), flush=True)
        tp = crop_pair(fr, rec, box)
        sheet_rows.append((f.name, tp or []))

    summ = {}
    for name in RES:
        v = [r[name]["psnr_lead"] for r in rows if r[name]["psnr_lead"] is not None]
        hpx = [r[name]["lead_box_h_px"] for r in rows if r[name]["lead_box_h_px"]]
        summ[name] = {"psnr_lead_mean": round(float(np.mean(v)), 2), "psnr_lead_min": round(float(np.min(v)), 2),
                      "lead_h_px_median": int(np.median(hpx))}
    v = [r["psnr_lead"] for r in refs if r["psnr_lead"] is not None]
    summ["v8_1024_reference"] = {"psnr_lead_mean": round(float(np.mean(v)), 2), "psnr_lead_min": round(float(np.min(v)), 2)}
    (OUT / "vae_check.json").write_text(json.dumps({"summary": summ, "shots": rows, "reference": refs}, indent=1))
    print("SUMMARY", json.dumps(summ, indent=1))

    # Sheet: one row per clip: [480 in, 480 rec, 576 in, 576 rec] (skits) or [1024 in, rec] (v8).
    tw, th = 220, 240
    img = Image.new("RGB", (tw * 4, th * len(sheet_rows)), "white")
    d = ImageDraw.Draw(img)
    for i, (name, tiles) in enumerate(sheet_rows):
        for j, t in enumerate(tiles[:4]):
            img.paste(t, (j * tw, i * th + 16))
        d.text((2, i * th), name[-60:], fill="black")
    img.save(OUT / "vae_check_sheet.png")
    print(f"sheet: {OUT / 'vae_check_sheet.png'}  (columns: 480 input, 480 recon, 576 input, 576 recon)")


if __name__ == "__main__":
    main()
