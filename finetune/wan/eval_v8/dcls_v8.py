#!/usr/bin/env python3
"""Tier-1 eval: a diffusion classifier on held-out clips. Cheap, low-variance, paired.

WHY. v7 selected checkpoints with face-SSIM between prompts, and ANALYSIS_v7 showed that
metric recovers the SEED 67-81% of the time and the LABEL at chance: it could not have
resolved a checkpoint choice. Generation-based gates are also expensive (~6.4 min/clip),
so a matrix large enough to tell checkpoints apart (~350 clips each) is out of reach.

WHAT. For a real clip x0 with conditioning frame y, the flow-matching loss of the model
under caption c is an ELBO-style proxy for -log p(x0 | c, y):

    E_c = sum_k || v_theta(x_tk, t_k, c, y) - (eps_k - x0) ||^2 ,  x_t = (1-t) x0 + t eps

With the SAME (t_k, eps_k) for every candidate caption (common random numbers), the
argmin over labels is a Bayes classifier under a uniform label prior (Li et al. 2023,
"Your Diffusion Model is Secretly a Zero-Shot Classifier"). It measures exactly the
controllability question - does the label caption explain the pixels better than every
other label - with forward passes only, and no sampler in the loop.

Reported per split (holdout = QF2_R, an angle never trained; seen = QF2_L, trained):
  acc      top-1 label accuracy (chance 1/8 expression, 1/5 motion)
  rank     mean rank of the true label (1 = best)
  margin   (min_{c != true} E_c - E_true) / E_true   - >0 means the true label wins
  loss     E_true per timestep: held-out denoising loss, the classic val loss
The noise draws depend only on (clip, k), so they are IDENTICAL across checkpoints: the
difference between two checkpoints is a paired comparison, not two independent samples.

    python dcls_v8.py --configs golden v7expr v8@100 v8@200 [--k 4] [--split holdout seen]
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOLDOUT, EVAL, DIT_LOW, T5, MUSUBI, lora_config  # noqa: E402

sys.path.insert(0, str(MUSUBI / "src"))
from musubi_tuner.wan.configs import WAN_CONFIGS  # noqa: E402
from musubi_tuner.wan.modules.model import load_wan_model  # noqa: E402
from musubi_tuner.wan.modules.t5 import T5EncoderModel  # noqa: E402

# Inside the low-noise expert's range (< 0.9), ordered most-informative first: content
# (which expression) is decided at mid-high noise, texture at low noise.
T_GRID = (0.65, 0.85, 0.45, 0.25)


def load_latents(entry):
    n = entry["frames"]
    stem = entry["clip"][:-4]
    hits = sorted((HOLDOUT / "cache" / f"f{n}").glob(f"{stem}_*_wan.safetensors"))
    if not hits:
        raise FileNotFoundError(f"no latent cache for {entry['clip']} - run cache step")
    with safe_open(str(hits[0]), "pt") as s:
        keys = list(s.keys())
        x0 = s.get_tensor(next(k for k in keys if k.startswith("latents_") and "image" not in k))
        y = s.get_tensor(next(k for k in keys if k.startswith("latents_image")))
    return x0, y


def encode_captions(texts, device):
    cfg = WAN_CONFIGS["i2v-A14B"]
    te = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
                        weight_path=str(T5), fp8=False)
    out = {}
    with torch.no_grad():
        uniq = sorted(set(texts))
        for i in range(0, len(uniq), 16):
            chunk = uniq[i:i + 16]
            for t, e in zip(chunk, te(chunk, device)):
                out[t] = e.to("cpu", torch.float16)
    del te
    gc.collect(); torch.cuda.empty_cache()
    return out


def change_mask(x0, floor=0.05):
    """Where the TRUE clip changes: |x0[f] - x0[0]| (channel mean), normalised, with frame 0
    zeroed. Frame 0 is the conditioning image - its error is label-independent noise - and
    most of the frame is background / static body that no caption can explain. The mask is
    a function of the ground truth only, never of a candidate caption, so every candidate
    is scored on the same pixels. `floor` keeps a little weight everywhere else."""
    d = (x0 - x0[:, :1]).abs().mean(0, keepdim=True)          # 1 x F x H x W
    d = d / (d.amax() + 1e-8)
    w = floor + (1 - floor) * d
    w[:, 0] = 0.0
    return w.expand_as(x0).contiguous()


def noise_for(clip, k, shape):
    seed = int(hashlib.sha256(f"{clip}|{k}".encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def load_model(name, device):
    low, _ = lora_config(name)
    for p, _m in low:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    sds = [load_file(str(p)) for p, _ in low]
    mults = [m for _, m in low]
    # Load to CPU and move once merged: loading straight to the GPU while merging LoRAs
    # peaks well above the 28.6 GB the fp16 DiT needs.
    model = load_wan_model(WAN_CONFIGS["i2v-A14B"], device, str(DIT_LOW), "sdpa", False,
                           "cpu", torch.float16, False,
                           lora_weights_list=sds, lora_multipliers=mults)
    return model.to(device).eval()


@torch.no_grad()
def score_clip(model, entry, emb, k_count, device, bs=1):
    # bs=1: Wan2.2 builds a per-token fp32 modulation tensor (~4.5 GB per sample at 24k
    # tokens), so batching candidates OOMs; the forward is compute-bound anyway.
    x0, y = load_latents(entry)
    x0 = x0.float(); y = y.to(device, torch.float16)
    C, F, H, W = x0.shape
    seq_len = F * H * W // 4                      # patch (1, 2, 2)
    labels = list(entry["candidates"])
    E = np.zeros((len(labels), k_count))
    w = change_mask(x0).to(device)
    for k in range(k_count):
        t = T_GRID[k % len(T_GRID)]
        eps = noise_for(entry["clip"], k, x0.shape)
        xt = ((1 - t) * x0 + t * eps).to(device, torch.float16)
        target = (eps - x0).to(device)
        for i in range(0, len(labels), bs):
            chunk = labels[i:i + bs]
            ctx = [emb[entry["candidates"][lab]].to(device) for lab in chunk]
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model([xt] * len(chunk),
                             t=torch.full((len(chunk),), t * 1000.0, device=device),
                             context=ctx, seq_len=seq_len, y=[y] * len(chunk))
            for j, p in enumerate(pred):
                E[i + j, k] = float((((p.float() - target) ** 2) * w).sum() / w.sum())
    tot = E.sum(1)
    ti = labels.index(entry["label"])
    order = np.argsort(tot)
    others = np.delete(tot, ti)
    return {"clip": entry["clip"], "label": entry["label"], "kind": entry["kind"],
            "char": entry["char"], "split": entry["split"],
            "pred": labels[int(order[0])], "rank": int(np.where(order == ti)[0][0]) + 1,
            "margin": float((others.min() - tot[ti]) / tot[ti]),
            "loss_true": [float(v) for v in E[ti]], "E": {l: float(v) for l, v in zip(labels, tot)}}


def summarise(rows):
    out = {}
    for split in sorted({r["split"] for r in rows}):
        for kind in ("expression", "motion"):
            rs = [r for r in rows if r["split"] == split and r["kind"] == kind]
            if not rs:
                continue
            n_lab = len(rs[0]["E"])
            out[f"{split}/{kind}"] = {
                "n": len(rs), "chance": round(1 / n_lab, 3),
                "acc": round(float(np.mean([r["pred"] == r["label"] for r in rs])), 3),
                "rank": round(float(np.mean([r["rank"] for r in rs])), 2),
                "margin": round(float(np.mean([r["margin"] for r in rs])), 4),
                "loss": round(float(np.mean([np.mean(r["loss_true"]) for r in rs])), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=2, help="(t, eps) draws per clip")
    ap.add_argument("--split", nargs="*", default=["holdout"])
    ap.add_argument("--kind", choices=("expression", "motion"), help="one kind only")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    ap.add_argument("--out", default=str(EVAL / "dcls"))
    a = ap.parse_args()
    device = torch.device("cuda")
    index = [e for e in json.loads((HOLDOUT / "index.json").read_text()) if e["split"] in a.split]
    if a.kind:
        index = [e for e in index if e["kind"] == a.kind]
    if a.limit:
        index = index[:a.limit]
    emb = encode_captions([c for e in index for c in e["candidates"].values()], device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in a.configs:
        dest = out / f"{name.replace('/', '_').replace(':', '_')}_k{a.k}.json"
        if a.kind or a.limit:
            dest = dest.with_name(dest.stem + f"_{a.kind or 'all'}_n{len(index)}.json")
        if dest.exists():
            print(f"== {name}: cached -> {dest}"); continue
        t0 = time.time()
        model = load_model(name, device)
        rows = [score_clip(model, e, emb, a.k, device) for e in index]
        res = {"config": name, "k": a.k, "t_grid": T_GRID, "summary": summarise(rows),
               "rows": rows, "seconds": round(time.time() - t0)}
        dest.write_text(json.dumps(res, indent=1))
        print(f"== {name} ({res['seconds']} s)")
        for key, v in res["summary"].items():
            print(f"   {key:<20} acc {v['acc']:.3f} (chance {v['chance']})  rank {v['rank']}  "
                  f"margin {v['margin']:+.4f}  loss {v['loss']:.5f}  n={v['n']}")
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
