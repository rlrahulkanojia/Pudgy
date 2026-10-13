#!/usr/bin/env python3
"""Tier 1 (PLAN §7) and G0.3 (PLAN §4): a diffusion classifier on held-out real clips, for
Kandinsky 5.0 Pro I2V through musubi's own training forward.

For a real clip x0 (start frame y), caption c and draw k (t_k, eps_k):
    E_c = sum_k  sum_{frames >= 1} w * || v(x_tk, t_k, c, y) - (eps_k - x0) ||^2 / sum w,
    x_t = (1 - t) x0 + t eps,  t in T_GRID = (0.65, 0.85),  k = 2
w is the change mask of the TRUE clip (where it moves relative to frame 0; frame 0 weighted
0, a 0.05 floor elsewhere), so every candidate caption is scored on the same pixels. The
noise depends only on (clip, k), so it is identical across candidates, layouts and
checkpoints: every comparison is paired. argmin_c E_c is a Bayes classifier under a uniform
label prior (Li et al. 2023). The definitions, the mask and the reported numbers are v8's
(finetune/wan/eval_v8/dcls_v8.py @ 4832f98, copied), so per-clip rows have the same schema
and can be compared clip by clip with v8's Tier-1 file (compare_v8.py).

The forward is musubi's Kandinsky5NetworkTrainer.call_dit (patched), i.e. exactly the
training forward, with the first-frame layout chosen per call:
  A      official Kandinsky inference: clean first-frame latent in the noisy input at
         frame 0, visual_cond zero, mask at frame 0. G0.3 (2026-10-13): the checkpoint
         expects A (lowest loss on 26/26 held-out clips), so expv1 trains and generates in A
         (generator patched: --i2v_layout A) and Tier 1 scores in A — the default here
  A1     musubi's stock generator: A plus visual_cond[0] (15.6% higher loss than A)
  B      visual_cond[0] only
  stock  musubi trainer as shipped (first+last frame in visual_cond, mask on every frame) = B'
The latents come from the expv1 cacher (VAE directory, --first_frame_image_encode), which
build_refs.py's tier1.toml sets up.

    python eval/dcls.py score --configs base                         # Tier 1, layout A
    python eval/dcls.py score --configs base --layouts A A1 B stock --true-only   # G0.3
    python eval/dcls.py score --configs expv1@300 expv1@550 --split holdout seen
    python eval/dcls.py g03                                          # G0.3 report
  configs: base | expv1@<step> (WORK/output/expv1-k5pro-a-step<8d>.safetensors) | file:<lora>
"""
import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv1_common as C  # noqa: E402
import stats as S  # noqa: E402

TIER1 = C.REFS / "tier1"
OUT = C.EVAL / "dcls"
LORA_RUN = C.WORK / "output"


def change_mask(x0, floor=0.05):
    """v8's mask: |x0[f] - x0[0]| (channel mean), normalised, frame 0 zeroed. x0: C,F,H,W."""
    d = (x0 - x0[:, :1]).abs().mean(0, keepdim=True)
    d = d / (d.amax() + 1e-8)
    w = floor + (1 - floor) * d
    w[:, 0] = 0.0
    return w.expand_as(x0).contiguous()


def noise_for(clip, k, shape):
    import torch
    seed = int(hashlib.sha256(f"{clip}|{k}".encode()).hexdigest()[:8], 16)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def load_latents(entry):
    from safetensors.torch import load_file
    n = entry["frames"]
    stem = entry["clip"][:-4]
    hits = sorted((TIER1 / f"cache_f{n}").glob(f"{stem}_*_k5.safetensors"))
    if len(hits) != 1:
        raise FileNotFoundError(f"latent cache for {entry['clip']}: {hits} (run the tier1.toml cache step)")
    sd = load_file(str(hits[0]))
    x0 = next(v for k, v in sd.items() if k.startswith("latents_") and "image" not in k)
    img = next(v for k, v in sd.items() if k.startswith("latents_image"))
    if x0.shape[1] != C.latent_frames(n):
        raise RuntimeError(f"{entry['clip']}: {x0.shape[1]} latent frames for f{n}")
    return x0.float(), img.float()


def encode_texts(texts, device):
    """Each caption encoded ALONE with the video template, sliced exactly as musubi's text
    cacher does (kandinsky5_cache_text_encoder_outputs.encode_and_save_batch). Cached."""
    import torch
    key = hashlib.sha256("\n\0".join(sorted(texts)).encode()).hexdigest()[:16]
    cache = TIER1 / f"text_{key}.pt"
    if cache.exists():
        return torch.load(cache)
    sys.path.insert(0, str(C.MUSUBI / "src"))
    from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
    conf = SimpleNamespace(qwen=SimpleNamespace(checkpoint_path=str(C.QWEN), max_length=512),
                           clip=SimpleNamespace(checkpoint_path=str(C.CLIP), max_length=77))
    te = get_text_embedder(conf, device=device, quantized_qwen=False)
    out = {}
    with torch.no_grad():
        for t in sorted(set(texts)):
            emb, cu, am = te.encode([t], type_of_content="video")
            e = emb["text_embeds"].cpu()
            if cu is not None and e.dim() == 2:
                e = e[int(cu[0]):int(cu[1])]
            m = am[0].bool().flatten().cpu()
            if m.numel() != e.shape[0]:
                m = m[m] if m.sum().item() == e.shape[0] else torch.ones(e.shape[0], dtype=torch.bool)
            out[t] = (e, emb["pooled_embed"][0].cpu(), m)
    del te
    gc.collect(); torch.cuda.empty_cache()
    torch.save(out, cache)
    return out


class Scorer:
    """musubi's trainer, loaded once; call_dit is the training forward."""

    def __init__(self, device):
        import torch
        sys.path.insert(0, str(C.MUSUBI / "src"))
        from musubi_tuner.hv_train_network import setup_parser_common
        from musubi_tuner.kandinsky5_train_network import Kandinsky5NetworkTrainer, kandinsky5_setup_parser
        self.torch = torch
        self.device = device
        p = kandinsky5_setup_parser(setup_parser_common())
        self.args = p.parse_args(["--task", C.TASK, "--dit", str(C.DIT), "--vae", str(C.VAE_DIR),
                                  "--mixed_precision", "bf16", "--sdpa", "--dataset_config", "unused.toml",
                                  "--i2v_layout", "A"])
        self.tr = Kandinsky5NetworkTrainer()
        self.tr.handle_model_specific_args(self.args)
        self.acc = SimpleNamespace(device=device,
                                   autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        self.base = None
        self.dit = None
        self.loaded = None

    def load(self, config):
        torch = self.torch
        if self.loaded == config:
            return
        self.dit = None
        gc.collect(); torch.cuda.empty_cache()
        dit = self.tr.load_transformer(self.acc, self.args, str(C.DIT), "sdpa", False, "cpu", torch.bfloat16)
        if config != "base":
            from safetensors.torch import load_file
            from musubi_tuner.networks import lora_kandinsky
            path = lora_path(config)
            sd = load_file(str(path))
            net = lora_kandinsky.create_arch_network_from_weights(1.0, sd, unet=dit, for_inference=True)
            net.merge_to(None, dit, sd, device=torch.device("cpu"), non_blocking=False)
        self.dit = dit.to(self.device).eval()
        self.dit.requires_grad_(False)
        self.loaded = config

    def loss(self, x0, img, eps, t, text, layout, w):
        torch = self.torch
        self.tr._i2v_layout = layout
        e, pe, m = text
        xt = (1 - t) * x0 + t * eps
        batch = {"text_embeds": [e], "pooled_embed": [pe], "attention_mask": [m], "latents_image": [img]}
        with torch.no_grad():
            out = self.tr.call_dit(self.args, self.acc, self.dit, x0[None], batch, eps[None], xt[None],
                                   torch.tensor([t * 1000.0]), torch.bfloat16)
        pred = out.pred[0].float()                       # F', C, H, W
        F = x0.shape[1]
        if pred.shape[0] == F:                           # stock keeps frame 0; drop it here
            pred = pred[1:]
        tgt = (eps - x0).permute(1, 0, 2, 3)[1:].to(pred.device)
        ww = w.permute(1, 0, 2, 3)[1:].to(pred.device)
        return float((((pred - tgt) ** 2) * ww).sum() / ww.sum())


def lora_path(config):
    if config.startswith("file:"):
        return Path(config[5:])
    if config.startswith("expv1@"):
        step = config[6:]
        return LORA_RUN / f"expv1-k5pro-a-step{int(step):08d}.safetensors"
    raise ValueError(f"unknown config {config!r}")


def score_clip(sc, entry, texts, layouts, k_count, true_only=False):
    x0, img = load_latents(entry)
    w = change_mask(x0)
    x0d, imgd = x0.to(sc.device), img.to(sc.device)
    labels = [entry["label"]] if true_only else list(entry["candidates"])
    E = {lay: np.zeros((len(labels), k_count)) for lay in layouts}
    for k in range(k_count):
        t = C.T_GRID[k % len(C.T_GRID)]
        eps = noise_for(entry["clip"], k, tuple(x0.shape)).to(sc.device)
        for i, lab in enumerate(labels):
            tx = tuple(v.to(sc.device) for v in texts[entry["candidates"][lab]])
            for lay in layouts:
                E[lay][i, k] = sc.loss(x0d, imgd, eps, t, tx, lay, w)
    rows = {}
    ti = labels.index(entry["label"])
    if true_only:                     # G0.3 needs only the true caption's loss
        return {lay: {"clip": entry["clip"], "label": entry["label"], "kind": entry["kind"],
                      "char": entry["char"], "split": entry["split"], "source": entry["source"],
                      "loss_true": [float(v) for v in E[lay][0]]} for lay in layouts}
    for lay in layouts:
        tot = E[lay].sum(1)
        order = np.argsort(tot)
        others = np.delete(tot, ti)
        rows[lay] = {"clip": entry["clip"], "label": entry["label"], "kind": entry["kind"],
                     "char": entry["char"], "split": entry["split"], "source": entry["source"],
                     "pred": labels[int(order[0])], "rank": int(np.where(order == ti)[0][0]) + 1,
                     "margin": float((others.min() - tot[ti]) / tot[ti]),
                     "loss_true": [float(v) for v in E[lay][ti]],
                     "E": {lab: float(v) for lab, v in zip(labels, tot)}}
    return rows


def summarise(rows):
    out = {}
    for split in sorted({r["split"] for r in rows}):
        for kind in ("expression", "motion"):
            rs = [r for r in rows if r["split"] == split and r["kind"] == kind]
            if not rs:
                continue
            out[f"{split}/{kind}"] = {
                "n": len(rs), "chance": round(1 / len(rs[0]["E"]), 3),
                "acc": round(float(np.mean([r["pred"] == r["label"] for r in rs])), 3),
                "rank": round(float(np.mean([r["rank"] for r in rs])), 2),
                "margin": round(float(np.mean([r["margin"] for r in rs])), 4),
                "loss": round(float(np.mean([np.mean(r["loss_true"]) for r in rs])), 5)}
    return out


def dest_for(config, layout, k, kind=None, n=None, true_only=False):
    name = f"{config.replace('/', '_').replace(':', '_')}__{layout}_k{k}" + ("_true" if true_only else "")
    if kind or n:
        name += f"_{kind or 'all'}_n{n}"
    return OUT / f"{name}.json"


def cmd_score(a):
    import torch
    device = torch.device("cuda")
    index = [e for e in C.load_json(TIER1 / "index.json", []) if e["split"] in a.split]
    if a.kind:
        index = [e for e in index if e["kind"] == a.kind]
    if a.limit:
        index = index[:a.limit]
    if not index:
        sys.exit(f"no Tier-1 clips in {TIER1} (run build_refs.py and the cache step)")
    texts = encode_texts([c for e in index for c in e["candidates"].values()], device)
    sc = Scorer(device)
    for config in a.configs:
        todo = [lay for lay in a.layouts
                if not dest_for(config, lay, a.k, a.kind, a.limit and len(index), a.true_only).exists()]
        if not todo:
            print(f"== {config}: cached"); continue
        t0 = time.time()
        sc.load(config)
        per = {lay: [] for lay in todo}
        for i, e in enumerate(index, 1):
            for lay, row in score_clip(sc, e, texts, todo, a.k, a.true_only).items():
                per[lay].append(row)
            print(f"  [{i}/{len(index)}] {e['clip']}  {time.time() - t0:.0f}s", flush=True)
        for lay in todo:
            res = {"config": config, "layout": lay, "k": a.k, "t_grid": list(C.T_GRID),
                   "lengths": C.TIER1_LEN, "true_only": a.true_only,
                   "summary": None if a.true_only else summarise(per[lay]), "rows": per[lay],
                   "seconds": round(time.time() - t0)}
            dest = dest_for(config, lay, a.k, a.kind, a.limit and len(index), a.true_only)
            C.write_json(dest, res)
            print(f"== {config} [{lay}] -> {dest}")
            for key, v in (res["summary"] or {}).items():
                print(f"   {key:<20} acc {v['acc']:.3f} (chance {v['chance']})  rank {v['rank']}  "
                      f"margin {v['margin']:+.4f}  loss {v['loss']:.5f}  n={v['n']}")


def cmd_g03(a):
    """PLAN §4 G0.3: the layout with the lower paired held-out loss is the one the checkpoint
    was trained on. Paired per clip (true caption, same noise), frames >= 1, change-masked."""
    res = {lay: (C.load_json(dest_for("base", lay, a.k, true_only=True)) or C.load_json(dest_for("base", lay, a.k)))
           for lay in ("A", "A1", "B", "stock")}
    res = {k: v for k, v in res.items() if v}
    if len(res) < 2:
        sys.exit("need base scores for >= 2 layouts: dcls.py score --configs base --layouts A A1 B stock")
    rows = {lay: {r["clip"]: r for r in v["rows"] if r["split"] == "holdout"} for lay, v in res.items()}
    clips = sorted(set.intersection(*[set(r) for r in rows.values()]))
    loss = {lay: np.array([np.mean(rows[lay][c]["loss_true"]) for c in clips]) for lay in rows}
    mean = {lay: float(v.mean()) for lay, v in loss.items()}
    ranked = sorted(mean, key=mean.get)
    best = ranked[0]
    pairs = {}
    for lay in ranked[1:]:
        d = loss[lay] - loss[best]                       # > 0: `best` explains the clip better
        pairs[f"{lay}-{best}"] = {"mean_diff": float(d.mean()), "rel_diff": float(d.mean() / loss[best].mean()),
                                  "best_lower_on": f"{int((d > 0).sum())}/{len(d)}",
                                  "sign_test_p": S.sign_test(int((d > 0).sum()), int((d < 0).sum())),
                                  "wilcoxon_p": S.wilcoxon_p(d),
                                  "ci95_mean_diff": S.bootstrap_ci(d)}
    a1_vs_a = None
    if "A" in loss and "A1" in loss:
        d = loss["A"] - loss["A1"]
        a1_vs_a = {"mean_diff_A_minus_A1": float(d.mean()), "A1_lower_on": f"{int((d > 0).sum())}/{len(d)}",
                   "wilcoxon_p": S.wilcoxon_p(d), "ci95": S.bootstrap_ci(d)}
    decided = all(p["wilcoxon_p"] < 0.05 and p["mean_diff"] > 0 for p in pairs.values())
    gen = "A1"   # what musubi's generator does unpatched (G0.2)
    if best == gen:
        action = "A1 wins: train in A1; the stock generator already matches"
    elif best == "A":
        action = ("A wins: train in A (--i2v_layout A) and generate with the patched generator "
                  "(kandinsky5_generate_video.py --i2v_layout A); G0.2 gen_A checks the match")
    else:
        action = f"{best} wins: patch the GENERATOR to {best} and train in {best} (PLAN §4 G0.3)"
    out = {"gate": "G0.3", "n_clips": len(clips), "k": a.k, "t_grid": list(C.T_GRID),
           "mean_loss_true": mean, "ranking": ranked, "winner": best,
           "winner_significant_vs_all": decided, "pairs": pairs, "a1_vs_a": a1_vs_a,
           "accuracy": {lay: res[lay].get("summary") for lay in res},
           "generator_layout": gen, "action": action,
           "rule": "lowest paired mean true-caption loss on the 26 QF2_R clips (PLAN §4); "
                   "significance by Wilcoxon signed-rank, reported not gated"}
    C.write_json(C.PHASE0 / "g03_layout.json", out)
    print(f"G0.3 mean loss: " + "  ".join(f"{k} {mean[k]:.5f}" for k in ranked))
    for k, p in pairs.items():
        print(f"  {k}: +{p['rel_diff']*100:.1f}%  {best} lower on {p['best_lower_on']}  "
              f"Wilcoxon p={p['wilcoxon_p']:.2g}")
    print(f"-> winner {best}; {action}\n-> {C.PHASE0 / 'g03_layout.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--layouts", nargs="+", default=["A"], choices=["A", "A1", "B", "stock"])
    s.add_argument("--k", type=int, default=C.K_DRAWS)
    s.add_argument("--split", nargs="*", default=["holdout"])
    s.add_argument("--kind", choices=("expression", "motion"))
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--true-only", action="store_true", help="score the true caption only (G0.3)")
    g = sub.add_parser("g03")
    g.add_argument("--k", type=int, default=C.K_DRAWS)
    a = ap.parse_args()
    {"score": cmd_score, "g03": cmd_g03}[a.cmd](a)


if __name__ == "__main__":
    main()
