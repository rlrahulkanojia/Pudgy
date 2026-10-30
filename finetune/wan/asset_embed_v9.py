#!/usr/bin/env python3
"""Learned asset embedding for v9c2 (training_approach/v9/RESEARCH_asset_memory.md section 3).

The trainer never runs the text encoder: it reads cached umT5 outputs, one [L x 4096] matrix
per caption. An asset is stored as a matrix E [K x 4096] that replaces the rows of the
asset phrase's token span, wherever the phrase occurs. E starts as the cached rows of the
phrase itself, so at step 0 the model is exactly the base, and any later change is E's alone.
Captions without the phrase never see E: they are bit-identical to the base.

  span   : token span of the phrase in a caption, found with the same tokenizer as the cache
  cache  : copy a latent cache dir, adding `varlen_assetspan` [2] to every *_te file
  check  : V0 - spans found in every phone caption, row counts match, splice is exact
  train  : run musubi wan_train_network with E as an extra trainable tensor (patched in memory)
           --base-weights-high <lora>: with musubi's two-expert mode (--dit_high_noise), merge this
           LoRA into the HIGH-noise expert as it loads. musubi's --base_weights only reaches the
           expert active at load time (low noise), so without this the high-noise expert would
           train without its golden LoRA (RESEARCH_asset_memory section 3).
           --freeze-asset: E is spliced in as a constant (pivotal stage 2: train the LoRA, not E).
           --preserve [--preserve-weight L]: on samples WITHOUT the asset phrase (replay), the target is
           the base model's own prediction (same network, LoRA multiplier 0, no grad), so the loss
           there is L * ||v_theta - v_base||^2 (RESEARCH_v9c_fixes section 3, fix 2). Batch size 1.

    python asset_embed_v9.py check --jsonl-dir /workspace/data_v9c --cache /workspace/wan_cache/latents_v9c
    python asset_embed_v9.py cache --jsonl-dir /workspace/data_v9c --cache /workspace/wan_cache/latents_v9c \
        --out /workspace/wan_cache/latents_v9c_emb
    python asset_embed_v9.py train --init /workspace/wan_cache/latents_v9c_emb/asset_init.safetensors \
        --asset-lr 5e-3 --embed-only -- <wan_train_network.py args>
"""
import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "assets" / "registry.toml"
SPAN_KEY = "varlen_assetspan"
TE_KEY = "varlen_t5_bfloat16"
MUSUBI = Path(os.environ.get("MUSUBI", "/workspace/musubi-tuner"))


def phrase(asset="phone"):
    reg = tomllib.loads(REGISTRY.read_text())
    for sect in reg.values():
        if isinstance(sect, dict) and asset in sect and "phrase" in sect[asset]:
            return sect[asset]["phrase"]
    raise KeyError(f"no phrase for {asset} in {REGISTRY}")


def tokenizer():
    sys.path.insert(0, str(MUSUBI / "src"))
    from musubi_tuner.wan.modules.tokenizers import HuggingfaceTokenizer
    return HuggingfaceTokenizer(name="Wan-AI/Wan2.1-T2V-14B", subfolder="google/umt5-xxl",
                                seq_len=512, clean="whitespace")


def token_ids(tok, text):
    ids, mask = tok([text], return_mask=True, add_special_tokens=True)
    return ids[0, : int(mask[0].sum())].tolist()        # includes the closing </s>


def find_span(cap_ids, ph_ids):
    """[start, end) of ph_ids inside cap_ids, or (0, 0). The phrase must occur at most once."""
    n = len(ph_ids)
    hits = [i for i in range(len(cap_ids) - n + 1) if cap_ids[i:i + n] == ph_ids]
    if len(hits) > 1:
        raise ValueError("phrase occurs more than once in a caption")
    return (hits[0], hits[0] + n) if hits else (0, 0)


def splice(context, spans, E, offset=False):
    """Replace rows [s, e) of each context with E (offset: add E to them). Rows outside the
    span are untouched; a (0, 0) span returns the context object itself, so non-asset
    captions are bit-identical."""
    out = []
    for ctx, (s, e) in zip(context, spans):
        s, e = int(s), int(e)
        if e <= s:
            out.append(ctx)
            continue
        mid = ctx[s:e] + E.to(ctx.dtype) if offset else E.to(ctx.dtype)
        out.append(torch.cat([ctx[:s], mid, ctx[e:]], 0))
    return out


def captions(jsonl_dir):
    """te-cache basename -> caption, from the prep_v9 jsonl files (workspace paths)."""
    m = {}
    for f in sorted(Path(jsonl_dir).glob("*.workspace.jsonl")):
        for line in f.read_text().splitlines():
            j = json.loads(line)
            m[Path(j["video_path"]).stem + "_wan_te.safetensors"] = j["caption"]
    return m


def spans_for(jsonl_dir, ph):
    tok = tokenizer()
    ph_ids = token_ids(tok, ph)[:-1]                      # drop </s>
    caps = captions(jsonl_dir)
    return {k: (find_span(token_ids(tok, c), ph_ids), len(token_ids(tok, c))) for k, c in caps.items()}, ph_ids


def cmd_check(a):
    ph = phrase()
    sp, ph_ids = spans_for(a.jsonl_dir, ph)
    te = {p.name: p for p in Path(a.cache).rglob("*_wan_te.safetensors")}
    n_asset = bad_len = missing = 0
    init, spread = None, []
    for name, ((s, e), n_tok) in sp.items():
        if name not in te:
            missing += 1
            continue
        ctx = load_file(te[name])[TE_KEY]
        bad_len += ctx.shape[0] != n_tok                  # cache rows must equal token count
        if e > s:
            n_asset += 1
            rows = ctx[s:e].float()
            init = rows if init is None else init
            spread.append(rows)
            assert splice([ctx], [(s, e)], ctx[s:e])[0].equal(ctx), "identity splice changed rows"
        else:
            assert splice([ctx], [(0, 0)], torch.zeros(e - s + 3, ctx.shape[1]))[0] is ctx
    phone_caps = sum(1 for k in sp if "phone" in k)
    if spread:
        R = torch.stack(spread)
        rel = ((R - R.mean(0)).flatten(1).norm(dim=1) / R.flatten(1).norm(dim=1))
        print(f"replace-mode init: per-caption distance from the mean rows {rel.mean():.3f} "
              f"(max {rel.max():.3f}) of the rows' norm")
    print(f"phrase: {len(ph_ids)} tokens  |  captions {len(sp)}  te files {len(te)}  missing {missing}")
    print(f"asset spans found {n_asset} / {phone_caps} phone captions  |  row-count mismatches {bad_len}")
    ok = missing == 0 and bad_len == 0 and n_asset == phone_caps
    print("V0 PASS" if ok else "V0 FAIL")
    sys.exit(0 if ok else 1)


def cmd_cache(a):
    """Mirror the cache dir: latents symlinked, te files rewritten with the span key. E's init
    is the mean of the phrase rows over the asset captions (one fixed start for every caption)."""
    ph = phrase()
    sp, ph_ids = spans_for(a.jsonl_dir, ph)
    src, out = Path(a.cache), Path(a.out)
    rows = []
    for p in sorted(src.rglob("*.safetensors")):
        dst = out / p.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not p.name.endswith("_wan_te.safetensors"):
            if not dst.exists():
                dst.symlink_to(p)
            continue
        sd = load_file(p)
        (s, e), _ = sp.get(p.name, ((0, 0), 0))
        sd[SPAN_KEY] = torch.tensor([s, e], dtype=torch.int64)
        if e > s:
            rows.append(sd[TE_KEY][s:e].float())
        save_file(sd, dst)
    E0 = torch.stack(rows).mean(0)
    save_file({"E": torch.zeros_like(E0).contiguous()}, out / "asset_init_offset.safetensors",
              metadata={"phrase": ph, "mode": "offset"})
    save_file({"E": E0.contiguous(), "phrase_tokens": torch.tensor(ph_ids)}, out / "asset_init.safetensors",
              metadata={"phrase": ph, "n_captions": str(len(rows))})
    print(f"wrote {out}: {len(rows)} asset captions, E0 {tuple(E0.shape)}")


def merge_lora_into(model, path, multiplier=1.0):
    """Merge a musubi-format LoRA into a loaded WanModel in place: W += m * (up @ down) * alpha/rank,
    computed in fp32 on CPU and written back in the weight's own dtype and device (the same
    LoRANetwork.merge_to path musubi uses for --base_weights)."""
    from musubi_tuner.networks import lora_wan
    wdt = next(m.weight.dtype for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert wdt in (torch.float16, torch.bfloat16, torch.float32), f"refusing to merge into {wdt} weights (fp8?)"
    sd = load_file(path)
    net = lora_wan.create_arch_network_from_weights(multiplier, sd, unet=model, for_inference=True)
    n_mod = len({k.split(".")[0] for k in sd if k.endswith("lora_down.weight")})
    assert len(net.unet_loras) == n_mod, f"{path}: {n_mod} LoRA modules in file, {len(net.unet_loras)} matched the model"
    net.merge_to(None, model, sd, None, "cpu")
    print(f"[asset] merged {n_mod} modules of {Path(path).name} (x{multiplier}) into the high-noise expert", flush=True)
    return n_mod


def cmd_train(a, rest):
    """Patch musubi in memory, then hand over to its own main()."""
    sys.path.insert(0, str(MUSUBI / "src"))
    sys.argv = [str(MUSUBI / "src/musubi_tuner/wan_train_network.py"), *rest]
    import musubi_tuner.wan_train_network as W
    from musubi_tuner.networks import lora as L

    dev = torch.device("cuda")
    expert_steps = {"high": 0, "low": 0, "preserve": 0}   # samples per expert; preservation samples
    assert not (a.embed_only and a.freeze_asset), "--embed-only and --freeze-asset leave nothing to train"
    E0 = load_file(a.init)["E"].to(dev, torch.float32)
    E = E0.clone() if a.freeze_asset else torch.nn.Parameter(E0.clone())
    state = {}
    E_init = E.detach().clone()
    if os.environ.get("ASSET_DEBUG"):
        E.register_hook(lambda g: print(f"[asset-debug] E grad norm {g.norm():.3e}", flush=True))

    orig_prep = L.LoRANetwork.prepare_optimizer_params

    def prep(self, *args, **kw):
        state["net"] = self
        params, descs = orig_prep(self, *args, **kw)
        if a.embed_only:
            for p in self.parameters():
                p.requires_grad_(False)
            params, descs = [], []
        if not a.freeze_asset:
            params.append({"params": [E], "lr": a.asset_lr})
            descs.append("asset_embed")
        return params, descs

    L.LoRANetwork.prepare_optimizer_params = prep
    orig_tp = L.LoRANetwork.get_trainable_params
    L.LoRANetwork.get_trainable_params = lambda self: [*orig_tp(self)] + ([] if a.freeze_asset else [E])

    orig_save = L.LoRANetwork.save_weights

    def save(self, file, dtype, metadata):
        if not a.embed_only:
            orig_save(self, file, dtype, metadata)
        side = str(file).rsplit(".", 1)[0] + ".asset.safetensors"
        print(f"[asset] samples per expert so far: {expert_steps}", flush=True)
        save_file({"E": E.detach().float().cpu().contiguous()}, side,
                  metadata={"phrase": phrase(), "delta_norm": f"{(E - E_init).norm():.4f}",
                           "init_norm": f"{E_init.norm():.4f}",
                           "samples_high": str(expert_steps["high"]), "samples_low": str(expert_steps["low"]),
                           "samples_preserve": str(expert_steps["preserve"]), "frozen": str(a.freeze_asset)})

    L.LoRANetwork.save_weights = save

    orig_call = W.WanNetworkTrainer.call_dit

    def call_dit(self, args, accelerator, transformer, latents, batch, *rest_, **kw):
        if getattr(self, "high_low_training", False):
            expert_steps["high" if self.next_model_is_high_noise else "low"] += 1
        if os.environ.get("ASSET_DEBUG"):
            print(f"[asset-debug] keys {sorted(batch)} spans {[x.tolist() for x in batch.get('assetspan', [])]} "
                  f"high={getattr(self, 'next_model_is_high_noise', None)} grad_enabled={torch.is_grad_enabled()}", flush=True)
        if "assetspan" in batch:
            batch["t5"] = splice([t.to(dev) for t in batch["t5"]], batch["assetspan"], E, a.offset)
            if os.environ.get("ASSET_DEBUG"):
                print(f"[asset-debug] after splice: requires_grad {[t.requires_grad for t in batch['t5']]}", flush=True)
        spans = [x.tolist() for x in batch.get("assetspan", [])]
        if a.preserve and spans and all(e <= s_ for s_, e in spans):        # replay sample: distil the base
            assert len(spans) == 1, "--preserve assumes batch size 1"
            net = state["net"]
            # Block swap in training mode moves blocks during a forward and relies on the BACKWARD pass to
            # move them back. The base pass has no backward, so run it in forward-only mode and reset block
            # placement afterwards: the same switch musubi itself makes around sample generation (each
            # switch drains in-flight transfers and calls prepare_block_swap_before_forward).
            m = accelerator.unwrap_model(transformer)
            swap = bool(getattr(m, "blocks_to_swap", 0))
            if swap:
                m.switch_block_swap_for_inference()
            with torch.no_grad():
                net.set_multiplier(0.0)
                try:
                    base = orig_call(self, args, accelerator, transformer, latents, batch, *rest_, **kw)
                finally:
                    net.set_multiplier(1.0)
                    if swap:
                        m.switch_block_swap_for_training()
            out = orig_call(self, args, accelerator, transformer, latents, batch, *rest_, **kw)
            expert_steps["preserve"] += 1
            w = a.preserve_weight ** 0.5
            return W.DiTOutput(pred=out.pred * w, target=base.pred.detach() * w)
        return orig_call(self, args, accelerator, transformer, latents, batch, *rest_, **kw)

    W.WanNetworkTrainer.call_dit = call_dit

    if a.base_weights_high:
        assert "--dit_high_noise" in rest, "--base-weights-high needs musubi's --dit_high_noise"
        high_path = os.path.realpath(rest[rest.index("--dit_high_noise") + 1])
        orig_load = W.load_wan_model

        def load_wan_model(config, device, dit_path, *args_, **kw):
            model = orig_load(config, device, dit_path, *args_, **kw)
            if os.path.realpath(dit_path) == high_path:      # before load_transformer enables block swap
                merge_lora_into(model, a.base_weights_high, a.base_weights_high_mult)
            return model

        W.load_wan_model = load_wan_model
    W.main()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("check", "cache"):
        p = sub.add_parser(n)
        p.add_argument("--jsonl-dir", required=True)
        p.add_argument("--cache", required=True)
        if n == "cache":
            p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--init", required=True)
    p.add_argument("--asset-lr", type=float, default=5e-3)
    p.add_argument("--embed-only", action="store_true")
    p.add_argument("--offset", action="store_true", help="E is added to the phrase rows (init 0)")
    p.add_argument("--base-weights-high", help="LoRA merged into the high-noise expert (two-expert mode)")
    p.add_argument("--freeze-asset", action="store_true", help="splice E as a constant; train only the LoRA")
    p.add_argument("--preserve", action="store_true", help="distil the base on samples without the asset phrase")
    p.add_argument("--preserve-weight", type=float, default=1.0)
    p.add_argument("--base-weights-high-mult", type=float, default=1.0)
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    {"check": cmd_check, "cache": cmd_cache}.get(a.cmd, lambda a: cmd_train(a, rest))(a)


if __name__ == "__main__":
    main()
