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


def cmd_train(a, rest):
    """Patch musubi in memory, then hand over to its own main()."""
    sys.path.insert(0, str(MUSUBI / "src"))
    sys.argv = [str(MUSUBI / "src/musubi_tuner/wan_train_network.py"), *rest]
    import musubi_tuner.wan_train_network as W
    from musubi_tuner.networks import lora as L

    dev = torch.device("cuda")
    E = torch.nn.Parameter(load_file(a.init)["E"].to(dev, torch.float32))
    E_init = E.detach().clone()

    orig_prep = L.LoRANetwork.prepare_optimizer_params

    def prep(self, *args, **kw):
        params, descs = orig_prep(self, *args, **kw)
        if a.embed_only:
            for p in self.parameters():
                p.requires_grad_(False)
            params, descs = [], []
        params.append({"params": [E], "lr": a.asset_lr})
        descs.append("asset_embed")
        return params, descs

    L.LoRANetwork.prepare_optimizer_params = prep
    orig_tp = L.LoRANetwork.get_trainable_params
    L.LoRANetwork.get_trainable_params = lambda self: [*orig_tp(self), E]

    orig_save = L.LoRANetwork.save_weights

    def save(self, file, dtype, metadata):
        if not a.embed_only:
            orig_save(self, file, dtype, metadata)
        side = str(file).rsplit(".", 1)[0] + ".asset.safetensors"
        save_file({"E": E.detach().float().cpu().contiguous()}, side,
                  metadata={"phrase": phrase(), "delta_norm": f"{(E - E_init).norm():.4f}",
                           "init_norm": f"{E_init.norm():.4f}"})

    L.LoRANetwork.save_weights = save

    orig_call = W.WanNetworkTrainer.call_dit

    def call_dit(self, args, accelerator, transformer, latents, batch, *rest_, **kw):
        if "assetspan" in batch:
            batch["t5"] = splice([t.to(dev) for t in batch["t5"]], batch["assetspan"], E, a.offset)
        return orig_call(self, args, accelerator, transformer, latents, batch, *rest_, **kw)

    W.WanNetworkTrainer.call_dit = call_dit
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
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    {"check": cmd_check, "cache": cmd_cache}.get(a.cmd, lambda a: cmd_train(a, rest))(a)


if __name__ == "__main__":
    main()
