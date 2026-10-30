#!/usr/bin/env python3
"""Tier-1 for a learned asset embedding (asset_embed_v9.py): the v9 diffusion classifier with
E spliced into every candidate caption that contains the asset phrase.

Only the registry phrase is replaced, so on prop clips the true caption carries E and the
distractors ("a phone", "a small red book", ...) do not: the margin is then a direct measure
of what E adds (bar P2 in RESEARCH_v9c_fixes section 5). Captions without the phrase are
untouched, so v8-label rows are bit-identical to the base config and are skipped by default.

    python dcls_asset_v9.py --config v8merged --asset <ckpt>.asset.safetensors --tag e5@100
    python dcls_asset_v9.py --config v8merged --asset ... --offset --k 4 --split holdout seen
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_v9 as C  # noqa: E402
import dcls_v9  # noqa: E402,F401  (wires dcls_v8 to the v9 clip set and configs)
import dcls_v8 as D  # noqa: E402
import asset_embed_v9 as A  # noqa: E402


# Registry text revisions since the index was built (each applied to stored captions in order).
REVISIONS = [
    ("the iFin phone, a small pale grey phone with a black screen, three camera lenses on the back and a little tuft on top",
     "the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top"),
    ("the screen turned toward itself", "holding it with its back toward the viewer"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v8merged")
    ap.add_argument("--asset", required=True, help="*.asset.safetensors (or asset_init*.safetensors)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--offset", action="store_true")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--split", nargs="*", default=["holdout", "seen"])
    ap.add_argument("--kinds", nargs="*", default=["prop"])
    ap.add_argument("--expert", choices=("low", "high"), default="low",
                    help="high: score the high-noise expert (its LoRA list) at t in (0.95, 0.92)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.expert == "high":
        from safetensors.torch import load_file as _lf
        from musubi_tuner.wan.configs import WAN_CONFIGS
        from musubi_tuner.wan.modules.model import load_wan_model

        def load_high(name, device):
            _, high = C.lora_config(name)
            for p, _m in high:
                if not Path(p).exists():
                    raise FileNotFoundError(p)
            model = load_wan_model(WAN_CONFIGS["i2v-A14B"], device, str(C.C8.DIT_HIGH), "sdpa", False, "cpu",
                                   torch.float16, False, lora_weights_list=[_lf(str(p)) for p, _ in high],
                                   lora_multipliers=[m for _, m in high])
            return model.to(device).eval()

        D.load_model = load_high
        D.T_GRID = (0.95, 0.92)               # inside the high-noise expert's range (t >= 0.9)
    index = [e for e in json.loads((D.HOLDOUT / "index.json").read_text())
             if e["split"] in a.split and e["kind"] in a.kinds]
    # The index stores captions compiled when it was built (2026-10-24). Phone candidates are recompiled from
    # the LIVE registry with build_assets_v9's own call, so E is spliced into the wording it was trained on.
    # Check: each recompiled caption = the stored one with only the known registry revisions applied.
    props = {(j["char"], j["label"], j["angle"]): j for j in C.P9.discover_props({"props_skipped": []})}
    n_prop = 0
    for e in index:
        if e["kind"] != "prop":
            continue
        job = props[(e["char"], e["prop_label"], e["angle"])]
        new = {k: C.phone_caption(job, v) for k, v in C.BINDING.items()}
        for k, old in e["candidates"].items():
            exp = old
            for o, n_ in REVISIONS:
                exp = exp.replace(o, n_)
            assert exp == new[k], f"{e['clip']}/{k}: recompiled caption differs beyond the known registry revisions"
        e["candidates"] = new
        n_prop += 1
    texts = [c for e in index for c in e["candidates"].values()]
    emb = D.encode_captions(texts, dev)

    tok = A.tokenizer()
    ph_ids = A.token_ids(tok, A.phrase())[:-1]
    E = load_file(a.asset)["E"].float()
    n = 0
    for t in list(emb):
        s, e = A.find_span(A.token_ids(tok, t), ph_ids)
        if e > s:
            assert emb[t].shape[0] == len(A.token_ids(tok, t)), "cache rows != tokens"
            emb[t] = A.splice([emb[t].float()], [(s, e)], E, a.offset)[0].to(torch.float16)
            n += 1
    print(f"spliced E into {n} of {len(emb)} captions ({n_prop} phone clips)")
    n_true = len({e["candidates"]["phone"] for e in index if e["kind"] == "prop"})
    assert n == n_true, f"E spliced into {n} captions, expected the {n_true} distinct true phone captions"

    t0 = time.time()
    model = D.load_model(a.config, dev)
    rows = [D.score_clip(model, e, emb, a.k, dev) for e in index]
    out = C.RESULTS / "dcls" / f"asset_{a.tag}_k{a.k}.json"   # expert recorded inside
    out.write_text(json.dumps({"config": a.config, "asset": a.asset, "offset": a.offset, "k": a.k, "expert": a.expert,
                               "t_grid": D.T_GRID, "summary": dcls_v9.summarise(rows), "rows": rows,
                               "seconds": round(time.time() - t0)}, indent=1))
    for key, s in dcls_v9.summarise(rows).items():
        print(f"   {key:20s} {s}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
