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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v8merged")
    ap.add_argument("--asset", required=True, help="*.asset.safetensors (or asset_init*.safetensors)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--offset", action="store_true")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--split", nargs="*", default=["holdout", "seen"])
    ap.add_argument("--kinds", nargs="*", default=["prop"])
    a = ap.parse_args()
    dev = torch.device("cuda")
    index = [e for e in json.loads((D.HOLDOUT / "index.json").read_text())
             if e["split"] in a.split and e["kind"] in a.kinds]
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
    print(f"spliced E into {n} of {len(emb)} captions")

    t0 = time.time()
    model = D.load_model(a.config, dev)
    rows = [D.score_clip(model, e, emb, a.k, dev) for e in index]
    out = C.RESULTS / "dcls" / f"asset_{a.tag}_k{a.k}.json"
    out.write_text(json.dumps({"config": a.config, "asset": a.asset, "offset": a.offset, "k": a.k,
                               "t_grid": D.T_GRID, "summary": dcls_v9.summarise(rows), "rows": rows,
                               "seconds": round(time.time() - t0)}, indent=1))
    for key, s in dcls_v9.summarise(rows).items():
        print(f"   {key:20s} {s}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
