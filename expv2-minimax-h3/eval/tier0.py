#!/usr/bin/env python3
"""Tier 0 (PLAN §7.2) on CPU, no torch: LoRA size and update coherence per checkpoint.

For every RUNS/<run>/step-<samples>.safetensors (optimizer step = samples / 8, C.opt_step):

  norm            ||dW||_F summed over modules, dW = B @ A (DiffSynth's peft LoRA has
                  alpha = rank, so the scale is 1). Computed through r x r Gram matrices,
                  ||BA||^2 = tr((B^T B)(A A^T)) - never the full 33B-shaped product.
  norm_by_group   the same per target (qkv_proj / out_proj / fc1 / fc2)
  eff_rank        mean over modules of the entropy effective rank of dW (exp of the
                  entropy of the normalised singular values; D5: Wan used ~9 of 32)
  update_norm     ||W_k - W_{k-1}|| between consecutive saved checkpoints (W_0 = 0)
  update_cos      cos(W_k - W_{k-1}, W_{k+1} - W_k) - B18: noisy constant-LR updates
                  show near-zero or negative cosines; accum 8 + cosine LR should not

Inner products of low-rank products: <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)).

    python expv2-minimax-h3/eval/tier0.py --run expv2-h3-main
    python expv2-minimax-h3/eval/tier0.py --run expv2-h3-s12 expv2-h3-s222
-> EVAL/tier0/<run>.json
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv2_common as C  # noqa: E402

STEP_RE = re.compile(r"^step-(\d+)\.safetensors$")
_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "BF16": None}


def read_safetensors(path):
    """Minimal reader: 8-byte little-endian header length, JSON header, raw buffer.
    BF16 is widened to float32 by shifting the 16 stored bits into the high half."""
    path = Path(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    buf = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        s, e = spec["data_offsets"]
        raw = buf[s:e]
        dt = spec["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw.tobytes(), dtype="<u2").astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt in _DT:
            arr = np.frombuffer(raw.tobytes(), dtype=np.dtype(_DT[dt]).newbyteorder("<")).astype(np.float32)
        else:
            raise ValueError(f"{path.name}: unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(spec["shape"])
    return out


def lora_pairs(sd):
    """{module: (A [r, in], B [out, r])} from peft-style keys (lora_A/lora_B, optional
    '.default'), also kohya-style lora_down/lora_up."""
    pairs = {}
    for k in sd:
        for a_tag, b_tag in ((".lora_A.default.weight", ".lora_B.default.weight"),
                             (".lora_A.weight", ".lora_B.weight"),
                             (".lora_down.weight", ".lora_up.weight")):
            if k.endswith(a_tag):
                mod = k[: -len(a_tag)]
                bk = mod + b_tag
                if bk in sd:
                    pairs[mod] = (sd[k].astype(np.float64), sd[bk].astype(np.float64))
    return pairs


def group_of(mod):
    for g in ("qkv_proj", "out_proj", "fc1", "fc2"):
        if mod.endswith(g):
            return g
    return "other"


def inner(p, q):
    """<B1 A1, B2 A2>_F for p = (A1, B1), q = (A2, B2)."""
    (A1, B1), (A2, B2) = p, q
    return float(np.trace((B1.T @ B2) @ (A2 @ A1.T)))


def eff_rank(A, B):
    """Entropy effective rank of B @ A via the r x r problem: the singular values of BA are
    sqrt of the eigenvalues of (A A^T)^(1/2) (B^T B) (A A^T)^(1/2)."""
    G = A @ A.T
    w, V = np.linalg.eigh(G)
    Gh = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    ev = np.clip(np.linalg.eigvalsh(Gh @ (B.T @ B) @ Gh), 0, None)
    s = np.sqrt(ev)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def list_ckpts(run):
    d = C.RUNS / run
    out = []
    for p in d.glob("step-*.safetensors"):
        m = STEP_RE.match(p.name)
        if m:
            s = int(m.group(1))
            out.append((C.opt_step(s), s, p))
    return sorted(out, key=lambda t: t[1])


def analyse(run, ckpts):
    loaded = []
    for opt, samples, p in ckpts:
        pairs = lora_pairs(read_safetensors(p))
        if not pairs:
            sys.exit(f"{p}: no LoRA A/B pairs found")
        loaded.append((opt, samples, p, pairs))
    mods = sorted(set.intersection(*[set(x[3]) for x in loaded]))

    def G(i, j):
        """<W_i, W_j> summed over modules; index -1 is W_0 = 0."""
        if i < 0 or j < 0:
            return 0.0
        return sum(inner(loaded[i][3][m], loaded[j][3][m]) for m in mods)

    gram = {}

    def g(i, j):
        k = (min(i, j), max(i, j))
        if k not in gram:
            gram[k] = G(*k)
        return gram[k]

    rows = []
    for k, (opt, samples, p, pairs) in enumerate(loaded):
        by_group = {}
        for m in mods:
            by_group.setdefault(group_of(m), 0.0)
            by_group[group_of(m)] += inner(pairs[m], pairs[m])
        upd = g(k, k) + g(k - 1, k - 1) - 2 * g(k - 1, k)
        row = {"ckpt": f"{run}@{opt}", "opt_step": opt, "samples": samples, "file": p.name,
               "modules": len(mods),
               "norm": float(np.sqrt(max(g(k, k), 0.0))),
               "norm_by_group": {gk: float(np.sqrt(max(v, 0.0))) for gk, v in sorted(by_group.items())},
               "eff_rank": float(np.mean([eff_rank(*pairs[m]) for m in mods])),
               "update_norm": float(np.sqrt(max(upd, 0.0))),
               "update_cos": None}
        if k + 1 < len(loaded):
            # <W_k - W_{k-1}, W_{k+1} - W_k>
            dot = g(k, k + 1) - g(k, k) - g(k - 1, k + 1) + g(k - 1, k)
            nxt = g(k + 1, k + 1) + g(k, k) - 2 * g(k, k + 1)
            den = np.sqrt(max(upd, 0.0) * max(nxt, 0.0))
            row["update_cos"] = float(dot / den) if den > 0 else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--files", nargs="*", help="explicit checkpoint files (testing); overrides discovery")
    a = ap.parse_args()
    for run in a.run:
        if a.files:
            ckpts = []
            for f in a.files:
                m = STEP_RE.match(Path(f).name)
                s = int(m.group(1)) if m else 0
                ckpts.append((C.opt_step(s), s, Path(f)))
            ckpts.sort(key=lambda t: t[1])
        else:
            ckpts = list_ckpts(run)
        if not ckpts:
            print(f"[{run}] no checkpoints under {C.RUNS / run}")
            continue
        rows = analyse(run, ckpts)
        out = {"run": run, "checkpoints": rows,
               "note": "update_cos near 0 or negative = noisy updates (B18); opt_step = samples / 8"}
        dest = C.EVAL / "tier0" / f"{run}.json"
        C.write_json(dest, out)
        print(f"[{run}] {len(rows)} checkpoints -> {dest}")
        for r in rows:
            uc = "  -  " if r["update_cos"] is None else f"{r['update_cos']:+.3f}"
            print(f"  step {r['opt_step']:4d}  norm {r['norm']:.4f}  upd {r['update_norm']:.4f}  "
                  f"cos {uc}  eff_rank {r['eff_rank']:.1f}")


if __name__ == "__main__":
    main()
