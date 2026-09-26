#!/usr/bin/env python3
"""Tier-0 (free, CPU) weight-space tools for the v8 run.

  traj  <run_dir> [--tb]    trajectory diagnostics over every saved checkpoint
  swa   <run_dir> --from S --to E [--rank R]   average checkpoints, re-compress exactly
  geom  <a.safetensors> <b.safetensors>        cosine / subspace overlap of two LoRAs

A kohya/musubi module stores A = lora_down (r x in), B = lora_up (out x r), alpha, and
contributes dW = (alpha / r) B A. Everything below is computed from r x r cores:
    <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)),   sigma(U V) = sigma(R_U R_V^T)
so no out x in matrix is ever formed. v8 trains a FRESH LoRA on a merged base (v7fixed),
so each checkpoint IS the v8 task vector tau(t) - nothing to subtract.

traj reports, per checkpoint t:  ||tau(t)||, cos(tau(t), tau(last)), and fits
    ||tau(t)||^2 = D^2 t^2 + sigma^2 t
i.e. systematic drift D vs random-walk diffusion sigma. In v7 (batch 1, constant LR) the
drift share of ||tau||^2 was 5% at step 500 and 19% at 2500 - the update was mostly noise,
which is why the behavioural sweep was flat. v8's accumulation + cosine decay should push
the drift share up; if it does not, that is a finding worth stopping for.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

torch.set_num_threads(32)
DT = torch.float64


def load(path):
    sd = load_file(str(path))
    mods = {}
    for k, v in sd.items():
        m = re.match(r"(.*)\.(lora_down|lora_up)\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.to(DT).flatten(1)
        elif k.endswith(".alpha"):
            mods.setdefault(k[:-6], {})["alpha"] = float(v)
    return {n: ((d.get("alpha", d["lora_down"].shape[0]) / d["lora_down"].shape[0]),
                d["lora_up"], d["lora_down"]) for n, d in mods.items()}


def ip(X, Y):
    s = 0.0
    for cx, Bx, Ax in X:
        for cy, By, Ay in Y:
            s += cx * cy * torch.trace((Bx.T @ By) @ (Ay @ Ax.T)).item()
    return s


def gip(L1, L2, names):
    return sum(ip([L1[n]], [L2[n]]) for n in names)


def ckpts(run):
    out = []
    for p in Path(run).glob("*-step*.safetensors"):
        m = re.search(r"-step(\d+)\.safetensors$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def traj(run, tb=False):
    cs = ckpts(run)
    if len(cs) < 2:
        print("need >= 2 step checkpoints"); return None
    L = {s: load(p) for s, p in cs}
    names = sorted(L[cs[0][0]])
    steps = [s for s, _ in cs]
    last = steps[-1]
    nn = {s: math.sqrt(max(gip(L[s], L[s], names), 0)) for s in steps}
    cos_last = {s: gip(L[s], L[last], names) / (nn[s] * nn[last] + 1e-30) for s in steps}
    # consecutive increments d_i = tau(s_i) - tau(s_{i-1})
    def inc(a, b):
        return {n: [L[b][n], (-L[a][n][0], L[a][n][1], L[a][n][2])] for n in names}
    incs = [inc(a, b) for a, b in zip(steps, steps[1:])]
    def icos(X, Y):
        num = sum(ip(X[n], Y[n]) for n in names)
        return num / math.sqrt(sum(ip(X[n], X[n]) for n in names) * sum(ip(Y[n], Y[n]) for n in names))
    inc_cos = [icos(incs[i], incs[i + 1]) for i in range(len(incs) - 1)]
    t = np.array(steps, float); y = np.array([nn[s] ** 2 for s in steps])
    (D2, s2), *_ = np.linalg.lstsq(np.stack([t ** 2, t], 1), y, rcond=None)
    slope = float(np.polyfit(np.log(t), np.log([nn[s] for s in steps]), 1)[0])
    res = {"steps": steps, "norm": {s: round(nn[s], 4) for s in steps},
           "cos_to_last": {s: round(cos_last[s], 4) for s in steps},
           "consecutive_increment_cos": [round(c, 4) for c in inc_cos],
           "loglog_slope": round(slope, 3),
           "drift_D2": float(D2), "noise_sigma2": float(s2),
           "drift_share": {s: round(float(D2 * s * s / max(D2 * s * s + s2 * s, 1e-30)), 3)
                           for s in steps}}
    Path(run, "trajectory.json").write_text(json.dumps(res, indent=1))
    print(f"steps {steps}\n||tau|| {[res['norm'][s] for s in steps]}\n"
          f"log-log slope {slope:.3f} (1 = drift, 0.5 = random walk)\n"
          f"consecutive increment cos {res['consecutive_increment_cos']}\n"
          f"drift share of ||tau||^2 {res['drift_share']}")
    if tb:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(str(Path(run) / "logs" / "diag"))
        for s in steps:
            w.add_scalar("diag/tau_norm", nn[s], s)
            w.add_scalar("diag/cos_to_last", cos_last[s], s)
            w.add_scalar("diag/drift_share", res["drift_share"][s], s)
        for s, c in zip(steps[2:], inc_cos):
            w.add_scalar("diag/consecutive_increment_cos", c, s)
        w.close()
    return res


def swa(run, s_from, s_to, rank=None, out=None):
    """Average dW over checkpoints in [s_from, s_to] and re-compress to `rank` exactly:
    stack all K factor pairs (rank K*r), QR both sides, SVD the (K r x K r) core, keep the
    top `rank` singular directions. Reports the energy discarded by the truncation."""
    cs = [(s, p) for s, p in ckpts(run) if s_from <= s <= s_to]
    if len(cs) < 2:
        sys.exit(f"need >= 2 checkpoints in [{s_from}, {s_to}]")
    Ls = [load(p) for _, p in cs]
    names = sorted(Ls[0])
    with safe_open(str(cs[-1][1]), "pt") as f:
        meta = f.metadata() or {}
        dtype = f.get_tensor(next(k for k in f.keys() if k.endswith("lora_up.weight"))).dtype
    r0 = Ls[0][names[0]][2].shape[0]
    rank = rank or r0
    sd, lost = {}, []
    for n in names:
        U = torch.cat([L[n][0] * L[n][1] / len(Ls) for L in Ls], 1)
        V = torch.cat([L[n][2] for L in Ls], 0)
        Qu, Ru = torch.linalg.qr(U); Qv, Rv = torch.linalg.qr(V.T)
        Uc, S, Vh = torch.linalg.svd(Ru @ Rv.T)
        lost.append(float((S[rank:] ** 2).sum() / (S ** 2).sum()))
        sq = S[:rank].sqrt()
        B = (Qu @ Uc[:, :rank]) * sq               # out x rank
        A = (sq[:, None] * Vh[:rank]) @ Qv.T       # rank x in
        sd[f"{n}.lora_up.weight"] = B.to(dtype).contiguous()
        sd[f"{n}.lora_down.weight"] = A.to(dtype).contiguous()
        sd[f"{n}.alpha"] = torch.tensor(float(rank))   # scale alpha/rank = 1: dW = B A
    tag = f"swa{cs[0][0]}-{cs[-1][0]}"
    out = out or Path(run) / f"{Path(run).name}-{tag}.safetensors"
    meta = dict(meta, ss_network_dim=str(rank), ss_network_alpha=str(rank),
                v8_swa=json.dumps({"from": [s for s, _ in cs], "rank": rank}))
    save_file(sd, str(out), metadata=meta)
    print(f"wrote {out}\n  averaged {len(cs)} checkpoints {[s for s, _ in cs]} -> rank {rank}; "
          f"energy discarded by truncation: mean {np.mean(lost):.4f}, max {max(lost):.4f}")
    return out


def combine(parts, out):
    """Exact sum of LoRAs as ONE LoRA: sum_i m_i (a_i/r_i) B_i A_i = [m_i s_i B_i] [A_i].

    Rank = sum of the input ranks, alpha = rank (scale 1). Used to build the v8 base
    L_e + L_m - G (v7 expression + v7 motion - v2 golden = G + tau_e + tau_m): merged in
    one step instead of three fp16 round-trips that add, then remove, a whole golden."""
    Ls = [(load(p), m) for p, m in parts]
    names = sorted(Ls[0][0])
    for L, _ in Ls[1:]:
        if sorted(L) != names:
            sys.exit("module sets differ between inputs")
    with safe_open(str(parts[0][0]), "pt") as f:
        dtype = f.get_tensor(next(k for k in f.keys() if k.endswith("lora_up.weight"))).dtype
    sd = {}
    for n in names:
        B = torch.cat([m * L[n][0] * L[n][1] for L, m in Ls], 1)
        A = torch.cat([L[n][2] for L, _ in Ls], 0)
        sd[f"{n}.lora_up.weight"] = B.to(dtype).contiguous()
        sd[f"{n}.lora_down.weight"] = A.to(dtype).contiguous()
        sd[f"{n}.alpha"] = torch.tensor(float(A.shape[0]))
    rank = next(iter(sd.values())).shape[1]
    meta = {"ss_network_module": "networks.lora_wan", "ss_network_dim": str(rank),
            "ss_network_alpha": str(rank),
            "v8_combined_from": json.dumps([[str(p), m] for p, m in parts])}
    save_file(sd, str(out), metadata=meta)
    print(f"wrote {out}: {len(names)} modules, rank {rank}")
    return out


def geom(a, b):
    A, B = load(a), load(b)
    names = sorted(set(A) & set(B))
    num = gip(A, B, names)
    na, nb = math.sqrt(gip(A, A, names)), math.sqrt(gip(B, B, names))
    print(f"||a|| {na:.4f}  ||b|| {nb:.4f}  cos {num / (na * nb):.4f}  modules {len(names)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("traj"); t.add_argument("run"); t.add_argument("--tb", action="store_true")
    s = sp.add_parser("swa"); s.add_argument("run"); s.add_argument("--from", dest="s_from", type=int, required=True)
    s.add_argument("--to", dest="s_to", type=int, required=True); s.add_argument("--rank", type=int)
    g = sp.add_parser("geom"); g.add_argument("a"); g.add_argument("b")
    c = sp.add_parser("combine", help="exact sum: --part file mult (repeat)")
    c.add_argument("out"); c.add_argument("--part", nargs=2, action="append", required=True)
    a = ap.parse_args()
    if a.cmd == "traj":
        traj(a.run, a.tb)
    elif a.cmd == "swa":
        swa(a.run, a.s_from, a.s_to, a.rank)
    elif a.cmd == "combine":
        combine([(Path(p), float(m)) for p, m in a.part], Path(a.out))
    else:
        geom(a.a, a.b)
