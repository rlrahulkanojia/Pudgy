"""Weight-space geometry of the v7 LoRAs, computed exactly from the low-rank factors.

A kohya/musubi LoRA module stores A = lora_down (r x in), B = lora_up (out x r), alpha.
Its weight delta is  dW = (alpha/r) * B @ A.

A general sum of low-rank terms  X = sum_i c_i B_i A_i  is represented as a list of
(c, B, A). Frobenius inner products never materialise out x in matrices:
    <B1 A1, B2 A2>_F = tr(A1^T B1^T B2 A2) = tr( (B1^T B2) (A2 A1^T) )     (r x r cores)
Singular values of X = U V (U = [c_i B_i], V = [A_i]) come from QR of U and V^T:
    U = Qu Ru,  V^T = Qv Rv   =>   sigma(X) = sigma(Ru Rv^T)
"""
import sys, glob, os, re, json, collections
import torch
from safetensors.torch import load_file

torch.set_num_threads(48)
# Checkpoints downloaded from pudgy/v2/weights/curated and pudgy/v7/weights, flattened
# with "/" -> "__" (see ANALYSIS_v7.md, "Reproduce").
W = os.environ.get("V7_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "w"))
DT = torch.float64


def load(path):
    sd = load_file(path)
    mods = {}
    for k, v in sd.items():
        m = re.match(r"(.*)\.(lora_down|lora_up)\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.to(DT)
        elif k.endswith(".alpha"):
            mods.setdefault(k[:-6], {})["alpha"] = float(v)
    out = {}
    for name, d in mods.items():
        A, B = d["lora_down"], d["lora_up"]
        if A.dim() > 2:  # conv-shaped linear
            A, B = A.flatten(1), B.flatten(1)
        r = A.shape[0]
        out[name] = (d.get("alpha", r) / r, B, A)
    return out


def ip(X, Y):
    s = 0.0
    for cx, Bx, Ax in X:
        for cy, By, Ay in Y:
            s += cx * cy * torch.trace((Bx.T @ By) @ (Ay @ Ax.T)).item()
    return s


def svals(X):
    U = torch.cat([c * B for c, B, _ in X], 1)
    V = torch.cat([A for _, _, A in X], 0)
    _, Ru = torch.linalg.qr(U)
    _, Rv = torch.linalg.qr(V.T)
    return torch.linalg.svdvals(Ru @ Rv.T)


def left_basis(X, k):
    """Top-k left singular vectors of X (out x k)."""
    U = torch.cat([c * B for c, B, _ in X], 1)
    V = torch.cat([A for _, _, A in X], 0)
    Qu, Ru = torch.linalg.qr(U)
    _, Rv = torch.linalg.qr(V.T)
    Uc, _, _ = torch.linalg.svd(Ru @ Rv.T)
    return Qu @ Uc[:, :k]


def right_basis(X, k):
    U = torch.cat([c * B for c, B, _ in X], 1)
    V = torch.cat([A for _, _, A in X], 0)
    _, Ru = torch.linalg.qr(U)
    Qv, Rv = torch.linalg.qr(V.T)
    _, _, Vh = torch.linalg.svd(Ru @ Rv.T)
    return Qv @ Vh[:k].T


def tau(L, G, name):
    """Task vector of checkpoint L relative to the golden init G: L - G."""
    c, B, A = L[name]
    c0, B0, A0 = G[name]
    return [(c, B, A), (-c0, B0, A0)]


def nrm(X):
    return max(ip(X, X), 0.0) ** 0.5


def cos(X, Y):
    return ip(X, Y) / (nrm(X) * nrm(Y) + 1e-30)


def f(p):
    return os.path.join(W, p)
