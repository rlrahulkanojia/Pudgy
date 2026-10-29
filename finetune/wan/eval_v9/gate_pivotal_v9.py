#!/usr/bin/env python3
"""Gate for v9c2 candidates (RESEARCH_v9c_fixes section 5, bars fixed before the run).

Paired against the base on the same clips and noise (dcls JSONs from dcls_v9 / dcls_asset_v9):
  R1  per kind (expression, motion), held out: bootstrap 95% CI of mean delta-margin has lower bound
      >= -0.003, and top-1 accuracy drops by at most 1 clip
  R2  v8-label loss ratio (geometric mean over held-out clips), per kind, <= 1.05 at EACH t
  P1  held-out phone loss (true caption, summed over t) ratio vs base < 1.00
  P2  text-attributable gain on phone clips: mean over clips of
      dlog E(true caption) - mean dlog E(distractor captions) < 0, bootstrap 95% CI excluding 0
R1/R2 use --base-v8 (v8 labels); P1/P2 use --base-prop (the base scored with the SAME phone wording).
--gating R2 makes only R2 decide (high-noise expert: label margins at t >= 0.9 are weak by construction,
v8 NOTES #5, so R1 is reported there but not gated); P1/P2 are skipped when no phone clips were scored.

    python gate_pivotal_v9.py --cand asset_pivotal_k2.json --base-v8 v8merged_k2.json \
        --base-prop asset_base_newwording_k2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402

RNG = np.random.default_rng(0)


def rows(p):
    p = Path(p)
    if not p.is_absolute():
        p = C.RESULTS / "dcls" / p
    return {r["clip"]: r for r in json.loads(p.read_text())["rows"]}


def boot_ci(x, n=5000):
    """95% interval for the mean: the WIDER of the percentile bootstrap and the Student-t interval.
    With n = 4-16 clips the percentile bootstrap is too narrow, which would make R1/P2 lenient."""
    from scipy import stats
    x = np.asarray(x, float)
    m = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    blo, bhi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.inf
    return float(x.mean()), float(min(blo, x.mean() - half)), float(max(bhi, x.mean() + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base-v8", default="v8merged_k2.json")
    ap.add_argument("--base-prop")
    ap.add_argument("--gating", nargs="*", default=["R1", "R2", "P1", "P2"])
    a = ap.parse_args()
    cand, b8 = rows(a.cand), rows(a.base_v8)
    bp = rows(a.base_prop) if a.base_prop else {}
    out, ok = [], True
    gate = lambda bar, p: p or bar not in a.gating      # a non-gating bar never fails the gate

    for kind in ("expression", "motion"):
        cs = [c for c in cand if c in b8 and cand[c]["kind"] == kind and cand[c]["split"] == "holdout"]
        if not cs:
            out.append(f"R1 {kind:10s} no clips scored"); ok &= gate("R1", False); continue
        dm = np.array([cand[c]["margin"] - b8[c]["margin"] for c in cs])
        m, lo, hi = boot_ci(dm)
        drop = sum(b8[c]["rank"] == 1 for c in cs) - sum(cand[c]["rank"] == 1 for c in cs)
        p = lo >= -0.003 and drop <= 1
        ok &= gate("R1", p)
        out.append(f"R1{'' if 'R1' in a.gating else ' (report)'} {kind:10s} n={len(cs):2d} dmargin {m:+.4f} CI [{lo:+.4f}, {hi:+.4f}]  top-1 drop {drop:+d}  "
                   f"{'PASS' if p else 'FAIL'}")

    nt = len(next(iter(cand.values()))["loss_true"])
    for kind in ("expression", "motion"):                      # per kind, per t: pooling would hide a kind
        cs = [c for c in cand if c in b8 and cand[c]["kind"] == kind and cand[c]["split"] == "holdout"]
        if not cs:
            out.append(f"R2 {kind:10s} no clips scored"); ok &= gate("R2", False); continue
        for t in range(nt):
            r = float(np.exp(np.mean([np.log(cand[c]["loss_true"][t] / b8[c]["loss_true"][t]) for c in cs])))
            p = r <= 1.05
            ok &= gate("R2", p)
            out.append(f"R2 {kind:10s} t-index {t}  loss ratio x{r:.3f}  {'PASS' if p else 'FAIL'}")

    ph = [c for c in cand if c in bp and cand[c]["kind"] == "prop"]
    ho = [c for c in ph if cand[c]["split"] == "holdout"]
    if not ho:
        out.append("P1/P2 n/a (no phone clips scored)")
        print("\n".join(out)); print("GATE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    r1 = sum(sum(cand[c]["loss_true"]) for c in ho) / sum(sum(bp[c]["loss_true"]) for c in ho)
    p = r1 < 1.0
    ok &= gate("P1", p)
    out.append(f"P1 held-out phone loss ratio x{r1:.3f} (n={len(ho)})  {'PASS' if p else 'FAIL'}")

    g = []
    for c in ph:
        lab = cand[c]["label"]
        d = {k: np.log(cand[c]["E"][k] / bp[c]["E"][k]) for k in cand[c]["E"]}
        true = d.get(lab, d.get("phone"))
        g.append(true - np.mean([v for k, v in d.items() if k not in (lab, "phone")]))
    m, lo, hi = boot_ci(np.array(g))
    p = hi < 0
    ok &= gate("P2", p)
    out.append(f"P2 text-attributable gain {m:+.4f} CI [{lo:+.4f}, {hi:+.4f}] (n={len(g)})  {'PASS' if p else 'FAIL'}")

    print("\n".join(out))
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
