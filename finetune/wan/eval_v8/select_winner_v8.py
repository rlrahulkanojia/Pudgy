#!/usr/bin/env python3
"""Pick the v8 checkpoint to take into Tier 2, by a rule fixed before the results exist.

Rule (holdout split, dcls k=2, paired noise):
  1. score = expression margin + motion margin   (margins are the label-control signal;
     accuracy moves in 0.06/0.10 steps at n = 16/10 and is not used)
  2. eligible only if score > v7fixed's score AND both kind losses < v7fixed's
  3. ties (score within 0.001) go to the lower mean loss; if still tied, an SWA is
     preferred over a single checkpoint, since averaging removes random-walk noise
Prints the winning config name (e.g. "v8@swa400-600") and writes eval_v8/winner.json.
Exit 1 if nothing is eligible.
"""
import json
import sys
from pathlib import Path

D = Path("/workspace/eval_v8/dcls")


def summary(cfg):
    p = D / f"{cfg}_k2.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    e, m = s.get("holdout/expression"), s.get("holdout/motion")
    if not e or not m:
        return None
    return {"config": cfg, "score": round(e["margin"] + m["margin"], 5),
            "loss": round((e["loss"] + m["loss"]) / 2, 6), "expr": e, "motion": m}


def main(cands=None):
    base = summary("v7fixed")
    if base is None:
        sys.exit("v7fixed baseline missing")
    cands = cands or sorted(p.name[:-8] for p in D.glob("v8@*_k2.json"))
    rows = [r for r in (summary(c) for c in cands) if r]
    elig = [r for r in rows if r["score"] > base["score"]
            and r["expr"]["loss"] < base["expr"]["loss"] and r["motion"]["loss"] < base["motion"]["loss"]]
    for r in sorted(rows, key=lambda r: -r["score"]):
        flag = "eligible" if r in elig else "not eligible"
        print(f"  {r['config']:<16} score {r['score']:+.4f}  loss {r['loss']:.5f}  ({flag})", file=sys.stderr)
    print(f"  {'v7fixed (bar)':<16} score {base['score']:+.4f}  loss {base['loss']:.5f}", file=sys.stderr)
    if not elig:
        sys.exit(1)
    best = max(r["score"] for r in elig)
    tied = [r for r in elig if best - r["score"] <= 0.001]
    tied.sort(key=lambda r: (r["loss"], 0 if "swa" in r["config"] else 1))
    win = tied[0]
    Path("/workspace/eval_v8/winner.json").write_text(json.dumps(
        {"winner": win["config"], "score": win["score"], "loss": win["loss"],
         "bar_v7fixed": {"score": base["score"], "loss": base["loss"]},
         "candidates": rows}, indent=1))
    print(win["config"])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
