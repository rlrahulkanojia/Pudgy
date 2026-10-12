#!/usr/bin/env python3
"""PLAN §7.3 — the Tier-1 winner rule, applied to one run's checkpoints.

Reads EVAL/tier1/<run>@<step>.json (eval/dcls.py, INTERFACES §3) and the motion probe
EVAL/scores/<run>@<step>/probe.json (eval/score.py). The rule, fixed before training:

  1. eligible = checkpoints whose held-out losses beat the base on >= 15/16 expression
     and >= 6/7 motion clips (summary.<kind>.below_base, scaled as ceil(n*15/16) and
     ceil(n*6/7) if n differs); among them take the highest
     score = expression mean_margin + motion mean_margin;
  2. if several are within 0.001 of that score, take the EARLIEST;
  3. motion veto: if the pick's probe motion energy is < 0.8x the best probed
     checkpoint's, take the best-motion checkpoint within 0.002 of the top score.

No eligible checkpoint -> winner null, with the best-scoring checkpoint reported as
`best_ineligible` (the sanity gate in §6.3 failed; do not run Tier 2 on it silently).

    python expv2-minimax-h3/eval/select_winner.py --run expv2-h3-main
    python expv2-minimax-h3/eval/select_winner.py --run expv2-h3-d
-> EVAL/reports/winner.json  ({<run>: result}, merged across runs)
"""
import argparse
import datetime
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv2_common as C  # noqa: E402

TIE = 0.001
VETO_RATIO = 0.8
VETO_WINDOW = 0.002


def candidates(run):
    rows = []
    for p in sorted((C.EVAL / "tier1").glob(f"{run}@*.json")):
        name = p.stem
        step = name.split("@", 1)[1]
        if not step.isdigit():
            continue
        t1 = C.load_json(p, {})
        s = t1.get("summary", {})
        e, m = s.get("expression", {}), s.get("motion", {})
        if e.get("mean_margin") is None or m.get("mean_margin") is None:
            print(f"  skip {name}: summary incomplete")
            continue
        probe = C.load_json(C.EVAL / "scores" / name / "probe.json", {}) or {}
        rows.append({"ckpt": name, "opt_step": int(step),
                     "score": e["mean_margin"] + m["mean_margin"],
                     "expr_margin": e["mean_margin"], "motion_margin": m["mean_margin"],
                     "expr_acc": e.get("acc"), "motion_acc": m.get("acc"),
                     "expr_below_base": e.get("below_base"), "expr_n": e.get("n"),
                     "motion_below_base": m.get("below_base"), "motion_n": m.get("n"),
                     "probe_motion_energy": (probe.get("summary") or {}).get("motion_energy_median")})
    return sorted(rows, key=lambda r: r["opt_step"])


def eligible(r):
    ne, nm = r["expr_n"] or 16, r["motion_n"] or 7
    be, bm = r["expr_below_base"], r["motion_below_base"]
    if be is None or bm is None:
        return False
    return be >= math.ceil(ne * 15 / 16) and bm >= math.ceil(nm * 6 / 7)


def choose(rows):
    elig = [r for r in rows if eligible(r)]
    out = {"n_checkpoints": len(rows), "n_eligible": len(elig), "winner": None,
           "steps": [], "veto": None}
    if not elig:
        if rows:
            out["best_ineligible"] = max(rows, key=lambda r: r["score"])["ckpt"]
        return out
    top = max(r["score"] for r in elig)
    tied = [r for r in elig if r["score"] >= top - TIE]
    pick = min(tied, key=lambda r: r["opt_step"])
    out["steps"].append(f"rule 1-2: top score {top:.5f}; {len(tied)} within {TIE}; earliest = {pick['ckpt']}")
    probed = [r for r in rows if r["probe_motion_energy"] is not None]
    if probed and pick["probe_motion_energy"] is not None:
        best_motion = max(r["probe_motion_energy"] for r in probed)
        ratio = pick["probe_motion_energy"] / best_motion if best_motion else None
        out["veto"] = {"pick_motion": pick["probe_motion_energy"], "best_probed_motion": best_motion,
                       "ratio": ratio, "fired": bool(ratio is not None and ratio < VETO_RATIO)}
        if out["veto"]["fired"]:
            near = [r for r in elig if r["score"] >= top - VETO_WINDOW and r["probe_motion_energy"] is not None]
            if near:
                pick = max(near, key=lambda r: r["probe_motion_energy"])
                out["steps"].append(f"rule 3: veto fired (ratio {ratio:.3f} < {VETO_RATIO}); "
                                    f"best motion within {VETO_WINDOW} = {pick['ckpt']}")
            else:
                out["steps"].append("rule 3: veto fired but no probed checkpoint within the window; pick kept")
    elif not probed:
        out["steps"].append("rule 3: no probe scores - veto not evaluated (run score.py --suites probe)")
    out["winner"] = pick["ckpt"]
    out["winner_row"] = pick
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    rows = candidates(a.run)
    if not rows:
        sys.exit(f"no tier1/{a.run}@<step>.json files under {C.EVAL / 'tier1'}")
    res = choose(rows)
    res.update({"run": a.run, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "table": rows})
    dest = C.EVAL / "reports" / "winner.json"
    allres = C.load_json(dest, {}) or {}
    allres[a.run] = res
    C.write_json(dest, allres)
    for r in rows:
        flag = "E" if eligible(r) else " "
        print(f"  {flag} {r['ckpt']:24s} score {r['score']:+.5f}  expr {r['expr_below_base']}/{r['expr_n']}  "
              f"motion {r['motion_below_base']}/{r['motion_n']}  probe {r['probe_motion_energy']}")
    print("\n".join(res["steps"]))
    print(f"winner[{a.run}] = {res['winner']}  -> {dest}")


if __name__ == "__main__":
    main()
