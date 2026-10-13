#!/usr/bin/env python3
"""PLAN §7 winner rule, from Tier-1 rows (eval/dcls.py) and the motion probe
(eval/motion_probe.py -> EVAL/reports/motion_probe.json).

  1. Among checkpoints whose expression AND motion held-out true-caption losses beat the base
     on >= 90% of clips, take the highest expression + motion mean margin.
  2. If several are within 0.001 of the top, take the EARLIEST.
  3. Motion veto: if that checkpoint's probed median motion energy is < 0.8x the best probed
     checkpoint's, take the best-motion probed checkpoint within 0.002 of the top score.
     (A pick that was not probed must be probed first: the script stops and says so.)

    python eval/winner.py
Writes EVAL/reports/winner.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv1_common as C  # noqa: E402

DCLS = C.EVAL / "dcls"


def holdout(cfg):
    d = C.load_json(DCLS / f"{cfg}__A_k2.json")
    return {r["clip"]: r for r in d["rows"] if r["split"] == "holdout"} if d else None


def main():
    base = holdout("base")
    probe = C.load_json(C.EVAL / "reports" / "motion_probe.json")
    if not probe:
        sys.exit("run eval/motion_probe.py first")
    energy = {c: v["median_energy"] for c, v in probe["configs"].items() if c.startswith("expv1@")}
    rows = []
    for p in sorted(DCLS.glob("expv1@*__A_k2.json"), key=lambda p: int(p.name.split("@")[1].split("__")[0])):
        cfg = p.name.split("__")[0]
        ck = holdout(cfg)
        r = {"config": cfg, "step": int(cfg.split("@")[1]), "energy": energy.get(cfg)}
        ok, score = True, 0.0
        for kind in ("expression", "motion"):
            rs = [x for x in ck.values() if x["kind"] == kind]
            frac = float(np.mean([np.mean(x["loss_true"]) < np.mean(base[x["clip"]]["loss_true"]) for x in rs]))
            m = float(np.mean([x["margin"] for x in rs]))
            acc = float(np.mean([x["pred"] == x["label"] for x in rs]))
            r[kind] = {"frac_below_base": frac, "margin": m, "acc": acc}
            ok &= frac >= 0.9
            score += m
        r.update(eligible=ok, score=score)
        rows.append(r)
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        sys.exit("no checkpoint beats the base on >= 90% of held-out clips in both kinds")
    top = max(r["score"] for r in elig)
    pick = min((r for r in elig if r["score"] >= top - 0.001), key=lambda r: r["step"])
    best_e = max(energy.values())
    steps = [f"step 1-2: {pick['config']} (score {pick['score']:+.4f}; top {top:+.4f})"]
    if pick["energy"] is None:
        sys.exit(f"{pick['config']} was not probed - add it to scripts/run_probe.sh and re-run")
    veto = pick["energy"] < 0.8 * best_e
    if veto:
        near = [r for r in elig if r["score"] >= top - 0.002]
        unprobed = [r["config"] for r in near if r["energy"] is None]
        if unprobed:
            sys.exit(f"motion veto applies but {unprobed} (within 0.002) were not probed")
        new = max(near, key=lambda r: r["energy"])
        steps.append(f"step 3: VETO - energy {pick['energy']:.3f} < 0.8 x best {best_e:.3f}; "
                     f"best-motion within 0.002 of top: {new['config']} ({new['energy']:.3f})")
        pick = new
    else:
        steps.append(f"step 3: no veto - energy {pick['energy']:.3f} >= 0.8 x best {best_e:.3f}")
    out = {"winner": pick["config"], "trace": steps, "best_probed_energy": best_e,
           "d1_triggered": probe["d1"]["triggered"], "rows": rows}
    C.write_json(C.EVAL / "reports" / "winner.json", out)
    print(f"{'config':<11} {'elig':<5} {'score':>8}  {'expr acc/margin':>16}  {'motion acc/margin':>17}  energy")
    for r in rows:
        e = f"{r['energy']:.3f}" if r["energy"] is not None else "-"
        print(f"{r['config']:<11} {str(r['eligible']):<5} {r['score']:+.4f}  "
              f"{r['expression']['acc']:.3f}/{r['expression']['margin']:+.4f}  "
              f"   {r['motion']['acc']:.3f}/{r['motion']['margin']:+.4f}  {e}")
    print("\n".join(steps))
    print(f"-> WINNER: {pick['config']}   (D1 arm-B trigger: {'YES' if out['d1_triggered'] else 'no'})")


if __name__ == "__main__":
    main()
