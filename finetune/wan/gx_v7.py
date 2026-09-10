#!/usr/bin/env python3
"""G-X — the motion expert A/B, batched (one model load per arm, not per clip).

Decides plan section 4: motion on the HIGH-noise expert (evidence #2: motion is global
and temporal, and high-noise already carries exactly that) or on the LOW-noise expert
(v5's only demonstrated-safe expert). v5's negative was expression-on-high-noise, which
is a different proposition — so this has never been tested and is a measurement, not an
assumption.

Pass = the arm whose action outputs stay distinguishable (pairwise SSIM < 0.95 across
all pairs) AND which retains prompt response (subject x-range > 200 px). v5 calibration:
0.9692 = prompt ignored, 0.9340 = prompt works, 0.8 px = motion collapse.

    python gx_v7.py --seeds 42 43 44
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_v7 as g
import sweep_v7 as sw

ARMS = {
    "M-high": ("high", Path("/workspace/wan_output/pudgy-v7-motion-highnoise/pudgy-v7-motion-highnoise.safetensors")),
    "M-low":  ("low",  Path("/workspace/wan_output/pudgy-v7-motion-lownoise/pudgy-v7-motion-lownoise.safetensors")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--outroot", default="/workspace/eval_v7/gx")
    a = ap.parse_args()
    # standing_idle is excluded: it is the counter-class (the character should NOT move),
    # so including it in a "do the actions differ" matrix would reward the wrong thing.
    labels = a.labels or [l for l in g.MOTION if l != "standing_idle"]

    verdict = {}
    for arm, (side, ckpt) in ARMS.items():
        if not ckpt.exists():
            sys.exit(f"missing {arm} checkpoint: {ckpt}")
        out = Path(a.outroot) / arm
        out.mkdir(parents=True, exist_ok=True)
        pf = out / "prompts.txt"
        index = sw.build_prompt_file(labels, a.seeds, "motion", pf)
        print(f"== {arm} ({side}-noise): {len(index)} generations, 1 model load", flush=True)
        clips = sw.generate_batch(ckpt, pf, out, side=side)
        if len(clips) == len(index):
            clips = sw.rename_clips(clips, index)
        by = {(i["character"], i["label"], i["seed"]): c for i, c in zip(index, clips)}

        rows, xr = [], []
        for char in ("Pax", "Polly"):
            for seed in a.seeds:
                for x, y in itertools.combinations(labels, 2):
                    kx, ky = (char, x, seed), (char, y, seed)
                    if kx not in by or ky not in by:
                        continue
                    s = g.ssim_pair(g.read_video(by[kx]), g.read_video(by[ky]))
                    rows.append({"character": char, "seed": seed, "pair": f"{x}|{y}",
                                 "ssim": round(s, 4), "distinct": s < g.WHOLE_DISTINCT})
        for k, p in by.items():
            xr.append(g.subject_stats_flat(g.read_video(p), g.bg_rgb("white"))["x_range_px"])
        verdict[arm] = {
            "expert": side, "checkpoint": ckpt.name, "n_clips": len(clips),
            "rows": rows,
            "all_distinct": bool(rows) and all(r["distinct"] for r in rows),
            "worst_ssim": round(max(r["ssim"] for r in rows), 4) if rows else None,
            "mean_ssim": round(float(np.mean([r["ssim"] for r in rows])), 4) if rows else None,
            "mean_x_range_px": round(float(np.mean(xr)), 1) if xr else 0.0,
            "retains_motion": bool(xr) and float(np.mean(xr)) > g.XRANGE_NOVEL,
        }
        v = verdict[arm]
        print(f"   mean_ssim={v['mean_ssim']} worst={v['worst_ssim']} "
              f"x_range={v['mean_x_range_px']}px distinct={v['all_distinct']} "
              f"motion={v['retains_motion']}", flush=True)

    winners = [k for k, v in verdict.items() if v["all_distinct"] and v["retains_motion"]]
    decision = (winners[0] if len(winners) == 1 else
                min(winners, key=lambda k: verdict[k]["worst_ssim"]) if winners else None)
    res = {"gate": "G-X", "arms": verdict, "winners": winners, "decision": decision,
           "pass": bool(winners),
           "note": ("If M-high wins, BOTH experts end up trained and v5's frozen-partner "
                    "protection is gone — G-S then becomes mandatory (plan section 4).")}
    Path(a.outroot, "gx_result.json").write_text(json.dumps(res, indent=2))
    print("\n== G-X ==")
    for k, v in verdict.items():
        print(f"  {k:7s} ssim(mean/worst)={v['mean_ssim']}/{v['worst_ssim']}  "
              f"x_range={v['mean_x_range_px']}px  distinct={v['all_distinct']}  motion={v['retains_motion']}")
    print(f"  -> decision: {decision or 'NO ARM PASSES'}")


if __name__ == "__main__":
    main()
