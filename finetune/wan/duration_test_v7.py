#!/usr/bin/env python3
"""Does a motion action survive past the length it was trained at?

Every motion label has a hard ceiling set by the client's delivered footage — the plan
refuses to loop-tile training clips to fake longer ones — so the longest motion rung is
33 frames (1.38 s). This asks for 49 frames (2.04 s) and measures what happens.

`walking` is excluded: its only rung is 13 frames, so 49 would be a 3.7x extrapolation
and would tell us nothing a gentler stretch does not. `standing_idle` is excluded too —
it is DERIVED from frozen neutral footage (plan 3.1), so "it does not move" is the known
starting condition, not a finding.

Three measurements per clip:
  length_ok     the generator returned the frames we asked for
  animates      mean frame-to-frame MAE vs a generated static reference (0.0049)
  tail_ratio    animation in the LAST third vs the FIRST third. This is the failure this
                test exists to catch: an action that completes inside its trained length
                and then FREEZES scores fine on a whole-clip average while being useless.
                ~1.0 = sustained; << 1.0 = ran out of action.

    python duration_test_v7.py --frames 49 --seeds 42
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_v7 as g
import sweep_v7 as sw

CKPT = Path("/workspace/wan_output/pudgy-v7-motion-lownoise/pudgy-v7-motion-lownoise.safetensors")
STATIC_MAE = 0.0049          # generated neutral-expression clip, measured
ACTIONS = ["jumping", "waving", "sitting", "running"]


def mae(v, lo=None, hi=None):
    f = v.astype(np.float32) / 255.0
    f = f[lo:hi]
    return float(np.mean(np.abs(f[1:] - f[:-1]))) if len(f) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=49, help="stretched length (4N+1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--chars", nargs="+", default=["Pax"])
    ap.add_argument("--actions", nargs="*", default=ACTIONS)
    ap.add_argument("--outroot", default="/workspace/eval_v7/duration")
    a = ap.parse_args()
    if (a.frames - 1) % 4:
        sys.exit(f"--frames must be 4N+1; {a.frames} is not")

    out = Path(a.outroot); out.mkdir(parents=True, exist_ok=True)
    lines, index = [], []
    for char in a.chars:
        start = g.start_frame(char, "motion")
        for lab in a.actions:
            natural = max(g.ladder_for("motion", lab))
            for n in sorted({natural, a.frames}):
                for seed in a.seeds:
                    # No "opening frames only" suffix above the natural length — that
                    # marker means TRUNCATED, and asking for more is the opposite.
                    cap = g.caption_for(char, "motion", lab, nframes=n)
                    lines.append(f"{cap} --i {start} --f {n} --d {seed}")
                    index.append({"character": char, "label": lab, "frames": n,
                                  "seed": seed, "natural": natural,
                                  "stretch": round(n / natural, 2)})
    pf = out / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")
    print(f"== duration test: {len(index)} generations, 1 model load", flush=True)

    clips = sw.generate_batch(CKPT, pf, out, side="low")
    named = []
    for clip, meta in zip(clips, index):
        dest = clip.with_name(f"{meta['character'].lower()}_{meta['label']}_f{meta['frames']}_s{meta['seed']}.mp4")
        if dest != clip and not dest.exists():
            clip.rename(dest)
        named.append(dest)

    rows = []
    for p, meta in zip(named, index):
        v = g.read_video(p)
        t = len(v)
        third = max(2, t // 3)
        head, tail = mae(v, 0, third), mae(v, t - third, t)
        rows.append({**meta, "produced": t, "length_ok": t == meta["frames"],
                     "frame_mae": round(mae(v), 4),
                     "animates": mae(v) > STATIC_MAE * 2,
                     "head_mae": round(head, 4), "tail_mae": round(tail, 4),
                     "tail_ratio": round(tail / head, 2) if head > 1e-6 else 0.0})
    res = {"gate": "duration-probe", "checkpoint": CKPT.name,
           "static_reference_mae": STATIC_MAE, "rows": rows}
    (out / "duration_result.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'action':10s} {'frames':>7s} {'stretch':>8s} {'len_ok':>7s} {'mae':>7s} {'tail/head':>10s}")
    for r in rows:
        print(f"  {r['label']:8s} {r['produced']:7d} {r['stretch']:7.2f}x {str(r['length_ok']):>7s} "
              f"{r['frame_mae']:7.4f} {r['tail_ratio']:9.2f}")
    stretched = [r for r in rows if r["stretch"] > 1.0]
    if stretched:
        held = [r for r in stretched if r["animates"] and r["tail_ratio"] > 0.6]
        print(f"\n  {len(held)}/{len(stretched)} stretched clips still animating at the tail "
              f"(ratio > 0.6)")


if __name__ == "__main__":
    main()
