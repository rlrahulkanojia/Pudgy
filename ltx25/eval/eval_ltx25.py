#!/usr/bin/env python3
"""
eval_ltx25.py - evaluation harness for the standalone LTX-2.5 experiment.

STANDALONE. Shares nothing with the v1-v7 eval tooling.

THE ONE IDEA THIS TOOL EXISTS FOR
---------------------------------
The stated success bar is "as close to the training video as possible". With I2V and a
supplied start frame, a model that simply REPLAYS the training clip satisfies that bar
perfectly. A previous generation of this project measured exactly that failure: 0.8 px of
motion when started from a training frame, against 221-267 px from a novel one.

So closeness to training footage is only meaningful on start frames the model has never
seen, and the gap between those two conditions is the headline number. `probe` measures it.

A second trap this tool guards against: the client's own footage contains genuine frozen
holds (LAUGHING holds ~9 frames mid-clip, CRYING's last 8 frames are identical, JUMPING
ends on 4). A naive "frozen frames are bad" threshold would score the client's own
animation as defective. `baseline` measures the source first so the threshold is earned.

Subcommands
-----------
  baseline     measure the SOURCE corpus (frozen %, motion, per label) -> baseline.json
  startframes  emit the I2V pose library + an unseen-frame eval set
  score        score generated clips against the baseline, per label and character
  probe        memorisation probe: training start frames vs unseen ones

Nothing here runs a model. Generation is done with ltx-pipelines (NOT the trainer's own
validation sampler - see PIPELINE.md 10.3 for why); this tool measures what comes out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GRAY = 160          # analysis resolution; structural measures do not need full res
FROZEN_EPS = 0.5    # mean abs 8-bit difference below which a frame pair is "frozen"


# --------------------------------------------------------------------------- decoding

def read_gray(path: Path, size: int = GRAY) -> np.ndarray | None:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = size * size
    c = len(out) // n
    if c == 0:
        return None
    return np.frombuffer(out[:c * n], dtype=np.uint8).reshape(c, size, size).astype(np.float32)


def read_rgba(path: Path, size: int = GRAY) -> np.ndarray | None:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={size}:{size},format=rgba", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = size * size * 4
    c = len(out) // n
    if c == 0:
        return None
    return np.frombuffer(out[:c * n], dtype=np.uint8).reshape(c, size, size, 4)


# --------------------------------------------------------------------------- measures

def measures(g: np.ndarray) -> dict:
    """Structural motion measures on a grayscale clip.

    centroid_travel is the proxy for "did anything actually happen": the summed
    displacement of the frame's intensity centroid, scaled to a 1024px frame so the
    numbers are comparable with the 0.8px-vs-221px figure that motivated this tool.
    """
    if len(g) < 2:
        return {"frames": int(len(g)), "motion_energy": 0.0, "centroid_travel": 0.0,
                "frozen_pct": 100.0, "max_step": 0.0}

    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    ys, xs = np.mgrid[0:g.shape[1], 0:g.shape[2]]
    dev = np.abs(g - g.mean())
    w = dev.sum(axis=(1, 2)) + 1e-6
    cy = (dev * ys).sum(axis=(1, 2)) / w
    cx = (dev * xs).sum(axis=(1, 2)) / w
    scale = 1024.0 / g.shape[1]
    travel = float(np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2).sum() * scale)

    return {
        "frames": int(len(g)),
        "motion_energy": round(float(d.mean()), 3),
        "centroid_travel": round(travel, 2),
        "frozen_pct": round(100.0 * float((d < FROZEN_EPS).sum()) / len(d), 1),
        "max_step": round(float(d.max()), 2),
    }


# -------------------------------------------------------------------------- baseline

def cmd_baseline(args) -> None:
    raw = args.raw
    clips = sorted(raw.rglob("*.mov"))
    if not clips:
        sys.exit(f"no .mov under {raw}")

    per_label: dict[str, list[dict]] = defaultdict(list)
    print(f"measuring {len(clips)} source clips")
    for i, c in enumerate(clips, 1):
        g = read_gray(c)
        if g is None:
            continue
        label = c.parent.parent.name.upper().replace("MOTION_", "")
        per_label[label].append(measures(g))
        if i % 40 == 0:
            print(f"  {i}/{len(clips)}")

    summary = {}
    for label, rows in sorted(per_label.items()):
        summary[label] = {
            "n": len(rows),
            "frozen_pct_median": round(float(np.median([r["frozen_pct"] for r in rows])), 1),
            "frozen_pct_max": round(max(r["frozen_pct"] for r in rows), 1),
            "centroid_travel_median": round(float(np.median([r["centroid_travel"] for r in rows])), 2),
            "motion_energy_median": round(float(np.median([r["motion_energy"] for r in rows])), 3),
        }

    allrows = [r for rows in per_label.values() for r in rows]
    out = {
        "source": str(raw),
        "clips": len(allrows),
        "note": ("Thresholds for generated clips must be set RELATIVE to these numbers. "
                 "The client's own animation contains deliberate frozen holds; scoring "
                 "generated output against 0% frozen would fail the source itself."),
        "corpus": {
            "frozen_pct_median": round(float(np.median([r["frozen_pct"] for r in allrows])), 1),
            "frozen_pct_p90": round(float(np.percentile([r["frozen_pct"] for r in allrows], 90)), 1),
            "centroid_travel_median": round(float(np.median([r["centroid_travel"] for r in allrows])), 2),
        },
        "per_label": summary,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\ncorpus frozen%: median {out['corpus']['frozen_pct_median']}, "
          f"p90 {out['corpus']['frozen_pct_p90']}")
    print(f"corpus centroid travel: median {out['corpus']['centroid_travel_median']} px")
    print("\nper label:")
    for k, v in summary.items():
        print(f"  {k:<12} n={v['n']:>3}  frozen%={v['frozen_pct_median']:>5} "
              f"(max {v['frozen_pct_max']:>5})  travel={v['centroid_travel_median']:>7} px")
    print(f"\n-> {args.out}")


# ----------------------------------------------------------------------- startframes

def cmd_startframes(args) -> None:
    """Emit the I2V pose library, and an eval set the model will never have trained on.

    PIPELINE.md 6: because ladder rungs are head-truncations, frame 0 is shared across
    every rung of a source, so an unseen start frame must come from a held-out angle or a
    mid-clip time offset. Both sets are written separately and must never be mixed.
    """
    ds = json.loads((args.dataset / "manifest.json").read_text())
    hold_angle = ds["holdout_angle"]

    prod = args.out / "pose_library"
    unseen = args.out / "unseen"
    prod.mkdir(parents=True, exist_ok=True)
    unseen.mkdir(parents=True, exist_ok=True)

    clips_dir = args.dataset / "clips"
    n_prod = n_unseen = 0
    for rec in ds["records"]:
        src = clips_dir / rec["file"]
        if not src.exists():
            continue
        held = rec["angle"] == hold_angle
        # Production frames come from frame 0 (what a user would supply). Eval frames come
        # from a mid-clip offset of a HELD-OUT angle, so neither the pose nor the viewpoint
        # was trained on.
        if held:
            if n_unseen >= args.n:
                continue
            idx, dest, n_unseen = rec["frames"] // 2, unseen, n_unseen + 1
        else:
            if n_prod >= args.n:
                continue
            idx, dest, n_prod = 0, prod, n_prod + 1
        stem = Path(rec["file"]).stem
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", str(dest / f"{stem}_f{idx}.png")],
            check=False)

    print(f"pose_library (frame 0, trained angles): {n_prod} -> {prod}")
    print(f"unseen (mid-clip, {hold_angle} only):   {n_unseen} -> {unseen}")
    print("\nScore ONLY on generations from `unseen/`. Generations from `pose_library/`")
    print("are the memorisation probe's control condition, never the score.")


# ------------------------------------------------------------------------------ score

def cmd_score(args) -> None:
    base = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    gens = sorted(list(args.gen.rglob("*.mp4")) + list(args.gen.rglob("*.gif")))
    if not gens:
        sys.exit(f"no clips under {args.gen}")

    rows = []
    for c in gens:
        g = read_gray(c)
        if g is None:
            continue
        m = measures(g)
        m["file"] = c.name
        # label inferred from filename when it follows the prep naming convention
        parts = c.stem.split("_")
        m["label"] = parts[1].upper() if len(parts) > 1 else "?"
        m["character"] = parts[0].upper() if parts else "?"
        rows.append(m)

    print(f"scored {len(rows)} generated clips\n")
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)

    print(f"{'label':<12}{'n':>4}{'frozen%':>10}{'travel px':>12}{'vs source':>26}")
    for label, rs in sorted(by.items()):
        fz = float(np.median([r["frozen_pct"] for r in rs]))
        tv = float(np.median([r["centroid_travel"] for r in rs]))
        note = ""
        if base and label in base.get("per_label", {}):
            b = base["per_label"][label]
            dt = tv / b["centroid_travel_median"] if b["centroid_travel_median"] else 0
            note = (f"travel {dt:.2f}x source"
                    + ("  <<< SUSPICIOUS" if dt < 0.25 else ""))
        print(f"{label:<12}{len(rs):>4}{fz:>10.1f}{tv:>12.1f}{note:>26}")

    if base:
        print(f"\nsource corpus frozen% median was {base['corpus']['frozen_pct_median']} "
              f"(p90 {base['corpus']['frozen_pct_p90']}) - judge against that, not against 0.")
    print("\n⚠️  These are STRUCTURAL measures only. No metric here sees identity.")
    print("    A clip can score perfectly and be completely off-model. Eye-check every")
    print("    result before drawing a conclusion.")


# ------------------------------------------------------------------------------ probe

def cmd_probe(args) -> None:
    """The headline number: does motion collapse when started from a TRAINING frame?"""
    def collect(d: Path) -> list[dict]:
        out = []
        for c in sorted(list(d.rglob("*.mp4")) + list(d.rglob("*.gif"))):
            g = read_gray(c)
            if g is not None:
                out.append(measures(g))
        return out

    trained = collect(args.trained)
    unseen = collect(args.unseen)
    if not trained or not unseen:
        sys.exit("need clips in BOTH --trained and --unseen")

    tt = float(np.median([r["centroid_travel"] for r in trained]))
    tu = float(np.median([r["centroid_travel"] for r in unseen]))
    ft = float(np.median([r["frozen_pct"] for r in trained]))
    fu = float(np.median([r["frozen_pct"] for r in unseen]))

    print("MEMORISATION PROBE")
    print(f"  from TRAINING start frames  n={len(trained):<4} travel={tt:8.2f} px  frozen={ft:5.1f}%")
    print(f"  from UNSEEN  start frames   n={len(unseen):<4} travel={tu:8.2f} px  frozen={fu:5.1f}%")

    ratio = tu / tt if tt > 1e-6 else float("inf")
    print(f"\n  unseen / trained travel ratio = {ratio:.2f}x")
    if ratio > 4.0:
        print("\n  ❌ FAIL. Motion collapses when the model starts from a frame it trained on.")
        print("     This is the failure a previous generation measured (0.8 px vs 221-267 px).")
        print("     The corpus is teaching start-frame recall, not action. Fix the ladder")
        print("     before adding data - more data will not help.")
    elif ratio > 1.8:
        print("\n  ⚠️  MARGINAL. Some start-frame dependence. Inspect per label before trusting.")
    else:
        print("\n  ✅ PASS. Motion is comparable from seen and unseen start frames, so the")
        print("     score on unseen frames reflects behaviour rather than recall.")


# ------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    RAW = Path.home() / "Documents/Projects/Saksham/Pudgy/Data/raw/iteration_4/LSLTTT-Project"
    DS = Path.home() / "Documents/Projects/Saksham/Pudgy/Data/processed/ltx25_experiment"

    b = sub.add_parser("baseline", help="measure the source corpus")
    b.add_argument("--raw", type=Path, default=RAW)
    b.add_argument("--out", type=Path, default=Path("ltx25/eval/baseline.json"))
    b.set_defaults(func=cmd_baseline)

    s = sub.add_parser("startframes", help="emit the pose library and unseen eval frames")
    s.add_argument("--dataset", type=Path, default=DS)
    s.add_argument("--out", type=Path, default=DS / "start_frames")
    s.add_argument("--n", type=int, default=120)
    s.set_defaults(func=cmd_startframes)

    c = sub.add_parser("score", help="score generated clips against the source baseline")
    c.add_argument("--gen", type=Path, required=True)
    c.add_argument("--baseline", type=Path, default=Path("ltx25/eval/baseline.json"))
    c.set_defaults(func=cmd_score)

    p = sub.add_parser("probe", help="memorisation probe")
    p.add_argument("--trained", type=Path, required=True, help="generations from TRAINING frames")
    p.add_argument("--unseen", type=Path, required=True, help="generations from UNSEEN frames")
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
