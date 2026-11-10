#!/usr/bin/env python3
"""Phase 0 on GPU 1 (Training_Approach_v10.md section 5.0, Q18): the baseline first, then #1 while
Stage 1 trains. Thin orchestration of suite_v10 / g4_v10; every step is resumable, and --max-new N
spreads at most N new clips over the steps in order, then exits 0 with "REMAINING <k>".

  p2  Phase 0 #2, the "before" for G1, G2 and G4: v8merged (+ golden + E5) on
        - the G1/G2 jobs: 34 jobs x seeds 42/43/44 at 480x848 (102 I2V clips) + the 8 FLF clips
        - the v8 suites: core 52 + compose 18 at 1024^2
      then, once all 180 clips exist: G2 scores and sheets for v8merged, G4 scores for v8merged.
  p1  Phase 0 #1 (informs Stage 2 only): v9b@100 and v9b@200 vs v8merged on
        - the 8-job v9b scene suite (v9b's 6 held-out skits, f49 from the span start, seeds 42/43/44,
          480x848; the same jobs as eval_v9b/suite_v9b.py, rebuilt into the eval set; E5 on, a no-op
          here: no v9b held-out caption names the phone) - v8merged's clips are p2's, shared
        - core 52 (1024^2)
      then the G2 scores, sheets (real / v8merged / v9b@100 / v9b@200) and runs/<RUN>/phase0/p1.json.
  report   rewrite p1.json and print its table from what exists.

The stock suite_v9b route (`cd eval_v9b && RUN_NAME=pudgy-v9b-lownoise python suite_v9b.py --configs
v8merged v9b@100 v9b@200`) needs /workspace/eval_v9b from eval_for_v9b.tar and has no --max-new; this
one needs neither. v9b checkpoints: Azure v9b/weights/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000{100,200}
-> /workspace/wan_output/pudgy-v9b-lownoise/ (V9B_RUN_DIR).

Box (CUDA_VISIBLE_DEVICES=1 by default):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY phase0_v10.py p2 [--max-new N]     # ~12-14 h (G1/G2, est.) + ~5-6 h (G4 at 1024^2, est.)
    $PY phase0_v10.py p1 [--max-new N]     # 48 scene clips (~4-5 h) + 104 core clips (~8 h), est.
    $PY phase0_v10.py report
Outputs: runs/$RUN_NAME/suite/... and runs/$RUN_NAME/g4/... (as suite_v10 / g4_v10), runs/$RUN_NAME/phase0/p1.json
Exit codes: 0 (last line "REMAINING <k>"; 0 = this phase is complete and scored).
"""
import argparse
import json
import sys

import common_v10 as C  # noqa: E402  (first: CUDA_VISIBLE_DEVICES)
import numpy as np
import g4_v10 as G4  # noqa: E402
import suite_v10 as SU  # noqa: E402

V9B_CONFIGS = ("v9b@100", "v9b@200")


def run_steps(steps, max_new):
    """steps: [(callable(max_new) -> remaining)]; spends max_new across them in order."""
    left = 0
    for fn in steps:
        if max_new is not None and max_new <= 0:
            left += fn(0)                     # counts what is still to do, generates nothing
            continue
        before = fn(0)
        rem = fn(max_new)
        if max_new is not None:
            max_new -= before - rem
        left += rem
    return left


def p2(max_new=None):
    left = run_steps([lambda n: SU.gen("v8merged", max_new=n),
                      lambda n: G4.gen("v8merged", max_new=n)], max_new)
    if not left:
        SU.score(["v8merged"]); SU.sheets(["v8merged"])
        G4.score(["v8merged"])
    return left


def p1(max_new=None):
    steps = [lambda n: SU.gen("v8merged", only="v9b", flf=False, max_new=n)]
    steps += [lambda n, c=c: SU.gen(c, only="v9b", flf=False, max_new=n) for c in V9B_CONFIGS]
    steps += [lambda n, c=c: G4.gen(c, suites=("core",), max_new=n) for c in ("v8merged",) + V9B_CONFIGS]
    left = run_steps(steps, max_new)
    if not left:
        cfgs = ["v8merged", *V9B_CONFIGS]
        SU.score(cfgs, only="v9b", flf=False); SU.sheets(cfgs, only="v9b")
        G4.score(cfgs, suites=("core",))
        report()
    return left


def report():
    scores = C.load_json(SU.ROOT / "scores.json", {})
    g4 = C.load_json(G4.ROOT / "scores.json", {}).get("core", {})
    jobs = SU.load_jobs("v9b")
    out = {"jobs": [j["tag"] for j in jobs], "configs": {}}
    for cfg in ("v8merged", *V9B_CONFIGS):
        rs = [scores.get(f"{cfg}|{j['tag']}_s{s}") for j in jobs for s in C.SEEDS]
        rs = [r for r in rs if r]
        row = {"clips": len(rs)}
        if rs:
            ids = [r["identity"] for r in rs if r["identity"] is not None]
            row.update(dup_flagged=sum(r["dup_flag"] for r in rs),
                       zoom_over=sum(1 for r in rs if r["zoom"] and r["zoom"]["verdict"] == "zoom"),
                       motion_median=round(float(np.median([r["motion"] for r in rs])), 3),
                       identity_mean=round(float(np.mean(ids)), 4) if ids else None,
                       ocr_fail=sum(r["ocr"]["verdict"] == "fail" for r in rs),
                       frozen_over_035=sum(r["frozen"] > 0.35 for r in rs), jump_over_8=sum(r["jump"] > 8 for r in rs),
                       drift_mean=round(float(np.mean([r["drift"] for r in rs])), 2))
        if cfg in g4:
            row["core_label_nn"] = {k: v["label_nn"] for k, v in g4[cfg]["control"].items()}
        out["configs"][cfg] = row
    out["sheets"] = str(SU.ROOT / "sheets")
    out["question"] = ("did low-noise scene training (v9b-1) make the held-out scenes visibly better despite failing R1? "
                       "The sheets decide; the numbers flag what to look at (Training_Approach_v10 5.0 #1)")
    C.write_json(C.RESULTS / "phase0" / "p1.json", out)
    for cfg, r in out["configs"].items():
        print(f"{cfg:<10} {json.dumps(r)}")
    print(f"-> {C.RESULTS / 'phase0' / 'p1.json'}; sheets {out['sheets']}")


def main():
    ap = argparse.ArgumentParser(description="v10 Phase 0 #1 / #2")
    ap.add_argument("phase", choices=("p1", "p2", "report"))
    ap.add_argument("--max-new", type=int, help="generate at most N new clips, then exit 0 with 'REMAINING <k>'")
    a = ap.parse_args()
    if a.phase == "report":
        report(); return 0
    left = (p1 if a.phase == "p1" else p2)(a.max_new)
    print(f"REMAINING {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
