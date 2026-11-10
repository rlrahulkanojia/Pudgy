#!/usr/bin/env python3
"""G5: the phone on the v10 stack (Training_Approach_v10.md section 5.1): ID1 on the 24-clip confirm
tier with E5, bar <= 1 / 24 defective (E5 alone on v8merged: 0 / 24).

This is arms_v9c2.py's `--tier confirm --arms E5` (same 24 jobs: the 8 phone starts x seeds 42 / 7 /
1234, 41 frames, 1024^2, the 2026-10-27 registry wording, E5 offset-spliced into the pinned phrase,
gen_v8's sampler settings; same scorer: phone_id_v9.score). Why a wrapper and not the stock command:
  - arms_v9c2 resolves configurations through common_v9, which has no high-noise-only `v10@S`;
  - its output dir is named only by the arm (runs/<RUN>/arms_v9c2/confirm/E5) and gen_v8 skips
    existing clips, so a second config (or restored v9c2 clips) would be silently re-scored (the
    stale-clip trap). Here every configuration gets its own dir: runs/<RUN>/g5/<cfg>/.
  - the jobs and start frames come from the eval set (/workspace/eval_v10/g5, built on the Mac with
    build_assets_v9's code), so the box needs neither raw iteration_6 nor /workspace/eval_v9/keyframes.
    `--jobs-from-arms` rebuilds them on the box with arms_v9c2.jobs_for instead (needs both).
The stock route, for a config common_v9 knows (e.g. the E5 baseline itself), stays:
    cd eval_v9 && RUN_NAME=pudgy-v10-g5-base E5_CONFIG=v8merged python arms_v9c2.py --tier confirm --arms E5 --asset $E5

Box (CUDA_VISIBLE_DEVICES=1 by default; ~6.6 h per config at 1024^2 f41):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY g5_v10.py gen --config v10@150 [--max-new N]     # last line REMAINING <k>, exit 0
    $PY g5_v10.py gate --config v10@150                  # scores ID1, writes id1.json, exit 0 PASS / 10 FAIL / 2 missing
    $PY g5_v10.py all --config v10@150 [--max-new N]
Outputs: /workspace/eval_v10/runs/$RUN_NAME/g5/<cfg>/<tag>.mp4, id1.json
"""
import argparse
import json
import sys

import common_v10 as C  # noqa: E402  (first: CUDA_VISIBLE_DEVICES)
import phone_id_v9 as ID  # noqa: E402

ROOT = C.RESULTS / "g5"
MAX_DEFECTS = 1
EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 10, 2


def load_jobs(from_arms=False):
    if from_arms:
        import arms_v9c2 as AR
        fn = AR.arms(str(C.E5))["E5"][2]
        return [fn(j) for j in AR.jobs_for("confirm", 41)]
    jobs = json.loads((C.G5DIR / "jobs.json").read_text())
    assert len(jobs) == 24 and all(j["frames"] == 41 for j in jobs), "eval set g5/jobs.json is not the confirm tier"
    return jobs


def gen(config, max_new=None, from_arms=False):
    js = load_jobs(from_arms)
    _, rem = C.generate(config, js, ROOT / C.cfg_dir(config), size=None, e5=True, max_new=max_new)
    return rem


def gate(config, from_arms=False):
    js = load_jobs(from_arms)
    d = ROOT / C.cfg_dir(config)
    miss = [j["tag"] for j in js if not (d / f"{j['tag']}.mp4").exists()]
    if miss:
        print(f"G5 {config}: {len(miss)} clips missing (e.g. {miss[:2]})")
        return EXIT_MISSING
    rows = [ID.score(str(d / f"{j['tag']}.mp4")) | {"tag": j["tag"], "split": j.get("split")} for j in js]
    n_def = sum(r["defect"] for r in rows)
    ok = n_def <= MAX_DEFECTS
    C.write_json(d / "id1.json", {"config": config, "arm": "E5 (arms_v9c2 confirm tier)", "defects": n_def,
                                  "n": len(rows), "bar": f"<= {MAX_DEFECTS}/24", "pass": ok, "rows": rows})
    for r in rows:
        if r["defect"]:
            print(f"   DEFECT {r['tag']}: min ratio {r['min_ratio']}, run {r['longest_low_run']}")
    print(f"G5 {config}: {n_def}/{len(rows)} defective (bar <= {MAX_DEFECTS})  {'PASS' if ok else 'FAIL'}  -> {d / 'id1.json'}")
    return EXIT_PASS if ok else EXIT_FAIL


def main():
    ap = argparse.ArgumentParser(description="v10 G5: phone ID1, confirm tier, E5")
    ap.add_argument("cmd", choices=("gen", "gate", "all"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, help="generate at most N new clips, then exit 0 with 'REMAINING <k>'")
    ap.add_argument("--jobs-from-arms", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("gen", "all"):
        left = gen(a.config, a.max_new, a.jobs_from_arms)
        if a.cmd == "gen" or left:
            print(f"REMAINING {left}"); return 0
    return gate(a.config, a.jobs_from_arms)


if __name__ == "__main__":
    sys.exit(main())
