"""Q22 (2026-11-10): G2 identity fails only when the 95% CI of the paired (job, seed) difference
v10 - v8merged lies wholly below 0. Runs the real suite_v10.gate on synthetic scores (no GPU, no models).

    <python with numpy scipy opencv-python-headless pillow> t_g2_identity.py <repo root>
"""
import json, os, sys, tempfile, random
from pathlib import Path
repo = Path(sys.argv[1]); ev = Path(tempfile.mkdtemp()); os.environ.update(V10_EVAL=str(ev), RUN_NAME="q22")
(ev / "g1").mkdir(parents=True)
jobs = [{"tag": f"job{i}", "camera": "static", "frames": 49} for i in range(10)]
(ev / "g1" / "jobs.json").write_text(json.dumps([dict(j, source="v10") for j in jobs]))
sys.path.insert(0, str(repo / "finetune/wan/eval_v10")); sys.path.insert(0, str(repo / "finetune/wan"))
import common_v10 as C, suite_v10 as S
def run(delta, noise):
    rng = random.Random(1); sc = {}
    for j in jobs:
        sc[f"_ref|{j['tag']}"] = {"zoom": {"zoom_pct": 0.0, "verdict": "ok"}}
        for s in C.SEEDS:
            base = 0.9 + rng.uniform(-0.02, 0.02)
            for cfg, v in (("v8merged", base), ("v10@100", base + delta + rng.gauss(0, noise))):
                sc[f"{cfg}|{j['tag']}_s{s}"] = {"dup_flag": False, "zoom": {"zoom_pct": 0.0, "verdict": "ok"},
                                                "motion": 1.0, "identity": v, "ocr": {"verdict": "ok", "new": []}}
    S.ROOT.mkdir(parents=True, exist_ok=True); (S.ROOT / "scores.json").write_text(json.dumps(sc))
    rc = S.gate("v10@100")
    rep = json.loads((S.ROOT / "g2_v10_100.json").read_text())
    return rep["checks"]["identity"], rep["status"], rc
cases = [("small drop in noise", -0.003, 0.04, None), ("draw measurably worse", -0.008, 0.03, None),
         ("clear drop -0.05", -0.05, 0.01, False), ("equal", 0.0, 0.01, True), ("better +0.03", 0.03, 0.01, True)]
seen_straddle = False
for name, delta, noise, want in cases:
    idc, status, rc = run(delta, noise)
    rule = idc["ci"][1] >= 0                      # Q22: fail only if the CI lies wholly below 0
    seen_straddle |= idc["ci"][0] < 0 <= idc["ci"][1] and idc["paired_diff"] < 0
    ok = idc["pass"] == rule and (want is None or idc["pass"] == want)
    print(f"{'ok  ' if ok else 'FAIL'} {name:<24} diff {idc['paired_diff']:+.4f} CI {idc['ci']} pass={idc['pass']} status={status} exit={rc}")
    assert ok
assert seen_straddle, "no case had a negative mean with a CI straddling 0"
print("Q22 identity rule: ALL PASS (a negative mean whose CI straddles 0 passes; a CI wholly below 0 fails)")
