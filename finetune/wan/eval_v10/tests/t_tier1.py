"""Mac tests for the v10 eval package (no GPU). <W> = a scratch dir; <PY> = a venv with numpy, scipy, opencv,
pillow, torch, transformers, rapidocr_onnxruntime. First build a Mac-rooted copy of the eval set:
    cd finetune/wan/eval_v10 && PUDGY_DATA=<Data> <PY> build_eval_v10.py --out <W>/eval_mac --root <W>/eval_mac
    mkdir -p <W>/tests && cp tests/fake_gen_asset.py <W>/tests/
Then:
    <PY> tests/t_tier1.py <W> <repo> <PY>          # Tier-1 gate / early stop / summary exit codes on synthetic results
    PUDGY_DATA=<Data> V10_EVAL=<W>/eval_mac RUN_NAME=mac-gen <PY> tests/t_gen.py <W> <repo>/finetune/wan
                                                   # generation against a fake musubi (--interactive protocol,
                                                   # E5 args, --max-new / REMAINING, 848x480 vs 1024^2, FLF, G5 gate)
    PUDGY_DATA=<Data> V10_EVAL=<W>/eval_mac RUN_NAME=mac-selftest <PY> suite_v10.py selftest   # G2 on real shots
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
SP = Path(sys.argv[1]); REPO = Path(sys.argv[2]); PY = sys.argv[3]
ev = SP / "t1eval"; shutil.rmtree(ev, ignore_errors=True)
(ev / "scenes").mkdir(parents=True)
shutil.copy(SP / "eval_mac/scenes/index.json", ev / "scenes/index.json")
idx = json.loads((ev / "scenes/index.json").read_text())
out = ev / "runs/t1test/tier1"; out.mkdir(parents=True)
rng = np.random.default_rng(1)
base_E = {e["clip"]: {k: float(rng.uniform(0.5, 1.0)) for k in e["candidates"]} for e in idx}
reg = [(f"r{i}", "expression" if i < 16 else "motion") for i in range(26)]
base_L = {c: [float(rng.uniform(0.3, 0.6)), float(rng.uniform(0.3, 0.6))] for c, _ in reg}

def rows(E, ratio_reg, gain, sd=0.01):
    sc = []
    for e in idx:
        noise = rng.normal(0, sd) if sd else 0.0
        EE = {k: v * (np.exp(gain + noise) if k == "true" else 1.0) for k, v in E[e["clip"]].items()}
        order = sorted(EE, key=EE.get)
        sc.append({"clip": e["clip"], "label": "true", "kind": "scene", "char": e["char"], "split": "holdout",
                   "loss_true": [EE["true"] / 2] * 2, "E": EE, "pred": order[0], "rank": order.index("true") + 1,
                   "margin": 0.0, "source": e["source"], "phone": e["phone"], "camera": e["camera"]})
    rg = [{"clip": c, "label": "x", "kind": k, "char": "Pax", "split": "holdout",
           "loss_true": [v * ratio_reg * float(np.exp(rng.normal(0, 0.005))) for v in base_L[c]], "E": {"x": 1.0}} for c, k in reg]
    return {"config": "", "t_grid": [0.95, 0.92], "rows_scene": sc, "rows_reg": rg}

def put(name, ratio, gain):
    (out / f"{name}_k2.json").write_text(json.dumps(rows(base_E, ratio, gain, 0.0 if name == "v8merged" else 0.01)))

def run(*args):
    env = dict(os.environ, V10_EVAL=str(ev), RUN_NAME="t1test")
    r = subprocess.run([PY, "tier1_v10.py", *args], cwd=REPO / "finetune/wan/eval_v10", env=env, capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()

put("v8merged", 1.0, 0.0)
cases = [("v10_50 fails G4-fast (x1.08)", [("v10_50", 1.08, 0.0)], ["gate", "--step", "50"], 10),
         ("v10_100 fails again -> early stop", [("v10_100", 1.07, 0.0)], ["gate", "--step", "100"], 20),
         ("v10_100 holds", [("v10_100", 1.00, -0.02)], ["gate", "--step", "100"], 0),
         ("v10_100 G3 worse -> early stop", [("v10_100", 1.00, +0.05)], ["gate", "--step", "100"], 20),
         ("v10_150 holds", [("v10_100", 1.00, -0.02), ("v10_150", 1.03, 0.0)], ["gate", "--step", "150"], 0),
         ("v10_200 G3 worse, no stop after 100", [("v10_200", 1.00, +0.05)], ["gate", "--step", "200"], 10),
         ("summary -> candidate v10@150", [], ["summary"], 0),
         ("summary --before 150 -> v10@100", [], ["summary", "--before", "150"], 0),
         ("summary --before 100 -> none", [], ["summary", "--before", "100"], 30),
         ("missing step 250 -> 2", [], ["gate", "--step", "250"], 2)]
bad = 0
for desc, puts, args, want in cases:
    for n, r, g in puts:
        put(n, r, g)
    code, lines = run(*args)
    ok = code == want
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {desc}: exit {code} (want {want}) | {lines[-1] if lines else ''}")
    if args[0] == "gate" and code in (0, 10, 20):
        print("     ", [l for l in lines if l.startswith("v10@")][0][:230])
print("FAILURES", bad)
s = json.loads((out / "summary.json").read_text())
print("summary.json candidate:", s["candidate"], "| steps:", [(r["step"], r["holds"]) for r in s["steps"]])
