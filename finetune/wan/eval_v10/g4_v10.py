#!/usr/bin/env python3
"""G4: the v8 prompts still work (Training_Approach_v10.md section 5.1, Q15).

The v8 suites unchanged: `core` (eval_v8/suite_v8.core: 13 labels x Pax/Polly x FRONT/QF2_R start,
seed 42, f21 expression / f13 motion = 52 clips) and `compose` (suite_v8.compose: 3 actions x 3
expressions x 2 characters, f25 = 18 clips), generated at 1024^2 as v8 was (gen_v8's settings, E5 on,
a no-op for these prompts), scored by eval_v8/score_v8.score_dir (CLIP ViT-L/14 label 1-NN, zero-shot).
The start frames come from the eval set (/workspace/eval_v10/g4/keyframes = Azure v8/eval/keyframes),
so /workspace/eval_v8 is not needed. This is the shim the inventory asked for: suite_v8's job
builders with the v10 configurations, instead of gen_v8.run's v8-only lora_config.

Gate (fixed before the data):
  core     per kind (expression, motion): label_nn averaged over Pax and Polly >= v8merged's - 0.1
           (per character reported; read as "no more than 0.1 below", an improvement never fails)
  compose  motion_zs >= v8merged's - 0.1

Box (CUDA_VISIBLE_DEVICES=1 by default):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY g4_v10.py gen --config v8merged [--suites core compose] [--max-new N]   # 70 clips, ~5-6 h (1024^2)
    $PY g4_v10.py gen --config v10@150
    $PY g4_v10.py score --configs v8merged v10@150 [--suites core compose]
    $PY g4_v10.py gate --cand v10@150
    $PY g4_v10.py all --config v10@150 [--max-new N]     # gen; then (when complete) score both + gate
Outputs: /workspace/eval_v10/runs/$RUN_NAME/g4/<suite>/<cfg>/<tag>.mp4 + meta.json,
         g4/scores.json ({suite: {config: score_v8 output}}), g4/g4_<cand>.json
Exit codes: gen 0 with last line "REMAINING <k>" (0 = complete); gate 0 PASS / 10 FAIL / 2 missing
            clips or scores; `all` = gen's contract while k > 0, then gate's.
"""
import argparse
import json
import sys
from pathlib import Path

import common_v10 as C  # noqa: E402  (first: CUDA_VISIBLE_DEVICES)
import suite_v8 as S8  # noqa: E402  (eval_v8: core() / compose() job builders)

ROOT = C.RESULTS / "g4"
SUITES = ("core", "compose")
TOL = 0.1
EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 10, 2


def keyframe(char, view="FRONT", bg="white"):
    return C.G4KF / f"{char.lower()}_{view}_{bg}.png"


S8.keyframe = keyframe                 # core() / compose() resolve start frames through this


def jobs(suite):
    return S8.SUITES[suite]()


def out_dir(suite, config):
    return ROOT / suite / C.cfg_dir(config)


def gen(config, suites=SUITES, max_new=None):
    left = 0
    for s in suites:
        js = jobs(s)
        d = out_dir(s, config)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({j["tag"]: j["meta"] for j in js}, indent=1))
        miss = [j["start"] for j in js if not Path(j["start"]).exists()]
        if miss:
            sys.exit(f"missing start frames {miss[:2]} (eval set g4/keyframes)")
        budget = None if max_new is None else max(0, max_new)
        before = sum(not (d / f"{j['tag']}.mp4").exists() for j in js)
        _, rem = C.generate(config, js, d, size=None, e5=True, max_new=budget)
        if max_new is not None:
            max_new -= before - rem
        left += rem
    return left


def complete(suite, config):
    d = out_dir(suite, config)
    return all((d / f"{j['tag']}.mp4").exists() for j in jobs(suite))


def score(configs, suites=SUITES, rescore=False):
    with C.locked(ROOT / "scores.json"):
        return _score(configs, suites, rescore)


def _score(configs, suites, rescore):
    import score_v8
    dest = ROOT / "scores.json"
    res = C.load_json(dest, {})
    for s in suites:
        for cfg in configs:
            if not complete(s, cfg):
                print(f"   {s}/{cfg}: clips missing, not scored"); continue
            if cfg in res.get(s, {}) and not rescore:
                continue
            res.setdefault(s, {})[cfg] = score_v8.score_dir(s, out_dir(s, cfg))
            C.write_json(dest, res)
            print(f"   scored {s}/{cfg}")
    return res


def gate(cand, base="v8merged"):
    res = C.load_json(ROOT / "scores.json", {})
    try:
        cc, bc = res["core"][cand]["control"], res["core"][base]["control"]
        mc, mb = res["compose"][cand]["motion_zs"], res["compose"][base]["motion_zs"]
    except KeyError as e:
        print(f"G4: missing scores ({e}); run gen + score for {cand} and {base}")
        return EXIT_MISSING
    out, ok = {"cand": cand, "base": base, "tolerance": TOL}, True
    for kind in ("expression", "motion"):
        keys = [k for k in bc if k.endswith("/" + kind) and k in cc]
        c = sum(cc[k]["label_nn"] for k in keys) / len(keys)
        b = sum(bc[k]["label_nn"] for k in keys) / len(keys)
        p = c >= b - TOL
        ok &= p
        out[f"core_{kind}"] = {"label_nn": round(c, 3), "base": round(b, 3), "diff": round(c - b, 3), "pass": p,
                               "within_0.1_two_sided": abs(c - b) <= TOL,
                               "per_char": {k: [cc[k]["label_nn"], bc[k]["label_nn"]] for k in keys}}
    p = mc >= mb - TOL
    ok &= p
    out["compose_motion_zs"] = {"cand": mc, "base": mb, "diff": round(mc - mb, 3), "pass": p,
                                "within_0.1_two_sided": abs(mc - mb) <= TOL,
                                "expr_zs": [res["compose"][cand].get("expr_zs"), res["compose"][base].get("expr_zs")]}
    out["status"] = "PASS" if ok else "FAIL"
    C.write_json(ROOT / f"g4_{C.cfg_dir(cand)}.json", out)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:<18} {json.dumps({x: y for x, y in v.items() if x != 'per_char'})}")
    print(f"G4 {cand} vs {base}: {out['status']}  -> {ROOT / f'g4_{C.cfg_dir(cand)}.json'}")
    return EXIT_PASS if ok else EXIT_FAIL


def main():
    ap = argparse.ArgumentParser(description="v10 G4: v8 core 52 + compose 18")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--config", required=True)
    s = sub.add_parser("score"); s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--rescore", action="store_true")
    t = sub.add_parser("gate"); t.add_argument("--cand", required=True); t.add_argument("--base", default="v8merged")
    al = sub.add_parser("all"); al.add_argument("--config", required=True); al.add_argument("--base", default="v8merged")
    for p in (g, s, al):
        p.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    for p in (g, al):
        p.add_argument("--max-new", type=int, help="generate at most N new clips, then exit 0 with 'REMAINING <k>'")
    a = ap.parse_args()
    if a.cmd == "gen":
        print(f"REMAINING {gen(a.config, a.suites, a.max_new)}"); return 0
    if a.cmd == "score":
        score(a.configs, a.suites, a.rescore); return 0
    if a.cmd == "gate":
        return gate(a.cand, a.base)
    left = gen(a.config, a.suites, a.max_new)
    if left:
        print(f"REMAINING {left}"); return 0
    score([a.base, a.config], a.suites)
    return gate(a.config, a.base)


if __name__ == "__main__":
    sys.exit(main())
