#!/usr/bin/env python3
"""Tier 1 per v10 checkpoint, on GPU 1 (Training_Approach_v10.md section 5.1; Q13, spec "Eval").

The v8 diffusion classifier (eval_v8/dcls_v8: same change mask, same per-(clip, k) noise, so two
configurations are a PAIRED comparison) run on the HIGH-noise expert at t = 0.95 / 0.92 (as
dcls_asset_v9 --expert high), with E5 spliced into every caption that holds the phone phrase
(as at inference and in training, Q17). Two clip sets, one model load per configuration:

  G3      held-out scene windows (/workspace/eval_v10/scenes, build_eval_v10.py), the true caption
          against its distractors. Per window, the text-attributable gain
              g = [ln E_cand(true) - ln E_base(true)] - mean_d [ln E_cand(d) - ln E_base(d)]
          with a 95% CI = the WIDER of the percentile bootstrap and Student-t (gate_pivotal_v9).
          REPORTED, not gating (Q13): "worse" = mean > 0 with the CI excluding 0.
  G4-fast the v8-label regression set (/workspace/eval_v9/holdout, HOLDOUT split = QF2_R, expression
          + motion; latents restored from eval_for_v9b.tar), loss under the TRUE caption only:
          per kind and per t, geometric-mean loss ratio cand / base <= 1.05 (gate_pivotal_v9 R2).

Base = v8merged (high = golden), scored ONCE and reused (its JSON is cached). The fix for
dcls_asset_v9's splice assertion (it counts phone PROP clips only, so it fails when a scene caption
holds the iFin phrase): here every caption with the phrase is spliced, and the count of captions
with the phrase vs spliced is recorded in the result.

Tier 1 HOLDS at a checkpoint = G4-fast passes AND G3 is not "worse".
Early stop (only at step 100): G4-fast failed at BOTH step 50 and step 100, OR G3 "worse" at 100.
Candidate = the latest checkpoint whose Tier 1 holds (`summary`; `summary --before S` = the one
before S, when S failed G2/G4/G5).

Box commands (CUDA_VISIBLE_DEVICES=1 is the default, see common_v10):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    RUN_NAME=pudgy-v10-highnoise $PY tier1_v10.py run --step 50     # base once (~15-20 min) + v10@50 (~15-20 min), gate
    $PY tier1_v10.py gate --step 100                                  # re-gate from the saved JSONs
    $PY tier1_v10.py summary                                          # table, candidate as the LAST line
    $PY tier1_v10.py summary --before 250                             # previous Tier-1 checkpoint
    $PY tier1_v10.py score --configs v8merged                         # scoring only (baseline)
Inputs: <ckpt> /workspace/wan_output/$RUN_NAME/$RUN_NAME-step000000SS.safetensors; latents in
/workspace/eval_v10/scenes/cache/f33 (README of the eval set) and /workspace/eval_v9/holdout/cache.
Outputs: /workspace/eval_v10/runs/$RUN_NAME/tier1/{v8merged_k2.json, v10_<S>_k2.json, gate_step<S>.json,
         summary.json, summary_before<S>.json}

Exit codes (run / gate):  0  Tier 1 holds at S             -> keep training
                         10  Tier 1 does not hold at S     -> keep training (S is not a candidate)
                         20  EARLY STOP (step-100 rule)    -> stop training, report
                          2  missing input (checkpoint, latents, base result)
                          1  crash
           (summary):    0  a candidate exists (last stdout line = v10@S);  30  none holds
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import common_v10 as C  # noqa: E402  (first: sets CUDA_VISIBLE_DEVICES before torch)
import numpy as np

T_HIGH = (0.95, 0.92)          # inside the high-noise expert's range (t >= 0.9), as dcls_asset_v9 --expert high
R2_MAX = 1.05
FIRST_STEP, STOP_STEP = 50, 100
EXIT_HOLDS, EXIT_FAILS, EXIT_STOP, EXIT_MISSING, EXIT_NO_CANDIDATE = 0, 10, 20, 2, 30
OUT = C.RESULTS / "tier1"


def result_path(name, k):
    return OUT / f"{C.cfg_dir(name)}_k{k}.json"


# ---------------------------------------------------------------------------------------------
# scoring (GPU)
# ---------------------------------------------------------------------------------------------
def gpu_modules():
    import torch
    import dcls_v8 as D                        # imports musubi (sys.path from eval_v8/common.MUSUBI)
    import asset_embed_v9 as A
    return torch, D, A


def load_high(name, device):
    """The HIGH-noise DiT with the configuration's high-noise LoRA list merged (dcls_asset_v9.load_high)."""
    torch, D, _ = gpu_modules()
    from safetensors.torch import load_file
    from musubi_tuner.wan.configs import WAN_CONFIGS
    from musubi_tuner.wan.modules.model import load_wan_model
    _, high = C.lora_config(name)
    for p, _m in high:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    model = load_wan_model(WAN_CONFIGS["i2v-A14B"], device, str(C.C8.DIT_HIGH), "sdpa", False, "cpu",
                           torch.float16, False, lora_weights_list=[load_file(str(p)) for p, _ in high],
                           lora_multipliers=[m for _, m in high])
    return model.to(device).eval()


def embed(texts, device):
    """T5 context per caption (dcls_v8.encode_captions), E5 added to the phone-phrase rows."""
    torch, D, A = gpu_modules()
    from safetensors.torch import load_file
    emb = D.encode_captions(texts, device)
    tok = A.tokenizer()
    ph = A.token_ids(tok, C.E5_PHRASE)[:-1]
    E = load_file(str(C.E5))["E"].float()
    assert E.shape[0] == len(ph), f"E5 has {E.shape[0]} rows, the phrase spans {len(ph)} tokens"
    n_text = n_spliced = 0
    for t in list(emb):
        if C.E5_PHRASE not in t:
            continue
        n_text += 1
        ids = A.token_ids(tok, t)
        s, e = A.find_span(ids, ph)
        if e > s:
            assert emb[t].shape[0] == len(ids), "cache rows != tokens"
            emb[t] = A.splice([emb[t].float()], [(s, e)], E, True)[0].to(torch.float16)
            n_spliced += 1
    print(f"   E5: {n_spliced} of {n_text} captions with the phone phrase spliced ({len(emb)} captions)", flush=True)
    return emb, {"captions_with_phrase": n_text, "spliced": n_spliced, "asset": str(C.E5)}


def score(model, entry, emb, k_count, device, holdout, labels=None):
    """dcls_v8.score_clip at the high-noise t grid, optionally on a subset of the candidates
    (the regression set needs the true caption only). Identical numbers per candidate: each
    candidate's loss depends only on (clip, k, caption)."""
    torch, D, _ = gpu_modules()
    D.HOLDOUT = holdout                                   # load_latents() reads <holdout>/cache/f<n>/
    x0, y = D.load_latents(entry)
    x0 = x0.float(); y = y.to(device, torch.float16)
    Cc, F, H, W = x0.shape
    seq_len = F * H * W // 4
    labels = labels or list(entry["candidates"])
    E = np.zeros((len(labels), k_count))
    w = D.change_mask(x0).to(device)
    with torch.no_grad():
        for k in range(k_count):
            t = T_HIGH[k % len(T_HIGH)]
            eps = D.noise_for(entry["clip"], k, x0.shape)
            xt = ((1 - t) * x0 + t * eps).to(device, torch.float16)
            target = (eps - x0).to(device)
            for i, lab in enumerate(labels):
                ctx = [emb[entry["candidates"][lab]].to(device)]
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model([xt], t=torch.full((1,), t * 1000.0, device=device), context=ctx,
                                 seq_len=seq_len, y=[y])
                E[i, k] = float((((pred[0].float() - target) ** 2) * w).sum() / w.sum())
    tot = E.sum(1)
    ti = labels.index(entry["label"])
    row = {"clip": entry["clip"], "label": entry["label"], "kind": entry["kind"], "char": entry["char"],
           "split": entry["split"], "loss_true": [float(v) for v in E[ti]],
           "E": {lab: float(v) for lab, v in zip(labels, tot)}}
    for key in ("source", "skit", "phone", "camera"):
        if key in entry:
            row[key] = entry[key]
    if len(labels) > 1:
        order = np.argsort(tot)
        others = np.delete(tot, ti)
        row.update(pred=labels[int(order[0])], rank=int(np.where(order == ti)[0][0]) + 1,
                   margin=float((others.min() - tot[ti]) / tot[ti]))
    return row


def sets():
    scene = json.loads((C.SCENES / "index.json").read_text())
    reg = [e for e in json.loads((C.REGRESSION / "index.json").read_text())
           if e["split"] == "holdout" and e["kind"] in ("expression", "motion")]
    return scene, reg


def check_latents(index, holdout):
    miss = [e["clip"] for e in index
            if not list((holdout / "cache" / f"f{e['frames']}").glob(f"{e['clip'][:-4]}_*_wan.safetensors"))]
    return miss


def score_configs(names, k=2, reg_all=False):
    scene, reg = sets()
    for idx, root in ((scene, C.SCENES), (reg, C.REGRESSION)):
        miss = check_latents(idx, root)
        if miss:
            print(f"no latent cache for {len(miss)} clips under {root}/cache (e.g. {miss[:2]}); cache them first "
                  f"(eval set README: wan_cache_latents.py --dataset_config {C.SCENES}/scenes_f33.toml ... --i2v)")
            sys.exit(EXIT_MISSING)
    todo = [n for n in names if not result_path(n, k).exists()]
    for n in names:
        if n not in todo:
            print(f"== {n}: cached -> {result_path(n, k)}")
    if not todo:
        return
    for n in todo:
        miss = [str(p) for p, _ in C.lora_config(n)[1] if not Path(p).exists()]
        if miss:
            print(f"missing high-noise weights for {n}: {miss}"); sys.exit(EXIT_MISSING)
    torch, D, _ = gpu_modules()
    device = torch.device("cuda")
    texts = [c for e in scene for c in e["candidates"].values()] + \
            [c for e in reg for lab, c in e["candidates"].items() if reg_all or lab == e["label"]]
    emb, e5 = embed(texts, device)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in todo:
        t0 = time.time()
        model = load_high(name, device)
        t_load = time.time() - t0
        rows_scene = [score(model, e, emb, k, device, C.SCENES) for e in scene]
        rows_reg = [score(model, e, emb, k, device, C.REGRESSION, None if reg_all else [e["label"]]) for e in reg]
        res = {"config": name, "expert": "high", "k": k, "t_grid": list(T_HIGH), "e5": e5,
               "high_loras": [[str(p), m] for p, m in C.lora_config(name)[1]],
               "rows_scene": rows_scene, "rows_reg": rows_reg,
               "seconds": round(time.time() - t0), "load_seconds": round(t_load)}
        C.write_json(result_path(name, k), res)
        print(f"== {name}: {len(rows_scene)} scene windows + {len(rows_reg)} regression clips in "
              f"{res['seconds']} s (load {res['load_seconds']} s) -> {result_path(name, k)}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# gate (CPU)
# ---------------------------------------------------------------------------------------------
def g4_fast(cand, base):
    b = {r["clip"]: r for r in base["rows_reg"]}
    out, ok = {}, True
    for kind in ("expression", "motion"):
        rs = [r for r in cand["rows_reg"] if r["kind"] == kind and r["clip"] in b]
        if not rs:
            out[kind] = {"n": 0}; ok = False; continue
        per_t = {}
        for ti, t in enumerate(cand["t_grid"]):
            per_t[str(t)] = round(float(np.exp(np.mean([np.log(r["loss_true"][ti] / b[r["clip"]]["loss_true"][ti])
                                                        for r in rs]))), 4)
        p = all(v <= R2_MAX for v in per_t.values())
        ok &= p
        out[kind] = {"n": len(rs), "loss_ratio": per_t, "pass": p}
    return {"bar": f"geometric-mean loss ratio <= {R2_MAX} per kind and t", "kinds": out, "pass": bool(ok)}


def gains(cand, base, sel=lambda r: True):
    b = {r["clip"]: r for r in base["rows_scene"]}
    g = []
    for r in cand["rows_scene"]:
        if r["clip"] not in b or not sel(r):
            continue
        bb = b[r["clip"]]
        dt = np.log(r["E"]["true"]) - np.log(bb["E"]["true"])
        dd = [np.log(r["E"][c]) - np.log(bb["E"][c]) for c in r["E"] if c != "true"]
        if dd:
            g.append(float(dt - np.mean(dd)))
    return g


def g3(cand, base):
    g = gains(cand, base)
    m, lo, hi = C.ci95(g)
    verdict = "worse" if (m > 0 and lo > 0) else "better" if (m < 0 and hi < 0) else "no change"
    b = {r["clip"]: r for r in base["rows_scene"]}
    rs = [r for r in cand["rows_scene"] if r["clip"] in b]
    sub = {}
    for key, sel in (("v10", lambda r: r.get("source") == "v10"), ("v9b", lambda r: r.get("source") == "v9b"),
                     ("phone", lambda r: bool(r.get("phone"))), ("camera_move", lambda r: r.get("camera", "static") != "static")):
        gg = gains(cand, base, sel)
        if len(gg) >= 2:
            sm, slo, shi = C.ci95(gg)
            sub[key] = {"n": len(gg), "gain": round(sm, 4), "ci": [round(slo, 4), round(shi, 4)]}
    return {"n": len(g), "gain": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "verdict": verdict,
            "true_loss_ratio": round(float(np.exp(np.mean([np.log(r["E"]["true"] / b[r["clip"]]["E"]["true"]) for r in rs]))), 4),
            "acc_base": round(float(np.mean([b[r["clip"]]["pred"] == "true" for r in rs])), 3),
            "acc": round(float(np.mean([r["pred"] == "true" for r in rs])), 3), "by": sub,
            "note": "reported, not gating (Q13); 'worse' (gain > 0, CI excluding 0) feeds the step-100 stop"}


def gate_step(step, k=2):
    base, cand = C.load_json(result_path("v8merged", k)), C.load_json(result_path(f"v10@{step}", k))
    if base is None or cand is None:
        return None
    a, b = g4_fast(cand, base), g3(cand, base)
    return {"step": int(step), "config": f"v10@{step}", "g4_fast": a, "g3": b,
            "holds": bool(a["pass"] and b["verdict"] != "worse")}


def decide(step, k=2):
    """-> (gate dict with the early-stop decision, exit code)."""
    gs = gate_step(step, k)
    if gs is None:
        print(f"missing Tier-1 results for v8merged or v10@{step} under {OUT}")
        sys.exit(EXIT_MISSING)
    stop, why = False, []
    if int(step) == STOP_STEP:
        prev = gate_step(FIRST_STEP, k)
        if prev is None:
            why.append(f"step {FIRST_STEP} not scored: the G4-fast half of the rule cannot fire")
        elif not prev["g4_fast"]["pass"] and not gs["g4_fast"]["pass"]:
            stop = True; why.append(f"G4-fast failed at steps {FIRST_STEP} and {STOP_STEP}")
        if gs["g3"]["verdict"] == "worse":
            stop = True; why.append(f"G3 caption use measurably worse at step {STOP_STEP}")
    gs["early_stop"] = {"stop": stop, "reasons": why, "rule": "at step 100: G4-fast fails at 50 and 100, or G3 worse"}
    C.write_json(OUT / f"gate_step{int(step)}.json", gs)
    return gs, EXIT_STOP if stop else EXIT_HOLDS if gs["holds"] else EXIT_FAILS


def show(gs):
    a, b = gs["g4_fast"], gs["g3"]
    r2 = "  ".join(f"{kind[:4]} " + "/".join(f"{v:.3f}" for v in x.get("loss_ratio", {}).values())
                   for kind, x in a["kinds"].items())
    print(f"v10@{gs['step']:<5} G4-fast {'PASS' if a['pass'] else 'FAIL'} ({r2})   "
          f"G3 {b['gain']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] n={b['n']} {b['verdict']}   "
          f"acc {b['acc_base']}->{b['acc']}   Tier 1 {'HOLDS' if gs['holds'] else 'does not hold'}"
          + (f"   EARLY STOP: {'; '.join(gs['early_stop']['reasons'])}" if gs.get("early_stop", {}).get("stop") else ""))


def steps_scored(k=2):
    out = []
    for p in OUT.glob(f"v10_*_k{k}.json"):
        m = re.match(rf"v10_(\d+)_k{k}\.json$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def summary(k=2, before=None):
    rows = []
    for s in steps_scored(k):
        gs = gate_step(s, k)
        if gs:
            rows.append(gs); show(gs)
    ok = [g["step"] for g in rows if g["holds"] and (before is None or g["step"] < before)]
    cand = max(ok) if ok else None
    stop = C.load_json(OUT / f"gate_step{STOP_STEP}.json", {}).get("early_stop")
    C.write_json(OUT / ("summary.json" if before is None else f"summary_before{before}.json"), {"steps": rows, "candidate": f"v10@{cand}" if cand else None,
                                        "before": before, "early_stop_at_100": stop,
                                        "rule": "candidate = the latest checkpoint whose Tier 1 holds"})
    print(f"v10@{cand}" if cand else "NONE")
    return EXIT_HOLDS if cand else EXIT_NO_CANDIDATE


def main():
    ap = argparse.ArgumentParser(description="v10 Tier 1 (G3 + G4-fast) on the high-noise expert")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--step", required=True, type=int)
    g = sub.add_parser("gate"); g.add_argument("--step", required=True, type=int)
    s = sub.add_parser("summary"); s.add_argument("--before", type=int)
    c = sub.add_parser("score"); c.add_argument("--configs", nargs="+", required=True)
    c.add_argument("--reg-all", action="store_true", help="score every v8 label on the regression set (R1 report)")
    for p in (r, g, s, c):
        p.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "score":
        score_configs(a.configs, a.k, a.reg_all); return 0
    if a.cmd == "summary":
        return summary(a.k, a.before)
    if a.cmd == "run":
        if not C.ckpt(a.step).exists():
            print(f"no checkpoint {C.ckpt(a.step)}"); return EXIT_MISSING
        score_configs(["v8merged", f"v10@{a.step}"], a.k)
    gs, code = decide(a.step, a.k)
    show(gs)
    summary(a.k)
    print(f"exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
