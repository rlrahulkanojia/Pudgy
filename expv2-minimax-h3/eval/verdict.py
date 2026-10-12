#!/usr/bin/env python3
"""PLAN §9 — the pre-registered verdict, bars V1-V11, per arm; plus the §6.6 arm-D trigger.

All bars are on the arm's WINNER (reports/winner.json, select_winner.py), from unseen QF2_R
start frames, over 3 seeds, at f22, excluding running.

  V1  VAE          phase0/vae_gate/report.json "gate" == PASS (metric bars + the 4x eye check)
  V2  style        rubric (reports/rubric_<ckpt>.json): >= 90% of core QF2_R clips pass
                   (flat fills, intact outlines, no 3D shading, on model); colour 100%
  V3  Tier 1       tier1/<ckpt>.json: acc >= 0.75 expression, >= 0.70 motion; loss below
                   the base on >= 15/16 expression and >= 6/7 motion clips
  V4  control      CLIP 1-NN (leave-angle-out) >= 0.45 expression, >= 0.40 motion
  V5  motion       per trained motion: median motion energy >= 0.8x source, median frozen
                   fraction <= source + 0.15
  V6  camera       median |zoom_rel| <= 0.03 (zoom minus the label's source zoom: jumping's
                   shadow shrinks by design); gb corner drift median <= 5 (/255); ground
                   spread median <= max(2x source, floor) - the source ground is exactly
                   flat (spread 0), so the floor is G0.1's flat-fill round-trip std, else 1.0
  V7  identity     median over labels of (late identity / source) >= 0.9
  V8  memorise     CLIP 1-NN accuracy FRONT - QF2_R <= 0.15
  V9  first frame  >= 95% of clips: ff_psnr >= G0.1 mean 768 round-trip PSNR - 3 dB and
                   ff_dino >= 0.95
  V10 cadence      per label: median twos index <= source + 0.10, for >= 10 of 12 labels
  V11 colour       median over labels of (saturation / source) >= 0.9 and of
                   (fill flatness / source) <= 1.5   (fill flatness = local 5x5 luminance std)

Ratios "vs source" are taken per label, then the median over labels, so the label mix of
the bank cannot move a bar. Outcome: FAIL at V1 | FAIL at V2/V11 | FAIL at V3 (not in the
plan's table: the labels were not learned) | INCOMPLETE (inputs missing) | PASS | PARTIAL.

    python expv2-minimax-h3/eval/verdict.py --rubric-template --ckpt expv2-h3-main@250
    python expv2-minimax-h3/eval/verdict.py                 # arm A (+ arm D if it has a winner)
-> EVAL/reports/verdict.json, verdict.md
"""
import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv2_common as C  # noqa: E402
import metrics as M  # noqa: E402

ARM_RUN = {"A": "expv2-h3-main", "D": "expv2-h3-d"}
SPREAD_FLOOR_DEFAULT = 1.0


def bar(passed, value, rule, **detail):
    return {"pass": passed, "value": value, "rule": rule, **({"detail": detail} if detail else {})}


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def src_med(baseline, label, k):
    try:
        return baseline["labels"][label][k]["median"]
    except (KeyError, TypeError):
        return None


def rubric_path(ckpt):
    return C.EVAL / "reports" / f"rubric_{ckpt}.json"


def write_rubric_template(ckpt, force):
    core = C.load_json(C.EVAL / "scores" / ckpt / "core.json")
    gen = C.EVAL / "gen" / ckpt / "core"
    rows = []
    for d in sorted(p for p in gen.glob("*__s*") if p.is_dir()):
        item_id, _, seed = d.name.rpartition("__s")
        if f"_{C.HOLDOUT_ANGLE}__" not in item_id:
            continue
        view = d / "preview.mp4" if (d / "preview.mp4").exists() else d
        label = item_id.split("__")[1].split("_")[1] if "__" in item_id else None
        rows.append(M.rubric_row(view, rid=f"{item_id}__s{seed}", item_id=item_id, seed=int(seed),
                                 label=label, untrained=label == "running"))
    if not rows and core:
        rows = [M.rubric_row("", rid=f"{r['item_id']}__s{r['seed']}", item_id=r["item_id"], seed=r["seed"],
                             label=r["label"], untrained=r["label"] == "running")
                for r in core["clips"] if r["angle"] == C.HOLDOUT_ANGLE]
    dest = rubric_path(ckpt)
    old = C.load_json(dest)
    if old and not force and any(M.rubric_pass(r) is not None for r in old.get("clips", [])):
        sys.exit(f"{dest} already has ratings; --force to overwrite")
    C.write_json(dest, M.rubric_template(rows))
    print(f"rubric template ({len(rows)} clips) -> {dest}")


def evaluate(arm, ckpt, baseline, vae, rt_psnr, spread_floor):
    missing = []
    core = C.load_json(C.EVAL / "scores" / ckpt / "core.json")
    gb = C.load_json(C.EVAL / "scores" / ckpt / "gb.json")
    t1 = C.load_json(C.EVAL / "tier1" / f"{ckpt}.json")
    rub = C.load_json(rubric_path(ckpt))
    for name, obj in (("scores/core", core), ("scores/gb", gb), ("tier1", t1),
                      ("rubric", rub), ("source_baseline", baseline), ("vae_gate", vae)):
        if not obj:
            missing.append(name)
    rows = [r for r in (core or {}).get("clips", []) if r["label"] in C.TRAINED_LABELS]
    q = [r for r in rows if r["angle"] == C.HOLDOUT_ANGLE]
    by_label = defaultdict(list)
    for r in q:
        by_label[r["label"]].append(r)
    bars = {}

    # V1
    gate = (vae or {}).get("gate")
    bars["V1"] = bar(None if gate in (None, "PENDING_EYE") else gate == "PASS", gate,
                     "phase0/vae_gate/report.json gate == PASS (metrics + eye check)",
                     eye_check=(vae or {}).get("eye_check"))

    # V2
    if rub:
        rr = [x for x in rub.get("clips", []) if not x.get("untrained") and x.get("label") != "running"]
        vals = [M.rubric_pass(x) for x in rr]
        rated = [v for v in vals if v is not None]
        col = [x.get("colour_correct") for x in rr if x.get("colour_correct") is not None]
        complete = len(rated) == len(rr) and len(col) == len(rr) and rr
        rate = float(np.mean(rated)) if rated else None
        crate = float(np.mean(col)) if col else None
        washed = [x.get("washed_out") for x in rr if x.get("washed_out") is not None]
        bars["V2"] = bar((rate >= 0.90 and crate == 1.0) if complete else None,
                         {"pass_rate": rate, "colour_rate": crate},
                         ">= 0.90 rubric pass and colour 100% on core QF2_R",
                         rated=len(rated), n=len(rr),
                         washed_out_rate=float(np.mean(washed)) if washed else None)
    else:
        bars["V2"] = bar(None, None, ">= 0.90 rubric pass and colour 100% on core QF2_R")

    # V3
    if t1:
        s = t1.get("summary", {})
        e, m = s.get("expression", {}), s.get("motion", {})
        ok = None
        if None not in (e.get("acc"), m.get("acc"), e.get("below_base"), m.get("below_base")):
            ok = (e["acc"] >= 0.75 and m["acc"] >= 0.70
                  and e["below_base"] >= int(np.ceil(e.get("n", 16) * 15 / 16))
                  and m["below_base"] >= int(np.ceil(m.get("n", 7) * 6 / 7)))
        bars["V3"] = bar(ok, {"expr_acc": e.get("acc"), "motion_acc": m.get("acc"),
                              "expr_below_base": f"{e.get('below_base')}/{e.get('n')}",
                              "motion_below_base": f"{m.get('below_base')}/{m.get('n')}"},
                         "acc >= 0.75 expr / 0.70 motion; below base >= 15/16 and 6/7")
    else:
        bars["V3"] = bar(None, None, "tier1 missing")

    def acc(rs, key="nn_label_clip"):
        v = [r[key] == r["label"] for r in rs if r.get(key) is not None]
        return (float(np.mean(v)) if v else None), len(v)

    # V4
    ae, ne = acc([r for r in q if r["kind"] == "expression"])
    am, nm = acc([r for r in q if r["kind"] == "motion"])
    bars["V4"] = bar(None if None in (ae, am) else (ae >= 0.45 and am >= 0.40),
                     {"expr": ae, "motion": am}, "CLIP 1-NN >= 0.45 expr, >= 0.40 motion",
                     n_expr=ne, n_motion=nm)

    # V5
    per = {}
    for lab in C.MOTION_TRAINED:
        rs = by_label.get(lab, [])
        ge, gf = med([r["motion_energy"] for r in rs]), med([r.get("frozen_fraction") for r in rs])
        se, sf = src_med(baseline, lab, "motion_energy"), src_med(baseline, lab, "frozen_fraction")
        ok = None if None in (ge, gf, se, sf) else (ge >= 0.8 * se and gf <= sf + 0.15)
        per[lab] = {"energy": ge, "src_energy": se, "frozen": gf, "src_frozen": sf, "pass": ok}
    v5 = [p["pass"] for p in per.values()]
    bars["V5"] = bar(None if None in v5 else all(v5), per,
                     "per motion: energy >= 0.8x source, frozen <= source + 0.15")

    # V6
    zr = med([abs(r["zoom_rel"]) for r in q if r.get("zoom_rel") is not None])
    gbr = [r for r in (gb or {}).get("clips", [])]
    cd = med([r["corner_drift"] for r in gbr])
    cd_max = max([r["corner_drift"] for r in gbr if r.get("corner_drift") is not None], default=None)
    gs = med([r.get("ground_spread") for r in q])
    ss = med([src_med(baseline, lab, "ground_spread") for lab in C.TRAINED_LABELS])
    spread_bar = max(2 * ss, spread_floor) if ss is not None else None
    ok6 = None if None in (zr, cd, gs, spread_bar) else (zr <= 0.03 and cd <= 5.0 and gs <= spread_bar)
    bars["V6"] = bar(ok6, {"zoom_abs_rel_median": zr, "gb_corner_drift_median": cd,
                           "gb_corner_drift_max": cd_max, "ground_spread_median": gs,
                           "ground_spread_bar": spread_bar},
                     "|zoom_rel| <= 3%; gb corner drift <= 5/255; spread <= max(2x source, floor)",
                     spread_floor=spread_floor, source_spread=ss)

    def ratio_over_labels(k, labels=C.TRAINED_LABELS):
        rs = []
        for lab in labels:
            g, s = med([r.get(k) for r in by_label.get(lab, [])]), src_med(baseline, lab, k)
            if g is not None and s:
                rs.append(g / s)
        return med(rs), len(rs)

    # V7
    li, nli = ratio_over_labels("late_identity")
    bars["V7"] = bar(None if li is None else li >= 0.9, li,
                     "median over labels of late identity / source >= 0.9", labels=nli)

    # V8
    af, _ = acc([r for r in rows if r["angle"] == "FRONT"])
    aq, _ = acc(q)
    gap = None if None in (af, aq) else af - aq
    bars["V8"] = bar(None if gap is None else gap <= 0.15, gap,
                     "CLIP 1-NN FRONT - QF2_R <= 0.15", front=af, qf2r=aq)

    # V9
    if rt_psnr is not None and q:
        okc = [r.get("ff_psnr") is not None and r.get("ff_dino") is not None
               and r["ff_psnr"] >= rt_psnr - 3.0 and r["ff_dino"] >= 0.95 for r in q]
        have = all(r.get("ff_dino") is not None for r in q)
        frac = float(np.mean(okc))
        bars["V9"] = bar(frac >= 0.95 if have else None, frac,
                         ">= 95% clips: ff_psnr >= rt - 3 dB and ff_dino >= 0.95",
                         rt_psnr=rt_psnr, psnr_floor=rt_psnr - 3.0)
    else:
        bars["V9"] = bar(None, None, "needs G0.1 h3_mean_psnr_768 and core scores")

    # V10
    per10 = {}
    for lab in C.TRAINED_LABELS:
        g, s = med([r.get("twos_index") for r in by_label.get(lab, [])]), src_med(baseline, lab, "twos_index")
        per10[lab] = None if None in (g, s) else (g <= s + 0.10)
    known = [v for v in per10.values() if v is not None]
    bars["V10"] = bar((sum(known) >= 10) if len(known) == len(C.TRAINED_LABELS) else None,
                      sum(known), "median twos <= source + 0.10 for >= 10 of 12 labels", per_label=per10)

    # V11
    sr, _ = ratio_over_labels("saturation")
    fr, _ = ratio_over_labels("fill_flatness")
    bars["V11"] = bar(None if None in (sr, fr) else (sr >= 0.9 and fr <= 1.5),
                      {"saturation_ratio": sr, "fill_flatness_ratio": fr},
                      "saturation >= 0.9x source; fill flatness <= 1.5x source")

    p = {k: v["pass"] for k, v in bars.items()}
    fails = [k for k in ("V1", "V2", "V11") if p[k] is False]
    unknown = [k for k, v in p.items() if v is None]
    if p["V1"] is False:
        outcome = "FAIL at V1"
    elif fails:
        outcome = "FAIL at " + "/".join(fails)
    elif p["V3"] is False:
        outcome = "FAIL at V3"
    elif unknown:
        outcome = "INCOMPLETE"
    elif all(p.values()):
        outcome = "PASS"
    else:
        outcome = "PARTIAL"
    return {"arm": arm, "ckpt": ckpt, "outcome": outcome, "bars": bars,
            "failed": [k for k, v in p.items() if v is False], "unknown": unknown,
            "missing_inputs": missing, "n_core_qf2r": len(q)}


def arm_d_trigger(res_a, rub):
    """§6.6: V11 fails; or V2 fails with washed-out on >= 20% of core QF2_R clips; or the
    probe's saturation ratio < 0.9 on >= 3 consecutive probed checkpoints before step 300."""
    v11 = res_a["bars"]["V11"]["pass"] is False
    v2 = res_a["bars"]["V2"]
    washed = (v2.get("detail") or {}).get("washed_out_rate")
    v2w = v2["pass"] is False and washed is not None and washed >= 0.20
    probes = []
    for d in sorted((C.EVAL / "scores").glob(f"{ARM_RUN['A']}@*")):
        step = d.name.split("@")[1]
        pr = C.load_json(d / "probe.json")
        if step.isdigit() and int(step) < 300 and pr:
            probes.append((int(step), (pr.get("summary") or {}).get("saturation_ratio_median")))
    probes.sort()
    run, longest = 0, 0
    for _, r in probes:
        run = run + 1 if (r is not None and r < 0.9) else 0
        longest = max(longest, run)
    probe = longest >= 3
    return {"fired": bool(v11 or v2w or probe), "v11_fail": v11, "v2_washed_out": v2w,
            "washed_out_rate": washed, "probe_low_saturation_run": longest, "probes": probes}


def short(v):
    """Compact cell text: numbers rounded, nested per-label dicts left to verdict.json."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        flat = {k: x for k, x in v.items() if not isinstance(x, dict)}
        if not flat:
            return "per label: see verdict.json"
        return ", ".join(f"{k}={short(x)}" for k, x in flat.items())
    return str(v)


def to_md(out):
    lines = [f"# expv2 verdict ({out['created']})", ""]
    arms = [a for a in ("A", "D") if a in out]
    lines.append("| bar | " + " | ".join(f"arm {a} ({out[a]['ckpt']})" for a in arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for k in [f"V{i}" for i in range(1, 12)]:
        cells = []
        for a in arms:
            b = out[a]["bars"][k]
            mark = {True: "PASS", False: "FAIL", None: "—"}[b["pass"]]
            cells.append(f"{mark} {short(b['value'])}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("| **outcome** | " + " | ".join(f"**{out[a]['outcome']}**" for a in arms) + " |")
    lines += ["", f"Overall: **{out['overall']}**"]
    if "arm_d_trigger" in out:
        lines.append(f"Arm D trigger (§6.6): {'FIRED' if out['arm_d_trigger']['fired'] else 'not fired'}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["A", "D", "both"], default="both")
    ap.add_argument("--ckpt", default=None, help="override the winner (single arm only)")
    ap.add_argument("--rt-psnr", type=float, default=None, help="override G0.1's mean 768 round-trip PSNR")
    ap.add_argument("--rubric-template", action="store_true", help="write reports/rubric_<ckpt>.json and exit")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    winners = C.load_json(C.EVAL / "reports" / "winner.json", {}) or {}
    if a.rubric_template:
        ckpt = a.ckpt or (winners.get(ARM_RUN["A" if a.arm == "both" else a.arm]) or {}).get("winner")
        if not ckpt:
            sys.exit("no --ckpt and no winner")
        write_rubric_template(ckpt, a.force)
        return

    baseline = C.load_json(C.PHASE0 / "source_baseline.json")
    vae = C.load_json(C.PHASE0 / "vae_gate" / "report.json")
    rt = a.rt_psnr if a.rt_psnr is not None else (vae or {}).get("h3_mean_psnr_768")
    floor = SPREAD_FLOOR_DEFAULT
    try:
        ff = [x.get("fill_std") for x in vae["verdict"]["bars"]["flat_fill"] if x.get("fill_std") is not None]
        if ff:
            floor = max(floor, float(max(ff)))
    except (KeyError, TypeError):
        pass

    out = {"created": datetime.datetime.now().isoformat(timespec="seconds")}
    for arm in (("A", "D") if a.arm == "both" else (a.arm,)):
        ckpt = a.ckpt if (a.ckpt and a.arm != "both") else (winners.get(ARM_RUN[arm]) or {}).get("winner")
        if not ckpt:
            if arm == "A" or a.arm == arm:
                print(f"arm {arm}: no winner in reports/winner.json (run select_winner.py --run {ARM_RUN[arm]})")
            continue
        out[arm] = evaluate(arm, ckpt, baseline, vae, rt, floor)
    if "A" in out:
        out["arm_d_trigger"] = arm_d_trigger(out["A"], C.load_json(rubric_path(out["A"]["ckpt"])))
    outs = [out[k]["outcome"] for k in ("A", "D") if k in out]
    out["overall"] = "PASS" if "PASS" in outs else (" / ".join(outs) if outs else "no arms evaluated")
    C.write_json(C.EVAL / "reports" / "verdict.json", M.clean(out))
    (C.EVAL / "reports" / "verdict.md").write_text(to_md(out))
    print(to_md(out))


if __name__ == "__main__":
    main()
