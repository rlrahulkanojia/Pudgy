#!/usr/bin/env python3
"""Is expv1 better than v8@550? The pre-registered, paired, statistical answer (PLAN §8b).

Two levels, both on the SAME held-out material for both models:

  Tier 1 — "does the model KNOW the label?" (likelihood; no sampler in the loop)
      The 26 held-out QF2_R source clips. Each model scores every label caption of the
      clip's kind on that clip; correct = the true caption explains it best. v8's per-clip
      rows: v8/eval/dcls/v8@550_k2.json (f13 motion / f21 expression); expv1's:
      eval/dcls/<config>__A_k2.json (f13 / f17). Paired by (character, label).
  Tier 2 — "does the model SHOW the label?" (generation)
      Clips generated from the same start frame (neutral frame 0 at FRONT and QF2_R, white),
      same (character, label, view), seed 42: 13 labels x 2 characters x 2 views = 52
      pairs. Both are scored by the same code at the same window (H2H_WINDOW: f13 motion,
      f17 expression; v8's f21 clips are trimmed, its 1024^2 frames resized to 512^2) against
      the same real-clip bank (leave-angle-out 1-NN, same character, the kind's labels).

Endpoints (fixed before any expv1 checkpoint exists):
  PRIMARY (superiority, one-sided exact McNemar, Holm-adjusted over the two):
    P1  Tier-1 held-out accuracy, pooled over both kinds (26 pairs)
    P2  Tier-2 CLIP 1-NN label accuracy (52 pairs)
  NON-INFERIORITY (must all hold for "better"; margins are the tolerance, not the target):
    N1  blinded style rubric pass rate: 90% CI lower bound of (expv1 - v8) >= -0.10,
        and expv1 colour correct on 100% of clips
    N2  motion (motion clips): 90% CI lower bound of mean log(expv1 ratio / v8 ratio) >=
        log 0.8, where ratio = motion energy / the label's source median at the window
    N3  camera: 90% CI upper bound of mean (|zoom_rel| expv1 - |zoom_rel| v8) <= +0.01
    N4  late-frame identity: 90% CI lower bound of mean (expv1 - v8) DINOv2 sim >= -0.05
    N5  ground: expv1 median corner drift <= 5/255 (PLAN §9 V6, absolute)
  SECONDARY (reported, not gated): per-kind accuracies, DINOv2 1-NN, mean true-label rank
  (Wilcoxon), seen-angle (QF2_L) Tier-1, motion ratio medians.

Verdict:
  BETTER          every N holds, at least one primary significant after Holm (p < 0.05),
                  and neither primary's point estimate favours v8
  WORSE           v8 significantly better on a primary (Holm, one-sided p < 0.05)
  NOT BETTER      a non-inferiority gate fails (expv1 trades away something v8 has)
  PROBABLY BETTER every N holds, both primaries favour expv1 with posterior P(better) >= 0.80,
                  but nothing significant: the data cannot prove it at this n
  NO DETECTABLE DIFFERENCE   otherwise
  (INTERIM: Tier 1 only, when no Tier-2 clips exist yet; never more than an early read.)
"Replacement candidate" additionally needs expv1's PLAN §9 verdict to be PASS.

Confounds (stated in the output): v8 ran at 1024^2 and saw QF2_R through its v7 parent;
captions are worded differently; v8's Tier-1 expression clips are f21, expv1's f17; one
seed per model in Tier 2; the losses are not comparable across models, only the decisions.

    python eval/compare_v8.py --a-dcls EVAL/dcls/base__A_k2.json --a-name base      # interim
    python eval/compare_v8.py --a-gen EVAL/gen/expv1@550/core --a-dcls ... --rubric-template
    python eval/compare_v8.py --a-gen EVAL/gen/expv1@550/core --a-dcls ... --a-name expv1@550
  Validation (any two v8-format runs): --a-dcls V8_REF/dcls/golden_k2.json
      --a-gen V8_REF/suites/core/golden --a-name golden  (expected: WORSE than v8)
"""
import argparse
import datetime
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv1_common as C  # noqa: E402
import metrics as M  # noqa: E402
import stats as S  # noqa: E402

WHITE = C.GROUNDS["white"]
CLIP_RE = re.compile(r"^(pax|polly)_([a-z]+)_(FRONT|QF2_R)_f(\d+)_s(\d+)\.mp4$")
ALPHA = 0.05
NI = {"rubric": -0.10, "motion_log_ratio": float(np.log(0.8)), "zoom": 0.01, "identity": -0.05,
      "corner_drift": 5.0}
P_BETTER_BAR = 0.80
CONFOUNDS = [
    "v8 generated at 1024^2 (resized to 512^2 here); expv1 at 512^2",
    "v8 saw QF2_R through its v7 parent; expv1 never trained on QF2_R",
    "caption wording differs (v8's one-line template vs expv1's PLAN §5.5 template); same action clauses",
    "Tier 1: v8's expression clips are f21, expv1's f17 (Kandinsky's VAE truncates f21); motion f13 for both",
    "Tier 1 losses are not comparable across models; only each model's per-clip decision is compared",
    "Tier 2: one seed (42) per model; v8's f21 expression clips trimmed to f17",
]


# --------------------------------------------------------------------------- Tier 1
def tier1(a_path, b_path, split="holdout"):
    A = C.load_json(a_path)
    B = C.load_json(b_path)
    if not A or not B:
        return None
    key = lambda r: (r["char"], r["label"])  # noqa: E731
    ra = {key(r): r for r in A["rows"] if r["split"] == split}
    rb = {key(r): r for r in B["rows"] if r["split"] == split}
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    out = {"n": len(common), "missing_in_a": sorted(map(list, set(rb) - set(ra))),
           "missing_in_b": sorted(map(list, set(ra) - set(rb)))}
    for kind in ("all", "expression", "motion"):
        ks = [k for k in common if kind == "all" or ra[k]["kind"] == kind]
        if not ks:
            continue
        ca = np.array([ra[k]["pred"] == ra[k]["label"] for k in ks])
        cb = np.array([rb[k]["pred"] == rb[k]["label"] for k in ks])
        rank_d = [rb[k]["rank"] - ra[k]["rank"] for k in ks]           # > 0: A ranks the truth higher
        out[kind] = {"n": len(ks), "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                     "mcnemar": S.mcnemar(int((ca & ~cb).sum()), int((cb & ~ca).sum())),
                     "acc_diff": S.paired_rate_diff(ca, cb),
                     "rank_a": float(np.mean([ra[k]["rank"] for k in ks])),
                     "rank_b": float(np.mean([rb[k]["rank"] for k in ks])),
                     "rank_wilcoxon_p_a_better": S.wilcoxon_p(rank_d, "greater")}
    out["pairs"] = [{"char": k[0], "label": k[1], "kind": ra[k]["kind"],
                     "a_pred": ra[k]["pred"], "b_pred": rb[k]["pred"],
                     "a_rank": ra[k]["rank"], "b_rank": rb[k]["rank"]} for k in common]
    return out


# --------------------------------------------------------------------------- Tier 2
def clips_in(d):
    out = {}
    for p in Path(d).glob("*.mp4"):
        m = CLIP_RE.match(p.name)
        if m and int(m.group(5)) == 42:
            out[(m.group(1).capitalize(), m.group(2), m.group(3))] = p
    return out


def load_window(p, window):
    fr = M.read_video(p, size=C.SIZE)
    if len(fr) < window:
        raise RuntimeError(f"{p.name}: {len(fr)} frames < window {window}")
    return fr[:window]


def measure(job):
    p, window, h = job
    fr = load_window(p, window)
    m, _ = M.clip_measures(fr, WHITE, h=h)
    return str(p), m


def source_stats(job):
    b, w = job
    fr = M.read_png_dir(b["png_dir"], w)
    m, _ = M.clip_measures(fr, WHITE)
    return b["label"], m["motion_energy"], m["zoom"]


def tier2(a_dir, b_dir, workers, emb):
    A, B = clips_in(a_dir), clips_in(b_dir)
    keys = sorted(set(A) & set(B))
    if not keys:
        return None, []
    bank = [b for b in M.load_bank(C.REFS) if b["label"] in C.LABELS]
    # V5 denominators: each label's source median motion energy and zoom at the window
    jobs = [(b, C.H2H_WINDOW[b["kind"]]) for b in bank if b["frames"] >= C.H2H_WINDOW[b["kind"]]]
    src = {}
    with ProcessPoolExecutor(workers) as ex:
        for lab, e, z in ex.map(source_stats, jobs, chunksize=2):
            src.setdefault(lab, {"e": [], "z": []})
            src[lab]["e"].append(e)
            src[lab]["z"].append(z)
    src_med = {lab: {"energy": float(np.median(v["e"])),
                     "zoom": float(np.median([z for z in v["z"] if z is not None] or [np.nan]))}
               for lab, v in src.items()}
    mjobs = []
    for k in keys:
        w = C.H2H_WINDOW[C.kind_of(k[1])]
        mjobs += [(A[k], w, None), (B[k], w, None)]
    with ProcessPoolExecutor(workers) as ex:
        meas = dict(ex.map(measure, mjobs, chunksize=1))
    banks = {}
    for w in sorted(set(C.H2H_WINDOW.values())):
        ok = [b for b in bank if b["frames"] >= w]
        banks[w] = (ok, *M.bank_embeddings(emb, ok, w, C.REFS / "bank" / "_emb"))
    rows = []
    for k in keys:
        char, label, view = k
        kind = C.kind_of(label)
        w = C.H2H_WINDOW[kind]
        ok, Cv, Dv = banks[w]
        labs = list(C.EXPR_LABELS if kind == "expression" else C.MOTION_LABELS)
        row = {"char": char, "label": label, "kind": kind, "view": view, "window": w}
        for tag, p in (("a", A[k]), ("b", B[k])):
            fr = load_window(p, w)
            qc, qd = emb.clip_embed(fr)
            lc, _ = M.nn_label(qc, Cv, ok, char, labs, view)
            ld, _ = M.nn_label(qd, Dv, ok, char, labs, view)
            m = meas[str(p)]
            masks0 = M.frame_masks(fr[0], None)["char"]
            masks1 = M.frame_masks(fr[-1], None)["char"]
            ident = emb.dino_sim(M.square_crop(fr[0], masks0), M.square_crop(fr[-1], masks1))
            se = src_med.get(label, {}).get("energy")
            sz = src_med.get(label, {}).get("zoom")
            row[tag] = {"file": str(p), "nn_clip": lc, "nn_dino": ld,
                        "correct_clip": lc == label, "correct_dino": ld == label,
                        "motion_energy": m["motion_energy"],
                        "motion_ratio": (m["motion_energy"] / se) if se else None,
                        "zoom_rel": (m["zoom"] - sz) if (m["zoom"] is not None and sz is not None
                                                         and np.isfinite(sz)) else None,
                        "corner_drift": m["corner_drift"], "late_identity": ident}
        rows.append(row)
    return {"src_med": src_med}, rows


def tier2_stats(rows):
    def col(tag, k, kind=None):
        return [r[tag][k] for r in rows if kind is None or r["kind"] == kind]

    out = {"n": len(rows)}
    for key in ("correct_clip", "correct_dino"):
        for kind in (None, "expression", "motion"):
            ca = np.array(col("a", key, kind), bool)
            cb = np.array(col("b", key, kind), bool)
            if len(ca) == 0:
                continue
            out[f"{key}/{kind or 'all'}"] = {"n": int(len(ca)), "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                                             "mcnemar": S.mcnemar(int((ca & ~cb).sum()), int((cb & ~ca).sum())),
                                             "acc_diff": S.paired_rate_diff(ca, cb)}
    mot = [r for r in rows if r["kind"] == "motion" and r["a"]["motion_ratio"] and r["b"]["motion_ratio"]]
    lr = [float(np.log(max(r["a"]["motion_ratio"], 1e-6) / max(r["b"]["motion_ratio"], 1e-6))) for r in mot]
    out["motion_log_ratio"] = {"n": len(lr), "mean": float(np.mean(lr)) if lr else None,
                               "ci90": S.bootstrap_ci(lr, level=0.90),
                               "median_ratio_a": float(np.median([r["a"]["motion_ratio"] for r in mot])) if mot else None,
                               "median_ratio_b": float(np.median([r["b"]["motion_ratio"] for r in mot])) if mot else None,
                               "wilcoxon_p": S.wilcoxon_p(lr)}
    zd = [abs(r["a"]["zoom_rel"]) - abs(r["b"]["zoom_rel"]) for r in rows
          if r["kind"] == "motion" and r["a"]["zoom_rel"] is not None and r["b"]["zoom_rel"] is not None]
    out["zoom_absdiff"] = {"n": len(zd), "mean": float(np.mean(zd)) if zd else None,
                           "ci90": S.bootstrap_ci(zd, level=0.90), "wilcoxon_p": S.wilcoxon_p(zd)}
    idd = [r["a"]["late_identity"] - r["b"]["late_identity"] for r in rows]
    out["identity_diff"] = {"n": len(idd), "mean": float(np.mean(idd)), "ci90": S.bootstrap_ci(idd, level=0.90),
                            "wilcoxon_p": S.wilcoxon_p(idd)}
    out["corner_drift_median"] = {"a": float(np.median(col("a", "corner_drift"))),
                                  "b": float(np.median(col("b", "corner_drift")))}
    return out


# --------------------------------------------------------------------------- rubric (blinded)
def write_blind(frames, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{C.SIZE}x{C.SIZE}", "-r", str(C.FPS), "-i", "-", "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", str(dest)], input=frames.tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-300:])


def rubric_template(rows, a_name, b_name, out_dir):
    order = [(r, t) for r in rows for t in ("a", "b")]
    random.Random(S.SEED).shuffle(order)
    clips, key = [], {}
    for i, (r, t) in enumerate(order, 1):
        rid = f"r{i:03d}"
        dest = out_dir / "clips" / f"{rid}.mp4"
        write_blind(load_window(Path(r[t]["file"]), r["window"]), dest)
        clips.append(M.rubric_row(dest, rid=rid))
        key[rid] = {"model": a_name if t == "a" else b_name, "side": t, "char": r["char"],
                    "label": r["label"], "view": r["view"]}
    tpl = M.rubric_template(clips)
    tpl["blinded"] = True
    return tpl, key


def rubric_stats(rubric, key, rows):
    by = {}
    for row in rubric.get("clips", []):
        k = key.get(row["rid"])
        if k:
            by[(k["side"], k["char"], k["label"], k["view"])] = row
    pa, pb, colour_a = [], [], []
    for r in rows:
        ra = by.get(("a", r["char"], r["label"], r["view"]))
        rb = by.get(("b", r["char"], r["label"], r["view"]))
        if not ra or not rb:
            continue
        va, vb = M.rubric_pass(ra), M.rubric_pass(rb)
        if va is None or vb is None:
            continue
        pa.append(va); pb.append(vb)
        if ra.get("colour_correct") is not None:
            colour_a.append(bool(ra["colour_correct"]))
    if not pa:
        return None
    pa, pb = np.array(pa, bool), np.array(pb, bool)
    return {"n": int(len(pa)), "pass_a": float(pa.mean()), "pass_b": float(pb.mean()),
            "diff": S.paired_rate_diff(pa, pb, level=0.90),
            "mcnemar": S.mcnemar(int((pa & ~pb).sum()), int((pb & ~pa).sum())),
            "colour_a": float(np.mean(colour_a)) if colour_a else None}


# --------------------------------------------------------------------------- verdict
def verdict(t1, t2, rub):
    prim = {}
    if t1 and "all" in t1:
        prim["P1_tier1_acc"] = t1["all"]["mcnemar"]
    if t2 and "correct_clip/all" in t2:
        prim["P2_tier2_clip_nn"] = t2["correct_clip/all"]["mcnemar"]
    adj_a = S.holm({k: v["p_A_better"] for k, v in prim.items()}) if prim else {}
    adj_b = S.holm({k: v["p_B_better"] for k, v in prim.items()}) if prim else {}
    ni = {}
    if t2:
        mlr, zd, idd = t2["motion_log_ratio"], t2["zoom_absdiff"], t2["identity_diff"]
        ni["N2_motion"] = None if mlr["ci90"][0] is None else mlr["ci90"][0] >= NI["motion_log_ratio"]
        ni["N3_camera"] = None if zd["ci90"][1] is None else zd["ci90"][1] <= NI["zoom"]
        ni["N4_identity"] = None if idd["ci90"][0] is None else idd["ci90"][0] >= NI["identity"]
        ni["N5_ground"] = t2["corner_drift_median"]["a"] <= NI["corner_drift"]
    ni["N1_style"] = (None if not rub else
                      (rub["diff"]["ci"][0] is not None and rub["diff"]["ci"][0] >= NI["rubric"]
                       and rub["colour_a"] == 1.0))
    pending = [k for k, v in ni.items() if v is None]
    failed = [k for k, v in ni.items() if v is False]
    sig_a = [k for k, p in adj_a.items() if p < ALPHA]
    sig_b = [k for k, p in adj_b.items() if p < ALPHA]
    favours_b = [k for k, v in prim.items() if v["c"] > v["b"]]
    all_pb = all(v["p_better_posterior"] >= P_BETTER_BAR for v in prim.values()) if prim else False
    if not t2:
        label = "INTERIM (Tier 1 only): " + (
            "A significantly better on P1" if sig_a else
            "B significantly better on P1" if sig_b else "no significant difference on P1")
    elif sig_b:
        label = "WORSE"
    elif failed:
        label = "NOT BETTER"
    elif pending:
        label = "PENDING: " + ", ".join(pending) + " not measured yet (rubric not rated?)"
    elif sig_a and not favours_b:
        label = "BETTER"
    elif all_pb and not favours_b:
        label = "PROBABLY BETTER"
    else:
        label = "NO DETECTABLE DIFFERENCE"
    power = {k: {"n_discordant": v["n_discordant"], "b_needed_for_p<0.05": S.min_b_for_significance(v["n_discordant"])}
             for k, v in prim.items()}
    return {"label": label, "primary": prim, "holm_p_A_better": adj_a, "holm_p_B_better": adj_b,
            "non_inferiority": ni, "ni_failed": failed, "ni_pending": pending, "power": power}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-name", default="expv1")
    ap.add_argument("--a-dcls", help="A's Tier-1 json (expv1: EVAL/dcls/<config>__A_k2.json)")
    ap.add_argument("--a-gen", help="A's seed-42 core clips (<char>_<label>_<view>_f<N>_s42.mp4)")
    ap.add_argument("--b-name", default="v8@550")
    ap.add_argument("--b-dcls", default=str(C.V8_REF / "dcls" / "v8@550_k2.json"))
    ap.add_argument("--b-gen", default=str(C.V8_REF / "suites" / "core" / "v8@550"))
    ap.add_argument("--out", default=None, help="default EVAL/reports/compare_<a>_vs_<b>.json")
    ap.add_argument("--rubric", default=None, help="filled blinded rubric (default next to --out)")
    ap.add_argument("--rubric-template", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    tag = f"{a.a_name}_vs_{a.b_name}".replace("/", "_").replace("@", "-")
    out = Path(a.out or C.EVAL / "reports" / f"compare_{tag}.json")
    rub_path = Path(a.rubric or out.with_name(f"rubric_{tag}.json"))
    key_path = rub_path.with_name(rub_path.stem + "_key.json")

    t1 = tier1(a.a_dcls, a.b_dcls) if a.a_dcls else None
    t1_seen = tier1(a.a_dcls, a.b_dcls, split="seen") if a.a_dcls else None
    t2, rows, extra = None, [], {}
    if a.a_gen:
        emb = M.Embedder()
        extra, rows = tier2(a.a_gen, a.b_gen, a.workers, emb)
        if a.rubric_template:
            tpl, key = rubric_template(rows, a.a_name, a.b_name, rub_path.parent / f"rubric_{tag}")
            C.write_json(rub_path, tpl)
            C.write_json(key_path, key)
            print(f"blinded rubric ({len(tpl['clips'])} clips) -> {rub_path}\nkey -> {key_path} (do not open while rating)")
            return
        t2 = tier2_stats(rows) if rows else None
    rub = None
    if rub_path.exists() and key_path.exists():
        rub = rubric_stats(C.load_json(rub_path, {}), C.load_json(key_path, {}), rows)
    v = verdict(t1, t2, rub)
    s9 = (C.load_json(C.EVAL / "reports" / "verdict.json", {}) or {}).get("outcome")
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "a": {"name": a.a_name, "dcls": a.a_dcls, "gen": a.a_gen},
           "b": {"name": a.b_name, "dcls": a.b_dcls, "gen": a.b_gen},
           "verdict": v["label"], "decision": v,
           "replacement_candidate": bool(v["label"] == "BETTER" and s9 == "PASS"),
           "plan_s9_outcome": s9,
           "tier1_holdout": t1, "tier1_seen": t1_seen, "tier2": t2, "rubric": rub,
           "source_medians_at_window": extra.get("src_med") if extra else None,
           "alpha": ALPHA, "margins": NI, "p_better_bar": P_BETTER_BAR,
           "confounds": CONFOUNDS, "tier2_pairs": rows}
    C.write_json(out, M.clean(res))
    print(f"== {a.a_name} vs {a.b_name}: {v['label']}")
    if t1:
        for kind in ("all", "expression", "motion"):
            if kind in t1:
                x = t1[kind]; m = x["mcnemar"]
                print(f"  Tier1 {kind:<10} acc {x['acc_a']:.3f} vs {x['acc_b']:.3f}  b={m['b']} c={m['c']}  "
                      f"p(A>B)={m['p_A_better']:.3g} p(B>A)={m['p_B_better']:.3g}  P(A better)={m['p_better_posterior']:.2f}")
    if t2:
        for kind in ("all", "expression", "motion"):
            x = t2.get(f"correct_clip/{kind}")
            if x:
                m = x["mcnemar"]
                print(f"  Tier2 CLIP {kind:<10} acc {x['acc_a']:.3f} vs {x['acc_b']:.3f}  b={m['b']} c={m['c']}  "
                      f"p(A>B)={m['p_A_better']:.3g} p(B>A)={m['p_B_better']:.3g}  P(A better)={m['p_better_posterior']:.2f}")
        print(f"  motion log-ratio mean {t2['motion_log_ratio']['mean']}  ci90 {t2['motion_log_ratio']['ci90']}")
        print(f"  identity diff {t2['identity_diff']['mean']:.3f} ci90 {t2['identity_diff']['ci90']}  "
              f"zoom |.| diff {t2['zoom_absdiff']['mean']} ci90 {t2['zoom_absdiff']['ci90']}  "
              f"drift {t2['corner_drift_median']}")
    print(f"  non-inferiority {v['non_inferiority']}  Holm p(A>B) {v['holm_p_A_better']}  p(B>A) {v['holm_p_B_better']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
