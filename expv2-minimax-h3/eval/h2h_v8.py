#!/usr/bin/env python3
"""PLAN §8 — the head-to-head with v8@550: the replacement read.

v8@550's `core` clips already exist (Azure pudgy/v8/eval/suites/core/v8@550/, downloaded
by the RUNBOOK): 52 mp4 named <char>_<label>_<view>_f<N>_s42.mp4, f13 motion / f21
expression, 1024^2, seed 42, FRONT + QF2_R, plus jobs_v8@550.json and meta.json.

Comparable set: the 12 labels both models trained x 2 characters x {FRONT, QF2_R}, seed 42
= 48 pairs. The H3 clip (EVAL/gen/<ckpt>/core/<item>__s42) is trimmed to the first 13
(motion) or 21 (expression) frames; v8's is downscaled to 768^2 (Lanczos). Both are scored
by the same code against the same real-clip bank at the same window:

  1-NN     CLIP (primary) and DINOv2, leave-angle-out, same character, the kind's trained
           labels (score.py's rule). Per pair: H3 correct? v8 correct?
           b = H3-only correct, c = v8-only correct; exact two-sided binomial p on b vs c
           (McNemar): p = min(1, 2 * sum_{k <= min(b,c)} C(b+c, k) / 2^(b+c)).
  V2       rubric pass rate per model, from a BLINDED rubric (--rubric-template writes the
           96 clips as same-size, same-length mp4s under shuffled ids, and the key apart).
  V5       motion ratio: clip motion energy / the label's source median at the same window.
  V6       corner drift vs white and zoom (score.py's definitions).

Rule (fixed in the plan):
  H3 PASSES §9, b > c, and H3 rubric rate >= v8's  -> "replacement candidate"
                                                      ("clear" when p < 0.05)
  H3 PASSES §9 but b <= c or a lower rubric rate     -> "not a replacement on today's data"
  H3 does not pass                                   -> "diagnostic only" (A13)
Never a replacement before §12 (running is untested).

Confounds stated in the output: resolutions and prompt wording differ; v8 saw QF2_R through
its v7 parent; H3 clips are trimmed from 22 frames; n = 1 seed per model. The start frames
follow the same rule in both (frame 0 of the character's `neutral` source at that view, on
white, 1.00x: v8's build_assets_v8.py, expv2's eval_items.json), rendered at 1024^2 and
768^2 respectively.

    python expv2-minimax-h3/eval/h2h_v8.py --v8-dir /workspace/v8_core/v8@550 --rubric-template
    #   ... rate EVAL/reports/h2h_rubric.json (blind) ...
    python expv2-minimax-h3/eval/h2h_v8.py --v8-dir /workspace/v8_core/v8@550
-> EVAL/reports/h2h_v8.json
"""
import argparse
import datetime
import math
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv2_common as C  # noqa: E402
import metrics as M  # noqa: E402

WHITE = C.GROUNDS["white"][1]
V8_RE = re.compile(r"^(pax|polly)_([a-z]+)_(FRONT|QF2_R)_f(\d+)_s(\d+)\.mp4$")
WINDOW = {"motion": 13, "expression": 21}
V8_NAME = "v8@550"
CONFOUNDS = [
    "resolution: v8 1024^2 downscaled to 768^2; H3 native 768^2",
    "prompt wording differs (v8's caption template vs H3's three-field format)",
    "v8 saw QF2_R through its v7 parent; H3 never trained on QF2_R",
    "H3 clips trimmed from 22 frames to 13 (motion) / 21 (expression)",
    "n = 1 seed per model (seed 42)",
    "start frames: same rule (neutral frame 0 per character and view), rendered at 1024^2 (v8) vs 768^2 (expv2)",
]


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def v8_clips(v8_dir):
    out = {}
    for p in Path(v8_dir).glob("*.mp4"):
        m = V8_RE.match(p.name)
        if m and int(m.group(5)) == 42:
            out[(m.group(1).capitalize(), m.group(2), m.group(3))] = (p, int(m.group(4)))
    return out


def h3_item(items, char, label, angle):
    for it in items:
        if (it["suite"] == "core" and it["char"] == char and it["label"] == label
                and it["angle"] == angle and it["ground"] == "white"):
            return it
    return None


def pairs(items, ckpt, v8_dir):
    v8 = v8_clips(v8_dir)
    out, missing = [], []
    for char in C.CHARS:
        for label in C.TRAINED_LABELS:
            for angle in ("FRONT", C.HOLDOUT_ANGLE):
                kind = C.kind_of(label)
                it = h3_item(items, char, label, angle)
                h3_dir = C.EVAL / "gen" / ckpt / "core" / f"{it['item_id']}__s42" if it else None
                v = v8.get((char, label, angle))
                rec = {"char": char, "label": label, "kind": kind, "angle": angle,
                       "window": WINDOW[kind], "h3_dir": str(h3_dir) if h3_dir else None,
                       "v8_file": str(v[0]) if v else None, "v8_frames": v[1] if v else None}
                if not (h3_dir and h3_dir.is_dir() and v):
                    missing.append(rec)
                else:
                    out.append(rec)
    return out, missing


def load_pair_frames(rec):
    w = rec["window"]
    h3 = M.read_png_dir(rec["h3_dir"])[:w]
    v8 = M.read_video(rec["v8_file"], size=C.SIZE)[:w]
    return h3, v8


def measure_pair(job):
    rec, h = job
    h3, v8 = load_pair_frames(rec)
    m3, _ = M.clip_measures(h3, WHITE, h=h)
    m8, _ = M.clip_measures(v8, WHITE, h=h)
    return rec, m3, m8, len(h3), len(v8)


def source_energy(job):
    b, w = job
    fr = M.read_png_dir(b["png_dir"], w)
    m, _ = M.clip_measures(fr, WHITE)
    return b["label"], w, m["motion_energy"], m["zoom"]


def write_blind_clip(frames, dest):
    """Same size, same length, same codec for both models, so the rater cannot tell."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", str(C.FPS), "-i", "-", "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", str(dest)],
                       input=frames.tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-400:])


def rubric_template(prs, ckpt):
    rows, key = [], {}
    order = [(r, model) for r in prs for model in ("h3", V8_NAME)]
    random.Random(20261012).shuffle(order)
    vdir = C.EVAL / "reports" / "h2h_rubric_clips"
    for i, (r, model) in enumerate(order, 1):
        rid = f"r{i:03d}"
        h3, v8 = load_pair_frames(r)
        write_blind_clip(h3 if model == "h3" else v8, vdir / f"{rid}.mp4")
        rows.append(M.rubric_row(vdir / f"{rid}.mp4", rid=rid))
        key[rid] = {"model": "h3" if model == "h3" else V8_NAME, "ckpt": ckpt if model == "h3" else V8_NAME,
                    "char": r["char"], "label": r["label"], "angle": r["angle"]}
    tpl = M.rubric_template(rows)
    tpl["blinded"] = True
    return tpl, key


def rubric_rates(rubric, key):
    rates = {}
    for model in ("h3", V8_NAME):
        rs = [row for row in rubric.get("clips", []) if key.get(row["rid"], {}).get("model") == model]
        vals = [M.rubric_pass(row) for row in rs]
        rated = [v for v in vals if v is not None]
        col = [row.get("colour_correct") for row in rs if row.get("colour_correct") is not None]
        rates[model] = {"n": len(rs), "rated": len(rated),
                        "pass_rate": float(np.mean(rated)) if rated else None,
                        "colour_rate": float(np.mean(col)) if col else None}
    return rates


def h3_outcome(ckpt):
    v = C.load_json(C.EVAL / "reports" / "verdict.json", {}) or {}
    arm = "D" if ckpt.startswith("expv2-h3-d") else "A"
    return (v.get(arm) or {}).get("outcome"), arm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v8-dir", required=True, help="local copy of pudgy/v8/eval/suites/core/v8@550/")
    ap.add_argument("--ckpt", default=None, help="H3 ckpt name (default: reports/winner.json[expv2-h3-main])")
    ap.add_argument("--corpus", default=str(C.CORPUS))
    ap.add_argument("--rubric", default=str(C.EVAL / "reports" / "h2h_rubric.json"))
    ap.add_argument("--rubric-template", action="store_true", help="write the blinded rubric + key and exit")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    ckpt = a.ckpt or ((C.load_json(C.EVAL / "reports" / "winner.json", {}) or {})
                      .get("expv2-h3-main", {}).get("winner"))
    if not ckpt:
        sys.exit("no --ckpt and no winner in reports/winner.json")
    items = C.load_json(Path(a.corpus) / "eval_items.json", [])
    prs, missing = pairs(items, ckpt, a.v8_dir)
    print(f"{len(prs)} comparable pairs, {len(missing)} missing", flush=True)
    for r in missing[:10]:
        print(f"  missing: {r['char']} {r['label']} {r['angle']}  h3={r['h3_dir']}  v8={r['v8_file']}")

    rubric_path = Path(a.rubric)
    key_path = rubric_path.with_name(rubric_path.stem + "_key.json")
    if a.rubric_template:
        tpl, key = rubric_template(prs, ckpt)
        C.write_json(rubric_path, tpl)
        C.write_json(key_path, key)
        print(f"blind rubric ({len(tpl['clips'])} clips) -> {rubric_path}\nkey -> {key_path} (do not open while rating)")
        return

    baseline = C.load_json(C.PHASE0 / "source_baseline.json", {}) or {}
    hl = {lab: (baseline.get("labels", {}).get(lab) or {}).get("h") for lab in C.TRAINED_LABELS}
    bank = M.load_bank(a.corpus)
    trained_bank = [b for b in bank if b["label"] in C.TRAINED_LABELS]

    # source medians at each kind's window (V5 denominators)
    src_jobs = [(b, WINDOW[b["kind"]]) for b in trained_bank]
    src = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for lab, w, e, z in ex.map(source_energy, src_jobs, chunksize=1):
            src.setdefault(lab, {"energy": [], "zoom": []})
            src[lab]["energy"].append(e)
            src[lab]["zoom"].append(z)
    src_med = {lab: {"energy": float(np.median(v["energy"])), "zoom": float(np.median(v["zoom"]))}
               for lab, v in src.items()}

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        measured = list(ex.map(measure_pair, [(r, hl.get(r["label"])) for r in prs], chunksize=1))

    emb = M.Embedder()
    banks = {}
    for w in set(WINDOW.values()):
        ok = [b for b in trained_bank if b["frames"] >= w]
        banks[w] = (ok, *M.bank_embeddings(emb, ok, w, C.EVAL / "scores" / "_bank"))

    rows = []
    for rec, m3, m8, n3, n8 in measured:
        ok, Cv, Dv = banks[rec["window"]]
        labs = list(C.EXPR_LABELS) if rec["kind"] == "expression" else list(C.MOTION_TRAINED)
        h3f, v8f = load_pair_frames(rec)
        row = {k: rec[k] for k in ("char", "label", "kind", "angle", "window")}
        for tag, fr, m, n in (("h3", h3f, m3, n3), ("v8", v8f, m8, n8)):
            qc, qd = emb.clip_embed(fr)
            lc, _ = M.nn_label(qc, Cv, ok, rec["char"], labs, rec["angle"])
            ld, _ = M.nn_label(qd, Dv, ok, rec["char"], labs, rec["angle"])
            se = src_med.get(rec["label"], {}).get("energy")
            sz = src_med.get(rec["label"], {}).get("zoom")
            row[tag] = {"frames": n, "nn_clip": lc, "nn_dino": ld,
                        "correct_clip": lc == rec["label"], "correct_dino": ld == rec["label"],
                        "motion_energy": m["motion_energy"],
                        "motion_ratio": (m["motion_energy"] / se) if se else None,
                        "corner_drift": m["corner_drift"], "zoom": m["zoom"],
                        "zoom_rel": (m["zoom"] - sz) if (m["zoom"] is not None and sz is not None) else None,
                        "frozen_fraction": m["frozen_fraction"], "saturation": m["saturation"]}
        rows.append(row)

    def mcnemar(key):
        b = sum(1 for r in rows if r["h3"][key] and not r["v8"][key])
        c = sum(1 for r in rows if r["v8"][key] and not r["h3"][key])
        return {"b_h3_only": b, "c_v8_only": c, "both": sum(1 for r in rows if r["h3"][key] and r["v8"][key]),
                "neither": sum(1 for r in rows if not r["h3"][key] and not r["v8"][key]),
                "p_exact_two_sided": binom_two_sided(b, c),
                "acc_h3": float(np.mean([r["h3"][key] for r in rows])) if rows else None,
                "acc_v8": float(np.mean([r["v8"][key] for r in rows])) if rows else None}

    def med(tag, k, kind=None):
        v = [r[tag][k] for r in rows if (kind is None or r["kind"] == kind) and r[tag][k] is not None]
        return float(np.median(v)) if v else None

    stats = {"clip": mcnemar("correct_clip"), "dino": mcnemar("correct_dino")}
    rubric = C.load_json(rubric_path, {}) or {}
    key = C.load_json(key_path, {}) or {}
    rates = rubric_rates(rubric, key) if rubric and key else None
    outcome, arm = h3_outcome(ckpt)
    b, c, p = stats["clip"]["b_h3_only"], stats["clip"]["c_v8_only"], stats["clip"]["p_exact_two_sided"]
    r3 = rates and rates["h3"]["pass_rate"]
    r8 = rates and rates[V8_NAME]["pass_rate"]
    if outcome is None:
        label = "pending: no §9 verdict for this arm yet (run verdict.py)"
    elif outcome != "PASS":
        label = "diagnostic only: H3 did not pass §9 (read to separate data- from base-limited failures, A13)"
    elif r3 is None or r8 is None:
        label = "pending: rubric not rated"
    elif b > c and r3 >= r8:
        label = "clear replacement candidate" if p < 0.05 else "replacement candidate"
    else:
        label = "not a replacement on today's data"
    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "h3_ckpt": ckpt, "h3_arm": arm, "h3_outcome": outcome, "v8": V8_NAME,
           "v8_dir": str(a.v8_dir), "n_pairs": len(rows), "missing": missing,
           "mcnemar": stats, "rubric": rates,
           "motion_ratio_median": {"h3": med("h3", "motion_ratio", "motion"), "v8": med("v8", "motion_ratio", "motion")},
           "corner_drift_median": {"h3": med("h3", "corner_drift"), "v8": med("v8", "corner_drift")},
           "zoom_rel_median": {"h3": med("h3", "zoom_rel", "motion"), "v8": med("v8", "zoom_rel", "motion")},
           "source_medians_at_window": src_med,
           "label": label,
           "note": "Never a replacement before PLAN §12: H3 does not cover running on today's data.",
           "confounds": CONFOUNDS, "pairs": rows}
    dest = C.EVAL / "reports" / "h2h_v8.json"
    C.write_json(dest, M.clean(out))
    print(f"CLIP 1-NN: H3 {stats['clip']['acc_h3']:.3f} vs v8 {stats['clip']['acc_v8']:.3f}; "
          f"b={b} c={c} p={p:.4f}; rubric h3={r3} v8={r8}\n-> {label}\n-> {dest}")


if __name__ == "__main__":
    main()
