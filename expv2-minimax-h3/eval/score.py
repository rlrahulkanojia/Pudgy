#!/usr/bin/env python3
"""Tier 2 scoring (PLAN §7.2, §7.4, §9): every generated clip of a checkpoint, measured the
way the source baseline measured the real clips (eval/metrics.py), plus CLIP / DINOv2
1-NN label recovery against the real-clip bank at the same frame window (B3).

Reads   EVAL/gen/<ckpt>/<suite>/<item_id>__s<seed>/0000.png ...   (eval/generate.py)
        CORPUS/eval_items.json, CORPUS/bank/, EVAL/phase0/source_baseline.json
Writes  EVAL/scores/<ckpt>/<suite>.json                              (INTERFACES §3)

1-NN (B3). Query = the generated clip's embedding (score_v8's 6-frame back-half mean).
Candidates = bank clips of the SAME character and the same kind's TRAINED labels (running
items: the four trained motions + running), truncated to the generated clip's length.
  nn_label_clip / nn_label_dino      leave-angle-out: every bank clip at the start frame's
                                     angle is dropped for every label (the start frame came
                                     from one of them). PRIMARY - used by V4 and V8.
  nn_label_clip_full / _dino_full    the whole bank. Reported only.

Frozen fraction and twos index use the label's hold threshold h from the source baseline;
without source_baseline.json they are null. Run G0.5 first.

    python expv2-minimax-h3/eval/score.py --ckpt base --suites core,native
    python expv2-minimax-h3/eval/score.py --ckpt expv2-h3-main@250 --suites core,native,gb
    python expv2-minimax-h3/eval/score.py --ckpt expv2-h3-main@all --suites probe
"""
import argparse
import datetime
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv2_common as C  # noqa: E402
import metrics as M  # noqa: E402

SUITES = ("core", "native", "gb", "probe")
ROW_MEASURES = ("motion_energy", "frozen_fraction", "twos_index", "centroid_travel", "zoom",
                "zoom_rel", "head_line_drift", "corner_drift", "ground_spread", "late_identity",
                "saturation", "fill_flatness", "fill_std_global", "ff_psnr", "ff_dino",
                "motion_ratio", "saturation_ratio")


def parse_clip_dir(d):
    item_id, _, seed = d.name.rpartition("__s")
    return item_id, int(seed)


def candidate_labels(item):
    if item["kind"] == "expression":
        return list(C.EXPR_LABELS)
    return list(C.MOTION_TRAINED) + (["running"] if item["label"] == "running" else [])


def baseline_for(baseline, label, frames):
    """The source block a clip is judged against: the label at the clip's window
    (native windows are keyed '<label>@<w>')."""
    if not baseline:
        return None
    if frames in (39, 56):
        b = baseline.get("native", {}).get(f"{label}@{frames}")
        if b:
            return b
    return baseline.get("labels", {}).get(label)


def _med(block, k):
    try:
        return block[k]["median"]
    except (KeyError, TypeError):
        return None


def measure_clip(job):
    """Worker: the mask-based measures of one generated clip (no embeddings)."""
    d, item, seed, h, corpus = job
    frames = M.read_png_dir(d)
    rgb = C.GROUNDS[item["ground"]][1]
    start = M.read_png_dir_one(Path(corpus) / item["start_frame"]) if item.get("start_frame") else None
    m, _ = M.clip_measures(frames, rgb, h=h, start_frame=start)
    return str(d), m


def resolve_ckpts(names):
    """'<run>@all' expands to every ckpt name that has generations; others pass through."""
    out = []
    for n in names:
        if n.endswith("@all"):
            run = n[:-4]
            out += sorted((p.name for p in (C.EVAL / "gen").glob(f"{run}@*") if p.is_dir()),
                          key=lambda s: int(s.split("@")[1]) if s.split("@")[1].isdigit() else 0)
        else:
            out.append(n)
    return out


def summarise_suite(rows):
    by_ka = defaultdict(list)
    for r in rows:
        kind = "running" if r["label"] == "running" else r["kind"]
        by_ka[f"{kind}/{r['angle']}"].append(r)
        by_ka[f"{kind}/all"].append(r)

    def acc(rs, key):
        v = [r[key] == r["label"] for r in rs if r.get(key) is not None]
        return (float(np.mean(v)) if v else None), len(v)

    out = {"by_kind_angle": {}, "by_label": {}}
    for k, rs in sorted(by_ka.items()):
        e = {"n": len(rs)}
        for key in ("nn_label_clip", "nn_label_dino", "nn_label_clip_full", "nn_label_dino_full"):
            e[f"acc_{key[9:]}"], e[f"n_{key[9:]}"] = acc(rs, key)
        out["by_kind_angle"][k] = e
    by_label = defaultdict(list)
    for r in rows:
        by_label[(r["label"], r["angle"])].append(r)
    for (lab, ang), rs in sorted(by_label.items()):
        out["by_label"][f"{lab}/{ang}"] = {"n": len(rs), **{
            k: M.summarise([r.get(k) for r in rs])["median"] for k in ROW_MEASURES}}
    motion = [r for r in rows if r["label"] in C.MOTION_TRAINED]
    out["motion_energy_median"] = M.summarise([r["motion_energy"] for r in motion])["median"]
    out["saturation_ratio_median"] = M.summarise([r.get("saturation_ratio") for r in rows])["median"]
    out["saturation_median"] = M.summarise([r["saturation"] for r in rows])["median"]
    return out


def score(ckpt, suite, items, baseline, bank, emb, workers, corpus):
    d = C.EVAL / "gen" / ckpt / suite
    dirs = sorted(p for p in d.glob("*__s*") if p.is_dir() and any(p.glob("*.png")))
    if not dirs:
        print(f"  [{ckpt}/{suite}] no generations under {d}")
        return None
    jobs, meta = [], {}
    for p in dirs:
        item_id, seed = parse_clip_dir(p)
        if item_id not in items:
            print(f"  skip {p.name}: item not in eval_items.json")
            continue
        it = items[item_id]
        src = baseline_for(baseline, it["label"], it["frames"])
        h = src.get("h") if src else None
        jobs.append((p, it, seed, h, corpus))
        meta[str(p)] = (item_id, seed, it, src)
    print(f"  [{ckpt}/{suite}] {len(jobs)} clips", flush=True)
    measured = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, m in ex.map(measure_clip, jobs, chunksize=1):
            measured[path] = m

    bank_cache = {}
    rows = []
    for path, m in measured.items():
        item_id, seed, it, src = meta[path]
        row = {"item_id": item_id, "seed": seed, "char": it["char"], "label": it["label"],
               "kind": it["kind"], "angle": it["angle"], "ground": it["ground"],
               "frames": m["frames"], **{k: m.get(k) for k in m}}
        sz = _med(src, "zoom")
        row["zoom_rel"] = (m["zoom"] - sz) if (m["zoom"] is not None and sz is not None) else None
        se, ss = _med(src, "motion_energy"), _med(src, "saturation")
        row["motion_ratio"] = m["motion_energy"] / se if se else None
        row["saturation_ratio"] = m["saturation"] / ss if (ss and m["saturation"] is not None) else None
        for k in ("nn_label_clip", "nn_label_dino", "nn_label_clip_full", "nn_label_dino_full"):
            row[k] = None
        if emb is not None:
            frames = M.read_png_dir(path)
            w = len(frames)
            if w not in bank_cache:
                ok = [b for b in bank if b["frames"] >= w or (b["label"] == "running" and w <= 22)]
                Cv, Dv = M.bank_embeddings(emb, ok, w, C.EVAL / "scores" / "_bank")
                bank_cache[w] = (ok, Cv, Dv)
            ok, Cv, Dv = bank_cache[w]
            qc, qd = emb.clip_embed(frames)
            labs = candidate_labels(it)
            row["nn_label_clip"], row["nn_sim_clip"] = M.nn_label(qc, Cv, ok, it["char"], labs, it["angle"])
            row["nn_label_dino"], row["nn_sim_dino"] = M.nn_label(qd, Dv, ok, it["char"], labs, it["angle"])
            row["nn_label_clip_full"], _ = M.nn_label(qc, Cv, ok, it["char"], labs)
            row["nn_label_dino_full"], _ = M.nn_label(qd, Dv, ok, it["char"], labs)
            masks0 = M.frame_masks(frames[0], None)
            masks1 = M.frame_masks(frames[-1], None)
            row["late_identity"] = emb.dino_sim(M.square_crop(frames[0], masks0["char"]),
                                                M.square_crop(frames[-1], masks1["char"]))
            if it.get("start_frame"):
                sf = M.read_png_dir_one(Path(corpus) / it["start_frame"])
                union = M.frame_masks(sf, None)["char"] | masks0["char"]
                row["ff_dino"] = emb.dino_sim(M.square_crop(sf, union), M.square_crop(frames[0], union))
        rows.append(row)
    rows.sort(key=lambda r: (r["item_id"], r["seed"]))
    out = {"ckpt": ckpt, "suite": suite,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "embed": emb.revisions() if emb else None,
           "source_baseline": bool(baseline),
           "clips": rows, "summary": summarise_suite(rows)}
    dest = C.EVAL / "scores" / ckpt / f"{suite}.json"
    C.write_json(dest, M.clean(out))
    s = out["summary"]["by_kind_angle"]
    msg = ", ".join(f"{k} clip={v['acc_clip']}" for k, v in s.items() if k.endswith("QF2_R"))
    print(f"  -> {dest}  {msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="ckpt names: base, <run>@<optstep>, <run>@all")
    ap.add_argument("--suites", default="core,native,gb,probe")
    ap.add_argument("--corpus", default=str(C.CORPUS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-embed", action="store_true", help="skip CLIP/DINOv2 (1-NN, ff_dino, late identity)")
    a = ap.parse_args()

    corpus = Path(a.corpus)
    items = {it["item_id"]: it for it in C.load_json(corpus / "eval_items.json", [])}
    if not items:
        sys.exit(f"no eval items in {corpus / 'eval_items.json'}")
    baseline = C.load_json(C.PHASE0 / "source_baseline.json")
    if not baseline:
        print("WARNING: no phase0/source_baseline.json - frozen/twos/ratios will be null (run G0.5)")
    bank = M.load_bank(corpus)
    emb = None if a.no_embed else M.Embedder()
    for ckpt in resolve_ckpts(a.ckpt):
        for suite in a.suites.split(","):
            if suite not in SUITES:
                sys.exit(f"unknown suite {suite}")
            score(ckpt, suite, items, baseline, bank, emb, a.workers, corpus)


if __name__ == "__main__":
    main()
