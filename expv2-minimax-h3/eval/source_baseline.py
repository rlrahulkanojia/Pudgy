#!/usr/bin/env python3
"""G0.5 — the source baseline: the real clips measured exactly as generations will be.

Reads the real-clip bank (CORPUS/bank, INTERFACES §1: every usable source once, on white,
zoom 1.00, at its longest legal rung) and measures every PLAN G0.5 quantity per label, on
the frames each suite uses:

    window 22      every trained label (`core`, `gb`, `probe`); running at its 17 frames
    window 39/56   angry, laughing, blushing @39 and neutral @56 (`native`), where the
                   source reaches it

The hold threshold h is fixed here, per label, as 25% of the median MOVING-pair difference
over all of that label's clips in the window (metrics.hold_threshold). Frozen fraction and
twos index of every generation are computed with the same h, so sources and generations
are judged by one rule. These numbers are the denominators of PLAN §9 (B6).

CPU only. DINOv2 (late-frame identity) runs if torch + transformers are importable and
--no-embed is not given.

    python expv2-minimax-h3/eval/source_baseline.py --workers 16
    python expv2-minimax-h3/eval/source_baseline.py --no-embed          # Mac / no torch
-> EVAL/phase0/source_baseline.json
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

WHITE = C.GROUNDS["white"][1]
MEASURES = ("motion_energy", "frozen_fraction", "twos_index", "centroid_travel",
            "shadow_width0", "zoom", "head_line_drift", "ground_spread", "late_identity",
            "saturation", "fill_flatness", "fill_std_global", "char_area0")


def windows_for(entry):
    """The suite windows this bank clip contributes to."""
    if entry["label"] == "running":
        return [min(entry["frames"], 17)]
    ws = [22] if entry["frames"] >= 22 else []
    for w, labels in C.LONG_RUNG_LABELS.items():
        if entry["label"] in labels and entry["frames"] >= w:
            ws.append(w)
    return ws


def measure(job):
    """Worker: one bank clip, every window. h-dependent measures are filled in later."""
    entry, wins = job
    frames = M.read_png_dir(entry["png_dir"])
    masks = M.clip_masks(frames, WHITE)
    out = []
    for w in wins:
        m, pd = M.clip_measures(frames[:w], WHITE, h=None, masks=masks[:w])
        out.append((w, m, pd.tolist()))
    return entry["bank_id"], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(C.CORPUS))
    ap.add_argument("--out", default=str(C.PHASE0 / "source_baseline.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-embed", action="store_true", help="skip DINOv2 late-frame identity")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N bank clips")
    a = ap.parse_args()

    bank = M.load_bank(a.corpus)
    if a.limit:
        bank = bank[:a.limit]
    jobs = [(b, windows_for(b)) for b in bank]
    jobs = [j for j in jobs if j[1]]
    print(f"bank: {len(bank)} clips, {len(jobs)} with a suite window", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (bid, rows) in enumerate(ex.map(measure, jobs, chunksize=1), 1):
            results[bid] = rows
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    by_id = {b["bank_id"]: b for b in bank}
    # h per (label, window) from every clip's pair diffs in that window
    pds = defaultdict(list)
    for bid, rows in results.items():
        for w, m, pd in rows:
            pds[(by_id[bid]["label"], w)].extend(pd)
    h = {k: M.hold_threshold(v) for k, v in pds.items()}
    # the label's h at its main window applies to every window of that label
    h_label = {lab: h[(lab, w)] for (lab, w) in h if w in (22, 17)}

    emb = None
    if not a.no_embed:
        try:
            emb = M.Embedder()
        except Exception as e:  # noqa: BLE001
            print(f"no embedder ({e}); late_identity skipped", flush=True)

    clips = []
    for bid, rows in results.items():
        b = by_id[bid]
        for w, m, pd in rows:
            hl = h_label.get(b["label"], h.get((b["label"], w)))
            pd = np.array(pd)
            m["frozen_fraction"] = M.frozen_fraction(pd, hl)
            m["twos_index"] = M.twos_index(pd, hl)
            if emb is not None:
                fr = M.read_png_dir(b["png_dir"], w)
                m0, m1 = M.frame_masks(fr[0], WHITE), M.frame_masks(fr[-1], WHITE)
                m["late_identity"] = emb.dino_sim(M.square_crop(fr[0], m0["char"]),
                                                  M.square_crop(fr[-1], m1["char"]))
            clips.append({"bank_id": bid, "char": b["char"], "label": b["label"],
                          "kind": b["kind"], "angle": b["angle"], "split": b["split"],
                          "untrained": b["untrained"], "window": w, **m})

    def block(rows, label, w):
        out = {"kind": C.kind_of(label), "untrained": label == "running", "window": w,
               "n": len(rows), "h": h_label.get(label, h.get((label, w)))}
        for k in MEASURES:
            out[k] = M.summarise([r[k] for r in rows])
        return out

    labels, native = {}, {}
    for lab in C.ALL_LABELS:
        main_w = 17 if lab == "running" else 22
        rows = [c for c in clips if c["label"] == lab and c["window"] == main_w]
        if rows:
            labels[lab] = block(rows, lab, main_w)
        for w, labs in C.LONG_RUNG_LABELS.items():
            if lab in labs:
                rows = [c for c in clips if c["label"] == lab and c["window"] == w]
                if rows:
                    native[f"{lab}@{w}"] = block(rows, lab, w)

    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "bank": str(Path(a.corpus) / "bank"), "n_bank": len(bank),
           "hold_fraction": M.HOLD_FRACTION, "hold_eps": M.HOLD_EPS,
           "embed": emb.revisions() if emb else None,
           "definitions": "eval/metrics.py; frozen/twos use labels[<label>].h",
           "labels": labels, "native": native, "clips": clips}
    C.write_json(a.out, M.clean(out))
    print(f"-> {a.out}")
    for lab, r in labels.items():
        def med(k):
            v = r[k]["median"]
            return "  -  " if v is None else f"{v:6.3f}"
        print(f"  {lab:10s} n={r['n']:2d} w={r['window']} h={r['h']:.3f} energy={med('motion_energy')} "
              f"frozen={med('frozen_fraction')} twos={med('twos_index')} zoom={med('zoom')} "
              f"sat={med('saturation')} flat={med('fill_flatness')}")


if __name__ == "__main__":
    main()
