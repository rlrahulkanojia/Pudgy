#!/usr/bin/env python3
"""Build expv1's evaluation references from the raw ProRes sources (CPU; idempotent).

  REFS/tier1/      the diffusion-classifier clip sets (PLAN §7 Tier 1, G0.3):
                   holdout = every QF2_R source (26, never trained), seen = the same
                   (character, label) at QF2_L (trained). One clip per source at its kind's
                   TIER1_LEN (f13 motion / f17 expression), 1.00x, white, 512^2, lossless;
                   index.json lists, per clip, the caption of EVERY label of its kind at the
                   same angle/length/ground (the candidates); tier1.toml caches the latents
                   with the expv1 cacher (VAE directory + --first_frame_image_encode).
  REFS/bank/       the real-clip bank for Tier-2 1-NN (and the v8 head-to-head): every
                   single-character source on white at 512^2, first min(real frames, 25)
                   frames as PNGs, in metrics.load_bank's format. split = holdout for QF2_R.
  REFS/start/      start frames: frame 0 of each character's `neutral` source at FRONT and
                   QF2_R, white, 1.00x (v8's build_assets_v8 rule, so the §8 comparison
                   starts both models from the same drawing).

    python eval/build_refs.py [--workers 32]
    python src/musubi_tuner/kandinsky5_cache_latents.py --dataset_config REFS/tier1/tier1.toml \
        --vae <VAE_DIR> --first_frame_image_encode          # run from the musubi checkout
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv1_common as C  # noqa: E402

BANK_MAX = 25
WHITE = C.GROUNDS["white"]


def tier1(jobs, workers):
    out = C.REFS / "tier1"
    rows, index = {}, []
    todo = []
    for j in jobs:
        ang = j["angle"].split("_t")[0]
        if ang not in (C.HOLDOUT_ANGLE, C.SEEN_ANGLE) or "_t" in j["angle"]:
            continue
        n = C.TIER1_LEN[j["kind"]]
        if j["gap"] < n:
            print(f"  ~ skip {j['char']}/{j['label']}/{ang}: real frames stop at {j['gap']} < f{n}")
            continue
        split = "holdout" if ang == C.HOLDOUT_ANGLE else "seen"
        name = f"{j['char'].lower()}_{j['label']}_{ang}__white_f{n}"
        labels = C.EXPR_LABELS if j["kind"] == "expression" else C.MOTION_LABELS
        cands = {lab: C.caption(j["char"], lab, ang, n) for lab in labels}
        index.append({"clip": name + ".mp4", "char": j["char"], "label": j["label"], "kind": j["kind"],
                      "angle": ang, "split": split, "frames": n, "source": j["src"], "candidates": cands})
        rows.setdefault(n, []).append(name)
        todo.append((j, n, out / f"f{n}" / f"{name}.mp4", cands[j["label"]]))

    def make(t):
        j, n, dest, cap = t
        if not dest.exists():
            try:
                C.write_mp4(C.composite(C.decode_checked(j["src"], n), WHITE), dest)
            except RuntimeError as e:          # A7: tears on every decode -> excluded, reported
                print(f"  x {dest.name}: {e}")
                return dest.name
        dest.with_suffix(".txt").write_text(cap)
        return None

    with ThreadPoolExecutor(workers) as ex:
        torn = {x for x in ex.map(make, todo) if x}
    index = [r for r in index if r["clip"] not in torn]
    rows = {n: [x for x in v if x + ".mp4" not in torn] for n, v in rows.items()}
    lines = ["[general]", f"resolution = [{C.SIZE}, {C.SIZE}]", 'caption_extension = ".txt"',
             "batch_size = 1", "enable_bucket = false", ""]
    for n in sorted(rows):
        lines += ["[[datasets]]", f'video_directory = "{out / f"f{n}"}"', f'cache_directory = "{out / f"cache_f{n}"}"',
                  f"target_frames = [{n}]", 'frame_extraction = "head"', "source_fps = 24.0", ""]
    (out / "tier1.toml").write_text("\n".join(lines))
    C.write_json(out / "index.json", index)
    by = {}
    for r in index:
        by[(r["split"], r["kind"])] = by.get((r["split"], r["kind"]), 0) + 1
    print(f"tier1: {len(index)} clips {by} -> {out}")
    if sum(v for (s, _), v in by.items() if s == "holdout") != 26:
        sys.exit("holdout must be exactly the 26 QF2_R sources (PLAN §5.3)")


def bank(jobs, workers):
    out = C.REFS / "bank"
    rows = []

    def make(j):
        n = min(j["gap"], BANK_MAX)
        if n < min(C.H2H_WINDOW.values()):
            print(f"  ~ bank skip {j['src']}: {n} real frames")
            return None
        bid = f"{j['char'].lower()}_{j['label']}_{j['angle']}"
        d = out / "png" / bid
        if not (d.is_dir() and len(list(d.glob("*.png"))) == n):
            fr = None
            for m in sorted({n, 17, 13}, reverse=True):     # a tear past f13/f17 still leaves a window
                if m > n:
                    continue
                try:
                    fr = C.composite(C.decode_checked(j["src"], m), WHITE)
                    n = m
                    break
                except RuntimeError as e:
                    err = e
            if fr is None:
                print(f"  x bank {bid}: {err}")
                return None
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(fr):
                Image.fromarray(f).save(d / f"{i:04d}.png")
        return {"bank_id": bid, "char": j["char"], "label": j["label"], "kind": j["kind"],
                "angle": j["angle"].split("_t")[0], "frames": n, "source": j["src"],
                "split": "holdout" if j["angle"].startswith(C.HOLDOUT_ANGLE) else "train"}

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(make, jobs) if r]
    C.write_json(out / "index.json", {"clips": rows, "size": C.SIZE, "ground": "white"})
    print(f"bank: {len(rows)} clips -> {out}")


def starts(jobs):
    out = C.REFS / "start"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for char in C.CHARS:
        for view in ("FRONT", C.HOLDOUT_ANGLE):
            j = next((j for j in jobs if j["char"] == char and j["label"] == "neutral" and j["angle"] == view), None)
            if not j:
                sys.exit(f"no neutral source for {char} {view}")
            img = C.composite(C.decode_checked(j["src"], 1), WHITE)[0]
            dest = out / f"{char.lower()}_{view}_white.png"
            Image.fromarray(img).save(dest)
            meta[dest.name] = {"char": char, "view": view, "ground": "white", "zoom": 1.0, "source": j["src"]}
    C.write_json(out / "start.json", meta)
    print(f"start frames: {len(meta)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    jobs, skipped = C.discover()
    with ThreadPoolExecutor(a.workers) as ex:
        for j, g in zip(jobs, ex.map(lambda j: C.real_frame_gap(j["src"]), jobs)):
            j["gap"] = g
    print(f"{len(jobs)} sources ({len(skipped)} skipped at discovery)")
    tier1(jobs, a.workers)
    bank(jobs, a.workers)
    starts(jobs)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
