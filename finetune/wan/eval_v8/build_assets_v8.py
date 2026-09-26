#!/usr/bin/env python3
"""Build the v8 evaluation assets from the raw sources (CPU only, idempotent).

1. Start frames  eval_v8/keyframes/<char>_<VIEW>_<bg>.png
   Frame 0 of each character's `neutral` source at VIEWS, composited at 1.00x onto white,
   plus FRONT on the two never-trained grounds (lavender, sky-blue) for G-B. QF2_R is the
   holdout angle - its neutral source was never trained, so that start frame is unseen.

2. Diffusion-classifier clip set  eval_v8/holdout/
   Every holdout source (QF2_R, 26 of them) and, as the SEEN control, the same
   (character, label) at the mirror angle QF2_L, which WAS trained. Each clip is cut at
   its kind's common bucket (f13 motion / f21 expression) at 1.00x on white, i.e. exactly
   the geometry every label was trained at. A musubi config is written so the latents
   are cached by the trainer's own code, and index.json lists, per clip, the candidate
   caption for EVERY label of its kind at the same view/length/ground - the classifier
   scores the true caption against all of them.

    python eval_v8/build_assets_v8.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (P, KF, HOLDOUT, VIEWS, UNSEEN_BG, EXPR_LABELS, MOTION_LABELS,  # noqa: E402
                    caption, kind_of)

SET = Path("/workspace/data_v8")


def sources():
    """(char, label, angle) -> {src, md5, split} from the v8 manifest."""
    man = json.loads((SET / "manifest.json").read_text())
    out = {}
    for c in man["clips"]:
        out[(c["character"], c["label"], c["angle"].split("_t")[0])] = {
            "src": c["source"], "split": "train"}
    for h in man["holdout"]:
        out[(h["character"], h["label"], h["angle"].split("_t")[0])] = {
            "src": h["source"], "split": "holdout"}
    return out


def first_frame(src, zoom=1.00):
    return P.read_rgba_zoomed(Path(src), 1, zoom)[0]


def build_keyframes(src_map):
    KF.mkdir(parents=True, exist_ok=True)
    meta = {}
    for char in ("Pax", "Polly"):
        for view in VIEWS:
            s = src_map.get((char, "neutral", view))
            if not s:
                print(f"  ! no neutral source for {char} {view}"); continue
            fr = first_frame(s["src"])
            bgs = ["white"] + (list(UNSEEN_BG) if view == "FRONT" else [])
            for bg in bgs:
                rgb = P.BACKGROUNDS[bg][1] if bg in P.BACKGROUNDS else UNSEEN_BG[bg][1]
                img = P.composite(fr[None], rgb)[0]
                dest = KF / f"{char.lower()}_{view}_{bg}.png"
                Image.fromarray(img).save(dest)
                meta[dest.name] = {"char": char, "view": view, "bg": bg, "zoom": 1.0,
                                   "source": s["src"], "split": s["split"]}
    (KF / "keyframes.json").write_text(json.dumps(meta, indent=1))
    print(f"keyframes: {len(meta)} -> {KF}")


def build_holdout(src_map):
    clips = HOLDOUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    index, rows = [], {}
    for (char, label, angle), s in sorted(src_map.items()):
        if angle not in ("QF2_R", "QF2_L"):
            continue
        if label not in EXPR_LABELS and label not in MOTION_LABELS:
            continue
        kind = kind_of(label)
        n = P.COMMON[kind]
        gap = P.real_frame_gap(Path(s["src"]))
        if gap is not None and gap < n:
            print(f"  ~ skip {char}/{label}/{angle}: real frames stop at #{gap} < f{n}")
            continue
        split = "holdout" if angle == "QF2_R" else "seen"
        name = f"{char.lower()}_{label}_{angle}__white_f{n}.mp4"
        dest = clips / name
        if not dest.exists():
            job = {"src": Path(s["src"]), "kind": kind, "label": label}
            try:
                fr = P.decode_checked(job, n, 1.00)
            except RuntimeError as e:
                print(f"  ✗ {name}: {e}"); continue
            P.write_mp4(P.composite(fr[:n], P.BACKGROUNDS["white"][1]), dest)
        labels = EXPR_LABELS if kind == "expression" else MOTION_LABELS
        cands = {lab: caption(char, lab, angle, nframes=n) for lab in labels}
        index.append({"clip": name, "char": char, "label": label, "kind": kind,
                      "angle": angle, "split": split, "frames": n, "source": s["src"],
                      "candidates": cands})
        rows.setdefault(n, []).append({"video_path": str(dest), "caption": cands[label]})
    lines = ["[general]", "resolution = [1024, 1024]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    for n, rs in sorted(rows.items()):
        jp = HOLDOUT / f"holdout_f{n}.jsonl"
        jp.write_text("".join(json.dumps(r) + "\n" for r in rs))
        lines += ["[[datasets]]", f'video_jsonl_file = "{jp}"',
                  f'cache_directory = "{HOLDOUT}/cache/f{n}"', f"target_frames = [{n}]",
                  'frame_extraction = "head"', "num_repeats = 1", ""]
    (HOLDOUT / "holdout.toml").write_text("\n".join(lines))
    (HOLDOUT / "index.json").write_text(json.dumps(index, indent=1))
    by = {}
    for r in index:
        by[r["split"]] = by.get(r["split"], 0) + 1
    print(f"classifier set: {len(index)} clips {by} -> {HOLDOUT}")


if __name__ == "__main__":
    m = sources()
    build_keyframes(m)
    build_holdout(m)
