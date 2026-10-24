#!/usr/bin/env python3
"""Build the v9c-phone evaluation assets from the raw sources (CPU only, idempotent).

1. Diffusion-classifier clip set  /workspace/eval_v9/holdout/
   a. REGRESSION (the gate): for every v8 label and character, the QF2_R source (holdout,
      never trained) and the QF2_L source (seen), cut at the kind's common bucket (f13
      motion / f21 expression), 1.00x on white - exactly build_assets_v8's geometry, but
      from prep_v9's discovery, so `confused` is the clean V2 re-export and sitting has
      its second take. Candidates: every label of the kind.
   b. PHONE: the QF2_R (holdout) and QF2_L (seen) phone sources, from the start of their
      clean window, f21, 1.00x on white. Candidates: the training caption, and the same
      caption with the prop phrase swapped for a generic phone and other hand-held objects
      (common_v9.BINDING). Held-out loss under the true caption measures learning; the
      argmin over candidates measures whether the NAME is bound to this phone.
   A musubi config is written so the latents are cached by the trainer's own code.

2. Start frames  /workspace/eval_v9/keyframes/
   phone_<char>_<label>_<VIEW>.png - frame 0 of the phone window (QF2_R unseen, FRONT seen)
   neutral_<char>_FRONT.png        - frame 0 of neutral: no phone in frame (text-call, leak)

    PUDGY_DATA=/workspace/Data python finetune/wan/eval_v9/build_assets_v9.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402

P, P9, C8 = C.P, C.P9, C.C8
WHITE = P.BACKGROUNDS["white"][1]


def write_clip(frames, dest):
    if not dest.exists():
        P.write_mp4(P.composite(frames, WHITE), dest)


def main():
    clips = C.HOLDOUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    C.KF.mkdir(parents=True, exist_ok=True)
    pool, props, _rep = P9.discover_v9()
    index, rows, kf = [], {}, {}

    # a. regression set (v8 labels)
    for j in sorted(pool, key=lambda x: (x["char"], x["label"], x["angle"])):
        base = j["angle"].split("_t")[0]
        if base not in ("QF2_R", "QF2_L"):
            continue
        labels = C8.EXPR_LABELS if j["kind"] == "expression" else C8.MOTION_LABELS
        if j["label"] not in labels:
            continue
        n = P.COMMON[j["kind"]]
        gap = P.real_frame_gap(j["src"])
        if gap is not None and gap < n:
            print(f"  ~ skip {j['src'].name}: real frames stop at #{gap} < f{n}"); continue
        name = f"{j['char'].lower()}_{j['label']}_{j['angle']}__white_f{n}.mp4"
        try:
            if not (clips / name).exists():
                write_clip(P.decode_checked(j, n, 1.00)[:n], clips / name)
        except RuntimeError as e:
            print(f"  ✗ {name}: {e}"); continue
        cands = {lab: C8.caption(j["char"], lab, base, nframes=n) for lab in labels}
        index.append({"clip": name, "char": j["char"], "label": j["label"], "kind": j["kind"],
                      "angle": j["angle"], "split": "holdout" if base == "QF2_R" else "seen",
                      "frames": n, "source": str(j["src"]), "candidates": cands})
        rows.setdefault(n, []).append({"video_path": str(clips / name),
                                       "caption": cands[j["label"]]})

    # b. phone set + phone keyframes
    n = P9.PROP_FLOOR
    for j in sorted(props, key=lambda x: (x["char"], x["label"], x["angle"])):
        if j["angle"] not in ("QF2_R", "QF2_L", "FRONT"):
            continue
        ok, why = P9.preflight_prop(j)
        if not ok or j["window"][1] < n:
            print(f"  ~ skip {j['src'].name}: {why or 'window too short'}"); continue
        jj = dict(j, rungs=[n])
        try:
            fr = P9.decode_window(jj, 1.00)
        except RuntimeError as e:
            print(f"  ✗ {j['src'].name}: {e}"); continue
        if j["angle"] in ("QF2_R", "FRONT"):
            dest = C.KF / f"phone_{j['char'].lower()}_{j['label']}_{j['angle']}.png"
            Image.fromarray(P.composite(fr[:1], WHITE)[0]).save(dest)
            kf[dest.name] = {"char": j["char"], "label": j["label"], "view": j["angle"],
                             "source": str(j["src"]), "window_start": j["window"][0],
                             "split": "holdout" if j["angle"] == "QF2_R" else "train"}
        if j["angle"] == "FRONT":
            continue
        name = f"{j['char'].lower()}_{j['label']}_{j['angle']}__white_f{n}.mp4"
        write_clip(fr[:n], clips / name)
        cands = {k: C.phone_caption(j, v) for k, v in C.BINDING.items()}
        index.append({"clip": name, "char": j["char"], "label": "phone", "kind": "prop",
                      "prop_label": j["label"], "angle": j["angle"],
                      "split": "holdout" if j["angle"] == "QF2_R" else "seen", "frames": n,
                      "source": str(j["src"]), "window": j["window"], "candidates": cands})
        rows.setdefault(n, []).append({"video_path": str(clips / name), "caption": cands["phone"]})

    # neutral FRONT start frames (no phone in frame)
    for j in pool:
        if j["label"] == "neutral" and j["angle"] == "FRONT":
            dest = C.KF / f"neutral_{j['char'].lower()}_FRONT.png"
            fr = P.read_rgba_zoomed(j["src"], 1, 1.00)
            Image.fromarray(P.composite(fr, WHITE)[0]).save(dest)
            kf[dest.name] = {"char": j["char"], "label": "neutral", "view": "FRONT",
                             "source": str(j["src"]), "split": "train"}

    lines = ["[general]", f"resolution = [{P.SIZE}, {P.SIZE}]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    for f, rs in sorted(rows.items()):
        jp = C.HOLDOUT / f"holdout_f{f}.jsonl"
        jp.write_text("".join(json.dumps(r) + "\n" for r in rs))
        lines += ["[[datasets]]", f'video_jsonl_file = "{jp}"',
                  f'cache_directory = "{C.HOLDOUT}/cache/f{f}"', f"target_frames = [{f}]",
                  'frame_extraction = "head"', "num_repeats = 1", ""]
    (C.HOLDOUT / "holdout.toml").write_text("\n".join(lines))
    (C.HOLDOUT / "index.json").write_text(json.dumps(index, indent=1))
    (C.KF / "keyframes.json").write_text(json.dumps(kf, indent=1))
    by = {}
    for r in index:
        k = f"{r['split']}/{r['kind']}"
        by[k] = by.get(k, 0) + 1
    print(f"classifier set: {len(index)} clips {dict(sorted(by.items()))} -> {C.HOLDOUT}")
    print(f"keyframes: {len(kf)} -> {C.KF}")


if __name__ == "__main__":
    main()
