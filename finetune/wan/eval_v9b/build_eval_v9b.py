#!/usr/bin/env python3
"""Build the v9b held-out scene evaluation set (CPU only, idempotent).

From the whole skits in scenes/holdout_v9b.json (never trained), using their curation:

1. Tier-1 windows  /workspace/eval_v9b/scenes/
   One f33 window at the start of every kept span of >= 33 frames, encoded exactly as
   prep_v9b encodes training windows (480x848, 24 fps). Candidate captions, all compiled
   by captions_v9b from the same span (same anchors, setting and camera clause), differing
   only in the ACTION:
     true    the window's own beats
     order   the same beats in reverse order (when the window shows >= 2 beats)
     swap    Pax and Polly exchanged in the beats (two-shots only)
     other   the beats of another held-out window with the same leads
   The loss under `true` vs base measures learning; true vs the distractors measures
   whether the model uses the caption (the text-attributable gain, Training_Approach_v9
   section 5.4; RESEARCH_v9c_fixes P2).

2. Tier-2 jobs  /workspace/eval_v9b/scenes/suite_jobs.json
   Up to 2 spans per held-out skit with >= 49 frames: the span's first frame (PNG,
   480x848), the true caption for 49 frames, and the real f49 clip for the motion ratio.

    PUDGY_DATA=/workspace/Data python finetune/wan/eval_v9b/build_eval_v9b.py
"""
import copy
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9b as C  # noqa: E402
import prep_v9b as PB  # noqa: E402
import captions_v9b as CAP  # noqa: E402

T1_FRAMES = 33
T2_FRAMES = 49
T2_PER_SKIT = 2


def swap_leads(text):
    m = {"Pax": "Polly", "Polly": "Pax", "he": "she", "she": "he", "He": "She", "She": "He",
         "his": "her", "her": "his", "His": "Her", "Her": "His", "him": "her"}
    return re.sub(r"\b(Pax|Polly|he|she|He|She|his|her|His|Her|him)\b", lambda x: m[x.group(1)], text)


def with_beats(sp, a, b, texts):
    """A copy of span sp whose beats over [a, b) are `texts`, spread evenly."""
    s2 = copy.deepcopy(sp)
    n = len(texts)
    cuts = [a + round(i * (b - a) / n) for i in range(n + 1)]
    s2["beats"] = [{"start": cuts[i], "end": cuts[i + 1], "text": t} for i, t in enumerate(texts)]
    return s2


def main():
    held = json.loads(PB.HOLDOUT_FILE.read_text())["skits"]
    rows = {r["skit"]: r for r in json.loads(PB.SKITS.read_text())}
    clips = C.SCENES / "clips"; clips.mkdir(parents=True, exist_ok=True)
    kfdir = C.SCENES / "keyframes"; kfdir.mkdir(parents=True, exist_ok=True)
    wins, t2 = [], []
    for skit in held:
        d = json.loads((PB.SCENES / rows[skit]["out"]).read_text())
        need = []
        n2 = 0
        for si, sp in enumerate(d["spans"]):
            if not sp.get("keep") or sp["end"] - sp["start"] < T1_FRAMES:
                continue
            a = sp["start"]
            if PB.lead_frames(sp, a, a + T1_FRAMES) < PB.LEAD_MIN_SHARE * T1_FRAMES:
                continue
            leads = tuple(sorted(c["id"] for c in sp["characters"] if c["id"] in CAP.LEAD_NAMES))
            beats = CAP.pick_beats(sp["beats"], a, a + T1_FRAMES)
            w = {"skit": skit, "span": si, "a": a, "b": a + T1_FRAMES, "leads": leads, "sp": sp, "d": d,
                 "texts": [b["text"] for b in beats],
                 "name": PB.out_name(d, si, a, T1_FRAMES).replace("scene_", "held_")}
            wins.append(w); need.append((w["name"], a, a + T1_FRAMES))
            if sp["end"] - sp["start"] >= T2_FRAMES and n2 < T2_PER_SKIT:
                n2 += 1
                nm = PB.out_name(d, si, a, T2_FRAMES).replace("scene_", "real_")
                t2.append({"tag": nm[:-4], "skit": skit, "span": si, "frames": T2_FRAMES,
                           "prompt": CAP.compile_window(d, sp, a, a + T2_FRAMES),
                           "start": str(kfdir / f"{nm[:-4]}.png"), "real": str(clips / nm),
                           "two_shot": len(leads) == 2})
                need.append((nm, a, a + T2_FRAMES))
        todo = [x for x in need if not (clips / x[0]).exists()] + \
               [x for x in need if x[0].startswith("real_") and not (kfdir / f"{x[0][:-4]}.png").exists()]
        if todo:
            v = PB.decode_skit(PB.RAW / skit, d["frames"])
            for nm, a, b in need:
                if not (clips / nm).exists():
                    PB.write_mp4(v[a:b].copy(), clips / nm)
                if nm.startswith("real_"):
                    Image.fromarray(v[a]).save(kfdir / f"{nm[:-4]}.png")

    index = []
    for i, w in enumerate(wins):
        d, sp, a, b = w["d"], w["sp"], w["a"], w["b"]
        cands = {"true": CAP.compile_window(d, sp, a, b)}
        if len(w["texts"]) >= 2:
            cands["order"] = CAP.compile_window(d, with_beats(sp, a, b, w["texts"][::-1]), a, b)
        if len(w["leads"]) == 2:
            cands["swap"] = CAP.compile_window(d, with_beats(sp, a, b, [swap_leads(t) for t in w["texts"]]), a, b)
        others = [o for o in wins if o is not w and o["leads"] == w["leads"] and o["texts"] != w["texts"]]
        others.sort(key=lambda o: (o["skit"] != w["skit"], abs(wins.index(o) - i)))
        if others:
            cands["other"] = CAP.compile_window(d, with_beats(sp, a, b, others[0]["texts"]), a, b)
        cands = {k: v for k, v in cands.items() if k == "true" or v != cands["true"]}
        index.append({"clip": w["name"], "char": "+".join(w["leads"]), "label": "true", "kind": "scene",
                      "split": "holdout", "frames": T1_FRAMES, "skit": w["skit"], "span": w["span"],
                      "window": [a, b], "candidates": cands})
    (C.SCENES / "index.json").write_text(json.dumps(index, indent=1))
    jl = C.SCENES / f"scenes_f{T1_FRAMES}.jsonl"
    with open(jl, "w") as f:
        for e in index:
            f.write(json.dumps({"video_path": str(clips / e["clip"]), "caption": e["candidates"]["true"]}) + "\n")
    (C.SCENES / "scenes.toml").write_text("\n".join([
        "# v9b held-out scene windows (build_eval_v9b.py): latents for the diffusion classifier.",
        "[general]", f"resolution = [{C.SW}, {C.SH}]", "batch_size = 1", "enable_bucket = true",
        "bucket_no_upscale = true", "", "[[datasets]]", f'video_jsonl_file = "{jl}"',
        f'cache_directory = "{C.SCENES / "cache" / f"f{T1_FRAMES}"}"', f"target_frames = [{T1_FRAMES}]",
        'frame_extraction = "head"', "num_repeats = 1", ""]))
    (C.SCENES / "suite_jobs.json").write_text(json.dumps(t2, indent=1))
    nd = sum(len(e["candidates"]) - 1 for e in index)
    print(f"held-out skits {len(held)}: {len(index)} Tier-1 windows (f{T1_FRAMES}), {nd} distractor captions "
          f"({sum('order' in e['candidates'] for e in index)} order, {sum('swap' in e['candidates'] for e in index)} swap, "
          f"{sum('other' in e['candidates'] for e in index)} other); {len(t2)} Tier-2 jobs (f{T2_FRAMES})")
    print(f"  {C.SCENES}/index.json · scenes.toml · suite_jobs.json")


if __name__ == "__main__":
    main()
