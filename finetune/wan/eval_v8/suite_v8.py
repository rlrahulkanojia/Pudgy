#!/usr/bin/env python3
"""Tier-2 suites: generate (batched, one model load per config) and score.

  python suite_v8.py --suite core    --configs golden v7fixed v8@300
  python suite_v8.py --suite compose --configs v7naive v7fixed v8@300
  python suite_v8.py --suite all     --configs v8@300 --dry

Suites (clip counts are per config; ~4-6.5 min per clip on an A100 at 1024^2):
  core     controllability. Every label x both characters x {FRONT, QF2_R} start frame x
           seed 42, at the kind's common bucket (f21 expression / f13 motion) - the one
           length where every label of the kind was trained. QF2_R is the holdout angle.
           26 labels-x-chars x 2 views = 52 clips.                                   ~4 h
  compose  motion x expression in ONE prompt (3 actions x 3 expressions x 2 chars = 18).
           The v7 stacking question: v7naive (2G), v7fixed (golden subtracted), v8.  ~1.5 h
  gb       unseen grounds (lavender, sky-blue): 3 labels x 2 chars x 2 grounds = 12.  ~1 h
  gd       duration: the 4 cyclic actions at the f49 tiled rung + 2 held expressions
           at f57, both chars = 12. Scored for repetition, not just length.        ~2 h
  guide    label-contrastive guidance: confused/neutral/happy/laughing with the SAME
           caption at the neutral label as the negative prompt vs the default
           negative, both chars = 16. No training needed - an inference knob.      ~1.5 h
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (P, EVAL, CHARS, EXPR_LABELS, MOTION_LABELS, caption, keyframe,  # noqa: E402
                    kind_of)
import gen_v8  # noqa: E402


def core():
    jobs = []
    for char in CHARS:
        for lab in EXPR_LABELS + MOTION_LABELS:
            for view in ("FRONT", "QF2_R"):
                n = P.COMMON[kind_of(lab)]
                jobs.append({"tag": f"{char.lower()}_{lab}_{view}_f{n}_s42",
                             "prompt": caption(char, lab, view, nframes=n),
                             "start": str(keyframe(char, "FRONT" if view == "FRONT" else view)),
                             "frames": n, "seed": 42,
                             "meta": {"char": char, "label": lab, "kind": kind_of(lab),
                                      "view": view, "seed": 42}})
    return jobs


def compose():
    jobs = []
    for char in CHARS:
        for act in ("walking", "waving", "jumping"):
            for emo in ("angry", "crying", "laughing"):
                m, e = P.MOTION[act], P.EXPRESSIONS[emo]
                prompt = (f"{P.STYLE}{P.ANCHORS[char]}, {m['action']}, while {e['action']}; "
                          f"{m['label']}, {e['label']}; static close-up shot, eye level, "
                          f"{P.FRONT}; {P.BACKGROUNDS['white'][0]}.")
                jobs.append({"tag": f"{char.lower()}_{act}+{emo}_f25_s42", "prompt": prompt,
                             "start": str(keyframe(char)), "frames": 25, "seed": 42,
                             "meta": {"char": char, "motion": act, "expression": emo,
                                      "seed": 42}})
    return jobs


def gb():
    jobs = []
    for char in CHARS:
        for bg in ("lavender", "skyblue"):
            for lab in ("happy", "angry", "waving"):
                n = P.COMMON[kind_of(lab)]
                jobs.append({"tag": f"{char.lower()}_{lab}_{bg}_f{n}_s42",
                             "prompt": caption(char, lab, "FRONT", nframes=n, bg=bg),
                             "start": str(keyframe(char, "FRONT", bg)), "frames": n,
                             "seed": 42, "meta": {"char": char, "label": lab, "bg": bg}})
    return jobs


def gd():
    jobs = []
    for char in CHARS:
        for lab, n in (("walking", 49), ("waving", 49), ("jumping", 49), ("sitting", 49),
                       ("neutral", 57), ("laughing", 57)):
            jobs.append({"tag": f"{char.lower()}_{lab}_f{n}_s42",
                         "prompt": caption(char, lab, "FRONT", nframes=n),
                         "start": str(keyframe(char)), "frames": n, "seed": 42,
                         "meta": {"char": char, "label": lab, "frames": n,
                                  "trained_top": max(P.natural_buckets(
                                      {"kind": kind_of(lab), "label": lab})),
                                  "tiled_rung": n == P.TILE_FRAMES}})
    return jobs


def guide():
    jobs = []
    for char in CHARS:
        for lab in ("confused", "neutral", "happy", "laughing"):
            n = P.COMMON["expression"]
            pos = caption(char, lab, "FRONT", nframes=n)
            neg = caption(char, "neutral" if lab != "neutral" else "happy", "FRONT", nframes=n)
            for mode in ("default", "contrastive"):
                j = {"tag": f"{char.lower()}_{lab}_{mode}_f{n}_s42", "prompt": pos,
                     "start": str(keyframe(char)), "frames": n, "seed": 42,
                     "meta": {"char": char, "label": lab, "mode": mode}}
                if mode == "contrastive":
                    j["neg"] = neg
                jobs.append(j)
    return jobs


SUITES = {"core": core, "compose": compose, "gb": gb, "gd": gd, "guide": guide}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES) + ["all"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    suites = sorted(SUITES) if a.suite == "all" else [a.suite]
    for s in suites:
        jobs = SUITES[s]()
        for cfg in a.configs:
            out = EVAL / "suites" / s / cfg.replace("/", "_").replace(":", "_")
            gen_v8.run(cfg, jobs, out, dry=a.dry)
            (out / "meta.json").write_text(json.dumps(
                {j["tag"]: j["meta"] for j in jobs}, indent=1))
            if a.dry:
                for j in jobs[:3]:
                    print(f"      {j['tag']}: {j['prompt'][:150]}...")
    if not a.dry and not a.no_score:
        import score_v8
        for s in suites:
            score_v8.score_suite(s, a.configs)


if __name__ == "__main__":
    main()
