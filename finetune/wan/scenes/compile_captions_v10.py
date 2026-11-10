#!/usr/bin/env python3
"""Compile the v10 training captions: one row per training window, cut exactly as prep_v9b.py
cuts them (BUCKETS f81/f49/f33/f21 from each span's start, plus an end window; a window with a
lead in under half its frames is skipped). Each caption comes from captions_v9b.compile_window
and passes prep_v9b's lint.

Three sets; `trainable` says which ones train under the decisions taken so far
(Training_Approach_v10.md section 7):
  kept         kept spans (camera locked)                         trainable
  u3_camera    dropped camera_move spans filled with every field  trainable (U3 = yes, 2026-11-04)
  u5_lookalike dropped lookalike spans filled as the lead         not trainable until the client
                                                                  answers (U5; only Fixer is left)

    PY=<python with numpy>  PUDGY_DATA=<Data root>
    $PY scenes/compile_captions_v10.py   # writes $PUDGY_DATA/processed/v10_scenes_748/captions_v10_windows.jsonl
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prep_v9b as PV  # noqa: E402  (windows, lead_frames, lint_caption, LEAD_MIN_SHARE)
import captions_v9b as CAP  # noqa: E402
import curate_check as CK  # noqa: E402

INDEX = HERE / "skits_v10.json"
# A derived training set lives in Data/processed, named for the run that consumes it (Data/README.md).
OUT = PV.P.DATA / "processed" / "v10_scenes_748" / "captions_v10_windows.jsonl"
TRAINABLE = {"kept": True, "u3_camera": True, "u5_lookalike": False}   # U3 = yes; U5 answered 2026-11-05 except Fixer


def span_set(sp):
    if sp.get("keep"):
        return "kept"
    if sp.get("beats") and sp.get("drop_reason") == "camera_move":
        return "u3_camera"
    if sp.get("beats") and sp.get("drop_reason") == "lookalike":
        return "u5_lookalike"
    return None


def main():
    idx = json.loads(INDEX.read_text())["skits"]
    rows, skipped, probs, bad = [], collections.Counter(), [], []
    for it in idx:
        f = HERE / it["out"]
        errs, _ = CK.check(f)
        if errs:
            bad.append(f"{f.name}: {errs[0]}"); continue
        raw = f.read_bytes(); d = json.loads(raw)
        for si, sp in enumerate(d["spans"]):
            s = span_set(sp)
            if not s:
                continue
            for wa, wb in PV.windows(sp):
                n = wb - wa
                if PV.lead_frames(sp, wa, wb) < PV.LEAD_MIN_SHARE * n:
                    skipped[f"{s}: no lead for half the window"] += 1; continue
                cap = CAP.compile_window(d, sp, wa, wb)
                pr = PV.lint_caption(cap)
                if pr:
                    probs.append(f"{d['skit']} [{wa},{wb}) {s}: {pr[0]}")
                rows.append({"set": s, "trainable": TRAINABLE[s], "skit": d["skit"], "curation": it["out"],
                             "curation_md5": hashlib.md5(raw).hexdigest(), "span": si,
                             "a": wa, "b": wb, "frames": n, "fps": it["fps"],
                             "camera": sp.get("camera", "static"), "instagram": it.get("instagram", ""),
                             "two_shot": sum(c["id"] in CAP.LEAD_NAMES for c in sp["characters"]) == 2,
                             "caption": cap})
    if bad:
        sys.exit("curation errors (run curate_check.py):\n  " + "\n  ".join(bad))
    if probs:
        sys.exit("caption lint failed:\n  " + "\n  ".join(probs[:20]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} windows -> {OUT}  (trainable now: {sum(r['trainable'] for r in rows)})")
    for s in ("kept", "u3_camera", "u5_lookalike"):
        sel = [r for r in rows if r["set"] == s]
        if not sel:
            continue
        bk = collections.Counter(r["frames"] for r in sel)
        secs = sum(r["frames"] / r["fps"] for r in sel)
        print(f"  {s:<13} {len(sel):>4} windows from {len({r['skit'] for r in sel}):>3} skits, "
              f"{secs / 60:5.1f} min, buckets {dict(sorted(bk.items()))}, two-shots {sum(r['two_shot'] for r in sel)}")
    for k, v in skipped.items():
        print(f"  skipped {v} windows ({k})")


if __name__ == "__main__":
    main()
