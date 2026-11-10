"""The v10 scene rows that may train, in one place, so the holdout, prep and the eval builder
agree. Reads `processed/v10_scenes_748/captions_v10_windows.jsonl` (compile_captions_v10.py)
and applies the decisions of Training_Approach_v10.md section 0:

- `trainable` rows only (U3 camera moves in; U5 lookalikes, i.e. Fixer, out);
- Q4: `PP_CuteFit-BASE.mp4` (29.97 fps) out;
- Q5: any window whose span lists an extra other than the black-and-white penguin out
  (unnamed side characters, unanswered by the client);
- every row's curation file must still hash to the md5 it was compiled from.

    from rows_v10 import load_rows
    rows, dropped = load_rows(data_root)      # dropped: {reason: [rows]}
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTIONS = Path("processed/v10_scenes_748/captions_v10_windows.jsonl")
DROP_SKITS = {"iteration_7/293_videos/PP_CuteFit-BASE.mp4": "q4_29.97fps"}
ACCEPTED_EXTRAS = ("black-and-white penguin",)     # the client accepted it (U5, 2026-11-05)
LEADS = {"pax", "polly"}


def curation(path, _cache={}):
    if path not in _cache:
        raw = (HERE / path).read_bytes()
        _cache[path] = (json.loads(raw), hashlib.md5(raw).hexdigest())
    return _cache[path]


def unnamed_extras(d, span):
    """Descriptions of the span's extras that the client has not accepted."""
    cast = d.get("cast", {})
    out = []
    for c in span.get("characters", []):
        if c["id"] in LEADS:
            continue
        desc = (cast.get(c["id"]) or {}).get("desc", c["id"])
        if not any(a in desc for a in ACCEPTED_EXTRAS):
            out.append(desc)
    return out


def load_rows(data_root):
    rows = [json.loads(line) for line in open(Path(data_root) / CAPTIONS)]
    keep, dropped = [], defaultdict(list)
    for r in rows:
        d, md5 = curation(r["curation"])
        if md5 != r["curation_md5"]:
            raise SystemExit(f"{r['curation']} changed since the captions were compiled: "
                             "re-run scenes/compile_captions_v10.py")
        if not r["trainable"]:
            dropped["not_trainable_" + r["set"]].append(r)
        elif r["skit"] in DROP_SKITS:
            dropped[DROP_SKITS[r["skit"]]].append(r)
        elif unnamed_extras(d, d["spans"][r["span"]]):
            dropped["q5_unnamed_extra"].append(r)
        else:
            keep.append(r)
    return keep, dict(dropped)


if __name__ == "__main__":
    import os
    keep, dropped = load_rows(os.environ.get("PUDGY_DATA", "/workspace/Data"))
    print(f"{len(keep)} rows train ({sum(r['frames'] for r in keep) / 24 / 60:.1f} min, "
          f"{len({r['skit'] for r in keep})} skits)")
    for k, v in sorted(dropped.items()):
        print(f"  dropped {k}: {len(v)} rows, {len({r['skit'] for r in v})} skits")
