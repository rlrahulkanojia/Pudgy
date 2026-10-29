#!/usr/bin/env python3
"""Recompile the captions of an existing prep_v9 set from the CURRENT asset registry, without
re-decoding or re-compositing any clip (a registry wording change only touches captions).

For every prop clip in the set's manifest, the new caption comes from prep_v9's own
prop_caption(job, background, zoom) with the live registry, i.e. exactly what a full prep run
would write. Two checks guard it:
  * the old caption with only the registry's text changes applied (--old/--new pairs) must equal
    the recompiled caption, character for character;
  * captions of non-prop clips (replay) are copied unchanged and must not contain any --old text.

Writes <out>/dataset_*.workspace.jsonl (video paths unchanged) plus a phone-only text-cache
config, and prints which clips changed.

    python recaption_v9.py --src /workspace/data_v9c --out /workspace/data_v9c2 \
        --pair "OLD PHRASE" "NEW PHRASE" --pair "OLD CLAUSE" "NEW CLAUSE"
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval_v9"))
import common_v9 as C  # noqa: E402  (imports prep_v8 as C.P and prep_v9 as C.P9 with the live registry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prep_v9 set dir (with manifest.json and *.workspace.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("OLD", "NEW"),
                    help="registry text change, applied to old captions as the cross-check")
    ap.add_argument("--cache", default="/workspace/wan_cache/latents_v9c2", help="cache root for the te config")
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out)
    man = json.loads((src / "manifest.json").read_text())
    jobs = {(j["char"], j["label"], j["angle"]): j for j in C.P9.discover_props({"props_skipped": []})}

    new_caps = {}
    for rec in man["clips"]:
        if rec["kind"] != "prop":
            continue
        job = jobs[(rec["character"], rec["label"], rec["angle"])]
        for o in rec["outputs"]:
            new_caps[o["name"]] = C.P9.prop_caption(job, C.P.BACKGROUNDS[o["background"]][0], o["zoom"])

    out.mkdir(parents=True, exist_ok=True)
    changed, kept, buckets = 0, 0, {}
    for jf in sorted(src.glob("dataset_*.workspace.jsonl")):
        lines = []
        for line in jf.read_text().splitlines():
            j = json.loads(line)
            name = Path(j["video_path"]).name
            if name in new_caps:
                expect = j["caption"]
                for old, new in a.pair:
                    assert expect.count(old) == 1, f"{name}: {old!r} occurs {expect.count(old)}x"
                    expect = expect.replace(old, new)
                assert expect == new_caps[name], f"{name}: recompiled caption differs from the text-change check"
                j["caption"] = new_caps[name]
                changed += 1
                buckets.setdefault(jf.name, []).append(j)
            else:
                assert not any(old in j["caption"] for old, _ in a.pair), f"{name}: non-prop caption has old text"
                kept += 1
            lines.append(json.dumps(j, ensure_ascii=False))
        (out / jf.name).write_text("\n".join(lines) + "\n")
    assert changed == len(new_caps), f"{changed} captions rewritten, manifest lists {len(new_caps)} prop clips"

    # phone-only jsonl + text-cache config (one [[datasets]] per frame bucket, same layout as prep_v9)
    blocks = []
    for name, rows in sorted(buckets.items()):
        f = int(name.split("_f")[-1].split(".")[0])
        pj = out / name.replace(".workspace.jsonl", "_props.workspace.jsonl")
        pj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        blocks.append(f'[[datasets]]\nvideo_jsonl_file = "{pj}"\ncache_directory = "{a.cache}/f{f}"\n'
                      f'target_frames = [{f}]\nframe_extraction = "head"\nnum_repeats = 1\n# {len(rows)} prop clips\n')
    (out / "te_cache_props.toml").write_text(
        "# Text-encoder cache for the recaptioned prop clips only (recaption_v9.py).\n"
        "[general]\nresolution = [1024, 1024]\nbatch_size = 1\nenable_bucket = true\nbucket_no_upscale = true\n\n"
        + "\n".join(blocks))
    (out / "recaption.json").write_text(json.dumps({"src": str(src), "pairs": a.pair, "changed": changed,
                                                    "kept": kept, "registry": C.P9.R.snapshot(C.P9.REG)
                                                    if hasattr(C.P9, "R") else None}, indent=1, default=str))
    print(f"recaptioned {changed} prop clips, kept {kept} other captions -> {out}")


if __name__ == "__main__":
    main()
