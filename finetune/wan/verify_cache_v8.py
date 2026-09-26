#!/usr/bin/env python3
"""Pre-launch check: every clip in the v8 config has a complete, loadable cache.

For each [[datasets]] block: each jsonl row must have exactly one latent cache file
(<stem>_*_<W>x<H>_wan.safetensors) whose `latents_*` tensor has the bucket's latent frame
count (1 + (n-1)/4) and a `latents_image_*` tensor, plus a text-encoder cache
(<stem>_wan_te.safetensors). A cache interrupted mid-write fails to open and is reported
(and, with --fix, deleted so `--skip_existing` re-encodes it).

    python finetune/wan/verify_cache_v8.py [--config ...] [--fix]
"""
import argparse
import json
import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from safetensors import safe_open


def check(row, cache, n):
    stem = Path(row["video_path"]).stem
    lat = list(cache.glob(f"{stem}_*_wan.safetensors"))
    te = list(cache.glob(f"{stem}_wan_te.safetensors"))
    errs = []
    if len(lat) != 1:
        errs.append(f"{len(lat)} latent files")
    else:
        try:
            with safe_open(str(lat[0]), "pt") as f:
                keys = list(f.keys())
                k = next(k for k in keys if k.startswith("latents_") and "image" not in k)
                F = f.get_slice(k).get_shape()[1]
                if F != 1 + (n - 1) // 4:
                    errs.append(f"latent frames {F} != {1 + (n - 1) // 4}")
                if not any(k.startswith("latents_image") for k in keys):
                    errs.append("no latents_image (I2V conditioning)")
        except Exception as e:
            errs.append(f"unreadable latent: {e.__class__.__name__}")
            return stem, errs, lat
    if len(te) != 1:
        errs.append(f"{len(te)} text-encoder files")
    else:
        try:
            with safe_open(str(te[0]), "pt") as f:
                list(f.keys())
        except Exception as e:
            errs.append(f"unreadable te: {e.__class__.__name__}")
            return stem, errs, te
    return stem, errs, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/workspace/data_v8/dataset_config_joint_v8.workspace.toml")
    ap.add_argument("--fix", action="store_true", help="delete unreadable cache files")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    total, bad = 0, []
    for ds in cfg["datasets"]:
        n = ds["target_frames"][0]
        cache = Path(ds["cache_directory"])
        rows = [json.loads(l) for l in open(ds["video_jsonl_file"])]
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(lambda r: check(r, cache, n), rows))
        errs = [(s, e, f) for s, e, f in res if e]
        total += len(rows)
        print(f"  f{n:<3} {len(rows):4d} clips  {len(rows) - len(errs):4d} ok  {len(errs)} bad")
        bad += errs
    for s, e, files in bad[:20]:
        print(f"    ✗ {s}: {'; '.join(e)}")
    if a.fix:
        for _, _, files in bad:
            for f in files:
                f.unlink(); print(f"    deleted {f.name}")
    print(f"{total} clips, {len(bad)} with cache problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
