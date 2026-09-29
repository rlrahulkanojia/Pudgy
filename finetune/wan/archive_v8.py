#!/usr/bin/env python3
"""Archive everything the v8 run produced to Azure, then VERIFY it.

The GPU box is not persistent (workspace_is_volume=false) and is being deleted, so this
is the complete list of what must survive it. Layout under pudgy/v8/:

  weights/pudgy-v8-joint-lownoise/   12 step checkpoints, final, SWA 400-600   (LoRA tau)
  base/pudgy-v7-fixed-combined.safetensors   THE v8 BASE - v8 checkpoints are useless
                                     without it (load it, then tau, both at x1.0)
  state/pudgy-v8-joint-lownoise/<step state>/   optimizer/scheduler/RNG for --resume
  logs/pudgy-v8-joint-lownoise/      tensorboard, trainer log segments, monitor, status,
                                     trajectory
  logs/box/                          prep / cache / setup / smoke logs
  eval/                              dcls results, holdout clip set, keyframes, suites,
                                     winner.json, gate/final logs

NOT archived, deliberately (rebuildable, see REPORT_v8 "Rebuild"): the Wan base models
(public, Hugging Face), the 20 GB latent/T5 cache (2.5 h to rebuild), the classifier-set
cache, the venv. raw/ and processed/v8_joint_2096 are already mirrored and MD5-verified.
Secrets (.env, gh/W&B credentials) are never uploaded.

Upload is idempotent (skips blobs whose size matches). --verify downloads every weight
file back and compares MD5 against the local file; everything else is size-checked.

    python finetune/wan/archive_v8.py            # upload what's missing
    python finetune/wan/archive_v8.py --verify   # upload + byte-level check of weights
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from azure.storage.blob import BlobServiceClient

RUN = Path("/workspace/wan_output/pudgy-v8-joint-lownoise")
BASE = Path("/workspace/wan_output/v7_baseline/pudgy-v7-fixed-combined.safetensors")
EVAL = Path("/workspace/eval_v8")
P = "v8"


def env():
    for line in Path("/workspace/Pudgy/.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def items():
    out = []
    for f in sorted(RUN.glob("*.safetensors")):
        out.append((f, f"{P}/weights/{RUN.name}/{f.name}", True))
    out.append((BASE, f"{P}/base/{BASE.name}", True))
    for d in sorted(RUN.glob("*-state")):
        for f in sorted(d.iterdir()):
            out.append((f, f"{P}/state/{RUN.name}/{d.name}/{f.name}", False))
    for f in sorted((RUN / "logs").rglob("*")):
        if f.is_file():
            out.append((f, f"{P}/logs/{RUN.name}/{f.relative_to(RUN / 'logs')}", False))
    for f in [*RUN.glob("*.log"), RUN / "trajectory.json", RUN / "status.json",
              *Path("/var/log/portal").glob("pudgy-v8-*.log*")]:
        if f.is_file():
            out.append((f, f"{P}/logs/{RUN.name}/{f.name}", False))
    for f in [Path("/workspace/prep_v8.log"), Path("/workspace/cache_v8.log"),
              Path("/workspace/setup_wan_env.log"), *Path("/workspace/smoke_v8").glob("*")]:
        if f.is_file():
            out.append((f, f"{P}/logs/box/{f.name}", False))
    for f in sorted(EVAL.rglob("*")):
        if f.is_file() and "/.raw_" not in str(f) and "/cache/" not in str(f):
            out.append((f, f"{P}/eval/{f.relative_to(EVAL)}", False))
    return out


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    env()
    cc = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
    have = {b.name: b.size for b in cc.list_blobs(name_starts_with=f"{P}/")}
    its = items()
    todo = [(f, b, w) for f, b, w in its if have.get(b) != f.stat().st_size]
    total = sum(f.stat().st_size for f, _, _ in todo)
    print(f"{len(its)} files tracked, {len(todo)} to upload ({total / 1e9:.2f} GB)")

    def up(t):
        f, b, _ = t
        with open(f, "rb") as fh:
            cc.upload_blob(b, fh, overwrite=True, max_concurrency=8)
        return b
    with ThreadPoolExecutor(6) as ex:
        for i, b in enumerate(ex.map(up, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print(f"  uploaded {i}/{len(todo)}  {b}", flush=True)

    have = {b.name: b.size for b in cc.list_blobs(name_starts_with=f"{P}/")}
    bad = [(b, f.stat().st_size, have.get(b)) for f, b, _ in its if have.get(b) != f.stat().st_size]
    print(f"size check: {len(its) - len(bad)}/{len(its)} match")
    for b, ls, rs in bad[:10]:
        print(f"  ✗ {b}: local {ls} azure {rs}")
    if a.verify:
        weights = [(f, b) for f, b, w in its if w]

        def check(t):
            f, b = t
            h = hashlib.md5()
            for chunk in cc.download_blob(b, max_concurrency=8).chunks():
                h.update(chunk)
            return b, md5_file(f) == h.hexdigest()
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(check, weights))
        ok = sum(r for _, r in res)
        print(f"MD5 check (downloaded back): {ok}/{len(res)} weight files byte-identical")
        for b, r in res:
            if not r:
                print(f"  ✗ MD5 MISMATCH {b}")
        bad += [(b, 0, 0) for b, r in res if not r]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
