#!/usr/bin/env python3
"""Mirror expv1's work to Azure every INTERVAL seconds (C1: /workspace is not persistent).

Uploads new or changed files under /workspace/expv1/{output,eval,logs} to
pudgy/expv1-kandinsky-50-video-pro/: every LoRA checkpoint, the NEWEST resume state only,
tensorboard logs, eval JSON/mp4/png, logs. Skips caches and refs (rebuildable).
The connection string comes from the environment (/workspace/.env); it is never printed.

    python scripts/mirror_expv1.py [--once] [--interval 900]
"""
import argparse
import os
import time
from pathlib import Path

W = Path("/workspace/expv1")
PREFIX = "expv1-kandinsky-50-video-pro/"
SUBS = ("output", "eval", "logs")


def env():
    for line in Path("/workspace/.env").read_text().splitlines():
        if line.startswith("AZURE_STORAGE_CONNECTION_STRING="):
            os.environ.setdefault("AZURE_STORAGE_CONNECTION_STRING", line.split("=", 1)[1].strip().strip('"'))


def files():
    states = sorted((W / "output").glob("*-state"))
    old_states = set(states[:-1])
    for sub in SUBS:
        for f in (W / sub).rglob("*"):
            if not f.is_file() or any(p in old_states for p in f.parents):
                continue
            if f.suffix == ".pt" and f.stat().st_size > 500e6:
                continue
            yield f


def sync(cc, seen):
    n = 0
    for f in files():
        st = f.stat()
        key = (st.st_size, int(st.st_mtime))
        if seen.get(f) == key or time.time() - st.st_mtime < 30:      # skip files still being written
            continue
        with open(f, "rb") as fh:
            cc.upload_blob(PREFIX + str(f.relative_to(W)), fh, overwrite=True, max_concurrency=4)
        seen[f] = key
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    a = ap.parse_args()
    env()
    from azure.storage.blob import ContainerClient
    cc = ContainerClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"], "pudgy")
    seen = {}
    while True:
        try:
            n = sync(cc, seen)
            print(f"[{time.strftime('%FT%T')}] mirrored {n} file(s)", flush=True)
        except Exception as e:                 # a transient Azure error must not kill the mirror
            print(f"[{time.strftime('%FT%T')}] mirror error: {type(e).__name__}: {e}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
