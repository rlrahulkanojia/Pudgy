#!/usr/bin/env python3
"""
Mirror v6 training artefacts to Azure *while the run is in progress*.

v5 uploaded at the end (azure_upload_v5.py). That is fine right up until it isn't:
the v5 GPU box was destroyed with its disk, `/workspace` is not a persistent volume
here either, and a v6 run is 40-50 h. Uploading only on completion means the entire
run is one preemption away from being gone (Training_Approach_v6 risk 9.8).

So this watches the output dir and pushes each checkpoint as it lands.

  python azure_mirror_v6.py --watch /workspace/wan_output/pudgy-expr-v6-lownoise
  python azure_mirror_v6.py --once  /workspace/wan_output/pudgy-expr-v6-lownoise

Layout written (matching the existing v1/v2/v4/v5 convention in container `pudgy`):

  v6/weights/   LoRA checkpoints
  v6/logs/      tensorboard events + trainer stdout
  v6/docs/      configs, dataset jsonl/toml, the plan

Deliberately NOT uploaded: `*-state/` resume directories. They are ~15 GB per
checkpoint and v5 made the same call. Pass --with-states if you specifically need
exact-resume capability off-box.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    sys.exit("pip install azure-storage-blob")

CONTAINER = os.environ.get("AZURE_CONTAINER", "pudgy")
# Files written in the last SETTLE seconds are skipped: a 300 MB checkpoint being
# flushed would otherwise upload truncated and look like a valid blob. Same guard the
# v2 upload script used.
SETTLE = 45


def load_env(path="/workspace/.env"):
    if not os.environ.get("AZURE_STORAGE_CONNECTION_STRING") and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def classify(p: Path, root: Path, with_states: bool):
    """Local path -> blob name, or None to skip."""
    rel = p.relative_to(root)
    parts = rel.parts
    if any(part.endswith("-state") for part in parts) and not with_states:
        return None
    if p.suffix == ".safetensors":
        return f"weights/{p.name}"
    # Drop a leading "logs/" from the relative path before re-prefixing it, or a file
    # already under logs/ lands at logs/logs/... in the container.
    tail = rel.as_posix()
    if parts and parts[0] == "logs":
        tail = Path(*parts[1:]).as_posix() if len(parts) > 1 else p.name
    if "logs" in parts or p.suffix in (".log", ".txt") or "events.out" in p.name:
        return f"logs/{tail}"
    if p.suffix in (".toml", ".json", ".jsonl", ".yaml", ".md"):
        return f"docs/{rel.as_posix()}"
    return None


def sweep(cc, root: Path, prefix: str, seen: dict, with_states: bool, quiet=False):
    now = time.time()
    pushed = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        blobpart = classify(p, root, with_states)
        if blobpart is None:
            continue
        st = p.stat()
        if now - st.st_mtime < SETTLE:            # still being written
            continue
        key = str(p)
        if seen.get(key) == (st.st_size, int(st.st_mtime)):
            continue
        blob = f"{prefix.rstrip('/')}/{blobpart}"
        try:
            with open(p, "rb") as fh:
                cc.upload_blob(name=blob, data=fh, overwrite=True,
                               max_concurrency=8)
            seen[key] = (st.st_size, int(st.st_mtime))
            pushed += 1
            if not quiet:
                print(f"  ↑ {blob}  ({st.st_size/1e6:.1f} MB)", flush=True)
        except Exception as e:                    # never let mirroring kill training
            print(f"  ! failed {blob}: {type(e).__name__}: {e}", flush=True)
    return pushed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--prefix", default="v6")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--with-states", action="store_true",
                    help="also upload *-state/ resume dirs (~15 GB per checkpoint)")
    args, _ = ap.parse_known_args()

    load_env()
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set (expected in /workspace/.env)")
    cc = BlobServiceClient.from_connection_string(cs).get_container_client(CONTAINER)

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    seen = {}
    print(f"mirroring {root} -> {CONTAINER}/{args.prefix}/  "
          f"(states {'INCLUDED' if args.with_states else 'skipped'})", flush=True)

    if args.once or not args.watch:
        n = sweep(cc, root, args.prefix, seen, args.with_states)
        print(f"done: {n} file(s) uploaded", flush=True)
        return

    # Watch mode: runs as a child of the training script, dies with it.
    while True:
        try:
            sweep(cc, root, args.prefix, seen, args.with_states, quiet=True)
        except Exception as e:
            print(f"  ! sweep error: {type(e).__name__}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
