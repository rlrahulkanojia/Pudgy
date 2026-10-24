#!/usr/bin/env python3
"""Pull from / push to the Azure `pudgy` container, size-idempotent, no extra deps.

Reads AZURE_STORAGE_CONNECTION_STRING from the environment or /workspace/Pudgy/.env
(parsed here: python-dotenv is not in the trainer venv). Used by v9_box.sh.

    python azure_sync.py pull --dest /workspace/Data raw/iteration_6/Samples/ --skip .zip
    python azure_sync.py pull --dest /workspace/restore v8/base/ v8/weights/.../step00000550
    python azure_sync.py push --prefix v9c/weights/pudgy-v9c-phone-lownoise /workspace/wan_output/pudgy-v9c-phone-lownoise/*.safetensors
    python azure_sync.py push --prefix v9c/eval --tree /workspace/eval_v9 --exclude /cache/ /.raw_
    python azure_sync.py ls v9c/

A blob whose size matches the local file is skipped, so re-runs only move what changed.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV_FILES = (Path("/workspace/Pudgy/.env"), Path(__file__).resolve().parents[2] / ".env")


def container():
    if "AZURE_STORAGE_CONNECTION_STRING" not in os.environ:
        for p in ENV_FILES:
            if p.exists():
                for line in p.read_text().splitlines():
                    if "=" in line and not line.lstrip().startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"'))
    if "AZURE_STORAGE_CONNECTION_STRING" not in os.environ:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set and no .env found")
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"], connection_timeout=30,
        read_timeout=300).get_container_client(os.environ.get("AZURE_CONTAINER", "pudgy"))


def pull(cc, prefixes, dest, skip):
    blobs = [b for p in prefixes for b in cc.list_blobs(name_starts_with=p)
             if not any(s in b.name for s in skip)]
    if not blobs:
        sys.exit(f"nothing on Azure under {prefixes}")

    def get(b):
        p = Path(dest) / b.name
        if p.exists() and p.stat().st_size == b.size:
            return 0
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with open(tmp, "wb") as f:
            cc.download_blob(b.name, max_concurrency=4).readinto(f)
        tmp.rename(p)
        return b.size
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(get, blobs))
    print(f"{len(blobs)} blobs under {prefixes} -> {dest} ({n / 1e9:.2f} GB downloaded)")


def push(cc, prefix, files, tree, exclude):
    items = []
    if tree:
        root = Path(tree)
        for f in sorted(root.rglob("*")):
            if f.is_file() and not any(e in str(f) for e in exclude):
                items.append((f, f"{prefix}/{f.relative_to(root).as_posix()}"))
    for f in files:
        f = Path(f)
        if f.is_file():
            items.append((f, f"{prefix}/{f.name}"))
    have = {b.name: b.size for b in cc.list_blobs(name_starts_with=f"{prefix}/")}
    todo = [(f, b) for f, b in items if have.get(b) != f.stat().st_size]

    def up(t):
        f, b = t
        with open(f, "rb") as fh:
            cc.upload_blob(b, fh, overwrite=True, max_concurrency=8)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(up, todo))
    have = {b.name: b.size for b in cc.list_blobs(name_starts_with=f"{prefix}/")}
    bad = [b for f, b in items if have.get(b) != f.stat().st_size]
    print(f"{prefix}: {len(items)} files, {len(todo)} uploaded, size check "
          f"{len(items) - len(bad)}/{len(items)}")
    if bad:
        sys.exit(f"size mismatch on {len(bad)} blob(s), e.g. {bad[:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pl = sp.add_parser("pull"); pl.add_argument("prefixes", nargs="+")
    pl.add_argument("--dest", required=True); pl.add_argument("--skip", nargs="*", default=[])
    ps = sp.add_parser("push"); ps.add_argument("files", nargs="*")
    ps.add_argument("--prefix", required=True); ps.add_argument("--tree")
    ps.add_argument("--exclude", nargs="*", default=[])
    ls = sp.add_parser("ls"); ls.add_argument("prefix")
    a = ap.parse_args()
    cc = container()
    if a.cmd == "pull":
        pull(cc, a.prefixes, a.dest, a.skip)
    elif a.cmd == "push":
        push(cc, a.prefix.rstrip("/"), a.files, a.tree, a.exclude)
    else:
        for b in cc.list_blobs(name_starts_with=a.prefix):
            print(f"{b.size / 1e6:10.1f} MB  {b.name}")


if __name__ == "__main__":
    main()
