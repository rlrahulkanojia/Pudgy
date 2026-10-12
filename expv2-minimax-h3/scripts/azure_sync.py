#!/usr/bin/env python3
"""Azure Blob sync for expv2, both directions. The box disk is not persistent (C1), so the
push side runs on a timer from the first hour.

    # pull what the box needs (container `pudgy`)
    python expv2-minimax-h3/scripts/azure_sync.py pull raw       # raw/iteration_{3,4,5}/ -> $PUDGY_DATA/raw/
    python expv2-minimax-h3/scripts/azure_sync.py pull corpus    # processed/expv2_h3/  -> $PUDGY_DATA/processed/expv2_h3/
    python expv2-minimax-h3/scripts/azure_sync.py pull v8        # v8/eval/suites/core/v8@550/ -> $EXPV2_WORK/v8_550/

    # push results (prefix expv2-minimax-h3/)
    python expv2-minimax-h3/scripts/azure_sync.py push eval weights logs
    python expv2-minimax-h3/scripts/azure_sync.py push eval weights logs --loop 900   # every 15 min
    python expv2-minimax-h3/scripts/azure_sync.py push corpus    # processed/expv2_h3/ (after prep on the box)

Idempotent: a blob whose size matches the local file is skipped (and, where Azure stored a
Content-MD5, the MD5 too). `--dry` lists what would move. Uploads keep relative paths (C7).

Auth: AZURE_STORAGE_CONNECTION_STRING from the environment (configs/env.sh reads it from
/workspace/.env). Never printed. The pull/push pattern follows finetune/wan/services/
pull_raw.py and finetune/wan/azure_upload_v7.py (read, not imported).
"""
import argparse
import base64
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv2_common as C  # noqa: E402

PULL = {
    "raw": [(f"raw/iteration_{i}/", C.DATA / f"raw/iteration_{i}") for i in (3, 4, 5)],
    "corpus": [("processed/expv2_h3/", C.CORPUS)],
    "v8": [("v8/eval/suites/core/v8@550/", C.WORK / "v8_550")],
}


def push_map():
    P = C.AZURE_PREFIX
    return {
        "eval": [(C.EVAL, f"{P}/eval/", None)],
        # LoRA checkpoints + the run's own records; the stage-1 caches are rebuildable and big.
        "weights": [(C.RUNS, f"{P}/weights/", (".safetensors", ".json", ".csv", ".log"))],
        "logs": [(C.WORK / "logs", f"{P}/logs/", None)],
        "corpus": [(C.CORPUS, "processed/expv2_h3/", None)],
    }


def container():
    from azure.storage.blob import BlobServiceClient
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs:
        env = Path("/workspace/.env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("AZURE_STORAGE_CONNECTION_STRING"):
                    cs = line.split("=", 1)[1].strip().strip('"')
    if not cs:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set and no /workspace/.env (RUNBOOK step 1)")
    return BlobServiceClient.from_connection_string(cs).get_container_client(C.AZURE_CONTAINER)


def md5_b64(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def pull(cc, what, dry, workers):
    jobs = []
    for prefix, dest in PULL[what]:
        for b in cc.list_blobs(name_starts_with=prefix):
            local = Path(dest) / b.name[len(prefix):]
            if local.exists() and local.stat().st_size == b.size:
                continue
            jobs.append((b.name, b.size, local))
    total = sum(s for _, s, _ in jobs)
    print(f"pull {what}: {len(jobs)} blobs, {total / 1e9:.2f} GB to fetch", flush=True)
    if dry:
        for name, size, local in jobs[:20]:
            print(f"  {name} -> {local}")
        return

    def get(job):
        name, _, local = job
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        with open(tmp, "wb") as f:
            cc.download_blob(name, max_concurrency=4).readinto(f)
        tmp.replace(local)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, jobs))
    print(f"pull {what}: done", flush=True)


def push(cc, what, dry, workers):
    from azure.storage.blob import ContentSettings
    jobs = []
    for root, prefix, suffixes in push_map()[what]:
        root = Path(root)
        if not root.exists():
            print(f"push {what}: {root} does not exist yet - skipped")
            continue
        remote = {b.name: b for b in cc.list_blobs(name_starts_with=prefix)}
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            if f.name.endswith((".part", ".tmp")) or f.name == ".DS_Store":
                continue
            if suffixes and not f.name.endswith(suffixes):
                continue
            name = prefix + f.relative_to(root).as_posix()
            b = remote.get(name)
            if b is not None and b.size == f.stat().st_size:
                cm = b.content_settings.content_md5 if b.content_settings else None
                if not cm or base64.b64encode(bytes(cm)).decode() == md5_b64(f):
                    continue
            jobs.append((f, name))
    total = sum(f.stat().st_size for f, _ in jobs)
    print(f"push {what}: {len(jobs)} files, {total / 1e9:.2f} GB", flush=True)
    if dry:
        for f, name in jobs[:20]:
            print(f"  {f} -> {name}")
        return

    def put(job):
        f, name = job
        md5 = base64.b64decode(md5_b64(f))
        with open(f, "rb") as fh:
            cc.upload_blob(name, fh, overwrite=True, max_concurrency=4,
                           content_settings=ContentSettings(content_md5=bytearray(md5)))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(put, jobs))
    print(f"push {what}: done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["pull", "push"])
    ap.add_argument("what", nargs="+")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--loop", type=int, default=0, help="push every N seconds until killed")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    valid = PULL if args.direction == "pull" else push_map()
    bad = [w for w in args.what if w not in valid]
    if bad:
        sys.exit(f"unknown {args.direction} target(s) {bad}; choose from {sorted(valid)}")
    cc = container()
    while True:
        for what in args.what:
            try:
                (pull if args.direction == "pull" else push)(cc, what, args.dry, args.workers)
            except Exception as e:                      # a timer must survive one bad pass
                if not args.loop:
                    raise
                print(f"{args.direction} {what}: FAILED this pass: {e!r}", flush=True)
        if not args.loop or args.direction == "pull":
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
