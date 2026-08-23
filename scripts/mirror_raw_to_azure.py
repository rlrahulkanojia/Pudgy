#!/usr/bin/env python3
"""
Mirror `Data/raw/` to Azure Blob — the client deliveries, which are the one part of the
data tree that cannot be regenerated.

Background: a 2026-08-21 audit found Azure held only run artifacts (weights, eval, logs,
docs) plus 7 of the 69 iteration_3 clips. Everything else in `raw/` — 2.6 GB of client
footage across three deliveries — existed in a single local copy. `processed/` is
deliberately NOT mirrored: a prep script plus a raw delivery reproduces it byte-for-byte,
which `v5_happy_28` demonstrated.

Blob layout mirrors the local tree exactly:

    pudgy/raw/iteration_1/30_videos/...
    pudgy/raw/iteration_2/72_videos/...
    pudgy/raw/iteration_3/03_expression_clips/...
    pudgy/raw/guidelines/...

The pre-existing `pudgy/interation_3/` prefix (7 Batch-1 Pax/happy clips, misspelled) is
left untouched so existing blob URLs keep resolving. Those same clips now also live under
`raw/iteration_3/`; the duplicate costs 40 MB.

Idempotent: a blob whose size and MD5 already match the local file is skipped, so re-runs
after a new delivery upload only what changed.

Usage:
    python scripts/mirror_raw_to_azure.py --dry-run
    python scripts/mirror_raw_to_azure.py
    python scripts/mirror_raw_to_azure.py --only iteration_3
    python scripts/mirror_raw_to_azure.py --verify        # check, upload nothing
"""
import argparse
import base64
import hashlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

RAW = Path("/Users/rahul/Documents/Projects/Saksham/Pudgy/Data/raw")
CONTAINER = "pudgy"
PREFIX = "raw"
SKIP_NAMES = {".DS_Store"}

CONTENT_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".pdf": "application/pdf",
    ".png": "image/png", ".md": "text/markdown; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def conn_string() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("AZURE_STORAGE_CONNECTION_STRING"):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit(f"no AZURE_STORAGE_CONNECTION_STRING in {env}")


def md5_of(path: Path) -> bytes:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def collect(only: str | None) -> list[tuple[Path, str]]:
    """(local_path, blob_name) pairs, sorted largest-first so the long tail overlaps."""
    items = []
    for p in RAW.rglob("*"):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        rel = p.relative_to(RAW)
        if only and rel.parts[0] != only:
            continue
        items.append((p, f"{PREFIX}/{rel.as_posix()}"))
    return sorted(items, key=lambda it: -it[0].stat().st_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would upload")
    ap.add_argument("--verify", action="store_true", help="compare only; upload nothing")
    ap.add_argument("--only", help="restrict to one delivery folder, e.g. iteration_3")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    items = collect(args.only)
    total = sum(p.stat().st_size for p, _ in items)
    log(f"{len(items)} files, {total/1e9:.2f} GB under {RAW}\n")

    cc = BlobServiceClient.from_connection_string(conn_string()).get_container_client(CONTAINER)

    def sync(path: Path, blob_name: str) -> tuple[str, int]:
        """-> (state, bytes_sent). state in {uploaded, skipped, mismatch, missing}."""
        local_md5 = md5_of(path)
        size = path.stat().st_size
        bc = cc.get_blob_client(blob_name)
        try:
            props = bc.get_blob_properties()
            if props.size == size and props.content_settings.content_md5 == local_md5:
                return "skipped", 0
            state = "mismatch"
        except ResourceNotFoundError:
            state = "missing"

        if args.dry_run or args.verify:
            return state, 0

        ct = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        with open(path, "rb") as f:
            bc.upload_blob(f, overwrite=True, max_concurrency=4,
                           content_settings=ContentSettings(content_type=ct,
                                                            content_md5=local_md5))
        return "uploaded", size

    counts = {"uploaded": 0, "skipped": 0, "mismatch": 0, "missing": 0}
    sent = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(sync, p, b): (p, b) for p, b in items}
        for i, fut in enumerate(as_completed(futs), 1):
            p, b = futs[fut]
            try:
                state, n = fut.result()
            except Exception as e:
                log(f"  ✗ [{i}/{len(items)}] {b}  — {type(e).__name__}: {e}")
                counts["mismatch"] += 1
                continue
            counts[state] += 1
            sent += n
            mark = {"uploaded": "↑", "skipped": "=", "missing": "+", "mismatch": "≠"}[state]
            if state != "skipped" or args.verify:
                log(f"  {mark} [{i}/{len(items)}] {b}  ({p.stat().st_size/1e6:.1f} MB)")

    log("")
    if args.dry_run or args.verify:
        log(f"would upload : {counts['missing']} missing + {counts['mismatch']} differing")
        log(f"already match: {counts['skipped']}")
        if args.verify and not (counts["missing"] or counts["mismatch"]):
            log("\n✅ every local file has a size+MD5-identical blob")
    else:
        log(f"uploaded {counts['uploaded']} files ({sent/1e9:.2f} GB), skipped {counts['skipped']}")


if __name__ == "__main__":
    main()
