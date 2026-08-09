#!/usr/bin/env python3
"""
Populate the dashboard's Azure container (`AZURE_DASHBOARD_CONTAINER`, default
"pudgy-dashboard") with the final-output videos the Streamlit app streams.

Two sources, tried in this order per mapping:

1. **Blob mirror (primary).** The canonical artifacts live in the training container
   (`AZURE_SOURCE_CONTAINER`, default "pudgy") under `v*/inference/...` and `v*/eval/...`.
   Copying blob->blob is server-side: no download, no local checkout needed.
2. **Local directory (fallback).** If a run's videos happen to be on disk, they are
   uploaded directly.

The local-only version of this script silently did nothing: every path it referenced
(`docs/training_reports/v1`, `.../v2`, `training_approach/v4/final_videos/*`) was removed
from the repo in 87b77b2 "add v5 and clear files", so `gather()` skipped all of them and
reported "Nothing to upload". The mirror below is derived from what the dashboard actually
serves today, so re-running it reproduces the live container exactly.

Usage:
    python upload_assets.py [--dry-run] [--only v5]
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from azure.storage.blob import (
    BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions,
)

SOURCE_CONTAINER = os.environ.get("AZURE_SOURCE_CONTAINER", "pudgy")

# (version, source prefix in SOURCE_CONTAINER, dest prefix in the dashboard container)
BLOB_SOURCES = [
    ("v1", "v1/eval/final.mp4",              "v1"),   # single golden clip
    ("v2", "v2/eval/showcase",               "v2"),
    ("v4", "v4/inference/BEST",              "v4/BEST"),
    ("v4", "v4/inference/FINAL",             "v4/FINAL"),
    ("v4", "v4/inference/FINAL2",            "v4/FINAL2"),
    ("v4", "v4/inference/phase2_variations", "v4/phase2_variations"),
    ("v5", "v5/inference",                   "v5"),
]

# Optional local fallbacks — used only if the directory exists.
LOCAL_SOURCES = [
    ("v4", "training_approach/v4/final_videos/BEST",              "v4/BEST"),
    ("v4", "training_approach/v4/final_videos/FINAL",             "v4/FINAL"),
    ("v4", "training_approach/v4/final_videos/FINAL2",            "v4/FINAL2"),
    ("v4", "training_approach/v4/final_videos/phase2_variations", "v4/phase2_variations"),
    ("v5", "docs/training_reports/v5/final_videos",               "v5"),
]


def _account(conn, field):
    for part in conn.split(";"):
        if part.startswith(field + "="):
            return part[len(field) + 1:]
    sys.exit(f"{field} not found in AZURE_STORAGE_CONNECTION_STRING")


def source_url(conn, blob_name):
    """SAS-signed read URL so the service can copy server-side."""
    sas = generate_blob_sas(
        account_name=_account(conn, "AccountName"),
        container_name=SOURCE_CONTAINER,
        blob_name=blob_name,
        account_key=_account(conn, "AccountKey"),
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    return (f"https://{_account(conn,'AccountName')}.blob.core.windows.net/"
            f"{SOURCE_CONTAINER}/{blob_name}?{sas}")


def gather_blobs(svc, only=None):
    """Yield (source_blob, dest_blob, size) for every mp4 to mirror."""
    src = svc.get_container_client(SOURCE_CONTAINER)
    if not src.exists():
        print(f"  source container '{SOURCE_CONTAINER}' not found — skipping blob mirror")
        return
    for version, s_prefix, d_prefix in BLOB_SOURCES:
        if only and version != only:
            continue
        if s_prefix.lower().endswith(".mp4"):           # a single named blob
            blobs = [b for b in src.list_blobs(name_starts_with=s_prefix)
                     if b.name == s_prefix]
        else:
            blobs = [b for b in src.list_blobs(name_starts_with=s_prefix.rstrip("/") + "/")
                     if b.name.lower().endswith(".mp4")]
        if not blobs:
            print(f"  skip (no mp4s at {SOURCE_CONTAINER}/{s_prefix})")
            continue
        for b in blobs:
            yield b.name, f"{d_prefix}/{os.path.basename(b.name)}", b.size


def gather_local(only=None):
    for version, local_dir, blob_prefix in LOCAL_SOURCES:
        if only and version != only:
            continue
        abs_dir = os.path.join(REPO_ROOT, local_dir)
        if not os.path.isdir(abs_dir):
            continue
        for fn in sorted(os.listdir(abs_dir)):
            if fn.lower().endswith(".mp4"):
                p = os.path.join(abs_dir, fn)
                yield p, f"{blob_prefix}/{fn}", os.path.getsize(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would upload, don't upload")
    ap.add_argument("--only", help="restrict to one version, e.g. v5")
    args = ap.parse_args()

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set — check .env")
    container = os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
    svc = BlobServiceClient.from_connection_string(conn)

    mirrored = list(gather_blobs(svc, args.only))
    local = list(gather_local(args.only))
    # a local copy of the same dest wins (it is the fresher, hand-placed file)
    local_dests = {d for _, d, _ in local}
    mirrored = [m for m in mirrored if m[1] not in local_dests]

    if not mirrored and not local:
        print("Nothing to upload.")
        return

    print(f"Target container '{container}':")
    for _, dest, size in mirrored:
        print(f"  [mirror] {dest}  ({size/1e6:.1f} MB)")
    for _, dest, size in local:
        print(f"  [local ] {dest}  ({size/1e6:.1f} MB)")
    if args.dry_run:
        print("\n--dry-run: not uploading.")
        return

    try:
        svc.create_container(container)
        print(f"\ncreated container: {container}")
    except Exception as e:
        if "ContainerAlreadyExists" not in str(e):
            raise
        print(f"\ncontainer exists: {container}")
    cc = svc.get_container_client(container)

    total, n = 0, 0
    for src_blob, dest, size in mirrored:
        t0 = time.time()
        copy = cc.get_blob_client(dest).start_copy_from_url(source_url(conn, src_blob))
        status = copy.get("copy_status")
        for _ in range(60):                       # server-side copy is usually instant
            if status == "success":
                break
            time.sleep(1)
            status = cc.get_blob_client(dest).get_blob_properties().copy.status
        if status != "success":
            print(f"  ✖ {dest}: copy status {status}")
            continue
        cc.get_blob_client(dest).set_http_headers(
            ContentSettings(content_type="video/mp4"))
        total += size; n += 1
        print(f"  ✔ {dest}  ({size/1e6:.1f} MB, {time.time()-t0:.1f}s)")

    for path, dest, size in local:
        t0 = time.time()
        with open(path, "rb") as f:
            cc.upload_blob(dest, f, overwrite=True, max_concurrency=8,
                           content_settings=ContentSettings(content_type="video/mp4"))
        total += size; n += 1
        print(f"  ✔ {dest}  ({size/1e6:.1f} MB, {time.time()-t0:.1f}s)")

    acct = _account(conn, "AccountName")
    print(f"\nDone: {n} blobs, {total/1e6:.1f} MB -> "
          f"https://{acct}.blob.core.windows.net/{container}/")


if __name__ == "__main__":
    main()
