#!/usr/bin/env python3
"""
One-off uploader: push local final-output videos into the dashboard's Azure container
(`AZURE_DASHBOARD_CONTAINER`, default "pudgy-dashboard") so the Streamlit app never has
to bundle or read large video files locally — it only needs Azure to be reachable.

Usage:
    source ../.venv-dashboard/bin/activate   # or your own env with azure-storage-blob + python-dotenv
    python upload_assets.py [--dry-run]

Re-run anytime new final videos are added locally; existing blobs are overwritten in place.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from azure.storage.blob import BlobServiceClient, ContentSettings

# (local dir relative to repo root, blob prefix)
SOURCES = [
    ("docs/training_reports/v1", "v1"),
    ("docs/training_reports/v2", "v2"),
    ("training_approach/v4/final_videos/BEST", "v4/BEST"),
    ("training_approach/v4/final_videos/FINAL", "v4/FINAL"),
    ("training_approach/v4/final_videos/FINAL2", "v4/FINAL2"),
    ("training_approach/v4/final_videos/phase2_variations", "v4/phase2_variations"),
    ("docs/training_reports/v5/final_videos", "v5"),
]


def gather():
    for local_dir, blob_prefix in SOURCES:
        abs_dir = os.path.join(REPO_ROOT, local_dir)
        if not os.path.isdir(abs_dir):
            print(f"  skip (not found): {local_dir}")
            continue
        for fn in sorted(os.listdir(abs_dir)):
            if not fn.lower().endswith(".mp4"):
                continue
            yield os.path.join(abs_dir, fn), f"{blob_prefix}/{fn}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would upload, don't upload")
    args = ap.parse_args()

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set — check .env")
    container = os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")

    items = list(gather())
    if not items:
        print("Nothing to upload — no local final-video files found.")
        return

    print(f"Found {len(items)} video(s) to upload to container '{container}':")
    for local, blob in items:
        size_mb = os.path.getsize(local) / 1e6
        print(f"  {blob}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n--dry-run: not uploading.")
        return

    svc = BlobServiceClient.from_connection_string(conn)
    try:
        svc.create_container(container)
        print(f"\ncreated container: {container}")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            print(f"\ncontainer exists: {container}")
        else:
            raise
    cc = svc.get_container_client(container)

    total_bytes, n = 0, 0
    for local, blob in items:
        t0 = time.time()
        size = os.path.getsize(local)
        with open(local, "rb") as f:
            cc.upload_blob(
                blob, f, overwrite=True, max_concurrency=8,
                content_settings=ContentSettings(content_type="video/mp4"),
            )
        total_bytes += size
        n += 1
        print(f"  ✔ {blob}  ({size/1e6:.1f} MB, {time.time()-t0:.1f}s)")

    acct = conn.split("AccountName=")[1].split(";")[0]
    print(f"\nDone: {n} blobs, {total_bytes/1e6:.1f} MB -> "
          f"https://{acct}.blob.core.windows.net/{container}/")


if __name__ == "__main__":
    main()
