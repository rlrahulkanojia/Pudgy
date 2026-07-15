#!/usr/bin/env python3
"""
Reusable Azure Blob uploader for Pudgy Wan2.2-A14B (Training Approach v2).

Container = "folder" v2-decoupled-identity-motion (Azure container names can't have
underscores; this is the 3-word v2 thesis: decouple identity from motion).

Auth + container come from env (see /workspace/.env, untracked):
    AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER

Usage (via the wrapper that sources .env):
    bash azure_upload.sh weights logs output      # initial push
    bash azure_upload.sh weights output           # re-push final weights + eval later
    bash azure_upload.sh all                       # + optimizer state dirs (resume backup)

Blob layout inside the container:
    weights/<ckpt>.safetensors        LoRA checkpoints (current run)
    logs/<name>.log                    training / eval logs
    output/<relpath>                   docs/training_reports/v2 (videos, montages, report, ep10 weights)
    state/<relpath>                    *-state optimizer dirs (only with 'state' or 'all')
"""
import os, sys, time, glob
from azure.storage.blob import BlobServiceClient, ContentSettings

CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.environ.get("AZURE_CONTAINER", "v2-decoupled-identity-motion")
WEIGHTS_DIR = "/workspace/wan_output/pudgy-wan22-a14b-lownoise"
LOGS_DIR = "/workspace/logs"
OUTPUT_DIR = "/workspace/Pudgy/docs/training_reports/v2"
MIN_AGE_S = 30          # skip files written within the last 30s (avoid partial checkpoints)

if not CONN:
    sys.exit("AZURE_STORAGE_CONNECTION_STRING not set — source /workspace/.env first.")

svc = BlobServiceClient.from_connection_string(CONN)
try:
    svc.create_container(CONTAINER)
    print(f"created container: {CONTAINER}")
except Exception as e:
    if "ContainerAlreadyExists" in str(e):
        print(f"container exists: {CONTAINER}")
    else:
        raise
cc = svc.get_container_client(CONTAINER)


def _ct(path):
    if path.endswith(".mp4"):  return "video/mp4"
    if path.endswith(".png"):  return "image/png"
    if path.endswith(".md"):   return "text/markdown"
    if path.endswith(".log") or path.endswith(".txt"): return "text/plain"
    return "application/octet-stream"


def upload(local_path, blob_name):
    size = os.path.getsize(local_path)
    age = time.time() - os.path.getmtime(local_path)
    if age < MIN_AGE_S:
        print(f"  skip (still being written, {age:.0f}s old): {blob_name}")
        return 0
    t0 = time.time()
    with open(local_path, "rb") as f:
        cc.upload_blob(blob_name, f, overwrite=True, max_concurrency=8,
                       content_settings=ContentSettings(content_type=_ct(local_path)))
    print(f"  ✔ {blob_name}  ({size/1e6:.1f} MB, {time.time()-t0:.1f}s)")
    return size


def gather(category):
    """yield (local_path, blob_name) pairs for a category."""
    if category == "weights":
        for p in sorted(glob.glob(f"{WEIGHTS_DIR}/*.safetensors")):
            yield p, f"weights/{os.path.basename(p)}"
    elif category == "logs":
        for p in sorted(glob.glob(f"{LOGS_DIR}/*.log")):
            yield p, f"logs/{os.path.basename(p)}"
    elif category == "output":
        for root, _, files in os.walk(OUTPUT_DIR):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, OUTPUT_DIR)
                yield p, f"output/{rel}"
    elif category == "state":
        for root, _, files in os.walk(WEIGHTS_DIR):
            if not root.rstrip("/").endswith("-state") and "-state/" not in root + "/":
                continue
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, WEIGHTS_DIR)
                yield p, f"state/{rel}"


def main():
    cats = sys.argv[1:] or ["weights", "logs", "output"]
    if "all" in cats:
        cats = ["weights", "logs", "output", "state"]
    total = 0
    n = 0
    for cat in cats:
        print(f"[{cat}]")
        for local, blob in gather(cat):
            s = upload(local, blob)
            if s:
                total += s; n += 1
    acct = CONN.split("AccountName=")[1].split(";")[0]
    print(f"\nDone: {n} blobs, {total/1e9:.2f} GB -> "
          f"https://{acct}.blob.core.windows.net/{CONTAINER}/")


if __name__ == "__main__":
    main()
