"""Back up the whole long-form job to Azure (the box's disk is not a persistent volume).

  deliverables: every file in final/ is checked against its delivery path in pudgy/2026-10-28_longform/
                (root, stories/, skits/) by MD5 and uploaded if missing or different
  workdir:      code, keyframes, cutouts, takes (incl. rejected), QA, logs, docs, skit keyframes and the
                speed test -> pudgy/2026-10-28_longform/workdir/<same tree>

Skipped: the client's own source material (raw/, skits/src/ - already in Azure under raw/iteration_2 and
raw/iteration_6), final/.uncorrected (superseded), caches, and .env.

python backup_azure.py [--dry]
"""
import hashlib
import mimetypes
import os
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

ROOT = Path("/workspace/longform_client")
PREFIX = "2026-10-28_longform/"
WORKDIR = [ROOT / p for p in ("keyframes", "cutouts", "takes", "qa", "logs", "docs", "skits/keys", "skits/keys480")]
WORKDIR_FILES = [p for p in ROOT.glob("*") if p.is_file() and p.suffix in (".py", ".png", ".txt")]
SPEEDTEST = Path("/workspace/speedtest")
SKIP_PARTS = {"__pycache__", ".uncorrected", "src"}


def ctype(p):
    return {".mp4": "video/mp4", ".pdf": "application/pdf", ".md": "text/markdown", ".py": "text/x-python",
            ".log": "text/plain", ".sh": "text/x-shellscript", ".json": "application/json",
            ".manual": "text/plain", ".out": "text/plain"}.get(p.suffix) or mimetypes.guess_type(p.name)[0] \
        or "application/octet-stream"


def plan():
    out = []
    for p in sorted((ROOT / "final").rglob("*")):
        if p.is_file() and not SKIP_PARTS & set(p.parts):
            rel = str(p.relative_to(ROOT / "final"))
            out.append((p, PREFIX + ("samples/" + rel if "light4" in rel else rel)))   # the speed add-on sample
    for d in WORKDIR:
        for p in sorted(d.rglob("*")):
            if p.is_file() and not SKIP_PARTS & set(p.parts):
                out.append((p, PREFIX + "workdir/" + str(p.relative_to(ROOT))))
    for p in WORKDIR_FILES:
        out.append((p, PREFIX + "workdir/" + p.name))
    for p in [SPEEDTEST / "speedtest.py", SPEEDTEST / "speedtest.log", SPEEDTEST / "run_after_batch.sh",
              *sorted((SPEEDTEST / "out").rglob("*"))]:
        if p.is_file():
            out.append((p, PREFIX + "workdir/speedtest/" + str(p.relative_to(SPEEDTEST))))
    return out


def main():
    dry = "--dry" in sys.argv
    c = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
    existing = {b.name: (b.content_settings.content_md5 and bytes(b.content_settings.content_md5)) for b in
                c.list_blobs(name_starts_with=PREFIX)}
    todo = plan()
    up = same = 0
    total = 0
    for p, name in todo:
        data = p.read_bytes()
        md5 = hashlib.md5(data).digest()
        if existing.get(name) == md5:
            same += 1
            continue
        up += 1; total += len(data)
        if dry:
            continue
        c.upload_blob(name, data, overwrite=True, content_settings=ContentSettings(content_type=ctype(p), content_md5=md5))
    print(f"{len(todo)} files: {same} already in Azure (same MD5), {up} {'to upload' if dry else 'uploaded'} "
          f"({total / 1e6:.1f} MB)")
    if dry:
        return
    # verify every planned blob is there with the right size and MD5
    after = {b.name: (b.size, bytes(b.content_settings.content_md5 or b"")) for b in c.list_blobs(name_starts_with=PREFIX)}
    bad = [name for p, name in todo if after.get(name) != (p.stat().st_size, hashlib.md5(p.read_bytes()).digest())]
    print(f"verified {len(todo) - len(bad)}/{len(todo)}" + (f"; MISMATCH: {bad[:10]}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
