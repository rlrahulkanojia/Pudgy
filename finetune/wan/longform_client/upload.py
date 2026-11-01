"""Upload the finished client videos to Azure: pudgy/<YYYY-MM-DD>_longform/[<sub>/] (new folder;
nothing else is written). Blobs already there with the same MD5 are left alone, so it can be
re-run after every finished video. Verifies every blob by size + MD5 after upload and prints a
read-only link (7 days) for each file.

python upload.py <YYYY-MM-DD> [--dry] [--sub samples] [files...]   (default: final/*.mp4 + README)
"""
import hashlib
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.storage.blob import (BlobSasPermissions, BlobServiceClient, ContentSettings,
                                generate_blob_sas)

ROOT = Path("/workspace/longform_client")


def link(c, name, days=7):
    key = dict(kv.split("=", 1) for kv in os.environ["AZURE_STORAGE_CONNECTION_STRING"].split(";") if kv)["AccountKey"]
    sas = generate_blob_sas(c.account_name, c.container_name, name, account_key=key,
                            permission=BlobSasPermissions(read=True),
                            expiry=datetime.now(timezone.utc) + timedelta(days=days))
    return f"{c.get_blob_client(name).url}?{sas}"


def main(date, dry=False, files=None, sub=""):
    prefix = f"{date}_longform/" + (f"{sub.strip('/')}/" if sub else "")
    if not files:
        files = sorted((ROOT / "final").glob("*_1080x1920.mp4")) + [ROOT / "final/README.md"]
    files = [Path(f) for f in files if Path(f).exists()]
    c = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
    existing = {b.name: bytes(b.content_settings.content_md5 or b"") for b in c.list_blobs(name_starts_with=prefix)}
    for f in files:
        name, data = prefix + f.name, f.read_bytes()
        md5 = hashlib.md5(data).digest()
        if existing.get(name) == md5:
            print("same    " + name)
            continue
        print(("DRY " if dry else "") + ("replace " if name in existing else "upload  ") + name, len(data))
        if dry:
            continue
        ctype = {".mp4": "video/mp4", ".pdf": "application/pdf"}.get(f.suffix, "text/markdown")
        c.upload_blob(name, data, overwrite=True,
                      content_settings=ContentSettings(content_type=ctype, content_md5=md5))
    if dry:
        return {}
    bad, links = [], {}
    for f in files:
        p = c.get_blob_client(prefix + f.name).get_blob_properties()
        if p.size != f.stat().st_size or bytes(p.content_settings.content_md5 or b"") != hashlib.md5(f.read_bytes()).digest():
            bad.append(f.name)
        links[f.name] = link(c, prefix + f.name)
    print(f"verified {len(files) - len(bad)}/{len(files)} blobs under pudgy/{prefix}" + (f" - MISMATCH {bad}" if bad else ""))
    if bad:
        sys.exit(1)
    return links


if __name__ == "__main__":
    a = sys.argv[2:]
    sub = a[a.index("--sub") + 1] if "--sub" in a else ""
    files = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--sub")]
    for n, u in main(sys.argv[1], "--dry" in a, files, sub).items():
        print(n, u)
