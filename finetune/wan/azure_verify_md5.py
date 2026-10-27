#!/usr/bin/env python3
"""Check Azure copies by CONTENT (Content-MD5), not size, and re-upload the ones that differ.

azure_sync.py push skips a blob whose size equals the local file's, so a same-size edit (a
checksum file, a re-saved checkpoint) never reaches Azure. Blobs without a stored MD5 (large
block uploads) are reported, not re-uploaded unless --fix-missing.

    python azure_verify_md5.py [--fix] [--fix-missing] PREFIX=LOCAL_DIR_OR_FILE ...
"""
import base64
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import azure_sync as AS  # noqa: E402


def md5_b64(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def main():
    fix, fix_missing = "--fix" in sys.argv, "--fix-missing" in sys.argv
    cc = AS.container()
    bad = 0
    for arg in (a for a in sys.argv[1:] if not a.startswith("--")):
        prefix, local = arg.split("=", 1)
        local = Path(local)
        pairs = [(f, f"{prefix}/{f.relative_to(local).as_posix()}") for f in sorted(local.rglob("*")) if f.is_file()] \
            if local.is_dir() else [(local, f"{prefix}/{local.name}")]
        have = {b.name: b for b in cc.list_blobs(name_starts_with=f"{prefix}/")}
        n_ok = n_missing_md5 = 0
        for f, name in pairs:
            b = have.get(name)
            remote = b.content_settings.content_md5 if b else None
            remote = base64.b64encode(bytes(remote)).decode() if remote else None
            if b is not None and remote is None:
                n_missing_md5 += 1
                if not fix_missing:
                    continue
            if b is not None and remote == md5_b64(f):
                n_ok += 1
                continue
            bad += 1
            print(f"  DIFFERS {name} ({'absent' if b is None else 'no md5' if remote is None else 'content'})")
            if fix or (fix_missing and remote is None):
                with open(f, "rb") as fh:
                    cc.upload_blob(name, fh, overwrite=True, max_concurrency=8)
                print(f"    re-uploaded {name}")
        print(f"{prefix}: {len(pairs)} files, md5 match {n_ok}, no stored md5 {n_missing_md5}")
    sys.exit(0 if not bad or fix else 1)


if __name__ == "__main__":
    main()
