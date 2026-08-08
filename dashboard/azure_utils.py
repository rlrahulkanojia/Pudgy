"""Azure Blob Storage helpers for the dashboard (read side: list + SAS-signed streaming URLs)."""
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

CONTAINER = os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
SAS_TTL_HOURS = 12


def _conn_str():
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set — add it to .env and restart."
        )
    return conn


@lru_cache(maxsize=1)
def _service_client():
    return BlobServiceClient.from_connection_string(_conn_str())


@lru_cache(maxsize=1)
def _account_key():
    for part in _conn_str().split(";"):
        if part.startswith("AccountKey="):
            return part[len("AccountKey="):]
    raise RuntimeError("AccountKey not found in AZURE_STORAGE_CONNECTION_STRING")


@lru_cache(maxsize=1)
def _account_name():
    for part in _conn_str().split(";"):
        if part.startswith("AccountName="):
            return part[len("AccountName="):]
    raise RuntimeError("AccountName not found in AZURE_STORAGE_CONNECTION_STRING")


def container_exists():
    try:
        return _service_client().get_container_client(CONTAINER).exists()
    except Exception:
        return False


def list_videos(prefix):
    """List .mp4 blobs under a prefix, returning sorted [(blob_name, display_name)]."""
    if not container_exists():
        return []
    cc = _service_client().get_container_client(CONTAINER)
    p = prefix.rstrip("/") + "/"
    out = []
    for blob in cc.list_blobs(name_starts_with=p):
        if not blob.name.lower().endswith(".mp4"):
            continue
        display = blob.name[len(p):]
        out.append((blob.name, display))
    return sorted(out, key=lambda x: x[1])


def signed_url(blob_name):
    """Return a read-only SAS URL for a blob, valid for SAS_TTL_HOURS."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=SAS_TTL_HOURS)
    sas = generate_blob_sas(
        account_name=_account_name(),
        container_name=CONTAINER,
        blob_name=blob_name,
        account_key=_account_key(),
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{_account_name()}.blob.core.windows.net/{CONTAINER}/{blob_name}?{sas}"
