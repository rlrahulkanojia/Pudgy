"""Find and talk to the GPU inference API (finetune/wan/infer_api) from the dashboard.

The GPU box is switched on only when it is needed and its public address changes on every start,
so the server announces itself in Azure: <dashboard container>/inference/server.json holds its URL
and a heartbeat refreshed every 30 s, and is marked "stopped" on a clean shutdown. A box that is
simply switched off stops heart-beating. INFERENCE_API_URL overrides the lookup (a fixed address,
or a local server while developing).

Settings (App Service application settings, or the repo .env when run locally):
  INFERENCE_API_KEY   the key the server was started with (required to generate)
  INFERENCE_API_URL   optional fixed address, e.g. http://127.0.0.1:8765
"""
import json
import os
from datetime import datetime, timezone

import requests

import azure_utils

REGISTRY_BLOB = "inference/server.json"
STALE_AFTER_S = 120          # 4 missed heartbeats
HEALTH_TIMEOUT_S = 6


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def api_key():
    return os.environ.get("INFERENCE_API_KEY", "").strip()


def _age_s(stamp):
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - t).total_seconds()


def human_age(seconds):
    if seconds is None:
        return "an unknown time"
    m = int(seconds // 60)
    if m < 2:
        return "a minute"
    if m < 120:
        return f"{m} minutes"
    h = m // 60
    return f"{h} hours" if h < 48 else f"{h // 24} days"


def discover():
    """-> {online: bool, url, reason, health, registered}. Never raises: an unreachable server is
    a normal state for this dashboard, not an error."""
    url = os.environ.get("INFERENCE_API_URL", "").strip().rstrip("/")
    reg = None
    if not url:
        reg = azure_utils.read_json(REGISTRY_BLOB)
        if not reg or not reg.get("url"):
            return {"online": False, "url": None, "health": None, "registered": None,
                    "reason": "No inference server has been started yet."}
        url = reg["url"].rstrip("/")
        if reg.get("state") == "stopped":
            return {"online": False, "url": url, "health": None, "registered": reg,
                    "reason": f"The inference server was shut down {human_age(_age_s(reg.get('stopped_utc')))} ago."}
        age = _age_s(reg.get("heartbeat_utc"))
        if age is None or age > STALE_AFTER_S:
            return {"online": False, "url": url, "health": None, "registered": reg,
                    "reason": f"The GPU machine was last heard from {human_age(age)} ago, so it is "
                              f"probably switched off."}
    try:
        r = requests.get(f"{url}/v1/health", timeout=HEALTH_TIMEOUT_S)
        r.raise_for_status()
        health = r.json()
        if health.get("service") != "pudgy-inference":
            raise ValueError("not a Pudgy inference server")
    except (requests.RequestException, ValueError) as e:
        why = "timed out" if isinstance(e, requests.Timeout) else "refused or unreachable" \
            if isinstance(e, requests.ConnectionError) else str(e)[:120]
        return {"online": False, "url": url, "health": None, "registered": reg,
                "reason": f"The inference server is registered but not answering ({why}). It may be "
                          f"starting up or shutting down."}
    return {"online": True, "url": url, "health": health, "registered": reg, "reason": None}


def _call(url, method, path, timeout=20, **kw):
    try:
        r = requests.request(method, f"{url}{path}", headers={"X-API-Key": api_key()}, timeout=timeout, **kw)
    except requests.Timeout:
        raise ApiError(0, "The inference server did not answer in time. Try again in a moment.")
    except requests.RequestException:
        raise ApiError(0, "The inference server could not be reached. It may have been switched off.")
    if r.status_code == 401:
        raise ApiError(401, "The dashboard's API key was not accepted (INFERENCE_API_KEY).")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, list):         # FastAPI's own validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
        raise ApiError(r.status_code, detail or f"The server answered {r.status_code}.")
    return r


def info(url):
    return _call(url, "GET", "/v1/info").json()


def preview_prompt(url, guided=None, prompt=None):
    body = {"guided": guided} if guided is not None else {"prompt": prompt}
    return _call(url, "POST", "/v1/prompt", json=body).json()


def submit(url, params, start=None, end=None):
    """start/end: (filename, bytes, mime) or None."""
    files = {}
    if start:
        files["start_image"] = start
    if end:
        files["end_image"] = end
    return _call(url, "POST", "/v1/jobs", timeout=60, data={"params": json.dumps(params)}, files=files).json()


def jobs(url, limit=12):
    return _call(url, "GET", f"/v1/jobs?limit={limit}").json()["jobs"]


def cancel(url, job_id):
    return _call(url, "DELETE", f"/v1/jobs/{job_id}").json()


def keyframe_png(url, name):
    return _call(url, "GET", f"/v1/keyframes/{name}").content


def video_bytes(url, job_id):
    return _call(url, "GET", f"/v1/jobs/{job_id}/video", timeout=120).content
