#!/usr/bin/env python3
"""Publish live v7 training status to the dashboard container as JSON.

The dashboard is client-facing and reads only from Azure — it has no access to the GPU
box — so "how far along is training" has to be pushed out, not pulled. This parses the
trainer's own stdout (the tqdm `steps:` bar is the authoritative progress source) and
writes one small JSON blob the dashboard renders.

Run it on the box, on a timer alongside azure_upload_v7.py:
    python finetune/wan/publish_status_v7.py
"""
import json, os, re, subprocess, sys, datetime as dt
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

DASH = os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
BLOB = "v7/status.json"
RUNS = [
    ("expression (low-noise)", "/workspace/train_v7_expr.log",
     "/workspace/wan_output/pudgy-v7-expr-lownoise"),
    ("motion A/B — high-noise", "/workspace/train_v7_motion_high.log",
     "/workspace/wan_output/pudgy-v7-motion-highnoise"),
    ("motion A/B — low-noise", "/workspace/train_v7_motion_low.log",
     "/workspace/wan_output/pudgy-v7-motion-lownoise"),
]
# tqdm writes with \r; the last match in the file is the newest sample.
BAR = re.compile(r"steps:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)s/it")
LOSS = re.compile(r"avr_loss=([\d.]+)")
EPOCH = re.compile(r"epoch (\d+)/(\d+)")


def parse(logpath):
    p = Path(logpath)
    if not p.exists():
        return None
    text = p.read_text(errors="ignore").replace("\r", "\n")
    bars = BAR.findall(text)
    if not bars:
        return None
    pct, cur, tot, elapsed, eta, rate = bars[-1]
    losses = LOSS.findall(text)
    epochs = EPOCH.findall(text)
    return {
        "percent": int(pct), "step": int(cur), "total_steps": int(tot),
        "steps_remaining": int(tot) - int(cur),
        "elapsed": elapsed, "eta": eta, "sec_per_step": float(rate),
        "loss": float(losses[-1]) if losses else None,
        "epoch": f"{epochs[-1][0]}/{epochs[-1][1]}" if epochs else None,
        "oom": "OutOfMemoryError" in text,
    }


def running(outdir_name):
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        return f"output_name {outdir_name}" in out
    except Exception:
        return False


def main():
    runs = []
    for label, log, outdir in RUNS:
        st = parse(log)
        if st is None:
            runs.append({"name": label, "state": "not started"})
            continue
        od = Path(outdir)
        ckpts = sorted(od.glob("*.safetensors")) if od.exists() else []
        done = st["step"] >= st["total_steps"]
        st.update({
            "name": label,
            "state": "running" if running(od.name) else ("complete" if done else "stopped"),
            "checkpoints": len(ckpts),
            "latest_checkpoint": ckpts[-1].name if ckpts else None,
        })
        runs.append(st)

    doc = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "runs": runs}
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set (expected in /workspace/.env)")
    cc = BlobServiceClient.from_connection_string(cs).get_container_client(DASH)
    cc.upload_blob(BLOB, json.dumps(doc, indent=2), overwrite=True,
                   content_settings=ContentSettings(content_type="application/json"))
    print(json.dumps(doc, indent=2))
    print(f"\n== published {DASH}/{BLOB}", file=sys.stderr)


if __name__ == "__main__":
    main()
