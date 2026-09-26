#!/usr/bin/env python3
"""v8 run monitor: status + weight-space diagnostics + Azure mirror, on a timer.

/workspace on this box is NOT persistent (vast-capabilities: workspace_is_volume=false);
v5 lost its training data that way and v7 survived only because of a 30-minute mirror.
This is that mirror for v8, plus the CPU-only diagnostics that need no GPU:

  every --every seconds (default 600):
    * parse the trainer log (tqdm `steps:` bar + avr_loss)      -> status.json
    * if a new checkpoint landed: lora_tools_v8 traj --tb        -> trajectory.json +
      tensorboard scalars under logs/diag (tau norm, drift share, increment cosine)
    * mirror to Azure pudgy/v8/ (size-idempotent):
        weights/<run>/*.safetensors   logs/<run>/**   eval/**   status.json
      resume states only with --with-states (large: optimizer moments)

Client dashboard publishing is OFF unless --dashboard is passed: that container is
client-facing, so it is an explicit decision, not a side effect of monitoring.

    python finetune/wan/monitor_v8.py                 # loop
    python finetune/wan/monitor_v8.py --once --dry-run
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(os.environ.get("V8_RUN", "/workspace/wan_output/pudgy-v8-joint-lownoise"))
LOG = Path(os.environ.get("V8_LOG", "/var/log/portal/pudgy-v8-train.log"))
EVAL = Path("/workspace/eval_v8")
PREFIX = "v8"
HERE = Path(__file__).resolve().parent
BAR = re.compile(r"steps:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)s/it")
LOSS = re.compile(r"avr_loss=([\d.]+)")
WANDB = re.compile(r"View run at (https://wandb\.ai/\S+)")


def env():
    p = Path("/workspace/Pudgy/.env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def status():
    s = {"run": RUN.name, "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if LOG.exists():
        tail = LOG.read_bytes()[-200_000:].decode("utf-8", "replace")
        bars = BAR.findall(tail)
        if bars:
            pct, done, total, el, rem, spi = bars[-1]
            s.update(step=int(done), total=int(total), elapsed=el, remaining=rem,
                     sec_per_it=float(spi))
        losses = LOSS.findall(tail)
        if losses:
            s["avr_loss"] = float(losses[-1])
        w = WANDB.findall(LOG.read_bytes()[:2_000_000].decode("utf-8", "replace"))
        if w:
            s["wandb"] = w[-1]
        s["finished"] = "== done:" in tail
        s["error"] = next((l for l in tail.splitlines()[::-1]
                           if "Error" in l or "Traceback" in l), None)
    s["checkpoints"] = sorted(p.name for p in RUN.glob("*.safetensors"))
    tj = RUN / "trajectory.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        s["diag"] = {"loglog_slope": t["loglog_slope"],
                     "drift_share_last": list(t["drift_share"].values())[-1],
                     "increment_cos": t["consecutive_increment_cos"][-3:]}
    return s


def diag(seen):
    ck = sorted(RUN.glob("*-step*.safetensors"))
    if len(ck) >= 2 and len(ck) != seen:
        subprocess.run([sys.executable, str(HERE / "eval_v8/lora_tools_v8.py"), "traj",
                        str(RUN), "--tb"], capture_output=True)
    return len(ck)


def collect(with_states):
    items = []
    for f in RUN.glob("*.safetensors"):
        items.append((f, f"{PREFIX}/weights/{RUN.name}/{f.name}"))
    for f in (RUN / "logs").rglob("*") if (RUN / "logs").exists() else []:
        if f.is_file():
            items.append((f, f"{PREFIX}/logs/{RUN.name}/{f.relative_to(RUN / 'logs')}"))
    for name in ("trajectory.json", "status.json"):
        if (RUN / name).exists():
            items.append((RUN / name, f"{PREFIX}/logs/{RUN.name}/{name}"))
    if LOG.exists():
        items.append((LOG, f"{PREFIX}/logs/{RUN.name}/{LOG.name}"))
    if with_states:
        for d in RUN.glob("*-state"):
            for f in d.rglob("*"):
                if f.is_file():
                    items.append((f, f"{PREFIX}/state/{RUN.name}/{d.name}/{f.relative_to(d)}"))
    if EVAL.exists():
        for f in EVAL.rglob("*"):
            if f.is_file() and "/.raw_" not in str(f) and "/cache/" not in str(f):
                items.append((f, f"{PREFIX}/eval/{f.relative_to(EVAL)}"))
    return items


def mirror(dry, with_states):
    from azure.storage.blob import BlobServiceClient
    cc = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
    have = {b.name: b.size for b in cc.list_blobs(name_starts_with=f"{PREFIX}/")}
    todo = [(f, b) for f, b in collect(with_states) if have.get(b) != f.stat().st_size]
    for f, b in todo:
        if dry:
            print(f"   would upload {b} ({f.stat().st_size / 1e6:.1f} MB)")
            continue
        with open(f, "rb") as fh:
            cc.upload_blob(b, fh, overwrite=True, max_concurrency=4)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--mirror-every", type=int, default=1800)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-states", action="store_true")
    ap.add_argument("--dashboard", action="store_true",
                    help="ALSO publish status.json to the client-facing dashboard container")
    a = ap.parse_args()
    env()
    seen, last_mirror = 0, 0.0
    while True:
        RUN.mkdir(parents=True, exist_ok=True)
        seen = diag(seen)
        s = status()
        (RUN / "status.json").write_text(json.dumps(s, indent=1))
        msg = f"[{s['updated']}] step {s.get('step', '?')}/{s.get('total', '?')} loss {s.get('avr_loss', '?')}"
        if time.time() - last_mirror >= a.mirror_every or a.once:
            try:
                n = mirror(a.dry_run, a.with_states)
                msg += f" | mirrored {n} file(s)"
                last_mirror = time.time()
            except Exception as e:                  # never let the mirror kill the monitor
                msg += f" | mirror FAILED: {e}"
            if a.dashboard and not a.dry_run:
                from azure.storage.blob import BlobServiceClient, ContentSettings
                BlobServiceClient.from_connection_string(
                    os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client(
                    os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")).upload_blob(
                    f"{PREFIX}/status.json", json.dumps(s), overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"))
        print(msg, flush=True)
        if a.once or s.get("finished"):
            if s.get("finished") and not a.once:
                mirror(a.dry_run, a.with_states)     # final push after the last save
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
