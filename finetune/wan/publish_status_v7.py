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



# What each post-training stage is FOR, in plain language. The dashboard is read by
# people who will not know what a "gate" is, so each entry says what question the stage
# answers and what happens next — not what it technically measures.
STAGES = {
    "gx": {
        "name": "Choosing where movement lives",
        "focus": "The model has two halves — one handles broad motion, the other fine "
                 "detail. We trained the movement skills into each half separately and "
                 "are now generating the same actions from both to see which version "
                 "takes direction better. The winner becomes the movement model; the "
                 "other is kept for the record.",
        "decides": "Which of the two movement models we keep and finish training.",
        "root": "/workspace/eval_v7/gx",
        "log": "/workspace/gx_v7.log",
    },
    "sweep": {
        "name": "Finding the best checkpoint",
        "focus": "Training saves a snapshot every 250 steps. This re-generates the same "
                 "set of expressions from several snapshots to find which one takes "
                 "direction best — more training is not automatically better.",
        "decides": "Which saved snapshot becomes the delivered model.",
        "root": "/workspace/eval_v7/sweep",
        "log": "/workspace/sweep_coarse.log",
    },
    "gates": {
        "name": "Checking the model behaves",
        "focus": "Generating test clips to confirm each expression and movement responds "
                 "to the prompt, holds up at different clip lengths and shot sizes, and "
                 "has not damaged anything the previous model could already do.",
        "decides": "Whether the model is ready to hand over.",
        "root": "/workspace/eval_v7",
        "log": "/workspace/gates_v7.log",
    },
}


def active_stage():
    """Which post-training stage is running, and how far along.

    Progress is counted from the batch prompt file (one line = one clip to generate)
    against the clips actually written, so it stays honest if a batch is restarted.
    """
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    except Exception:
        ps = ""
    key = ("gx" if "gx_v7.py" in ps else
           "sweep" if "sweep_v7.py" in ps else
           "gates" if "gates_v7.py" in ps else None)
    if key is None:
        return None
    meta = STAGES[key]
    root = Path(meta["root"])
    total = done = 0
    for pf in root.glob("*/prompts.txt"):
        total += len([l for l in pf.read_text().splitlines() if l.strip()])
    # Count generations from the trainer log, NOT written mp4s. musubi's --from_file
    # mode holds every latent in memory and decodes the whole batch only after the last
    # generation, so the output directory stays empty for hours and then fills at once —
    # a file-count progress bar would sit at 0% and jump straight to 100%.
    lg = Path(meta["log"])
    if lg.exists():
        txt = lg.read_text(errors="ignore").replace("\r", "\n")
        hits = re.findall(r"Processing prompt (\d+)/(\d+)", txt)
        if hits:
            done = int(hits[-1][0])
    out = {"key": key, "name": meta["name"], "focus": meta["focus"],
           "decides": meta["decides"], "state": "running"}
    if total:
        out.update({"done": done, "total": total,
                    "percent": int(100 * done / total)})
    return out


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
           "stage": active_stage(),
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
