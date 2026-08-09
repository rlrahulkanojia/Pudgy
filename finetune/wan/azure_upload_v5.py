#!/usr/bin/env python3
"""
Upload the v5 pilot artifacts to Azure, following the existing container layout.

Layout mirrors v1/ v2/ v4/ in the `pudgy` container:

    v5/
    ├── weights/
    │   ├── highnoise/            run 1 — expression on the MOTION expert (rejected by A/B)
    │   ├── lownoise/             run 2 — expression on the IDENTITY expert (GOLDEN)
    │   └── lora_happy_lownoise_GOLDEN_ep18.safetensors   <- the winner, named per v2 convention
    ├── eval/                     generated videos + montages for all 4 test modes
    ├── logs/                     tensorboard events + trainer stdout
    └── docs/                     report, plan docs, training/eval scripts, dataset config

Resume states (`*-state/`, 15.6 GB) are excluded by default — same as the v2 run, where
`state/` was only pushed with an explicit flag. Pass --with-states to include them.
"""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.storage.blob import BlobServiceClient

CONTAINER = "pudgy"
PREFIX = "v5"


def conn_string() -> str:
    for line in open("/workspace/.env"):
        if line.startswith("AZURE_STORAGE_CONNECTION_STRING"):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("no AZURE_STORAGE_CONNECTION_STRING in /workspace/.env")


def collect(with_states: bool) -> list[tuple[Path, str]]:
    """(local_path, blob_name) pairs."""
    items: list[tuple[Path, str]] = []

    runs = {
        "highnoise": Path("/workspace/wan_output/pudgy-happy-expr-highnoise-v1"),
        "lownoise":  Path("/workspace/wan_output/pudgy-happy-expr-lownoise-v1"),
    }
    for tag, d in runs.items():
        if not d.exists():
            continue
        for f in sorted(d.glob("*.safetensors")):
            items.append((f, f"{PREFIX}/weights/{tag}/{f.name}"))
        for f in sorted(d.rglob("events.out.tfevents.*")):
            items.append((f, f"{PREFIX}/logs/{tag}/{f.name}"))
        if with_states:
            for f in sorted(d.glob("*-state/*")):
                items.append((f, f"{PREFIX}/state/{tag}/{f.parent.name}/{f.name}"))

    # the A/B winner, surfaced at the top of weights/ under the v2 GOLDEN naming
    golden = runs["lownoise"] / "pudgy-happy-expr-lownoise-v1.safetensors"
    if golden.exists():
        items.append((golden, f"{PREFIX}/weights/lora_happy_lownoise_GOLDEN_ep18.safetensors"))

    ev = Path("/workspace/eval_v5")
    # is_file() matters: musubi's --output_type both creates DIRECTORIES named "*.mp4",
    # so a bare glob yields directories and the upload dies on IsADirectoryError.
    for f in list(ev.rglob("*.mp4")) + list(ev.rglob("*.png")):
        if f.is_file():
            items.append((f, f"{PREFIX}/eval/{f.relative_to(ev)}"))

    for f in [Path("/workspace/train_high.log"), Path("/workspace/train_low.log")]:
        if f.exists():
            items.append((f, f"{PREFIX}/logs/{f.name}"))

    repo = Path("/workspace/Pudgy")
    docs = [
        "docs/training_reports/v5/REPORT_happy_pilot.md",
        "training_approach/v5/Training_Approach_v5.md",
        "training_approach/v5/Training_Approach_v5_Happy_Expression_LoRA.md",
        "finetune/wan/prep_happy_v5.py",
        "finetune/wan/train_pudgy_happy_expr.sh",
        "finetune/wan/eval_happy_v5.sh",
        "finetune/wan/wandb_mirror.py",
        "finetune/wan/dataset_config_happy.toml",
        "finetune/wan/dataset_happy.jsonl",
        "finetune/wan/azure_upload_v5.py",
    ]
    for rel in docs:
        f = repo / rel
        if f.exists():
            items.append((f, f"{PREFIX}/docs/{Path(rel).name}"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-states", action="store_true", help="also upload resume states (+15.6 GB)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    svc = BlobServiceClient.from_connection_string(conn_string())
    cc = svc.get_container_client(CONTAINER)
    existing = {b.name: b.size for b in cc.list_blobs(name_starts_with=PREFIX)}

    items = collect(args.with_states)
    todo = [(p, n) for p, n in items if existing.get(n) != p.stat().st_size]
    total = sum(p.stat().st_size for p, _ in todo)
    print(f"{len(items)} artifacts; {len(todo)} to upload ({total/1e9:.2f} GB); "
          f"{len(items)-len(todo)} already present and identical")
    if args.dry_run:
        for p, n in todo[:25]:
            print(f"  {p.stat().st_size/1e6:9.1f} MB  {n}")
        return

    done = [0]
    def push(pair):
        p, n = pair
        with open(p, "rb") as fh:
            cc.upload_blob(name=n, data=fh, overwrite=True, max_concurrency=4)
        done[0] += 1
        print(f"  [{done[0]}/{len(todo)}] {n}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(push, it) for it in todo]):
            fut.result()

    after = list(cc.list_blobs(name_starts_with=PREFIX))
    print(f"\nDONE — {PREFIX}/ now holds {len(after)} blobs, "
          f"{sum(b.size for b in after)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
