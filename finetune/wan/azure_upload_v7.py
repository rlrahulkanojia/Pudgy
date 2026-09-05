#!/usr/bin/env python3
"""
Upload the v7 run artifacts to Azure, following the existing container layout.

v7 differs from v5/v6 in one structural way that this script has to reflect: **there are
THREE runs, not one**, because motion and expression train as separate LoRAs on separate
experts and the motion expert is chosen by an A/B:

    v7/
    ├── weights/
    │   ├── expr-lownoise/        expression LoRA — low-noise expert (settled by v5's A/B)
    │   ├── motion-highnoise/     motion A/B arm A
    │   ├── motion-lownoise/      motion A/B arm B
    │   └── <golden>.safetensors  whichever checkpoints the gates select, surfaced at top
    ├── eval/                     gate matrices + montages
    ├── logs/                     tensorboard events + trainer stdout
    └── docs/                     plan, prep/train/eval/gate scripts, dataset configs

Run it on the GPU box after the gates. Idempotent: a blob whose size already matches is
skipped, so a re-run after more eval only pushes what is new.

    python finetune/wan/azure_upload_v7.py --dry-run
    python finetune/wan/azure_upload_v7.py
    python finetune/wan/azure_upload_v7.py --with-states     # + resume states (large)

Auth: reads AZURE_STORAGE_CONNECTION_STRING from /workspace/.env — the same file the box
already needs for `az storage blob download-batch`. See the v7 handover doc, step 1.
"""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.storage.blob import BlobServiceClient

CONTAINER = "pudgy"
PREFIX = "v7"

# Run name -> output dir. These match the NAME= values in the handover runbook; if you
# rename a run, rename it here too or its weights are silently not uploaded.
RUNS = {
    "expr-lownoise":    Path("/workspace/wan_output/pudgy-v7-expr-lownoise"),
    "motion-highnoise": Path("/workspace/wan_output/pudgy-v7-motion-highnoise"),
    "motion-lownoise":  Path("/workspace/wan_output/pudgy-v7-motion-lownoise"),
}
EVAL_DIRS = [Path("/workspace/eval_v7"), Path("/workspace/gates_v7")]
REPO = Path("/workspace/Pudgy")
DOCS = [
    "training_approach/v7/Training_Approach_v7.md",
    "training_approach/v7/GPU_HANDOFF_v7.md",
    "finetune/wan/prep_v7.py",
    "finetune/wan/train_pudgy_v7.sh",
    "finetune/wan/azure_upload_v7.py",
    "finetune/wan/eval_v6.sh",
    "finetune/wan/gates_v6.py",
    "finetune/wan/gate_gl_v6.py",
    "finetune/wan/gate_gf_v6.py",
    "finetune/wan/gates_remaining_v6.py",
    "finetune/wan/sample_epochs_v6.py",
    "docs/documents/Client_Data_Request_Round4.md",
    "docs/documents/Corrupt_Files_To_Reexport.md",
]


def conn_string() -> str:
    env = Path("/workspace/.env")
    if not env.exists():
        sys.exit("no /workspace/.env — copy it to the box (handover step 1)")
    for line in env.read_text().splitlines():
        if line.startswith("AZURE_STORAGE_CONNECTION_STRING"):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("no AZURE_STORAGE_CONNECTION_STRING in /workspace/.env")


def collect(with_states: bool) -> list[tuple[Path, str]]:
    """(local_path, blob_name) pairs. Missing runs are skipped, not fatal — the motion
    A/B loser may legitimately have been deleted, and expression may finish first."""
    items: list[tuple[Path, str]] = []
    for tag, d in RUNS.items():
        if not d.exists():
            print(f"  (no output dir for {tag} — skipping)")
            continue
        for f in sorted(d.glob("*.safetensors")):
            items.append((f, f"{PREFIX}/weights/{tag}/{f.name}"))
        for f in sorted(d.rglob("events.out.tfevents.*")):
            items.append((f, f"{PREFIX}/logs/{tag}/{f.name}"))
        if with_states:
            for f in sorted(d.glob("*-state/*")):
                items.append((f, f"{PREFIX}/state/{tag}/{f.parent.name}/{f.name}"))

    for ev in EVAL_DIRS:
        if not ev.exists():
            continue
        # is_file() matters: musubi's --output_type both creates DIRECTORIES named
        # "*.mp4", so a bare glob yields directories and the upload dies on IsADirectory.
        for f in list(ev.rglob("*.mp4")) + list(ev.rglob("*.png")) + list(ev.rglob("*.json")):
            if f.is_file():
                items.append((f, f"{PREFIX}/eval/{ev.name}/{f.relative_to(ev)}"))

    for f in sorted(Path("/workspace").glob("train_v7_*.log")):
        items.append((f, f"{PREFIX}/logs/{f.name}"))

    for rel in DOCS:
        f = REPO / rel
        if f.exists():
            items.append((f, f"{PREFIX}/docs/{Path(rel).name}"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-states", action="store_true",
                    help="also upload resume states (large — same policy as v2/v5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cc = BlobServiceClient.from_connection_string(conn_string()).get_container_client(CONTAINER)
    existing = {b.name: b.size for b in cc.list_blobs(name_starts_with=PREFIX)}

    items = collect(args.with_states)
    todo = [(p, n) for p, n in items if existing.get(n) != p.stat().st_size]
    total = sum(p.stat().st_size for p, _ in todo)
    print(f"{len(items)} artifacts; {len(todo)} to upload ({total/1e9:.2f} GB); "
          f"{len(items)-len(todo)} already present and identical")
    if args.dry_run:
        for p, n in todo[:30]:
            print(f"  {p.stat().st_size/1e6:9.1f} MB  {n}")
        return
    if not todo:
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
