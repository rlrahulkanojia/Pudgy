#!/usr/bin/env python3
"""
Epoch-wise sample renders for the v6 run — two random expressions per checkpoint.

Purpose: a cheap visual quality-vs-epoch curve, so the golden checkpoint can be picked
by eye rather than by loss. v5 §4.4 found motion responsiveness *decayed* with training
(ep04 moved 259 px, ep18 only 221 px), so the best checkpoint is often early and a
per-epoch visual series is how you see that coming.

⚠️ THIS CANNOT RUN CONCURRENTLY WITH v6 TRAINING ON A SINGLE 80 GB CARD.
Measured on this box: training's VRAM oscillates 28 -> 70 GB as the sampler moves between
the 21-frame and 57-frame buckets, so free memory swings between ~53 GB and ~11 GB. A
concurrent job that allocates during a low-water window will push training over the limit
when the next f57 batch lands, and training dies. (v5 *did* run eval alongside training,
but v5's training held a steady ~54 GB and had no 57-frame bucket — that precedent does
not transfer.) The script therefore refuses to start unless it sees enough free VRAM, and
`--min-free` is the guard, not a suggestion.

Three safe ways to use it:

  # 1. After training finishes — renders every epoch checkpoint in one pass
  python sample_epochs_v6.py --all

  # 2. On a SECOND box, live, while this one trains. Checkpoints are mirrored to Azure
  #    every epoch, so another instance can follow along at zero risk to the run.
  python sample_epochs_v6.py --watch --from-azure

  # 3. Follow a local dir, but only render when the GPU is genuinely free
  python sample_epochs_v6.py --watch

Selection: two distinct (character, emotion) pairs per epoch, drawn with a seed derived
from the epoch number — so the pairs differ across epochs (you see more of the space)
but re-running any epoch reproduces the same pick.
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import caption_for  # noqa: E402  — same captions the model was trained on
from prep_expressions_v6 import EMOTIONS  # noqa: E402

HERE = Path(__file__).resolve().parent
KF = Path("/workspace/eval_v6/keyframes")
OUTROOT = Path("/workspace/eval_v6/samples")
TRAIN_OUT = Path("/workspace/wan_output/pudgy-expr-v6-lownoise")
CHARS = ("Pax", "Polly")

# Wan 14B I2V with fp8_scaled + heavy block-swap + lazy loading needs roughly this much.
# Deliberately conservative: the cost of being wrong is killing a 35 h training run.
MIN_FREE_MIB = 24000


def free_vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return -1


def training_alive():
    return subprocess.run(["pgrep", "-f", "wan_train_network.py"],
                          capture_output=True).returncode == 0


def epoch_of(path: Path):
    """pudgy-expr-v6-lownoise-000003.safetensors -> 3.

    musubi writes the intermediate epochs with a zero-padded suffix but the FINAL
    epoch as the bare output name (`pudgy-expr-v6-lownoise.safetensors`, no number).
    Matching only the numbered form silently drops the last checkpoint — which is the
    one most likely to be the golden, so the omission would be both invisible and
    maximally annoying. Return None here and let `with_final()` assign it.
    """
    m = re.search(r"-(\d{6})\.safetensors$", path.name)
    return int(m.group(1)) if m else None


def with_final(paths):
    """Order checkpoints, giving the un-numbered final file epoch = max numbered + 1."""
    numbered = sorted((p for p in paths if epoch_of(p)), key=epoch_of)
    final = [p for p in paths if epoch_of(p) is None
             and p.name.endswith(".safetensors") and "-state" not in p.name]
    out = [(epoch_of(p), p) for p in numbered]
    nxt = (out[-1][0] + 1) if out else 1
    for p in final:
        out.append((nxt, p))
        nxt += 1
    return out


def pick_two(epoch):
    """Two distinct (character, emotion) pairs, reproducible per epoch."""
    combos = [(c, e) for c in CHARS for e in EMOTIONS]
    return random.Random(1000 + epoch).sample(combos, 2)


def render(ckpt: Path, epoch: int, frames: int, seed: int, force: bool, blkswap=0):
    outdir = OUTROOT / f"epoch_{epoch:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    picks = pick_two(epoch)
    print(f"\n=== epoch {epoch:02d} :: {ckpt.name} ===")
    print(f"    picks: {', '.join(f'{c}/{e}' for c, e in picks)}")

    for char, emo in picks:
        tag = f"ep{epoch:02d}_{char.lower()}_{emo}_s{seed}"
        if (outdir / f"{tag}.mp4").exists():
            print(f"    skip {tag} (exists)")
            continue
        # Two independent guards, and the process check is the one that matters.
        #
        # A free-VRAM check alone is NOT sufficient and it is worth being explicit about
        # why: training's demand is periodic, not steady. Sampled during a 21-frame batch
        # it reports ~53 GB free and the check passes — then the next 57-frame batch needs
        # 70 GB, finds it taken, and the run dies. The instantaneous reading says nothing
        # about the peak that is coming a few steps later. So while a v6 training process
        # exists on this box, refuse outright.
        if training_alive() and not force:
            print(f"    ABORT: v6 training is running on this GPU. Its VRAM demand is "
                  f"periodic (28 -> 70 GB), so free memory right now ({free_vram_mib()} "
                  f"MiB) does not predict the next 57-frame batch. Render after training, "
                  f"or use --from-azure on a second box.")
            return False
        free = free_vram_mib()
        if not force and free < MIN_FREE_MIB:
            print(f"    ABORT: only {free} MiB free, need {MIN_FREE_MIB}.")
            return False
        env = dict(os.environ)
        env.update({
            "CKPT": str(ckpt), "PROMPT": caption_for(char, emo),
            "START": str(KF / f"{char.lower()}_neutral_start.png"),
            "FRAMES": str(frames), "SEED": str(seed), "TAG": tag,
            "OUTDIR": str(outdir),
            # Block-swap is a MEMORY tool and a large throughput tax (v5 4.7: 27.5 vs
            # 13.7 s/it). The guard above already guarantees training is not running, so
            # the whole card is free and 39 buys nothing. Default 0; raise only if this
            # ever has to share the GPU.
            "BLKSWAP": str(blkswap),
        })
        r = subprocess.run(["bash", str(HERE / "eval_v6.sh")], env=env,
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            print(f"    FAILED {tag}")
            return False
        print(f"    rendered {tag}")

    # One contact sheet per epoch so the series can be flipped through quickly.
    for mp4 in sorted(outdir.glob("*.mp4")):
        png = mp4.with_name(mp4.stem + "_montage.png")
        if not png.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                            "-vf", "scale=180:-1,tile=7x3", str(png)], check=False)
    return True


def local_checkpoints():
    return with_final(list(TRAIN_OUT.glob("*.safetensors")))


def pull_from_azure(dest: Path):
    """Mirror v6/weights/ down from Azure — for use on a second box."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        sys.exit("pip install azure-storage-blob")
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs and Path("/workspace/.env").exists():
        for line in Path("/workspace/.env").read_text().splitlines():
            if line.startswith("AZURE_STORAGE_CONNECTION_STRING"):
                cs = line.split("=", 1)[1].strip().strip('"')
    if not cs:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set")
    cc = BlobServiceClient.from_connection_string(cs).get_container_client(
        os.environ.get("AZURE_CONTAINER", "pudgy"))
    got = []
    for b in cc.list_blobs(name_starts_with="v6/weights/"):
        if not b.name.endswith(".safetensors"):
            continue
        p = dest / Path(b.name).name
        if p.exists() and p.stat().st_size == b.size:
            got.append(p); continue
        print(f"    pulling {Path(b.name).name} ({b.size/1e6:.0f} MB)")
        with open(p, "wb") as f:
            f.write(cc.download_blob(b.name, max_concurrency=8).readall())
        got.append(p)
    return sorted((p for p in got if epoch_of(p)), key=lambda p: epoch_of(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="render every checkpoint, once")
    ap.add_argument("--watch", action="store_true", help="follow new checkpoints as they appear")
    ap.add_argument("--from-azure", action="store_true",
                    help="pull checkpoints from Azure v6/weights/ (use on a second box)")
    ap.add_argument("--ckpt", type=Path, help="render one specific checkpoint")
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--blkswap", type=int, default=0,
                    help="blocks_to_swap for inference; 0 when the GPU is free")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--min-free", type=int, default=24000)
    ap.add_argument("--force", action="store_true",
                    help="DANGEROUS: render even with insufficient free VRAM. On the "
                         "training box this can OOM-kill the run.")
    ap.add_argument("--azure-cache", type=Path, default=Path("/workspace/v6_ckpts"))
    args = ap.parse_args()

    global MIN_FREE_MIB
    MIN_FREE_MIB = args.min_free

    if not KF.exists():
        sys.exit(f"keyframes missing at {KF} — run prep_eval_keyframes_v6.py first")

    if args.force and training_alive():
        print("!! --force with training running. Measured free VRAM swings to ~11 GB on")
        print("!! 57-frame batches; this can kill the run. Continuing because you asked.")

    if args.ckpt:
        e = epoch_of(args.ckpt) or 0
        sys.exit(0 if render(args.ckpt, e, args.frames, args.seed, args.force,
                              args.blkswap) else 1)

    done = set()
    while True:
        cks = pull_from_azure(args.azure_cache) if args.from_azure else local_checkpoints()
        todo = [(e, c) for e, c in cks if e not in done]
        if not todo and not args.watch:
            print("no new checkpoints" if cks else
                  f"no checkpoints yet in {TRAIN_OUT} — epoch 1 lands ~3.2 h in")
        for e, c in todo:
            if render(c, e, args.frames, args.seed, args.force, args.blkswap):
                done.add(e)
            else:
                break                      # stop on refusal/failure; retry next cycle
        if not args.watch:
            break
        if len(done) >= 11 and not training_alive():
            print("all epochs rendered and training has finished — done")
            break
        time.sleep(args.interval)

    print(f"\nsamples -> {OUTROOT}")


if __name__ == "__main__":
    main()
