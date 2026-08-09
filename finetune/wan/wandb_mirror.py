#!/usr/bin/env python3
"""
Mirror a musubi tensorboard run into wandb, live.

Why this exists: run 1 was launched before a WANDB_API_KEY was available, so it
logs to tensorboard. Restarting it to get native wandb would throw away hours of
A100 time. This tails the tfevents file instead and republishes each scalar into
a wandb run, so the run shows up live and complete without touching training.

  python wandb_mirror.py <run_name> [--logdir DIR] [--poll SEC]

Idempotent: the wandb run id is derived from the run name and resumed, so
restarting the mirror appends rather than creating a duplicate run.
"""
import argparse, os, time, glob, sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

OUT_ROOT = Path("/workspace/wan_output")


def find_event_dir(run_name: str) -> Path | None:
    hits = sorted(glob.glob(str(OUT_ROOT / run_name / "logs" / "*" / "*" / "events.out.tfevents.*")))
    return Path(hits[-1]).parent if hits else None


def training_alive(run_name: str) -> bool:
    tag = "lownoise" if "lownoise" in run_name else "highnoise"
    return os.system(f"pgrep -f 'wan_train_network.*happy' >/dev/null 2>&1") == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    ev_dir = find_event_dir(args.run_name)
    while ev_dir is None:
        print(f"waiting for tfevents under {args.run_name} ...", flush=True)
        time.sleep(20)
        ev_dir = find_event_dir(args.run_name)
    print(f"mirroring {ev_dir}", flush=True)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "pudgy"),
        entity=os.environ.get("WANDB_ENTITY", "rlrahulkanojia"),
        name=args.run_name,
        id=args.run_name.replace(".", "-"),
        resume="allow",
        config={
            "base": "Wan2.2-I2V-A14B",
            "expert": "low-noise" if "lownoise" in args.run_name else "high-noise",
            "init_from": "v2 lora_*_GOLDEN_ep40 (rank16/alpha32)",
            "dataset": "v5 pilot: Pax/happy, 7 angles x 4 backgrounds = 28 clips",
            "resolution": "1024x1024x21",
            "rank": 16, "alpha": 32, "lr": 3e-5,
            "epochs": 18, "steps": 1008,
            "source": "azure pudgy/interation_3/03_expression_clips/Pax/happy",
        },
    )
    print(f"wandb run: {run.url}", flush=True)
    Path("/workspace/wandb_urls.txt").open("a").write(f"{args.run_name}\t{run.url}\n")

    last = -1
    idle = 0
    while True:
        ea = EventAccumulator(str(ev_dir)); ea.Reload()
        tags = ea.Tags().get("scalars", [])
        by_step: dict[int, dict] = {}
        for t in tags:
            for s in ea.Scalars(t):
                if s.step > last:
                    by_step.setdefault(s.step, {})[t] = s.value
        if by_step:
            for step in sorted(by_step):
                run.log(by_step[step], step=step)
            last = max(by_step)
            idle = 0
            print(f"  logged through step {last}", flush=True)
        else:
            idle += 1

        if not training_alive(args.run_name) and idle >= 3:
            print("training finished and no new scalars — closing mirror", flush=True)
            break
        time.sleep(args.poll)

    run.finish()


if __name__ == "__main__":
    main()
