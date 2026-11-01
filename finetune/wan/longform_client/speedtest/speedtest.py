"""Isolated speed test, scheduled after the long-form batch: the approved full settings plus
  1. SageAttention 1.0.6 (--attn_mode sageattn; installed only in ./venv, which reads the production
     venv's other packages through a .pth and never changes it)
  2. Wan2.1-style time embedding (--force_v2_1_time_embedding)
  3. torch.compile (--compile; caches in ./cache)
on walking take 1 - same prompt, keyframe and seed as the approved sample A - compared with that take.

python speedtest.py   -> out/walking_fast_t1.mp4, out/compare.{json,png}, out/compare_1440x1280.mp4,
                         Azure pudgy/2026-10-28_longform/speedtest/
Writes nothing under /workspace/longform_client.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LOG = HERE / "speedtest.log"            # this script's stdout (run_after_batch.sh), incl. musubi's progress bars
sys.path.insert(0, "/workspace/longform_client")
import produce as PR                    # noqa: E402  (also points gen_v8 at the v8merged LoRA)
import assemble_client as AC            # noqa: E402
import upload                           # noqa: E402
import gen_v8                           # noqa: E402

BASE_S_PER_STEP = 82.0                  # sdpa, measured in the batch (logs/stream.log)
REF = PR.TAKES_DIR / "walking_chain_t1.mp4"
FAST = ["--attn_mode", "sageattn", "--force_v2_1_time_embedding", "--compile", "--compile_cache_size_limit", "32"]


def step_times():
    """Seconds per denoising step, from the progress bar's elapsed time."""
    txt = LOG.read_text(errors="ignore").replace("\r", "\n")
    el = {}
    for step, m, s in re.findall(r"\|\s*(\d+)/25 \[(\d+):(\d+)<", txt):
        el[int(step)] = int(m) * 60 + int(s)
    steps = sorted(el)
    return [el[b] - el[a] for a, b in zip(steps, steps[1:]) if b == a + 1]


def main():
    gen_v8.PY = HERE / "venv/bin/python"
    os.environ["GEN_EXTRA"] = " ".join(["--video_size", "1280", "720", *FAST])   # later --attn_mode wins
    os.environ["TRITON_CACHE_DIR"] = str(HERE / "cache/triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(HERE / "cache/inductor")
    job = dict(PR.jobs_for(1, PR.select(["walking"]))[0], tag="walking_fast_t1")
    t = time.time()
    take = gen_v8.run("v8merged", [job], OUT)["walking_fast_t1"]
    wall = (time.time() - t) / 60

    ref, new = AC.decode(REF)[:AC.FRAMES], AC.decode(take)[:AC.FRAMES]
    rf, nf = ref.astype(np.float32), new.astype(np.float32)
    mse = ((rf - nf) ** 2).mean((1, 2, 3))
    psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
    motion = lambda v: float(np.median(np.abs(np.diff(v[:, ::4, ::4], axis=0)).mean((1, 2, 3))))
    border = np.zeros((AC.H, AC.W), bool); border[:60] = border[-60:] = True; border[:, :40] = border[:, -40:] = True
    drift = lambda v: round(float(np.abs(v[-1] - v[0])[border].mean()), 2)
    st = step_times()
    steady = float(np.median(st)) if st else None
    res = dict(settings=FAST, take_minutes_incl_load_and_compile=round(wall, 1),
               s_per_step_median=steady, s_per_step_baseline=BASE_S_PER_STEP,
               speedup=round(BASE_S_PER_STEP / steady, 2) if steady else None,
               slowest_steps_s=sorted(st)[-3:], psnr_vs_sampleA_mean=round(float(psnr.mean()), 1),
               psnr_vs_sampleA_min=round(float(psnr.min()), 1), motion_sampleA=round(motion(rf), 2),
               motion_fast=round(motion(nf), 2), border_drift_sampleA=drift(rf), border_drift_fast=drift(nf))
    (OUT / "compare.json").write_text(json.dumps(res, indent=1))

    idx = np.linspace(0, AC.FRAMES - 1, 8).round().astype(int)
    rows = [np.concatenate([np.asarray(Image.fromarray(v[i]).resize((180, 320))) for i in idx], 1) for v in (ref, new)]
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / "compare.png")
    side = OUT / "compare_1440x1280.mp4"      # left: sample A (current settings), right: fast settings
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{2 * AC.W}x{AC.H}", "-r", "24", "-i", "-", "-vf", "format=yuv420p",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart",
                            "-an", str(side)], stdin=subprocess.PIPE)
    enc.stdin.write(np.concatenate([ref, new], 2).tobytes()); enc.stdin.close()
    if enc.wait():
        sys.exit("encode failed")
    links = upload.main("2026-10-28", files=[side], sub="speedtest")
    print("## SPEEDTEST result", json.dumps(res), flush=True)
    print(f"## SPEEDTEST compare video (left: current, right: fast): {links[side.name]}", flush=True)


if __name__ == "__main__":
    main()
