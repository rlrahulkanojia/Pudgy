#!/usr/bin/env python3
"""Tier-2 generation: batched two-expert I2V with explicit LoRA STACKS per expert.

v7's generate path took one LoRA per expert. v8 needs lists: the v8 task vector rides on
top of its training base, v7fixed (= v7 expression + v7 motion - v2 golden, one exact
rank-48 file), and the three-file cross-check is [L_e, L_m, G x -1] (common.lora_config). musubi merges every entry additively with its own multiplier.

One model load per (config) - all jobs for a config go through a single --from_file
batch. Outputs are renamed to <tag>.mp4; existing clips are skipped, so a re-run after a
crash or a metric change costs no GPU.

Job line format (musubi --from_file):  <prompt> --i <start.png> --f <frames> --d <seed> [--n <neg>]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIT_LOW, DIT_HIGH, VAE, T5, MUSUBI, PY, lora_config)  # noqa: E402

STEPS, SHIFT, GUIDE, BOUNDARY = 25, 5.0, 5.0, 0.9     # v7 eval settings, unchanged


def run(config, jobs, outdir, dry=False):
    """jobs: [{tag, prompt, start, frames, seed, neg?}] -> {tag: Path}"""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    want = {j["tag"]: outdir / f"{j['tag']}.mp4" for j in jobs}
    pending = [j for j in jobs if not want[j["tag"]].exists()]
    print(f"   [{config}] {len(jobs)} clips, {len(pending)} to generate -> {outdir}", flush=True)
    if not pending or dry:
        return want
    low, high = lora_config(config)
    for p, _ in low + high:
        if not Path(p).exists():
            sys.exit(f"missing LoRA for {config}: {p}")
    raw = outdir / f".raw_{config.replace('/', '_').replace(':', '_')}"
    raw.mkdir(exist_ok=True)
    for f in raw.glob("*.mp4"):
        f.unlink()
    pf = raw / "prompts.txt"
    lines = []
    for j in pending:
        line = f"{j['prompt']} --i {j['start']} --f {j['frames']} --d {j['seed']}"
        if j.get("neg"):
            line += f" --n {j['neg']}"
        lines.append(line)
    pf.write_text("\n".join(lines) + "\n")
    cmd = [str(PY), "src/musubi_tuner/wan_generate_video.py",
           "--task", "i2v-A14B", "--dit", str(DIT_LOW), "--dit_high_noise", str(DIT_HIGH),
           "--timestep_boundary", str(BOUNDARY), "--vae", str(VAE), "--t5", str(T5),
           "--lora_weight", *[str(p) for p, _ in low],
           "--lora_multiplier", *[str(m) for _, m in low],
           "--lora_weight_high_noise", *[str(p) for p, _ in high],
           "--lora_multiplier_high_noise", *[str(m) for _, m in high],
           "--video_size", "1024", "1024", "--fps", "24",
           "--infer_steps", str(STEPS), "--flow_shift", str(SHIFT),
           "--guidance_scale", str(GUIDE), "--attn_mode", "sdpa",
           "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--from_file", str(pf), "--save_path", str(raw), "--output_type", "video"]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    (raw / "cmd.json").write_text(json.dumps(cmd, indent=1))
    r = subprocess.run(cmd, cwd=str(MUSUBI), env=env)
    if r.returncode:
        sys.exit(f"generation failed for {config} (see {raw})")
    produced = sorted(raw.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if len(produced) != len(pending):
        sys.exit(f"{config}: {len(produced)} clips for {len(pending)} prompts - refusing to "
                 f"guess the mapping (musubi names by timestamp)")
    for src, j in zip(produced, pending):       # mtime order == prompt-file order
        src.rename(want[j["tag"]])
    (outdir / f"jobs_{config.replace('/', '_').replace(':', '_')}.json").write_text(
        json.dumps(jobs, indent=1))
    return want
