#!/usr/bin/env python3
"""
Batched v6 rendering — one model load per CHECKPOINT instead of one per clip.

Why this exists. Profiling the per-clip renderer showed the GPU sitting at 0-20%
utilisation while memory climbed 4 -> 45 GB: the cost is loading and fp8-quantising two
14B experts (~57 GB) from disk, not diffusion. At ~20 min/clip that overhead dominated
completely. musubi supports `--from_file`, which runs many prompts against a single
loaded model — v5's report already recorded the same finding ("one model load instead of
ten... 2x faster"), and the per-clip design here failed to apply it.

`--lora_weight` is a GLOBAL argument, not per-prompt, so a batch cannot span checkpoints:
one batch == one checkpoint. That still collapses N clips per checkpoint into one load.

Prompt-file format (musubi `parse_prompt_line`):
    <prompt text> --w 1024 --h 1024 --f 21 --d 42 --i /path/start.png

  python batch_render_v6.py --ckpt <lora.safetensors> --epoch 11 \
      --picks "Polly:surprised,Pax:angry"
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import caption_for  # noqa: E402

KF = Path("/workspace/eval_v6/keyframes")
OUTROOT = Path("/workspace/eval_v6/samples")
M = Path("/workspace/wan_models")
GOLD = Path("/workspace/wan_output/v2_golden")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
REPO = "/workspace/musubi-tuner"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--epoch", required=True, type=int)
    ap.add_argument("--picks", required=True,
                    help='comma list of Char:emotion, e.g. "Pax:happy,Polly:angry"')
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--blkswap", type=int, default=0)
    args = ap.parse_args()

    picks = []
    for tok in args.picks.split(","):
        c, e = tok.strip().split(":")
        picks.append((c.strip(), e.strip()))

    outdir = OUTROOT / f"epoch_{args.epoch:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    todo = [(c, e) for c, e in picks
            if not (outdir / f"ep{args.epoch:02d}_{c.lower()}_{e}_s{args.seed}.mp4").exists()]
    if not todo:
        print(f"epoch {args.epoch:02d}: nothing to render")
        return

    scratch = Path(tempfile.mkdtemp(prefix=f"batch_ep{args.epoch:02d}_", dir=str(outdir)))
    lines = []
    for char, emo in todo:
        start = KF / f"{char.lower()}_neutral_start.png"
        lines.append(f"{caption_for(char, emo)} --w 1024 --h 1024 "
                     f"--f {args.frames} --d {args.seed} --i {start}")
    pf = scratch / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")

    print(f"epoch {args.epoch:02d} :: {args.ckpt.name}")
    for c, e in todo:
        print(f"    {c}/{e}")
    print(f"    ONE model load for {len(todo)} clip(s)")

    dit = M / "comfy22/split_files/diffusion_models"
    cmd = [
        PY, "src/musubi_tuner/wan_generate_video.py",
        "--task", "i2v-A14B",
        "--dit", str(dit / "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        "--dit_high_noise", str(dit / "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        "--timestep_boundary", "0.9",
        "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
        "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
        "--lora_weight", str(args.ckpt), "--lora_multiplier", "1.0",
        "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
        "--lora_multiplier_high_noise", "1.0",
        "--fps", "24", "--infer_steps", str(args.steps),
        "--flow_shift", "5.0", "--guidance_scale", "5.0",
        "--attn_mode", "sdpa",
        "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
        "--from_file", str(pf),
        "--save_path", str(scratch), "--output_type", "video",
    ]
    if args.blkswap > 0:
        cmd += ["--blocks_to_swap", str(args.blkswap), "--lazy_loading"]

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode:
        sys.exit(f"batch render failed for epoch {args.epoch}")

    # musubi names outputs by timestamp; they come out in prompt-file order.
    produced = sorted(scratch.glob("*.mp4"))
    if len(produced) != len(todo):
        print(f"!! expected {len(todo)} clips, got {len(produced)}")
    for (char, emo), src in zip(todo, produced):
        dest = outdir / f"ep{args.epoch:02d}_{char.lower()}_{emo}_s{args.seed}.mp4"
        src.rename(dest)
        print(f"    -> {dest.name}")
    for leftover in scratch.glob("*"):
        leftover.unlink()
    scratch.rmdir()


if __name__ == "__main__":
    main()
