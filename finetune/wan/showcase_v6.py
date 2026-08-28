#!/usr/bin/env python3
"""
v6 showcase — a curated demonstration set from the golden checkpoint.

Unlike the gate renders, which deliberately probe the model's weak points, this is
meant to show what v6 actually does well, while still being honest about coverage.
Every clip runs in ONE model load (`--from_file`): inference cost here is dominated by
loading and fp8-quantising two 14B experts (~57 GB), not by generation.

Composition — 12 clips, chosen so each earns its place:

  core (8)      both characters x all four emotions, close-up, seed 42.
                This is the capability v6 was trained for and the thing to look at first.
  unseen bg (2) Pax happy and Polly surprised on LAVENDER — a ground never trained.
                v5 measured 2/255 background drift on an unseen ground; this shows
                whether v6 kept that generalisation.
  shot size (1) Pax angry at WIDE. Framing is promptable via the shot-size ladder, and
                the open question is whether expression stays legible when the face is
                ~55% linear size before the 8x VAE.
  long hold (1) Polly neutral at 57 frames. The only sustained-hold data in the
                programme; v5's expression relaxed by f11 on 21 frames.

  python showcase_v6.py --ckpt <golden.safetensors> [--seed 42]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import caption_for  # noqa: E402
from prep_expressions_v6 import EMOTIONS  # noqa: E402

KF = Path("/workspace/eval_v6/keyframes")
OUT = Path("/workspace/eval_v6/showcase")
M = Path("/workspace/wan_models")
GOLD = Path("/workspace/wan_output/v2_golden")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
REPO = "/workspace/musubi-tuner"


# Four variants per (character, emotion). Ground AND seed both change, so the set
# demonstrates range rather than four near-identical takes: it shows the expression
# survives a background swap and is not a fluke of one seed. All four grounds here
# were trained, so this is a fair showcase, not a robustness test — the unseen-ground
# probes below are what test generalisation.
VARIANTS = [("white", 42), ("blue", 7), ("peach", 123), ("mint", 5)]


def build(seed):
    """(label, char, emotion, zoom, bg, start-keyframe stem, frames)."""
    items = []
    for char in ("Pax", "Polly"):
        for emo in EMOTIONS:
            for bg, sd in VARIANTS:
                items.append((f"{char.lower()}_{emo}_{bg}_s{sd}", char, emo, 1.00, bg,
                              f"{char.lower()}_neutral" if bg == "white"
                              else f"{char.lower()}_neutral_{bg}", 21, sd))
    items.append(("pax_happy_UNSEEN_lavender", "Pax", "happy", 1.00, "lavender",
                  "pax_neutral_lavender", 21, seed))
    items.append(("polly_surprised_UNSEEN_lavender", "Polly", "surprised", 1.00,
                  "lavender", "polly_neutral_lavender", 21, seed))
    items.append(("pax_angry_WIDE", "Pax", "angry", 0.55, "white",
                  "pax_neutral_wide", 21, seed))
    items.append(("polly_neutral_LONG57", "Polly", "neutral", 1.00, "white",
                  "polly_neutral", 57, seed))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    items = build(args.seed)
    todo = [i for i in items if not (OUT / f"{i[0]}.mp4").exists()]
    print(f"showcase :: {args.ckpt.name}")
    print(f"  {len(items)} clips, {len(todo)} to render, ONE model load")
    for lab, char, emo, zoom, bg, kf, fr, sd in todo:
        print(f"    {lab:<38} {char}/{emo} zoom {zoom} bg {bg} {fr}f seed {sd}")
    if args.dry_run or not todo:
        return

    scratch = Path(tempfile.mkdtemp(prefix="showcase_", dir=str(OUT)))
    lines = []
    for lab, char, emo, zoom, bg, kf, fr, sd in todo:
        start = KF / f"{kf}_start.png"
        if not start.exists():
            sys.exit(f"missing keyframe {start}")
        lines.append(f"{caption_for(char, emo, zoom=zoom, bg=bg)} "
                     f"--w 1024 --h 1024 --f {fr} --d {sd} --i {start}")
    pf = scratch / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")

    dit = M / "comfy22/split_files/diffusion_models"
    cmd = [
        PY, "src/musubi_tuner/wan_generate_video.py", "--task", "i2v-A14B",
        "--dit", str(dit / "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        "--dit_high_noise", str(dit / "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        "--timestep_boundary", "0.9",
        "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
        "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
        "--lora_weight", str(args.ckpt), "--lora_multiplier", "1.0",
        "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
        "--lora_multiplier_high_noise", "1.0",
        "--fps", "24", "--infer_steps", str(args.steps), "--flow_shift", "5.0",
        "--guidance_scale", "5.0", "--attn_mode", "sdpa",
        "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
        "--from_file", str(pf), "--save_path", str(scratch), "--output_type", "video",
    ]
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if subprocess.run(cmd, cwd=REPO, env=env).returncode:
        sys.exit("showcase render failed")

    produced = sorted(scratch.glob("*.mp4"))
    if len(produced) != len(todo):
        sys.exit(f"expected {len(todo)} clips, got {len(produced)}")
    for (lab, *_rest), src in zip(todo, produced):
        dest = OUT / f"{lab}.mp4"
        src.rename(dest)
        print(f"    -> {dest.name}")
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()

    # Contact sheet per clip, for quick review.
    for mp4 in sorted(OUT.glob("*.mp4")):
        png = mp4.with_name(mp4.stem + "_sheet.png")
        if not png.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                            "-vf", "select='not(mod(n\\,4))',scale=200:-1,tile=6x1",
                            str(png)], check=False)
    print(f"\nshowcase -> {OUT}")


if __name__ == "__main__":
    main()
