#!/usr/bin/env python3
"""
G-L — the length de-confound gate. The most important test of the v6 run.

The problem it tests. In `v6_expressions_272` each emotion ships at exactly one frame
count (21 happy / 29 surprised / 37 angry / 57 neutral) and no two emotions share one.
Clip length therefore predicts emotion with 100% accuracy in the training data. musubi
buckets by frame count, so "6 latent frames" and "happy" co-occur in every single
example. The fix (a second truncated copy of each longer emotion, so the short bucket
carries all four) was ruled out — original lengths only — so the confound went into
training unmitigated and this gate is the safeguard.

The failure mode is concrete and would make the model unusable: **ask for 21 frames and
get happy no matter what you prompt.**

Method. One checkpoint, one start frame, one seed per pass. Generate every emotion at
every length — the full 4x4 — then ask two questions:

  prompt effect  At a FIXED length, do the four emotion prompts diverge? If the model
                 reads length instead of the caption, they will collapse together. This
                 is the pass/fail signal.
  length effect  For a FIXED emotion, how much does changing length change the result?
                 Large values here mean length is steering content.

Measured on the FACE crop, not the whole frame: ~85% of these frames are body, outline
and flat background that the prompt cannot alter, and whole-frame SSIM understates a
facial change badly enough to put a visually obvious difference 0.0015 from "identical"
(see gates_v6.ssim_pair).

Everything runs in ONE model load via --from_file. Loading and fp8-quantising the two
14B experts costs ~15 min and dominates a per-clip loop; the 4x4 matrix is a single
checkpoint, so it batches perfectly.

  python gate_gl_v6.py --ckpt <lora.safetensors> [--seeds 42,7,123]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import (  # noqa: E402
    FACE_DISTINCT, caption_for, face_box, read_video, ssim_pair,
)
from prep_expressions_v6 import EMOTIONS  # noqa: E402

KF = Path("/workspace/eval_v6/keyframes")
OUT = Path("/workspace/eval_v6/gl")
M = Path("/workspace/wan_models")
GOLD = Path("/workspace/wan_output/v2_golden")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
REPO = "/workspace/musubi-tuner"
DEFAULT_LENGTHS = [21, 29, 37, 57]
LENGTHS = list(DEFAULT_LENGTHS)


def render_matrix(ckpt, char, seed, outdir, steps=25, blkswap=0):
    """All (emotion x length) clips for one seed, in a single model load."""
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(e, n) for e in EMOTIONS for n in LENGTHS
            if not (outdir / f"gl_{char.lower()}_{e}_f{n}_s{seed}.mp4").exists()]
    if not todo:
        print(f"  seed {seed}: complete")
        return
    scratch = Path(tempfile.mkdtemp(prefix=f"gl_s{seed}_", dir=str(outdir)))
    start = KF / f"{char.lower()}_neutral_start.png"
    lines = [f"{caption_for(char, e)} --w 1024 --h 1024 --f {n} --d {seed} --i {start}"
             for e, n in todo]
    pf = scratch / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")
    print(f"  seed {seed}: {len(todo)} clips in ONE model load")

    dit = M / "comfy22/split_files/diffusion_models"
    cmd = [
        PY, "src/musubi_tuner/wan_generate_video.py",
        "--task", "i2v-A14B",
        "--dit", str(dit / "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        "--dit_high_noise", str(dit / "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        "--timestep_boundary", "0.9",
        "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
        "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
        "--lora_weight", str(ckpt), "--lora_multiplier", "1.0",
        "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
        "--lora_multiplier_high_noise", "1.0",
        "--fps", "24", "--infer_steps", str(steps),
        "--flow_shift", "5.0", "--guidance_scale", "5.0", "--attn_mode", "sdpa",
        "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
        "--from_file", str(pf), "--save_path", str(scratch), "--output_type", "video",
    ]
    if blkswap > 0:
        cmd += ["--blocks_to_swap", str(blkswap), "--lazy_loading"]
    env = dict(os.environ); env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if subprocess.run(cmd, cwd=REPO, env=env).returncode:
        sys.exit("G-L render failed")

    produced = sorted(scratch.glob("*.mp4"))
    if len(produced) != len(todo):
        sys.exit(f"expected {len(todo)} clips, got {len(produced)} — cannot map reliably")
    for (e, n), src in zip(todo, produced):
        src.rename(outdir / f"gl_{char.lower()}_{e}_f{n}_s{seed}.mp4")
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()


def score(char, seeds, outdir):
    fb = face_box(1.00)
    vids = {}
    for e in EMOTIONS:
        for n in LENGTHS:
            for s in seeds:
                p = outdir / f"gl_{char.lower()}_{e}_f{n}_s{s}.mp4"
                if p.exists():
                    vids[(e, n, s)] = read_video(p)

    prompt_effect, length_effect = [], []
    for s in seeds:
        for n in LENGTHS:
            for a, b in itertools.combinations(EMOTIONS, 2):
                if (a, n, s) in vids and (b, n, s) in vids:
                    v = ssim_pair(vids[(a, n, s)], vids[(b, n, s)], region=fb)
                    prompt_effect.append({"length": n, "pair": f"{a}|{b}", "seed": s,
                                          "face_ssim": round(v, 4),
                                          "distinct": v < FACE_DISTINCT})
        for e in EMOTIONS:
            for x, y in itertools.combinations(LENGTHS, 2):
                if (e, x, s) in vids and (e, y, s) in vids:
                    v = ssim_pair(vids[(e, x, s)], vids[(e, y, s)], region=fb)
                    length_effect.append({"emotion": e, "lengths": f"{x}|{y}",
                                          "seed": s, "face_ssim": round(v, 4)})

    print(f"\n=== G-L: prompt effect at each length ({char}) ===")
    print("  at a fixed length, do the 4 emotion prompts diverge?")
    for n in LENGTHS:
        rows = [r for r in prompt_effect if r["length"] == n]
        if not rows:
            continue
        vals = [r["face_ssim"] for r in rows]
        nd = sum(1 for r in rows if r["distinct"])
        print(f"  f{n:<3} mean {np.mean(vals):.4f}  worst {max(vals):.4f}  "
              f"distinct {nd}/{len(rows)}")

    print(f"\n=== G-L: length effect per emotion ({char}) ===")
    print("  for a fixed emotion, how much does changing length change it?")
    for e in EMOTIONS:
        rows = [r for r in length_effect if r["emotion"] == e]
        if rows:
            vals = [r["face_ssim"] for r in rows]
            print(f"  {e:<10} mean {np.mean(vals):.4f}  min {min(vals):.4f}")

    # Distinguish "no data" from "failed". An empty matrix reporting FAIL would be a
    # false negative of exactly the kind this harness exists to avoid.
    if not prompt_effect:
        ok = None
    else:
        ok = all(r["distinct"] for r in prompt_effect)
    worst = max(prompt_effect, key=lambda r: r["face_ssim"]) if prompt_effect else None
    return {"gate": "G-L", "character": char, "seeds": seeds,
            "prompt_effect": prompt_effect, "length_effect": length_effect,
            "worst_prompt_pair": worst, "pass": ok,
            "threshold_face_ssim": FACE_DISTINCT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--char", default="Pax")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--blkswap", type=int, default=0)
    ap.add_argument("--lengths", default=None,
                    help="comma list, e.g. 21 — restrict the matrix. Golden selection needs "
                         "one length; all four costs 4x for no extra selection signal.")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.lengths:
        globals()["LENGTHS"] = [int(x) for x in args.lengths.split(",") if x.strip()]
    outdir = OUT / args.ckpt.stem
    print(f"G-L :: {args.ckpt.name} :: {args.char} :: seeds {seeds}")
    print(f"matrix: {len(EMOTIONS)} emotions x {len(LENGTHS)} lengths = "
          f"{len(EMOTIONS)*len(LENGTHS)} clips per seed")

    if not args.score_only:
        for s in seeds:
            render_matrix(args.ckpt, args.char, s, outdir, args.steps, args.blkswap)

    res = score(args.char, seeds, outdir)
    rep = OUT / f"G-L_{args.ckpt.stem}_{args.char}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(res, indent=2))

    print(f"\n{'='*62}")
    verdict = "NO DATA (nothing rendered)" if res["pass"] is None else (
        "PASS" if res["pass"] else "FAIL")
    print(f"G-L: {verdict}")
    if res["worst_prompt_pair"]:
        w = res["worst_prompt_pair"]
        print(f"  worst pair: {w['pair']} at f{w['length']} = {w['face_ssim']} "
              f"(threshold < {FACE_DISTINCT})")
    if len(seeds) < 3:
        print(f"  NOTE: {len(seeds)} seed(s). A single seed can FAIL the gate but cannot")
        print(f"        pass it — v4 5.1 found seed 42 alone hid 40-66% failure rates.")
    print(f"  report: {rep}")


if __name__ == "__main__":
    main()
