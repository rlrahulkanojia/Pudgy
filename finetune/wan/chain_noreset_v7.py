#!/usr/bin/env python3
"""Continuous long-form video by NO-RESET beat chaining.

The existing v7/eval/chain10s/ builds 10 s by restarting each expression block from the
neutral keyframe (`reset: true` in its manifest). That guarantees a visible pop at every
block boundary — the character snaps back to a canonical pose mid-video.

This chains without resets: beat N+1 is conditioned on the LAST FRAME of beat N, so the
join is frame-exact by construction. Beat N+1's first frame IS beat N's last frame, so no
cut can appear at the seam; the duplicate is dropped when the beats are concatenated.

Two things this does not fix, both documented and both real:

  * DRIFT. Every beat re-encodes the previous output, so identity/colour/framing error
    compounds. Nothing in v7 has tested a chain this long — REPORT_v7 section 3.4 only ever
    stretched a single generation to 2.9x its trained length.
  * THE REST POSE. REPORT_v7 section 3.5: beat chaining "needs a clean rest pose to hand
    off between beats", and `standing_idle` is derived from frozen `neutral` footage, so
    the model teaches "stop" as "freeze". That gap is on the critical path for long video
    and needs delivered idle footage to close.

Beat length is the longest bucket the label was actually trained at — longer beats mean
fewer joins mean less accumulated drift. Only `neutral` reaches f57; the longest
expressive buckets are `angry` and `laughing` at f37.

    python chain_noreset_v7.py --label laughing --seconds 10
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_v7 as g  # noqa: E402
from prep_v7 import FRONT  # noqa: E402

M = Path("/workspace/wan_models")
REPO = Path("/workspace/musubi-tuner")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
GOLD = Path("/workspace/wan_output/v2_golden")
CKPT = ("/workspace/weights/v7/expr-lownoise/"
        "pudgy-v7-expr-lownoise-step00002500.safetensors")
FPS = 24


def last_frame(mp4, dest):
    """Pull the final frame — the conditioning image for the next beat."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-1",
                    "-i", str(mp4), "-update", "1", "-q:v", "1", str(dest)], check=True)
    return dest


def generate(prompt, start, frames, seed, outdir, tag):
    """One beat. Single-prompt mode so the mp4 lands immediately (--from_file defers
    every decode to the end of the batch, which is useless when beat N+1 needs beat N)."""
    scratch = outdir / f".raw_{tag}"
    subprocess.run(["rm", "-rf", str(scratch)], check=True)
    scratch.mkdir(parents=True)
    cmd = [PY, str(REPO / "src/musubi_tuner/wan_generate_video.py"),
           "--task", "i2v-A14B",
           "--dit", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
           "--dit_high_noise", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
           "--timestep_boundary", "0.9",
           "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
           "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
           "--lora_weight", CKPT, "--lora_multiplier", "1.0",
           "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
           "--lora_multiplier_high_noise", "1.0",
           "--video_size", "1024", "1024", "--video_length", str(frames), "--fps", str(FPS),
           "--infer_steps", "25", "--flow_shift", "5.0", "--guidance_scale", "5.0",
           "--image_path", str(start),
           "--prompt", prompt, "--seed", str(seed), "--attn_mode", "sdpa",
           "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--save_path", str(scratch), "--output_type", "video"]
    # musubi re-fetches the T5 tokenizer from the Hub on every process start. One process
    # per beat means one API call per beat, which earns a 429 rate-limit partway through a
    # long chain. The tokenizer is already in HF_HOME, so serve it from cache and never ask.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if subprocess.run(cmd, cwd=str(REPO), env=env).returncode:
        sys.exit(f"generation failed on {tag}")
    mp4s = sorted(scratch.glob("*.mp4"))
    if not mp4s:
        sys.exit(f"no mp4 produced for {tag}")
    dest = outdir / f"{tag}.mp4"
    mp4s[-1].rename(dest)
    subprocess.run(["rm", "-rf", str(scratch)], check=True)
    return dest


def concat(beats, dest, total_frames, workdir):
    """Concatenate, dropping each beat's first frame (it duplicates the previous last)."""
    frames_dir = workdir / "frames"
    subprocess.run(["rm", "-rf", str(frames_dir)], check=True)
    frames_dir.mkdir(parents=True)
    n = 0
    for i, b in enumerate(beats):
        tmp = workdir / f".ex{i}"
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        tmp.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(b),
                        str(tmp / "f%04d.png")], check=True)
        fs = sorted(tmp.iterdir())
        if i:
            fs = fs[1:]                      # drop the duplicated seam frame
        for f in fs:
            if n >= total_frames:
                break
            n += 1
            f.rename(frames_dir / f"{n:05d}.png")
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", str(dest)], check=True)
    subprocess.run(["rm", "-rf", str(frames_dir)], check=True)
    return n


# Longest bucket each expression was actually trained at. Beats are cut to the label's
# own ceiling: longer beats mean fewer joins mean less accumulated drift. Only `neutral`
# reaches f57; `happy` and `confused` never got past f21.
MAX_BUCKET = {"neutral": 57, "angry": 37, "laughing": 37,
              "crying": 29, "surprised": 29, "happy": 21, "confused": 21}


def plan_beats(labels, total):
    """Split `total` frames across `labels` in order, each beat at that label's ceiling.

    Beat 1 contributes all its frames; every later beat contributes one fewer, because
    its first frame duplicates the previous beat's last and is dropped at concat.
    """
    share = total // len(labels)
    beats, acc = [], 0
    for li, lab in enumerate(labels):
        f = MAX_BUCKET[lab]
        target = total if li == len(labels) - 1 else share * (li + 1)
        while acc < target:
            beats.append((lab, f))
            acc += f if not beats[:-1] else f - 1
            if acc >= total:
                break
        if acc >= total:
            break
    return beats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="laughing",
                    help="single expression; ignored when --labels is given")
    ap.add_argument("--labels", nargs="+",
                    help="expression sequence, e.g. --labels angry happy crying. "
                         "Prompt changes at each block boundary but the chain NEVER "
                         "resets: the new expression is conditioned on the previous "
                         "beat's last frame, so the change is generated, not cut.")
    ap.add_argument("--char", default="Pax")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--beat-frames", type=int,
                    help="override; default is each label's own trained ceiling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--tag", default=None, help="output basename")
    ap.add_argument("--out", default="/workspace/eval_v7/chain_noreset")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    total = int(round(a.seconds * FPS))
    labels = a.labels or [a.label]
    for l in labels:
        if l not in MAX_BUCKET:
            sys.exit(f"unknown expression {l!r}; known: {sorted(MAX_BUCKET)}")
    if a.beat_frames:
        plan = []
        acc = 0
        while acc < total:
            plan.append((labels[min(len(labels) - 1, acc * len(labels) // total)],
                         a.beat_frames))
            acc += a.beat_frames if len(plan) == 1 else a.beat_frames - 1
    else:
        plan = plan_beats(labels, total)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start = Path(f"/workspace/eval_v7/keyframes/{a.char.lower()}_neutral_start.png")
    if not start.exists():
        sys.exit(f"missing start frame {start}")

    print(f"{a.char} / {' -> '.join(labels)} / {a.seconds}s = {total} frames")
    print(f"{len(plan)} beats, NO reset, seed {a.seed}")
    for i, (lab, f) in enumerate(plan, 1):
        print(f"  beat {i:2d}  {lab:<10} f{f}")
    if a.dry:
        return

    beats, meta = [], []
    cur = start
    for i, (lab, frames) in enumerate(plan, 1):
        tag = f"beat{i:02d}_{lab}_f{frames}"
        mp4 = out / f"{tag}.mp4"
        prompt = g.caption_for(a.char, "expression", lab, zoom=1.00, bg=a.bg,
                               clause=FRONT, nframes=frames)
        if mp4.exists():
            # Resume: a completed beat is deterministic given its start frame and seed,
            # so regenerating it would cost 15 min to reproduce the same bytes.
            print(f"== beat {i}/{len(plan)}  SKIP, already present", flush=True)
        else:
            print(f"== beat {i}/{len(plan)}  {lab} f{frames} from {Path(cur).name}",
                  flush=True)
            mp4 = generate(prompt, cur, frames, a.seed, out, tag)
        beats.append(mp4)
        meta.append({"beat": i, "label": lab, "frames": frames,
                     "start": Path(cur).name, "clip": mp4.name, "reset": False})
        nxt = out / f"{tag}_last.png"
        cur = nxt if nxt.exists() else last_frame(mp4, nxt)

    name = a.tag or f"{a.char.lower()}_{'-'.join(labels)}_{int(a.seconds)}s_noreset_s{a.seed}"
    final = out / f"{name}.mp4"
    n = concat(beats, final, total, out)
    (out / f"{name}_manifest.json").write_text(json.dumps(
        {"character": a.char, "labels": labels, "seed": a.seed, "reset": False,
         "beats": len(plan), "frames": n, "seconds": n / FPS, "chain": meta}, indent=2))
    print(f"\nwrote {final}  ({n} frames, {n/FPS:.2f}s)")


if __name__ == "__main__":
    main()
