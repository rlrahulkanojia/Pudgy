#!/usr/bin/env python3
"""Batched G-C / G-M sweep across v7 checkpoints — ONE model load per checkpoint.

Why this exists: gates_v7.generate() shells out to eval_v7.sh per clip, and each call
pays a full two-expert load. Measured on this box: 226 s load + 381 s denoise. Run that
way a 3-seed G-C is 42 loads per checkpoint — the load alone is 2.6 h of pure overhead,
and a 17-checkpoint sweep is ~76 h.

musubi's wan_generate_video.py --from_file takes a prompt file, one generation per line
(`<prompt> --i <image> --f <frames> --d <seed>`), so the whole matrix runs inside a
single load. Same generation path, same flags as eval_v7.sh — only the batching differs.

    python sweep_v7.py --ckpt <lora> --seeds 42 --labels happy angry neutral
    python sweep_v7.py --sweep 500 1000 1750 2500 3500 --seeds 42
"""
import argparse, itertools, json, os, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_v7 as g

M = Path("/workspace/wan_models")
REPO = Path("/workspace/musubi-tuner")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
GOLD = Path("/workspace/wan_output/v2_golden")
RUN = Path("/workspace/wan_output/pudgy-v7-expr-lownoise")


def build_prompt_file(labels, seeds, kind, dest):
    """One line per generation. Captions come from gates_v7 (which imports prep_v7),
    so eval prompts sit in the training distribution by construction."""
    lines, index = [], []
    n = g.COMMON[kind]
    for char in ("Pax", "Polly"):
        start = g.start_frame(char, kind)
        for lab in labels:
            for seed in seeds:
                cap = g.caption_for(char, kind, lab, nframes=n)
                lines.append(f"{cap} --i {start} --f {n} --d {seed}")
                index.append({"character": char, "label": lab, "seed": seed})
    dest.write_text("\n".join(lines) + "\n")
    return index


def generate_batch(ckpt, prompt_file, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(REPO / "src/musubi_tuner/wan_generate_video.py"),
           "--task", "i2v-A14B",
           "--dit", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
           "--dit_high_noise", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
           "--timestep_boundary", "0.9",
           "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
           "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
           # expression checkpoint on low-noise; high-noise stays the UNTOUCHED v2 golden
           # so any regression remains bisectable (plan section 4).
           "--lora_weight", str(ckpt), "--lora_multiplier", "1.0",
           "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
           "--lora_multiplier_high_noise", "1.0",
           "--video_size", "1024", "1024", "--fps", "24",
           "--infer_steps", "25", "--flow_shift", "5.0", "--guidance_scale", "5.0",
           "--attn_mode", "sdpa", "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--from_file", str(prompt_file),
           "--save_path", str(outdir), "--output_type", "video"]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode:
        sys.exit(f"batch generation failed for {ckpt}")
    # musubi names outputs by timestamp; sort by mtime to recover prompt-file order.
    return sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)


def rename_clips(clips, index):
    """Give each clip its (character, label, seed) name.

    musubi names outputs `<timestamp>_<seed>_.mp4`, which is unreviewable: you cannot
    tell which expression a clip is without replaying the prompt order. The gate maths
    works off mtime order either way, but a human reading the results cannot.
    """
    out = []
    for clip, meta in zip(clips, index):
        dest = clip.with_name(f"{meta['character'].lower()}_{meta['label']}_s{meta['seed']}.mp4")
        if dest != clip:
            clip.rename(dest)
        out.append(dest)
    return out


def score_gc(clips, index, labels, seeds):
    by = {(i["character"], i["label"], i["seed"]): c for i, c in zip(index, clips)}
    fb = g.face_box(1.00)
    rows = []
    for char in ("Pax", "Polly"):
        for seed in seeds:
            for a, b in itertools.combinations(labels, 2):
                ka, kb = (char, a, seed), (char, b, seed)
                if ka not in by or kb not in by:
                    continue
                va, vb = g.read_video(by[ka]), g.read_video(by[kb])
                face = g.ssim_pair(va, vb, region=fb)
                rows.append({"character": char, "seed": seed, "pair": f"{a}|{b}",
                             "ssim_face": round(face, 4),
                             "ssim_whole": round(g.ssim_pair(va, vb), 4),
                             "distinct": face < g.FACE_DISTINCT})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--sweep", nargs="*", type=int)
    ap.add_argument("--kind", default="expression")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--outroot", default="/workspace/eval_v7/sweep")
    a = ap.parse_args()

    labels = a.labels or list(g.table(a.kind))
    ckpts = []
    if a.ckpt:
        ckpts = [Path(a.ckpt)]
    for step in (a.sweep or []):
        p = RUN / f"pudgy-v7-expr-lownoise-step{step:08d}.safetensors"
        if not p.exists():
            sys.exit(f"missing checkpoint {p}")
        ckpts.append(p)

    summary = []
    for ck in ckpts:
        tag = ck.stem.replace("pudgy-v7-expr-lownoise-", "")
        out = Path(a.outroot) / tag
        pf = out / "prompts.txt"
        out.mkdir(parents=True, exist_ok=True)
        index = build_prompt_file(labels, a.seeds, a.kind, pf)
        print(f"== {tag}: {len(index)} generations, 1 model load", flush=True)
        clips = generate_batch(ck, pf, out)
        if len(clips) == len(index):
            clips = rename_clips(clips, index)
        if len(clips) != len(index):
            print(f"   !! {len(clips)} clips for {len(index)} prompts — scoring what exists")
        rows = score_gc(clips, index, labels, a.seeds)
        worst = max(rows, key=lambda r: r["ssim_face"]) if rows else None
        res = {"checkpoint": tag, "n_clips": len(clips), "rows": rows, "worst_pair": worst,
               "mean_ssim_face": round(float(np.mean([r["ssim_face"] for r in rows])), 4) if rows else None,
               "pass": bool(rows) and all(r["distinct"] for r in rows)}
        (out / "gc_result.json").write_text(json.dumps(res, indent=2))
        summary.append(res)
        print(f"   mean_face_ssim={res['mean_ssim_face']} worst={worst['ssim_face'] if worst else '-'} pass={res['pass']}", flush=True)

    Path(a.outroot, "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n== sweep summary ==")
    for s in summary:
        print(f"  {s['checkpoint']:>16s}  mean_face_ssim={s['mean_ssim_face']}  pass={s['pass']}")


if __name__ == "__main__":
    main()
