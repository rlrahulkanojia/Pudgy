#!/usr/bin/env python3
"""Angle probe — expression promptability from a NON-FRONT start frame.

Every v7 gate that has run (G-X, G-C, G-P, G-D) drove I2V from a front-view frame:
prep_eval_keyframes_v7.py hardcodes FRONT. But front is only 11% of the training set —
1,508 of 2,272 clips are three-quarter and 508 are profile. So the headline "125/126
pairs distinct" is a front-view result, and nothing is known about the other 89%.

This probes that. For each (character, non-front view) it takes frame 0 of the real
neutral clip at that view as the conditioning frame, then prompts for an expression
with the matching angle clause. Angle therefore stays constant between the start frame
and the caption; only the expression label varies.

Shot size is pinned to 1.00x ("static close-up shot") — the unzoomed rung — so the
shot ladder is not a second moving part. Zoom is assigned per (clip, background) in
prep_v7, so the background is whichever one carries 1.00x for that cell; it is read
from the dataset jsonl rather than assumed, and the caption names it.

Captions come from gates_v7.caption_for (which imports prep_v7), so the prompts sit in
the training distribution by construction rather than being retyped here.

    python angle_probe_v7.py --sample 5            # reproducible random draw
    python angle_probe_v7.py                       # the full 48-cell matrix
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_v7 as g  # noqa: E402
from prep_v7 import clause_for_slug  # noqa: E402  — single source of caption truth

M = Path("/workspace/wan_models")
REPO = Path("/workspace/musubi-tuner")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
GOLD = Path("/workspace/wan_output/v2_golden")
DATA = Path("/workspace/data_v7")
CLIPS = DATA / "clips"
KF = Path("/workspace/eval_v7/keyframes/angle")

VIEWS = [f"QF{d}_{s}" for d in (1, 2, 3) for s in ("L", "R")] + ["SIDE_L", "SIDE_R"]
CHARS = ["pax", "polly"]
ZOOM = 1.00
ZOOM_TEXT = "static close-up shot"


def unzoomed_bg(char, view, nframes=21):
    """Which background carries the 1.00x rung for this cell. Read, never assumed."""
    jl = DATA / f"dataset_expression_f{nframes}.workspace.jsonl"
    pat = re.compile(rf"^{char}_neutral_{view}__(\w+)_f{nframes}\.mp4$")
    hits = []
    for line in open(jl):
        r = json.loads(line)
        m = pat.match(r["video_path"].split("/")[-1])
        if m and ZOOM_TEXT in r["caption"]:
            hits.append(m.group(1))
    if not hits:
        sys.exit(f"no {ZOOM}x neutral clip for {char}/{view}")
    return sorted(hits)[0]


def keyframe(char, view, bg, nframes=21):
    """Frame 0 of the real neutral clip = the training start frame, already at 1.00x."""
    src = CLIPS / f"{char}_neutral_{view}__{bg}_f{nframes}.mp4"
    if not src.exists():
        sys.exit(f"missing clip {src}")
    KF.mkdir(parents=True, exist_ok=True)
    dest = KF / f"{char}_neutral_{view}_z100_start.png"
    if not dest.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-vf", "select=eq(n\\,0)", "-vframes", "1", str(dest)], check=True)
    return dest


def build(cells, seed, nframes):
    lines, index = [], []
    for char, view, label in cells:
        bg = unzoomed_bg(char, view, nframes)
        start = keyframe(char, view, bg, nframes)
        cap = g.caption_for(char.capitalize(), "expression", label,
                            zoom=ZOOM, bg=bg, clause=clause_for_slug(view),
                            nframes=nframes)
        lines.append(f"{cap} --i {start} --f {nframes} --d {seed}")
        index.append({"character": char, "view": view, "label": label,
                      "bg": bg, "seed": seed, "start": str(start), "caption": cap})
    return lines, index


def generate(ckpt, prompt_file, outdir):
    """One model load for the whole matrix (sweep_v7.py's --from_file path, same flags)."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(REPO / "src/musubi_tuner/wan_generate_video.py"),
           "--task", "i2v-A14B",
           "--dit", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
           "--dit_high_noise", str(M / "comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
           "--timestep_boundary", "0.9",
           "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
           "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
           # expression golden on low-noise; partner stays the UNTOUCHED v2 golden
           "--lora_weight", str(ckpt), "--lora_multiplier", "1.0",
           "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
           "--lora_multiplier_high_noise", "1.0",
           "--video_size", "1024", "1024", "--fps", "24",
           "--infer_steps", "25", "--flow_shift", "5.0", "--guidance_scale", "5.0",
           "--attn_mode", "sdpa", "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--from_file", str(prompt_file),
           "--save_path", str(outdir), "--output_type", "video"]
    # musubi re-fetches the T5 tokenizer from the Hub on every process start, which earns
    # a 429 rate-limit once enough runs stack up on one IP. It is already in HF_HOME, so
    # serve it from cache and never ask.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if subprocess.run(cmd, cwd=str(REPO), env=env).returncode:
        sys.exit("batch generation failed")
    return sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/weights/v7/expr-lownoise/"
                                      "pudgy-v7-expr-lownoise-step00002500.safetensors")
    ap.add_argument("--labels", nargs="+", default=["happy", "angry", "surprised"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--sample", type=int, help="draw N random cells instead of all")
    ap.add_argument("--sample-seed", type=int, default=7, help="makes the draw reproducible")
    ap.add_argument("--out", default="/workspace/eval_v7/angle_probe")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    cells = [(c, v, l) for c in CHARS for v in VIEWS for l in a.labels]
    if a.sample:
        cells = sorted(random.Random(a.sample_seed).sample(cells, a.sample))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, index = build(cells, a.seed, a.frames)
    pf = out / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")

    print(f"{len(cells)} generations, 1 model load  (sample-seed {a.sample_seed})\n")
    for i, m in enumerate(index, 1):
        print(f"  {i}. {m['character']:<5} {m['view']:<7} {m['label']:<10} bg={m['bg']:<6} "
              f"{Path(m['start']).name}")
    if a.dry:
        print("\n--dry: nothing generated")
        return

    clips = generate(Path(a.ckpt), pf, out)
    named = []
    for clip, m in zip(clips, index):
        dest = clip.with_name(f"{m['character']}_{m['label']}_{m['view']}_z100_s{m['seed']}.mp4")
        if dest != clip:
            clip.rename(dest)
        m["clip"] = dest.name
        named.append(dest)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(named)} clips -> {out}")
    for p in named:
        print("  ", p.name)


if __name__ == "__main__":
    main()
