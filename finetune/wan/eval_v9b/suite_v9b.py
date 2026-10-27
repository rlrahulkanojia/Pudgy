#!/usr/bin/env python3
"""Tier 2 for v9b: generate from held-out skit spans and score (Training_Approach_v9.md
section 5.4, "looks right"; jobs from build_eval_v9b.py).

Per job (a held-out span's first frame + its true caption, 49 frames at 480x848) and seed
(42, 43, 44), for each configuration:
  dup       frames with two bodies of one lead's colour (longform/dupcheck.py's detector)
  drift     mean |last frames - start frame| on the border strip (qa_take: camera / zoom creep)
  frozen    share of frame pairs with ~no change (qa_take's FROZEN at > 0.35)
  jump      largest single-frame change / median change (qa_take's POP at > 8)
  motion    mean per-frame change of the generated clip / the real span's (bar 0.7-1.3)
Contact sheets per job: the real span on top, then one row per configuration (seed 42).
The sheets are the verdict on "leads on-model for the whole span"; numbers flag what to
look at.

Generation is gen_v8's two-expert call (same steps, shift, guidance, boundary), at the
scene size instead of 1024^2. gen_v8.run hard-codes 1024^2 and is shared with v9c, so the
call is rebuilt here rather than changed there.

    python suite_v9b.py --configs v8merged v9b@400
    python suite_v9b.py --configs v8merged --seeds 42 --limit 2      # quick look
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9b as C  # noqa: E402
import gen_v8 as G  # noqa: E402

sys.path.insert(0, str(C.WAN / "longform"))
import dupcheck as DC  # noqa: E402

SEEDS = (42, 43, 44)


def generate(config, jobs, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    want = {j["tag"]: outdir / f"{j['tag']}.mp4" for j in jobs}
    pending = [j for j in jobs if not want[j["tag"]].exists()]
    print(f"   [{config}] {len(jobs)} clips, {len(pending)} to generate -> {outdir}", flush=True)
    if not pending:
        return want
    low, high = C.lora_config(config)
    for p, _ in low + high:
        if not Path(p).exists():
            sys.exit(f"missing LoRA for {config}: {p}")
    raw = outdir / f".raw_{config.replace('/', '_').replace(':', '_').replace('@', '_')}"
    raw.mkdir(exist_ok=True)
    for f in raw.glob("*.mp4"):
        f.unlink()
    pf = raw / "prompts.txt"
    pf.write_text("".join(f"{j['prompt']} --i {j['start']} --f {j['frames']} --d {j['seed']}\n" for j in pending))
    cmd = [str(G.PY), "src/musubi_tuner/wan_generate_video.py",
           "--task", "i2v-A14B", "--dit", str(G.DIT_LOW), "--dit_high_noise", str(G.DIT_HIGH),
           "--timestep_boundary", str(G.BOUNDARY), "--vae", str(G.VAE), "--t5", str(G.T5),
           "--lora_weight", *[str(p) for p, _ in low], "--lora_multiplier", *[str(m) for _, m in low],
           "--lora_weight_high_noise", *[str(p) for p, _ in high],
           "--lora_multiplier_high_noise", *[str(m) for _, m in high],
           "--video_size", str(C.SH), str(C.SW), "--fps", "24",          # musubi: height width
           "--infer_steps", str(G.STEPS), "--flow_shift", str(G.SHIFT),
           "--guidance_scale", str(G.GUIDE), "--attn_mode", "sdpa",
           "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--from_file", str(pf), "--save_path", str(raw), "--output_type", "video"]
    cmd += os.environ.get("GEN_EXTRA", "").split()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    (raw / "cmd.json").write_text(json.dumps(cmd, indent=1))
    if subprocess.run(cmd, cwd=str(G.MUSUBI), env=env).returncode:
        sys.exit(f"generation failed for {config} (see {raw})")
    produced = sorted(raw.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if len(produced) != len(pending):
        sys.exit(f"{config}: {len(produced)} clips for {len(pending)} prompts - refusing to guess the mapping")
    for src, j in zip(produced, pending):
        src.rename(want[j["tag"]])
    return want


def frames(p, w=180, h=320):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(p), "-vf", f"scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def score(clip, start_png, real):
    v = frames(clip).astype(np.float32)
    d = np.abs(np.diff(v, axis=0)).mean((1, 2, 3))
    ref = np.asarray(Image.open(start_png).convert("RGB").resize((180, 320)), np.float32)
    border = np.zeros(v.shape[1:3], bool); border[:24] = border[-24:] = True; border[:, :14] = border[:, -14:] = True
    rv = frames(real).astype(np.float32)
    rd = np.abs(np.diff(rv, axis=0)).mean((1, 2, 3))
    big = DC.frames(clip)
    dup = sum(DC.bodies(f, *DC.PAX) >= 2 or DC.bodies(f, *DC.POLLY) >= 2 for f in big)
    return {"frames": len(v), "dup_frames": int(dup),
            "drift": round(float(np.abs(v[-4:].mean(0) - ref)[border].mean()), 2),
            "frozen": round(float((d < 0.15).mean()), 3),
            "jump": round(float(d.max() / max(np.median(d), 1e-6)), 2),
            "motion": round(float(d.mean() / max(rd.mean(), 1e-6)), 3)}


def sheet(job, clips, dest, n=7):
    rows = [("real", job["real"])] + clips
    tw, th = 120, 213
    img = Image.new("RGB", (tw * n + 90, th * len(rows)), "white")
    dr = ImageDraw.Draw(img)
    for r, (lab, p) in enumerate(rows):
        v = frames(p, tw, th)
        idx = np.linspace(0, len(v) - 1, n).round().astype(int)
        dr.text((2, r * th + 4), lab[:14], fill="black")
        for i, k in enumerate(idx):
            img.paste(Image.fromarray(v[k]), (90 + i * tw, r * th))
    img.save(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    base_jobs = json.loads((C.SCENES / "suite_jobs.json").read_text())
    if a.limit:
        base_jobs = base_jobs[:a.limit]
    jobs = [dict(j, tag=f"{j['tag']}_s{s}", seed=s, base=j["tag"]) for j in base_jobs for s in a.seeds]
    root = C.RESULTS / "suite"
    scores = json.loads((root / "scores.json").read_text()) if (root / "scores.json").exists() else {}
    for cfg in a.configs:
        got = generate(cfg, jobs, root / cfg.replace("@", "_"))
        for j in jobs:
            key = f"{cfg}|{j['tag']}"
            if key not in scores:
                scores[key] = score(got[j["tag"]], j["start"], j["real"]) | {"config": cfg, "job": j["base"],
                                                                             "seed": j["seed"], "two_shot": j["two_shot"]}
        (root / "scores.json").write_text(json.dumps(scores, indent=1))
    for j in base_jobs:
        clips = [(cfg, root / cfg.replace("@", "_") / f"{j['tag']}_s{a.seeds[0]}.mp4") for cfg in a.configs]
        sheet(j, clips, root / f"{j['tag']}.sheet.png")
    print(f"{'config':<12} {'clips':>5} {'dup clips':>9} {'dup frames':>10} {'drift':>6} {'frozen>0.35':>11} {'pop>8':>6} {'motion (median)':>15}")
    for cfg in a.configs:
        rs = [v for v in scores.values() if v["config"] == cfg and v["seed"] in a.seeds]
        if not rs:
            continue
        print(f"{cfg:<12} {len(rs):>5} {sum(r['dup_frames'] > 0 for r in rs):>9} {sum(r['dup_frames'] for r in rs):>10} "
              f"{np.mean([r['drift'] for r in rs]):>6.2f} {sum(r['frozen'] > 0.35 for r in rs):>11} "
              f"{sum(r['jump'] > 8 for r in rs):>6} {np.median([r['motion'] for r in rs]):>15.2f}")
    print(f"sheets: {root}/*.sheet.png")


if __name__ == "__main__":
    main()
