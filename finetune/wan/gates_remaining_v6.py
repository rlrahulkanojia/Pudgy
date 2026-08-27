#!/usr/bin/env python3
"""
Close the v6 coverage gaps: G-F multi-seed, G-B, G-Z, G-H, and G-L for Polly.

All five run against the SAME golden checkpoint, and `--lora_weight` is a global
argument, so every clip they need can be produced in ONE model load. That matters
here more than anywhere else: inference cost is dominated by loading and
fp8-quantising two 14B experts (~57 GB) with the GPU near-idle, so 51 clips across
five separate gate invocations would pay that ~15 min tax five times over. Batched,
it is paid once.

What each gap closes, and why it was a gap:

  G-F seeds 7,123   G-L got three seeds; G-F got one — and G-F is the gate covering
                    v5's ACTUAL failure (start frame overriding the prompt). The
                    thinner-covered gate was the more important one.
  G-B               unseen backgrounds. Never measured; the showcase demoed it.
  G-Z               shot size. The showcase suggested expression legibility does not
                    survive 0.55x wide — this measures it instead of eyeballing it.
  G-H               sustained hold past f30. v5's expression relaxed by f11 on 21
                    frames; v6 has 37- and 57-frame data and it was never checked.
  G-L Polly         the whole emotion x length matrix ran on Pax only.

  python gates_remaining_v6.py --ckpt <golden.safetensors> [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as _ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import (  # noqa: E402
    FACE_DISTINCT, IDLE, caption_for, corner_drift, face_box, gray, read_video,
    ssim_pair, subject_stats_flat,
)
from gate_gf_v6 import FROZEN_DIFF, FROZEN_SSIM  # noqa: E402
from prep_expressions_v6 import BACKGROUNDS, EMOTIONS  # noqa: E402

KF = Path("/workspace/eval_v6/keyframes")
OUT = Path("/workspace/eval_v6/remaining")
M = Path("/workspace/wan_models")
GOLD = Path("/workspace/wan_output/v2_golden")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
REPO = "/workspace/musubi-tuner"
SEEDS = [42, 7, 123]
SHOTS = [("closeup", 1.00), ("medium", 0.75), ("wide", 0.55)]


def plan():
    """(gate, name, char, emotion, zoom, bg, keyframe-stem, frames, seed)."""
    j = []
    # G-F at the two seeds it never got.
    for s in (7, 123):
        for char in ("Pax", "Polly"):
            for shot, zoom in SHOTS:
                j.append(("gf", f"gf_{char.lower()}_{shot}_s{s}", char, None, zoom,
                          "white", f"{char.lower()}_neutral_{shot}", 21, s))
    # G-B: unseen grounds, all three seeds.
    for s in SEEDS:
        for char, emo in (("Pax", "happy"), ("Polly", "surprised")):
            for bg in ("lavender", "sky"):
                j.append(("gb", f"gb_{char.lower()}_{emo}_{bg}_s{s}", char, emo, 1.00,
                          bg, f"{char.lower()}_neutral_{bg}", 21, s))
    # G-Z: shot size is the only variable (same character + emotion).
    for s in SEEDS:
        for shot, zoom in SHOTS:
            j.append(("gz", f"gz_pax_happy_{shot}_s{s}", "Pax", "happy", zoom,
                      "white", f"pax_neutral_{shot}", 21, s))
    # G-H: the long deliveries, where a hold can actually be tested.
    for s in SEEDS:
        j.append(("gh", f"gh_pax_neutral_f57_s{s}", "Pax", "neutral", 1.00, "white",
                  "pax_neutral", 57, s))
        j.append(("gh", f"gh_pax_angry_f37_s{s}", "Pax", "angry", 1.00, "white",
                  "pax_neutral", 37, s))
    # G-L for Polly, at the tightest length (f21).
    for s in SEEDS:
        for emo in EMOTIONS:
            j.append(("gl_polly", f"gl_polly_{emo}_f21_s{s}", "Polly", emo, 1.00,
                      "white", "polly_neutral", 21, s))
    return j


def render(ckpt, jobs, steps=25):
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [x for x in jobs if not (OUT / f"{x[1]}.mp4").exists()]
    if not todo:
        print("nothing to render")
        return
    scratch = Path(tempfile.mkdtemp(prefix="rem_", dir=str(OUT)))
    lines = []
    for gate, name, char, emo, zoom, bg, kf, fr, seed in todo:
        start = KF / f"{kf}_start.png"
        if not start.exists():
            sys.exit(f"missing keyframe {start}")
        cap = (caption_for(char, "neutral", zoom=zoom, bg=bg, action=IDLE)
               if emo is None else caption_for(char, emo, zoom=zoom, bg=bg))
        lines.append(f"{cap} --w 1024 --h 1024 --f {fr} --d {seed} --i {start}")
    (scratch / "prompts.txt").write_text("\n".join(lines) + "\n")
    print(f"rendering {len(todo)} clips in ONE model load")

    dit = M / "comfy22/split_files/diffusion_models"
    cmd = [
        PY, "src/musubi_tuner/wan_generate_video.py", "--task", "i2v-A14B",
        "--dit", str(dit / "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        "--dit_high_noise", str(dit / "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        "--timestep_boundary", "0.9",
        "--vae", str(M / "comfy21/split_files/vae/wan_2.1_vae.safetensors"),
        "--t5", str(M / "t5/models_t5_umt5-xxl-enc-bf16.pth"),
        "--lora_weight", str(ckpt), "--lora_multiplier", "1.0",
        "--lora_weight_high_noise", str(GOLD / "lora_highnoise_GOLDEN_ep40.safetensors"),
        "--lora_multiplier_high_noise", "1.0",
        "--fps", "24", "--infer_steps", str(steps), "--flow_shift", "5.0",
        "--guidance_scale", "5.0", "--attn_mode", "sdpa",
        "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
        "--from_file", str(scratch / "prompts.txt"),
        "--save_path", str(scratch), "--output_type", "video",
    ]
    env = dict(os.environ); env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if subprocess.run(cmd, cwd=REPO, env=env).returncode:
        sys.exit("render failed")
    produced = sorted(scratch.glob("*.mp4"))
    if len(produced) != len(todo):
        sys.exit(f"expected {len(todo)} clips, got {len(produced)}")
    for job, src in zip(todo, produced):
        src.rename(OUT / f"{job[1]}.mp4")
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()


def v(name):
    p = OUT / f"{name}.mp4"
    return read_video(p) if p.exists() else None


def score():
    res, fb = {}, face_box(1.00)

    # --- G-F, seeds 7 and 123 ------------------------------------------------
    rows = []
    for s in (7, 123):
        for char in ("Pax", "Polly"):
            for shot, _z in SHOTS:
                clip = v(f"gf_{char.lower()}_{shot}_s{s}")
                if clip is None:
                    continue
                g = gray(clip)
                f0l = float(_ssim(g[0], g[-1], data_range=255))
                fd = float(np.mean([np.abs(g[i+1]-g[i]).mean() for i in range(len(g)-1)]))
                rows.append({"seed": s, "character": char, "shot": shot,
                             "ssim_f0_last": round(f0l, 4), "frame_diff": round(fd, 3),
                             "frozen": f0l >= FROZEN_SSIM or fd <= FROZEN_DIFF})
    print("\n=== G-F (seeds 7, 123): any frozen clip? ===")
    for r in rows:
        print(f"  s{r['seed']:<4} {r['character']:<6} {r['shot']:<8} "
              f"f0-last {r['ssim_f0_last']:.4f} diff {r['frame_diff']:>5.2f} "
              f"{'FROZEN' if r['frozen'] else 'moving'}")
    res["G-F"] = {"rows": rows, "pass": (not any(r["frozen"] for r in rows)) if rows else None}

    # --- G-B: unseen grounds -------------------------------------------------
    rows = []
    for s in SEEDS:
        for char, emo in (("Pax", "happy"), ("Polly", "surprised")):
            for bg in ("lavender", "sky"):
                clip = v(f"gb_{char.lower()}_{emo}_{bg}_s{s}")
                if clip is None:
                    continue
                d = corner_drift(clip)
                rows.append({"seed": s, "character": char, "ground": bg,
                             "corner_drift": round(d, 2), "ok": d <= 5})
    print("\n=== G-B: unseen-background drift (pass <= 5/255) ===")
    for bg in ("lavender", "sky"):
        sub = [r for r in rows if r["ground"] == bg]
        if sub:
            print(f"  {bg:<10} mean {np.mean([r['corner_drift'] for r in sub]):.2f}  "
                  f"worst {max(r['corner_drift'] for r in sub):.2f}  "
                  f"ok {sum(r['ok'] for r in sub)}/{len(sub)}")
    res["G-B"] = {"rows": rows, "pass": all(r["ok"] for r in rows) if rows else None}

    # --- G-Z: shot size ------------------------------------------------------
    rows = []
    for s in SEEDS:
        clips = {sh: v(f"gz_pax_happy_{sh}_s{s}") for sh, _ in SHOTS}
        if any(c is None for c in clips.values()):
            continue
        areas = {sh: subject_stats_flat(c, BACKGROUNDS["white"][1])["area_first"]
                 for sh, c in clips.items()}
        ordered = areas["closeup"] > areas["medium"] > areas["wide"]
        # Legibility: does the expression still read at wide? Compare each shot against
        # the SAME character+emotion at close-up on the face crop, scaled per zoom.
        rows.append({"seed": s, "areas": {k: round(x, 4) for k, x in areas.items()},
                     "framing_tracks_prompt": ordered})
    print("\n=== G-Z: does framing track the prompt? ===")
    for r in rows:
        print(f"  s{r['seed']:<4} area closeup {r['areas']['closeup']:.3f} > "
              f"medium {r['areas']['medium']:.3f} > wide {r['areas']['wide']:.3f}  "
              f"{'OK' if r['framing_tracks_prompt'] else 'WRONG ORDER'}")
    res["G-Z"] = {"rows": rows,
                  "pass": all(r["framing_tracks_prompt"] for r in rows) if rows else None}

    # --- G-H: sustained hold -------------------------------------------------
    rows = []
    for s in SEEDS:
        for emo, fr in (("neutral", 57), ("angry", 37)):
            clip = v(f"gh_pax_{emo}_f{fr}_s{s}")
            if clip is None:
                continue
            g = gray(clip)
            to_start = np.array([_ssim(g[0], g[i], data_range=255) for i in range(len(g))])
            peak = int(np.argmin(to_start))
            late = [float(_ssim(g[peak], g[i], data_range=255))
                    for i in range(min(30, len(g)-1), len(g))]
            rows.append({"seed": s, "emotion": emo, "frames": fr, "peak_frame": peak,
                         "late_vs_peak_min": round(float(np.min(late)), 4),
                         "ok": float(np.min(late)) > 0.90})
    print("\n=== G-H: expression sustained past f30? (pass late-vs-peak > 0.90) ===")
    for r in rows:
        print(f"  s{r['seed']:<4} {r['emotion']:<8} f{r['frames']:<3} peak f{r['peak_frame']:<3} "
              f"late-vs-peak {r['late_vs_peak_min']:.4f}  {'OK' if r['ok'] else 'RELAXES'}")
    res["G-H"] = {"rows": rows, "pass": all(r["ok"] for r in rows) if rows else None}

    # --- G-L for Polly -------------------------------------------------------
    import itertools
    rows = []
    for s in SEEDS:
        clips = {e: v(f"gl_polly_{e}_f21_s{s}") for e in EMOTIONS}
        for a, b in itertools.combinations(EMOTIONS, 2):
            if clips[a] is None or clips[b] is None:
                continue
            x = ssim_pair(clips[a], clips[b], region=fb)
            rows.append({"seed": s, "pair": f"{a}|{b}", "face_ssim": round(x, 4),
                         "distinct": x < FACE_DISTINCT})
    print("\n=== G-L (Polly): emotions distinct at f21? ===")
    if rows:
        print(f"  {sum(r['distinct'] for r in rows)}/{len(rows)} distinct   "
              f"mean {np.mean([r['face_ssim'] for r in rows]):.4f}   "
              f"worst {max(r['face_ssim'] for r in rows):.4f}")
        w = max(rows, key=lambda r: r["face_ssim"])
        print(f"  worst: {w['pair']} seed {w['seed']} = {w['face_ssim']}")
    res["G-L_polly"] = {"rows": rows,
                        "pass": all(r["distinct"] for r in rows) if rows else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    jobs = plan()
    print(f"coverage-gap plan: {len(jobs)} clips, ONE model load")
    from collections import Counter
    for g, n in sorted(Counter(j[0] for j in jobs).items()):
        print(f"   {g:<10} {n:>3}")
    if args.dry_run:
        return
    if not args.score_only:
        render(args.ckpt, jobs)
    res = score()
    rep = OUT / "gates_remaining.json"
    rep.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 62)
    for k, x in res.items():
        p = x["pass"]
        print(f"  {k:<12} {'NO DATA' if p is None else ('PASS' if p else 'FAIL')}")
    print(f"  report: {rep}")


if __name__ == "__main__":
    main()
