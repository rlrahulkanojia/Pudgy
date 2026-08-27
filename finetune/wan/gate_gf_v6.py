#!/usr/bin/env python3
"""
G-F — does the TRAINING start frame trigger an expression regardless of the prompt?

This is the failure that killed v5's high-noise run. All 28 v5 clips began from roughly
one design-sheet pose and always ended happy, so that frame became a near-deterministic
trigger: driven from it, the model fired the happy arc no matter what was asked, and
motion collapsed to 0.8 px (versus 221-267 px from a novel frame). v5 4.4's conclusion
was that it had learned "this frame -> happy" rather than the expression itself.

v6 has two structural mitigations, and G-F is the joint test of both:
  * contrastive labelling  — the same frame maps to four different labelled outcomes
  * the shot-size ladder   — 68 sources rendered at 1.00x/0.75x/0.55x, so there is no
                             single canonical start pixel-pattern to memorise

Method. Drive from the TRAINING start frame at all three shot sizes, with a prompt that
asks for NO expression and some head motion. Two independent readings:

  motion     is the clip FROZEN? A collapsed clip barely changes from its start frame.
             Note the plan's "x-range > 100 px" is NOT used as the criterion — see the
             FROZEN_* constants below for why it is wrong for an in-place turn.
  no-trigger compare this idle output against the four EMOTION outputs generated from the
             same frame (the G-L f21 set). If the idle clip closely matches one of them,
             the frame fired that expression despite the prompt asking for none.

The second reading is the one that matters and it is why this gate needs G-L's clips as
a reference set — a motion number alone cannot see an expression firing.

  python gate_gf_v6.py --ckpt <lora.safetensors> [--seeds 42]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_v6 import (  # noqa: E402
    FACE_DISTINCT, IDLE, caption_for, face_box, gray, read_video, ssim_pair,
    subject_stats_flat,
)
from skimage.metrics import structural_similarity as _ssim

# "Did the subject move?" needs a metric matched to the motion being ASKED FOR.
#
# The plan specifies x-range > 100 px, calibrated from v5, where the novel-frame clip had
# the character WALKING TOWARD CAMERA (221-267 px). Our idle prompt asks it to stand still
# and TURN ITS HEAD. Rotation in place changes almost every pixel while moving the centroid
# barely at all, so x-range reads ~20-70 px on clips that are visibly performing the
# requested action perfectly. Measured here: Pax close-up reads 31 px yet clearly turns to
# look left across the clip.
#
# So x-range is kept as informational, and the pass/fail criterion is "not frozen":
#   FROZEN_SSIM  f0-vs-last SSIM at/above this means the clip barely changed
#   FROZEN_DIFF  mean adjacent-frame absolute difference at/below this means static
# A genuinely collapsed clip (v5's failure) sits at ~1.00 / ~0.0. Observed here:
# 0.88-0.96 and 0.99-2.21.
FROZEN_SSIM = 0.985
FROZEN_DIFF = 0.30
from prep_expressions_v6 import BACKGROUNDS, EMOTIONS  # noqa: E402

KF = Path("/workspace/eval_v6/keyframes")
OUT = Path("/workspace/eval_v6/gf")
GL = Path("/workspace/eval_v6/gl")
M = Path("/workspace/wan_models")
GOLD = Path("/workspace/wan_output/v2_golden")
PY = "/workspace/Pudgy/.venv-wan/bin/python"
REPO = "/workspace/musubi-tuner"
SHOTS = [("closeup", 1.00), ("medium", 0.75), ("wide", 0.55)]


def render(ckpt, seeds, outdir, steps=25, frames=21):
    outdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for char in ("Pax", "Polly"):
        for shot, zoom in SHOTS:
            for s in seeds:
                if not (outdir / f"gf_{char.lower()}_{shot}_s{s}.mp4").exists():
                    todo.append((char, shot, zoom, s))
    if not todo:
        print("  nothing to render")
        return
    scratch = Path(tempfile.mkdtemp(prefix="gf_", dir=str(outdir)))
    lines = []
    for char, shot, zoom, s in todo:
        start = KF / f"{char.lower()}_neutral_{shot}_start.png"
        # action=IDLE asks for motion and explicitly NO expression.
        lines.append(f"{caption_for(char, 'neutral', zoom=zoom, action=IDLE)} "
                     f"--w 1024 --h 1024 --f {frames} --d {s} --i {start}")
    pf = scratch / "prompts.txt"
    pf.write_text("\n".join(lines) + "\n")
    print(f"  {len(todo)} clips in ONE model load")

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
        "--from_file", str(pf), "--save_path", str(scratch), "--output_type", "video",
    ]
    env = dict(os.environ); env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if subprocess.run(cmd, cwd=REPO, env=env).returncode:
        sys.exit("G-F render failed")
    produced = sorted(scratch.glob("*.mp4"))
    if len(produced) != len(todo):
        sys.exit(f"expected {len(todo)}, got {len(produced)}")
    for (char, shot, _z, s), src in zip(todo, produced):
        src.rename(outdir / f"gf_{char.lower()}_{shot}_s{s}.mp4")
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()


def score(seeds, outdir, gl_dir):
    rows = []
    print("\n=== G-F motion: does the subject move when asked? ===")
    for char in ("Pax", "Polly"):
        for shot, zoom in SHOTS:
            for s in seeds:
                p = outdir / f"gf_{char.lower()}_{shot}_s{s}.mp4"
                if not p.exists():
                    continue
                v = read_video(p)
                st = subject_stats_flat(v, BACKGROUNDS["white"][1])
                g = gray(v)
                f0_last = float(_ssim(g[0], g[-1], data_range=255))
                fdiff = float(np.mean([np.abs(g[i + 1] - g[i]).mean()
                                       for i in range(len(g) - 1)]))
                frozen = f0_last >= FROZEN_SSIM or fdiff <= FROZEN_DIFF
                rows.append({"character": char, "shot": shot, "seed": s,
                             "x_range_px": round(st["x_range_px"], 1),
                             "ssim_f0_last": round(f0_last, 4),
                             "frame_diff": round(fdiff, 3),
                             "frozen": frozen,
                             "area_first": round(st["area_first"], 4),
                             "area_last": round(st["area_last"], 4)})
                print(f"  {char:<6} {shot:<8} s{s}  f0-last {f0_last:.4f}  "
                      f"diff {fdiff:>5.2f}  {'FROZEN' if frozen else 'moving'}"
                      f"   (x-range {st['x_range_px']:>6.1f} px, informational)")

    # The reading that actually matters: is the idle clip a disguised emotion?
    trig = []
    print("\n=== G-F trigger: does the idle prompt secretly fire an expression? ===")
    print("  idle output vs each EMOTION output from the SAME frame (face SSIM)")
    for s in seeds:
        idle_p = outdir / f"gf_pax_closeup_s{s}.mp4"
        if not idle_p.exists():
            continue
        idle = read_video(idle_p)
        fb = face_box(1.00)
        best = None
        for e in EMOTIONS:
            ep = gl_dir / f"gl_pax_{e}_f21_s{s}.mp4"
            if not ep.exists():
                continue
            v = ssim_pair(idle, read_video(ep), region=fb)
            trig.append({"seed": s, "emotion": e, "face_ssim": round(v, 4),
                         "matches": v >= FACE_DISTINCT})
            print(f"    seed {s}  idle vs {e:<10} {v:.4f}"
                  f"{'   <-- MATCHES (trigger fired)' if v >= FACE_DISTINCT else ''}")
            if best is None or v > best[1]:
                best = (e, v)
        if best:
            print(f"    closest emotion: {best[0]} at {best[1]:.4f} "
                  f"(>= {FACE_DISTINCT} would mean the frame overrode the prompt)")

    motion_ok = (not any(r["frozen"] for r in rows)) if rows else None
    trig_ok = (not any(t["matches"] for t in trig)) if trig else None
    return {"gate": "G-F", "seeds": seeds, "motion": rows, "trigger": trig,
            "motion_pass": motion_ok, "no_trigger_pass": trig_ok,
            "pass": (motion_ok and trig_ok) if (motion_ok is not None
                                                and trig_ok is not None) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    outdir = OUT / args.ckpt.stem
    gl_dir = GL / args.ckpt.stem
    print(f"G-F :: {args.ckpt.name} :: seeds {seeds}")
    if not args.score_only:
        render(args.ckpt, seeds, outdir)
    res = score(seeds, outdir, gl_dir)
    rep = OUT / f"G-F_{args.ckpt.stem}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(res, indent=2))
    verdict = "NO DATA" if res["pass"] is None else ("PASS" if res["pass"] else "FAIL")
    print(f"\n{'='*62}\nG-F: {verdict}")
    print(f"  motion    : {res['motion_pass']}   (no clip frozen)")
    print(f"  no-trigger: {res['no_trigger_pass']}   (idle never matches an emotion)")
    print(f"  report: {rep}")


if __name__ == "__main__":
    main()
