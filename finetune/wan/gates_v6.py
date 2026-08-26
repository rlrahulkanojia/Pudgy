#!/usr/bin/env python3
"""
v6 gate harness — generate the gate matrices, measure them, and return a verdict.

Why this exists: v5's headline lesson (report section 4.3 note) is that weight-space
distance and loss both said "fine" while the model had in fact stopped responding to
the prompt. Only a behavioural test caught it. So every gate here drives the actual
inference path and measures the produced video; nothing is inferred from training
metrics.

Gates implemented (Training_Approach_v6 section 7):

  G-C  controllability  — same start frame, prompt is the only variable, 4 emotions
                          x 2 characters. Pass: all 6 pairwise SSIMs < 0.95.
  G-L  length de-confound — every emotion at every length. Pass: emotion tracks the
                          PROMPT, not the frame count. This is the gate that matters
                          most for this run, because the length<->emotion bijection
                          could not be fixed in the data (see below).
  G-M  motion preserved — neutral + head-turn from a NOVEL frame. Pass: subject moves.
  G-F  no frame trigger — same, from a TRAINING frame, at all 3 shot sizes.
  G-B  background invariance — two unseen grounds. Pass: corner drift <= 5/255.
  G-Z  shot size        — shot-size clause is the only variable.
  G-H  hold             — 57-frame neutral / 37-frame angry sustained past f30.

  G-P is not a separate matrix: it is G-C read per character, so it comes for free.

IMPORTANT — this dataset ships the length confound unfixed. Each emotion exists at
exactly one length (21 happy / 29 surprised / 37 angry / 57 neutral), so clip length
predicts emotion perfectly. The only no-new-data fix was emitting truncated copies,
which was ruled out (original lengths only). G-L is therefore the primary safeguard
rather than a formality: if it fails, the recovery is
`prep_expressions_v6.py --common-bucket 21` + retrain (plan section 10.2).

A metrics caveat, stated because v5 was burned by exactly this: the plan's numeric
thresholds (e.g. "x-range > 200 px") were produced by v5's own measurement code, which
did not survive the box. The estimators below are re-implementations, so absolute
values are NOT guaranteed comparable to v5's. Treat the thresholds as provisional and
read the per-run numbers relative to each other until one full run re-anchors them.
Every metric is written to the JSON so it can be re-scored later without regenerating.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_expressions_v6 import (  # noqa: E402  — the single source of caption truth
    ANCHORS, BACKGROUNDS, EMOTIONS, STYLE, ZOOMS,
)
from prep_eval_keyframes_v6 import UNSEEN  # noqa: E402

HERE = Path(__file__).resolve().parent
KF = Path("/workspace/eval_v6/keyframes")
OUT = Path("/workspace/eval_v6")

FRONT = "facing the camera directly, front view"

# G-C distinctness threshold on the FACE crop. Provisional: derived from 3 pairs at one
# seed on epochs 2-3 (different prompts 0.84-0.88, same prompt 0.95). Re-anchor once a
# full multi-seed sweep exists; until then read pairs near the line as undecided rather
# than as passes. The whole-frame 0.95 bar from v5 is kept alongside for comparability.
FACE_DISTINCT = 0.92
# The neutral/idle action clause, used when a gate must ask for NO expression. It is
# phrased in the same register as the trained action clauses so it stays in
# distribution; it is not one of the four trained emotion labels.
IDLE = ("standing still and turning its head slowly to look to one side, gentle "
        "bouncy idle motion")


def caption_for(char, emotion, zoom=1.00, bg="white", clause=FRONT, action=None):
    """Rebuild a training-distribution caption. Mirrors prep_expressions_v6.caption()."""
    act = action if action is not None else EMOTIONS[emotion]["action"]
    label = f"{emotion} expression" if action is None else "neutral expression"
    bg_desc = BACKGROUNDS[bg][0] if bg in BACKGROUNDS else UNSEEN[bg][1]
    return (f"{STYLE}{ANCHORS[char]}, {act}; {label}; "
            f"{ZOOMS[zoom]}, eye level, {clause}; {bg_desc}.")


# ---------------------------------------------------------------- video + metrics ---
def read_video(path):
    """Decode to (T,H,W,3) uint8 without pulling in a video lib."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def gray(v):
    return (v[..., 0] * 0.299 + v[..., 1] * 0.587 + v[..., 2] * 0.114).astype(np.float32)


def face_box(zoom=1.00, size=1024):
    """Crop around the head for a centred close-up, scaled by the shot-size zoom.

    Measured on the real renders: the head sits in the upper-middle of the frame at
    1.00x. The box is scaled about the frame centre so it still lands on the face at
    0.75x medium and 0.55x wide, where the character is composited smaller.
    """
    y0, y1, x0, x1 = 120, 470, 330, 700
    cy, cx = size / 2, size / 2
    def s(v, c):
        return int(round(c + (v - c) * zoom))
    return max(0, s(y0, cy)), min(size, s(y1, cy)), max(0, s(x0, cx)), min(size, s(x1, cx))


def ssim_pair(a, b, region=None):
    """Mean per-frame SSIM over the overlapping prefix of two clips.

    Compared frame-by-frame on the shared prefix rather than resampled to a common
    length: G-L deliberately compares clips of DIFFERENT lengths, and time-warping
    them would manufacture similarity that is not in the videos.

    `region` restricts the comparison to a crop. Use it for anything measuring an
    EXPRESSION change: about 85% of these frames are body, outline and flat background
    that the prompt never touches, so whole-frame SSIM is dominated by pixels that
    cannot differ and it badly understates the difference. Measured on real renders:

        pair                     whole-frame   face-region
        neutral vs happy            0.9485        0.8536
        angry   vs happy            0.9382        0.8387
        angry   vs neutral          0.9410        0.8821
        same prompt (control)       0.9834        0.9528

    Whole-frame put one genuine pair 0.0015 from failing; the face crop clears it by
    0.10. This is the same trap v4 hit with whole-frame optical flow under-counting a
    localized flipper wave.
    """
    n = min(len(a), len(b))
    ga, gb = gray(a[:n]), gray(b[:n])
    if region:
        y0, y1, x0, x1 = region
        ga, gb = ga[:, y0:y1, x0:x1], gb[:, y0:y1, x0:x1]
    return float(np.mean([ssim(ga[i], gb[i], data_range=255) for i in range(n)]))


def adjacent_ssim(v):
    g = gray(v)
    vals = [ssim(g[i], g[i + 1], data_range=255) for i in range(len(g) - 1)]
    return float(np.mean(vals)), float(np.min(vals))


def ssim_vs_f0(v):
    g = gray(v)
    return float(ssim(g[0], g[-1], data_range=255))


def subject_mask_flat(v, bg_rgb, thresh=28):
    """Subject mask on a known flat ground: pixels far from the background colour."""
    d = np.abs(v.astype(np.int16) - np.array(bg_rgb, np.int16)).sum(-1)
    return d > thresh


def subject_stats_flat(v, bg_rgb):
    """Area fraction and horizontal centre-of-mass range, for flat-ground clips."""
    m = subject_mask_flat(v, bg_rgb)
    areas, xs = [], []
    for t in range(len(m)):
        mt = m[t]
        areas.append(float(mt.mean()))
        if mt.any():
            xs.append(float(np.argwhere(mt)[:, 1].mean()))
    return {
        "area_first": areas[0], "area_last": areas[-1], "area_min": float(np.min(areas)),
        "x_range_px": float(np.max(xs) - np.min(xs)) if xs else 0.0,
    }


def motion_xrange(v):
    """Horizontal range of the motion-energy centroid, for clips over a real scene.

    On a static camera the only thing that changes is the character, so the centroid of
    |f_t - f_0| tracks where the movement is. Used where a colour key cannot work
    (the v1 skit frames have full backgrounds, not flat grounds).
    """
    g = gray(v)
    xs = []
    for t in range(1, len(g)):
        d = np.abs(g[t] - g[0])
        if d.sum() < 1e-3:
            continue
        col = d.sum(0)
        xs.append(float((col * np.arange(len(col))).sum() / col.sum()))
    if len(xs) < 2:
        return 0.0
    return float(np.max(xs) - np.min(xs))


def corner_drift(v, k=64):
    """Max per-channel background drift, measured in the four corners."""
    def corners(f):
        return np.stack([f[:k, :k], f[:k, -k:], f[-k:, :k], f[-k:, -k:]])
    a = corners(v[0].astype(np.int16)).reshape(4, -1, 3).mean(1)
    b = corners(v[-1].astype(np.int16)).reshape(4, -1, 3).mean(1)
    return float(np.abs(a - b).max())


# ------------------------------------------------------------------- generation -----
def generate(ckpt, tag, prompt, start, frames, seed, outdir, blkswap=0, dry=False):
    dest = Path(outdir) / f"{tag}.mp4"
    if dest.exists():
        return dest
    env = {
        "CKPT": str(ckpt), "PROMPT": prompt, "START": str(start),
        "FRAMES": str(frames), "SEED": str(seed), "TAG": tag,
        "OUTDIR": str(outdir), "BLKSWAP": str(blkswap),
    }
    if dry:
        print(f"   [dry] {tag}: f{frames} seed{seed} :: {prompt[:110]}...")
        return None
    import os
    e = dict(os.environ); e.update(env)
    r = subprocess.run(["bash", str(HERE / "eval_v6.sh")], env=e,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        sys.exit(f"generation failed for {tag}")
    return dest


# ------------------------------------------------------------------------ gates -----
def gate_gc(ckpt, seeds, outdir, dry):
    """G-C / G-P: prompt is the only variable. Pass = all 6 emotion pairs SSIM < 0.95."""
    clips, rows = {}, []
    for char in ("Pax", "Polly"):
        start = KF / f"{char.lower()}_neutral_start.png"
        for emo in EMOTIONS:
            for seed in seeds:
                tag = f"gc_{char.lower()}_{emo}_s{seed}"
                p = generate(ckpt, tag, caption_for(char, emo), start, 21, seed, outdir,
                             dry=dry)
                clips[(char, emo, seed)] = p
    if dry:
        return {"gate": "G-C", "status": "dry"}
    fb = face_box(1.00)
    for char in ("Pax", "Polly"):
        for seed in seeds:
            for a, b in itertools.combinations(EMOTIONS, 2):
                va, vb = read_video(clips[(char, a, seed)]), read_video(clips[(char, b, seed)])
                whole = ssim_pair(va, vb)
                face = ssim_pair(va, vb, region=fb)
                # Gate on the face crop: it is the region the prompt actually controls,
                # and it separates ~0.10 where whole-frame separates ~0.04. Whole-frame is
                # still recorded so these stay comparable with v5's published numbers.
                rows.append({"character": char, "seed": seed, "pair": f"{a}|{b}",
                             "ssim_face": round(face, 4), "ssim_whole": round(whole, 4),
                             "distinct": face < FACE_DISTINCT})
    worst = max(rows, key=lambda r: r["ssim_face"])
    per_char = {c: round(float(np.mean([r["ssim_face"] for r in rows if r["character"] == c])), 4)
                for c in ("Pax", "Polly")}
    gap = abs(per_char["Pax"] - per_char["Polly"])
    return {"gate": "G-C", "rows": rows, "worst_pair": worst,
            "pass": all(r["distinct"] for r in rows),
            "G-P_mean_ssim_per_character": per_char,
            "G-P_pass": gap <= 0.10 * max(per_char.values())}


def gate_gl(ckpt, seeds, outdir, dry):
    """G-L: every emotion at every length. The confound test.

    Two readings from one matrix:
      prompt_effect — at a FIXED length, do different emotion prompts diverge? If the
                      model reads length instead of the caption, they will not.
      length_effect — for a FIXED emotion, does changing length change the content
                      beyond the extra frames? Large divergence here means the length
                      is steering the emotion.
    """
    lengths = sorted({e["frames"] for e in EMOTIONS.values()})
    clips = {}
    for emo in EMOTIONS:
        for n in lengths:
            for seed in seeds:
                tag = f"gl_pax_{emo}_f{n}_s{seed}"
                clips[(emo, n, seed)] = generate(
                    ckpt, tag, caption_for("Pax", emo), KF / "pax_neutral_start.png",
                    n, seed, outdir, dry=dry)
    if dry:
        return {"gate": "G-L", "status": "dry"}
    prompt_effect, length_effect = [], []
    for seed in seeds:
        for n in lengths:
            for a, b in itertools.combinations(EMOTIONS, 2):
                s = ssim_pair(read_video(clips[(a, n, seed)]), read_video(clips[(b, n, seed)]))
                prompt_effect.append({"length": n, "pair": f"{a}|{b}", "seed": seed,
                                      "ssim": round(s, 4), "distinct": s < 0.95})
        for emo in EMOTIONS:
            for x, y in itertools.combinations(lengths, 2):
                s = ssim_pair(read_video(clips[(emo, x, seed)]), read_video(clips[(emo, y, seed)]))
                length_effect.append({"emotion": emo, "lengths": f"{x}|{y}", "seed": seed,
                                      "ssim": round(s, 4)})
    ok = all(r["distinct"] for r in prompt_effect)
    return {"gate": "G-L", "prompt_effect": prompt_effect, "length_effect": length_effect,
            "pass": ok,
            "note": ("emotion must track the prompt at EVERY length; a length where the "
                     "emotion prompts collapse together is the section 4.1 confound "
                     "surfacing -> recovery is --common-bucket 21 + retrain")}


def gate_gm(ckpt, seeds, outdir, dry):
    """G-M: motion survives, driven from a novel frame (real scene, not a flat ground)."""
    rows = []
    for char in ("pax", "polly"):
        name = {"pax": "Pax", "polly": "Polly"}[char]
        for seed in seeds:
            tag = f"gm_{char}_s{seed}"
            p = generate(ckpt, tag,
                         caption_for(name, "neutral", clause="in a room, medium shot",
                                     action=IDLE),
                         KF / f"novel_v1_{char}_start.png", 21, seed, outdir, dry=dry)
            if dry:
                continue
            v = read_video(p)
            adj, _ = adjacent_ssim(v)
            rows.append({"character": name, "seed": seed,
                         "motion_xrange_px": round(motion_xrange(v), 1),
                         "adjacent_ssim": round(adj, 4)})
    if dry:
        return {"gate": "G-M", "status": "dry"}
    return {"gate": "G-M", "rows": rows,
            "pass": all(r["motion_xrange_px"] > 200 for r in rows),
            "note": "threshold 200 px is v5's calibration under a different estimator"}


def gate_gf(ckpt, seeds, outdir, dry):
    """G-F: the exact v5 failure — does a TRAINING start frame trigger an expression?"""
    rows = []
    for char in ("pax", "polly"):
        name = {"pax": "Pax", "polly": "Polly"}[char]
        for shot, zoom in (("closeup", 1.00), ("medium", 0.75), ("wide", 0.55)):
            for seed in seeds:
                tag = f"gf_{char}_{shot}_s{seed}"
                p = generate(ckpt, tag,
                             caption_for(name, "neutral", zoom=zoom, action=IDLE),
                             KF / f"{char}_neutral_{shot}_start.png", 21, seed, outdir,
                             dry=dry)
                if dry:
                    continue
                v = read_video(p)
                st = subject_stats_flat(v, BACKGROUNDS["white"][1])
                rows.append({"character": name, "shot": shot, "seed": seed,
                             "x_range_px": round(st["x_range_px"], 1),
                             "area_first": round(st["area_first"], 4),
                             "area_last": round(st["area_last"], 4)})
    if dry:
        return {"gate": "G-F", "status": "dry"}
    return {"gate": "G-F", "rows": rows,
            "pass": all(r["x_range_px"] > 100 for r in rows),
            "note": ("x-range only detects the MOTION half. Whether a neutral prompt "
                     "silently fires an expression is a visual check on the montages — "
                     "v5 section 4.3 was caught by eye, not by a number.")}


def gate_gb(ckpt, seeds, outdir, dry):
    """G-B: unseen grounds. Pass = corner drift <= 5/255."""
    rows = []
    for char in ("pax", "polly"):
        name = {"pax": "Pax", "polly": "Polly"}[char]
        for ground in UNSEEN:
            for seed in seeds:
                tag = f"gb_{char}_{ground}_s{seed}"
                p = generate(ckpt, tag, caption_for(name, "happy", bg=ground),
                             KF / f"{char}_neutral_{ground}_start.png", 21, seed, outdir,
                             dry=dry)
                if dry:
                    continue
                v = read_video(p)
                rows.append({"character": name, "ground": ground, "seed": seed,
                             "corner_drift": round(corner_drift(v), 2)})
    if dry:
        return {"gate": "G-B", "status": "dry"}
    return {"gate": "G-B", "rows": rows,
            "pass": all(r["corner_drift"] <= 5 for r in rows)}


def gate_gz(ckpt, seeds, outdir, dry):
    """G-Z: shot size is promptable, and the expression survives at wide."""
    rows = []
    for shot, zoom in (("closeup", 1.00), ("medium", 0.75), ("wide", 0.55)):
        for seed in seeds:
            tag = f"gz_pax_happy_{shot}_s{seed}"
            p = generate(ckpt, tag, caption_for("Pax", "happy", zoom=zoom),
                         KF / "pax_neutral_start.png", 21, seed, outdir, dry=dry)
            if dry:
                continue
            v = read_video(p)
            st = subject_stats_flat(v, BACKGROUNDS["white"][1])
            rows.append({"shot": shot, "seed": seed,
                         "subject_area": round(st["area_first"], 4)})
    if dry:
        return {"gate": "G-Z", "status": "dry"}
    # Framing tracks the prompt if subject area orders closeup > medium > wide.
    by_shot = {}
    for r in rows:
        by_shot.setdefault(r["shot"], []).append(r["subject_area"])
    means = {k: float(np.mean(v)) for k, v in by_shot.items()}
    ordered = means.get("closeup", 0) > means.get("medium", 0) > means.get("wide", 0)
    return {"gate": "G-Z", "rows": rows, "mean_area_by_shot": means,
            "pass": ordered,
            "note": "expression legibility at wide is a visual check, not a number"}


def gate_gh(ckpt, seeds, outdir, dry):
    """G-H: does the expression hold past f30 on the long deliveries?"""
    rows = []
    for emo, n in (("neutral", 57), ("angry", 37)):
        for seed in seeds:
            tag = f"gh_pax_{emo}_f{n}_s{seed}"
            p = generate(ckpt, tag, caption_for("Pax", emo),
                         KF / "pax_neutral_start.png", n, seed, outdir, dry=dry)
            if dry:
                continue
            v = read_video(p)
            g = gray(v)
            # "Sustained" = late frames stay close to the peak-expression frame rather
            # than relaxing back toward the neutral start.
            to_start = np.array([ssim(g[0], g[i], data_range=255) for i in range(len(g))])
            peak = int(np.argmin(to_start))
            late = [float(ssim(g[peak], g[i], data_range=255))
                    for i in range(min(30, len(g) - 1), len(g))]
            rows.append({"emotion": emo, "frames": n, "seed": seed,
                         "peak_frame": peak,
                         "late_vs_peak_ssim_min": round(float(np.min(late)), 4)})
    if dry:
        return {"gate": "G-H", "status": "dry"}
    return {"gate": "G-H", "rows": rows,
            "pass": all(r["late_vs_peak_ssim_min"] > 0.90 for r in rows),
            "note": "v5 relaxed by f11 on 21 frames; these are the first long clips"}


GATES = {"gc": gate_gc, "gl": gate_gl, "gm": gate_gm, "gf": gate_gf,
         "gb": gate_gb, "gz": gate_gz, "gh": gate_gh}
EP2 = ["gc", "gl", "gm", "gf"]   # the ~epoch-2 set: fires before the full run is paid for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--gates", default="ep2",
                    help="'ep2' (gc,gl,gm,gf), 'all', or a comma list")
    ap.add_argument("--seeds", default="42,7,123",
                    help="v4 section 5.1 / v5 section 5.1: never judge on one seed")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the matrix and the prompts without generating")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    which = EP2 if args.gates == "ep2" else (
        list(GATES) if args.gates == "all" else args.gates.split(","))
    for g in which:
        if g not in GATES:
            sys.exit(f"unknown gate {g}; have {sorted(GATES)}")

    stem = args.ckpt.stem
    outdir = args.outdir or (OUT / "out" / stem)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.ckpt.exists():
        sys.exit(f"checkpoint not found: {args.ckpt}")

    print(f"checkpoint : {args.ckpt}")
    print(f"gates      : {', '.join(which)}")
    print(f"seeds      : {seeds}")
    print(f"outdir     : {outdir}\n")

    results = {}
    for g in which:
        print(f"--- {g.upper()} ---")
        results[g] = GATES[g](args.ckpt, seeds, outdir, args.dry_run)
        if not args.dry_run:
            r = results[g]
            print(f"   {r['gate']}: {'PASS' if r.get('pass') else 'FAIL'}")

    if args.dry_run:
        print("\n[dry run] nothing generated")
        return

    rep = OUT / f"gates_{stem}.json"
    rep.write_text(json.dumps(results, indent=2))
    print(f"\nreport: {rep}")
    print("\nsummary")
    for g, r in results.items():
        print(f"   {r['gate']:<5} {'PASS' if r.get('pass') else 'FAIL'}")
    print("\nNumbers are necessary, not sufficient — v5's regression was found by eye "
          "after the metrics said fine. Review the montages before declaring a golden.")


if __name__ == "__main__":
    main()
