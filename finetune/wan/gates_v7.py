#!/usr/bin/env python3
"""v7 behavioural gates — the v6 harness adapted on the three axes v7 changes.

The v6 scripts are NOT drop-in (GPU_HANDOFF_v7 section 7). Three differences:

  1. LABEL COUNT. v6 sized its matrices for four emotions. v7 has SEVEN expression
     labels (adds `confused`, `crying`, and `laughing`) and SIX motion labels. That
     turns G-C's 6 pairs per character into 21 (expression) / 15 (motion), which is
     why `--labels` exists to subset a sweep when a full matrix is not affordable.

  2. TWO EXPERTS. v6 trained low-noise only and always loaded the v2 high-noise
     golden as the frozen partner. In v7 the motion LoRA may also be trained
     (decided by G-X), so `--side` selects which expert carries the checkpoint and
     G-S loads two trained LoRAs at once. Every gate before G-S still runs against
     the UNTOUCHED v2 golden partner, so a regression stays bisectable — that is a
     plan requirement (section 4), not a default.

  3. G-D IS NEW. v6 emitted one length per label, so duration control could not be
     asked. v7's ladder makes it askable, and G-D is the gate the ladder exists for.

Captions are IMPORTED from prep_v7, never retyped, so eval prompts sit inside the
training distribution by construction (plan section 7).

Metrics are imported from gates_v6: they are geometry/statistics, independent of the
label set, and were calibrated on real v5/v6 renders. Re-deriving them would silently
break comparability with every published number.

  python gates_v7.py --gate gc --kind expression --ckpt <lora.safetensors> --seeds 42 43 44
  python gates_v7.py --gate gx --ckpt-high <M-high.safetensors> --ckpt-low <M-low.safetensors>
  python gates_v7.py --gate gs --ckpt-low <expr.safetensors> --ckpt-high <motion.safetensors>
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_v7 import (  # noqa: E402  — the single source of caption truth
    ANCHORS, BACKGROUNDS, EXPRESSIONS, MOTION, STYLE, ZOOMS, COMMON, FRONT,
    buckets_for,
)
from gates_v6 import (  # noqa: E402  — calibrated metrics, label-set independent
    read_video, ssim_pair, face_box, motion_xrange, corner_drift,
    subject_stats_flat, adjacent_ssim, gray as gray_,
)

HERE = Path(__file__).resolve().parent
KF = Path("/workspace/eval_v7/keyframes")
OUT = Path("/workspace/eval_v7")
# Which ground/zoom each keyframe actually has, written by prep_eval_keyframes_v7.
# G-F masks the subject against the flat ground, and the three shot sizes come from
# clips on DIFFERENT grounds, so this cannot be assumed.
try:
    KF_META = json.loads((KF / "keyframes.json").read_text())
except Exception:
    KF_META = {}

# Thresholds. Carried from v6/v5 calibration — see the plan's gate table (section 7).
FACE_DISTINCT = 0.92      # G-C on the face crop (v6 calibration)
WHOLE_DISTINCT = 0.95     # v5's whole-frame bar, kept for comparability
XRANGE_NOVEL = 200.0      # G-M: healthy 221-267 px, collapse 0.8 px
XRANGE_TRAIN = 100.0      # G-F, from a training start frame
XRANGE_IDLE = 50.0        # G-N: idle must NOT wander
CORNER_DRIFT = 5.0        # G-B: v5 measured 2/255
PARITY = 0.10             # G-P: Polly within 10% of Pax

# Grounds held out of training, for G-B. Same two v5/v6 reserved.
UNSEEN_BG = {
    "lavender": ("plain pastel lavender background", (222, 214, 240)),
    "skyblue":  ("plain pastel sky-blue background", (205, 232, 246)),
}


def table(kind):
    return EXPRESSIONS if kind == "expression" else MOTION


def caption_for(char, kind, label, zoom=1.00, bg="white", clause=FRONT, nframes=None,
                action=None):
    """Rebuild a training-distribution caption. Mirrors prep_v7.caption() exactly.

    prep_v7.caption() takes a `job` dict assembled during dataset build; the gates do
    not have one, so this reproduces the same string from the label tables. The
    `opening frames only` suffix is part of that contract: a clip shorter than the
    label's natural length was captioned that way in training, so a gate asking for a
    short rung must ask the same way or it is testing an out-of-distribution prompt.
    """
    rec = table(kind)[label]
    act = action if action is not None else rec["action"]
    lab = rec["label"]
    if nframes is not None and nframes < rec["frames"]:
        lab = f"{lab}, opening frames only"
    bg_desc = BACKGROUNDS[bg][0] if bg in BACKGROUNDS else UNSEEN_BG[bg][0]
    return (f"{STYLE}{ANCHORS[char]}, {act}; {lab}; "
            f"{ZOOMS[zoom]}, eye level, {clause}; {bg_desc}.")


def bg_rgb(bg):
    return BACKGROUNDS[bg][1] if bg in BACKGROUNDS else UNSEEN_BG[bg][1]


def natural_frames(kind, label):
    return table(kind)[label]["frames"]


def ladder_for(kind, label):
    """The frame lengths this label was actually emitted at.

    Delegates to prep_v7.buckets_for rather than re-deriving the ladder. G-D asks
    whether duration is promptable; if the gate generated a rung the dataset never
    contained, it would be measuring out-of-distribution behaviour and calling it a
    duration failure. The dataset builder is the only correct authority here.
    """
    return buckets_for({"kind": kind, "label": label})


# ------------------------------------------------------------------- generation -----
def generate(tag, prompt, start, frames, seed, outdir, side="low", ckpt=None,
             low=None, high=None, blkswap=0, dry=False):
    dest = Path(outdir) / f"{tag}.mp4"
    if dest.exists():
        return dest
    if dry:
        print(f"   [dry] {tag}: side={side} f{frames} seed{seed} :: {prompt[:100]}...")
        return None
    env = dict(os.environ)
    env.update({"PROMPT": prompt, "START": str(start), "FRAMES": str(frames),
                "SEED": str(seed), "TAG": tag, "OUTDIR": str(outdir),
                "BLKSWAP": str(blkswap), "SIDE": side})
    if ckpt:
        env["CKPT"] = str(ckpt)
    if low:
        env["LOW_LORA"] = str(low)
    if high:
        env["HIGH_LORA"] = str(high)
    r = subprocess.run(["bash", str(HERE / "eval_v7.sh")], env=env,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        sys.exit(f"generation failed for {tag}")
    return dest


def start_frame(char, kind):
    """Neutral/idle start frame for a character. Novel (unseen) frames live under
    KF/novel/ and are used by G-M; the trained-pose frames by G-F."""
    return KF / f"{char.lower()}_{'neutral' if kind == 'expression' else 'idle'}_start.png"


# ------------------------------------------------------------------------ gates -----
def gate_gc(a):
    """G-C / G-P: prompt is the only variable. Pass = every label pair distinguishable.

    7 expression labels -> 21 pairs per (character, seed); 6 motion -> 15.
    """
    labels = a.labels or list(table(a.kind))
    clips, rows = {}, []
    for char in ("Pax", "Polly"):
        st = start_frame(char, a.kind)
        for lab in labels:
            for seed in a.seeds:
                # Every label generated at the COMMON bucket so length is constant
                # across the matrix — otherwise G-C would re-introduce the very
                # length<->label confound the common bucket exists to break.
                n = COMMON[a.kind]
                tag = f"gc_{a.kind}_{char.lower()}_{lab}_s{seed}"
                clips[(char, lab, seed)] = generate(
                    tag, caption_for(char, a.kind, lab, nframes=n), st, n, seed,
                    a.outdir, side=a.side, ckpt=a.ckpt, dry=a.dry)
    if a.dry:
        return {"gate": "G-C", "kind": a.kind, "status": "dry",
                "pairs_per_char_seed": len(list(itertools.combinations(labels, 2)))}
    fb = face_box(1.00)
    for char in ("Pax", "Polly"):
        for seed in a.seeds:
            for x, y in itertools.combinations(labels, 2):
                vx, vy = read_video(clips[(char, x, seed)]), read_video(clips[(char, y, seed)])
                whole, face = ssim_pair(vx, vy), ssim_pair(vx, vy, region=fb)
                rows.append({"character": char, "seed": seed, "pair": f"{x}|{y}",
                             "ssim_face": round(face, 4), "ssim_whole": round(whole, 4),
                             "distinct": face < FACE_DISTINCT})
    per_char = {c: round(float(np.mean([r["ssim_face"] for r in rows if r["character"] == c])), 4)
                for c in ("Pax", "Polly")}
    gap = abs(per_char["Pax"] - per_char["Polly"])
    return {"gate": "G-C", "kind": a.kind, "rows": rows,
            "worst_pair": max(rows, key=lambda r: r["ssim_face"]),
            "pass": all(r["distinct"] for r in rows),
            "G-P_mean_ssim_per_character": per_char,
            "G-P_pass": gap <= PARITY * max(per_char.values())}


def gate_gd(a):
    """G-D (NEW in v7): duration control. Same label+character, length is the only variable.

    v6 could not ask this — it emitted one length per label, so length and label were
    bijective. The v7 ladder exists to make duration promptable, and this is the gate
    that decides whether it worked. Two things must hold: the output length tracks the
    REQUEST, and quality does not fall off at the short rungs.
    """
    labels = a.labels or list(table(a.kind))
    rows = []
    for char in ("Pax", "Polly"):
        st = start_frame(char, a.kind)
        for lab in labels:
            for n in ladder_for(a.kind, lab):
                for seed in a.seeds:
                    tag = f"gd_{a.kind}_{char.lower()}_{lab}_f{n}_s{seed}"
                    p = generate(tag, caption_for(char, a.kind, lab, nframes=n), st, n,
                                 seed, a.outdir, side=a.side, ckpt=a.ckpt, dry=a.dry)
                    if a.dry:
                        continue
                    v = read_video(p)
                    mean_adj, min_adj = adjacent_ssim(v)
                    rows.append({"character": char, "label": lab, "requested": n,
                                 "produced": len(v), "length_ok": len(v) == n,
                                 "seed": seed,
                                 "adj_ssim_mean": round(mean_adj, 4),
                                 "adj_ssim_min": round(min_adj, 4)})
    if a.dry:
        return {"gate": "G-D", "kind": a.kind, "status": "dry"}
    # Quality must not degrade at the SHORT rungs: compare each label's shortest rung
    # against its natural length on temporal coherence.
    degraded = []
    for char in ("Pax", "Polly"):
        for lab in labels:
            rs = [r for r in rows if r["character"] == char and r["label"] == lab]
            if len(rs) < 2:
                continue
            short = min(rs, key=lambda r: r["requested"])
            longr = max(rs, key=lambda r: r["requested"])
            if short["adj_ssim_mean"] < longr["adj_ssim_mean"] - 0.05:
                degraded.append({"character": char, "label": lab,
                                 "short": short["adj_ssim_mean"],
                                 "natural": longr["adj_ssim_mean"]})
    return {"gate": "G-D", "kind": a.kind, "rows": rows,
            "length_mismatches": [r for r in rows if not r["length_ok"]],
            "degraded_short_rungs": degraded,
            "pass": all(r["length_ok"] for r in rows) and not degraded}


def gate_gx(a):
    """G-X (NEW in v7): which expert gets motion. Decides plan section 4.

    M-high vs M-low on the SAME start frame with the action prompt as the only
    variable. The winning arm is the one whose action outputs stay distinguishable
    (pairwise SSIM < 0.95) AND which retains prompt response. v5 calibration:
    0.9692 = prompt ignored, 0.9340 = prompt works.

    This is a BEFORE-COMMITTING gate: it runs at ~2 epochs and picks the arm that
    then continues to the ceiling. Run it before spending the other ~25 h.
    """
    labels = a.labels or [l for l in MOTION if l != "standing_idle"][:5]
    arms = {"M-high": ("high", a.ckpt_high), "M-low": ("low", a.ckpt_low)}
    verdict = {}
    for arm, (side, ckpt) in arms.items():
        if not ckpt:
            continue
        clips, rows = {}, []
        for char in ("Pax", "Polly"):
            st = start_frame(char, "motion")
            for lab in labels:
                for seed in a.seeds:
                    n = COMMON["motion"]
                    tag = f"gx_{arm}_{char.lower()}_{lab}_s{seed}"
                    clips[(char, lab, seed)] = generate(
                        tag, caption_for(char, "motion", lab, nframes=n), st, n, seed,
                        a.outdir, side=side, ckpt=ckpt, dry=a.dry)
        if a.dry:
            verdict[arm] = {"status": "dry"}
            continue
        for char in ("Pax", "Polly"):
            for seed in a.seeds:
                for x, y in itertools.combinations(labels, 2):
                    vx = read_video(clips[(char, x, seed)])
                    vy = read_video(clips[(char, y, seed)])
                    s = ssim_pair(vx, vy)
                    rows.append({"character": char, "seed": seed, "pair": f"{x}|{y}",
                                 "ssim_whole": round(s, 4), "distinct": s < WHOLE_DISTINCT})
        # Prompt response: the subject must actually move, not just differ.
        xr = []
        for (char, lab, seed), p in clips.items():
            v = read_video(p)
            xr.append(subject_stats_flat(v, bg_rgb("white"))["x_range_px"])
        verdict[arm] = {"rows": rows, "all_distinct": all(r["distinct"] for r in rows),
                        "worst_ssim": round(max(r["ssim_whole"] for r in rows), 4),
                        "mean_x_range_px": round(float(np.mean(xr)), 1),
                        "retains_motion": float(np.mean(xr)) > XRANGE_NOVEL}
    if a.dry:
        return {"gate": "G-X", "status": "dry", "arms": list(verdict)}
    winners = [k for k, v in verdict.items()
               if v.get("all_distinct") and v.get("retains_motion")]
    return {"gate": "G-X", "arms": verdict, "winners": winners,
            "decision": (winners[0] if len(winners) == 1 else
                         (min(winners, key=lambda k: verdict[k]["worst_ssim"])
                          if winners else None)),
            "pass": bool(winners)}


def gate_gm(a):
    """G-M: motion preserved, from a NOVEL start frame. Pass = x-range > 200 px."""
    rows = []
    idle = ("standing still and turning its head slowly to look to one side, gentle "
            "bouncy idle motion")
    for char in ("Pax", "Polly"):
        st = KF / "novel" / f"{char.lower()}_novel_start.png"
        for seed in a.seeds:
            tag = f"gm_{char.lower()}_s{seed}"
            lab = "neutral" if a.kind == "expression" else "standing_idle"
            p = generate(tag, caption_for(char, a.kind, lab, action=idle), st,
                         COMMON[a.kind], seed, a.outdir, side=a.side, ckpt=a.ckpt, dry=a.dry)
            if a.dry:
                continue
            v = read_video(p)
            rows.append({"character": char, "seed": seed,
                         "x_range_px": round(motion_xrange(v), 1)})
    if a.dry:
        return {"gate": "G-M", "status": "dry"}
    return {"gate": "G-M", "rows": rows,
            "pass": all(r["x_range_px"] > XRANGE_NOVEL for r in rows)}


def gate_gn(a):
    """G-N: idle is promptable. EXPECTED TO FAIL until real idle footage lands.

    plan section 3.1 / prep_v7 DERIVE_IDLE_FROM_NEUTRAL: `standing_idle` is currently
    DERIVED from the `neutral` expression footage, whose body is frozen (measured MAE
    0.028-0.064). So this gate tests two things and reports them separately: the
    character must STOP (x-range < 50 px) and must still show LIFE (frame-to-frame
    change above the frozen-neutral floor). A pass on the first with a fail on the
    second is the documented current state, not a new defect.
    """
    rows = []
    for char in ("Pax", "Polly"):
        st = start_frame(char, "motion")
        for seed in a.seeds:
            tag = f"gn_{char.lower()}_s{seed}"
            p = generate(tag, caption_for(char, "motion", "standing_idle"), st,
                         COMMON["motion"], seed, a.outdir, side=a.side, ckpt=a.ckpt,
                         dry=a.dry)
            if a.dry:
                continue
            v = read_video(p)
            g = v.astype(np.float32) / 255.0
            mae = float(np.mean(np.abs(g[1:] - g[:-1])))
            rows.append({"character": char, "seed": seed,
                         "x_range_px": round(subject_stats_flat(v, bg_rgb("white"))["x_range_px"], 1),
                         "frame_mae": round(mae, 4)})
    if a.dry:
        return {"gate": "G-N", "status": "dry"}
    stops = all(r["x_range_px"] < XRANGE_IDLE for r in rows)
    alive = all(r["frame_mae"] > 0.064 for r in rows)   # above frozen-neutral ceiling
    return {"gate": "G-N", "rows": rows, "stops": stops, "shows_life": alive,
            "pass": stops and alive,
            "note": "shows_life is EXPECTED False until real idle footage is delivered "
                    "(plan 3.1); stops=True with shows_life=False is the known state."}


def gate_gb(a):
    """G-B: background invariance on two UNSEEN grounds. Pass = corner drift <= 5/255."""
    labels = a.labels or list(table(a.kind))[:3]
    rows = []
    for char in ("Pax", "Polly"):
        st = start_frame(char, a.kind)
        for bg in UNSEEN_BG:
            for lab in labels:
                for seed in a.seeds:
                    tag = f"gb_{char.lower()}_{lab}_{bg}_s{seed}"
                    p = generate(tag, caption_for(char, a.kind, lab, bg=bg), st,
                                 COMMON[a.kind], seed, a.outdir, side=a.side,
                                 ckpt=a.ckpt, dry=a.dry)
                    if a.dry:
                        continue
                    rows.append({"character": char, "label": lab, "background": bg,
                                 "seed": seed, "corner_drift": round(corner_drift(read_video(p)), 2)})
    if a.dry:
        return {"gate": "G-B", "status": "dry"}
    return {"gate": "G-B", "rows": rows,
            "pass": all(r["corner_drift"] <= CORNER_DRIFT for r in rows)}


def gate_gs(a):
    """G-S: stacking. ONLY meaningful if M-high won G-X (plan section 4).

    Loads BOTH trained LoRAs and checks the two axes respond independently. This is
    the test that v5's frozen-partner protection has not been silently lost: with two
    trained experts there is no untouched partner left to A/B against, so the pair
    must be validated as a pair.
    """
    if not (a.ckpt_low and a.ckpt_high):
        sys.exit("G-S needs BOTH --ckpt-low and --ckpt-high")
    rows = []
    motions = a.labels or [l for l in MOTION if l != "standing_idle"][:3]
    exprs = list(EXPRESSIONS)[:3]
    for char in ("Pax", "Polly"):
        st = start_frame(char, "motion")
        for mo in motions:
            for ex in exprs:
                for seed in a.seeds:
                    # Both clauses in one caption: motion action + expression label.
                    cap = (f"{STYLE}{ANCHORS[char]}, {MOTION[mo]['action']}, "
                           f"{EXPRESSIONS[ex]['action']}; {MOTION[mo]['label']}; "
                           f"{ZOOMS[1.00]}, eye level, {FRONT}; {BACKGROUNDS['white'][0]}.")
                    tag = f"gs_{char.lower()}_{mo}_{ex}_s{seed}"
                    p = generate(tag, cap, st, COMMON["motion"], seed, a.outdir,
                                 side="both", low=a.ckpt_low, high=a.ckpt_high, dry=a.dry)
                    if a.dry:
                        continue
                    v = read_video(p)
                    rows.append({"character": char, "motion": mo, "expression": ex,
                                 "seed": seed,
                                 "x_range_px": round(motion_xrange(v), 1)})
    if a.dry:
        return {"gate": "G-S", "status": "dry"}
    return {"gate": "G-S", "rows": rows,
            "motion_retained": all(r["x_range_px"] > XRANGE_NOVEL for r in rows),
            "pass": all(r["x_range_px"] > XRANGE_NOVEL for r in rows)}



def gate_gl(a):
    """G-L: every label at every bucket of its kind. The length<->label confound test.

    Two readings from one matrix:
      prompt_effect — at a FIXED length, do different label prompts diverge? If the
                      model reads length instead of the caption, they will not.
      length_effect — for a FIXED label, does changing length change content beyond
                      the extra frames? Large divergence means length steers the label.

    v7 vs v6: v6 had one length per label, so its "every length" was the set of natural
    lengths. v7 has a real ladder, so the lengths tested are the label's OWN rungs plus
    the common floor — asking for a rung the label was never emitted at would test
    out-of-distribution behaviour and call it a confound.
    """
    labels = a.labels or list(table(a.kind))
    floor = COMMON[a.kind]
    clips, seen = {}, {}
    for char in ("Pax", "Polly") if a.both_chars else ("Pax",):
        st = start_frame(char, a.kind)
        for lab in labels:
            rungs = ladder_for(a.kind, lab)
            seen[lab] = rungs
            for n in rungs:
                for seed in a.seeds:
                    tag = f"gl_{a.kind}_{char.lower()}_{lab}_f{n}_s{seed}"
                    clips[(char, lab, n, seed)] = generate(
                        tag, caption_for(char, a.kind, lab, nframes=n), st, n, seed,
                        a.outdir, side=a.side, ckpt=a.ckpt, dry=a.dry)
    if a.dry:
        return {"gate": "G-L", "kind": a.kind, "status": "dry"}
    prompt_effect, length_effect = [], []
    chars = ("Pax", "Polly") if a.both_chars else ("Pax",)
    for char in chars:
        for seed in a.seeds:
            # prompt effect at the COMMON floor, the one bucket every label shares
            for x, y in itertools.combinations(labels, 2):
                kx, ky = (char, x, floor, seed), (char, y, floor, seed)
                if kx not in clips or ky not in clips:
                    continue
                sc = ssim_pair(read_video(clips[kx]), read_video(clips[ky]))
                prompt_effect.append({"character": char, "length": floor,
                                      "pair": f"{x}|{y}", "seed": seed,
                                      "ssim": round(sc, 4), "distinct": sc < WHOLE_DISTINCT})
            for lab in labels:
                for x, y in itertools.combinations(seen[lab], 2):
                    kx, ky = (char, lab, x, seed), (char, lab, y, seed)
                    if kx not in clips or ky not in clips:
                        continue
                    sc = ssim_pair(read_video(clips[kx]), read_video(clips[ky]))
                    length_effect.append({"character": char, "label": lab,
                                          "lengths": f"{x}|{y}", "seed": seed,
                                          "ssim": round(sc, 4)})
    return {"gate": "G-L", "kind": a.kind, "prompt_effect": prompt_effect,
            "length_effect": length_effect,
            "pass": all(r["distinct"] for r in prompt_effect),
            "note": ("label must track the prompt at EVERY length. A length where the "
                     "prompts collapse together is the confound surfacing -> the common "
                     "floor bucket is not doing its job (plan 10.2).")}


def gate_gf(a):
    """G-F: the exact v5 failure — does a TRAINING start frame trigger the label?

    Uses the per-shot-size training frames, and reads each frame's real ground from the
    keyframes.json sidecar: subject masking keys off the flat background colour, and
    the three shot sizes come from clips on DIFFERENT grounds (the manifest ties zoom to
    background), so assuming white would corrupt the measurement.
    """
    rows = []
    idle_action = ("standing still and turning its head slowly to look to one side, "
                   "gentle bouncy idle motion")
    neutral_label = "neutral" if a.kind == "expression" else "standing_idle"
    for char in ("Pax", "Polly"):
        c = char.lower()
        for shot, zoom in (("close", 1.00), ("medium", 0.75), ("wide", 0.55)):
            frame = KF / f"{c}_neutral_{shot}_start.png"
            if not frame.exists():
                continue
            ground = KF_META.get(frame.name, {}).get("background", "white")
            for seed in a.seeds:
                tag = f"gf_{c}_{shot}_s{seed}"
                p = generate(tag,
                             caption_for(char, a.kind, neutral_label, zoom=zoom,
                                         bg=ground, action=idle_action),
                             frame, COMMON[a.kind], seed, a.outdir,
                             side=a.side, ckpt=a.ckpt, dry=a.dry)
                if a.dry:
                    continue
                st = subject_stats_flat(read_video(p), bg_rgb(ground))
                rows.append({"character": char, "shot": shot, "seed": seed,
                             "ground": ground,
                             "x_range_px": round(st["x_range_px"], 1),
                             "area_first": round(st["area_first"], 4),
                             "area_last": round(st["area_last"], 4)})
    if a.dry:
        return {"gate": "G-F", "kind": a.kind, "status": "dry"}
    return {"gate": "G-F", "kind": a.kind, "rows": rows,
            "pass": all(r["x_range_px"] > XRANGE_TRAIN for r in rows),
            "note": ("x-range only detects the MOTION half. Whether a neutral prompt "
                     "silently fires a label is a VISUAL check — v5 4.3 was caught by "
                     "eye, not by a number.")}


def gate_gz(a):
    """G-Z: shot size is promptable, and the label stays legible at 0.55x wide."""
    lab = (a.labels or ["happy" if a.kind == "expression" else "waving"])[0]
    rows = []
    for char in ("Pax", "Polly") if a.both_chars else ("Pax",):
        st = start_frame(char, a.kind)
        for shot, zoom in (("close", 1.00), ("medium", 0.75), ("wide", 0.55)):
            for seed in a.seeds:
                tag = f"gz_{char.lower()}_{lab}_{shot}_s{seed}"
                p = generate(tag, caption_for(char, a.kind, lab, zoom=zoom), st,
                             COMMON[a.kind], seed, a.outdir, side=a.side,
                             ckpt=a.ckpt, dry=a.dry)
                if a.dry:
                    continue
                s = subject_stats_flat(read_video(p), bg_rgb("white"))
                rows.append({"character": char, "shot": shot, "seed": seed,
                             "subject_area": round(s["area_first"], 4)})
    if a.dry:
        return {"gate": "G-Z", "kind": a.kind, "status": "dry"}
    by = {}
    for r in rows:
        by.setdefault(r["shot"], []).append(r["subject_area"])
    means = {k: round(float(np.mean(v)), 4) for k, v in by.items()}
    ordered = means.get("close", 0) > means.get("medium", 0) > means.get("wide", 0)
    return {"gate": "G-Z", "kind": a.kind, "rows": rows, "mean_area_by_shot": means,
            "label": lab, "pass": ordered,
            "note": "legibility at 0.55x wide is a visual check, not a number"}


def gate_gh(a):
    """G-H: does the label HOLD past f30 on the long deliveries?

    v7 tests the three the plan names: 57-frame neutral, 37-frame angry, 37-frame
    laughing. `laughing` is new in v7 and is the programme's first real emotive hold.
    """
    from skimage.metrics import structural_similarity as _ssim
    targets = [("neutral", 57), ("angry", 37), ("laughing", 37)] if a.kind == "expression" \
        else [("standing_idle", 33), ("waving", 25)]
    rows = []
    for char in ("Pax", "Polly") if a.both_chars else ("Pax",):
        st = start_frame(char, a.kind)
        for lab, n in targets:
            if lab not in table(a.kind) or n not in ladder_for(a.kind, lab):
                continue
            for seed in a.seeds:
                tag = f"gh_{char.lower()}_{lab}_f{n}_s{seed}"
                p = generate(tag, caption_for(char, a.kind, lab, nframes=n), st, n,
                             seed, a.outdir, side=a.side, ckpt=a.ckpt, dry=a.dry)
                if a.dry:
                    continue
                v = read_video(p)
                g = gray_(v)
                # "Sustained" = late frames stay close to the PEAK-expression frame
                # rather than relaxing back toward the neutral start.
                to_start = np.array([_ssim(g[0], g[i], data_range=255) for i in range(len(g))])
                peak = int(np.argmin(to_start))
                late = [float(_ssim(g[peak], g[i], data_range=255))
                        for i in range(min(30, len(g) - 1), len(g))]
                rows.append({"character": char, "label": lab, "frames": n, "seed": seed,
                             "peak_frame": peak,
                             "late_vs_peak_ssim_min": round(float(np.min(late)), 4)})
    if a.dry:
        return {"gate": "G-H", "kind": a.kind, "status": "dry"}
    return {"gate": "G-H", "kind": a.kind, "rows": rows,
            "pass": all(r["late_vs_peak_ssim_min"] > 0.90 for r in rows),
            "note": "v5 relaxed by f11 on 21 frames; laughing is the first real emotive hold"}


def gate_gr(a):
    """G-R: no regression against v2's showcase prompts.

    Re-runs v2's own scene prompts through the v7 stack and measures temporal stability.
    The v2 published baseline is temporal SSIM 0.949 / struct-stability 0.880; the plan
    requires >= 0.94 / >= 0.87. Prompts are read from a file rather than retyped so this
    compares against what v2 actually generated.
    """
    # NO silent fallback. sample_prompts.txt is v6's in-training sampler prompts, NOT
    # v2's 10-scene showcase — scoring those and calling the result "no regression"
    # would be a meaningless pass. The v2 showcase text is not in the repo and not on
    # Azure (only the rendered mp4s under v2/eval/showcase/ and a report that omits the
    # prompts), so it has to come from the author.
    if not a.prompts:
        sys.exit("G-R needs v2's 10-scene showcase prompts: pass --prompts <file>.\n"
                 "  They are NOT recoverable from this repo or the Azure container — "
                 "v2/eval/showcase/ has the videos (01_construction ... 10_doorway_mug) "
                 "and v2/docs/REPORT_showcase_analysis.md, but neither carries the "
                 "prompt text. Ask the author for the exact strings; substituting "
                 "different prompts makes the comparison to v2's 0.949/0.880 baseline "
                 "meaningless.")
    pf = Path(a.prompts)
    if not pf.exists():
        sys.exit(f"G-R prompts file not found: {pf}")
    prompts = [l.strip() for l in pf.read_text().splitlines() if l.strip()][:10]
    rows = []
    for i, prompt in enumerate(prompts):
        for seed in a.seeds:
            tag = f"gr_scene{i:02d}_s{seed}"
            p = generate(tag, prompt, KF / "novel" / "pax_novel_start.png",
                         COMMON[a.kind], seed, a.outdir, side=a.side, ckpt=a.ckpt,
                         dry=a.dry)
            if a.dry:
                continue
            v = read_video(p)
            mean_adj, min_adj = adjacent_ssim(v)
            rows.append({"scene": i, "seed": seed,
                         "temporal_ssim": round(mean_adj, 4),
                         "struct_stability": round(min_adj, 4)})
    if a.dry:
        return {"gate": "G-R", "status": "dry", "n_prompts": len(prompts)}
    return {"gate": "G-R", "rows": rows,
            "mean_temporal_ssim": round(float(np.mean([r["temporal_ssim"] for r in rows])), 4),
            "mean_struct_stability": round(float(np.mean([r["struct_stability"] for r in rows])), 4),
            "pass": (all(r["temporal_ssim"] >= 0.94 for r in rows)
                     and all(r["struct_stability"] >= 0.87 for r in rows)),
            "note": "v2 baseline: temporal 0.949 / struct 0.880"}


GATES = {"gc": gate_gc, "gd": gate_gd, "gx": gate_gx, "gm": gate_gm,
         "gn": gate_gn, "gb": gate_gb, "gs": gate_gs, "gl": gate_gl,
         "gf": gate_gf, "gz": gate_gz, "gh": gate_gh, "gr": gate_gr}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True, choices=sorted(GATES))
    ap.add_argument("--kind", default="expression", choices=("expression", "motion"))
    ap.add_argument("--side", default="low", choices=("low", "high", "both"),
                    help="which expert carries --ckpt (v6 could only do 'low')")
    ap.add_argument("--ckpt", help="checkpoint under test")
    ap.add_argument("--ckpt-low", help="low-noise LoRA (G-X arm M-low, or G-S)")
    ap.add_argument("--ckpt-high", help="high-noise LoRA (G-X arm M-high, or G-S)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help=">=3 seeds is a plan requirement (evidence #10)")
    ap.add_argument("--labels", nargs="*", help="subset the label matrix")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--both-chars", action="store_true",
                    help="run single-character gates (G-L/G-Z/G-H) for Polly too")
    ap.add_argument("--prompts", help="G-R: file of v2 showcase prompts")
    ap.add_argument("--dry", action="store_true", help="print the matrix, generate nothing")
    a = ap.parse_args()

    if a.labels:
        bad = [l for l in a.labels if l not in table(a.kind)]
        if bad:
            sys.exit(f"unknown {a.kind} labels: {bad}\nvalid: {list(table(a.kind))}")
    a.outdir = Path(a.outdir or (OUT / f"{a.gate}_{a.kind}"))
    if not a.dry:
        a.outdir.mkdir(parents=True, exist_ok=True)

    if len(a.seeds) < 3 and not a.dry:
        print("!! fewer than 3 seeds — the plan requires >=3 (evidence #10)", file=sys.stderr)

    res = GATES[a.gate](a)
    print(json.dumps(res, indent=2, default=str))
    if a.dry:
        # Never persist a dry run. A file named `<gate>_result.json` gets mirrored to
        # Azure and read as a real verdict; a directory of `status: dry` stubs looks
        # exactly like a completed gate suite, which is worse than having nothing.
        print("\n== dry run: no result file written", file=sys.stderr)
        return
    dest = a.outdir / f"{a.gate}_result.json"
    dest.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n== wrote {dest}", file=sys.stderr)
    if not a.dry and res.get("pass") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
