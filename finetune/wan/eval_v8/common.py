"""Shared paths, LoRA configurations and caption construction for the v8 evaluation.

Every eval prompt is built by prep_v8.caption() itself, from a job dict of the same shape
the dataset builder uses, so eval prompts sit in the training distribution by
construction (the v6/v7 rule: import captions, never retype them).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prep_v8 as P  # noqa: E402

WS = Path("/workspace")
MODELS = WS / "wan_models"
DIT_LOW = MODELS / "comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH = MODELS / "comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
VAE = MODELS / "comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5 = MODELS / "t5/models_t5_umt5-xxl-enc-bf16.pth"
MUSUBI = WS / "musubi-tuner"
PY = WS / "Pudgy/.venv-wan/bin/python"

GOLD_LOW = WS / "wan_output/v2_golden/lora_lownoise_GOLDEN_ep40.safetensors"
GOLD_HIGH = WS / "wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors"
V7_EXPR = WS / "wan_output/v7_baseline/pudgy-v7-expr-lownoise-step00002500.safetensors"
V7_MOTION = WS / "wan_output/v7_baseline/pudgy-v7-motion-lownoise.safetensors"
# L_e + L_m - G as ONE exact rank-48 LoRA (= G + tau_e + tau_m): the v8 training base.
V7_COMBINED = WS / "wan_output/v7_baseline/pudgy-v7-fixed-combined.safetensors"
V8_RUN = WS / "wan_output/pudgy-v8-joint-lownoise"

EVAL = WS / "eval_v8"
KF = EVAL / "keyframes"
HOLDOUT = EVAL / "holdout"

CHARS = ("Pax", "Polly")
EXPR_LABELS = tuple(P.EXPRESSIONS)                    # 8, incl. blushing
MOTION_LABELS = tuple(k for k in P.MOTION if k != "standing_idle")   # 5 delivered actions
UNSEEN_BG = {"lavender": ("plain pastel lavender background", (222, 214, 240)),
             "skyblue": ("plain pastel sky-blue background", (205, 232, 246))}
# Start-frame views for the controllability matrix. QF2_R is the v8 HOLDOUT angle: the
# model never trained on it, so it measures generalisation, not memorisation.
VIEWS = ("FRONT", "QF1_L", "SIDE_R", "QF2_R")


def lora_config(name):
    """Named inference configurations: (low-noise [(path, mult)], high-noise [(path, mult)]).

    v8@<step>   v8 trains on top of v7fixed (merged at training time), so inference loads
                that same base explicitly and adds the v8 task vector on top.
    v7fixed     v7 expression + motion with the double-counted golden SUBTRACTED
                (L_e + L_m - G = G + tau_e + tau_m) - the exact combined file. This is
                v8 at step 0, i.e. the baseline v8 must beat.
    v7fixed3    the same sum as three separate files (1, 1, -1) - a cross-check.
    """
    hi = [(GOLD_HIGH, 1.0)]
    if name == "golden":
        return [(GOLD_LOW, 1.0)], hi
    if name == "v7expr":
        return [(V7_EXPR, 1.0)], hi
    if name == "v7motion":
        return [(V7_MOTION, 1.0)], hi
    if name == "v7fixed":
        return [(V7_COMBINED, 1.0)], hi
    if name == "v7fixed3":
        return [(V7_EXPR, 1.0), (V7_MOTION, 1.0), (GOLD_LOW, -1.0)], hi
    if name == "v7naive":          # what stacking the two v7 files naively gives - 2G
        return [(V7_EXPR, 1.0), (V7_MOTION, 1.0)], hi
    if name.startswith("v8@"):
        return [(V7_COMBINED, 1.0), (v8_ckpt(name[3:]), 1.0)], hi
    if name.startswith("file:"):   # any v8-style tau file on top of the v8 base
        return [(V7_COMBINED, 1.0), (Path(name[5:]), 1.0)], hi
    raise ValueError(f"unknown LoRA config {name!r}")


def v8_ckpt(step):
    if step in ("final", "last"):
        return V8_RUN / "pudgy-v8-joint-lownoise.safetensors"
    if step.startswith("swa"):
        return V8_RUN / f"pudgy-v8-joint-lownoise-{step}.safetensors"
    return V8_RUN / f"pudgy-v8-joint-lownoise-step{int(step):08d}.safetensors"


def kind_of(label):
    return "motion" if label in P.MOTION else "expression"


def job_for(char, label, view="FRONT"):
    kind = kind_of(label)
    slug = view if view == "FRONT" or view.startswith("SIDE") else view
    return {"char": char, "label": label, "kind": kind, "angle": slug,
            "clause": P.clause_for_slug(slug), "seam_ratio": 0.0, "src_frames": 10**6,
            "first_gap": None}


def caption(char, label, view="FRONT", nframes=None, zoom=1.00, bg="white"):
    """The training caption for (char, label, view, length). nframes defaults to the
    kind's common bucket, the one length where every label of the kind was trained."""
    job = job_for(char, label, view)
    n = nframes or P.COMMON[job["kind"]]
    bg_desc = P.BACKGROUNDS[bg][0] if bg in P.BACKGROUNDS else UNSEEN_BG[bg][0]
    return P.caption(job, bg_desc, zoom, n)


def keyframe(char, view="FRONT", bg="white"):
    return KF / f"{char.lower()}_{view}_{bg}.png"


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default
