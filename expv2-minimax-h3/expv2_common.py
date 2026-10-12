"""expv2 shared contract: paths, model ids, labels, grounds, shot sizes, holdout, the frozen
caption template, inference settings and the evaluation suites.

Every expv2 script imports this module and nothing else from the repo:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import expv2_common as C

Isolation (README): tables copied from `finetune/wan/prep_v8.py` (at b902bde) are marked
"copied"; nothing outside `expv2-minimax-h3/` is ever imported.

Paths resolve the same way on the Mac and on the GPU box:
    PUDGY_DATA                  the `Data/` tree   (box: /workspace/Data)
    EXPV2_WORK                  box scratch: caches, checkpoints, eval output (/workspace/expv2)
    DIFFSYNTH_MODEL_BASE_PATH   where DiffSynth looks for weights (/workspace/models)
    DIFFSYNTH_DIR               the pinned DiffSynth-Studio checkout (/workspace/DiffSynth-Studio)
`configs/env.sh` exports all four.
"""
import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parent          # expv2-minimax-h3/
REPO = EXP.parent


def _data_root():
    for p in (os.environ.get("PUDGY_DATA"), "/workspace/Data", REPO.parent.parent / "Data"):
        if p and Path(p).is_dir():
            return Path(p)
    return Path("/workspace/Data")


DATA = _data_root()
RAW = DATA / "raw"
CORPUS = DATA / "processed" / "expv2_h3"          # prep output, mirrored to Azure processed/expv2_h3/
WORK = Path(os.environ.get("EXPV2_WORK", "/workspace/expv2"))
MODELS = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/workspace/models"))
DIFFSYNTH = Path(os.environ.get("DIFFSYNTH_DIR", "/workspace/DiffSynth-Studio"))

CACHE = WORK / "cache"            # stage-1 caches: cache/train/f{22,39,56}/, cache/eval/<set>/
RUNS = WORK / "output"            # stage-2 runs: output/<run>/step-<samples>.safetensors
EVAL = WORK / "eval"              # every eval artefact; mirrored to Azure expv2-minimax-h3/eval/
PHASE0 = EVAL / "phase0"
PINS = PHASE0 / "pins.json"

AZURE_ACCOUNT = "pudgytraining"
AZURE_CONTAINER = "pudgy"
AZURE_PREFIX = "expv2-minimax-h3"  # {weights,eval,logs,docs}/

# ---------------------------------------------------------------------------------------
# Model ids. HuggingFace ids throughout: configs/env.sh sets DIFFSYNTH_DOWNLOAD_SOURCE=
# HuggingFace, and DiffSynth only redirects "MiniMax/MiniMax-H3" -> "MiniMaxAI/MiniMax-H3"
# inside from_pretrained, so we name the HF repo directly and the local layout is
# $DIFFSYNTH_MODEL_BASE_PATH/<repo id>/<file>.
# ---------------------------------------------------------------------------------------
H3_REPO = "MiniMaxAI/MiniMax-H3"
COMFY_REPO = "Comfy-Org/MiniMax-H3"
ADAPTER_REPO = "DiffSynth-Studio/MiniMax-H3-TrainingAdapter"
DIT_PRUNED = f"{COMFY_REPO}:diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
DIT_FULL = f"{COMFY_REPO}:diffusion_models/minimax_h3_fl2va_bf16.safetensors"   # G0.2a only
TEXT_ENCODER = f"{H3_REPO}:FL2VA/text_encoder/model*.safetensors"
VIDEO_VAE = f"{H3_REPO}:FL2VA/video_vae/source/model.safetensors"
AUDIO_VAE = f"{H3_REPO}:FL2VA/audio_vae/model.safetensors"
PROCESSOR = f"{H3_REPO}:FL2VA/processor/"
ADAPTER = f"{ADAPTER_REPO}:model_for_comfy_dit.safetensors"                     # arm D only

# DiffSynth-Studio HEAD on 2026-10-09 (v2.1.9, #1734). Has --training_cfg_scale and
# --audio_loss_weight. Everything in patches/ is written against this tree.
DIFFSYNTH_SHA = "acf2ad29c65f436633f57beec1caaa1296760331"

LORA_TARGETS = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
LORA_RANK = 32

# ---------------------------------------------------------------------------------------
# Geometry and length
# ---------------------------------------------------------------------------------------
SIZE = 768
FPS = 24
BUCKETS = (22, 39, 56)            # the 17n+5 grid (MODEL_NOTES 2.3)
FLOOR = 22
# Every corpus mp4 carries ONE extra copy of its last frame after the N real ones.
# DiffSynth's loader (FrameSamplerByRateMixin, fix_frame_rate=True) takes
# floor(container duration x 24) as the available length, and imageio reports the
# duration to 2 decimals: a 56-frame file reads as 2.33 s -> 55 frames -> snapped down
# to 39, silently. N+1 frames always yields >= N. The loader reads exactly --num_frames,
# so the tail frame is never trained. PNG copies hold exactly N. (MODEL_NOTES 2.17)
LOADER_TAIL = 1


def latents_for(n):
    """Frame count on the 17n+5 grid -> latent frames (22 -> 7, 39 -> 12, 56 -> 17)."""
    assert n % 17 == 5, n
    return (n - 5) // 17 * 5 + 2


def frames_of_latent(i):
    """Frames covered by latent i. The H3 VAE cuts 17-frame clips FROM FRAME 0 (padding the
    last with copies of the final frame) and drops the last 3 tokens
    (MiniMaxH3VideoVAE.encode_temporal: clip_length 17, token_drop 3), so each 17-frame
    chunk gives tokens covering (1, 4, 4, 4, 4) frames and the 5-frame remainder of 17n+5
    sits at the END. f22: {0} {1-4} {5-8} {9-12} {13-16} {17} {18-21}. Latent 0 = frame 0
    only (G0.2b confirms on the box)."""
    chunk, k = divmod(i, 5)
    base = 17 * chunk
    if k == 0:
        return [base]
    start = base + 1 + 4 * (k - 1)
    return list(range(start, start + 4))


CHUNK_STARTS = (17, 34, 51)      # first frame of each 17-frame VAE clip after the first


# ---------------------------------------------------------------------------------------
# Characters, labels and caption clauses
# ---------------------------------------------------------------------------------------
CHARS = ("Pax", "Polly")
# copied (prep_v8.ANCHORS). Colour is in every caption (A10).
ANCHORS = {
    "Pax": "Pax, a short round blue penguin",
    "Polly": "Polly, a short round pink penguin",
}

# Action clauses: copied verbatim from prep_v8.MOTION / prep_v8.EXPRESSIONS (written from
# the footage frame by frame; see the comments there). `tag` is the label clause text.
MOTION = {
    "walking": {"tag": "walk cycle",
                "action": ("walking in place with a steady waddling gait, feet alternating, "
                           "flippers swinging gently at its sides, body bobbing softly")},
    "waving": {"tag": "waving gesture",
               "action": ("standing and waving one flipper, raising it up beside its head and "
                          "swinging it side to side, eyes closed in a happy smile")},
    "sitting": {"tag": "seated idle",
                "action": ("sitting on the ground with its legs stretched out in front, settling "
                           "into a calm seated rest, only a soft idle breathing motion")},
    "jumping": {"tag": "jump cycle",
                "action": ("crouching down and then hopping straight up, flippers spreading out "
                           "wide, feet tucking under as it leaves the ground, then landing and "
                           "settling back down")},
    # Untrained in expv2 (17 frames < the 22 floor). Start frames and `core` prompts only.
    "running": {"tag": "run cycle",
                "action": ("running in place with a quick bounding stride, one foot lifting high, "
                           "flippers swept back, body leaning forward into the run")},
}
EXPRESSIONS = {
    "happy": {"tag": "happy expression",
              "action": ("breaking into a big happy smile, beak opening into a wide joyful "
                         "grin, eyebrows lifting, cheeks lifting")},
    "surprised": {"tag": "surprised expression",
                  "action": ("eyes widening in surprise, brows shooting up, beak opening into a "
                             "small round gasp, body pulling back slightly")},
    "angry": {"tag": "angry expression",
              "action": ("scowling into an angry glare, brows dropping and pressing together, "
                         "beak set in a hard frown, shoulders squaring")},
    "confused": {"tag": "confused expression",
                 "action": ("its expression turning puzzled, one eyebrow arching up while the other "
                            "eye narrows, beak shifting to one side in a quizzical look, then "
                            "holding the look")},
    "crying": {"tag": "crying expression",
               "action": ("beginning to cry, hunching down as its flippers come up beside its "
                          "face, teardrops welling and running from both eyes, brows angling up "
                          "and beak turning down in a sob, then holding the cry")},
    "neutral": {"tag": "neutral expression",
                "action": ("standing still with a calm neutral expression, body held steady and "
                           "beak closed, eyes open apart from a slow blink")},
    "blushing": {"tag": "blushing expression",
                 "action": ("blushing shyly, rosy blush marks appearing on both cheeks, eyes "
                            "glancing up bashfully with the beak small and closed, body shifting "
                            "slightly, then holding the shy look")},
    "laughing": {"tag": "laughing expression",
                 "action": ("laughing out loud, eyes squeezing shut with mirth, beak opening in "
                            "an open-mouthed laugh, flippers coming up to its belly as its body "
                            "rocks back, then holding the laugh")},
}
LABELS = {**{k: {**v, "kind": "expression"} for k, v in EXPRESSIONS.items()},
          **{k: {**v, "kind": "motion"} for k, v in MOTION.items()}}
EXPR_LABELS = tuple(EXPRESSIONS)                                  # 8
MOTION_TRAINED = ("walking", "waving", "sitting", "jumping")      # 4
TRAINED_LABELS = EXPR_LABELS + MOTION_TRAINED                     # 12
ALL_LABELS = TRAINED_LABELS + ("running",)                        # 13 (core suite)
LONG_RUNG_LABELS = {39: ("angry", "laughing", "blushing", "neutral"), 56: ("neutral",)}


def kind_of(label):
    return LABELS[label]["kind"]


# Angle clauses: copied (prep_v8.FRONT / SIDE / QUARTER).
FRONT = "facing the camera directly, front view"
SIDE = {"L": "seen from its left side, profile view",
        "R": "seen from its right side, profile view"}
QUARTER = {
    1: "turned slightly to its {side}, three-quarter front view",
    2: "turned further to its {side}, wide three-quarter view",
    3: "turned strongly to its {side}, near-profile three-quarter view",
}
HOLDOUT_ANGLE = "QF2_R"           # fixed (B9); every label, both characters
CALIB_ANGLE = "FRONT"             # Tier-1 t calibration uses trained-angle clips (B4)


def angle_clause(slug):
    """'FRONT' | 'SIDE_L' | 'SIDE_R' | 'QF{1,2,3}_{L,R}' -> caption clause."""
    if slug == "FRONT":
        return FRONT
    if slug.startswith("SIDE_"):
        return SIDE[slug[-1]]
    if slug.startswith("QF") and slug[2].isdigit():
        return QUARTER[int(slug[2])].format(side="left" if slug.endswith("L") else "right")
    raise ValueError(f"no clause for angle {slug!r}")


# Grounds. Trained four: copied (prep_v8.BACKGROUNDS RGB). Unseen two: copied
# (eval_v8/common.UNSEEN_BG RGB). Never trained (B15).
TRAINED_GROUNDS = {
    "white": ("white", (255, 255, 255)),
    "blue": ("pastel blue", (198, 222, 241)),
    "peach": ("pastel peach", (247, 219, 205)),
    "mint": ("pastel mint", (206, 235, 219)),
}
UNSEEN_GROUNDS = {
    "lavender": ("pastel lavender", (222, 214, 240)),
    "skyblue": ("pastel sky-blue", (205, 232, 246)),
}
GROUNDS = {**TRAINED_GROUNDS, **UNSEEN_GROUNDS}

# Shot-size ladder, zoom-out only (A4). Caption word per zoom.
ZOOMS = {1.00: "close-up", 0.75: "medium", 0.55: "wide"}

# ---------------------------------------------------------------------------------------
# The caption template (PLAN 5.5). FROZEN once the corpus is built: training and every
# evaluation prompt come from this one function. The label clause is the only part that
# changes between labels (B12); Tier 1 swaps exactly that clause.
# ---------------------------------------------------------------------------------------
CAPTION_HEAD = ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
STYLE = ("2D cartoon animation in the Pudgy Penguins style, hand-drawn and animated on ones "
         "at 24 frames per second, thick clean black outlines, flat pastel colors, no shading.")
CAMERA = "Fixed camera, locked-off shot, the character stays the same size."
MAX_CAPTION_TOKENS = 220          # Qwen3-VL text tokens, asserted by prep


def label_clause(label, truncated=False):
    """'<action>; <tag>[, opening frames only]' - the only label-dependent text."""
    spec = LABELS[label]
    tail = ", opening frames only" if truncated else ""
    return f"{spec['action']}; {spec['tag']}{tail}"


def caption(char, label, angle, ground, zoom=1.00, truncated=False):
    """The one caption template. `ground` is a key of GROUNDS, `zoom` a key of ZOOMS."""
    desc = (f"[Shot 1] {STYLE} {ANCHORS[char]}, {label_clause(label, truncated)}. "
            f"{ZOOMS[zoom].capitalize()} shot, eye level, {angle_clause(angle)}. {CAMERA} "
            f"Plain flat {GROUNDS[ground][0]} background.")
    return (f"{CAPTION_HEAD}\n\n"
            f"integrated_multimodal_description: {desc}\n\n"
            f"overall_soundscape: Silence.\n\n"
            f"non_diegetic_music: None.")


def clip_id(char, label, angle, ground, zoom, n):
    """e.g. pax_angry_QF1_L__mint_z075_f22"""
    return f"{char.lower()}_{label}_{angle}__{ground}_z{int(round(zoom * 100)):03d}_f{n}"


# ---------------------------------------------------------------------------------------
# Inference (PLAN 7.1) and evaluation constants
# ---------------------------------------------------------------------------------------
INFERENCE = {"cfg_scale": 1.0, "flow_shift": 12.0, "audio_flow_shift": 3.0,
             "num_inference_steps": 50, "height": SIZE, "width": SIZE,
             "tiled": True, "tile_size": 256, "tile_overlap": 64, "rand_device": "cpu"}
SEEDS = (42, 43, 44)
PROBE_SEED = 42
# Tier 1 noise levels. sigma = noise fraction (1 = pure noise). DiffSynth's model_fn
# receives timestep = 1000*sigma and uses t = 1 - sigma internally (clean = 1).
TIER1_SIGMA_GRID = (0.25, 0.5, 0.75, 0.875, 0.9375)
TIER1_NOISE_DRAWS = 2
TIER1_NOISE_SEED = 1234
PARITY_SIGMAS = (0.25, 0.5, 0.875)

GRAD_ACCUM = 8


def opt_step(samples):
    """DiffSynth's step-N checkpoint names count SAMPLES (micro-batches), not optimizer
    steps (ModelLogger.on_step_end increments per dataloader item). opt step = N / 8."""
    return samples // GRAD_ACCUM


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)
