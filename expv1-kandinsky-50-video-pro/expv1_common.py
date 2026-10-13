"""expv1 shared constants: paths, labels, the frozen caption grammar (PLAN §5.5), raw-source
discovery and decoding.

Provenance (copied, never imported; PLAN §5): the label action clauses, the angle parser,
bare-quarter resolution, md5 de-duplication, the real-frame-gap probe and the torn-frame
check come from finetune/wan/prep_v8.py @ 4832f98. The caption *template* is expv1's own
(PLAN §5.5); only the action clauses are shared with v8, so the two models are asked for
the same thing in different wording (a stated §8 confound).

Lengths: Kandinsky's VAE keeps only 13, 17 and 8k+1 frames whole (MODEL_NOTES §2.8), so
every length here is one of those.
"""
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("EXPV1_WORK", "/workspace/expv1"))
DATA = Path(os.environ.get("PUDGY_DATA", "/workspace/Data"))
RAW = DATA / "raw"
MUSUBI = Path(os.environ.get("MUSUBI_DIR", "/workspace/musubi-tuner"))
MODELS = Path(os.environ.get("MODELS", "/workspace/models"))
DIT = MODELS / "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s/model/kandinsky5pro_i2v_sft_5s.safetensors"
VAE_DIR = MODELS / "hunyuanvideo-community/HunyuanVideo"     # the DIRECTORY (MODEL_NOTES §2.9)
QWEN = MODELS / "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP = MODELS / "openai/clip-vit-large-patch14"
TASK = "k5-pro-i2v-5s-sd"

EVAL = WORK / "eval"
PHASE0 = EVAL / "phase0"
REFS = WORK / "refs"                 # Tier-1 clip sets, the real-clip bank, start frames
V8_REF = Path(os.environ.get("V8_REF", "/workspace/v8_ref"))   # local copy of pudgy/v8/eval/

SIZE = 512
FPS = 24
CHARS = ("Pax", "Polly")
HOLDOUT_ANGLE = "QF2_R"
SEEN_ANGLE = "QF2_L"                 # the trained mirror angle: Tier-1's "seen" control

# Tier-1 (diffusion classifier) clip length per kind. Expression is f17, not v8's f21:
# Kandinsky's VAE truncates f21 (MODEL_NOTES §2.8), and happy/confused sources are only
# 21 frames long, so f25 does not exist for every label. f17 is inside v8's f21 window.
TIER1_LEN = {"motion": 13, "expression": 17}
# The window both models are compared at in the §8 head-to-head: the same as TIER1_LEN.
# v8's core clips (f13 motion / f21 expression) are trimmed to it.
H2H_WINDOW = {"motion": 13, "expression": 17}
T_GRID = (0.65, 0.85)                # PLAN §4 G0.3 / §7 Tier 1
K_DRAWS = 2

ANCHORS = {"Pax": "Pax, a short round blue penguin",
           "Polly": "Polly, a short round pink penguin"}
STYLE = "2D cartoon animation in the Pudgy Penguins style, thick clean black outlines, flat pastel colors."
CAMERA = "Fixed camera, locked-off shot, the character stays the same size."
SHOTS = {1.00: "Close-up shot", 0.75: "Medium shot", 0.55: "Wide shot"}
GROUNDS = {"white": (255, 255, 255), "pastel blue": (198, 222, 241), "peach": (247, 219, 205),
           "mint": (206, 235, 219),
           # never trained (PLAN §5.3)
           "lavender": (222, 214, 240), "sky-blue": (205, 232, 246)}
NEG = ("photorealistic, 3D render, realistic lighting, gradient shading, soft shadows, textured background, "
       "blurry, distorted, deformed, low quality, worst quality")

# --- labels (action clauses copied from prep_v8.py) -------------------------------------------
# `native`: the shortest legal length holding the label's whole arc in the delivered footage
# (measured by prep_expv1: active length); laughing's arc (38-39 frames) does not fit before
# the sources end, so every laughing clip is 'opening frames only' (native 41 > any rung).
MOTION = {
    "walking": {"native": 13, "label": "walk cycle",
                "action": ("walking in place with a steady waddling gait, feet alternating, "
                           "flippers swinging gently at its sides, body bobbing softly")},
    "running": {"native": 17, "label": "run cycle",
                "action": ("running in place with a quick bounding stride, one foot lifting high, "
                           "flippers swept back, body leaning forward into the run")},
    "waving": {"native": 25, "label": "waving gesture",
               "action": ("standing and waving one flipper, raising it up beside its head and "
                          "swinging it side to side, eyes closed in a happy smile")},
    "sitting": {"native": 25, "label": "seated idle",
                "action": ("sitting on the ground with its legs stretched out in front, settling "
                           "into a calm seated rest, only a soft idle breathing motion")},
    "jumping": {"native": 33, "label": "jump cycle",
                "action": ("crouching down and then hopping straight up, flippers spreading out "
                           "wide, feet tucking under as it leaves the ground, then landing and "
                           "settling back down")},
}
EXPRESSIONS = {
    "happy": {"native": 17, "label": "happy expression",
              "action": ("breaking into a big happy smile, beak opening into a wide joyful "
                         "grin, eyebrows lifting, cheeks lifting")},
    "surprised": {"native": 25, "label": "surprised expression",
                  "action": ("eyes widening in surprise, brows shooting up, beak opening into a "
                             "small round gasp, body pulling back slightly")},
    "angry": {"native": 25, "label": "angry expression",
              "action": ("scowling into an angry glare, brows dropping and pressing together, "
                         "beak set in a hard frown, shoulders squaring")},
    "confused": {"native": 17, "label": "confused expression",
                 "action": ("its expression turning puzzled, one eyebrow arching up while the other "
                            "eye narrows, beak shifting to one side in a quizzical look, then "
                            "holding the look")},
    "crying": {"native": 25, "label": "crying expression",
               "action": ("beginning to cry, hunching down as its flippers come up beside its "
                          "face, teardrops welling and running from both eyes, brows angling up "
                          "and beak turning down in a sob, then holding the cry")},
    "neutral": {"native": 57, "label": "neutral expression",
                "action": ("standing still with a calm neutral expression, body held steady and "
                           "beak closed, eyes open apart from a slow blink")},
    "blushing": {"native": 33, "label": "blushing expression",
                 "action": ("blushing shyly, rosy blush marks appearing on both cheeks, eyes "
                            "glancing up bashfully with the beak small and closed, body shifting "
                            "slightly, then holding the shy look")},
    "laughing": {"native": 41, "label": "laughing expression",
                 "action": ("laughing out loud, eyes squeezing shut with mirth, beak opening in "
                            "an open-mouthed laugh, flippers coming up to its belly as its body "
                            "rocks back, then holding the laugh")},
}
EXPR_LABELS = tuple(EXPRESSIONS)
MOTION_LABELS = tuple(MOTION)
LABELS = EXPR_LABELS + MOTION_LABELS


def kind_of(label):
    return "motion" if label in MOTION else "expression"


def recipe(label):
    return (MOTION if label in MOTION else EXPRESSIONS)[label]


def k5_len_ok(n):
    """Lengths Kandinsky's VAE returns whole (MODEL_NOTES §2.8)."""
    return n in (1, 5, 9, 13, 17) or (n > 17 and (n - 1) % 8 == 0)


def latent_frames(n):
    return (n - 1) // 4 + 1


# --- angles (copied from prep_v8.py) ----------------------------------------------------------
SIDE = {"L": "seen from its left side, profile view", "R": "seen from its right side, profile view"}
QUARTER = {1: "turned slightly to its {side}, three-quarter front view",
           2: "turned further to its {side}, wide three-quarter view",
           3: "turned strongly to its {side}, near-profile three-quarter view"}
FRONT = "facing the camera directly, front view"
NOISE = ({"PAX", "POLLY", "EXPRESSION", "MOTION", "SURPRISED", "SURPRISE"}
         | {n.upper() for n in MOTION} | {n.upper() for n in EXPRESSIONS})


def clause_for_angle(slug):
    base = slug.split("_t")[0]
    if base == "FRONT":
        return FRONT
    if base.startswith("SIDE_"):
        return SIDE[base[-1]]
    m = re.fullmatch(r"QF([123])_([LR])", base)
    if m:
        return QUARTER[int(m.group(1))].format(side="left" if m.group(2) == "L" else "right")
    raise ValueError(f"no clause for angle {slug!r}")


def caption(char, label, angle, nframes, zoom=1.00, ground="white", native=None):
    """PLAN §5.5, frozen. The label clause is the only part that changes between labels.
    `native`: the source's own full length (prep); eval prompts use the label's recipe length."""
    r = recipe(label)
    lab = r["label"] if nframes >= (native or r["native"]) else f"{r['label']}, opening frames only"
    if "pink" in r["action"] and char == "Pax":
        raise AssertionError("'pink' in a Pax caption")
    return (f"{STYLE}\n{ANCHORS[char]}, {r['action']}; {lab}.\n"
            f"{SHOTS[zoom]}, eye level, {clause_for_angle(angle)}. {CAMERA}\n"
            f"Plain {ground} background.")


def parse_angle(stem):
    toks = [t for t in re.split(r"[_\-]", stem.upper()) if t and t not in NOISE]
    if not toks:
        return None
    if toks[0] in ("FRONT", "FR"):
        return "FRONT"
    if toks[0] == "SIDE" and len(toks) > 1 and toks[1][0] in "LR":
        return f"SIDE_{toks[1][0]}"
    if toks[0] in ("RIGHT", "LEFT"):
        return f"SIDE_{toks[0][0]}"
    if toks[0].startswith("QF"):
        joined = "".join(toks)
        m = re.search(r"([LR])", joined)
        if not m:
            return None
        digits = re.findall(r"\d", joined)
        if re.match(r"QF\d", toks[0]):
            return f"QF{min(max(int(digits[0]), 1), 3)}_{m.group(1)}"
        return f"QF?_{m.group(1)}"
    return None


def _resolve_bare_quarters(jobs):
    """prep_v8.resolve_bare_quarters: bare QF_<side>[n] names take the lowest free degrees
    in sorted filename order (the client's suffix means index in some folders, offset in
    others; verified against the footage in v8)."""
    groups, taken = collections.defaultdict(list), collections.defaultdict(set)
    for j in jobs:
        if not j["angle"].startswith("QF"):
            continue
        key = (j["char"], j["label"], j["angle"][-1])
        if j["angle"].startswith("QF?"):
            groups[key].append(j)
        else:
            taken[key].add(int(j["angle"][2]))
    for key, g in sorted(groups.items()):
        g.sort(key=lambda x: Path(x["src"]).name)
        free = [d for d in (1, 2, 3) if d not in taken[key]]
        if len(g) > len(free):
            sys.exit(f"{key}: {len(g)} bare quarter files but only {len(free)} free degrees")
        for j, deg in zip(g, free):
            j["angle"] = f"QF{deg}_{key[2]}"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover():
    """Every single-character source with a caption recipe, md5-deduplicated, iteration_3
    first (canonical names). Returns (jobs, skipped). Same rules as prep_v8.discover()."""
    jobs, skipped, seen = [], [], {}
    chars = {"PAX": "Pax", "POLLY": "Polly"}

    def add(src, char, label):
        slug = parse_angle(src.stem)
        if not slug:
            skipped.append((str(src), "unparseable angle")); return
        h = _md5(src)
        if h in seen:
            return
        seen[h] = str(src)
        jobs.append({"src": str(src), "md5": h, "char": char, "label": label,
                     "kind": kind_of(label), "angle": slug})

    it3 = RAW / "iteration_3/03_expression_clips"
    if it3.exists():
        for cd in sorted(p for p in it3.iterdir() if p.is_dir()):
            if cd.name not in ANCHORS:
                continue
            for ed in sorted(p for p in cd.iterdir() if p.is_dir()):
                if ed.name in EXPRESSIONS:
                    for c in sorted(ed.glob("*.mov")):
                        add(c, cd.name, ed.name)
    for root in (RAW / "iteration_4/LSLTTT-Project", RAW / "iteration_5"):
        ex = root / "EXPRESSIONS"
        if ex.exists():
            for ld in sorted(p for p in ex.iterdir() if p.is_dir()):
                key = {"SURPRISE": "surprised"}.get(ld.name, ld.name.lower())
                cds = [p for p in ld.iterdir() if p.is_dir()]
                if not cds or key not in EXPRESSIONS:
                    continue
                for cd in sorted(cds):
                    char = chars.get(cd.name.upper())
                    if not char:
                        continue
                    for c in sorted(cd.glob("*.mov")):
                        named = next((v for k, v in chars.items() if c.stem.upper().startswith(k)), None)
                        add(c, named or char, key)
        mo = root / "MOTION_CLIPS"
        if mo.exists():
            for ad in sorted(p for p in mo.iterdir() if p.is_dir()):
                key = ad.name.upper().replace("MOTION_", "").lower()
                if key not in MOTION:
                    continue
                for cd in sorted(p for p in ad.iterdir() if p.is_dir()):
                    char = chars.get(cd.name.upper())
                    if not char:
                        continue
                    for c in sorted(cd.glob("*.mov")):
                        other = [a for a in MOTION if a != key and a.upper() in c.stem.upper()]
                        if other:
                            skipped.append((str(c), f"folder says {key}, filename says {other[0]}"))
                            continue
                        add(c, char, key)
    _resolve_bare_quarters(jobs)
    groups = collections.defaultdict(list)
    for j in jobs:
        groups[(j["char"], j["label"], j["angle"])].append(j)
    for g in groups.values():
        if len(g) > 1:
            g.sort(key=lambda x: Path(x["src"]).name)
            for i, j in enumerate(g, 1):
                j["angle"] = f"{j['angle']}_t{i}"
    return jobs, skipped


def real_frame_gap(path):
    """Index of the first frame the stream does not contain (ffmpeg pads gaps with
    duplicates), or None. prep_v8.real_frame_gap."""
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    idx = sorted({int(round(float(t) * FPS)) for t in out if t.strip()})
    for i, k in enumerate(idx):
        if k != i:
            return i
    return len(idx)            # no gap: the first missing index is the end of the stream


def read_rgba(path, nframes, size=SIZE):
    """ProRes 4444 -> (n, size, size, 4) uint8 in ONE Lanczos resample from 1080."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"scale={size}:{size}:flags=lanczos", "-frames:v", str(nframes),
                          "-pix_fmt", "rgba", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 4)
    if arr.shape[0] < nframes:
        raise RuntimeError(f"{Path(path).name}: got {arr.shape[0]} frames, need {nframes}")
    return arr[:nframes]


TORN_FRAC = 0.6
DECODE_TRIES = 24


def decode_checked(path, n, size=SIZE):
    """Decode, retrying the non-deterministic ProRes tears (prep_v8.decode_checked)."""
    last = None
    for _ in range(DECODE_TRIES):
        fr = read_rgba(path, n, size)
        cover = (fr[..., 3] > 8).reshape(n, -1).mean(1)
        med = float(np.median(cover))
        torn = [int(i) for i in np.where(cover < TORN_FRAC * med)[0]] if med > 0 else list(range(n))
        if not torn:
            return fr
        last = torn
    raise RuntimeError(f"{Path(path).name}: torn frames {last} after {DECODE_TRIES} decodes")


def composite(frames, rgb):
    a = frames[..., 3:4].astype(np.float32) / 255.0
    return (frames[..., :3].astype(np.float32) * a + np.array(rgb, np.float32) * (1 - a)).round().astype(np.uint8)


def write_mp4(frames, dest, lossless=True):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    codec = ["-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast"] if lossless else \
        ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                        "-r", str(FPS), "-i", "-", *codec, "-an", str(dest)], input=frames.tobytes(),
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-400:])


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
