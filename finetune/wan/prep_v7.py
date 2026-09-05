#!/usr/bin/env python3
"""
v7 training-set prep — motion primitives + a 5th expression, from the iteration_4 delivery.

Successor to prep_expressions_v6.py. Two things forced a new script rather than a flag:

  1. iteration_4 is shaped `<CATEGORY>/<ACTION|EXPRESSION>/<CHAR>/`, the inverse of
     iteration_3's `<Char>/<emotion>/`. v6's discover() cannot read it.
  2. It is a whole-folder RE-DROP: 70 of its 180 files are byte-identical re-sends of all
     69 iteration_3 clips. Reading both trees naively trains those clips twice, so
     everything here is de-duplicated by **md5**, not by path.

It emits TWO datasets, because v5 settled that the target expert depends on what the clip
teaches, and mixing them in one LoRA re-creates the failure v5 diagnosed:

  v7_motion_*       5 in-place actions   -> HIGH-noise expert (global/temporal motion)
  v7_expressions_*  5 facial expressions -> LOW-noise expert  (identity/texture)

v5 §4.5-4.6 proved expression belongs on low-noise: continue-training high-noise gave
SSIM 0.9692 between opposite prompts (prompt ignored) vs 0.9340 on low-noise (prompt has
effect). v2's G1 finding is the mechanism — high-noise carries global composition and
motion, low-noise carries identity and fine detail. Motion is therefore the one kind of
content that *does* belong on the high-noise expert; it is not a re-litigation of v5.

Source clips are ProRes 4444 with a real alpha channel (yuva444p12le, 1080x1080, 24 fps),
same as iteration_3. Same two consequences, unchanged from v5/v6:

  1. A naive decode composites onto BLACK. We composite deliberately onto flat pastel
     grounds, so nothing teaches a black void.
  2. Alpha lets one performance be composited onto N backgrounds, which is what buys
     background-invariance rather than "this action on this background". v5 measured
     2/255 drift on a ground never trained.

Output: 1024x1024 (NOT the source 1080 - Wan's 8x VAE + 2x2 patchify needs an even latent
side, and 1080/8 = 135 is odd), silent mp4 + captions + jsonl + musubi config + manifest.
"""
import argparse, collections, hashlib, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 1024
FPS = 24

DATA = Path("/Users/rahul/Documents/Projects/Saksham/Pudgy/Data")
SRC_IT3 = DATA / "raw/iteration_3/03_expression_clips"
SRC_IT4 = DATA / "raw/iteration_4/LSLTTT-Project"
OUT_ROOT = DATA / "processed"

# Shot-size ladder — carried over from v6, and re-verified against iteration_4's alpha.
#
# ZOOM-OUT ONLY, measured not preferred. Across the 110 new clips the character's alpha
# bbox is a median 87% of frame height and a MAXIMUM of 94% (jumping, at the top of the
# hop); 6 clips touch the top edge. The largest centre-anchored zoom-IN that keeps the
# character whole is therefore ~1.06x - tighter even than v6's 1.15x, and visually
# indistinguishable from 1.0x. A real close-up would have to cut the body.
ZOOMS = {
    1.00: "static close-up shot",
    0.75: "static medium shot",
    0.55: "static wide shot",
}

# Flat pastel grounds — identical set to v5/v6, so the runs stay comparable and the two
# reserved eval grounds (lavender, sky-blue) stay unseen.
BACKGROUNDS = {
    "white": ("plain white studio background", (255, 255, 255)),
    "blue":  ("plain pastel blue background",  (198, 222, 241)),
    "peach": ("plain pastel peach background", (247, 219, 205)),
    "mint":  ("plain pastel mint background",  (206, 235, 219)),
}

# Identity anchors, verbatim from the v1/v2 caption convention and unchanged through
# v5/v6. Deliberately NOT v4's rare token (pxngn0): this line continue-trains the v2
# goldens, so captions must sit in the distribution they already saw.
#
# Colour is non-negotiable. v4 rendered Polly BLUE because 0 of its 33 solo-Polly captions
# contained the word "pink" (v4 §5.3). Every caption here names the colour.
ANCHORS = {
    "Pax":   "Pax, a short round blue penguin",
    "Polly": "Polly, a short round pink penguin",
}
STYLE = ("A 2D cartoon animation in the Pudgy Penguins style, with thick clean "
         "black outlines and flat pastel colors, showing ")

# ---------------------------------------------------------------------------
# Labels. `frames` is the largest musubi-legal (4k+1) count that fits the delivered
# length. Action clauses were written from the footage, frame by frame - not from the
# folder name. Two would have been wrong from the name alone:
#   - SITTING is a seated IDLE. The character is already seated in frame 0; it never
#     lowers itself. "sitting down" would mislabel every clip.
#   - JUMPING is a full hop cycle (crouch -> flippers out -> airborne, feet tucked,
#     shadow shrinking -> land -> settle), not a standing bounce.
# ---------------------------------------------------------------------------
MOTION = {
    "walking": {
        "frames": 13,   # source 16
        "loops": True,
        "action": ("walking in place with a steady waddling gait, feet alternating, "
                   "flippers swinging gently at its sides, body bobbing softly"),
        "label": "walk cycle",
    },
    "running": {
        "frames": 17,   # source 17
        "loops": False,
        "action": ("running in place with a quick bounding stride, one foot lifting high, "
                   "flippers swept back, body leaning forward into the run"),
        "label": "run cycle",
    },
    "waving": {
        "frames": 25,   # source 26
        "loops": True,
        "action": ("standing and waving one flipper, raising it up beside its head and "
                   "swinging it side to side, eyes closed in a happy smile"),
        "label": "waving gesture",
    },
    "sitting": {
        "frames": 25,   # source 28
        "loops": True,
        "action": ("sitting on the ground with its legs stretched out in front, settling "
                   "into a calm seated rest, only a soft idle breathing motion"),
        "label": "seated idle",
    },
    # PENDING DELIVERY (requested in Client_Data_Request_Round4 §3). The client has never
    # delivered an idle: `iteration_3/02_motion_clips/*/standing_idle/` is an empty scaffold
    # WE created from the Round 3 ask. Until it arrives this label simply finds no clips and
    # is skipped - motion has no counter-class for "stop", and gate G-N is expected to fail.
    #
    # `frames: None` means DERIVE from the delivered footage. It is deliberately not a
    # guess: Round 4 asks for 3-5 s, which at 24 fps is 72-120 frames - far longer than
    # anything delivered so far (max 60), and the single most expensive bucket in the set.
    "standing_idle": {
        # 33, deliberately capped rather than neutral's natural 57. At 57 the f37/f57
        # buckets would contain ONLY idle, re-creating the length<->label confound one
        # level up: ask for 57 frames of motion and you would always get idle. At 33 the
        # ladder is 13/21/33, all buckets that already hold other motion labels.
        # When a REAL breathing idle is delivered, set this back to None to derive from it.
        "frames": 33,
        "loops": True,
        # Honest to the CURRENT source, which is the `neutral` expression footage reused
        # (see DERIVE_IDLE_FROM_NEUTRAL). That footage does NOT breathe - the body is
        # frozen (measured MAE 0.028-0.064 across all 18 sources) and the only motion is a
        # blink. Saying "chest rising and falling" would be a caption that contradicts the
        # pixels, which is the exact class of error this pipeline exists to avoid.
        # Replace this clause with a breathing description ONLY when real idle footage
        # lands - see Client_Data_Request_Round4 section 3.
        "action": ("standing still and holding its position, body steady, flippers at its "
                   "sides, not performing any action, with an occasional blink"),
        "label": "standing idle",
    },
    "jumping": {
        "frames": 33,   # source 33
        "loops": True,
        "action": ("crouching down and then hopping straight up, flippers spreading out "
                   "wide, feet tucking under as it leaves the ground, then landing and "
                   "settling back down"),
        "label": "jump cycle",
    },
}

EXPRESSIONS = {
    "happy": {
        "frames": 21,   # source 21
        "action": ("breaking into a big happy smile, beak opening into a wide joyful "
                   "grin, eyebrows lifting, cheeks lifting"),
        "label": "happy expression",
    },
    "surprised": {
        "frames": 29,   # source 30
        "action": ("eyes widening in surprise, brows shooting up, beak opening into a "
                   "small round gasp, body pulling back slightly"),
        "label": "surprised expression",
    },
    "angry": {
        "frames": 37,   # source 40
        "action": ("scowling into an angry glare, brows dropping and pressing together, "
                   "beak set in a hard frown, shoulders squaring"),
        "label": "angry expression",
    },
    # NEW in the 2026-09-05 batch. Both clauses written from the footage, frame by frame.
    "confused": {
        "frames": 21,   # source 21
        "action": ("its expression turning puzzled, one eyebrow arching up while the other "
                   "eye narrows, beak shifting to one side in a quizzical look, then "
                   "holding the look"),
        "label": "confused expression",
    },
    "crying": {
        "frames": 29,   # source 31
        "action": ("beginning to cry, hunching down as its flippers come up beside its "
                   "face, teardrops welling and running from both eyes, brows angling up "
                   "and beak turning down in a sob, then holding the cry"),
        "label": "crying expression",
    },
    "neutral": {
        "frames": 57,   # source 60
        # Rewritten 2026-09-05 from the footage. The previous wording ("eyes open and
        # steady ... only a soft idle settle") was wrong on both counts: measured across
        # all 18 sources the BODY is essentially frozen (MAE 0.028-0.064) - it does not
        # settle - and the only real motion is a blink, in the face (head MAE 0.139-0.255,
        # 4-6x the body). Detected in 14/18; the 4 misses are SIDE_L/SIDE_R profiles where
        # the eye is barely visible edge-on.
        "action": ("standing still with a calm neutral expression, body held steady and "
                   "beak closed, eyes open apart from a slow blink"),
        "label": "neutral expression",
    },
    "laughing": {
        "frames": 37,   # source 39 - NEW in iteration_4
        "action": ("laughing out loud, eyes squeezing shut with mirth, beak opening in "
                   "an open-mouthed laugh, flippers coming up to its belly as its body "
                   "rocks back, then holding the laugh"),
        "label": "laughing expression",
    },
}

# The common short bucket that breaks the length<->label bijection. See BUCKET NOTE below.
COMMON = {"motion": 13, "expression": 21}

# Frame lengths emitted per label: a ladder from the kind's floor up to the label's
# natural (original) length. See buckets_for().
LADDER_STEPS = 3

# Reuse the `neutral` EXPRESSION footage as a motion counter-class ("standing idle").
#
# ⚠️ This is a DERIVED class, not a delivered one. The client has never shipped an idle;
# `neutral` is their expression label and this reuses the same footage under a second,
# also-true description (the character is standing and not acting). Two consequences to
# keep in view:
#   1. The footage is a FROZEN stand-still, not a living idle, so it teaches "stop" as
#      "freeze". Gate G-N should be read with that in mind.
#   2. The same source therefore appears in BOTH sets - as `neutral expression` on the
#      low-noise expert and as `standing idle` on the high-noise one. That is coherent
#      (separate LoRAs, separate experts, both captions true of the pixels) but it must be
#      stated wherever the set is described, or it reads as data we do not have.
# Set False to ship motion without a counter-class.
DERIVE_IDLE_FROM_NEUTRAL = True

# Which Wan2.2 expert each kind continue-trains. Settled by the v5 A/B, see module docstring.
EXPERT = {"motion": "high-noise", "expression": "low-noise"}

# One delivery, one folder: both kinds share clips/ on the box too, so the whole v7 set
# arrives with a single `az` pull. Only the latent cache is split per kind, because the two
# are separate training runs.
WS_CLIPS = "/workspace/data_v7/clips"
WS_CACHE = "/workspace/wan_cache/latents_v7"

SIDE = {"L": "seen from its left side, profile view",
        "R": "seen from its right side, profile view"}
QUARTER = {
    1: "turned slightly to its {side}, three-quarter front view",
    2: "turned further to its {side}, wide three-quarter view",
    3: "turned strongly to its {side}, near-profile three-quarter view",
}
FRONT = "facing the camera directly, front view"

# Tokens that carry no angle information. Every label name is noise for angle parsing.
NOISE = ({"PAX", "POLLY", "EXPRESSION", "MOTION", "SURPRISED"}
         | {n.upper() for n in MOTION} | {n.upper() for n in EXPRESSIONS}
         | {"SURPRISE"})


def parse_angle(stem):
    """Filename -> (slug, caption clause).

    The deliveries use FIVE angle vocabularies now: FRONT/FR, SIDE_L/Right, QF_L, QF1_L,
    QF_L2. Normalise them all here - nothing under raw/ is ever renamed, so this is the
    only place the client's inconsistency is absorbed.
    """
    toks = [t for t in re.split(r"[_\-]", stem.upper()) if t and t not in NOISE]
    if not toks:
        return None, None
    if toks[0] in ("FRONT", "FR"):
        return "FRONT", FRONT
    if toks[0] == "SIDE" and len(toks) > 1 and toks[1][0] in "LR":
        d = toks[1][0]
        return f"SIDE_{d}", SIDE[d]
    if toks[0] in ("RIGHT", "LEFT"):                    # v5's odd "Right.mov"
        d = toks[0][0]
        return f"SIDE_{d}", SIDE[d]
    if toks[0].startswith("QF"):
        joined = "".join(toks)
        m = re.search(r"([LR])", joined)
        if not m:
            return None, None
        d = m.group(1)
        digits = re.findall(r"\d", joined)
        # Two incompatible client conventions, which the digit alone cannot tell apart:
        #   EXPLICIT  QF1_R / QF2_R / QF3_R   - digit BEFORE the side letter IS the angle
        #   BARE      QF_R / QF_R1 / QF_R2    - Pax neutral: the suffix is an OFFSET (2,3)
        #             QF_L / QF_L2 / QF_L3    - Pax neutral & Polly: the suffix IS the index
        # Verified against the footage: Pax-neutral QF_R1 is the same angle as angry QF2_R,
        # and QF_R2 matches QF3_R. So a bare name's digit is not trustworthy - bare names
        # are re-indexed by sorted order in resolve_bare_quarters(). Only marked here.
        explicit = bool(re.match(r"QF\d", toks[0]))
        if explicit:
            deg = min(max(int(digits[0]), 1), 3)
            side = "left" if d == "L" else "right"
            return f"QF{deg}_{d}", QUARTER[deg].format(side=side)
        return f"QF?_{d}", None          # degree resolved later, by order
    return None, None


def resolve_bare_quarters(jobs):
    """Assign a degree to bare `QF_<side>[n]` names from their order within the group.

    The client's suffix means "index" in some folders and "offset" in others, so ordering is
    the only rule that survives both. Explicit `QF<n>_<side>` names keep their stated degree;
    the bare ones take the lowest degrees still free, in sorted filename order.
    """
    groups = collections.defaultdict(list)
    taken = collections.defaultdict(set)
    for j in jobs:
        if not j["angle"].startswith("QF"):
            continue
        key = (j["char"], j["label"], j["angle"][-1])
        if j["angle"].startswith("QF?"):
            groups[key].append(j)
        else:
            taken[key].add(int(j["angle"][2]))
    fixed = []
    for key, g in sorted(groups.items()):
        g.sort(key=lambda x: x["src"].name)
        free = [d for d in (1, 2, 3) if d not in taken[key]]
        if len(g) > len(free):
            sys.exit(f"{key}: {len(g)} bare quarter files but only {len(free)} free degrees")
        for j, deg in zip(g, free):
            side_l = key[2]
            fixed.append((j["src"].name, f"QF{deg}_{side_l}"))
            j["angle"] = f"QF{deg}_{side_l}"
            j["clause"] = QUARTER[deg].format(side="left" if side_l == "L" else "right")
    return fixed


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover():
    """Every source clip we know how to caption, de-duplicated by md5.

    Expressions come from iteration_3 FIRST, so the canonical filenames win and
    iteration_4's re-drop of the same bytes is reported as a duplicate rather than
    silently doubling the set. `laughing` exists only in iteration_4.

    Nothing is dropped silently: duplicates, unknown labels, unparseable angles and
    folder/filename disagreements are all reported.
    """
    jobs, skipped, dupes = [], [], []
    seen = {}

    def add(src, char, label, kind):
        slug, clause = parse_angle(src.stem)
        if not slug:
            skipped.append((str(src), "unparseable angle")); return
        h = md5(src)
        if h in seen:
            dupes.append((str(src), seen[h])); return
        seen[h] = str(src)
        jobs.append({"src": src, "md5": h, "char": char, "label": label,
                     "kind": kind, "angle": slug, "clause": clause})

    # --- expressions: iteration_3 (canonical) -------------------------------
    if SRC_IT3.exists():
        for char_dir in sorted(p for p in SRC_IT3.iterdir() if p.is_dir()):
            if char_dir.name not in ANCHORS:
                skipped.append((str(char_dir), "unknown character")); continue
            for emo_dir in sorted(p for p in char_dir.iterdir() if p.is_dir()):
                emo = emo_dir.name
                clips = sorted(emo_dir.glob("*.mov"))
                if emo not in EXPRESSIONS:
                    if clips:
                        skipped.append((f"iteration_3/{char_dir.name}/{emo}",
                                        f"no caption recipe ({len(clips)} clips)"))
                    continue
                for c in clips:
                    add(c, char_dir.name, emo, "expression")

    # --- iteration_4: motion + the new expression --------------------------
    CHARS = {"PAX": "Pax", "POLLY": "Polly"}
    it4_expr = SRC_IT4 / "EXPRESSIONS"
    if it4_expr.exists():
        for lab_dir in sorted(p for p in it4_expr.iterdir() if p.is_dir()):
            # folder names are upper-case and sometimes differ from our key
            key = {"SURPRISE": "surprised"}.get(lab_dir.name, lab_dir.name.lower())
            char_dirs = [p for p in lab_dir.iterdir() if p.is_dir()]
            if not char_dirs:                       # e.g. TURNAROUND (a single file)
                n = len(list(lab_dir.glob("*.mov")))
                if n:
                    skipped.append((f"iteration_4/EXPRESSIONS/{lab_dir.name}",
                                    f"identity reference, not a performance ({n} clips)"))
                continue
            if key not in EXPRESSIONS:
                skipped.append((f"iteration_4/EXPRESSIONS/{lab_dir.name}",
                                "no caption recipe")); continue
            for cd in sorted(char_dirs):
                char = CHARS.get(cd.name)
                if not char:
                    skipped.append((str(cd), "unknown character")); continue
                for c in sorted(cd.glob("*.mov")):
                    # A file whose NAME says a different character than its folder is
                    # misfiled (HAPPY/POLLY/PAX_HAPPY_SIDE_R.mov). Trust the name.
                    named = next((v for k, v in CHARS.items()
                                  if c.stem.upper().startswith(k)), None)
                    add(c, named or char, key, "expression")

    it4_motion = SRC_IT4 / "MOTION_CLIPS"
    if it4_motion.exists():
        for act_dir in sorted(p for p in it4_motion.iterdir() if p.is_dir()):
            key = act_dir.name.upper().replace("MOTION_", "").lower()
            # The client's folder name for the idle is not known yet; accept the likely ones.
            key = {"idle": "standing_idle", "stand": "standing_idle",
                   "standing": "standing_idle", "standingidle": "standing_idle",
                   "standing_still": "standing_idle"}.get(key, key)
            if key not in MOTION:
                skipped.append((f"iteration_4/MOTION_CLIPS/{act_dir.name}",
                                "no caption recipe")); continue
            for cd in sorted(p for p in act_dir.iterdir() if p.is_dir()):
                char = CHARS.get(cd.name)
                if not char:
                    skipped.append((str(cd), "unknown character")); continue
                for c in sorted(cd.glob("*.mov")):
                    # The folder says WALKING; three files inside it are named
                    # PAX_MOTION_WAVING_*. They are 16-frame walking-length renders and
                    # are NOT the real waving clips, so the filename is wrong - but we
                    # cannot know which angle they really are, and guessing would put a
                    # false angle clause in the caption. Skip, loudly.
                    other = [a for a in MOTION
                             if a != key and a.upper() in c.stem.upper()]
                    if other:
                        skipped.append((str(c.relative_to(DATA)),
                                        f"folder says {key}, filename says {other[0]} "
                                        f"- true angle unknowable, needs client answer"))
                        continue
                    add(c, char, key, "motion")

    if DERIVE_IDLE_FROM_NEUTRAL:
        derived = []
        for j in list(jobs):
            if j["kind"] == "expression" and j["label"] == "neutral":
                d = dict(j)
                d["kind"], d["label"] = "motion", "standing_idle"
                d["derived_from"] = "neutral"
                derived.append(d)
        jobs += derived
        if derived:
            print(f"derived {len(derived)} motion 'standing idle' sources from the "
                  f"`neutral` expression footage (DERIVE_IDLE_FROM_NEUTRAL)")

    rebased = resolve_bare_quarters(jobs)

    # Two takes can legitimately parse to the same angle. Sharing the caption is right;
    # sharing the output filename is not, or one take silently overwrites the other.
    collisions, groups = [], {}
    for j in jobs:
        groups.setdefault((j["char"], j["label"], j["angle"]), []).append(j)
    for (char, lab, slug), g in groups.items():
        if len(g) > 1:
            g.sort(key=lambda x: x["src"].name)
            for i, j in enumerate(g, 1):
                j["angle"] = f"{slug}_t{i}"
            collisions.append((f"{char}/{lab}/{slug}", [x["src"].name for x in g]))
    return jobs, skipped, dupes, collisions, rebased


def recipe(job):
    return (MOTION if job["kind"] == "motion" else EXPRESSIONS)[job["label"]]


def buckets_for(job):
    """The frame lengths this source is emitted at.

    BUCKET NOTE - this is the v6 mitigation that was recommended but never implemented.

    musubi buckets by frame count, so a label that appears at exactly one length makes
    sequence length a perfectly predictive shortcut for that label. v6 section 4.1 spelled
    this out and proposed a fix; the run never happened, so it was never applied. It is
    applied here, because v5 proved this model latches onto the most predictive available
    cue (all 28 v5 clips began from ~one frame, and that frame became a deterministic
    "happy" trigger - motion from it collapsed to 0.8 px).

    Every label is therefore ALSO emitted at the shortest common length for its kind, so
    that bucket alone carries all five labels and length predicts nothing.

    DURATION LADDER. The lengths are a ladder, not a pair, because duration is a training
    axis in its own right: at one or two lengths per label the model has little basis for
    generating anything in between. The ladder runs from the kind's floor up to the label's
    NATURAL length, evenly spaced over the legal 4k+1 values.

    **The original duration is always the top rung** - the longest bucket for every label is
    exactly what the client shot, and every shorter rung is a head truncation of that same
    footage. No frames are invented: nothing is loop-tiled or time-stretched. The four
    verified-cyclic actions *could* legitimately be wrapped to longer clips (a cycle has no
    arc to mislabel), but that would manufacture footage the client never delivered, so it
    is deliberately not done.

    The floor still carries every label of its kind, so the de-confounding above is
    preserved - now as a property of the ladder rather than a special case.
    """
    nat = recipe(job)["frames"]
    floor = COMMON[job["kind"]]
    if nat <= floor:
        return [nat]
    opts = [f for f in range(floor, nat + 1) if (f - 1) % 4 == 0]
    if len(opts) <= LADDER_STEPS:
        return opts
    idx = [round(i * (len(opts) - 1) / (LADDER_STEPS - 1)) for i in range(LADDER_STEPS)]
    return sorted({opts[i] for i in idx})


def read_rgba(path, nframes, size=SIZE):
    """Decode ProRes 4444 -> (nframes, size, size, 4) uint8, alpha preserved."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={size}:{size}",
         "-frames:v", str(nframes), "-pix_fmt", "rgba", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 4)
    if arr.shape[0] < nframes:
        raise RuntimeError(f"{path.name}: got {arr.shape[0]} frames, need {nframes}")
    return arr[:nframes]


def read_rgba_zoomed(path, nframes, zoom):
    """Decode at the zoomed size and centre it on a transparent SIZE canvas.

    One resample, straight from the 1080 source to round(SIZE*zoom) - not
    1080 -> 1024 -> target, which would soften the thick black outlines twice. The padding
    is fully transparent, so the flat ground shows through it after compositing, which
    means the ground is regenerated at the smaller scale rather than resampled: no halos.
    """
    if zoom >= 0.999:
        return read_rgba(path, nframes)
    n = int(round(SIZE * zoom))
    n -= n % 2                                   # keep it even
    small = read_rgba(path, nframes, size=n)
    canvas = np.zeros((nframes, SIZE, SIZE, 4), np.uint8)
    off = (SIZE - n) // 2
    canvas[:, off:off + n, off:off + n] = small
    return canvas


def assign_zooms(jobs):
    """One zoom per (clip, background) pair, decorrelated from every other factor.

    Round-robin over a stable enumeration rather than a hash: it makes the marginal counts
    near-exact by construction, and verify_zoom_balance asserts the result. Zoom must NOT
    track label, character, background or angle - a factor that predicts the label is
    exactly the confound the frame-length bijection already introduced.
    """
    ladder = sorted(ZOOMS)
    for i, j in enumerate(sorted(jobs, key=lambda x: (x["kind"], x["char"],
                                                      x["label"], x["angle"]))):
        j["zooms"] = {tag: ladder[(i + k) % len(ladder)]
                      for k, tag in enumerate(BACKGROUNDS)}
    return jobs


def verify_zoom_balance(jobs, tol=0.34):
    """Fail loudly if zoom correlates with any label the model could shortcut on."""
    ladder = sorted(ZOOMS)
    even = 1.0 / len(ladder)
    factors = {"label": lambda j, t_: j["label"], "character": lambda j, t_: j["char"],
               "background": lambda j, t_: t_,
               "angle": lambda j, t_: j["angle"].split("_t")[0]}
    problems = []
    for fname, key in factors.items():
        tab = collections.defaultdict(collections.Counter)
        for j in jobs:
            for tag in BACKGROUNDS:
                tab[key(j, tag)][j["zooms"][tag]] += 1
        for level, c in sorted(tab.items(), key=lambda kv: str(kv[0])):
            n = sum(c.values())
            for z in ladder:
                frac = c[z] / n
                if abs(frac - even) > tol:
                    problems.append(
                        f"{fname}={level}: zoom {z} is {frac:.0%} of {n} (want ~{even:.0%})")
    return problems


def verify_length_balance(jobs):
    """Assert no frame length is a perfect predictor of a label.

    The direct test of the BUCKET NOTE fix: after the common bucket is added, the shortest
    bucket must carry every label of its kind. If it does not, the shortcut is still there.
    """
    per_kind = collections.defaultdict(lambda: collections.defaultdict(set))
    for j in jobs:
        for n in buckets_for(j):
            per_kind[j["kind"]][n].add(j["label"])
    problems = []
    for kind, lengths in per_kind.items():
        want = set(MOTION if kind == "motion" else EXPRESSIONS)
        want = {l for l in want if any(j["label"] == l and j["kind"] == kind for j in jobs)}
        common = COMMON[kind]
        got = lengths.get(common, set())
        if got != want:
            problems.append(f"{kind}: common bucket f{common} carries {sorted(got)}, "
                            f"expected all of {sorted(want)}")
        for n, labs in sorted(lengths.items()):
            if n != common and len(labs) == 1 and len(want) > 1:
                lab = next(iter(labs))
                if common not in [b for b in lengths if lab in lengths[b]]:
                    problems.append(f"{kind}: f{n} uniquely predicts {lab}")
    return problems


PREFLIGHT_SIZE = 192          # cheap decode, structural check only
TORN_FRAC = 0.6               # a frame under 0.6x the clip's median alpha coverage is torn
# Measured tear rate per decode on the corrupt files: 17-50%, and essentially the same at
# a 192 px and a 1024 px decode, so the build resolution does not help. At the worst
# observed rate (50%) an 8-try budget still fails 0.39% of the time - and with ~15 damaged
# sources x 3 zooms there are ~45 chances per build, so a spurious drop actually happened.
# 24 tries puts that at 0.5^24 = 6e-8. Only the damaged files ever pay for a retry, and a
# clip already on disk is reused without decoding at all. The real fix is a client re-export.
DECODE_TRIES = 24


def find_torn(alpha, n):
    """Indices of torn frames within the first n. Alpha coverage is the detector: the
    character occupies a stable ~35% of frame, so a frame at a fraction of the clip's
    median coverage is decoder damage, not animation."""
    cover = (alpha > 8).reshape(len(alpha), -1).mean(1)
    med = float(np.median(cover))
    if med <= 0:
        return list(range(min(n, len(cover))))
    return [int(i) for i in np.where(cover[:n] < TORN_FRAC * med)[0]]


def preflight(job):
    """Cheap structural check: can this clip supply the frames we need at all?

    **ffprobe metadata lies about this delivery.** 16 iteration_4 files carry a ProRes
    `invalid frame header`; on those the container's `nb_frames` disagrees with what
    actually decodes. This catches the hard case (fewer frames exist than we need).

    It deliberately does NOT decide tearing. The corrupt streams decode
    **non-deterministically** - PAX_MOTION_SITTING_QF1_R yields torn frames at 14-16 on one
    decode and a clean run on the next, because the decoder sometimes conceals the bad
    frame by repeating its predecessor and sometimes emits the torn one. A pre-pass
    therefore cannot predict what the build decode will produce, so tearing is checked on
    the actual frames that get written (see decode_checked).
    """
    want = recipe(job)["frames"]
    need = 1 if want is None else max(buckets_for(job))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(job["src"]),
         "-vf", f"scale={PREFLIGHT_SIZE}:{PREFLIGHT_SIZE}",
         "-pix_fmt", "rgba", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, PREFLIGHT_SIZE, PREFLIGHT_SIZE, 4)[..., 3]
    if len(a) < need:
        return False, f"only {len(a)} frames decode, need {need} (container claims more)", len(a)
    return True, None, len(a)


def derive_frame_budgets(jobs, decoded):
    """Set `frames` for any label that declares it as None, from the delivered footage.

    The budget is the largest musubi-legal (4k+1) count that fits the SHORTEST clip in that
    label, so every source in the label can supply it. Derived rather than hardcoded because
    the length of a not-yet-delivered class cannot be known in advance.
    """
    per_label = collections.defaultdict(list)
    for j, n in zip(jobs, decoded):
        if recipe(j)["frames"] is None:
            per_label[(j["kind"], j["label"])].append(n)
    for (kind, lab), lens in sorted(per_label.items()):
        shortest = min(lens)
        legal = [f for f in range(5, shortest + 1) if (f - 1) % 4 == 0]
        if not legal:
            sys.exit(f"{lab}: shortest clip is {shortest} frames — too short for any "
                     f"musubi-legal (4k+1) length")
        table = MOTION if kind == "motion" else EXPRESSIONS
        table[lab]["frames"] = legal[-1]
        print(f"derived frame budget for {lab}: {legal[-1]} "
              f"(shortest of {len(lens)} clips = {shortest} frames "
              f"= {shortest/FPS:.2f} s)")


def decode_checked(job, need, zoom):
    """Decode, then verify no torn frame reached the array we are about to composite.

    Retries because the damage is non-deterministic: a re-decode of the same corrupt file
    often comes back clean. If it never does, the clip is dropped and reported - never
    silently written, and never "repaired" by cutting the torn frames out, which would
    leave a jump cut mid-motion and teach a discontinuity.
    """
    last = None
    for _ in range(DECODE_TRIES):
        frames = read_rgba_zoomed(job["src"], need, zoom)
        torn = find_torn(frames[..., 3], need)
        if not torn:
            return frames
        last = torn
    raise RuntimeError(f"torn frames {last} within the first {need} at zoom {zoom:.2f} "
                       f"after {DECODE_TRIES} decodes — corrupt ProRes, needs re-export")


def composite(frames, rgb):
    a = frames[..., 3:4].astype(np.float32) / 255.0
    fg = frames[..., :3].astype(np.float32)
    return (fg * a + np.array(rgb, np.float32) * (1 - a)).round().astype(np.uint8)


def write_mp4(frames, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-pix_fmt", "yuv420p", "-an", str(dest)],
        stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode:
        sys.exit(f"ffmpeg failed on {dest}")


def caption(job, bg_desc, zoom, nframes):
    """style + identity(+colour) + action + label + shot size + eye level + angle + ground.

    The `truncated` note matters: a clip cut to the common bucket shows the OPENING of the
    arc, not the whole thing. Saying so keeps the caption honest about what is on screen -
    v6 section 4.3 rejected sliding windows for exactly this reason (a window starting
    mid-arc is mislabelled).

    The note is attached to the LABEL, not the action clause. Every action clause is a
    gerund phrase ("crouching down and then hopping straight up"), so prefixing it produces
    "beginning to crouching down" - malformed, and it would have shipped on 55% of the set.
    Annotating the label instead is grammatical, and it keeps the action clause byte-
    identical between the two buckets, so length is the only thing that differs.
    """
    r = recipe(job)
    label = r["label"] if nframes >= r["frames"] else f"{r['label']}, opening frames only"
    return (f"{STYLE}{ANCHORS[job['char']]}, {r['action']}; {label}; "
            f"{ZOOMS[zoom]}, eye level, {job['clause']}; {bg_desc}.")


def clause_for_slug(slug):
    """Angle slug -> caption clause. Inverse of parse_angle, for --captions-only."""
    base = slug.split("_t")[0]
    if base == "FRONT":
        return FRONT
    if base.startswith("SIDE_"):
        return SIDE[base[-1]]
    m = re.fullmatch(r"QF([123])_([LR])", base)
    if m:
        deg, d = int(m.group(1)), m.group(2)
        return QUARTER[deg].format(side="left" if d == "L" else "right")
    raise ValueError(f"cannot rebuild clause for angle {slug!r}")


def recaption(out_dir):
    """Rewrite every jsonl + toml from the merged manifest, without touching the clips.

    Captions live only in the dataset jsonl - the mp4s are unaffected - so a caption fix
    costs seconds instead of a full re-encode.
    """
    man = json.loads((out_dir / "manifest.json").read_text())
    by_kind = {}
    for c in man["clips"]:
        kind = c["kind"]
        job = {"char": c["character"], "label": c["label"], "kind": kind,
               "angle": c["angle"], "clause": clause_for_slug(c["angle"])}
        outs = []
        for o in c["outputs"]:
            bg_desc, _ = BACKGROUNDS[o["background"]]
            outs.append({"name": o["name"], "frames": o["frames"],
                         "background": o["background"], "zoom": o["zoom"],
                         "truncated": o["truncated"],
                         "caption": caption(job, bg_desc, o["zoom"], o["frames"])})
        by_kind.setdefault(kind, []).append({"job": job, "outputs": outs})
    for kind, records in by_kind.items():
        _, counts = emit(records, out_dir, str(out_dir / "clips"),
                         str(out_dir / "cache" / kind), "", kind)
        emit(records, out_dir, WS_CLIPS, f"{WS_CACHE}/{kind}", ".workspace", kind)
        n = sum(len(r["outputs"]) for r in records)
        print(f"{out_dir.name}: {kind} recaptioned {n} clips, buckets {counts}")


def process(job, clips_dir, reencode=False):
    """Composite one source into every (zoom x ground x length) output it owes.

    Incremental by default: a clip already on disk is reused, and a zoom whose outputs are
    ALL present needs no decode at all. Extending the ladder therefore costs only the new
    rungs rather than a full re-encode. `--reencode` forces everything.
    """
    r = recipe(job)
    lens = buckets_for(job)
    need = max(lens)
    # One decode per distinct zoom, not per background.
    by_zoom = {}
    for tag in BACKGROUNDS:
        by_zoom.setdefault(job["zooms"][tag], []).append(tag)

    def name_of(tag, n):
        return f"{job['char'].lower()}_{job['label']}_{job['angle']}__{tag}_f{n}.mp4"

    todo = {}
    for zoom, tags in by_zoom.items():
        missing = [(tag, n) for tag in tags for n in lens
                   if reencode or not (clips_dir / name_of(tag, n)).exists()]
        if missing:
            todo[zoom] = missing

    # Decode every needed zoom BEFORE writing anything, so a source that turns out to be
    # damaged leaves no half-written set of clips behind.
    try:
        decoded = {zoom: decode_checked(job, need, zoom) for zoom in todo}
    except RuntimeError as e:
        return {"job": job, "outputs": [], "error": str(e), "reused": 0}

    out, reused = [], 0
    for zoom, tags in sorted(by_zoom.items()):
        for tag in tags:
            bg_desc, rgb = BACKGROUNDS[tag]
            for n in lens:
                name = name_of(tag, n)
                if (tag, n) in todo.get(zoom, []):
                    write_mp4(composite(decoded[zoom][:n], rgb), clips_dir / name)
                else:
                    reused += 1
                out.append({"name": name, "caption": caption(job, bg_desc, zoom, n),
                            "frames": n, "background": tag, "zoom": zoom,
                            "truncated": n < r["frames"]})
    return {"job": job, "outputs": out, "error": None, "reused": reused}


def emit(records, out, prefix, cache_root, suffix, kind):
    """One jsonl per frame bucket + a toml with one [[datasets]] block per bucket."""
    buckets = {}
    for rec in records:
        for o in rec["outputs"]:
            buckets.setdefault(o["frames"], []).append(
                {"video_path": f"{prefix.rstrip('/')}/{o['name']}", "caption": o["caption"]})
    lines = [f"# Musubi dataset config — v7 {kind} (auto-generated by prep_v7.py).",
             "# One [[datasets]] block per frame length: musubi buckets by frame count.",
             f"# f{COMMON[kind]} is the COMMON bucket and carries every label, so sequence",
             "# length is not a predictive shortcut for the label (see BUCKET NOTE).",
             "# 1024x1024, not the source 1080: Wan's 8x VAE + 2x2 patchify needs an even",
             "# latent side (1080/8 = 135 is odd).",
             "", "[general]", "resolution = [1024, 1024]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    for n in sorted(buckets):
        jp = out / f"dataset_{kind}_f{n}{suffix}.jsonl"
        with open(jp, "w") as f:
            for rec in buckets[n]:
                f.write(json.dumps(rec) + "\n")
        ref = jp if not suffix else f"{prefix.rsplit('/', 1)[0]}/{jp.name}"
        lines += ["[[datasets]]",
                  f'video_jsonl_file = "{ref}"',
                  f'cache_directory = "{cache_root}/f{n}"',
                  f"target_frames = [{n}]",
                  'frame_extraction = "head"',
                  "num_repeats = 1",
                  f"# {len(buckets[n])} clips", ""]
    cfg = out / f"dataset_config_{kind}_v7{suffix}.toml"
    cfg.write_text("\n".join(lines))
    return cfg, {n: len(v) for n, v in buckets.items()}


def summarise(kind, jobs):
    """Print what this kind will produce. Shared by --dry-run and the real build."""
    n_clips = sum(len(buckets_for(j)) for j in jobs) * len(BACKGROUNDS)
    print(f"\n=== {kind}: {len(jobs)} sources -> up to {n_clips} training clips")
    by = collections.Counter(f"{j['char']}/{j['label']}" for j in jobs)
    for k in sorted(by):
        print(f"    {k:<26} {by[k]} sources")
    bk = collections.Counter()
    for j in jobs:
        for n in buckets_for(j):
            bk[n] += len(BACKGROUNDS)
    print(f"    buckets: {dict(sorted(bk.items()))}   (f{COMMON[kind]} = common)")
    dist = collections.Counter(z for j in jobs for z in j["zooms"].values())
    print("    zoom ladder: " + " · ".join(
        f"{z:.2f}x {ZOOMS[z].replace('static ', '')} = {dist[z]}" for z in sorted(ZOOMS)))
    return n_clips


def build(jobs_by_kind, out_root, workers, preflight_dropped=(), reencode=False):
    """Build every kind into ONE v7 folder with a shared clips/ directory.

    One delivery, one folder. The two kinds train as two separate LoRAs (see the module
    docstring), but that is a *training* split, not a storage one - so they ship together
    and a GPU box pulls the whole delivery with one `az` command instead of two.

    Sharing clips/ is safe because output names carry the label
    (`<char>_<label>_<ANGLE>__<ground>_f<frames>.mp4`) and the motion and expression label
    sets are disjoint. It is asserted below rather than assumed.

    What stays per-kind is everything the trainer reads: a jsonl per frame bucket, a musubi
    config, and its own latent cache directory - because the two are separate runs against
    separate experts.
    """
    all_jobs = [j for k in jobs_by_kind for j in jobs_by_kind[k]]
    names = [f"{j['char'].lower()}_{j['label']}_{j['angle']}" for j in all_jobs]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        sys.exit(f"refusing to run: duplicate output names would overwrite: {dup}")

    out = out_root / "v7_primitives_build"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    results_by_kind, damaged_all = {}, []
    for kind, jobs in jobs_by_kind.items():
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda j: process(j, clips_dir, reencode), jobs))
            damaged = [r for r in results if r.get("error")]
        for r in damaged:
            print(f"    ✗ dropped {r['job']['src'].name}: {r['error']}")
            # A source dropped on THIS run may still have clips on disk from a previous
            # one, because the build is incremental. Leave them and the set silently
            # contains footage from a source we just rejected.
            j = r["job"]
            stale = list(clips_dir.glob(
                f"{j['char'].lower()}_{j['label']}_{j['angle']}__*.mp4"))
            for f in stale:
                f.unlink()
            if stale:
                print(f"      removed {len(stale)} stale clip(s) from an earlier build")
        damaged_all += [(kind, r) for r in damaged]
        results_by_kind[kind] = [r for r in results if not r.get("error")]

    # clips/ must contain EXACTLY the expected set. The incremental build reuses files by
    # name, so a rename (e.g. an angle re-index) leaves the old name behind as an orphan
    # that no jsonl references but that still ships in the folder. Sweep them.
    expected_names = {o["name"] for rs in results_by_kind.values()
                      for r in rs for o in r["outputs"]}
    orphans = [f for f in clips_dir.glob("*.mp4") if f.name not in expected_names]
    for f in orphans:
        f.unlink()
    if orphans:
        print(f"    removed {len(orphans)} orphaned clip(s) no longer in the set")

    expected = sum(len(r["outputs"]) for rs in results_by_kind.values() for r in rs)
    written = len(list(clips_dir.glob("*.mp4")))
    if written != expected:
        sys.exit(f"expected {expected} clips on disk, found {written}")

    # Rename to the real total now that damage is known, THEN write configs and manifest,
    # which embed absolute paths.
    final = out_root / f"v7_primitives_{written}"
    if final.exists() and final != out:
        sys.exit(f"refusing to overwrite existing {final}")
    if final != out:
        out.rename(final)
        out, clips_dir = final, final / "clips"
    reused = sum(r.get("reused", 0) for rs in results_by_kind.values() for r in rs)
    print(f"\nwrote {written} clips -> {clips_dir}"
          + (f"   ({reused} reused, {written - reused} newly encoded)" if reused else ""))
    if damaged_all:
        print(f"{len(damaged_all)} source(s) dropped as damaged")

    manifest_clips = []
    for kind, records in results_by_kind.items():
        cfg_local, counts = emit(records, out, str(clips_dir), str(out / "cache" / kind),
                                 "", kind)
        cfg_ws, _ = emit(records, out, WS_CLIPS, f"{WS_CACHE}/{kind}", ".workspace", kind)
        print(f"  {kind:<11} buckets {counts}")
        print(f"              {cfg_local.name} · {cfg_ws.name}")
        for r in records:
            manifest_clips.append(
                {"kind": kind, "source": str(r["job"]["src"]),
                 "source_md5": r["job"]["md5"], "character": r["job"]["char"],
                 "label": r["job"]["label"], "angle": r["job"]["angle"],
                 "derived_from": r["job"].get("derived_from"),
                 "source_frames": recipe(r["job"])["frames"],
                 "outputs": [{"name": o["name"], "background": o["background"],
                              "zoom": o["zoom"], "frames": o["frames"],
                              "truncated": o["truncated"]} for o in r["outputs"]]})

    manifest = {
        "generated_by": "prep_v7.py",
        "kinds": {k: {"expert": EXPERT[k], "common_bucket": COMMON[k],
                      "sources": len(v)} for k, v in results_by_kind.items()},
        "sources": [str(SRC_IT3), str(SRC_IT4)],
        "size": SIZE, "fps": FPS, "backgrounds": list(BACKGROUNDS),
        "zoom_ladder": {str(z): ZOOMS[z] for z in sorted(ZOOMS)},
        # Both stages of exclusion, so the manifest is the full record of what the
        # delivery could not supply: caught before the build, and caught during it.
        "dropped_damaged": (
            [{"kind": j["kind"], "source": str(j["src"]), "reason": why,
              "stage": "preflight"} for j, why in preflight_dropped]
            + [{"kind": k, "source": str(r["job"]["src"]), "reason": r["error"],
                "stage": "build"} for k, r in damaged_all]),
        "clips": manifest_clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {out / 'manifest.json'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("motion", "expression"))
    ap.add_argument("--no-zoom", action="store_true",
                    help="disable the shot-size ladder; every clip at 1.0x close-up")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the decode-and-check-for-damage pass (faster, unsafe: "
                         "16 iteration_4 files have corrupt ProRes streams)")
    ap.add_argument("--no-common-bucket", action="store_true",
                    help="emit natural lengths only (reproduces the v6-style confound; "
                         "for ablation, not for a real run)")
    ap.add_argument("--reencode", action="store_true",
                    help="re-encode every clip even if it already exists on disk")
    ap.add_argument("--captions-only", type=Path, metavar="SET_DIR",
                    help="rewrite the jsonl/toml of an already-built set from its "
                         "manifest, without re-encoding any video")
    args = ap.parse_args()

    if args.captions_only:
        recaption(args.captions_only)
        return

    if args.no_common_bucket:
        for k in COMMON:
            COMMON[k] = 10 ** 6      # nothing reaches it, so only natural lengths emit
        print("!! common bucket DISABLED — frame length will predict the label")

    jobs, skipped, dupes, collisions, rebased = discover()
    print(f"discovered {len(jobs)} unique source clips "
          f"({len(dupes)} md5 duplicates dropped)")
    for what, why in skipped:
        print(f"  ! skipped {what}: {why}")
    for what, first in dupes:
        print(f"  = duplicate {Path(what).name} == {Path(first).name}")
    for before, after in rebased:
        print(f"  ° re-indexed {before} -> {after}")
    for slug, srcs in collisions:
        print(f"  ~ disambiguated {slug}: {srcs} -> _t1.._t{len(srcs)}")

    preflight_dropped = []
    if not args.skip_preflight:
        print(f"preflight: decoding {len(jobs)} sources to check for damage …")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = list(ex.map(preflight, jobs))
        derive_frame_budgets(jobs, [v[2] for v in verdicts])
        kept, dropped = [], []
        for j, (ok, why, _n) in zip(jobs, verdicts):
            (kept if ok else dropped).append(j if ok else (j, why))
        for j, why in dropped:
            print(f"  ✗ damaged {j['src'].name}: {why}")
        print(f"preflight: {len(kept)} usable, {len(dropped)} damaged")
        jobs, preflight_dropped = kept, dropped

    if args.no_zoom:
        for j in jobs:
            j["zooms"] = {tag: 1.00 for tag in BACKGROUNDS}
        print("zoom ladder DISABLED (--no-zoom): every clip at 1.0x close-up")
    else:
        assign_zooms(jobs)

    lp = verify_length_balance(jobs)
    if lp and not args.no_common_bucket:
        for m in lp:
            print(f"  ! length imbalance {m}")
        sys.exit("refusing to run: frame length predicts a label (see above)")

    jobs_by_kind = {}
    for kind in ("motion", "expression"):
        if args.only and args.only != kind:
            continue
        sub = [j for j in jobs if j["kind"] == kind]
        if not sub:
            print(f"\n[{kind}] no sources found — skipping"); continue
        if not args.no_zoom:
            probs = verify_zoom_balance(sub)
            if probs:
                for m in probs:
                    print(f"  ! zoom imbalance {m}")
                sys.exit(f"refusing to run: zoom correlates with a label in {kind}")
        summarise(kind, sub)
        jobs_by_kind[kind] = sub

    if args.dry_run or not jobs_by_kind:
        return
    out = build(jobs_by_kind, args.out, args.workers, preflight_dropped,
                args.reencode)
    print(f"\nbuilt: {out}")


if __name__ == "__main__":
    main()
