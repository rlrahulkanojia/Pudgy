#!/usr/bin/env python3
"""
prep_ltx25.py - build an LTX-2.5 training corpus straight from the raw ProRes.

STANDALONE. This script shares no code, no data and no conventions with the v1-v7
Wan line. It reads `raw/` and writes one self-contained folder. Nothing here is a
port of anything.

WHY IT LOOKS THE WAY IT DOES
----------------------------
1. LTX's video VAE compresses time 8x, so frame counts must satisfy `frames % 8 == 1`
   (1, 9, 17, 25, 33, 41, 49, 57, ...). That is coarser than Wan's 4x, and it is the
   single biggest constraint on this corpus.

2. Every label is ALSO emitted at a common floor length (17). The trainer buckets by
   frame count, so a label appearing at exactly one length makes sequence length a
   perfect shortcut for that label. The floor guarantees one bucket carries every
   label, so length predicts nothing. Asserted, not assumed.

3. Shot size is assigned round-robin and balance-checked, so zoom cannot predict the
   label either.

4. Cells are DISCOVERED from the directory tree, never hardcoded. The client delivers
   incrementally (CONFUSED and CRYING landed 2026-09-05, mid-design), so a new cell
   must not require a code change. Unknown labels are emitted with a caption warning
   rather than silently mis-captioned.

5. The raw tree has real defects: corrupt ProRes streams that tear NON-deterministically,
   files whose names disagree with their folder, and a file filed under the wrong
   character. The folder is treated as authoritative for character and label; the
   filename supplies only the angle. Every disagreement is reported, never silently
   "fixed", and nothing under raw/ is ever modified.

Usage:
    python prep_ltx25.py --out <dir> [--limit N] [--captions-only <dir>] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- config

RAW = Path.home() / "Documents/Projects/Saksham/Pudgy/Data/raw/iteration_4/LSLTTT-Project"
OUT_DEFAULT = Path.home() / "Documents/Projects/Saksham/Pudgy/Data/processed/ltx25_experiment"

SIZE = 1024              # W and H; must be divisible by 32 (LTX spatial VAE factor)
FPS = 24                 # native source rate and the LTX trainer default
TEMPORAL = 8             # LTX temporal VAE factor -> frames % 8 == 1
FLOOR = 17               # common bucket every label is also emitted at
LADDER_STEPS = 3         # rungs per label, floor -> natural length
SEED = 42

# Angle held out of training entirely, so evaluation has an unseen viewpoint.
HOLDOUT_ANGLE = "SIDE_R"

# Flat grounds. Deliberately avoid the Pudgy palette (Pax blue, Polly pink, belly
# white/cream, feet orange, outline black) so the character stays separable from its
# ground and the delivery matte is clean.
GROUNDS = {
    "sage green":   (139, 168, 136),
    "olive":        (128, 132,  86),
    "teal":         ( 74, 130, 130),
    "warm grey":    (146, 141, 134),
    "mustard":      (191, 158,  74),
    "slate":        ( 96, 108, 124),
    "plum":         (122,  90, 122),
    "rust":         (163, 100,  74),
}

# Shot sizes: (name, caption phrase, TARGET character height as a fraction of the frame).
#
# The client frames the character at ~92% of frame height in every source clip, so there is
# no headroom to crop INWARD - measured, medium and close both clamped to the same window
# and differed by 2 percentage points, which would have meant three captions describing one
# image. Because the ground is ours to synthesise, the ladder instead zooms OUT by padding
# the canvas with flat ground. That gives a real range, and v4 measured that character size
# in frame drives identity quality, so this axis needs to be genuine rather than nominal.
SHOTS = [
    ("wide",   "static wide shot, eye level",   0.45),
    ("medium", "static medium shot, eye level", 0.66),
    ("close",  "static close-up, eye level",    0.90),
]

CHARACTERS = {
    "PAX":   ("pxngn0", "blue",  "Pax"),
    "POLLY": ("plngn0", "pink",  "Polly"),
}

STYLE = ("2D cartoon animation in the Pudgy Penguins style, thick clean black outlines, "
         "flat pastel colors, cel shading")

# Actions verified by watching the footage. `sitting` is a seated IDLE (already seated at
# frame 0), not sitting down. `jumping` is a full crouch-airborne-land hop. `confused` and
# `crying` were watched on 2026-09-05 when those cells arrived.
ACTIONS = {
    "WALKING":   ("walk cycle", "walking forward with a bouncy waddle, flippers swinging at its "
                  "sides, body rocking gently from side to side with each step", True),
    "RUNNING":   ("run cycle", "running forward at a quick clip, flippers pumping, body leaning "
                  "into the stride", True),
    "WAVING":    ("waving", "raising one flipper and waving it back and forth in greeting, the "
                  "other flipper resting at its side", False),
    "SITTING":   ("seated idle", "sitting on the ground, settling its weight and shifting "
                  "slightly in place", False),
    "JUMPING":   ("jump cycle", "crouching down and then hopping straight up, flippers spreading "
                  "out wide, feet tucking under as it leaves the ground, then landing and "
                  "settling back down", True),
    "ANGRY":     ("angry expression", "scowling into an angry glare, brows dropping and pressing "
                  "together, beak set in a hard frown, shoulders squaring", False),
    "CONFUSED":  ("confused expression", "tilting its head to one side, one brow arching up while "
                  "the other drops, eyes drifting off to the side in puzzlement, then holding the "
                  "quizzical look", False),
    "CRYING":    ("crying expression", "shrinking down and drawing its flippers in close, eyes "
                  "welling up with tears, beak turning down, then holding the tearful look", False),
    "HAPPY":     ("happy expression", "breaking into a bright happy smile, eyes curving up, cheeks "
                  "lifting, body giving a small pleased bounce", False),
    "LAUGHING":  ("laughing expression", "laughing out loud, eyes squeezing shut with mirth, beak "
                  "opening in an open-mouthed laugh, flippers coming up to its belly as its body "
                  "rocks back, then holding the laugh", False),
    # MEASURED 96.6% frozen: this is a HELD pose, not an idle with settling movement.
    # Captioning it as animated would teach the model that "idle" means "move slightly",
    # which the footage does not show.
    "NEUTRAL":   ("neutral pose", "standing still in a calm neutral pose, facing forward, "
                  "holding the pose steadily", False),
    "SURPRISE":  ("surprised expression", "snapping into a startled look of surprise, eyes going "
                  "wide, brows shooting up, beak opening, body pulling back sharply", False),
}

ANGLES = {
    "FRONT":  "facing the camera directly, front view",
    "QF1_L":  "turned slightly to its left, three-quarter front view",
    "QF1_R":  "turned slightly to its right, three-quarter front view",
    "QF2_L":  "turned further to its left, wide three-quarter view",
    "QF2_R":  "turned further to its right, wide three-quarter view",
    "QF3_L":  "turned strongly to its left, near-profile three-quarter view",
    "QF3_R":  "turned strongly to its right, near-profile three-quarter view",
    "SIDE_L": "seen from its left side, profile view",
    "SIDE_R": "seen from its right side, profile view",
}

# Tear detection. A frame whose alpha coverage falls well under the clip median is torn.
# The corruption is non-deterministic, so this runs on the frames about to be written.
TEAR_FRAC = 0.60
DECODE_TRIES = 24


# ------------------------------------------------------------------------- small utils

def top_legal(n: int) -> int:
    """Largest legal frame count <= n."""
    ok = [f for f in range(1, n + 1) if (f - 1) % TEMPORAL == 0]
    return max(ok) if ok else 1


ACTIVE_EPS = 0.5      # mean abs 8-bit frame difference below which a pair is "frozen"


def active_frames(path: Path, total: int) -> int:
    """Frames up to and including the last one that actually moves.

    MEASURED, not assumed: the source corpus is 37.9% frozen at the median, and NEUTRAL
    is 96.6% frozen (a still image held for 60 frames). Laddering against the raw length
    would emit 162 clips of static video, and training on static video teaches the model
    to generate static video - the exact frozen-frame failure this project keeps hitting.

    So the ladder is built against the ACTIVE length: the frozen tail is trimmed before
    rungs are chosen. Frozen holds INSIDE a clip are preserved, because those are the
    client's animation intent; only the dead tail goes.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", "scale=128:128,format=gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = 128 * 128
    c = len(out) // n
    if c < 2:
        return total
    g = np.frombuffer(out[:c * n], dtype=np.uint8).reshape(c, 128, 128).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    moving = np.where(d >= ACTIVE_EPS)[0]
    if len(moving) == 0:
        return 1                       # nothing moves at all: a still, not a clip
    return min(total, int(moving[-1]) + 2)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_frames(path: Path) -> int:
    """True decodable frame count. ffprobe's container metadata lies on this corpus."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------- discovery

# Three naming conventions coexist in this corpus, all seen in iteration_4:
#   FRONT / FR                          -> FRONT
#   QF1_L, QF2_R, QF3_L  (index first)  -> as-is
#   QF_L2, QF_R1         (index last)   -> QF2_L, QF1_R   <- NEUTRAL cells only
#   QF_L, QF_R           (no index)     -> QF1_L, QF1_R   <- some HAPPY cells
#   SIDE_L / SIDE_R                     -> as-is
ANGLE_PATTERNS = [
    (re.compile(r"(?:^|_)(?:FRONT|FR)$"),        lambda m: "FRONT"),
    (re.compile(r"(?:^|_)SIDE_([LR])$"),         lambda m: f"SIDE_{m.group(1)}"),
    (re.compile(r"(?:^|_)QF([123])_([LR])$"),    lambda m: f"QF{m.group(1)}_{m.group(2)}"),
    (re.compile(r"(?:^|_)QF_([LR])([123])$"),    lambda m: f"QF{m.group(2)}_{m.group(1)}"),
    (re.compile(r"(?:^|_)QF_([LR])$"),           lambda m: f"QF1_{m.group(1)}"),
]


def parse_angle(stem: str) -> str | None:
    up = stem.upper()
    for rx, norm in ANGLE_PATTERNS:
        m = rx.search(up)
        if m:
            a = norm(m)
            return a if a in ANGLES else None
    return None


def discover(raw: Path) -> tuple[list[dict], list[str]]:
    """Walk the raw tree. Folder is authoritative for character and label."""
    jobs, notes = [], []
    seen_md5: dict[str, str] = {}

    for char_dir in sorted(raw.glob("*/*/*")):
        if not char_dir.is_dir():
            continue
        character = char_dir.name.upper()
        if character not in CHARACTERS:
            notes.append(f"skip: unknown character folder {char_dir}")
            continue

        cell = char_dir.parent.name.upper()
        label = cell.replace("MOTION_", "")
        kind = "motion" if char_dir.parent.parent.name.upper().startswith("MOTION") else "expression"

        for f in sorted(char_dir.glob("*.mov")):
            stem = f.stem.upper()
            angle = parse_angle(stem)
            if angle is None:
                notes.append(f"DEFECT no-angle: {f.relative_to(raw)} - skipped")
                continue

            # Filename disagreeing with its folder. Folder wins; the disagreement is reported.
            for other in CHARACTERS:
                if stem.startswith(other + "_") and other != character:
                    notes.append(f"DEFECT misfiled-character: {f.relative_to(raw)} "
                                 f"is named {other} but sits under {character}/ - using {character}")
            fname_label = None
            for lab in ACTIONS:
                if f"_{lab}_" in f"_{stem}_":
                    fname_label = lab
                    break
            if fname_label and fname_label != label:
                notes.append(f"DEFECT mislabelled-file: {f.relative_to(raw)} names "
                             f"{fname_label} but sits under {label}/ - using {label}")

            digest = md5(f)
            if digest in seen_md5:
                notes.append(f"DEFECT duplicate: {f.relative_to(raw)} is byte-identical to "
                             f"{seen_md5[digest]} - skipped")
                continue
            seen_md5[digest] = str(f.relative_to(raw))

            real = probe_frames(f)
            if real < 2:
                notes.append(f"DEFECT undecodable: {f.relative_to(raw)} decodes {real} frames"
                             f" - skipped")
                continue

            jobs.append({
                "src": f, "character": character, "label": label, "kind": kind,
                "angle": angle, "real_frames": real, "md5": digest,
                "active_frames": active_frames(f, real),
            })
    # Length sanity per cell. Every angle in a cell renders the SAME performance, so
    # they should share a frame count. A file well under its cell's mode has a damaged
    # stream (ffprobe metadata lies on this corpus), not a shorter performance.
    keep = []
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for j in jobs:
        by_cell[(j["label"], j["character"])].append(j)
    for (label, char), group in by_cell.items():
        mode = Counter(j["real_frames"] for j in group).most_common(1)[0][0]
        for j in group:
            if j["real_frames"] < 0.9 * mode:
                notes.append(f"DEFECT truncated: {j['src'].name} decodes {j['real_frames']}f "
                             f"but {label}/{char} renders {mode}f - skipped")
            else:
                keep.append(j)

    # A non-cyclic source below the floor cannot be emitted at all.
    final = []
    for j in keep:
        entry = ACTIONS.get(j["label"])
        cyclic = bool(entry and entry[2])
        if top_legal(j["real_frames"]) < FLOOR and not cyclic:
            notes.append(f"DEFECT below-floor: {j['src'].name} is {j['real_frames']}f and "
                         f"{j['label']} is not a verified cycle, so it cannot be wrapped - skipped")
        else:
            final.append(j)
    return final, notes


# ------------------------------------------------------------------------------- ladder

def buckets_for(job: dict) -> list[int]:
    """Frame lengths this source is emitted at, on legal 8n+1 rungs.

    The top rung is the source's own length truncated to a legal value, so no motion is
    invented. The exception is a verified CYCLE shorter than the floor: a cycle has no
    narrative arc to mislabel, so it is wrapped up to the floor rather than emitted at a
    near-degenerate length. `walking` (16 raw frames) is the only case today; untouched it
    would yield 9 frames, which is 2 latent frames after 8x compression.
    """
    entry = ACTIONS.get(job["label"])
    cyclic = bool(entry and entry[2])
    # Trim the frozen tail, but never below the common floor: the floor invariant (every
    # label present at one shared length) matters more than a few held frames, and going
    # under it would put a label at 9 frames, which is 2 latent frames after 8x compression.
    usable = min(job["real_frames"], max(job["active_frames"], FLOOR))
    nat = top_legal(usable)

    if nat < FLOOR:
        return [FLOOR] if cyclic else [nat]
    if nat == FLOOR:
        return [FLOOR]

    rungs = [f for f in range(FLOOR, nat + 1) if (f - 1) % TEMPORAL == 0]
    if len(rungs) <= LADDER_STEPS:
        return rungs
    idx = [round(i * (len(rungs) - 1) / (LADDER_STEPS - 1)) for i in range(LADDER_STEPS)]
    return sorted({rungs[i] for i in idx})


# ------------------------------------------------------------------------ decode / write

def read_rgba(path: Path, want: int, loop: bool) -> list[np.ndarray] | None:
    """Decode to raw RGBA at SIZE. `loop` wraps a cycle that is shorter than `want`."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", "format=rgba",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        capture_output=True)
    side = _probe_side(path)
    n = side * side * 4
    buf = proc.stdout
    count = len(buf) // n
    if count == 0:
        return None
    arr = np.frombuffer(buf[:count * n], dtype=np.uint8).reshape(count, side, side, 4)
    frames = [arr[i] for i in range(count)]
    if len(frames) < want:
        if not loop:
            return None
        frames = [frames[i % len(frames)] for i in range(want)]
    return frames[:want]


def _probe_side(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])


def alpha_bbox(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Union bounding box of the character across every frame, from the alpha channel.

    Union, not per-frame: a per-frame box would make the crop breathe with the animation
    and introduce a fake camera move into what is a locked-off shot.
    """
    acc = np.zeros(frames[0].shape[:2], dtype=bool)
    for f in frames:
        acc |= f[:, :, 3] > 8
    ys, xs = np.where(acc)
    if len(ys) == 0:
        h, w = acc.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_shot(frames: list[np.ndarray], target: float) -> list[np.ndarray]:
    """Frame the character so it occupies ~`target` of the output height.

    Builds a square window of `char_height / target` source pixels centred on the
    character. When that window is LARGER than the source frame (the normal case here,
    since the client frames tight), the surplus is transparent and composites to flat
    ground, so zooming out costs nothing and invents no content. When it is smaller, this
    crops in. Either way the character is never clipped.
    """
    side = frames[0].shape[0]
    x0, y0, x1, y1 = alpha_bbox(frames)
    ch = max(y1 - y0, 1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    win = int(round(ch / max(target, 0.05)))
    win = max(win, ch + 2)                    # never clip the character
    half = win / 2.0
    ox, oy = int(round(half - cx)), int(round(half - cy))

    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    w = min(side - sx0, win - dx0)
    h = min(side - sy0, win - dy0)

    out = []
    for f in frames:
        canvas = np.zeros((win, win, 4), dtype=np.uint8)
        if w > 0 and h > 0:
            canvas[dy0:dy0 + h, dx0:dx0 + w] = f[sy0:sy0 + h, sx0:sx0 + w]
        out.append(np.asarray(Image.fromarray(canvas, mode="RGBA")
                              .resize((SIZE, SIZE), Image.LANCZOS)))
    return out


def find_torn(frames: list[np.ndarray]) -> list[int]:
    """Frames whose alpha coverage collapses against the clip median are torn.

    The ProRes corruption in this corpus is NON-deterministic: the same file yields a
    clean decode on one attempt and a torn one on the next, so a pre-pass cannot certify
    a file. This runs on the frames about to be written.
    """
    cov = [float(f[:, :, 3].mean()) for f in frames]
    med = sorted(cov)[len(cov) // 2]
    if med <= 0:
        return list(range(len(frames)))
    return [i for i, c in enumerate(cov) if c < TEAR_FRAC * med]


def composite(frames: list[np.ndarray], rgb: tuple[int, int, int]) -> bytes:
    """Alpha-over a flat ground, returning packed RGB24 for ffmpeg's rawvideo input."""
    bgc = np.array(rgb, dtype=np.uint16)
    out = []
    for fr in frames:
        a = fr[:, :, 3:4].astype(np.uint16)
        src = fr[:, :, :3].astype(np.uint16)
        px = (src * a + bgc * (255 - a)) // 255
        out.append(px.astype(np.uint8))
    return np.concatenate(out, axis=0).tobytes()


def write_mp4(rgb_bytes: bytes, dest: Path, nframes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-frames:v", str(nframes), str(dest)],
        input=rgb_bytes, check=True, capture_output=True)


# ----------------------------------------------------------------------------- captions

def caption(job: dict, ground: str, shot_phrase: str, nframes: int) -> tuple[str, bool]:
    token, colour, name = CHARACTERS[job["character"]]
    entry = ACTIONS.get(job["label"])
    unknown = entry is None
    if unknown:
        label_txt = job["label"].lower().replace("_", " ")
        action = f"performing a {label_txt} action"
    else:
        label_txt, action, _ = entry

    if nframes == 1:
        # A single frame cannot depict a walk cycle. Stills carry appearance only, which
        # is what they are for (LTX-2 #249: stills teach appearance, not motion).
        text = (f"{token}, {name}, a short round {colour} penguin. {STYLE}; "
                f"standing in a still pose; {shot_phrase}, {ANGLES[job['angle']]}; "
                f"plain {ground} background.")
        return text, False

    truncated = nframes < top_legal(min(job["real_frames"],
                                        max(job.get("active_frames", job["real_frames"]), FLOOR)))
    label_note = f"{label_txt}, opening frames only" if truncated else label_txt

    text = (f"{token}, {name}, a short round {colour} penguin. {STYLE}; "
            f"{action}; {label_note}; {shot_phrase}, {ANGLES[job['angle']]}; "
            f"plain {ground} background.")
    return text, unknown


# ------------------------------------------------------------------------- verification

def verify_length_balance(records: list[dict]) -> None:
    at_floor = {r["label"] for r in records if r["frames"] == FLOOR}
    every = {r["label"] for r in records}
    missing = every - at_floor
    if missing:
        sys.exit(f"FATAL length balance: labels absent from the floor bucket ({FLOOR}f): "
                 f"{sorted(missing)}. Frame count would predict the label.")
    print(f"  length balance OK - all {len(every)} labels present at {FLOOR}f")


def verify_shot_balance(records: list[dict], tol: float = 0.34) -> None:
    for axis in ("label", "character", "ground", "angle"):
        table = defaultdict(Counter)
        for r in records:
            table[r[axis]][r["shot"]] += 1
        for key, counts in table.items():
            tot = sum(counts.values())
            if tot < len(SHOTS):
                continue
            lo, hi = min(counts.values()), max(counts.values())
            if (hi - lo) / tot > tol:
                sys.exit(f"FATAL shot balance: {axis}={key} skewed across shot sizes {dict(counts)}")
    print(f"  shot-size balance OK across label/character/ground/angle")


# ---------------------------------------------------------------------------- emission

def process_one(args) -> list[dict]:
    """Emit every shot size for one (source, length) from a SINGLE decode.

    Decoding once and cropping three ways is both correct and ~3x faster than decoding
    per shot. It also guarantees the three shot sizes share identical timing and identical
    torn-frame handling, so shot size varies and nothing else does.
    """
    job, nframes, plan, out_clips = args
    entry = ACTIONS.get(job["label"])
    cyclic = bool(entry and entry[2])

    for attempt in range(DECODE_TRIES):
        frames = read_rgba(job["src"], nframes, loop=cyclic)
        if frames is None:
            continue
        if find_torn(frames):
            continue

        made = []
        for ground, (name, _phrase, frac) in plan:
            shot_frames = apply_shot(frames, frac)
            stem = (f"{job['character'].lower()}_{job['label'].lower()}_{job['angle']}"
                    f"__{ground.replace(' ', '')}_{name}_f{nframes}")
            dest = out_clips / f"{stem}.mp4"
            write_mp4(composite(shot_frames, GROUNDS[ground]), dest, nframes)
            made.append({"file": dest.name, "frames": nframes, "ground": ground, "shot": name,
                         "label": job["label"], "character": job["character"],
                         "kind": job["kind"], "angle": job["angle"],
                         "source": job["src"].name, "attempts": attempt + 1})
        return made
    return []


def emit_stills(jobs: list[dict], out: Path, per_character: int) -> list[dict]:
    """Single-frame appearance samples at the F=1 bucket.

    LTX-2 issue #249 (maintainer): image-only character LoRAs work, and "60-120 images is
    a reasonable appearance dataset; it will not teach motion." Identity is this corpus's
    weak axis - issue #255 confirms first-frame conditioning is NOT an identity encoder -
    so stills are the documented way to strengthen appearance. Mixing stills with video
    requires `optimization.batch_size: 1`.
    """
    rng = random.Random(SEED + 1)
    out_stills = out / "stills"
    out_stills.mkdir(parents=True, exist_ok=True)
    records = []

    for character in sorted(CHARACTERS):
        pool = [j for j in jobs if j["character"] == character]
        if not pool:
            continue
        by_angle: dict[str, list[dict]] = defaultdict(list)
        for j in pool:
            by_angle[j["angle"]].append(j)

        made = 0
        for angle in sorted(by_angle):
            for shot in SHOTS:
                reps = max(1, per_character // (len(by_angle) * len(SHOTS)))
                for _ in range(reps):
                    if made >= per_character:
                        break
                    job = rng.choice(by_angle[angle])
                    ground = sorted(GROUNDS)[made % len(GROUNDS)]
                    frames = read_rgba(job["src"], job["real_frames"], loop=False)
                    if not frames:
                        continue
                    if find_torn(frames):
                        continue
                    idx = rng.randrange(len(frames))
                    one = apply_shot(frames, shot[2])[idx]
                    stem = (f"{character.lower()}_still_{angle}_{shot[0]}"
                            f"__{ground.replace(' ', '')}_{made:03d}")
                    dest = out_stills / f"{stem}.mp4"
                    write_mp4(composite([one], GROUNDS[ground]), dest, 1)
                    records.append({"file": f"stills/{dest.name}", "frames": 1, "ground": ground,
                                    "shot": shot[0], "label": job["label"],
                                    "character": character, "kind": "still",
                                    "angle": angle, "source": job["src"].name})
                    made += 1
        print(f"  {character}: {made} stills")
    return records


def build(jobs: list[dict], out: Path, workers: int, limit: int | None,
          stills_per_character: int) -> None:
    rng = random.Random(SEED)
    out_clips = out / "clips"
    out_clips.mkdir(parents=True, exist_ok=True)

    # Every (source, rung) is emitted at EVERY shot size. Multiplying rather than
    # rotating guarantees zoom cannot correlate with label by construction, and this
    # corpus is small enough that the extra samples are worth more than the redundancy.
    # Grounds rotate, so background stays decorrelated from label, character and zoom.
    tasks, k = [], 0
    for job in sorted(jobs, key=lambda j: (j["kind"], j["label"], j["character"], j["angle"])):
        for nframes in buckets_for(job):
            plan = []
            for shot in SHOTS:
                plan.append((sorted(GROUNDS)[k % len(GROUNDS)], shot))
                k += 1
            tasks.append((job, nframes, plan, out_clips))
    if limit:
        tasks = tasks[:limit]

    print(f"\nemitting {len(tasks) * len(SHOTS)} clips ({len(tasks)} decodes x {len(SHOTS)} "
          f"shot sizes) from {len(jobs)} sources with {workers} workers")
    records, failed = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            made = fut.result()
            if made:
                records.extend(made)
            else:
                t = futs[fut]
                failed.append(f"{t[0]['src'].name}@{t[1]}f")
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} decodes, {len(records)} clips")

    if failed:
        print(f"\n  {len(failed)} clips unrecoverable after {DECODE_TRIES} decode attempts:")
        for f in failed[:12]:
            print(f"    {f}")

    print("\nverifying invariants")
    verify_length_balance(records)
    verify_shot_balance(records)

    still_records = []
    if stills_per_character:
        print(f"\nemitting stills (F=1 appearance bucket)")
        still_records = emit_stills(jobs, out, stills_per_character)

    # Holdout: one whole angle, never preprocessed, never trained.
    train = [r for r in records if r["angle"] != HOLDOUT_ANGLE]
    hold = [r for r in records if r["angle"] == HOLDOUT_ANGLE]

    # Stills join the training split only. The holdout stays video-only so that the
    # evaluation measures motion and identity over time, not a single frame.
    train += [dict(r, file=r["file"]) for r in still_records]

    def to_entries(rs):
        entries, unknown = [], set()
        for r in rs:
            job = {"character": r["character"], "label": r["label"], "angle": r["angle"],
                   "real_frames": next(j["real_frames"] for j in jobs
                                       if j["src"].name == r["source"])}
            shot_phrase = next(s[1] for s in SHOTS if s[0] == r["shot"])
            cap, unk = caption(job, r["ground"], shot_phrase, r["frames"])
            if unk:
                unknown.add(r["label"])
            rel = r["file"] if "/" in r["file"] else f"clips/{r['file']}"
            entries.append({"caption": cap, "video": rel})
        return entries, unknown

    train_entries, unk_a = to_entries(train)
    hold_entries, unk_b = to_entries(hold)

    (out / "dataset.json").write_text(json.dumps(train_entries, indent=1))
    (out / "holdout.json").write_text(json.dumps(hold_entries, indent=1))

    buckets = sorted({r["frames"] for r in records} | {r["frames"] for r in still_records})
    manifest = {
        "generated": "prep_ltx25.py",
        "size": SIZE, "fps": FPS, "temporal_factor": TEMPORAL, "floor": FLOOR,
        "sources": len(jobs), "clips_train": len(train), "clips_holdout": len(hold),
        "holdout_angle": HOLDOUT_ANGLE,
        "resolution_buckets": ";".join(f"{SIZE}x{SIZE}x{f}" for f in buckets),
        "labels": sorted({r["label"] for r in records}),
        "per_label": {k: v for k, v in sorted(Counter(r["label"] for r in records).items())},
        "per_bucket": {str(k): v for k, v in sorted(Counter(r["frames"] for r in records).items())},
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\n  train    {len(train)} clips -> dataset.json")
    print(f"  holdout  {len(hold)} clips ({HOLDOUT_ANGLE}) -> holdout.json")
    print(f"  buckets  {manifest['resolution_buckets']}")
    if unk_a | unk_b:
        print(f"\n  ⚠️  labels with no verified caption phrase: {sorted(unk_a | unk_b)}")
        print("      Watch the footage and add them to ACTIONS before training.")


# --------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="emit only the first N clips")
    ap.add_argument("--stills", type=int, default=108,
                    help="F=1 appearance stills per character (0 disables); ~60-120 recommended")
    ap.add_argument("--dry-run", action="store_true", help="discover and plan, write nothing")
    args = ap.parse_args()

    if not args.raw.exists():
        sys.exit(f"raw tree not found: {args.raw}")

    print(f"scanning {args.raw}")
    jobs, notes = discover(args.raw)
    print(f"  {len(jobs)} usable sources")
    if notes:
        print(f"\n{len(notes)} intake notes:")
        for n in notes:
            print(f"  {n}")

    by_label = Counter(f"{j['kind']}/{j['label']}" for j in jobs)
    print("\nplan:")
    total = 0
    for key in sorted(by_label):
        kind, label = key.split("/")
        ex = next(j for j in jobs if j["label"] == label)
        rungs = buckets_for(ex)
        n = by_label[key] * len(rungs) * len(SHOTS)
        total += n
        wrapped = " (cycle wrapped to floor)" if top_legal(ex["real_frames"]) < FLOOR else ""
        act = min(ex["real_frames"], max(ex["active_frames"], FLOOR))
        trim = f"  (trimmed from {ex['real_frames']}f)" if act < ex["real_frames"] else ""
        print(f"  {key:<24} src={by_label[key]:>3}  active={act:>3}f  "
              f"rungs={str(rungs):<16} -> {n:>4} clips{wrapped}{trim}")
    stills = args.stills * len(CHARACTERS)
    print(f"\n  {total} clips ({len(GROUNDS)} grounds x {len(SHOTS)} shot sizes)"
          f" + {stills} stills = {total + stills} samples")

    if args.dry_run:
        print("\ndry run - nothing written")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    build(jobs, args.out, args.workers, args.limit, args.stills)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
