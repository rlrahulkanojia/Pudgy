#!/usr/bin/env python3
"""
Expression training-set prep — v6, incremental over the whole `interation_3` delivery.

Supersedes prep_happy_v5.py, which hardcoded 7 Pax/happy clips. This one *scans*
`03_expression_clips/<Char>/<emotion>/` and builds from everything present, so each
new client drop is picked up by re-running it — no edits, no per-batch scripts.
The v5 pilot set is therefore included automatically, and the captions it generates
for those 7 clips are byte-identical to the 28 in dataset_happy.jsonl that v5 trained
on — so each run is a strict superset of the last, not a replacement.

Source clips are ProRes 4444 with a real alpha channel (yuva444p12le, 1080x1080,
24 fps). Two consequences, same as v5:

  1. A naive decode composites onto BLACK. We composite deliberately, onto flat
     pastel grounds, so nothing teaches a black void.
  2. Alpha lets the same performance be composited onto N backgrounds, which is
     what buys background-invariance rather than "this face on this background".

Frame budget is per-emotion because the deliveries differ in length (happy 21,
surprised 30, angry 40, neutral 60). Musubi needs 4k+1, so each emotion is
truncated to the largest legal count that fits, and each length becomes its own
[[datasets]] block (musubi buckets by frame count, and one block cannot hold two).

Output: 1024x1024 (NOT the source 1080 — Wan's 8x VAE + 2x2 patchify needs an even
latent side, and 1080/8 = 135 is odd), silent mp4 + captions + jsonl + dataset config.
"""
import argparse, hashlib, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 1024
FPS = 24

DEFAULT_SRC = Path("/Users/rahul/Documents/Projects/Saksham/Pudgy/Data/raw/iteration_3/03_expression_clips")
DEFAULT_OUT = Path("/Users/rahul/Documents/Projects/Saksham/Pudgy/Data/processed/v6_expressions_272")
WORKSPACE_CLIPS = "/workspace/data_v6/expressions_train"
WORKSPACE_CACHE = "/workspace/wan_cache/latents_expr"

# Shot-size ladder. The client spec asked for "Zoom: mix of close-up / medium / wide"
# and none arrived — every delivered clip is the same fixed framing, character filling
# ~91% of frame height. So we synthesise it from the alpha channel.
#
# ZOOM-OUT ONLY, deliberately. Measured across all 68 sources, the character's alpha
# bbox is a median 654x928 of 1024, and 10 clips already touch the top edge — so the
# largest centre-anchored zoom-IN that keeps the character whole is ~1.15x, which is
# visually indistinguishable from 1.0x. A real tight shot would have to cut the body.
#
# Compositing at a smaller scale onto the SAME flat ground is lossless in the way that
# matters: the ground is regenerated, not resampled, so no edge halos and no
# double-resample of the artwork (we re-decode from the 1080 source at the target size).
ZOOMS = {
    1.00: "static close-up shot",
    0.75: "static medium shot",
    0.55: "static wide shot",
}

# Flat pastel grounds — identical set to v5, so the two runs stay comparable.
BACKGROUNDS = {
    "white": ("plain white studio background", (255, 255, 255)),
    "blue":  ("plain pastel blue background",  (198, 222, 241)),
    "peach": ("plain pastel peach background", (247, 219, 205)),
    "mint":  ("plain pastel mint background",  (206, 235, 219)),
}

# Identity anchors, verbatim from the v1/v2 75-clip caption convention. Deliberately
# NOT v4's rare token (pxngn0): we continue-train the v2/v5 lineage, so the captions
# must sit in the same distribution.
ANCHORS = {
    "Pax":   "Pax, a short round blue penguin",
    "Polly": "Polly, a short round pink penguin",
}
STYLE = ("A 2D cartoon animation in the Pudgy Penguins style, with thick clean "
         "black outlines and flat pastel colors, showing ")

# Per-emotion action clause + the musubi-legal frame budget for that delivery.
EMOTIONS = {
    "happy": {
        "frames": 21,   # source 21
        "action": ("breaking into a big happy smile, beak opening into a wide joyful "
                   "grin, eyebrows lifting, cheeks lifting"),
    },
    "surprised": {
        "frames": 29,   # source 30
        "action": ("eyes widening in surprise, brows shooting up, beak opening into a "
                   "small round gasp, body pulling back slightly"),
    },
    "angry": {
        "frames": 37,   # source 40
        "action": ("scowling into an angry glare, brows dropping and pressing together, "
                   "beak set in a hard frown, shoulders squaring"),
    },
    "neutral": {
        "frames": 57,   # source 60
        "action": ("holding a calm neutral expression, eyes open and steady, beak "
                   "closed, only a soft idle settle"),
    },
}

SIDE = {"L": "seen from its left side, profile view",
        "R": "seen from its right side, profile view"}
QUARTER = {
    1: "turned slightly to its {side}, three-quarter front view",
    2: "turned further to its {side}, wide three-quarter view",
    3: "turned strongly to its {side}, near-profile three-quarter view",
}
NOISE = {"PAX", "POLLY", "EXPRESSION", "HAPPY", "ANGRY", "NEUTRAL", "SURPRISE", "SURPRISED"}


def parse_angle(stem):
    """Filename -> (slug, caption clause). The deliveries use four different angle
    vocabularies (FRONT/FR, QF_L, QF1_L, QF_L2, SIDE_L, Right); normalise them all."""
    toks = [t for t in re.split(r"[_\-]", stem.upper()) if t and t not in NOISE]
    if not toks:
        return None, None
    if toks[0] in ("FRONT", "FR"):
        return "FRONT", "facing the camera directly, front view"
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
        deg = int(digits[0]) if digits else 1
        deg = min(max(deg, 1), 3)
        side = "left" if d == "L" else "right"
        return f"QF{deg}_{d}", QUARTER[deg].format(side=side)
    return None, None


def probe_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


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

    One resample, straight from the 1080 source to round(SIZE*zoom) — not
    1080 -> 1024 -> target, which would soften the outlines twice. The padding is
    fully transparent, so the flat ground shows through it after compositing.
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

    Round-robin over a stable enumeration rather than a hash: it makes the marginal
    counts near-exact by construction, and `verify_zoom_balance` asserts the result.
    Zoom must NOT track emotion, character, background or angle — a factor that
    predicts the label is exactly the confound that clip-length already introduced
    (see Training_Approach_v6 section 4.1).
    """
    ladder = sorted(ZOOMS)                       # [0.55, 0.75, 1.00]
    for i, j in enumerate(sorted(jobs, key=lambda x: (x["char"], x["emotion"], x["angle"]))):
        j["zooms"] = {tag: ladder[(i + k) % len(ladder)]
                      for k, tag in enumerate(BACKGROUNDS)}
    return jobs


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


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(src):
    """Every <Char>/<emotion>/*.mov we know how to caption. Unknown emotions and
    unparseable angles are reported, never silently dropped."""
    jobs, skipped = [], []
    for char_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        char = char_dir.name
        if char not in ANCHORS:
            skipped.append((str(char_dir), "unknown character")); continue
        for emo_dir in sorted(p for p in char_dir.iterdir() if p.is_dir()):
            emo = emo_dir.name
            if emo not in EMOTIONS:
                clips = list(emo_dir.glob("*.mov"))
                if clips:
                    skipped.append((f"{char}/{emo}", f"no caption recipe ({len(clips)} clips)"))
                continue
            for clip in sorted(emo_dir.glob("*.mov")):
                slug, clause = parse_angle(clip.stem)
                if not slug:
                    skipped.append((str(clip), "unparseable angle")); continue
                jobs.append({"src": clip, "char": char, "emotion": emo,
                             "angle": slug, "clause": clause})

    # Two takes can legitimately parse to the same angle — neutral ships both
    # QF_R and QF_R1, and both really are "slightly turned right". Sharing the
    # caption is right; sharing the output filename is not, or one take silently
    # overwrites the other and vanishes from training. Suffix them by take.
    collisions, groups = [], {}
    for j in jobs:
        groups.setdefault((j["char"], j["emotion"], j["angle"]), []).append(j)
    for (char, emo, slug), g in groups.items():
        if len(g) > 1:
            g.sort(key=lambda x: x["src"].name)
            for i, j in enumerate(g, 1):
                j["angle"] = f"{slug}_t{i}"
            collisions.append((f"{char}/{emo}/{slug}", [x["src"].name for x in g]))
    return jobs, skipped, collisions


def verify_zoom_balance(jobs, tol=0.34):
    """Fail loudly if zoom correlates with any label the model could shortcut on.

    Clip length already predicts emotion perfectly in this dataset; a second such
    shortcut would be a self-inflicted repeat of that mistake. `tol` is the largest
    allowed deviation from an even split within any single factor level.
    """
    import collections
    ladder = sorted(ZOOMS)
    even = 1.0 / len(ladder)
    factors = {"emotion": lambda j, t_: j["emotion"], "character": lambda j, t_: j["char"],
               "background": lambda j, t_: t_, "angle": lambda j, t_: j["angle"].split("_t")[0]}
    problems = []
    for fname, key in factors.items():
        tab = collections.defaultdict(collections.Counter)
        for j in jobs:
            for tag in BACKGROUNDS:
                tab[key(j, tag)][j["zooms"][tag]] += 1
        for level, c in sorted(tab.items()):
            n = sum(c.values())
            for z in ladder:
                frac = c[z] / n
                if abs(frac - even) > tol:
                    problems.append(f"{fname}={level}: zoom {z} is {frac:.0%} of {n} (want ~{even:.0%})")
    return problems


def caption(job, bg_desc, zoom):
    e = EMOTIONS[job["emotion"]]
    return (f"{STYLE}{ANCHORS[job['char']]}, {e['action']}; {job['emotion']} expression; "
            f"{ZOOMS[zoom]}, eye level, {job['clause']}; {bg_desc}.")


def process(job, clips_dir):
    want = EMOTIONS[job["emotion"]]["frames"]
    have = probe_frames(job["src"])
    if have < want:
        raise RuntimeError(f"{job['src'].name}: {have} frames < required {want}")
    # Decode once per distinct zoom this clip needs, not once per background.
    by_zoom = {}
    for tag in BACKGROUNDS:
        by_zoom.setdefault(job["zooms"][tag], []).append(tag)

    out = []
    for zoom, tags in sorted(by_zoom.items()):
        frames = read_rgba_zoomed(job["src"], want, zoom)
        for tag in tags:
            bg_desc, rgb = BACKGROUNDS[tag]
            name = f"{job['char'].lower()}_{job['emotion']}_{job['angle']}__{tag}.mp4"
            dest = clips_dir / name
            write_mp4(composite(frames, rgb), dest)
            out.append({"name": name, "dest": dest,
                        "caption": caption(job, bg_desc, zoom),
                        "frames": want, "background": tag, "zoom": zoom})
    return {"job": job, "outputs": out, "src_frames": have, "used_frames": want}


def emit(records, out, prefix, cache_root, suffix):
    """One jsonl per frame-bucket + a toml with one [[datasets]] block per bucket."""
    buckets = {}
    for r in records:
        for o in r["outputs"]:
            buckets.setdefault(o["frames"], []).append(
                {"video_path": f"{prefix.rstrip('/')}/{o['name']}", "caption": o["caption"]})
    lines = ["# Musubi dataset config — v6 expressions (auto-generated by prep_expressions_v6.py).",
             "# One [[datasets]] block per frame length: musubi buckets by frame count and the",
             "# deliveries differ (happy 21 / surprised 29 / angry 37 / neutral 57).",
             "# 1024x1024, not the source 1080: Wan's 8x VAE + 2x2 patchify needs an even latent side.",
             "", "[general]", "resolution = [1024, 1024]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    for n in sorted(buckets):
        jp = out / f"dataset_expr_f{n}{suffix}.jsonl"
        with open(jp, "w") as f:
            for rec in buckets[n]:
                f.write(json.dumps(rec) + "\n")
        jsonl_ref = jp if not suffix else f"{WORKSPACE_CLIPS.rsplit('/', 1)[0]}/{jp.name}"
        lines += ["[[datasets]]",
                  f'video_jsonl_file = "{jsonl_ref}"',
                  f'cache_directory = "{cache_root}/f{n}"',
                  f"target_frames = [{n}]",
                  'frame_extraction = "head"',
                  "num_repeats = 1",
                  f"# {len(buckets[n])} clips", ""]
    cfg = out / f"dataset_config_expressions_v6{suffix}.toml"
    cfg.write_text("\n".join(lines))
    return cfg, {n: len(v) for n, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-zoom", action="store_true",
                    help="disable the shot-size ladder; every clip at 1.0x close-up "
                         "(reproduces the pre-zoom set)")
    args = ap.parse_args()

    jobs, skipped, collisions = discover(args.src)
    if args.no_zoom:
        for j in jobs:
            j["zooms"] = {tag: 1.00 for tag in BACKGROUNDS}
        print("zoom ladder DISABLED (--no-zoom): every clip at 1.0x close-up")
    else:
        assign_zooms(jobs)
        problems = verify_zoom_balance(jobs)
        if problems:
            for m in problems:
                print(f"  ! zoom imbalance {m}")
            sys.exit("refusing to run: zoom correlates with a label (see above)")
        import collections as _c
        dist = _c.Counter(z for j in jobs for z in j["zooms"].values())
        print("zoom ladder: " + " · ".join(
            f"{z:.2f}x {ZOOMS[z].replace('static ', '')} = {dist[z]}" for z in sorted(ZOOMS))
            + "   (balanced across emotion/character/background/angle)")
    print(f"discovered {len(jobs)} source clips -> {len(jobs) * len(BACKGROUNDS)} training clips")
    for what, why in skipped:
        print(f"  ! skipped {what}: {why}")
    for slug, srcs in collisions:
        print(f"  ~ disambiguated {slug}: {srcs} -> _t1.._t{len(srcs)}")
    by = {}
    for j in jobs:
        by[f"{j['char']}/{j['emotion']}"] = by.get(f"{j['char']}/{j['emotion']}", 0) + 1
    for k in sorted(by):
        print(f"    {k:<20} {by[k]} clips x {len(BACKGROUNDS)} bg = {by[k] * len(BACKGROUNDS)}")
    if args.dry_run:
        return

    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{j['char'].lower()}_{j['emotion']}_{j['angle']}" for j in jobs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        sys.exit(f"refusing to run: duplicate output names would overwrite: {dupes}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda j: process(j, clips_dir), jobs))

    written = len(list(clips_dir.glob("*.mp4")))
    expected = len(jobs) * len(BACKGROUNDS)
    if written != expected:
        sys.exit(f"expected {expected} clips on disk, found {written}")
    print(f"\nwrote {sum(len(r['outputs']) for r in records)} clips -> {clips_dir}")

    cfg_local, counts = emit(records, args.out, str(clips_dir),
                             str(args.out / "cache"), "")
    cfg_ws, _ = emit(records, args.out, WORKSPACE_CLIPS, WORKSPACE_CACHE, ".workspace")

    manifest = {
        "generated_by": "prep_expressions_v6.py",
        "source_root": str(args.src),
        "size": SIZE, "fps": FPS, "backgrounds": list(BACKGROUNDS),
        "zoom_ladder": {str(z): ZOOMS[z] for z in sorted(ZOOMS)},
        "clips": [{"source": str(r["job"]["src"]), "source_md5": md5(r["job"]["src"]),
                   "character": r["job"]["char"], "emotion": r["job"]["emotion"],
                   "angle": r["job"]["angle"], "source_frames": r["src_frames"],
                   "used_frames": r["used_frames"],
                   "outputs": [{"name": o["name"], "background": o["background"],
                                "zoom": o["zoom"]} for o in r["outputs"]]} for r in records],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"buckets: {counts}")
    print(f"config (local)     : {cfg_local}")
    print(f"config (gpu box)   : {cfg_ws}")
    print(f"manifest           : {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
