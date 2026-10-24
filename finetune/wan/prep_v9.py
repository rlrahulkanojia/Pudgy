#!/usr/bin/env python3
"""
v9c-phone training-set prep - the first PROP stage: the iFin phone + replay.

Plan: training_approach/v9/Training_Approach_v9.md (stage v9c-phone) and
training_approach/v9/Assets_v9.md. Built ON TOP of prep_v8.py, not as a fork: discovery
of iteration_3-5, angle parsing, damage checks, zoom/length ladders, compositing, captions
and encoding are imported from prep_v8 unchanged, so the replay clips are exactly the
distribution v8 learned. What v9 adds, and why:

  * iteration_6 (2026-10-24). A whole-project re-drop: 257 of its 323 clips are md5
    re-sends of iteration_3-5. Every flat file must hash to a known clip; anything new
    outside the handled folders is reported, never silently trained or dropped.
  * V1/V2 folders. CONFUSED/V2 is a re-export of the SAME performance (fixes the 4
    damaged V1 clips) -> it REPLACES every earlier confused source; training both would
    double-count one performance. MOTION_SITTING/V2 is a NEW performance -> a second take
    (`_t2`) beside V1.
  * PROPS. In-use prop footage is described by the asset registry
    (finetune/wan/assets/registry.toml): folder -> label + action clause, with the prop's
    phrase substituted. Captions are compiled from the registry, never typed here.
  * Clean WINDOWS, not head rungs. 11 Polly phone clips lose frames mid-clip (some at
    frames 1-5), and every phone clip ends on a 10-frame hold (frames 64-73 = frame 63).
    Each prop source trains on its longest run of real, animated frames, found from frame
    TIMESTAMPS. Decoding uses passthrough timing and maps frames by timestamp: ffmpeg's
    constant-rate output fills a gap with duplicates and can shift the next frame by one
    (seen on POLLY_..._STANDING_SIDE_R), so slicing by output index is wrong after a gap.
  * REPLAY. A stage is new data + a stratified share of everything before it (v9 section
    4): one output per (character, label) at its kind's common bucket first, so length
    never predicts a label, then the rest sampled across angles, grounds and rungs.
  * Interactions (two-character) are discovered, with the client-approved rename applied,
    and reported - they belong to v9b, not this stage.

Stills (the 24 phone views) are NOT trained: in image-to-video a 1-frame sample is
conditioned on itself, so it carries almost no signal. They are the inference keyframe
pixels and the eval reference instead (Assets_v9 section 4.1).

    PUDGY_DATA=/workspace/Data python finetune/wan/prep_v9.py --dry-run
    PUDGY_DATA=/workspace/Data python finetune/wan/prep_v9.py --workers 8
"""
import argparse
import collections
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "assets"))
import prep_v8 as P  # noqa: E402
import registry as R  # noqa: E402

DATA = P.DATA
SRC_IT6 = DATA / "raw/iteration_6/Samples"
OLD_ROOTS = [DATA / f"raw/iteration_{i}" for i in (3, 4, 5)]
OUT_ROOT = DATA / "processed"
STAGE = "v9c_phone"
WS_DATA = "/workspace/data_v9c"                 # the built set is linked here on the box
WS_CACHE = "/workspace/wan_cache/latents_v9c"

REG = R.load()
CHARS = {"PAX": "Pax", "POLLY": "Polly"}

# The props' shortest rung. 21 = prep_v8's expression common bucket, so f21 carries every
# expression AND every prop label: length alone cannot predict "phone".
PROP_FLOOR = P.COMMON["expression"]

# Raw is never renamed: corrections are keyed by md5, so a re-send of the same bytes is
# corrected too. Interactions are not built in this stage; the overrides are applied when
# they are discovered and recorded in the manifest for v9b.
OVERRIDES = {
    "652c16f45d4f212de7f3bb898f0db2cb": {
        "file": "INTERACTION_CLIPS/WALKING/INTERACTION_WAVING_SIDE_R.mov",
        "label": "walking", "angle": "SIDE_R",
        "why": "client-approved rename 2026-10-24: 17 frames like walking (waving is 26); "
               "walking had no SIDE_R"},
    "9133f3df65dad36ceb5b76fd224b4ae4": {
        "file": "INTERACTION_CLIPS/HUGGING/INTERACTION_HUGGING__QF3_L.mov",
        "label": "hugging", "angle": "QF3_L", "why": "doubled underscore"},
}

# How a V2 folder relates to what came before.
VERSIONS = {
    ("expression", "confused"): "replace",   # same performance, clean re-export
    ("motion", "sitting"): "take",           # a new performance: second take
}

ANGLE_TAIL = re.compile(r"_(FRONT|SIDE_[LR]|QF[123]_[LR])$")


def tail_angle(stem):
    """Angle from the END of a filename: props and interactions carry extra tokens
    (PRP, PHONE, TEXTING, INTERACTION, HUGGING) that prep_v8's parser would read as the
    angle. All their names use the explicit FRONT / SIDE_x / QFn_x convention."""
    m = ANGLE_TAIL.search(stem.upper().replace("__", "_"))
    return m.group(1) if m else None


def char_of(path, folder_char=None):
    """Trust the filename over the folder (prep_v8 rule: HAPPY/POLLY/PAX_HAPPY_SIDE_R)."""
    named = next((v for k, v in CHARS.items() if path.stem.upper().startswith(k)), None)
    return named or CHARS.get((folder_char or "").upper())


# ---------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------
def discover_v9():
    """All sources this stage can use, plus a full report of everything it did not use."""
    base, skipped, dupes, collisions, rebased = P.discover()          # iteration_3-5, as v8
    rep = {"v8_skipped": skipped, "v8_dupes": len(dupes), "v8_collisions": collisions,
           "v8_rebased": rebased, "it6_resent": 0, "it6_unknown_new": [],
           "superseded": [], "v2_dupes": [], "interactions": [], "overrides_applied": [],
           "props_skipped": []}
    known = {P.md5(p) for root in OLD_ROOTS for p in root.rglob("*.mov")}

    v2 = []
    for cat, kind in (("EXPRESSIONS", "expression"), ("MOTION_CLIPS", "motion")):
        root = SRC_IT6 / cat
        if not root.exists():
            continue
        for lab_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if kind == "expression":
                label = {"SURPRISE": "surprised"}.get(lab_dir.name, lab_dir.name.lower())
            else:
                label = lab_dir.name.upper().replace("MOTION_", "").lower()
            for f in sorted(lab_dir.rglob("*.mov")):
                rel = f.relative_to(lab_dir).parts
                if rel[0] != "V2":
                    if P.md5(f) in known:
                        rep["it6_resent"] += 1
                    else:
                        rep["it6_unknown_new"].append(str(f.relative_to(DATA)))
                    continue
                policy = VERSIONS.get((kind, label))
                if policy is None:
                    rep["it6_unknown_new"].append(f"{f.relative_to(DATA)} (V2 with no policy)")
                    continue
                slug, clause = P.parse_angle(f.stem)
                char = char_of(f, rel[1] if len(rel) > 2 else None)
                if not slug or slug.startswith("QF?") or not char:
                    rep["it6_unknown_new"].append(f"{f.relative_to(DATA)} (unparseable)")
                    continue
                v2.append({"src": f, "md5": P.md5(f), "char": char, "label": label,
                           "kind": kind, "angle": slug, "clause": clause,
                           "version": "V2", "policy": policy})

    jobs = list(base)
    seen = {j["md5"] for j in jobs}
    for (kind, label), policy in VERSIONS.items():
        mine = [j for j in v2 if (j["kind"], j["label"]) == (kind, label)]
        if not mine:
            continue
        if policy == "replace":
            gone = [j for j in jobs if (j["kind"], j["label"]) == (kind, label)]
            rep["superseded"] += [str(j["src"].relative_to(DATA)) for j in gone]
            jobs = [j for j in jobs if j not in gone]
        taken = {(j["char"], j["label"], j["angle"].split("_t")[0]) for j in jobs}
        v2_seen = set()
        for j in mine:
            if j["md5"] in v2_seen or (policy == "take" and j["md5"] in seen):
                rep["v2_dupes"].append(str(j["src"].relative_to(DATA))); continue
            v2_seen.add(j["md5"])
            if policy == "take" and (j["char"], j["label"], j["angle"]) in taken:
                j["angle"] = f"{j['angle']}_t2"
            jobs.append(j)

    # Interactions: discovered and reported (v9b), overrides applied.
    inter = SRC_IT6 / "INTERACTION_CLIPS"
    for f in sorted(inter.rglob("*.mov")) if inter.exists() else []:
        h = P.md5(f)
        ov = OVERRIDES.get(h)
        label = ov["label"] if ov else f.parent.name.lower()
        angle = ov["angle"] if ov else tail_angle(f.stem)
        if ov:
            rep["overrides_applied"].append({"file": str(f.relative_to(DATA)), **ov})
        rep["interactions"].append({"file": str(f.relative_to(DATA)), "md5": h,
                                    "label": label, "angle": angle})

    props = discover_props(rep)
    return jobs, props, rep


def discover_props(rep):
    out = []
    for aid, a in REG.items():
        if a.get("kind") != "prop" or not a.get("clips"):
            continue
        root = DATA / a["clips"]["clips_dir"]
        for act in a.get("actions", []):
            for f in sorted((root / act["folder"]).rglob("*.mov")):
                slug = tail_angle(f.stem)
                char = char_of(f, f.parent.name)
                if not slug or not char:
                    rep["props_skipped"].append(f"{f.relative_to(DATA)}: angle/character unreadable")
                    continue
                out.append({"src": f, "md5": P.md5(f), "char": char, "label": act["label"],
                            "kind": "prop", "asset": aid, "act": act, "angle": slug,
                            "clause": P.clause_for_slug(slug),
                            "animated": int(a["clips"]["animated_frames"])})
    return out


# ---------------------------------------------------------------------------------------
# Prop windows
# ---------------------------------------------------------------------------------------
def frame_indices(path):
    """Indices of the frames the stream really contains (timestamps are ground truth)."""
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    return sorted({int(round(float(t) * P.FPS)) for t in out if t.strip()})


def runs_of(idx, limit):
    runs, start, prev = [], None, None
    for k in [i for i in idx if i < limit] + [None]:
        if k is not None and prev is not None and k == prev + 1:
            prev = k; continue
        if start is not None:
            runs.append((start, prev - start + 1))
        start = prev = k
    return runs


def prop_rungs(length):
    """prep_v8's ladder (floor -> natural, 3 evenly spaced legal lengths), with the
    natural length capped by the clean run."""
    cap = max([f for f in range(5, length + 1) if (f - 1) % 4 == 0], default=0)
    opts = [f for f in range(PROP_FLOOR, cap + 1) if (f - 1) % 4 == 0]
    if len(opts) <= P.LADDER_STEPS:
        return opts
    idx = [round(i * (len(opts) - 1) / (P.LADDER_STEPS - 1)) for i in range(P.LADDER_STEPS)]
    return sorted({opts[i] for i in idx})


def preflight_prop(job):
    idx = frame_indices(job["src"])
    runs = runs_of(idx, job["animated"])
    if not runs:
        return False, "no decodable frames"
    start, length = max(runs, key=lambda r: (r[1], -r[0]))
    job["missing"] = sorted(set(range(job["animated"])) - set(idx))
    job["window"] = [start, length]
    job["rungs"] = prop_rungs(length)
    if not job["rungs"]:
        return False, f"longest clean run is {length} frames, below the f{PROP_FLOOR} floor"
    return True, None


_FPS_MODE = []


def _passthrough_flag():
    """`-fps_mode` is ffmpeg >= 5.1; older builds (Ubuntu 22.04 ships 4.4) use `-vsync`."""
    if not _FPS_MODE:
        h = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True,
                           text=True).stdout
        _FPS_MODE.extend(["-fps_mode", "passthrough"] if "-fps_mode" in h
                         else ["-vsync", "passthrough"])
    return _FPS_MODE


def read_rgba_indexed(path, size):
    """Decode every real frame (no gap filling) and its index from its timestamp."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", str(path),
                        "-vf", f"showinfo,scale={size}:{size}", *_passthrough_flag(),
                        "-pix_fmt", "rgba", "-f", "rawvideo", "-"], capture_output=True)
    v = np.frombuffer(p.stdout, np.uint8).reshape(-1, size, size, 4)
    pts = re.findall(r"Parsed_showinfo.*? pts_time:\s*([0-9.]+)", p.stderr.decode("utf-8", "replace"))
    return v, [int(round(float(t) * P.FPS)) for t in pts]


def decode_window(job, zoom):
    """The window [start, start + max rung) at `zoom`, alpha kept, torn frames retried."""
    start, n = job["window"][0], max(job["rungs"])
    size = P.SIZE if zoom >= 0.999 else int(round(P.SIZE * zoom)) // 2 * 2
    want = list(range(start, start + n))
    last = None
    for _ in range(P.DECODE_TRIES):
        v, idx = read_rgba_indexed(job["src"], size)
        if len(v) != len(idx):
            last = f"{len(v)} frames but {len(idx)} timestamps"; continue
        pos = {k: i for i, k in enumerate(idx)}
        if any(k not in pos for k in want):
            last = f"frames {[k for k in want if k not in pos]} did not decode"; continue
        arr = v[[pos[k] for k in want]]
        torn = P.find_torn(arr[..., 3], n)
        if torn:
            last = f"torn frames {[want[t] for t in torn]}"; continue
        if size != P.SIZE:
            canvas = np.zeros((n, P.SIZE, P.SIZE, 4), np.uint8)
            off = (P.SIZE - size) // 2
            canvas[:, off:off + size, off:off + size] = arr
            arr = canvas
        return arr
    raise RuntimeError(f"{job['src'].name}: {last} after {P.DECODE_TRIES} decodes")


# ---------------------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------------------
def anchor(char):
    return REG[char.lower()]["phrase"]


def prop_caption(job, bg_desc, zoom):
    """prep_v8's structure - style, anchor, action; label; shot, eye level, angle; ground -
    with the action and its prop wording from the registry. The texting is continuous
    (no arc), so a shorter rung is not marked as "opening frames only"."""
    act = job["act"]
    return (f"{P.STYLE}{anchor(job['char'])}, {R.action_clause(REG, job['asset'], act)}; "
            f"{act['label_text']}; {P.ZOOMS[zoom]}, eye level, {job['clause']}; {bg_desc}.")


def out_name(job, tag, n):
    return f"{job['char'].lower()}_{job['label']}_{job['angle']}__{tag}_f{n}.mp4"


# ---------------------------------------------------------------------------------------
# Replay selection
# ---------------------------------------------------------------------------------------
def replay_candidates(job):
    return [(tag, n) for tag in P.BACKGROUNDS for n in P.buckets_for(job)]


def select_replay(pool, total, seed):
    """`total` outputs from the replay pool, stratified by (character, label).

    Each group first gets one output at its kind's common bucket (f13 motion / f21
    expression) - the de-confounding property prep_v8 asserts - then the remaining quota
    is drawn at random across angles, grounds and rungs.
    """
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for j in sorted(pool, key=lambda x: (x["char"], x["label"], x["angle"])):
        for tag, n in replay_candidates(j):
            groups[(j["char"], j["label"])].append((j, tag, n))
    keys = sorted(groups)
    quota = {k: total // len(keys) for k in keys}
    for k in rng.sample(keys, total - sum(quota.values())):
        quota[k] += 1
    picked = []
    for k in keys:
        cands = groups[k][:]
        rng.shuffle(cands)
        common = P.COMMON[cands[0][0]["kind"]]
        first = next((c for c in cands if c[2] == common), None)
        take = ([first] if first else []) + [c for c in cands if c is not first]
        picked += take[:max(quota[k], 1 if first else 0)]
    return picked


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def build_prop(job, clips_dir, reencode):
    outs, todo = [], {}
    for tag in P.BACKGROUNDS:
        z = job["zooms"][tag]
        for n in job["rungs"]:
            name = out_name(job, tag, n)
            if reencode or not (clips_dir / name).exists():
                todo.setdefault(z, []).append((tag, n))
            outs.append({"name": name, "background": tag, "zoom": z, "frames": n})
    try:
        decoded = {z: decode_window(job, z) for z in todo}
    except RuntimeError as e:
        return {"job": job, "outputs": [], "error": str(e)}
    for z, items in todo.items():
        for tag, n in items:
            P.write_mp4(P.composite(decoded[z][:n], P.BACKGROUNDS[tag][1]),
                        clips_dir / out_name(job, tag, n))
    for o in outs:
        o["caption"] = prop_caption(job, P.BACKGROUNDS[o["background"]][0], o["zoom"])
        o["truncated"] = False
    return {"job": job, "outputs": outs, "error": None, "role": "new"}


def build_replay(job, picks, clips_dir, reencode):
    """Only the selected (ground, rung) outputs of one v8-kind source, encoded exactly as
    prep_v8.process would (same decode, tiling, compositing and caption)."""
    by_zoom = collections.defaultdict(list)
    for tag, n in picks:
        by_zoom[job["zooms"][tag]].append((tag, n))
    need = P.decode_need(job)
    outs = []
    try:
        decoded = {}
        for z, items in by_zoom.items():
            if reencode or any(not (clips_dir / out_name(job, t, n)).exists() for t, n in items):
                decoded[z] = P.decode_checked(job, need, z)
    except RuntimeError as e:
        return {"job": job, "outputs": [], "error": str(e)}
    r = P.recipe(job)
    for z, items in by_zoom.items():
        for tag, n in items:
            name = out_name(job, tag, n)
            tiled = n == P.TILE_FRAMES and P.tiles(job)
            if z in decoded:
                src = (P.tile(decoded[z][:job["src_frames"]], n, job["seam_ratio"]) if tiled
                       else decoded[z][:n])
                P.write_mp4(P.composite(src, P.BACKGROUNDS[tag][1]), clips_dir / name)
            bg_desc = P.BACKGROUNDS[tag][0]
            outs.append({"name": name, "background": tag, "zoom": z, "frames": n,
                         "truncated": n < r["frames"], "tiled": tiled,
                         "caption": P.caption(job, bg_desc, z, n)})
    return {"job": job, "outputs": outs, "error": None, "role": "replay"}


def emit(records, out, prefix, cache_root, suffix):
    buckets = collections.defaultdict(list)
    for rec in records:
        for o in rec["outputs"]:
            buckets[o["frames"]].append({"video_path": f"{prefix.rstrip('/')}/{o['name']}",
                                         "caption": o["caption"]})
    lines = [f"# Musubi dataset config - {STAGE} (auto-generated by prep_v9.py).",
             "# New prop clips + stratified replay; one [[datasets]] block per frame length.",
             f"# f{P.COMMON['motion']} carries every replayed motion label, f{PROP_FLOOR} every "
             "expression AND prop label.", "",
             "[general]", f"resolution = [{P.SIZE}, {P.SIZE}]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    for n in sorted(buckets):
        jp = out / f"dataset_{STAGE}_f{n}{suffix}.jsonl"
        with open(jp, "w") as f:
            for row in buckets[n]:
                f.write(json.dumps(row) + "\n")
        ref = jp if not suffix else f"{prefix.rsplit('/', 1)[0]}/{jp.name}"
        lines += ["[[datasets]]", f'video_jsonl_file = "{ref}"',
                  f'cache_directory = "{cache_root}/f{n}"', f"target_frames = [{n}]",
                  'frame_extraction = "head"', "num_repeats = 1",
                  f"# {len(buckets[n])} clips", ""]
    cfg = out / f"dataset_config_{STAGE}{suffix}.toml"
    cfg.write_text("\n".join(lines))
    return cfg, {n: len(v) for n, v in sorted(buckets.items())}


def check_lengths(records):
    """No frame length may predict a label: every label of a kind must reach its kind's
    common bucket, and every prop label the f21 floor (shared with the expressions)."""
    at = collections.defaultdict(set)
    labels = collections.defaultdict(set)
    for rec in records:
        j = rec["job"]
        for o in rec["outputs"]:
            at[o["frames"]].add(j["label"])
        labels[j["kind"]].add(j["label"])
    probs = []
    floors = {"motion": P.COMMON["motion"], "expression": P.COMMON["expression"],
              "prop": PROP_FLOOR}
    for kind, labs in labels.items():
        miss = labs - at.get(floors[kind], set())
        if miss:
            probs.append(f"{kind}: f{floors[kind]} is missing {sorted(miss)}")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay-frac", type=float, default=0.20,
                    help="share of the stage's clips that are replay (v9 section 4: 0.20)")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--reencode", action="store_true")
    ap.add_argument("--holdout-angle", nargs="*", default=None, metavar="ANGLE")
    a = ap.parse_args()

    probs = R.lint(REG, DATA)
    if probs:
        sys.exit("registry lint failed:\n  " + "\n  ".join(probs))
    for c in CHARS.values():
        if anchor(c) != P.ANCHORS[c]:
            sys.exit(f"registry anchor for {c} differs from prep_v8 - the base was trained on prep_v8's")
    if a.holdout_angle is not None:
        P.HOLDOUT_ANGLES.clear(); P.HOLDOUT_ANGLES.update(a.holdout_angle)

    pool, props, rep = discover_v9()
    held = lambda j: j["angle"].split("_t")[0] in P.HOLDOUT_ANGLES  # noqa: E731
    holdout = [j for j in pool + props if held(j)]
    pool = [j for j in pool if not held(j)]
    props = [j for j in props if not held(j)]
    print(f"iteration_6: {rep['it6_resent']} flat clips are md5 re-sends of iteration_3-5")
    for f in rep["it6_unknown_new"]:
        print(f"  ! NEW iteration_6 file not handled by this stage: {f}")
    print(f"superseded by CONFUSED/V2: {len(rep['superseded'])} earlier confused sources")
    print(f"V2 byte-copies of V1 dropped: {len(rep['v2_dupes'])}")
    print(f"interactions (v9b, not built): {len(rep['interactions'])} "
          f"({len(rep['overrides_applied'])} filename overrides applied)")
    for o in rep["overrides_applied"]:
        print(f"  ° {o['file']} -> {o['label']} {o['angle']} ({o['why']})")
    print(f"holdout {sorted(P.HOLDOUT_ANGLES)}: {len(holdout)} sources "
          f"({sum(j['kind'] == 'prop' for j in holdout)} prop)")

    # Prop windows.
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        verdicts = list(ex.map(preflight_prop, props))
    dropped = [(j, why) for j, (ok, why) in zip(props, verdicts) if not ok]
    props = [j for j, (ok, _) in zip(props, verdicts) if ok]
    for j in holdout:
        if j["kind"] == "prop":
            preflight_prop(j)
    for j, why in dropped:
        print(f"  ✗ prop {j['src'].name}: {why}")
    print(f"\nprops: {len(props)} training sources")
    for j in props:
        flag = f"  missing {j['missing']}" if j["missing"] else ""
        print(f"    {j['src'].name:<48} window {j['window'][0]:>2}+{j['window'][1]:<2} "
              f"rungs {j['rungs']}{flag}")
    P.assign_zooms(props)
    zp = P.verify_zoom_balance(props)
    if zp:
        sys.exit("zoom correlates with a prop label:\n  " + "\n  ".join(zp))
    n_new = sum(len(j["rungs"]) for j in props) * len(P.BACKGROUNDS)

    # Replay pool: v8's sources (+ V2), preflighted exactly as v8 did.
    print(f"\nreplay pool: preflighting {len(pool)} sources ...")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        pv = list(ex.map(P.preflight, pool))
    P.derive_frame_budgets(pool, [v[2] for v in pv])
    pool_dropped = [(j, why) for j, (ok, why, _n) in zip(pool, pv) if not ok]
    pool = [j for j, (ok, _w, _n) in zip(pool, pv) if ok]
    for j, why in pool_dropped:
        print(f"  ✗ replay source {j['src'].name}: {why}")
    P.assign_zooms(pool)
    n_replay = round(n_new * a.replay_frac / (1 - a.replay_frac)) if a.replay_frac > 0 else 0
    picks = select_replay(pool, n_replay, a.seed) if n_replay else []
    by_job = collections.defaultdict(list)
    for j, tag, n in picks:
        by_job[id(j)].append((tag, n))
    replay_jobs = [j for j in pool if id(j) in by_job]
    groups = collections.Counter(f"{j['char']}/{j['label']}" for j, _t, _n in picks)
    print(f"replay: {len(picks)} clips from {len(replay_jobs)} sources over {len(groups)} "
          f"(character, label) groups  (target {a.replay_frac:.0%} of {n_new + len(picks)})")
    for k in sorted(groups):
        print(f"    {k:<28} {groups[k]}")
    bk = collections.Counter()
    for j in props:
        for n in j["rungs"]:
            bk[n] += len(P.BACKGROUNDS)
    for _j, _t, n in picks:
        bk[n] += 1
    print(f"buckets (clips): {dict(sorted(bk.items()))}   total {sum(bk.values())}")
    if a.dry_run:
        return

    out = a.out / f"{STAGE}_build"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res_p = list(ex.map(lambda j: build_prop(j, clips_dir, a.reencode), props))
        res_r = list(ex.map(lambda j: build_replay(j, by_job[id(j)], clips_dir, a.reencode),
                            replay_jobs))
    bad = [r for r in res_p + res_r if r["error"]]
    for r in bad:
        print(f"  ✗ dropped {r['job']['src'].name}: {r['error']}")
    records = [r for r in res_p + res_r if not r["error"]]
    lp = check_lengths(records)
    if lp:
        sys.exit("refusing to finish: frame length predicts a label:\n  " + "\n  ".join(lp))

    expected = {o["name"] for r in records for o in r["outputs"]}
    for f in clips_dir.glob("*.mp4"):
        if f.name not in expected:
            f.unlink()
    written = len(list(clips_dir.glob("*.mp4")))
    if written != len(expected):
        sys.exit(f"expected {len(expected)} clips on disk, found {written}")
    final = a.out / f"{STAGE}_{written}"
    if final.exists():
        sys.exit(f"refusing to overwrite existing {final} (delete it or keep the build dir)")
    out.rename(final)
    out, clips_dir = final, final / "clips"

    cfg_local, counts = emit(records, out, str(clips_dir), str(out / "cache"), "")
    cfg_ws, _ = emit(records, out, f"{WS_DATA}/clips", WS_CACHE, ".workspace")

    def src_entry(j):
        e = {"kind": j["kind"], "source": str(j["src"]), "source_md5": j["md5"],
             "character": j["char"], "label": j["label"], "angle": j["angle"],
             "assets": [j["char"].lower()] + ([j["asset"]] if j.get("asset") else [])}
        for k in ("window", "missing", "rungs", "version", "policy", "first_gap",
                  "src_frames", "seam_ratio"):
            if j.get(k) is not None:
                e[k] = j[k]
        return e

    manifest = {
        "generated_by": "prep_v9.py", "stage": STAGE,
        "base": "pudgy-v8-merged = v7fixed + v8@550 (lora_tools_v8.py combine)",
        "registry": R.snapshot(REG),
        "replay_frac": a.replay_frac, "seed": a.seed, "prop_floor": PROP_FLOOR,
        "size": P.SIZE, "fps": P.FPS, "backgrounds": list(P.BACKGROUNDS),
        "zoom_ladder": {str(z): P.ZOOMS[z] for z in sorted(P.ZOOMS)},
        "holdout_angles": sorted(P.HOLDOUT_ANGLES),
        "holdout": [src_entry(j) for j in holdout],
        "counts": {"new": sum(len(r["outputs"]) for r in records if r["role"] == "new"),
                   "replay": sum(len(r["outputs"]) for r in records if r["role"] == "replay"),
                   "buckets": counts},
        "dropped": ([{"source": str(j["src"]), "reason": w, "stage": "preflight"}
                     for j, w in dropped + pool_dropped]
                    + [{"source": str(r["job"]["src"]), "reason": r["error"], "stage": "build"}
                       for r in bad]),
        "report": {k: v for k, v in rep.items() if k != "v8_skipped"}
                  | {"v8_skipped": [list(s) for s in rep["v8_skipped"]]},
        "clips": [src_entry(r["job"]) | {"role": r["role"], "outputs": r["outputs"]}
                  for r in records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nwrote {written} clips -> {clips_dir}")
    print(f"  new {manifest['counts']['new']} · replay {manifest['counts']['replay']} · "
          f"buckets {counts}")
    print(f"  {cfg_local.name} · {cfg_ws.name} · manifest.json")
    print(f"\nbuilt: {out}")


if __name__ == "__main__":
    main()
