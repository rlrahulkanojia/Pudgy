#!/usr/bin/env python3
"""
v9b-1 training-set prep: curated skit spans + 35% stratified v8 replay.

Plan: training_approach/v9/Training_Approach_v9.md section 0.1 (decisions D1-D10), 3.2-3.4
and 4; runbook: training_approach/v9/RUNBOOK_v9b.md. Inputs are the curator pass
(finetune/wan/scenes/curation/*.json, schema scenes/CURATION_v9b.md) and the asset
registry. Nothing here re-judges footage: every keep/drop, boundary and beat is the
curator's, checked by scenes/curate_check.py, which must pass before anything is built.

  * WINDOWS. Each kept span (locked camera, >= 21 frames, a lead visible) is cut into
    non-overlapping windows of 81 / 49 / 33 / 21 frames from its start, largest first
    (D6), plus one f21 window ending at the span's end when >= 11 frames would be left over.
    A window in which no lead is on screen for at least half its frames is skipped.
  * CAPTIONS are compiled by scenes/captions_v9b.py from the curator's beats (only the
    beats a window shows), the registry anchors, the setting and the camera clause (D9),
    then linted: no lead colour word outside the anchors, no v8 action clause (D10), no
    unresolved {ref}.
  * SIZE. 1080x1920 -> 480x854 (Lanczos) -> centre crop 480x848, 24 fps (D2). The one
    23.976 fps skit is written at 24 fps (0.1% faster, frames unchanged).
  * HOLDOUT. Whole skits listed in scenes/holdout_v9b.json are never trained.
  * REPLAY (D10). 35% of the stage is v8 clips, built exactly as prep_v8/prep_v9 build them
    (1024^2, flat grounds, v8 captions). Each (character, label) group gets at least one
    clip, and a label whose motion or expression word appears in N scene captions gets at
    least ceil(N / 4) replay clips, so a scene word cannot redefine the v8 label. The QF2_R
    angle stays held out (the regression eval set).

    PUDGY_DATA=/workspace/Data python finetune/wan/prep_v9b.py --dry-run
    PUDGY_DATA=/workspace/Data python finetune/wan/prep_v9b.py --workers 16
"""
import argparse
import collections
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "assets", HERE / "scenes"):
    sys.path.insert(0, str(p))
import prep_v8 as P  # noqa: E402
import prep_v9 as P9  # noqa: E402
import registry as R  # noqa: E402
import captions_v9b as CAP  # noqa: E402
import curate_check as CK  # noqa: E402

DATA = P.DATA
RAW = DATA / "raw"
OUT_ROOT = DATA / "processed"
STAGE = "v9b"
SCENES = HERE / "scenes"
CURATION = SCENES / "curation"
SKITS = SCENES / "skits_v9b.json"
HOLDOUT_FILE = SCENES / "holdout_v9b.json"
WS_DATA = "/workspace/data_v9b"
WS_CACHE = "/workspace/wan_cache/latents_v9b"

SW, SH = 480, 848              # D2: 9:16, both multiples of 16
SCALE_H = 854                  # 1920 * 480 / 1080 = 853.3 -> 854, then crop 3 px top and bottom
BUCKETS = (81, 49, 33, 21)     # D6
LEAD_MIN_SHARE = 0.5

# Words in a scene caption that touch a v8 label (D10). Expression labels count too: a
# scene "smiles" 400 times would otherwise redefine what the v8 happy caption predicts.
LABEL_WORDS = {
    "walking": r"\b(walk\w*|waddl\w*|strid\w*|step\w*)\b",
    "running": r"\b(run|runs|running|ran|sprint\w*|dash\w*)\b",
    "waving": r"\bwav(e|es|ing|ed)\b",
    "sitting": r"\b(sit|sits|sitting|sat|seated)\b",
    "jumping": r"\b(jump\w*|hop|hops|hopping|hopped|leap\w*|bounc\w*)\b",
    "happy": r"\b(smil\w*|grin\w*|happ\w*|beam\w*)\b",
    "surprised": r"\b(surpris\w*|gasp\w*|startl\w*|shock\w*)\b",
    "angry": r"\b(angr\w*|scowl\w*|glar\w*|frown\w*|furious)\b",
    "confused": r"\b(confus\w*|puzzl\w*|baffl\w*)\b",
    "crying": r"\b(cry|cries|crying|cried|sob\w*|tears?)\b",
    "laughing": r"\b(laugh\w*|giggl\w*|chuckl\w*)\b",
    "blushing": r"\b(blush\w*)\b",
}
ANCHOR_RATIO = 4               # >= 1 replay clip per 4 scene windows using the label's words


# ---------------------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------------------
def md5_file(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def load_curation(allow_missing):
    skits = json.loads(SKITS.read_text())
    docs, missing, bad = [], [], {}
    for row in skits:
        f = SCENES / row["out"]
        if not f.exists():
            missing.append(row["skit"]); continue
        errs, _w = CK.check(f)
        if errs:
            bad[f.name] = errs
        d = json.loads(f.read_text())
        d["_file"] = str(f.relative_to(HERE)); d["_md5"] = md5_file(f)
        docs.append(d)
    if bad:
        sys.exit("curation has errors - run scenes/curate_check.py:\n  " +
                 "\n  ".join(f"{k}: {v[0]}" for k, v in bad.items()))
    if missing and not allow_missing:
        sys.exit(f"{len(missing)} skits have no curation file yet (e.g. {missing[:3]}); "
                 "use --allow-missing only for a dry run")
    return docs, missing


def lead_frames(sp, a, b):
    """How many frames of [a, b) show at least one lead (from presence / enters / exits)."""
    on = np.zeros(b - a, bool)
    for c in sp["characters"]:
        if c["id"] not in CAP.LEAD_NAMES:
            continue
        s = c.get("enters_at", sp["start"]) if c.get("presence") in ("enters", "enters_exits") else sp["start"]
        e = c.get("exits_at", sp["end"]) if c.get("presence") in ("exits", "enters_exits") else sp["end"]
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            on[lo - a:hi - a] = True
    return int(on.sum())


END_REMAINDER = 11      # the end window then shares at most 10 of its 21 frames with the previous one


def windows(sp):
    """Non-overlapping windows from the span start, largest bucket first; then, if >= 11
    frames would be left over at the end, one f21 window ending exactly at the span end, so
    a span's last beat (often its payoff) is not lost to the remainder. (At >= 11 it shares
    at most 10 of its frames with the window before it.)"""
    out, a = [], sp["start"]
    while sp["end"] - a >= BUCKETS[-1]:
        n = next(x for x in BUCKETS if x <= sp["end"] - a)
        out.append((a, a + n))
        a += n
    if out and sp["end"] - a >= END_REMAINDER:
        out.append((sp["end"] - BUCKETS[-1], sp["end"]))
    return out


COLOUR_RE = re.compile(r"\b(blue|pink)\b", re.I)


def lint_caption(cap):
    probs = []
    body = cap
    for cid in CAP.LEAD_NAMES:
        body = body.replace(R.phrase(CAP.REG, cid), "")
    if COLOUR_RE.search(body):
        probs.append(f"lead colour word outside the anchors: {COLOUR_RE.search(body).group(0)!r}")
    hit = CK.ngrams(cap) & CK.V8_GRAMS
    if hit:
        probs.append(f"v8 action clause (D10): {sorted(hit)[0]!r}")
    if "{" in cap or "}" in cap:
        probs.append("unresolved {ref}")
    if len(cap.split()) > 300:
        probs.append(f"{len(cap.split())} words (T5 limit 512 tokens)")
    return probs


# ---------------------------------------------------------------------------------------
# Decode / encode
# ---------------------------------------------------------------------------------------
def decode_skit(path, expect):
    vf = f"scale={SW}:{SCALE_H}:flags=lanczos,crop={SW}:{SH}:0:{(SCALE_H - SH) // 2}"
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path), "-vf", vf,
                        *P9._passthrough_flag(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True).stdout
    v = np.frombuffer(b, np.uint8).reshape(-1, SH, SW, 3)
    if len(v) != expect:
        raise RuntimeError(f"{path.name}: decoded {len(v)} frames, curation says {expect}")
    return v


def write_mp4(frames, dest):
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{SW}x{SH}", "-r", str(P.FPS), "-i", "-", "-threads", "4",
                          "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                          "-pix_fmt", "yuv420p", "-an", str(dest)], stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed on {dest}")


def out_name(d, si, a, n):
    stem = re.sub(r"[^A-Za-z0-9]+", "", Path(d["skit"]).stem.replace("Base", ""))
    it = d["skit"].split("/")[0].replace("iteration_", "it")
    return f"scene_{it}_{stem}_s{si:02d}_{a:04d}_f{n}.mp4"


def build_skit(d, items, clips_dir, reencode):
    todo = [w for w in items if reencode or not (clips_dir / w["name"]).exists()]
    if todo:
        v = decode_skit(RAW / d["skit"], d["frames"])
        for w in todo:
            write_mp4(np.ascontiguousarray(v[w["a"]:w["b"]]), clips_dir / w["name"])
    return len(todo)


# ---------------------------------------------------------------------------------------
# Replay with label anchors
# ---------------------------------------------------------------------------------------
def anchor_minima(captions):
    counts = {lab: sum(bool(re.search(rx, c, re.I)) for c in captions) for lab, rx in LABEL_WORDS.items()}
    return counts, {lab: math.ceil(n / ANCHOR_RATIO) for lab, n in counts.items() if n}


def select_replay(pool, total, minima, seed):
    """`total` replay outputs stratified over (character, label), at least one per group at
    its kind's common bucket (prep_v9's rule), and each label's anchor minimum spread over
    both characters. Returns picks and the effective total."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for j in sorted(pool, key=lambda x: (x["char"], x["label"], x["angle"])):
        for tag, n in P9.replay_candidates(j):
            groups[(j["char"], j["label"])].append((j, tag, n))
    keys = sorted(groups)
    need = {k: 1 for k in keys}
    for lab, m in minima.items():
        ks = [k for k in keys if k[1] == lab]
        for i, k in enumerate(ks):
            need[k] = max(need[k], m // len(ks) + (1 if i < m % len(ks) else 0))
    total = max(total, sum(need.values()))
    quota = dict(need)
    spare = total - sum(quota.values())
    while spare > 0:                          # the rest: evenly, then at random
        for k in rng.sample(keys, min(spare, len(keys))):
            quota[k] += 1; spare -= 1
    picked = []
    for k in keys:
        cands = groups[k][:]
        rng.shuffle(cands)
        common = P.COMMON[cands[0][0]["kind"]]
        first = next((c for c in cands if c[2] == common), None)
        take = ([first] if first else []) + [c for c in cands if c is not first]
        if quota[k] > len(take):
            print(f"  ! {k}: wants {quota[k]} replay clips, only {len(take)} distinct exist - using all")
        picked += take[:quota[k]]
    return picked, total


# ---------------------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------------------
def emit(scene_rows, replay_rows, out, prefix, cache_root, suffix):
    lines = [f"# Musubi dataset config - {STAGE} (auto-generated by prep_v9b.py).",
             f"# Scene windows at {SW}x{SH} (9:16) + v8 replay at {P.SIZE}^2; one block per",
             "# (source, frame length). Plan: Training_Approach_v9.md section 0.1 / 4.", "",
             "[general]", f"resolution = [{SW}, {SH}]", "batch_size = 1",
             "enable_bucket = true", "bucket_no_upscale = true", ""]
    counts = {}
    for kind, rows, reso in (("scene", scene_rows, None), ("replay", replay_rows, (P.SIZE, P.SIZE))):
        by_n = collections.defaultdict(list)
        for r in rows:
            by_n[r["frames"]].append({"video_path": f"{prefix.rstrip('/')}/{r['name']}", "caption": r["caption"]})
        for n in sorted(by_n):
            jp = out / f"dataset_{STAGE}_{kind}_f{n}{suffix}.jsonl"
            with open(jp, "w") as f:
                for row in by_n[n]:
                    f.write(json.dumps(row) + "\n")
            ref = jp if not suffix else f"{prefix.rsplit('/', 1)[0]}/{jp.name}"
            lines += ["[[datasets]]", f'video_jsonl_file = "{ref}"',
                      f'cache_directory = "{cache_root}/{kind}_f{n}"', f"target_frames = [{n}]",
                      'frame_extraction = "head"', "num_repeats = 1"]
            if reso:
                lines.append(f"resolution = [{reso[0]}, {reso[1]}]")
            lines += [f"# {len(by_n[n])} {kind} clips", ""]
            counts[f"{kind}_f{n}"] = len(by_n[n])
    cfg = out / f"dataset_config_{STAGE}{suffix}.toml"
    cfg.write_text("\n".join(lines))
    return cfg, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="dry runs only: skip uncurated skits")
    ap.add_argument("--replay-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--reencode", action="store_true")
    a = ap.parse_args()
    if a.allow_missing and not a.dry_run:
        sys.exit("--allow-missing is for --dry-run only")

    probs = R.lint(R.load(), DATA)
    if probs:
        sys.exit("registry lint failed:\n  " + "\n  ".join(probs))
    for cid, nm in CAP.LEAD_NAMES.items():
        if R.phrase(CAP.REG, cid) != P.ANCHORS[nm]:
            sys.exit(f"registry anchor for {nm} differs from prep_v8 - the base was trained on prep_v8's")

    docs, missing = load_curation(a.allow_missing)
    held = set(json.loads(HOLDOUT_FILE.read_text())["skits"]) if HOLDOUT_FILE.exists() else set()
    if not held and not a.dry_run:
        sys.exit(f"no holdout chosen: write {HOLDOUT_FILE} first (Training_Approach_v9 section 3.2 #6)")
    unknown = held - {d["skit"] for d in docs} - set(missing)
    if unknown:
        sys.exit(f"holdout lists unknown skits: {sorted(unknown)}")

    # ---- scene windows ----
    scene, skipped, cap_probs = [], collections.Counter(), []
    per_skit = collections.defaultdict(list)
    for d in docs:
        if d["skit"] in held:
            continue
        for si, sp in enumerate(d["spans"]):
            if not sp.get("keep"):
                continue
            for wa, wb in windows(sp):
                n = wb - wa
                if lead_frames(sp, wa, wb) < LEAD_MIN_SHARE * n:
                    skipped["no lead for half the window"] += 1; continue
                cap = CAP.compile_window(d, sp, wa, wb)
                pr = lint_caption(cap)
                if pr:
                    cap_probs.append(f"{d['skit']} [{wa},{wb}): {pr[0]}")
                w = {"name": out_name(d, si, wa, n), "skit": d["skit"], "span": si, "a": wa, "b": wb,
                     "frames": n, "caption": cap, "curation": d["_file"], "curation_md5": d["_md5"],
                     "two_shot": sum(c["id"] in CAP.LEAD_NAMES for c in sp["characters"]) == 2}
                scene.append(w); per_skit[d["skit"]].append(w)
    if cap_probs:
        sys.exit("caption lint failed:\n  " + "\n  ".join(cap_probs[:20]))
    names = [w["name"] for w in scene]
    if len(names) != len(set(names)):
        sys.exit("duplicate output names")
    bk = collections.Counter(w["frames"] for w in scene)
    print(f"curation: {len(docs)} skits ({len(missing)} not curated yet), holdout {len(held)} skits")
    print(f"scene windows: {len(scene)} from {len(per_skit)} skits  buckets {dict(sorted(bk.items()))}  "
          f"({sum(w['frames'] for w in scene) / P.FPS / 60:.1f} min)  two-shots {sum(w['two_shot'] for w in scene)}")
    for k, v in skipped.items():
        print(f"  skipped: {v} windows ({k})")

    # ---- replay ----
    counts, minima = anchor_minima([w["caption"] for w in scene])
    print("D10 label words in scene captions -> replay minimum:")
    for lab in LABEL_WORDS:
        print(f"    {lab:<10} {counts[lab]:>4} windows -> >= {minima.get(lab, 0)}")
    pool, _props, _rep = P9.discover_v9()
    pool = [j for j in pool if j["angle"].split("_t")[0] not in P.HOLDOUT_ANGLES]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        pv = list(ex.map(P.preflight, pool))
    P.derive_frame_budgets(pool, [v[2] for v in pv])
    pool_dropped = [(j, why) for j, (ok, why, _n) in zip(pool, pv) if not ok]
    pool = [j for j, (ok, _w, _n) in zip(pool, pv) if ok]
    for j, why in pool_dropped:
        print(f"  ✗ replay source {j['src'].name}: {why}")
    P.assign_zooms(pool)
    target = round(len(scene) * a.replay_frac / (1 - a.replay_frac))
    picks, total = select_replay(pool, target, minima, a.seed)
    print(f"replay: {len(picks)} clips (target {target} = {a.replay_frac:.0%}; anchors raise it to {total}) "
          f"-> {len(picks) / (len(picks) + len(scene)):.0%} of the stage")
    grp = collections.Counter(f"{j['char']}/{j['label']}" for j, _t, _n in picks)
    print("    " + "  ".join(f"{k} {v}" for k, v in sorted(grp.items())))
    if a.dry_run:
        return

    out = a.out / f"{STAGE}_build"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    by_doc = {d["skit"]: d for d in docs}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        n_written = sum(ex.map(lambda kv: build_skit(by_doc[kv[0]], kv[1], clips_dir, a.reencode),
                               per_skit.items()))
    print(f"scene clips encoded: {n_written} (of {len(scene)})")
    by_job = collections.defaultdict(list)
    for j, tag, n in picks:
        by_job[id(j)].append((tag, n))
    replay_jobs = [j for j in pool if id(j) in by_job]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res_r = list(ex.map(lambda j: P9.build_replay(j, by_job[id(j)], clips_dir, a.reencode), replay_jobs))
    bad = [r for r in res_r if r["error"]]
    for r in bad:
        print(f"  ✗ replay dropped {r['job']['src'].name}: {r['error']}")
    replay = [{"name": o["name"], "frames": o["frames"], "caption": o["caption"],
               "source": str(r["job"]["src"]), "char": r["job"]["char"], "label": r["job"]["label"],
               "angle": r["job"]["angle"], "background": o["background"], "zoom": o["zoom"]}
              for r in res_r if not r["error"] for o in r["outputs"]]

    expected = {w["name"] for w in scene} | {r["name"] for r in replay}
    for f in clips_dir.glob("*.mp4"):
        if f.name not in expected:
            f.unlink()
    written = len(list(clips_dir.glob("*.mp4")))
    if written != len(expected):
        sys.exit(f"expected {len(expected)} clips on disk, found {written}")
    final = a.out / f"{STAGE}_{written}"
    if final.exists():
        sys.exit(f"refusing to overwrite existing {final}")
    out.rename(final)
    out, clips_dir = final, final / "clips"
    _cfg, cnt = emit(scene, replay, out, str(clips_dir), str(out / "cache"), "")
    cfg_ws, _ = emit(scene, replay, out, f"{WS_DATA}/clips", WS_CACHE, ".workspace")
    manifest = {
        "generated_by": "prep_v9b.py", "stage": STAGE,
        "base": "pudgy-v8-merged = v7fixed + v8@550 (D1)",
        "registry": R.snapshot(CAP.REG), "size": [SW, SH], "fps": P.FPS, "buckets": list(BUCKETS),
        "replay_frac": a.replay_frac, "seed": a.seed, "anchor_ratio": ANCHOR_RATIO,
        "label_word_counts": counts, "anchor_minima": minima,
        "holdout_skits": sorted(held), "holdout_angles": sorted(P.HOLDOUT_ANGLES),
        "uncurated": missing, "counts": cnt,
        "scene": scene, "replay": replay,
        "replay_dropped": [{"source": str(j["src"]), "reason": w} for j, w in pool_dropped]
                          + [{"source": str(r["job"]["src"]), "reason": r["error"]} for r in bad],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nbuilt {out}: {len(scene)} scene + {len(replay)} replay clips  {cnt}")
    print(f"  {cfg_ws.name} · manifest.json")


if __name__ == "__main__":
    main()
