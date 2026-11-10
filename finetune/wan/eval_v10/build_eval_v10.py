#!/usr/bin/env python3
"""Build the v10 evaluation set ON THE MAC (CPU only, deterministic, idempotent).

Writes <out> (default $PUDGY_DATA/processed/v10_eval) whose internal paths are all rooted at
--root (default /workspace/eval_v10), so the folder is restored verbatim on the box
(Azure processed/v10_eval/ -> /workspace/eval_v10/). Training_Approach_v10.md section 5; decisions
Q12-Q19.

Inputs: scenes/holdout_v10.json (16 iteration_7 skits + v9b's 6 held-out skits),
scenes/rows_v10.load_rows (the rows that may train; the held-out skits' rows are a subset), the
curation files, the raw skits ($PUDGY_DATA/raw/iteration_{1,2,7}), the v8 keyframes
(<repo>/v8_eval/inputs/keyframes, = Azure v8/eval/keyframes) and the raw phone clips
($PUDGY_DATA/raw/iteration_6/Samples/PROPS/PHONE, for the G5 start frames).

1. scenes/  Tier-1 windows for G3 (high-noise caption use)
   One f33 window per held-out span that has a window of >= 33 frames, starting at that span's
   first such window and with a lead in >= half of it (prep_v9b's rule). Clips are cut exactly
   as prep cuts training windows: 1080x1920 -> 480x854 (Lanczos) -> crop 480x848 at y=3, 24 fps,
   x264 crf 12 (prep_v9b.decode_skit / write_mp4); a PILLARBOX skit (4 px black edge columns,
   flagged in its curation notes and in prep_v10.PILLARBOX) is cut with prep_v10.vf_for (4 px off
   each side, 480x860, crop at y=6). None of the held-out skits is one; the builder checks every
   decoded skit's edge columns for a black bar either way. Candidate captions (all
   compiled by captions_v9b from the same span, differing only in the ACTION), as build_eval_v9b:
     true   the window's own beats       order  the beats reversed (>= 2 beats)
     swap   Pax and Polly exchanged (two-shots)   other  another held-out window's beats, same leads
   Windows with no distractor are left out. index.json is dcls_v8's index format;
   scenes_f33.toml caches the latents on the box (cache dir <root>/scenes/cache/f33).
   v9b's 6 held-out skits contribute their f33 windows too (source "v9b").
2. g1/  G1/G2 jobs: one per held-out span that has a window of >= 49 frames (the span's FIRST such
   window, its own length f49 or f81, its own training caption): first-frame PNG, the real clip of
   the same length, the real last-frame PNG; plus v9b's 8 Tier-2 jobs (build_eval_v9b's rule: up
   to 2 spans per skit, f49 from the span start). ~8 jobs carry "flf": true (report only, Q19).
3. g4/keyframes/  the 4 v8 start frames the core 52 / compose 18 suites use (pax/polly FRONT/QF2_R white).
4. g5/  the 24-clip phone confirm tier (arms_v9c2 --tier confirm, arm E5's jobs: 8 phone starts x
   seeds 42/7/1234, 41 frames), with its start frames rebuilt here exactly as build_assets_v9 does,
   so the box needs no raw data for G5.
5. README.md, manifest.json (counts + input md5s), SHA256SUMS (relative paths; `sha256sum -c` on the box).

    cd <repo>/finetune/wan/eval_v10
    PUDGY_DATA=<Data> <python with numpy + opencv + PIL> build_eval_v10.py
    PUDGY_DATA=<Data> python build_eval_v10.py --out /tmp/x --root /workspace/eval_v10 --force
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v10 as C  # noqa: E402  (sys.path: wan, eval_v8, eval_v9, assets, scenes)

sys.path.insert(0, str(C.WAN / "eval_v9b"))
import prep_v9b as PB  # noqa: E402
import prep_v10 as P10  # noqa: E402  (vf_for, PILLARBOX: the training cut; main session's file, read only)
import captions_v9b as CAP  # noqa: E402
import rows_v10  # noqa: E402
from build_eval_v9b import swap_leads, with_beats  # noqa: E402

T1_FRAMES, G1_MIN, V9B_T2_FRAMES, V9B_T2_PER_SKIT = 33, 49, 49, 2
REPO = C.WAN.parents[1]
V8_KF = REPO / "v8_eval" / "inputs" / "keyframes"
G4_KEYFRAMES = [f"{c}_{v}_white.png" for c in ("pax", "polly") for v in ("FRONT", "QF2_R")]
HOLDOUT_V10 = C.WAN / "scenes" / "holdout_v10.json"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def is_pillarbox(d):
    """Flagged in the curation notes, or listed in prep_v10.PILLARBOX (they must agree)."""
    notes = any("PILLARBOX" in (sp.get("notes") or "") for sp in d["spans"])
    listed = d["skit"] in P10.PILLARBOX
    if notes != listed:
        sys.exit(f"{d['skit']}: PILLARBOX in the curation notes = {notes}, in prep_v10.PILLARBOX = {listed}")
    return listed


def decode(skit, nframes, pillarbox):
    """The training cut: prep_v10.vf_for (1080x1920 -> 480x854 Lanczos -> crop 480x848 at y=3; a
    PILLARBOX skit: crop 4 px off each side -> 480x860 -> crop 480x848 at y=6)."""
    if not pillarbox:
        return PB.decode_skit(PB.RAW / skit, nframes)
    vf = P10.vf_for(skit)
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(PB.RAW / skit), "-vf", vf,
                        *PB.P9._passthrough_flag(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True).stdout
    v = np.frombuffer(b, np.uint8).reshape(-1, PB.SH, PB.SW, 3)
    if len(v) != nframes:
        raise RuntimeError(f"{skit}: decoded {len(v)} frames, curation says {nframes}")
    return v


def edge_bars(v):
    """Mean brightness of the outermost column and of the 5th column, per side, over the clip. A
    pillarbox left in reads ~0 at the edge and ~230 four columns in (PP-Foot: 0 / 228); a dark
    scene is dark in both. -> (left0, left4, right0, right4)."""
    m = v[::4].mean(axis=(0, 1, 3))
    return float(m[0]), float(m[4]), float(m[-1]), float(m[-5])


def looks_pillarboxed(e):
    l0, l4, r0, r4 = e
    return (l0 < 20 and l4 - l0 > 60) or (r0 < 20 and r4 - r0 > 60)


def lead_ids(sp):
    return tuple(sorted(c["id"] for c in sp["characters"] if c["id"] in CAP.LEAD_NAMES))


def name_for(d, si, a, n, prefix):
    return PB.out_name(d, si, a, n).replace("scene_", prefix)


# ---------------------------------------------------------------------------------------------
def collect(held_v10, rows, held_v9b):
    """-> (t1 windows, g1 jobs, skit docs) without touching pixels."""
    t1, g1, docs = [], [], {}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["skit"] in held_v10:
            by[r["skit"]][r["span"]].append(r)
    missing = sorted(set(held_v10) - set(by))
    if missing:
        sys.exit(f"held-out skits with no trainable rows: {missing}")
    for skit in sorted(by):
        d, _ = rows_v10.curation(by[skit][next(iter(by[skit]))][0]["curation"])
        docs[skit] = d
        for si in sorted(by[skit]):
            ws = sorted(by[skit][si], key=lambda w: w["a"])
            sp = d["spans"][si]
            w33 = [w for w in ws if w["frames"] >= T1_FRAMES]
            if w33:
                a = w33[0]["a"]
                if PB.lead_frames(sp, a, a + T1_FRAMES) >= PB.LEAD_MIN_SHARE * T1_FRAMES:
                    t1.append({"source": "v10", "skit": skit, "span": si, "a": a, "b": a + T1_FRAMES, "d": d, "sp": sp,
                               "leads": lead_ids(sp), "texts": [b["text"] for b in CAP.pick_beats(sp["beats"], a, a + T1_FRAMES)],
                               "camera": sp.get("camera", "static"), "name": name_for(d, si, a, T1_FRAMES, "held_")})
            w49 = [w for w in ws if w["frames"] >= G1_MIN]
            if w49:
                w = w49[0]
                cap = CAP.compile_window(d, sp, w["a"], w["b"])
                assert cap == w["caption"], f"{skit} span {si}: recompiled caption differs from captions_v10_windows.jsonl"
                g1.append({"source": "v10", "skit": skit, "span": si, "a": w["a"], "b": w["b"], "frames": w["frames"],
                           "prompt": w["caption"], "camera": w["camera"], "two_shot": w["two_shot"],
                           "tag": name_for(d, si, w["a"], w["frames"], "g1_")[:-4]})
    # v9b's 6 held-out skits (build_eval_v9b's rules, from the current curation files)
    v9b_rows = {r["skit"]: r for r in json.loads(PB.SKITS.read_text())}
    for skit in held_v9b:
        d = json.loads((PB.SCENES / v9b_rows[skit]["out"]).read_text())
        docs[skit] = d
        n2 = 0
        for si, sp in enumerate(d["spans"]):
            if not sp.get("keep") or sp["end"] - sp["start"] < T1_FRAMES:
                continue
            a = sp["start"]
            if PB.lead_frames(sp, a, a + T1_FRAMES) < PB.LEAD_MIN_SHARE * T1_FRAMES:
                continue
            t1.append({"source": "v9b", "skit": skit, "span": si, "a": a, "b": a + T1_FRAMES, "d": d, "sp": sp,
                       "leads": lead_ids(sp), "texts": [b["text"] for b in CAP.pick_beats(sp["beats"], a, a + T1_FRAMES)],
                       "camera": sp.get("camera", "static"), "name": name_for(d, si, a, T1_FRAMES, "held_")})
            if sp["end"] - sp["start"] >= V9B_T2_FRAMES and n2 < V9B_T2_PER_SKIT:
                n2 += 1
                g1.append({"source": "v9b", "skit": skit, "span": si, "a": a, "b": a + V9B_T2_FRAMES,
                           "frames": V9B_T2_FRAMES, "prompt": CAP.compile_window(d, sp, a, a + V9B_T2_FRAMES),
                           "camera": sp.get("camera", "static"), "two_shot": len(lead_ids(sp)) == 2,
                           "tag": name_for(d, si, a, V9B_T2_FRAMES, "g1_")[:-4]})
    return t1, g1, docs


def candidates(wins):
    """build_eval_v9b's distractors; an `other` whose beats name a cast member this skit does
    not have cannot compile and the next window is tried."""
    out = []
    for i, w in enumerate(wins):
        d, sp, a, b = w["d"], w["sp"], w["a"], w["b"]
        cands = {"true": CAP.compile_window(d, sp, a, b)}
        if len(w["texts"]) >= 2:
            cands["order"] = CAP.compile_window(d, with_beats(sp, a, b, w["texts"][::-1]), a, b)
        if len(w["leads"]) == 2:
            cands["swap"] = CAP.compile_window(d, with_beats(sp, a, b, [swap_leads(t) for t in w["texts"]]), a, b)
        others = [o for o in wins if o is not w and o["leads"] == w["leads"] and o["texts"] != w["texts"]]
        others.sort(key=lambda o: (o["skit"] != w["skit"], abs(wins.index(o) - i)))
        for o in others:
            try:
                cands["other"] = CAP.compile_window(d, with_beats(sp, a, b, o["texts"]), a, b)
                break
            except KeyError:
                continue
        cands = {k: v for k, v in cands.items() if k == "true" or v != cands["true"]}
        out.append(cands)
    return out


def pick_flf(jobs, n, seed):
    """~n jobs for the FLF report: iteration_7 only, at most one per skit, 2 camera moves, then
    static shots alternating two-shot / single, at least one naming the phone."""
    rng = random.Random(seed)
    pool = [j for j in jobs if j["source"] == "v10"]
    rng.shuffle(pool)
    pick, used = [], set()

    def take(pred, k):
        for j in pool:
            if len([x for x in pick if pred(x)]) >= k or len(pick) >= n:
                return
            if j["skit"] not in used and pred(j) and j not in pick:
                pick.append(j); used.add(j["skit"])

    take(lambda j: j["camera"] != "static", 2)
    take(lambda j: j["camera"] == "static" and j["phone"], 1)
    want_two = True
    while len(pick) < n:
        c = [j for j in pool if j not in pick and j["skit"] not in used and j["camera"] == "static"]
        if not c:
            c = [j for j in pool if j not in pick]
        if not c:
            break
        j = next((x for x in c if x["two_shot"] == want_two), c[0])
        pick.append(j); used.add(j["skit"]); want_two = not want_two
    return {j["tag"] for j in pick}


def build_g5(out, root):
    """build_assets_v9's phone start frames + arms_v9c2's confirm-tier E5 jobs (24)."""
    import common_v9 as C9
    import arms_v9c2 as AR
    P, P9 = C9.P, C9.P9
    white = P.BACKGROUNDS["white"][1]
    kfdir = out / "g5" / "keyframes"; kfdir.mkdir(parents=True, exist_ok=True)
    kf = {}
    n = P9.PROP_FLOOR
    for j in sorted(P9.discover_props({"props_skipped": []}), key=lambda x: (x["char"], x["label"], x["angle"])):
        if j["angle"] not in ("QF2_R", "FRONT"):
            continue
        ok, why = P9.preflight_prop(j)
        if not ok or j["window"][1] < n:
            print(f"  ~ g5 skip {j['src'].name}: {why or 'window too short'}"); continue
        dest = kfdir / f"phone_{j['char'].lower()}_{j['label']}_{j['angle']}.png"
        if not dest.exists():
            fr = P9.decode_window(dict(j, rungs=[n]), 1.00)
            Image.fromarray(P.composite(fr[:1], white)[0]).save(dest)
        kf[dest.name] = {"char": j["char"], "label": j["label"], "view": j["angle"],
                         "source": str(Path(j["src"]).relative_to(P.DATA)), "window_start": j["window"][0],
                         "split": "holdout" if j["angle"] == "QF2_R" else "train"}
    (kfdir / "keyframes.json").write_text(json.dumps(kf, indent=1))
    C9.KF = kfdir                                   # suite_v9.phone_jobs reads common_v9.KF
    fn = AR.arms("E5")["E5"][2]                     # new_wording (asserts the 2026-10-27 registry wording)
    jobs = [fn(j) for j in AR.jobs_for("confirm", 41)]
    for j in jobs:
        j["start"] = str(root / "g5" / "keyframes" / Path(j["start"]).name)
    (out / "g5" / "jobs.json").write_text(json.dumps(jobs, indent=1))
    return len(kf), len(jobs)


README = """# v10 evaluation set (`processed/v10_eval`, restored as `/workspace/eval_v10`)

Built on the Mac by `finetune/wan/eval_v10/build_eval_v10.py` ({built}); every path inside is rooted
at `{root}`, so restore the folder **verbatim** to that path (Azure `processed/v10_eval/`). Training_Approach_v10.md
section 5; the scripts that read it are in `finetune/wan/eval_v10/`.

| part | what | count |
|---|---|---|
| `scenes/` | Tier-1 (G3) f33 windows of held-out spans, cut exactly as prep cuts training windows (480x848, 24 fps); `index.json` (dcls format, true + distractor captions), `scenes_f33.jsonl` + `scenes_f33.toml` (latent cache -> `scenes/cache/f33`, made on the box) | {n_t1} windows ({n_t1_v10} iteration_7 + {n_t1_v9b} v9b), {n_d} distractors ({n_order} order, {n_swap} swap, {n_other} other); {n_t1_phone} windows name the phone |
| `g1/` | G1/G2 jobs: `jobs.json`; `first/<tag>.png` (start frame), `real/<tag>.mp4` (the real shot, same length), `last/<tag>.png` (its last frame, for FLF) | {n_g1} jobs ({n_g1_v10} iteration_7 + {n_g1_v9b} v9b; f49 {n_f49}, f81 {n_f81}; {n_static} static camera; {n_g1_phone} name the phone) x seeds 42/43/44 = {n_pairs} G1 pairs; {n_flf} marked `flf` (report only) |
| `g4/keyframes/` | the v8 start frames of the core 52 / compose 18 suites (`v8_eval/inputs/keyframes` = Azure `v8/eval/keyframes`) | {n_g4} |
| `g5/` | phone confirm tier (arm E5's 24 jobs: 8 starts x seeds 42/7/1234, 41 f, 1024^2); start frames rebuilt here as build_assets_v9 does | {n_g5kf} start frames, {n_g5} jobs |

Held out: {n_skits_v10} iteration_7 skits (`scenes/holdout_v10.json`) + v9b's {n_skits_v9b} held-out skits. None of them is
PILLARBOX-flagged ({pillar}). Captions: the v10 jobs use their row's caption from
`processed/v10_scenes_748/captions_v10_windows.jsonl` (recompiled and checked equal); v9b's are compiled by
`captions_v9b` from the current curation files.

## Rebuild (Mac)
```
cd <repo>/finetune/wan/eval_v10
PUDGY_DATA=<Data> <python with numpy, opencv, PIL> build_eval_v10.py [--out <dir>] [--root {root}] [--force]
```
Deterministic: same inputs, same files (`manifest.json` records the input md5s; `SHA256SUMS` every file).

## On the box
```
cd /workspace/eval_v10 && sha256sum -c --quiet SHA256SUMS      # no output = intact
# latents for the Tier-1 windows (once; 51 clips, a few minutes, est.):
cd /workspace/musubi-tuner && /workspace/Pudgy/.venv-wan/bin/python src/musubi_tuner/wan_cache_latents.py \\
    --dataset_config /workspace/eval_v10/scenes/scenes_f33.toml \\
    --vae /workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors --i2v --skip_existing --num_workers 16
```
Outputs of every eval go to `/workspace/eval_v10/runs/<RUN_NAME>/` (never into the files listed in SHA256SUMS).
The step order and every command are in the docstrings of `finetune/wan/eval_v10/` (common_v10.py prefetch,
tier1_v10.py, phase0_v10.py, suite_v10.py, g4_v10.py, g5_v10.py, g1_pairs_v10.py). Also needed on the box:
`/workspace/eval_v9/holdout` with its latent cache (Azure `v9b/handover/eval_for_v9b.tar`) for G4-fast, and
`rapidocr_onnxruntime` in the venv for G2's text check.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="default $PUDGY_DATA/processed/v10_eval")
    ap.add_argument("--root", default="/workspace/eval_v10", help="path the set is restored to on the box")
    ap.add_argument("--flf", type=int, default=8)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="re-cut clips and PNGs that already exist")
    ap.add_argument("--skip-g5", action="store_true")
    a = ap.parse_args()
    if "PUDGY_DATA" not in os.environ:
        sys.exit("set PUDGY_DATA to the Data root")
    data = Path(os.environ["PUDGY_DATA"])
    out = Path(a.out) if a.out else data / "processed" / "v10_eval"
    root = Path(a.root)
    t0 = time.time()

    hold = json.loads(HOLDOUT_V10.read_text())
    held_v10, held_v9b = hold["skits"], hold["plus_v9b_holdout"]
    assert held_v9b == json.loads(PB.HOLDOUT_FILE.read_text())["skits"], "holdout_v10's v9b list != holdout_v9b.json"
    rows, _dropped = rows_v10.load_rows(data)
    t1, g1, docs = collect(set(held_v10), rows, held_v9b)

    # pixels: one decode per skit, only if something is missing
    dirs = {k: out / k for k in ("scenes/clips", "g1/first", "g1/last", "g1/real")}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    pillar, edges = {}, {}
    for skit in sorted(docs):
        d = docs[skit]
        pillar[skit] = is_pillarbox(d)
        need_t1 = [w for w in t1 if w["skit"] == skit]
        need_g1 = [j for j in g1 if j["skit"] == skit]
        todo = a.force or any(not (dirs["scenes/clips"] / w["name"]).exists() for w in need_t1) or any(
            not (dirs[k] / f"{j['tag']}{ext}").exists() for j in need_g1
            for k, ext in (("g1/first", ".png"), ("g1/last", ".png"), ("g1/real", ".mp4")))
        if not todo:
            continue
        v = decode(skit, d["frames"], pillar[skit])
        edges[skit] = edge_bars(v)
        if looks_pillarboxed(edges[skit]):
            sys.exit(f"{skit}: black edge columns after the crop {edges[skit]} - "
                     f"{'PILLARBOX crop failed' if pillar[skit] else 'not PILLARBOX-flagged in its curation'}")
        for w in need_t1:
            PB.write_mp4(np.ascontiguousarray(v[w["a"]:w["b"]]), dirs["scenes/clips"] / w["name"])
        for j in need_g1:
            PB.write_mp4(np.ascontiguousarray(v[j["a"]:j["b"]]), dirs["g1/real"] / f"{j['tag']}.mp4")
            Image.fromarray(v[j["a"]]).save(dirs["g1/first"] / f"{j['tag']}.png")
            Image.fromarray(v[j["b"] - 1]).save(dirs["g1/last"] / f"{j['tag']}.png")
        print(f"  {skit.split('/')[-1]:<44} {len(need_t1)} T1, {len(need_g1)} G1  edge columns "
              f"{edges[skit][0]:.0f}|{edges[skit][1]:.0f} .. {edges[skit][3]:.0f}|{edges[skit][2]:.0f}"
              f"{'  PILLARBOX' if pillar[skit] else ''}", flush=True)

    # Tier-1 index
    cands = candidates(t1)
    index, dropped_nod = [], 0
    for w, cd in zip(t1, cands):
        if len(cd) < 2:
            dropped_nod += 1; continue
        phone = sorted(k for k, v in cd.items() if C.E5_PHRASE in v)
        for k, v in cd.items():
            assert v.count(C.E5_PHRASE) <= 1, f"{w['name']}/{k}: phone phrase twice"
            assert C.E5_PHRASE.lower() not in v.lower() or C.E5_PHRASE in v, \
                f"{w['name']}/{k}: phone phrase capitalised - E5 would not be spliced"
        index.append({"clip": w["name"], "char": "+".join(w["leads"]), "label": "true", "kind": "scene",
                      "split": "holdout", "frames": T1_FRAMES, "source": w["source"], "skit": w["skit"],
                      "span": w["span"], "window": [w["a"], w["b"]], "camera": w["camera"],
                      "phone": phone, "candidates": cd})
    sc = out / "scenes"
    (sc / "index.json").write_text(json.dumps(index, indent=1))
    jl = sc / f"scenes_f{T1_FRAMES}.jsonl"
    with open(jl, "w") as f:
        for e in index:
            f.write(json.dumps({"video_path": str(root / "scenes" / "clips" / e["clip"]),
                                "caption": e["candidates"]["true"]}, ensure_ascii=False) + "\n")
    (sc / f"scenes_f{T1_FRAMES}.toml").write_text("\n".join([
        "# v10 held-out scene windows (eval_v10/build_eval_v10.py): latents for the high-noise diffusion classifier (G3).",
        "[general]", f"resolution = [{PB.SW}, {PB.SH}]", "batch_size = 1", "enable_bucket = true",
        "bucket_no_upscale = true", "", "[[datasets]]",
        f'video_jsonl_file = "{root / "scenes" / jl.name}"',
        f'cache_directory = "{root / "scenes" / "cache" / f"f{T1_FRAMES}"}"', f"target_frames = [{T1_FRAMES}]",
        'frame_extraction = "head"', "num_repeats = 1", ""]))
    for p in sc.glob("clips/*.mp4"):                  # stale clips from an older holdout
        if p.name not in {e["clip"] for e in index}:
            p.unlink()

    # G1/G2 jobs
    for j in g1:
        j["phone"] = C.E5_PHRASE in j["prompt"]
    flf = pick_flf(g1, a.flf, a.seed)
    jobs = [{"tag": j["tag"], "source": j["source"], "skit": j["skit"], "span": j["span"], "window": [j["a"], j["b"]],
             "frames": j["frames"], "prompt": j["prompt"], "camera": j["camera"], "two_shot": j["two_shot"],
             "phone": j["phone"], "flf": j["tag"] in flf,
             "start": str(root / "g1" / "first" / f"{j['tag']}.png"),
             "real": str(root / "g1" / "real" / f"{j['tag']}.mp4"),
             "end": str(root / "g1" / "last" / f"{j['tag']}.png")} for j in g1]
    assert len({j["tag"] for j in jobs}) == len(jobs), "duplicate G1 tags"
    (out / "g1" / "jobs.json").write_text(json.dumps(jobs, indent=1, ensure_ascii=False))

    # G4 keyframes
    g4 = out / "g4" / "keyframes"; g4.mkdir(parents=True, exist_ok=True)
    for n in G4_KEYFRAMES:
        src = V8_KF / n
        if not src.exists():
            sys.exit(f"missing v8 keyframe {src} (Azure v8/eval/keyframes/)")
        assert Image.open(src).size == (1024, 1024), src
        if a.force or not (g4 / n).exists():
            shutil.copyfile(src, g4 / n)

    n_g5kf, n_g5 = (0, 0) if a.skip_g5 else build_g5(out, root)

    # manifest, README, checksums
    counts = {
        "n_t1": len(index), "n_t1_v10": sum(e["source"] == "v10" for e in index),
        "n_t1_v9b": sum(e["source"] == "v9b" for e in index), "t1_dropped_no_distractor": dropped_nod,
        "n_d": sum(len(e["candidates"]) - 1 for e in index),
        "n_order": sum("order" in e["candidates"] for e in index), "n_swap": sum("swap" in e["candidates"] for e in index),
        "n_other": sum("other" in e["candidates"] for e in index), "n_t1_phone": sum(bool(e["phone"]) for e in index),
        "n_g1": len(jobs), "n_g1_v10": sum(j["source"] == "v10" for j in jobs), "n_g1_v9b": sum(j["source"] == "v9b" for j in jobs),
        "n_f49": sum(j["frames"] == 49 for j in jobs), "n_f81": sum(j["frames"] == 81 for j in jobs),
        "n_static": sum(j["camera"] == "static" for j in jobs), "n_g1_phone": sum(j["phone"] for j in jobs),
        "n_pairs": 3 * len(jobs), "n_flf": len(flf), "n_g4": len(G4_KEYFRAMES), "n_g5kf": n_g5kf, "n_g5": n_g5,
        "n_skits_v10": len(held_v10), "n_skits_v9b": len(held_v9b),
    }
    pillar_held = sorted(s for s, p in pillar.items() if p)
    inputs = {"holdout_v10.json": md5(HOLDOUT_V10), "holdout_v9b.json": md5(PB.HOLDOUT_FILE),
              "captions_v10_windows.jsonl": md5(data / rows_v10.CAPTIONS),
              "registry.toml": md5(C.WAN / "assets" / "registry.toml"),
              "curation": {s: md5(PB.SCENES / (r["out"])) for s, r in
                           {**{x["skit"]: x for x in json.loads(PB.SKITS.read_text())},
                            **{x["skit"]: x for x in json.loads((C.WAN / "scenes" / "skits_v10.json").read_text())["skits"]}}.items()
                           if s in docs}}
    manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root": str(root),
                "command": "PUDGY_DATA=<Data> python finetune/wan/eval_v10/build_eval_v10.py", "counts": counts,
                "pillarbox_held": pillar_held, "edge_brightness": {k: [round(x, 1) for x in v] for k, v in edges.items()},
                "flf_jobs": sorted(flf), "inputs": inputs, "seconds": round(time.time() - t0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text(README.format(built=manifest["built"], root=root, pillar=", ".join(pillar_held) or "checked", **counts))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                   and "/runs/" not in str(p) and "/cache/" not in str(p))
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps(counts, indent=1))
    print(f"-> {out}  ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
