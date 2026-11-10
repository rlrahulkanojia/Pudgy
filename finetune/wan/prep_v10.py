#!/usr/bin/env python3
"""
v10 Stage 1 training-set prep (Training_Approach_v10.md section 0, 3.4-3.6). Runs on the Mac
(Q11): CPU and ffmpeg only. The box downloads the result and starts at caching.

Three parts, one musubi dataset config:

  * v10 SCENE WINDOWS: the rows of processed/v10_scenes_748/captions_v10_windows.jsonl that
    may train (scenes/rows_v10.py: trainable, minus Q4 CuteFit and Q5 unnamed side
    characters), minus the skits in scenes/holdout_v10.json. Windows and captions are
    compile_captions_v10.py's; every caption is re-compiled from its curation span and must
    match, then linted as prep_v9b lints. Cut from raw iteration_7: 1080x1920 -> 480x854
    (Lanczos) -> centre crop 480x848, 24 fps. Six skits have 4 px black columns at both edges
    (measured 2026-11-10 on every training skit): those are cropped to 1072 wide first, then
    scaled to 480x860 and centre-cropped to 480x848.
  * v9b SCENE WINDOWS (U7): the 348 windows v9b-1 trained on, frozen in
    scenes/v9b_535_scene_windows.json, re-cut from raw iteration_1/2 with the same frame
    ranges, crop and captions.
  * v8 REPLAY at 35% of the stage, built by prep_v9's builders exactly as v9b's (1024^2, flat
    grounds, v8 captions, QF2_R held out), with prep_v9b's label anchors (D10) computed over
    all scene captions.

Output: Data/processed/v10_<N>/ with clips/, dataset_config_v10[.workspace].toml,
dataset_v10_{scene,replay}_f<n>[.workspace].jsonl (one block per kind and length; cache
directories {scene,replay}_f<n>, which the preservation marker relies on), manifest.json and
README.md. The .workspace files point at /workspace/data_v10 and
/workspace/wan_cache/latents_v10.

    PUDGY_DATA=<Data> PY=<python with numpy> $PY finetune/wan/prep_v10.py --dry-run
    PUDGY_DATA=<Data> $PY finetune/wan/prep_v10.py --workers 8
"""
import argparse
import collections
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_v9b as PB  # noqa: E402  (puts prep_v8 / prep_v9 / registry / captions / curate_check on the path)
from prep_v9b import CAP, CK, P, P9, R  # noqa: E402
sys.path.insert(0, str(HERE / "scenes"))
from rows_v10 import curation, load_rows  # noqa: E402

DATA = P.DATA
RAW = DATA / "raw"
STAGE = "v10"
SCENES = HERE / "scenes"
HOLDOUT_FILE = SCENES / "holdout_v10.json"
V9B_WINDOWS = SCENES / "v9b_535_scene_windows.json"
WS_DATA = "/workspace/data_v10"
WS_CACHE = "/workspace/wan_cache/latents_v10"
SW, SH = PB.SW, PB.SH
PILLARBOX = {f"iteration_7/293_videos/PP-{n}-Base.mp4" for n in ("BFPockets", "Favor", "Foot", "Muscle", "Sniff", "Wall")}
PILLAR_PX = 4


def vf_for(skit):
    if skit in PILLARBOX:
        h = 860                                   # 1920 * 480 / 1072 = 859.7
        return f"crop=1080-{2 * PILLAR_PX}:1920:{PILLAR_PX}:0,scale={SW}:{h}:flags=lanczos,crop={SW}:{SH}:0:{(h - SH) // 2}"
    return f"scale={SW}:{PB.SCALE_H}:flags=lanczos,crop={SW}:{SH}:0:{(PB.SCALE_H - SH) // 2}"


def decode(skit, expect):
    path = RAW / skit
    b = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path), "-vf", vf_for(skit),
                        *P9._passthrough_flag(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True).stdout
    v = np.frombuffer(b, np.uint8).reshape(-1, SH, SW, 3)
    if len(v) != expect:
        raise RuntimeError(f"{path.name}: decoded {len(v)} frames, curation says {expect}")
    if skit in PILLARBOX and (v[:, :, :2].max() < 16 or v[:, :, -2:].max() < 16):
        raise RuntimeError(f"{path.name}: black edge columns survived the pillarbox crop")
    return v


def build_skit(skit, frames, items, clips_dir, reencode):
    todo = [w for w in items if reencode or not (clips_dir / w["name"]).exists()]
    if todo:
        v = decode(skit, frames)
        for w in todo:
            PB.write_mp4(np.ascontiguousarray(v[w["a"]:w["b"]]), clips_dir / w["name"])
    return len(todo)


def select_replay(pool, total, minima, seed):
    """prep_v9b.select_replay's rule (>= 1 per (character, label) at its kind's common bucket,
    each label's anchor minimum spread over both characters, the rest at random), with one
    change: a group's quota is capped at the clips that exist, and what it cannot take goes to
    groups that still have clips. v10's scene set is ~3x v9b's, so happy's anchor (one replay
    clip per 4 scene windows using a happy word) asks for more happy clips than exist; without
    this the stage would fall short of its replay share. Returns picks, effective total and
    the capped groups."""
    import random
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for j in sorted(pool, key=lambda x: (x["char"], x["label"], x["angle"])):
        for tag, n in P9.replay_candidates(j):
            groups[(j["char"], j["label"])].append((j, tag, n))
    keys = sorted(groups)
    cap = {k: len(groups[k]) for k in keys}
    need = {k: 1 for k in keys}
    for lab, m in minima.items():
        ks = [k for k in keys if k[1] == lab]
        for i, k in enumerate(ks):
            need[k] = max(need[k], m // len(ks) + (1 if i < m % len(ks) else 0))
    capped = {f"{k[0]}/{k[1]}": (need[k], cap[k]) for k in keys if need[k] > cap[k]}
    quota = {k: min(need[k], cap[k]) for k in keys}
    total = max(total, sum(quota.values()))
    spare = total - sum(quota.values())
    while spare > 0:
        open_ = [k for k in keys if quota[k] < cap[k]]
        if not open_:
            break
        for k in rng.sample(open_, min(spare, len(open_))):
            quota[k] += 1; spare -= 1
    picked = []
    for k in keys:
        cands = groups[k][:]
        rng.shuffle(cands)
        common = P.COMMON[cands[0][0]["kind"]]
        first = next((c for c in cands if c[2] == common), None)
        take = ([first] if first else []) + [c for c in cands if c is not first]
        picked += take[:quota[k]]
    return picked, total, capped


def emit(scene_rows, replay_rows, out, prefix, cache_root, suffix):
    lines = [f"# Musubi dataset config - {STAGE} Stage 1 (auto-generated by prep_v10.py).",
             f"# Scene windows (v10 iteration_7 + v9b) at {SW}x{SH} + v8 replay at {P.SIZE}^2; one block per",
             "# (kind, frame length). Cache directories replay_f<n> carry the preservation marker",
             "# (cache_mark_v10.py). Plan: training_approach/v10/Training_Approach_v10.md.", "",
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


def readme(out, m):
    c = m["counts"]
    sc = sum(v for k, v in c.items() if k.startswith("scene"))
    rp = sum(v for k, v in c.items() if k.startswith("replay"))
    (out / "README.md").write_text(f"""# {out.name}: v10 Stage 1 training set

Built by `finetune/wan/prep_v10.py` (code repo) on {m['built']}, for **v10 Stage 1** (a fresh
high-noise LoRA; plan `training_approach/v10/Training_Approach_v10.md`, box guide
`training_approach/v10/HANDOVER_v10.md`).

| part | clips | from |
|---|---|---|
| v10 scene windows | {m['n_scene_v10']} | `processed/v10_scenes_748` rows that may train (`scenes/rows_v10.py`), minus the {len(m['holdout_skits'])} held-out skits (`scenes/holdout_v10.json`); cut from `raw/iteration_7` |
| v9b scene windows | {m['n_scene_v9b']} | the 348 windows v9b-1 trained on (`scenes/v9b_535_scene_windows.json`, U7); cut from `raw/iteration_1`, `raw/iteration_2` |
| v8 replay | {rp} | `prep_v9` builders from `raw/iteration_3`-`6`, 1024², flat grounds, v8 captions; QF2_R held out |
| **total** | **{sc + rp}** | replay {100 * rp / (sc + rp):.0f}% |

Scene clips: 480×848, 24 fps, x264 crf 12. Six iteration_7 skits had 4 px black columns at both
edges; they were cropped before scaling (`PILLARBOX` in prep_v10.py).

Files: `clips/`, `dataset_config_v10.toml` (paths on this machine) and
`dataset_config_v10.workspace.toml` (paths on the GPU box: `/workspace/data_v10`, cache
`/workspace/wan_cache/latents_v10/{{scene,replay}}_f<n>`), one jsonl per block, `manifest.json`
(every clip with its source, window, caption and curation md5).

Rebuild: `PUDGY_DATA=<Data> python finetune/wan/prep_v10.py --workers 8` (deterministic
selection; the encoder version can change bytes, not content).
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--reencode", action="store_true")
    a = ap.parse_args()

    probs = R.lint(R.load(), DATA)
    if probs:
        sys.exit("registry lint failed:\n  " + "\n  ".join(probs))
    for cid, nm in CAP.LEAD_NAMES.items():
        if R.phrase(CAP.REG, cid) != P.ANCHORS[nm]:
            sys.exit(f"registry anchor for {nm} differs from prep_v8 - the base was trained on prep_v8's")

    # ---- v10 scene windows ----
    rows, dropped = load_rows(DATA)
    hold = json.loads(HOLDOUT_FILE.read_text())
    held = set(hold["skits"])
    if not held <= {r["skit"] for r in rows}:
        sys.exit(f"holdout lists skits with no trainable rows: {sorted(held - {r['skit'] for r in rows})}")
    scene, per_skit, frames_of, probs = [], collections.defaultdict(list), {}, []
    checked = set()
    for r in rows:
        if r["skit"] in held:
            continue
        d, md5 = curation(r["curation"])
        if r["curation"] not in checked:
            errs, _w = CK.check(SCENES / r["curation"])
            if errs:
                probs.append(f"{r['curation']}: {errs[0]}")
            checked.add(r["curation"])
        sp = d["spans"][r["span"]]
        if CAP.compile_window(d, sp, r["a"], r["b"]) != r["caption"]:
            probs.append(f"{r['skit']} [{r['a']},{r['b']}): caption differs from its curation - re-run compile_captions_v10.py")
        probs += [f"{r['skit']} [{r['a']},{r['b']}): {p}" for p in PB.lint_caption(r["caption"])]
        w = {"name": PB.out_name(d, r["span"], r["a"], r["frames"]), "src": "v10", "skit": r["skit"],
             "span": r["span"], "a": r["a"], "b": r["b"], "frames": r["frames"], "caption": r["caption"],
             "camera": r["camera"], "two_shot": r["two_shot"], "set": r["set"],
             "curation": r["curation"], "curation_md5": md5, "pillarbox": r["skit"] in PILLARBOX}
        scene.append(w); per_skit[r["skit"]].append(w); frames_of[r["skit"]] = d["frames"]
    n10 = len(scene)

    # ---- v9b scene windows (U7) ----
    v9 = json.loads(V9B_WINDOWS.read_text())
    for r in v9["windows"]:
        f = HERE / r["curation"]
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        if md5 != r["curation_md5"]:
            probs.append(f"{r['curation']} changed since v9b prep (md5) - the frozen windows may no longer match")
        if r["skit"] in v9["holdout_skits"]:
            probs.append(f"{r['skit']} is in v9b's holdout")
        probs += [f"{r['skit']} [{r['a']},{r['b']}): {p}" for p in PB.lint_caption(r["caption"])]
        d = json.loads(f.read_text())
        w = {"name": r["name"], "src": "v9b", "skit": r["skit"], "span": r["span"], "a": r["a"], "b": r["b"],
             "frames": r["frames"], "caption": r["caption"], "camera": "static", "two_shot": r["two_shot"],
             "set": "kept", "curation": r["curation"], "curation_md5": md5, "pillarbox": False}
        scene.append(w); per_skit[r["skit"]].append(w); frames_of[r["skit"]] = d["frames"]
    if probs:
        sys.exit("checks failed:\n  " + "\n  ".join(probs[:30]))
    names = [w["name"] for w in scene]
    if len(names) != len(set(names)):
        sys.exit("duplicate output names")
    bk = collections.Counter(w["frames"] for w in scene)
    print(f"v10 rows that may train: {len(rows)}; dropped " + ", ".join(f"{k} {len(v)}" for k, v in sorted(dropped.items())))
    print(f"holdout: {len(held)} skits ({sum(r['skit'] in held for r in rows)} windows) + v9b's {len(v9['holdout_skits'])}")
    print(f"scene windows: {len(scene)} = v10 {n10} + v9b {len(scene) - n10}, from {len(per_skit)} skits  "
          f"buckets {dict(sorted(bk.items()))}  ({sum(w['frames'] for w in scene) / P.FPS / 60:.1f} min)  "
          f"camera moves {sum(w['camera'] != 'static' for w in scene)}  pillarbox {sum(w['pillarbox'] for w in scene)}")

    # ---- replay (D10 anchors over every scene caption) ----
    counts, minima = PB.anchor_minima([w["caption"] for w in scene])
    print("D10 label words in scene captions -> replay minimum:")
    for lab in PB.LABEL_WORDS:
        print(f"    {lab:<10} {counts[lab]:>4} windows -> >= {minima.get(lab, 0)}")
    pool, _props, _rep = P9.discover_v9()
    pool = [j for j in pool if j["angle"].split("_t")[0] not in P.HOLDOUT_ANGLES]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        pv = list(ex.map(P.preflight, pool))
    P.derive_frame_budgets(pool, [v[2] for v in pv])
    pool_dropped = [(j, why) for j, (ok, why, _n) in zip(pool, pv) if not ok]
    pool = [j for j, (ok, _w, _n) in zip(pool, pv) if ok]
    for j, why in pool_dropped:
        print(f"  x replay source {j['src'].name}: {why}")
    P.assign_zooms(pool)
    target = round(len(scene) * a.replay_frac / (1 - a.replay_frac))
    picks, total, capped = select_replay(pool, target, minima, a.seed)
    print(f"replay: {len(picks)} clips (target {target} = {a.replay_frac:.0%}; anchors raise it to {total}) "
          f"-> {len(picks) / (len(picks) + len(scene)):.1%} of the stage")
    for g, (want, have) in capped.items():
        print(f"  ! {g}: the anchor wants {want} clips, {have} distinct exist - all used, the rest spread over other groups")
    grp = collections.Counter(f"{j['char']}/{j['label']}" for j, _t, _n in picks)
    print("    " + "  ".join(f"{k} {v}" for k, v in sorted(grp.items())))
    if a.dry_run:
        return

    out = a.out / f"{STAGE}_build"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        n_written = sum(ex.map(lambda kv: build_skit(kv[0], frames_of[kv[0]], kv[1], clips_dir, a.reencode),
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
        print(f"  x replay dropped {r['job']['src'].name}: {r['error']}")
    replay = [{"name": o["name"], "frames": o["frames"], "caption": o["caption"],
               "source": str(Path(r["job"]["src"]).relative_to(DATA)), "char": r["job"]["char"],
               "label": r["job"]["label"], "angle": r["job"]["angle"], "background": o["background"], "zoom": o["zoom"]}
              for r in res_r if not r["error"] for o in r["outputs"]]

    expected = {w["name"] for w in scene} | {r["name"] for r in replay}
    if len(expected) != len(scene) + len(replay):
        sys.exit("a replay clip name collides with a scene clip name")
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
    import datetime as dt
    manifest = {
        "generated_by": "prep_v10.py", "stage": STAGE, "built": dt.date.today().isoformat(),
        "plan": "training_approach/v10/Training_Approach_v10.md section 0 (Q1-Q20), 3.6",
        "base": "high-noise: wan2.2 i2v high-noise fp16 + lora_highnoise_GOLDEN_ep40 (base_weights); low-noise frozen pudgy-v8-merged",
        "registry": R.snapshot(CAP.REG), "size": [SW, SH], "fps": P.FPS, "buckets": list(PB.BUCKETS),
        "replay_frac": a.replay_frac, "seed": a.seed, "anchor_ratio": PB.ANCHOR_RATIO,
        "label_word_counts": counts, "anchor_minima": minima, "anchor_capped": capped,
        "holdout_skits": sorted(held), "v9b_holdout_skits": v9["holdout_skits"],
        "holdout_angles": sorted(P.HOLDOUT_ANGLES), "pillarbox_skits": sorted(PILLARBOX),
        "dropped_rows": {k: len(v) for k, v in dropped.items()},
        "n_scene_v10": n10, "n_scene_v9b": len(scene) - n10, "counts": cnt,
        "scene": scene, "replay": replay,
        "replay_dropped": [{"source": str(j["src"]), "reason": w} for j, w in pool_dropped]
                          + [{"source": str(r["job"]["src"]), "reason": r["error"]} for r in bad],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    readme(out, manifest)
    print(f"\nbuilt {out}: {len(scene)} scene + {len(replay)} replay clips  {cnt}")
    print(f"  {cfg_ws.name} · manifest.json · README.md")


if __name__ == "__main__":
    main()
