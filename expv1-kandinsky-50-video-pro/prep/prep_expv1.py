#!/usr/bin/env python3
"""expv1 corpus: raw ProRes alpha clips -> Data/processed/expv1_k5pro/ (PLAN §5).

Per source (single character, a caption recipe, not QF2_R, decodable):
  * lengths (§5.1, amended 2026-10-13): Kandinsky-legal only (13, 17, 8k+1; MODEL_NOTES §2.8)
      floor   the largest legal length every label reaches with >= half its sources
              (computed; asserted to contain every label)
      eval    the kind's Tier-1 / core length (f13 motion, f17 expression), so the lengths
              §7 and §8b measure at are trained lengths (an addition to §5.1's two rungs)
      top     the SHORTEST legal length that covers the whole active arc (active length =
              up to the last frame that still moves; the dead tail beyond it is trimmed,
              holds inside the clip are kept; A5), if the real frames reach it; otherwise
              the longest legal length before the first missing frame (A6), capped at 57
    a rung is captioned as the complete label only when it holds the whole arc (missing at
    most ARC_SLACK trailing active frames, and not cut by a missing-frame gap); every other
    rung is "<label>, opening frames only".
  * grounds (§5.4): white, pastel blue, peach, mint; one shot size per (source, ground) from
    the zoom-out ladder 1.00 / 0.75 / 0.55, round-robin, balance asserted against label,
    character, ground and angle (tolerance 0.34, prep_v8's rule).
  * captions: expv1_common.caption (PLAN §5.5), with the per-source truncation flag.

Exclusions (manifest.json, each with its reason): the QF2_R holdout (§5.3), files whose
label/angle cannot be known (A9), files that tear on all 24 decodes (A7), sources with too
few real frames for the floor (A6), and character disagreements: the character is the
one the measured body hue says (blue Pax / pink Polly); a source whose hue contradicts its
name is listed and, unless whitelisted in HUE_CALLS, makes prep exit (A9).

Assertions (§5.6; prep exits non-zero): exact frame count on every written file (ffprobe),
rungs cut before the first missing frame, floor contains every label of both kinds, zoom
balance, colour words (Pax: blue, never pink; Polly: pink), <= 200 Qwen tokens per caption,
no QF2_R / lavender / sky-blue in training, MD5-unique sources.

    python prep/prep_expv1.py --dry-run          # plan only: lengths, counts, exclusions
    python prep/prep_expv1.py --workers 48       # write clips + captions + dataset.toml
Then (musubi checkout, patched):
    python src/musubi_tuner/kandinsky5_cache_latents.py --dataset_config <OUT>/dataset.toml \
        --vae <VAE_DIR> --first_frame_image_encode
    python src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py --dataset_config <OUT>/dataset.toml \
        --text_encoder_qwen <QWEN> --text_encoder_clip <CLIP> --batch_size 16
"""
import argparse
import collections
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv1_common as C  # noqa: E402

OUT = C.DATA / "processed" / "expv1_k5pro"
CACHE = C.WORK / "cache"
TRAIN_GROUNDS = ("white", "pastel blue", "peach", "mint")
ZOOMS = (1.00, 0.75, 0.55)
MAX_LEN = 57
ACTIVE_EPS = 0.5          # grey levels: a frame pair below this inside the character is "not moving"
ZOOM_TOL = 0.34
ARC_SLACK = 3             # trailing active frames a "complete" rung may drop (0.125 s)
MAX_TOKENS = 200
# Manual calls for hue/name disagreements (A9): src basename -> character. Empty until one is needed.
HUE_CALLS = {}


def legal_upto(n):
    return [m for m in range(5, n + 1) if C.k5_len_ok(m)]


def body_hue(path):
    """Mean colour of saturated character pixels in frame 0 -> 'Pax' (blue) or 'Polly' (pink)."""
    fr = C.read_rgba(path, 1, 192)[0].astype(np.float32)
    a = fr[..., 3] > 200
    rgb = fr[..., :3]
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    m = a & (sat > 0.25) & (mx > 80)
    if m.sum() < 50:
        return None, {}
    r, g, b = (float(rgb[..., i][m].mean()) for i in range(3))
    return ("Pax" if b > r else "Polly"), {"r": round(r, 1), "g": round(g, 1), "b": round(b, 1), "px": int(m.sum())}


def active_length(path, n):
    """1 + index of the last frame that differs from its predecessor inside the character."""
    fr = C.read_rgba(path, n, 192).astype(np.float32)
    grey = fr[..., :3].mean(-1)
    alpha = fr[..., 3] > 8
    last = 0
    for t in range(1, n):
        m = alpha[t] | alpha[t - 1]
        if m.any() and np.abs(grey[t] - grey[t - 1])[m].mean() > ACTIVE_EPS:
            last = t
    return last + 1


def stream_frames(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return len([t for t in out.split() if t.strip()])


def probe(j):
    j["gap"] = C.real_frame_gap(j["src"])
    j["stream_frames"] = stream_frames(j["src"])
    j["hue_char"], j["hue"] = body_hue(j["src"])
    j["active"] = active_length(j["src"], min(j["gap"], 64)) if j["gap"] >= 5 else 0
    return j


def plan_lengths(jobs):
    # floor: largest legal length every label reaches with >= half its sources
    by = collections.defaultdict(list)
    for j in jobs:
        by[j["label"]].append(j["gap"])
    cands = [n for n in legal_upto(MAX_LEN)]
    floor = max(n for n in cands if all(sum(g >= n for g in gs) * 2 >= len(gs) for gs in by.values()))
    for j in jobs:
        avail = min(j["gap"], MAX_LEN)
        legal = legal_upto(avail)
        cover = [n for n in legal if n >= max(j["active"], floor)]
        top = cover[0] if cover else max(legal, default=0)
        cut_by_gap = j["gap"] < j["stream_frames"]
        complete = top >= j["active"] - ARC_SLACK and not (cut_by_gap and top >= j["gap"] - ARC_SLACK)
        # caption rule: a rung is the complete label iff it reaches `native`
        j["native"] = top if complete else top + 1
        j["top"] = top
        j["arc_complete"] = complete
        j["rungs"] = sorted({n for n in (floor, C.H2H_WINDOW[j["kind"]], top) if 0 < n <= top})
    return floor


def assign_zooms(jobs):
    """prep_v8.assign_zooms: round-robin over the ladder in (kind, char, label, angle) order,
    shifted per ground, so zoom tracks none of them."""
    for i, j in enumerate(sorted(jobs, key=lambda x: (x["kind"], x["char"], x["label"], x["angle"]))):
        j["zooms"] = {g: ZOOMS[(i + k) % len(ZOOMS)] for k, g in enumerate(TRAIN_GROUNDS)}


def verify_zoom_balance(jobs):
    even = 1.0 / len(ZOOMS)
    factors = {"label": lambda j, g: j["label"], "character": lambda j, g: j["char"],
               "ground": lambda j, g: g, "angle": lambda j, g: j["angle"].split("_t")[0]}
    problems = []
    for fname, key in factors.items():
        tab = collections.defaultdict(collections.Counter)
        for j in jobs:
            for g in TRAIN_GROUNDS:
                tab[key(j, g)][j["zooms"][g]] += 1
        for level, c in tab.items():
            n = sum(c.values())
            for z in ZOOMS:
                if abs(c[z] / n - even) > ZOOM_TOL:
                    problems.append(f"{fname}={level}: zoom {z} at {c[z]}/{n}")
    return problems


def zoomed(path, n, zoom):
    if zoom >= 0.999:
        return C.decode_checked(path, n)
    s = int(round(C.SIZE * zoom))
    s -= s % 2
    small = C.decode_checked(path, n, size=s)
    canvas = np.zeros((n, C.SIZE, C.SIZE, 4), np.uint8)
    o = (C.SIZE - s) // 2
    canvas[:, o:o + s, o:o + s] = small
    return canvas


def ffprobe_frames(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def write_source(j):
    """All rungs x grounds of one source. Returns (records, error)."""
    recs = []
    try:
        decoded = {}
        for g in TRAIN_GROUNDS:
            z = j["zooms"][g]
            if z not in decoded:
                decoded[z] = zoomed(j["src"], j["top"], z)
            fr = C.composite(decoded[z], C.GROUNDS[g])
            for n in j["rungs"]:
                stem = f"{j['char'].lower()}_{j['label']}_{j['angle']}__{g.replace(' ', '')}_z{int(z * 100):03d}_f{n}"
                dest = OUT / "clips" / f"f{n}" / f"{stem}.mp4"
                if not dest.exists() or ffprobe_frames(dest) != n:
                    C.write_mp4(fr[:n], dest)
                cap = C.caption(j["char"], j["label"], j["angle"].split("_t")[0], n, zoom=z, ground=g,
                                native=j["native"])
                dest.with_suffix(".txt").write_text(cap)
                recs.append({"file": str(dest), "frames": n, "char": j["char"], "label": j["label"],
                             "kind": j["kind"], "angle": j["angle"], "ground": g, "zoom": z,
                             "truncated": n < j["native"], "source": j["src"], "md5": j["md5"]})
    except RuntimeError as e:
        return [], str(e)
    return recs, None


def check_captions(recs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.QWEN))
    caps = {Path(r["file"]).with_suffix(".txt").read_text() for r in recs}
    worst = 0
    bad = []
    for c in caps:
        n = len(tok(c)["input_ids"])
        worst = max(worst, n)
        if n > MAX_TOKENS:
            bad.append((n, c[:80]))
    return worst, bad, len(caps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    jobs, skipped = C.discover()
    excluded = [{"src": s, "reason": r} for s, r in dict(skipped).items()]
    md5s = collections.Counter(j["md5"] for j in jobs)
    assert max(md5s.values()) == 1, "md5 duplicates survived discovery"
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = list(ex.map(probe, jobs))

    open_disagreements = []
    keep = []
    for j in jobs:
        name = Path(j["src"]).name
        if j["angle"].startswith(C.HOLDOUT_ANGLE):
            excluded.append({"src": j["src"], "reason": "holdout angle QF2_R (PLAN §5.3)"}); continue
        if j["hue_char"] and j["hue_char"] != j["char"]:
            call = HUE_CALLS.get(name)
            if call is None:
                open_disagreements.append({"src": j["src"], "named": j["char"], "hue": j["hue_char"], **j["hue"]})
                continue
            j["char_note"] = f"name says {j['char']}, hue says {j['hue_char']}; manual call {call}"
            j["char"] = call
        keep.append(j)
    if open_disagreements:
        print(json.dumps(open_disagreements, indent=1))
        sys.exit(f"{len(open_disagreements)} character disagreement(s) open: record a manual call in HUE_CALLS (A9)")

    floor = plan_lengths(keep)
    dropped = [j for j in keep if not j["rungs"] or j["top"] < floor]
    for j in dropped:
        excluded.append({"src": j["src"], "reason": f"first missing frame #{j['gap']} / active {j['active']} "
                                                    f"below the f{floor} floor (A6)"})
    keep = [j for j in keep if j not in dropped]
    labels_at_floor = {j["label"] for j in keep if floor in j["rungs"]}
    if labels_at_floor != set(C.LABELS):
        sys.exit(f"floor f{floor} misses labels {sorted(set(C.LABELS) - labels_at_floor)} (B11)")
    assign_zooms(keep)
    problems = verify_zoom_balance(keep)
    if problems:
        sys.exit("zoom balance: " + "; ".join(problems[:10]))

    n_clips = sum(len(j["rungs"]) * len(TRAIN_GROUNDS) for j in keep)
    per_len = collections.Counter(n for j in keep for n in j["rungs"] for _ in TRAIN_GROUNDS)
    perf = {(j["char"], j["label"]) for j in keep}
    print(f"sources {len(keep)} ({len(perf)} performances), floor f{floor}, clips {n_clips}")
    print("clips per length:", dict(sorted(per_len.items())))
    lab_tab = collections.defaultdict(lambda: collections.Counter())
    for j in keep:
        for n in j["rungs"]:
            lab_tab[j["label"]][n] += len(TRAIN_GROUNDS)
    for lab in C.LABELS:
        print(f"  {lab:<10} " + " ".join(f"f{n}:{c}" for n, c in sorted(lab_tab[lab].items())))
    # the Tier-1 / core prompts (recipe natives) must say what training said at the eval length
    for j in keep:
        n = C.H2H_WINDOW[j["kind"]]
        if n in j["rungs"] and (n < j["native"]) != (n < C.recipe(j["label"])["native"]):
            print(f"  ! eval caption disagrees with training at f{n}: {Path(j['src']).name} "
                  f"(top {j['top']}, complete {j['arc_complete']})")
    print(f"excluded {len(excluded)}:")
    for e in excluded:
        print(f"  - {Path(e['src']).name}: {e['reason']}")
    if a.dry_run:
        return

    recs, errors = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for j, (r, err) in zip(keep, ex.map(write_source, keep)):
            if err:
                errors.append({"src": j["src"], "reason": f"torn on every decode (A7): {err}"})
            recs += r
    excluded += errors
    # assertions (§5.6)
    bad_len = [r["file"] for r in recs if ffprobe_frames(r["file"]) != r["frames"]]
    if bad_len:
        sys.exit(f"{len(bad_len)} files with the wrong frame count, e.g. {bad_len[:3]}")
    for r in recs:
        cap = Path(r["file"]).with_suffix(".txt").read_text()
        if r["char"] == "Pax" and ("pink" in cap or "blue" not in cap):
            sys.exit(f"colour words wrong in {r['file']}")
        if r["char"] == "Polly" and "pink" not in cap:
            sys.exit(f"colour words wrong in {r['file']}")
        if r["angle"].startswith(C.HOLDOUT_ANGLE) or r["ground"] in ("lavender", "sky-blue"):
            sys.exit(f"held-out material in training: {r['file']}")
    worst, bad, n_caps = check_captions(recs)
    if bad:
        sys.exit(f"{len(bad)} captions over {MAX_TOKENS} Qwen tokens: {bad[:2]}")
    if {r["label"] for r in recs if r["frames"] == floor} != set(C.LABELS):
        sys.exit("floor bucket lost a label after decoding")

    lines = ["[general]", f"resolution = [{C.SIZE}, {C.SIZE}]", 'caption_extension = ".txt"',
             "batch_size = 1", "enable_bucket = false", ""]
    for n in sorted({r["frames"] for r in recs}):
        lines += ["[[datasets]]", f'video_directory = "{OUT / "clips" / f"f{n}"}"',
                  f'cache_directory = "{CACHE / f"f{n}"}"', f"target_frames = [{n}]",
                  'frame_extraction = "head"', "source_fps = 24.0", "num_repeats = 1", ""]
    (OUT / "dataset.toml").write_text("\n".join(lines))
    manifest = {"floor": floor, "eval_lengths": C.H2H_WINDOW, "grounds": TRAIN_GROUNDS, "zooms": ZOOMS,
                "n_clips": len(recs), "n_sources": len(keep) - len(errors), "performances": len(perf),
                "max_caption_tokens": worst, "n_captions": n_caps, "negative_prompt": C.NEG,
                "sources": [{k: j[k] for k in ("src", "md5", "char", "label", "kind", "angle", "gap",
                                               "stream_frames", "active", "top", "arc_complete", "rungs",
                                               "zooms", "hue")} | (
                                {"char_note": j["char_note"]} if "char_note" in j else {}) for j in keep],
                "excluded": excluded, "clips": recs}
    C.write_json(OUT / "manifest.json", manifest)
    print(f"wrote {len(recs)} clips ({n_caps} distinct captions, max {worst} tokens) -> {OUT}\n"
          f"dataset config -> {OUT / 'dataset.toml'}")


if __name__ == "__main__":
    main()
