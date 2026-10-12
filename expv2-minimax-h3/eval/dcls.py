#!/usr/bin/env python3
"""Tier 1 - the diffusion classifier (PLAN 7.2, 6.3; schema INTERFACES section 3).

Question per held-out clip: is the TRUE caption the best denoiser among the captions of its
kind? Expression clips are scored against the 8 expression captions (chance 0.125), motion
clips against the 4 trained motion captions (chance 0.25). Only the label clause differs
between candidates (prep builds them; truncation suffix held at the true clip's value).

Pass 1 (--encode; Qwen3-VL + VAEs, no DiT) runs the TRAINING unit chain on each clip
(h3lib.encode_training_path): clean video latents exactly as InputVideoEmbedder makes
them, the keyframe = frame 0 through KeyframeEncoder with seed 42, silent audio as
train.py's --silent_on_missing_audio, and prompt embeds + packed sequence per candidate.

Pass 2 (--ckpt; DiT only) for every (clip, sigma, draw): one shared noise draw (seeded
from C.TIER1_NOISE_SEED, clip, sigma, draw - identical for every checkpoint), noisy =
(1-sigma)x0 + sigma*eps, audio noised at the same sigma, then for each candidate the
flow-matching error |pred - (eps - x0)|^2:
  * on every video latent EXCEPT latent 0 (frame 0 alone; h3lib.latent_frames - G0.2b
    confirms the index on the box), and
  * change-masked: weight |z[f] - z[0]| (channel mean), normalised to mean 1 over the
    kept latents, so the static body does not swamp the part that moves.
The unmasked mean is stored too. DiffSynth's training_weight(t) is a per-sigma constant
and is not applied (it cannot change a ranking within a sigma).

Per clip: loss_true = mean over sigmas x draws of the true candidate; margin = mean of the
wrong candidates - loss_true; predicted = argmin. Distilled-model note (PLAN 7.2):
margins rank captions; their absolute size is not comparable with v8's.

    python eval/dcls.py --dry
    python eval/dcls.py --encode --set holdout,calib
    python eval/dcls.py --ckpt base,expv2-h3-s12@last,expv2-h3-s222@last --set holdout,calib --sigmas grid
    python eval/dcls.py --calibrate --pilots expv2-h3-s12@last,expv2-h3-s222@last     # CPU
    python eval/dcls.py --ckpt base,expv2-h3-main@all --sigmas calibrated
    python eval/dcls.py --summarize --ckpt base,expv2-h3-main@all                      # CPU
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3lib as L  # noqa: E402
C = L.C

OUT = C.EVAL / "tier1"
ENC = C.CACHE / "eval" / "tier1"
CALIB = OUT / "calibration.json"
KEYS = ("losses", "losses_unmasked")


def skey(s):
    return f"{float(s):.6f}"


# ---------------------------------------------------------------------------------------
# result files: schema lists <-> internal dicts
# ---------------------------------------------------------------------------------------

def load_result(name):
    """tier1/<name>.json -> internal {clip_id: {meta..., losses: {lab: {skey: [d...]}}}}"""
    rec = C.load_json(OUT / f"{name}.json")
    out = {}
    if not rec:
        return out
    sig = [skey(s) for s in rec.get("sigmas", [])]
    for c in rec.get("clips", []):
        d = {k: c[k] for k in ("clip_id", "split", "kind", "label") if k in c}
        for key in KEYS:
            d[key] = {}
            for lab, per_sig in c.get(key, {}).items():
                d[key][lab] = {sig[i]: [v for v in draws if v is not None]
                               for i, draws in enumerate(per_sig) if draws and any(v is not None for v in draws)}
        out[c["clip_id"]] = d
    return out


def save_result(name, internal, draws, summary):
    sigmas = sorted({s for c in internal.values() for lab in c["losses"].values() for s in lab}, key=float)
    clips = []
    for cid in sorted(internal):
        c = internal[cid]
        row = {k: c[k] for k in ("clip_id", "split", "kind", "label")}
        for key in KEYS:
            row[key] = {lab: [(per.get(s, []) + [None] * draws)[:draws] for s in sigmas]
                        for lab, per in c[key].items()}
        clips.append(row)
    C.write_json(OUT / f"{name}.json", {
        "ckpt": name, "sigmas": [float(s) for s in sigmas], "draws": draws,
        "noise_seed": C.TIER1_NOISE_SEED, "clips": clips, "summary": summary,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ---------------------------------------------------------------------------------------
# statistics (CPU)
# ---------------------------------------------------------------------------------------

def clip_mean(c, lab, sigmas, key="losses"):
    vals = [v for s in sigmas for v in c[key].get(lab, {}).get(skey(s), [])]
    return sum(vals) / len(vals) if vals else None


def per_clip(internal, sigmas, split, key="losses"):
    rows = []
    for c in internal.values():
        if c["split"] != split:
            continue
        means = {lab: clip_mean(c, lab, sigmas, key) for lab in c[key]}
        if any(v is None for v in means.values()) or c["label"] not in means:
            continue
        lt = means[c["label"]]
        wrong = [v for lab, v in means.items() if lab != c["label"]]
        rows.append({"clip_id": c["clip_id"], "kind": c["kind"], "label": c["label"],
                     "loss_true": lt, "margin": sum(wrong) / len(wrong) - lt,
                     "pred": min(means, key=means.get)})
    return rows


def summarize(internal, sigmas, base=None, split="holdout", key="losses"):
    rows = per_clip(internal, sigmas, split, key)
    brow = {r["clip_id"]: r for r in per_clip(base, sigmas, split, key)} if base else {}
    out = {"sigmas": [float(s) for s in sigmas], "split": split, "per_clip": rows}
    score = 0.0
    for kind in ("expression", "motion"):
        rk = [r for r in rows if r["kind"] == kind]
        if not rk:
            continue
        for r in rk:
            b = brow.get(r["clip_id"])
            r["base_loss_true"] = b["loss_true"] if b else None
        bb = [r for r in rk if r.get("base_loss_true") is not None]
        k = {"n": len(rk),
             "acc": round(sum(r["pred"] == r["label"] for r in rk) / len(rk), 4),
             "correct": sum(r["pred"] == r["label"] for r in rk),
             "mean_margin": sum(r["margin"] for r in rk) / len(rk),
             "mean_loss_true": sum(r["loss_true"] for r in rk) / len(rk),
             "below_base": sum(r["loss_true"] < r["base_loss_true"] for r in bb) if bb else None,
             "n_with_base": len(bb)}
        out[kind] = k
        score += k["mean_margin"]
    out["score"] = score            # expression + motion mean margin (PLAN 6.3.3, 7.3)
    return out


def calibrated_sigmas():
    cal = C.load_json(CALIB)
    return [float(s) for s in cal["sigmas"]] if cal and cal.get("sigmas") else None


def summary_sigmas(internal):
    have = sorted({float(s) for c in internal.values() for lab in c["losses"].values() for s in lab})
    cal = calibrated_sigmas()
    if cal and all(any(abs(h - s) < 1e-9 for h in have) for s in cal):
        return cal
    return have


def bootstrap_diff(rows_a, rows_b, n=10000, seed=0):
    """Paired bootstrap of (mean expr margin + mean motion margin)_a - (...)_b, resampling
    clips within each kind. Returns (diff, lo, hi)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    b = {r["clip_id"]: r for r in rows_b}
    diffs = {}
    for kind in ("expression", "motion"):
        d = [r["margin"] - b[r["clip_id"]]["margin"] for r in rows_a if r["kind"] == kind and r["clip_id"] in b]
        if d:
            diffs[kind] = np.array(d)
    point = float(sum(v.mean() for v in diffs.values()))
    boots = np.zeros(n)
    for v in diffs.values():
        idx = rng.integers(0, len(v), size=(n, len(v)))
        boots += v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def run_shift(name):
    """train_flow_shift recorded in the run's training_args.json (the shift patch adds it)."""
    run = name.split("@")[0]
    a = C.load_json(C.RUNS / run / "training_args.json") or {}
    return a.get("train_flow_shift")


def latest_result_name(spec):
    """'<run>@last' -> the newest checkpoint of that run that has a tier1 JSON, so
    --calibrate also works on a machine that holds the JSONs but not the weights."""
    if spec.endswith("@last"):
        run = spec[:-5]
        have = sorted((int(p.stem.split("@")[1]), p.stem) for p in OUT.glob(f"{run}@*.json")
                      if p.stem.split("@")[1].isdigit())
        if have:
            return have[-1][1]
    return L.resolve_ckpts(spec, must_exist=False)[0][0]


def calibrate(pilots, base_name="base"):
    """PLAN 6.3: (1) fix the two sigmas with the largest (pilot - base) margin on the calib
    set, averaged over both pilots; (2) the sanity gate; (3) the shift choice."""
    base = load_result(base_name)
    if not base:
        sys.exit(f"{OUT / (base_name + '.json')} missing: run --ckpt base --set holdout,calib --sigmas grid first")
    pres = {}
    for spec in pilots:
        name = latest_result_name(spec)
        pres[name] = load_result(name)
        if not pres[name]:
            sys.exit(f"{OUT / (name + '.json')} missing: run --ckpt {spec} --set holdout,calib --sigmas grid first")
    names = list(pres)

    gaps = {}
    for s in C.TIER1_SIGMA_GRID:
        bsum = summarize(base, [s], split="calib")
        g = []
        for n in names:
            ps = summarize(pres[n], [s], split="calib")
            g.append(sum(ps[k]["mean_margin"] - bsum[k]["mean_margin"]
                         for k in ("expression", "motion") if k in ps and k in bsum))
        gaps[skey(s)] = sum(g) / len(g)
    chosen = sorted(sorted(gaps, key=gaps.get, reverse=True)[:2], key=float)
    sig = [float(s) for s in chosen]

    summ = {n: summarize(pres[n], sig, base=base) for n in names}
    summ[base_name] = summarize(base, sig)

    sanity = {}
    for n in names:
        e, m = summ[n].get("expression", {}), summ[n].get("motion", {})
        need_e = math.ceil(0.75 * e.get("n", 0))
        need_m = math.ceil(0.75 * m.get("n", 0))
        ok = (e.get("below_base") or 0) >= need_e and (m.get("below_base") or 0) >= need_m
        mostly_above = ((e.get("below_base") or 0) < e.get("n", 0) / 2) and ((m.get("below_base") or 0) < m.get("n", 0) / 2)
        sanity[n] = {"pass": ok, "expression_below_base": e.get("below_base"), "expression_needed": need_e,
                     "motion_below_base": m.get("below_base"), "motion_needed": need_m,
                     "mostly_above_base": mostly_above}
    if any(v["pass"] for v in sanity.values()):
        action = "proceed"
    elif all(v["mostly_above_base"] for v in sanity.values()):
        action = "both pilots ABOVE the base on most clips: re-run the pilots at LR 5e-5 (PLAN 6.3.2)"
    else:
        action = "STOP: neither pilot passes - re-check the conditioning (G0.2b) before touching LR (PLAN 6.3.2)"

    shifts = {n: run_shift(n) for n in names}
    s12 = next((n for n in names if shifts[n] is not None and abs(float(shifts[n]) - 12) < 1e-6), names[0])
    other = next(n for n in names if n != s12) if len(names) > 1 else None
    shift = None
    if other:
        point, lo, hi = bootstrap_diff(summ[s12]["per_clip"], summ[other]["per_clip"])
        tie = lo <= 0 <= hi
        winner = s12 if (tie or point > 0) else other
        shift = {"s12": s12, "other": other, "diff_s12_minus_other": point, "ci95": [lo, hi], "tie": tie,
                 "winner": winner, "winner_train_flow_shift": shifts.get(winner) or (12.0 if winner == s12 else 2.22),
                 "rule": "higher held-out score (expr + motion mean margin); CI contains 0 -> S12 (PLAN 6.3.3)"}

    cal = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sigmas": sig, "frozen": True,
           "grid_gap_pilot_minus_base": gaps, "pilots": names, "pilot_shifts": shifts,
           "summaries": {n: {k: v for k, v in s.items() if k != "per_clip"} for n, s in summ.items()},
           "sanity": sanity, "action": action, "shift": shift}
    C.write_json(CALIB, cal)
    for n in names + [base_name]:   # refresh stored summaries at the frozen sigmas
        internal = pres.get(n, base)
        rec = C.load_json(OUT / f"{n}.json")
        save_result(n, internal, rec["draws"], summarize(internal, sig, base=None if n == base_name else base))
    print(L.dumps({k: cal[k] for k in ("sigmas", "grid_gap_pilot_minus_base", "sanity", "action", "shift")}))
    return cal


# ---------------------------------------------------------------------------------------
# GPU passes
# ---------------------------------------------------------------------------------------

def ordered_captions(item):
    caps = dict(item["candidates"])
    lab = item["label"]
    return {lab: caps[lab], **{k: v for k, v in caps.items() if k != lab}}


def encode(items, force=False):
    import torch
    ENC.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in items:
        p = ENC / f"{it['clip_id']}.pth"
        caps = ordered_captions(it)
        if p.exists() and not force:
            old = torch.load(str(p), map_location="cpu", weights_only=False)
            if old.get("caption_hashes") == {k: L.text_hash(v) for k, v in caps.items()}:
                continue
        todo.append((it, p, caps))
    print(f"pass 1: {len(todo)} clips to encode ({len(items) - len(todo)} cached)")
    if not todo:
        return
    pipe = L.load_pipe(("te", "vae", "audio_vae"))
    for i, (it, p, caps) in enumerate(todo):
        frames = L.read_frames(C.CORPUS / it["png_dir"], n=it["frames"])
        if frames[0].size != (C.SIZE, C.SIZE):
            raise ValueError(f"{it['clip_id']}: frames are {frames[0].size}, expected {C.SIZE}^2")
        t = time.time()
        enc = L.encode_training_path(pipe, frames, caps)
        enc.update({"clip_id": it["clip_id"], "split": it["split"], "kind": it["kind"], "label": it["label"]})
        assert enc["x0"].shape[2] == C.latents_for(it["frames"]), enc["x0"].shape
        torch.save(enc, str(p))
        print(f"  [{i + 1}/{len(todo)}] {it['clip_id']}  {len(caps)} captions  {time.time() - t:.1f}s")
    pipe = None
    L.free()


def score_ckpts(items, ckpts, sigmas, draws):
    import torch
    encs = {}
    for it in items:
        p = ENC / f"{it['clip_id']}.pth"
        if not p.exists():
            sys.exit(f"missing encoding {p}: run --encode first")
        encs[it["clip_id"]] = torch.load(str(p), map_location="cpu", weights_only=False)
    pipe = L.load_pipe(("dit",))
    L.onload(pipe, ["dit"])
    for name, path in ckpts:
        L.detach_lora(pipe)
        L.attach_lora(pipe, path)
        internal = load_result(name)
        base = load_result("base") if name != "base" else None
        t0 = time.time()
        nfwd = 0
        for it in items:
            cid = it["clip_id"]
            enc = encs[cid]
            c = internal.setdefault(cid, {"clip_id": cid, "split": it["split"], "kind": it["kind"],
                                          "label": it["label"], "losses": {}, "losses_unmasked": {}})
            x0 = enc["x0"].to(pipe.device).float()
            w = (x0[:, :, 1:] - x0[:, :, :1]).abs().mean(dim=1, keepdim=True)
            w = w / w.mean().clamp_min(1e-8)
            for s in sigmas:
                for d in range(draws):
                    if all(len(c["losses"].get(lab, {}).get(skey(s), [])) > d for lab in enc["cands"]):
                        continue
                    nv, na = L.fixed_noise(enc, cid, skey(s), d)
                    for lab in enc["cands"]:
                        pred, tgt = L.velocity_at(pipe, enc, lab, s, nv, na)
                        err = (pred - tgt) ** 2
                        kept = err[:, :, 1:]                     # drop latent 0 = frame 0
                        masked = float((kept * w).mean())
                        unmasked = float(kept.mean())
                        for key, v in (("losses", masked), ("losses_unmasked", unmasked)):
                            lst = c[key].setdefault(lab, {}).setdefault(skey(s), [])
                            if len(lst) == d:
                                lst.append(v)
                        nfwd += 1
            save_result(name, internal, draws, summarize(internal, summary_sigmas(internal), base=base))
        summ = summarize(internal, summary_sigmas(internal), base=base)
        save_result(name, internal, draws, summ)
        msg = " | ".join(f"{k} acc {summ[k]['acc']:.3f} margin {summ[k]['mean_margin']:.5f} below_base {summ[k]['below_base']}"
                         for k in ("expression", "motion") if k in summ)
        print(f"  {name}: {nfwd} forwards in {time.time() - t0:.0f}s  [{msg}]")
    L.detach_lora(pipe)
    pipe = None
    L.free()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encode", action="store_true", help="pass 1 (Qwen3-VL + VAEs)")
    ap.add_argument("--ckpt", default="", help="checkpoint specs (INTERFACES 2)")
    ap.add_argument("--set", default="holdout", help="holdout,calib")
    ap.add_argument("--sigmas", default="calibrated", help="grid | calibrated | comma list")
    ap.add_argument("--draws", type=int, default=C.TIER1_NOISE_DRAWS)
    ap.add_argument("--calibrate", action="store_true", help="PLAN 6.3 from existing grid results (CPU)")
    ap.add_argument("--pilots", default="expv2-h3-s12@last,expv2-h3-s222@last",
                    help="S12 first; the run's training_args.json train_flow_shift overrides the order")
    ap.add_argument("--summarize", action="store_true", help="recompute summaries (CPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate([s for s in args.pilots.split(",") if s])
        return 0

    sets = [s for s in args.set.split(",") if s]
    items = L.load_tier1_items(sets)
    if args.sigmas == "grid":
        sigmas = list(C.TIER1_SIGMA_GRID)
    elif args.sigmas == "calibrated":
        sigmas = calibrated_sigmas()
        if sigmas is None and (args.ckpt or args.dry):
            print(f"no {CALIB} yet; --sigmas calibrated falls back to the grid")
            sigmas = list(C.TIER1_SIGMA_GRID)
    else:
        sigmas = [float(s) for s in args.sigmas.split(",")]

    if args.dry:
        from collections import Counter
        probs = []
        cnt = Counter((it["split"], it["kind"]) for it in items)
        print(f"Tier 1: {len(items)} clips  {dict(cnt)}")
        for it in items:
            d = C.CORPUS / it["png_dir"]
            n = len(L.png_files(d))
            if n < it["frames"]:
                probs.append(f"{it['clip_id']}: {n} PNGs < {it['frames']}")
            want = len(C.EXPR_LABELS) if it["kind"] == "expression" else len(C.MOTION_TRAINED)
            if len(it["candidates"]) != want or it["label"] not in it["candidates"]:
                probs.append(f"{it['clip_id']}: {len(it['candidates'])} candidates (want {want}, incl. the true label)")
            skel = {c.replace(C.label_clause(lab, it.get("truncated", False)), "<L>")
                    for lab, c in it["candidates"].items()}
            if len(skel) != 1 or "<L>" not in next(iter(skel)):
                probs.append(f"{it['clip_id']}: candidates differ outside the label clause")
        if any(it["split"] == "holdout" for it in items):
            nh = Counter(it["kind"] for it in items if it["split"] == "holdout")
            print(f"  holdout by kind: {dict(nh)} (PLAN: expression 16, motion 7)")
        have = sum((ENC / f"{it['clip_id']}.pth").exists() for it in items)
        print(f"  pass-1 encodings cached: {have}/{len(items)} under {ENC}")
        print(f"  sigmas: {sigmas}  draws: {args.draws}")
        if args.ckpt:
            for name, path in L.resolve_ckpts(args.ckpt, must_exist=False):
                print(f"  ckpt {name:28s} {path or '(no LoRA)'}{'' if path is None or Path(path).exists() else '  [missing]'}")
            nc = sum(len(it["candidates"]) for it in items)
            print(f"  forwards per checkpoint: {nc * len(sigmas) * args.draws}")
        print("DRY: problems:\n  - " + "\n  - ".join(probs) if probs else "DRY: inputs OK")
        return 1 if probs else 0

    if args.summarize:
        names = [n for n, _ in L.resolve_ckpts(args.ckpt or "base", must_exist=False)]
        base = load_result("base")
        for n in names:
            internal = load_result(n)
            if not internal:
                print(f"  {n}: no results")
                continue
            rec = C.load_json(OUT / f"{n}.json")
            summ = summarize(internal, summary_sigmas(internal), base=None if n == "base" else base)
            save_result(n, internal, rec["draws"], summ)
            print(f"  {n}: " + " | ".join(f"{k} acc {summ[k]['acc']:.3f} margin {summ[k]['mean_margin']:.5f} "
                                          f"below_base {summ[k]['below_base']}" for k in ("expression", "motion") if k in summ))
        return 0

    if args.encode:
        encode(items, force=args.force)
    if args.ckpt:
        ckpts = L.resolve_ckpts(args.ckpt)
        if any(n == "base" for n, _ in ckpts):   # base first, so later summaries can count below_base
            ckpts = [c for c in ckpts if c[0] == "base"] + [c for c in ckpts if c[0] != "base"]
        score_ckpts(items, ckpts, sigmas, args.draws)
    if not (args.encode or args.ckpt):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
