#!/usr/bin/env python3
"""G1/G2 generation and the G2 automatic checks (Training_Approach_v10.md section 5.1; Q12, Q14, Q19).

Jobs: /workspace/eval_v10/g1/jobs.json (build_eval_v10.py): one per held-out span (26 iteration_7 +
v9b's 8), each with its first frame, its own caption and frame count (f49 / f81) and the real shot.

  gen     I2V, both experts, E5 on the phone phrase, 480x848, seeds 42/43/44 (seed-major order, so a
          --max-new slice fills whole seeds first), plus the FLF subset (jobs with "flf": first frame + the
          real LAST frame, seed 42; a visual report only, Q19). One model load per config; every
          clip is saved as it is made; existing clips are skipped (resumable).
  score   per clip, against its real shot (written to runs/<RUN>/suite/scores.json, skip existing):
            dup        frames with two bodies of one lead's colour (longform/dupcheck.py), gen vs real:
                       the clip is FLAGGED only if it has more such frames than the real shot (Q14)
            zoom       static-camera jobs: ORB similarity fit frame 0 -> every 4th frame on the
                       background with the leads masked (scenes/span_motion_check.fit, 540x960);
                       zoom % = max over the series of the median |scale - 1| of 3 consecutive fits
                       (a zoom persists, a one-fit spike does not); "unverifiable" when no fit has
                       >= 20 inliers (plain grounds)
            motion     mean per-frame change gen / real shot (suite_v9b, 180x320)
            identity   DINOv2-base cosine of each lead's crop (largest dupcheck HSV blob, padded square)
                       at 8 frames vs its frame-0 crop, mean over the leads in frame 0
            ocr        RapidOCR on frame 0 and every 6th frame: text (score >= 0.6, >= 3 alphanumerics)
                       absent from frame 0 AND from everything OCR reads in the real shot; in >= 2 sampled
                       frames = flagged, in 1 = suspect (listed)
            drift / frozen / jump   suite_v9b's qa numbers, reported only
  sheets  per job: the real shot, then each config's seed-42 clip (7 frames); FLF sheets apart
  gate    G2 for --cand vs --base (v8merged), I2V clips only. All must hold:
            dup       no flagged clip                       else REVIEW: eye-check the listed clips (Q14)
            zoom      no static-camera clip measured > 3%   (unverifiable clips listed, not failed; a job
                      whose locked REAL shot reads > 1.5% - a moving prop drags the fit - is reported, not gated)
            motion    median ratio over the clips in [0.7, 1.3]
            identity  mean identity(cand) >= mean identity(base), paired over (job, seed)
                      (the paired difference's CI is reported; the bar is the literal spec)
            ocr       no clip with new text in >= 2 sampled frames   else REVIEW (eye check)
  selftest  scores the REAL shots as if generated (config "real") and gates real vs real: every
            check must pass trivially (scorer sanity, runs on a laptop without a GPU)

Box (CUDA_VISIBLE_DEVICES=1 by default; RUN_NAME=pudgy-v10-highnoise):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY suite_v10.py gen --config v8merged [--max-new 6]          # Phase 0 #2: 102 I2V + 8 FLF clips (~12-14 h)
    $PY suite_v10.py gen --config v10@150                          # candidate
    $PY suite_v10.py score --configs v8merged v10@150              # CPU/GPU, ~1 min per clip incl. OCR (est.)
    $PY suite_v10.py sheets --configs v8merged v10@150
    $PY suite_v10.py gate --cand v10@150                           # writes runs/<RUN>/suite/g2_v10_150.json
    $PY suite_v10.py all --config v10@150                          # gen + score (cand and base) + sheets + gate
    $PY suite_v10.py gen --config v9b@100 --only v9b --no-flf      # Phase 0 #1 scene suite (phase0_v10.py)
Outputs: /workspace/eval_v10/runs/$RUN_NAME/suite/{<cfg>/<tag>_s<seed>.mp4, <cfg>/<tag>_flf_s42.mp4,
         scores.json, sheets/*.png, g2_<cand>.json}
Exit codes: gen 0 (last line "REMAINING <k>": clips still to make; with --max-new N it stops after N new
            clips; re-run to continue; 0 = complete); `all` prints REMAINING <k> and exits 0 before scoring
            while k > 0;
            gate 0 PASS / 11 REVIEW (flagged clips need an eye check) / 10 FAIL / 2 clips or scores missing.
Needs: rapidocr_onnxruntime in the venv (pip install rapidocr_onnxruntime); DINOv2 from the HF cache
(common_v10.py prefetch).
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import common_v10 as C  # noqa: E402  (first: CUDA_VISIBLE_DEVICES)
import cv2
import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("V9B_WORK", str(C.RESULTS / ".v9b_work"))     # curate_tools exits without it when PUDGY_DATA is set
import dupcheck as DC  # noqa: E402  (longform/dupcheck.py)
import span_motion_check as SM  # noqa: E402  (scenes/span_motion_check.py)

ROOT = C.RESULTS / "suite"
ZOOM_MAX, MOTION_LO, MOTION_HI = 3.0, 0.7, 1.3
ZOOM_REAL_MAX = 1.5      # a locked real shot reading more than this = the background fit is unreliable on that job
OCR_MIN_SCORE, OCR_MIN_LEN, OCR_STEP, ID_FRAMES = 0.6, 3, 6, 8
EXIT_PASS, EXIT_FAIL, EXIT_REVIEW, EXIT_MISSING = 0, 10, 11, 2


def load_jobs(only=None):
    jobs = json.loads((C.G1DIR / "jobs.json").read_text())
    return [j for j in jobs if not only or j["source"] == only]


def expand(jobs, seeds, flf=True, flf_only=False):
    """Seed-major I2V jobs, then the FLF jobs (seed 42)."""
    out = []
    if not flf_only:
        for s in seeds:
            for j in jobs:
                out.append({"tag": f"{j['tag']}_s{s}", "prompt": j["prompt"], "start": j["start"], "frames": j["frames"],
                            "seed": s, "base": j["tag"], "flf": False})
    if flf or flf_only:
        for j in jobs:
            if j.get("flf"):
                out.append({"tag": f"{j['tag']}_flf_s42", "prompt": j["prompt"], "start": j["start"], "end": j["end"],
                            "frames": j["frames"], "seed": 42, "base": j["tag"], "flf": True})
    return out


def cfg_out(config):
    return ROOT / C.cfg_dir(config)


# ---------------------------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------------------------
def frames(p, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(p), "-vf", f"scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def native(p):
    return frames(p, C.SW, C.SH)


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def dup_frames(p):
    return int(sum(DC.bodies(f, *DC.PAX) >= 2 or DC.bodies(f, *DC.POLLY) >= 2 for f in DC.frames(p)))


def zoom(p):
    """Background similarity fit frame 0 -> every 4th frame (and the last), leads masked. A camera
    zoom persists; a prop or an unmasked body part dragging one RANSAC fit does not (a real locked
    shot, Brainrot span 1, spikes to 2.7% at one frame and is back at 1.000 four frames later). So
    zoom % = the max over the series of the MEDIAN of 3 consecutive fits' |scale - 1| (the raw
    single-fit max is reported as zoom_pct_raw)."""
    v = frames(p, SM.W, SM.H)
    ks = sorted(set(list(range(4, len(v), 4)) + [len(v) - 1]))
    fits = [SM.fit(v[0], v[k]) for k in ks]
    good = [(k, f) for k, f in zip(ks, fits) if f and f["inliers"] >= SM.MIN_INL]
    if not good:
        return {"verdict": "unverifiable", "zoom_pct": None, "checked": 0, "of": len(ks)}
    dev = [abs(f["scale"] - 1) for _, f in good]
    runs = [float(np.median(dev[i:i + 3])) for i in range(max(1, len(dev) - 2))]
    i = int(np.argmax(runs))
    z = 100 * runs[i]
    return {"verdict": "zoom" if z > ZOOM_MAX else "ok", "zoom_pct": round(z, 2), "at_frame": good[min(i + 1, len(good) - 1)][0],
            "zoom_pct_raw": round(100 * max(dev), 2), "max_shift_px1080": max(f["shift"] for _, f in good),
            "checked": len(good), "of": len(ks)}


def qa(v180, rv180, start_png):
    v = v180.astype(np.float32)
    d = np.abs(np.diff(v, axis=0)).mean((1, 2, 3))
    rd = np.abs(np.diff(rv180.astype(np.float32), axis=0)).mean((1, 2, 3))
    ref = np.asarray(Image.open(start_png).convert("RGB").resize((180, 320)), np.float32)
    border = np.zeros(v.shape[1:3], bool); border[:24] = border[-24:] = True; border[:, :14] = border[:, -14:] = True
    return {"motion": round(float(d.mean() / max(rd.mean(), 1e-6)), 3),
            "drift": round(float(np.abs(v[-4:].mean(0) - ref)[border].mean()), 2),
            "frozen": round(float((d < 0.15).mean()), 3),
            "jump": round(float(d.max() / max(np.median(d), 1e-6)), 2)}


_M = {}


def dino():
    if not _M:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _M.update(torch=torch, dev=dev, model=AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval(),
                  proc=AutoImageProcessor.from_pretrained("facebook/dinov2-base"))
    return _M


def lead_box(f, lo, hi):
    """Bounding box (full-res) of the largest blob of a lead's body colour (dupcheck's mask at 270x480)."""
    small = cv2.resize(f, (DC.W, DC.H), interpolation=cv2.INTER_AREA)
    m = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_RGB2HSV), lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < DC.MIN_ABS * DC.W * DC.H:
        return None
    sx, sy = f.shape[1] / DC.W, f.shape[0] / DC.H
    x, y, w, h = (st[i, cv2.CC_STAT_LEFT] * sx, st[i, cv2.CC_STAT_TOP] * sy,
                  st[i, cv2.CC_STAT_WIDTH] * sx, st[i, cv2.CC_STAT_HEIGHT] * sy)
    px, py = 0.15 * w, 0.15 * h
    x0, y0 = int(max(0, x - px)), int(max(0, y - py))
    x1, y1 = int(min(f.shape[1], x + w + px)), int(min(f.shape[0], y + h + py))
    return x0, y0, x1, y1


def square(crop):
    h, w = crop.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(out)


def identity(v):
    """Mean over the leads present in frame 0 of the mean DINO cosine (lead crop at frame k vs frame 0)
    over ID_FRAMES frames; frames where the lead is not found are skipped (found share reported)."""
    m = dino(); t = m["torch"]
    ks = np.unique(np.linspace(1, len(v) - 1, ID_FRAMES).round().astype(int))
    per = {}
    for name, (lo, hi) in (("pax", DC.PAX), ("polly", DC.POLLY)):
        b0 = lead_box(v[0], lo, hi)
        if b0 is None:
            continue
        crops, idx = [square(v[0][b0[1]:b0[3], b0[0]:b0[2]])], []
        for k in ks:
            b = lead_box(v[k], lo, hi)
            if b is not None:
                crops.append(square(v[k][b[1]:b[3], b[0]:b[2]])); idx.append(int(k))
        if not idx:
            per[name] = {"sim": 0.0, "found": 0.0}
            continue
        with t.no_grad():
            e = m["model"](**m["proc"](images=crops, return_tensors="pt").to(m["dev"])).last_hidden_state[:, 0]
            e = t.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        sims = e[1:] @ e[0]
        per[name] = {"sim": round(float(sims.mean()), 4), "late": round(float(sims[len(sims) // 2:].mean()), 4),
                     "found": round(len(idx) / len(ks), 3)}
    if not per:
        return {"identity": None, "leads": {}}
    return {"identity": round(float(np.mean([x["sim"] for x in per.values()])), 4), "leads": per}


_OCR = []


def norm(t):
    return re.sub(r"[^0-9a-z]", "", t.lower())


def ocr_texts(f):
    if not _OCR:
        from rapidocr_onnxruntime import RapidOCR
        _OCR.append(RapidOCR())
    res, _ = _OCR[0](cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return sorted({norm(t) for _box, t, s in (res or []) if float(s) >= OCR_MIN_SCORE and len(norm(t)) >= OCR_MIN_LEN})


def known(t, ref):
    return any(t in r or r in t or difflib.SequenceMatcher(None, t, r).ratio() >= 0.6 for r in ref)


def ocr_scan(v):
    """{frame: texts} on frame 0 and every OCR_STEP-th frame (and the last)."""
    return {int(k): ocr_texts(v[k]) for k in [0] + list(range(OCR_STEP, len(v), OCR_STEP)) + [len(v) - 1]}


def ocr(v, real_texts=()):
    """Text in the clip that is absent from its frame 0 AND from everything OCR reads in the real
    shot (scene text the action reveals, e.g. Hype's PAX foam finger, and OCR's own readings of
    the drawing are then not counted). Self-test on the 34 real shots: with frame 0 alone, 4 of 34
    'fail'; relative to the real shot, by construction 0."""
    scan = ocr_scan(v)
    ref = list(scan[0]) + list(real_texts)
    hits = {k: new for k, ts in scan.items() if k and (new := [t for t in ts if not known(t, ref)])}
    return {"frame0": scan[0], "new": hits, "verdict": "fail" if len(hits) >= 2 else "suspect" if hits else "ok"}


def real_ref(job, scores):
    key = f"_ref|{job['tag']}"
    if key not in scores:
        rv = native(job["real"])
        scan = ocr_scan(rv)
        scores[key] = {"dup_frames": dup_frames(job["real"]),
                       "zoom": zoom(job["real"]) if job["camera"] == "static" else None,
                       "ocr_texts": sorted({t for ts in scan.values() for t in ts}), "ocr_scan": scan, "frames": len(rv)}
    return scores[key]


def score_clip(p, job, scores):
    ref = real_ref(job, scores)
    v = native(p)
    out = {"frames": len(v), "dup_frames": dup_frames(p), "dup_real": ref["dup_frames"]}
    out["dup_flag"] = out["dup_frames"] > out["dup_real"]
    out.update(qa(frames(p, 180, 320), frames(job["real"], 180, 320), job["start"]))
    out["zoom"] = zoom(p) if job["camera"] == "static" else None
    out.update(identity(v))
    out["ocr"] = ocr(v, ref["ocr_texts"])
    return out


# ---------------------------------------------------------------------------------------------
def gen(config, only=None, seeds=C.SEEDS, flf=True, flf_only=False, limit=0, max_new=None):
    """-> clips still to generate for this call (0 = complete)."""
    jobs = load_jobs(only)
    if limit:
        jobs = jobs[:limit]
    _, remaining = C.generate(config, expand(jobs, seeds, flf, flf_only), cfg_out(config), size=C.SCENE_SIZE,
                              e5=True, max_new=max_new)
    return remaining


def score(configs, only=None, seeds=C.SEEDS, flf=True, limit=0, rescore=False):
    with C.locked(ROOT / "scores.json"):
        return _score(configs, only, seeds, flf, limit, rescore)


def _score(configs, only, seeds, flf, limit, rescore):
    jobs = {j["tag"]: j for j in load_jobs(only)[:limit or None]}
    dest = ROOT / "scores.json"
    scores = C.load_json(dest, {})
    n = 0
    for cfg in configs:
        for x in expand(list(jobs.values()), seeds, flf):
            key = f"{cfg}|{x['tag']}"
            p = cfg_out(cfg) / f"{x['tag']}.mp4"
            if not p.exists() or (key in scores and not rescore):
                continue
            j = jobs[x["base"]]
            scores[key] = score_clip(p, j, scores) | {"config": cfg, "job": j["tag"], "seed": x["seed"], "flf": x["flf"],
                                                     "camera": j["camera"], "two_shot": j["two_shot"],
                                                     "phone": j["phone"], "source": j["source"]}
            n += 1
            if n % 10 == 0:
                C.write_json(dest, scores)
                print(f"   scored {n} clips", flush=True)
    C.write_json(dest, scores)
    print(f"scored {n} new clips -> {dest}")
    return scores


def sheet(rows, dest, n=7):
    tw, th = 120, 213
    img = Image.new("RGB", (tw * n + 110, th * len(rows)), "white")
    dr = ImageDraw.Draw(img)
    for r, (lab, p) in enumerate(rows):
        dr.text((2, r * th + 4), lab[:17], fill="black")
        if not Path(p).exists():
            dr.text((112, r * th + 100), "(missing)", fill="red"); continue
        v = frames(p, tw, th)
        for i, k in enumerate(np.linspace(0, len(v) - 1, n).round().astype(int)):
            img.paste(Image.fromarray(v[k]), (110 + i * tw, r * th))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def sheets(configs, only=None, seed=42):
    out = ROOT / "sheets"
    for j in load_jobs(only):
        sheet([("real", j["real"])] + [(c, cfg_out(c) / f"{j['tag']}_s{seed}.mp4") for c in configs],
              out / f"{j['tag']}.png")
        if j.get("flf"):
            sheet([("real", j["real"])] + [(f"{c} FLF", cfg_out(c) / f"{j['tag']}_flf_s42.mp4") for c in configs],
                  out / f"{j['tag']}_flf.png")
    print(f"sheets -> {out}")


def gate(cand, base="v8merged", only=None, seeds=C.SEEDS, limit=0):
    scores = C.load_json(ROOT / "scores.json", {})
    jobs = load_jobs(only)[:limit or None]
    want = [(j, s) for s in seeds for j in jobs]
    rc = {(j["tag"], s): scores.get(f"{cand}|{j['tag']}_s{s}") for j, s in want}
    rb = {(j["tag"], s): scores.get(f"{base}|{j['tag']}_s{s}") for j, s in want}
    miss = [f"{cand}|{k[0]}_s{k[1]}" for k, v in rc.items() if v is None] + \
           [f"{base}|{k[0]}_s{k[1]}" for k, v in rb.items() if v is None]
    rep = {"cand": cand, "base": base, "n_expected": len(want), "missing": miss}
    if miss:
        rep["status"] = "INCOMPLETE"
        C.write_json(ROOT / f"g2_{C.cfg_dir(cand)}.json", rep)
        print(f"G2 INCOMPLETE: {len(miss)} clips not generated or not scored (e.g. {miss[:3]})")
        return EXIT_MISSING
    c = list(rc.items())
    dup = [f"{k[0]}_s{k[1]}" for k, r in c if r["dup_flag"]]
    refz = {j["tag"]: (scores.get(f"_ref|{j['tag']}") or {}).get("zoom") for j in jobs}
    unreliable = {t for t, z in refz.items() if z and z.get("zoom_pct") is not None and z["zoom_pct"] > ZOOM_REAL_MAX}
    zc = [(k, r) for k, r in c if r["zoom"] and k[0] not in unreliable]
    zf = [(f"{k[0]}_s{k[1]}", r["zoom"]["zoom_pct"]) for k, r in zc if r["zoom"]["verdict"] == "zoom"]
    zu = [f"{k[0]}_s{k[1]}" for k, r in zc if r["zoom"]["verdict"] == "unverifiable"]
    nz = len(zc)
    mot = float(np.median([r["motion"] for _, r in c]))
    pairs = [(rc[k]["identity"], rb[k]["identity"]) for k in rc if rc[k]["identity"] is not None and rb[k]["identity"] is not None]
    idc, idb = float(np.mean([a for a, _ in pairs])), float(np.mean([b for _, b in pairs]))
    dm, dlo, dhi = C.ci95([a - b for a, b in pairs])
    of = [(f"{k[0]}_s{k[1]}", r["ocr"]["new"]) for k, r in c if r["ocr"]["verdict"] == "fail"]
    osus = [(f"{k[0]}_s{k[1]}", r["ocr"]["new"]) for k, r in c if r["ocr"]["verdict"] == "suspect"]
    checks = {
        "dup": {"flagged": dup, "n_flagged": len(dup), "base_n_flagged": sum(r["dup_flag"] for r in rb.values()),
                "pass": not dup, "bar": "no clip with more duplicate-lead frames than its real shot (flagged -> eye check)"},
        "zoom": {"n_static": nz, "over": zf, "unverifiable": len(zu), "unverifiable_clips": zu, "pass": not zf,
                 "jobs_unreliable": {t: refz[t]["zoom_pct"] for t in sorted(unreliable)},
                 "base_over": sum(1 for k, r in rb.items() if r["zoom"] and k[0] not in unreliable and r["zoom"]["verdict"] == "zoom"),
                 "bar": f"zoom <= {ZOOM_MAX}% on static-camera jobs (jobs whose locked real shot reads > {ZOOM_REAL_MAX}% "
                        f"are reported as unreliable, not gated)"},
        "motion": {"median": round(mot, 3), "base_median": round(float(np.median([r["motion"] for r in rb.values()])), 3),
                   "share_outside": round(float(np.mean([not MOTION_LO <= r["motion"] <= MOTION_HI for _, r in c])), 3),
                   "pass": MOTION_LO <= mot <= MOTION_HI, "bar": f"median motion ratio in [{MOTION_LO}, {MOTION_HI}]"},
        "identity": {"n_pairs": len(pairs), "cand": round(idc, 4), "base": round(idb, 4),
                     "paired_diff": round(dm, 4), "ci": [round(dlo, 4), round(dhi, 4)],
                     "not_measurably_worse": dhi >= 0, "pass": idc >= idb,
                     "bar": "mean DINO lead-crop similarity to frame 0 >= v8merged (literal)"},
        "ocr": {"fail": of, "suspect": osus, "pass": not of, "base_fail": sum(r["ocr"]["verdict"] == "fail" for r in rb.values()),
                "bar": "no text absent from frame 0 and from the real shot (in >= 2 sampled frames = flagged for an eye "
                       "check; in 1 = suspect, listed)"},
    }
    hard = all(checks[k]["pass"] for k in ("zoom", "motion", "identity"))
    flagged = sorted(set(dup + [x for x, _ in of]))
    status = "FAIL" if not hard else "REVIEW" if flagged else "PASS"
    rep.update(status=status, checks=checks, eye_check=flagged, ocr_suspects=[x for x, _ in osus],
               note="REVIEW = zoom, motion and identity hold, but clips flagged by dupcheck or OCR need an eye check "
                    "(Q14; OCR reads cartoon shapes as text) before G2 counts as passed")
    C.write_json(ROOT / f"g2_{C.cfg_dir(cand)}.json", rep)
    for k, v in checks.items():
        print(f"  {k:<9} {'PASS' if v['pass'] else 'FAIL'}  " + json.dumps({x: y for x, y in v.items()
              if x not in ('bar', 'pass', 'unverifiable_clips', 'flagged', 'fail', 'suspect')})[:220])
    if dup:
        print(f"  eye check (dup flagged): {dup}")
    if osus:
        print(f"  OCR suspects: {osus[:6]}")
    print(f"G2 {cand} vs {base}: {status}  -> {ROOT / f'g2_{C.cfg_dir(cand)}.json'}")
    return {"PASS": EXIT_PASS, "REVIEW": EXIT_REVIEW, "FAIL": EXIT_FAIL}[status]


def selftest(limit=0, only=None):
    """The real shots scored as config 'real' (seed 42 only) and gated against themselves."""
    d = cfg_out("real"); d.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(only)[:limit or None]
    for j in jobs:
        dst = d / f"{j['tag']}_s42.mp4"
        if not dst.exists():
            os.symlink(j["real"], dst)
    score(["real"], only, (42,), flf=False, limit=limit)
    return gate("real", "real", only, (42,), limit)


def main():
    ap = argparse.ArgumentParser(description="v10 G1/G2 suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--config", required=True)
    s = sub.add_parser("score"); s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--rescore", action="store_true")
    h = sub.add_parser("sheets"); h.add_argument("--configs", nargs="+", required=True)
    t = sub.add_parser("gate"); t.add_argument("--cand", required=True); t.add_argument("--base", default="v8merged")
    al = sub.add_parser("all"); al.add_argument("--config", required=True); al.add_argument("--base", default="v8merged")
    st = sub.add_parser("selftest")
    for p in (g, s, h, t, al, st):
        p.add_argument("--only", choices=("v10", "v9b"))
        p.add_argument("--limit", type=int, default=0)
    for p in (g, s, t, al):
        p.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    for p in (g, s, al):
        p.add_argument("--no-flf", action="store_true")
    for p in (g, al):
        p.add_argument("--flf-only", action="store_true")
        p.add_argument("--max-new", type=int, help="generate at most N new clips, then exit 0 with 'REMAINING <k>'")
    a = ap.parse_args()
    if a.cmd == "gen":
        print(f"REMAINING {gen(a.config, a.only, a.seeds, not a.no_flf, a.flf_only, a.limit, a.max_new)}")
        return 0
    if a.cmd == "score":
        score(a.configs, a.only, a.seeds, not a.no_flf, a.limit, a.rescore); return 0
    if a.cmd == "sheets":
        sheets(a.configs, a.only); return 0
    if a.cmd == "gate":
        return gate(a.cand, a.base, a.only, a.seeds, a.limit)
    if a.cmd == "selftest":
        return selftest(a.limit, a.only)
    if a.cmd == "all":
        left = gen(a.config, a.only, a.seeds, not a.no_flf, a.flf_only, a.limit, a.max_new)
        if left:
            print(f"REMAINING {left}")
            return 0
        score([a.base, a.config], a.only, a.seeds, not a.no_flf, a.limit)
        sheets([a.base, a.config], a.only)
        return gate(a.config, a.base, a.only, a.seeds, a.limit)


if __name__ == "__main__":
    sys.exit(main())
