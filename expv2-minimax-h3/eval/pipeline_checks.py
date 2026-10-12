#!/usr/bin/env python3
"""G0.2 a-h - pipeline checks (PLAN section 4). One GPU session, no training beyond
single steps. Each check writes into phase0/pipeline_checks/report.json (merged, so checks
can run in separate invocations) and exits 2 if any HARD check fails.

  a  parity: pruned vs full DiT velocity at C.PARITY_SIGMAS, one holdout clip, cached
     embeddings, fixed noise                                   HARD  rel L2 <= 1% at every sigma
  b  conditioning dump: one TRAINING step (DiffSynth stage-1 cache + MiniMaxH3TrainingModule)
     vs one SAMPLING step (the generation pass-1 path) on the same clip and caption:
     keyframe rows, their timestep and RoPE time, vision tokens in Qwen3-VL's input,
     end-frame rows, latents in the loss, the frame -> latent map  HARD  identical layouts,
     keyframe in both paths, no end-frame rows, frame 0 -> latent 0 only
  c  guidance-loss branch at --training_cfg_scale 4: what the no-grad unconditional pass
     keeps and drops                                            RECORD
  d  audio: silent fallback active; with --audio_loss_weight 0 the loss is the video term
                                                                HARD  audio contribution == 0
  e  loader: DiffSynth's own video operator on the corpus mp4s vs prep's PNGs (CPU)
                                                                HARD  N frames, bit-identical
  f  mixed lengths: f22 + f39 + f56 stage-1 caches in ONE stage-2 dataset, 20 samples
                                                                HARD  no error, mixed shapes.
     If it fails, PLAN says: drop the 39/56 rungs and train f22 only (pre-declared)
  g  two-pass vs one-pass generation on 2 core items          HARD  PSNR >= 50 dB every frame
  h  trainer flags / code: our patches applied, LR scheduler, save and resume semantics,
     the train-shift patch moves the sigma schedule (CPU)      HARD  patch flags present,
                                                                      DiffSynth at the pinned SHA

    python eval/pipeline_checks.py --dry
    python eval/pipeline_checks.py --checks h,e          # CPU-only, run first
    python eval/pipeline_checks.py                       # everything (GPU)
    python eval/pipeline_checks.py --checks g --g-steps 50

GPU order (models never co-resident beyond what DiffSynth's offload config allows):
  tiny stage-1 caches (b,c,d,f) -> Qwen3-VL + VAEs (a,b,g encodes; b latent map) ->
  training module DiT (b,c,d,f) -> pruned pipe (a,g) -> full DiT (a) -> one-pass pipe (g).
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3lib as L  # noqa: E402
import generate as G  # noqa: E402
C = L.C

OUT = C.PHASE0 / "pipeline_checks"
REPORT = OUT / "report.json"
ALL = "abcdefgh"
PATCH_FLAGS = ("--train_flow_shift", "--train_audio_flow_shift", "--seed",
               "--lr_scheduler", "--lr_warmup_steps", "--lr_min_ratio")
STOCK_FLAGS = ("--training_cfg_scale", "--audio_loss_weight", "--silent_on_missing_audio",
               "--save_steps", "--gradient_accumulation_steps", "--resume_from_checkpoint",
               "--preset_lora_path", "--extra_inputs", "--dataset_repeat")


def result(status, hard, **details):
    return {"status": status, "hard": hard, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **details}


def train_rows(F):
    p = C.CORPUS / f"metadata_f{F}.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def clip_id_of(row):
    return Path(row["video"]).stem


# ---------------------------------------------------------------------------------------
# h - flags and code (CPU)
# ---------------------------------------------------------------------------------------

def check_h():
    tp = L.train_py()
    help_txt = subprocess.run([sys.executable, str(tp), "--help"], cwd=str(C.DIFFSYNTH),
                              capture_output=True, text=True)
    txt = help_txt.stdout + help_txt.stderr
    missing_patch = [f for f in PATCH_FLAGS if f not in txt]
    missing_stock = [f for f in STOCK_FLAGS if f not in txt]
    src = {name: (C.DIFFSYNTH / rel).read_text() if (C.DIFFSYNTH / rel).exists() else ""
           for name, rel in (("runner", "diffsynth/diffusion/runner.py"),
                             ("logger", "diffsynth/diffusion/logger.py"),
                             ("train", L.TRAIN_PY),
                             ("tm", "diffsynth/diffusion/training_module.py"))}
    head = L.diffsynth_sha()
    diff = subprocess.run(["git", "-C", str(C.DIFFSYNTH), "diff", "--stat"], capture_output=True, text=True).stdout
    sched = {}
    try:
        from diffsynth.diffusion import FlowMatchScheduler
        for shift in (2.22, 3.0, 12.0):
            s = FlowMatchScheduler("MiniMax-H3")
            s.set_timesteps(1000, training=True, shift=shift)
            sig = s.sigmas.float()
            sched[str(shift)] = {q: round(float(sig.quantile(v)), 4) for q, v in (("q10", .1), ("median", .5), ("q90", .9))}
    except Exception as e:
        sched["error"] = f"{type(e).__name__}: {e}"
    logged = {}
    for run in L.RUN_NAMES:
        lp = C.RUNS / run / "train.log"
        if lp.exists():
            logged[run] = [ln.strip() for ln in lp.read_text(errors="ignore").splitlines()
                           if "[expv2]" in ln][:6]
    rec = {
        "diffsynth_head": head, "pinned": C.DIFFSYNTH_SHA, "at_pinned_sha": head == C.DIFFSYNTH_SHA,
        "working_tree_diff": diff.strip().splitlines()[-12:],
        "help_rc": help_txt.returncode, "patch_flags_missing": missing_patch, "stock_flags_missing": missing_stock,
        "has_max_steps_flag": "--max_steps" in txt or "--max_train_steps" in txt,
        "scheduler": {"stock_ConstantLR": "ConstantLR" in src["runner"],
                      "patched_build_lr_scheduler": "build_lr_scheduler" in src["runner"],
                      "note": "stock DiffSynth: torch ConstantLR (factor 1/3 for 5 steps) - no warmup/cosine"},
        "save_semantics": {"counts_samples": "self.num_steps += 1" in src["logger"],
                           "note": "ModelLogger.on_step_end runs once per dataloader item: step-N = N SAMPLES; "
                                   "opt step = N / grad_accum (configs/train.sh saves every 400 = 50 opt steps)"},
        "resume_semantics": {"weights_only": "def resume_from_checkpoint" in src["tm"] and "optimizer" not in
                             src["tm"].split("def resume_from_checkpoint", 1)[-1][:900],
                             "note": "--resume_from_checkpoint loads module weights only; no optimizer/LR/step state"},
        "train_shift_patch_applied": "args.train_flow_shift" in src["train"],
        "sigma_schedules": sched,
        "logged_sigmas": logged or "no run logs yet - re-run --checks h after the pilots to record them",
    }
    ok = help_txt.returncode == 0 and not missing_patch and not missing_stock and rec["at_pinned_sha"] \
        and rec["train_shift_patch_applied"] and rec["scheduler"]["patched_build_lr_scheduler"]
    return result("PASS" if ok else "FAIL", True, **rec)


# ---------------------------------------------------------------------------------------
# e - loader (CPU)
# ---------------------------------------------------------------------------------------

def check_e(all_clips=False, n_f22=40):
    import hashlib
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import LoadAudioWithTorchaudio
    picks = []
    for F in C.BUCKETS:
        rows = train_rows(F)
        if not all_clips and F == 22:
            random.Random(0).shuffle(rows)
            rows = rows[:n_f22]
        picks += [(F, r) for r in rows]
    bad, details = [], []
    for F, r in picks:
        op = UnifiedDataset.default_video_operator(       # exactly train.py's operator
            base_path=str(C.CORPUS), max_pixels=1024 * 1024, height=C.SIZE, width=C.SIZE,
            height_division_factor=32, width_division_factor=32, num_frames=F,
            time_division_factor=17, time_division_remainder=5, frame_rate=24, fix_frame_rate=True)
        frames = op(r["video"])
        png = L.read_frames(C.CORPUS / "png" / clip_id_of(r))
        got = [hashlib.md5(np.asarray(f).tobytes()).hexdigest() for f in frames]
        want = [hashlib.md5(np.asarray(f).tobytes()).hexdigest() for f in png]
        same = len(got) == F and got == want[:F] and len(png) == F
        audio = LoadAudioWithTorchaudio(num_frames=F, time_division_factor=17, time_division_remainder=5,
                                        frame_rate=24, fix_frame_rate=True)(str(C.CORPUS / r["input_audio"]))
        d = {"clip": clip_id_of(r), "F": F, "loaded": len(frames), "png": len(png), "bit_identical": same,
             "audio_none": audio is None}
        if not same or audio is not None:
            bad.append(d)
        details.append(d)
    meta = {}
    try:
        import imageio
        for F in C.BUCKETS:
            rows = train_rows(F)
            if rows:
                rd = imageio.get_reader(str(C.CORPUS / rows[0]["video"]))
                md = rd.get_meta_data()
                meta[f"f{F}"] = {"fps": md.get("fps"), "duration": md.get("duration"),
                                 "count_frames": rd.count_frames(),
                                 "floor_duration_x24": int(np.floor((md.get("duration") or 0) * 24))}
                rd.close()
    except Exception as e:
        meta["error"] = str(e)
    return result("PASS" if picks and not bad else "FAIL", True, checked=len(picks), failures=bad[:20],
                  imageio_meta=meta, note="mp4s carry N+1 frames (LOADER_TAIL); the loader must return exactly N")


# ---------------------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------------------

def capture_dit(dit):
    """Forward pre-hook recording what the DiT actually receives (model_fn passes every
    argument by keyword, pipelines/minimax_h3_audio_video.py:1042-1062)."""
    rec = []

    def hook(mod, args, kwargs):
        ut = kwargs["unique_timesteps"].float().cpu()
        inv = kwargs["inverse_indices"].view(-1).long().cpu()
        rec.append({"row_t": ut[inv], "token_tags": kwargs["token_tags"].view(-1).cpu(),
                    "img_pos": kwargs["img_pos_info"]["position_ids"].view(-1).cpu(),
                    "pos_ids": kwargs["img_position_ids"][0].float().cpu(),
                    "seq_len": int(kwargs["x"].shape[1]),
                    "text_len": int(kwargs["text_pos_info"]["position_ids"].numel())})
    return rec, dit.register_forward_pre_hook(hook, with_kwargs=True)


def layout(cap, n_latents, latent_hw=C.SIZE // 16):
    frame_rows = (latent_hw // 2) ** 2
    img_pos = cap["img_pos"]
    cond = len(img_pos) - n_latents * frame_rows
    cond_rows = img_pos[:cond]
    vid_rows = img_pos[cond:]
    tl = cap["text_len"]
    return {
        "seq_len": cap["seq_len"], "text_len": tl, "frame_rows": frame_rows,
        "cond_rows": int(cond), "cond_frames": cond / frame_rows,
        "cond_row_t": sorted({round(float(v), 5) for v in cap["row_t"][cond_rows]}),
        "video_row_t": sorted({round(float(v), 5) for v in cap["row_t"][vid_rows]}),
        "cond_rope_t": sorted({round(float(v), 4) for v in cap["pos_ids"][cond_rows, 0]}),
        "first_video_rope_t": round(float(cap["pos_ids"][vid_rows[0], 0]), 4) if len(vid_rows) else None,
        "cond_rope_t_equals_text_len": bool(cond and all(abs(float(v) - tl) < 1e-6 for v in cap["pos_ids"][cond_rows, 0])),
        "token_tag_counts": {int(k): int(v) for k, v in zip(*np.unique(cap["token_tags"].numpy(), return_counts=True))},
    }


def vision_tokens(cap):
    """Vision (image-pad + start/end) tokens in Qwen3-VL's input, read from the text slice
    of the packed token tags the DiT receives (PackedSequenceBuilder copies text_token_tags
    there, minimax_h3_audio_video.py:836; presentation tags, minimax_h3_text_encoder.py)."""
    from diffsynth.models.minimax_h3_text_encoder import PRESENTATION_VIDEO_TAG
    return int((cap["token_tags"][:cap["text_len"]] == PRESENTATION_VIDEO_TAG).sum())


def latent_map(pipe, frames):
    """Empirical frame -> latent map from the H3 VAE: perturb frame f, re-encode, record the
    latents that change. A causal encoder also moves later latents of the same chunk, so
    the FIRST changed latent is the one covering frame f."""
    import torch
    with torch.no_grad():
        pipe.load_models_to_device(["video_vae"])

        def enc(fr):
            x = pipe.preprocess_video(fr, torch_dtype=torch.float32, min_value=0, device=pipe.device)
            return pipe.video_vae.encode_video(x, dtype=pipe.torch_dtype).float()
        z0 = enc(frames)
        out = {}
        for f in range(len(frames)):
            fr = list(frames)
            a = np.asarray(fr[f]).copy()
            h, w = a.shape[:2]
            a[h // 3:h // 3 + 96, w // 3:w // 3 + 96] = 255 - a[h // 3:h // 3 + 96, w // 3:w // 3 + 96]
            from PIL import Image
            fr[f] = Image.fromarray(a)
            dz = (enc(fr) - z0).abs().amax(dim=(0, 1, 3, 4))
            changed = [int(i) for i in torch.nonzero(dz > 1e-3).view(-1)]
            out[f] = {"first": changed[0] if changed else None, "changed": changed}
    return out, list(z0.shape)


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", default=ALL)
    ap.add_argument("--clip", default="", help="training clip id for b/c/d (default: first f22 row)")
    ap.add_argument("--holdout-clip", default="", help="holdout clip id for a (default: first holdout)")
    ap.add_argument("--g-items", default="", help="core item ids for g (default: 1 expression + 1 motion QF2_R)")
    ap.add_argument("--g-steps", type=int, default=C.INFERENCE["num_inference_steps"])
    ap.add_argument("--all-loader", action="store_true", help="e: every training clip, not a sample")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    checks = [c for c in args.checks.replace(",", "") if c in ALL]

    rows22 = train_rows(22)
    row_b = next((r for r in rows22 if clip_id_of(r) == args.clip), rows22[0] if rows22 else None)
    hold = C.load_json(C.CORPUS / "tier1_items.json") or []
    hold = [h for h in hold if h["split"] == "holdout"]
    item_a = next((h for h in hold if h["clip_id"] == args.holdout_clip), hold[0] if hold else None)
    core = [it for it in (C.load_json(C.CORPUS / "eval_items.json") or []) if it["suite"] == "core"]
    if args.g_items:
        items_g = [it for it in core if it["item_id"] in args.g_items.split(",")]
    else:
        items_g = [next((it for it in core if it["angle"] == C.HOLDOUT_ANGLE and it["kind"] == k), None)
                   for k in ("expression", "motion")]
        items_g = [it for it in items_g if it]

    if args.dry:
        probs = L.check_dry_inputs([("DiffSynth checkout", C.DIFFSYNTH), ("train.py", L.train_py()),
                                    ("tier1_items.json", C.CORPUS / "tier1_items.json"),
                                    ("eval_items.json", C.CORPUS / "eval_items.json")])
        print(f"checks: {checks}")
        print(f"  b/c/d clip : {clip_id_of(row_b) if row_b else None}")
        print(f"  a clip     : {item_a['clip_id'] if item_a else None}")
        print(f"  g items    : {[it['item_id'] for it in items_g]}")
        for F in C.BUCKETS:
            print(f"  metadata_f{F}.csv rows: {len(train_rows(F))}")
        plan = L.tiny_cache_plan()
        print(f"  tiny caches: {[f'f{F}' for F in plan]} -> {L.PROBE}")
        if any(c in checks for c in "bcd") and not row_b:
            probs.append("no metadata_f22.csv row for b/c/d")
        if "a" in checks and not item_a:
            probs.append("no holdout clip for a")
        if "g" in checks and len(items_g) < 2:
            probs.append("g needs 2 core items")
        if "f" in checks and len(plan) < 3:
            print("  (f: fewer than 3 buckets present - f tests what exists)")
        for x in ("dit_pruned", "dit_full"):
            spec = C.DIT_PRUNED if x == "dit_pruned" else C.DIT_FULL
            repo, pat = spec.rsplit(":", 1)
            p = C.MODELS / repo / pat
            print(f"  {x}: {p} {'present' if p.exists() else '[not on this machine]'}")
        print("DRY: problems:\n  - " + "\n  - ".join(probs) if probs else "DRY: inputs OK")
        return 1 if probs else 0

    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    hdr = L.report_header("G0.2 pipeline checks")

    def save():
        L.merge_report(REPORT, {**hdr, "gpu": L.gpu_info() if any(c in checks for c in "abcdfg") else None,
                                "checks": res})

    if "h" in checks:
        res["h"] = check_h()
        save()
    if "e" in checks:
        res["e"] = check_e(args.all_loader)
        save()
    gpu = [c for c in checks if c in "abcdfg"]
    if not gpu:
        return finish(res)

    import torch
    log = {}
    caches = L.build_tiny_caches(log=log) if any(c in checks for c in "bcdf") else {}
    clip_b = clip_id_of(row_b) if row_b else None

    # ---- pass: Qwen3-VL + VAEs -----------------------------------------------------------
    enc_a = enc_s = lmap = None
    enc_g = {}
    if any(c in checks for c in "abg"):
        pipe = L.load_pipe(("te", "vae", "audio_vae"))
        if "a" in checks and item_a:
            fr = L.read_frames(C.CORPUS / item_a["png_dir"], n=item_a["frames"])
            enc_a = L.encode_training_path(pipe, fr, {item_a["label"]: item_a["candidates"][item_a["label"]]})
        if "b" in checks and row_b:
            fr = L.read_frames(C.CORPUS / "png" / clip_b, n=22)
            enc_s = L.encode_generation(pipe, row_b["prompt"], fr[0], 22, 42)
            lmap, zshape = latent_map(pipe, fr)
            log["latent_map_shape"] = zshape
        if "g" in checks:
            memo = {}
            for it in items_g:
                enc_g[it["item_id"]] = L.encode_generation(pipe, it["caption"], G.start_image(it), it["frames"], 42,
                                                           memo=memo, memo_key=it["caption"])
        pipe = None
        L.free()

    # ---- pass: training module (DiT + LoRA, guidance loss) -----------------------------
    if any(c in checks for c in "bcdf"):
        m = L.load_train_module(cfg_scale=4.0, audio_loss_weight=0.0)
        pipe = m.pipe
        data = L.load_cache_item(caches[22][0]) if 22 in caches else None
        if data is None:
            for c in "bcd":
                if c in checks:
                    res[c] = result("ERROR", True, error="no f22 probe cache")
        if "b" in checks and data is not None:
            res["b"] = check_b(m, pipe, data, enc_s, lmap, row_b)
            save()
        if "c" in checks and data is not None:
            res["c"] = check_c(m, pipe, data)
            save()
        if "d" in checks and data is not None:
            res["d"] = check_d(m, pipe, data)
            save()
        if "f" in checks:
            res["f"] = check_f(m, caches)
            save()
        m = pipe = data = None
        L.free()

    # ---- pass: pruned pipeline (a, g two-pass) -----------------------------------------
    vel_p, frames_two = {}, {}
    if ("a" in checks and enc_a is not None) or "g" in checks:
        pipe = L.load_pipe(("dit", "vae"))
        L.onload(pipe, ["dit"])
        if "a" in checks and enc_a is not None:
            vel_p = parity_velocities(pipe, enc_a, item_a)
        if "g" in checks:
            for it in items_g:
                frames_two[it["item_id"]] = L.denoise(pipe, enc_g[it["item_id"]], steps=args.g_steps)
        pipe = None
        L.free()

    # ---- pass: full DiT (a) ---------------------------------------------------------------
    if "a" in checks and enc_a is not None:
        pipe = L.load_pipe(("dit_full",))
        L.onload(pipe, ["dit"])
        vel_f = parity_velocities(pipe, enc_a, item_a)
        pipe = None
        L.free()
        rel = {s: float((vel_p[s] - vel_f[s]).norm() / vel_f[s].norm()) for s in vel_p}
        ok = all(v <= 0.01 for v in rel.values())
        res["a"] = result("PASS" if ok else "FAIL", True, clip=item_a["clip_id"], rel_l2=rel, bar=0.01,
                          note=("FAIL -> stop: answer the build question first (MODEL_NOTES 2.6). "
                                "Delete the full DiT afterwards (PLAN 3)."))
        save()

    # ---- pass: one-pass pipeline (g) ------------------------------------------------------
    if "g" in checks:
        pipe = L.load_pipe(("te", "dit", "vae", "audio_vae"))
        I = C.INFERENCE
        details = {}
        ok = True
        for it in items_g:
            with torch.no_grad():
                one, _ = pipe(prompt=it["caption"], negative_prompt=" ", height=I["height"], width=I["width"],
                              num_frames=it["frames"], num_inference_steps=args.g_steps, seed=42,
                              rand_device=I["rand_device"], cfg_scale=I["cfg_scale"], flow_shift=I["flow_shift"],
                              audio_flow_shift=I["audio_flow_shift"], tiled=I["tiled"], tile_size=I["tile_size"],
                              tile_overlap=I["tile_overlap"], keyframes=[G.start_image(it)], keyframe_indices=[0])
            two = frames_two[it["item_id"]]
            a, b = L.frames_to_u8(two), L.frames_to_u8(one)
            ps = [L.psnr_u8(a[i], b[i]) for i in range(min(len(a), len(b)))]
            okk = len(a) == len(b) == it["frames"] and min(ps) >= 50.0
            ok &= okk
            details[it["item_id"]] = {"frames": [len(a), len(b)], "psnr_min": round(min(ps), 2),
                                      "psnr_mean": round(float(np.mean(ps)), 2), "pass": okk}
            L.save_frames(two, OUT / "dumps" / "g" / it["item_id"] / "two_pass")
            L.save_frames(one, OUT / "dumps" / "g" / it["item_id"] / "one_pass")
        pipe = None
        L.free()
        res["g"] = result("PASS" if ok else "FAIL", True, steps=args.g_steps, bar_psnr=50.0, items=details)
        save()

    res.setdefault("_log", log)
    save()
    return finish(res)


def parity_velocities(pipe, enc, item):
    out = {}
    lab = item["label"]
    for s in C.PARITY_SIGMAS:
        nv, na = L.fixed_noise(enc, "parity", item["clip_id"], f"{s:.6f}")
        pred, _ = L.velocity_at(pipe, enc, lab, s, nv, na)
        out[s] = pred.cpu()
    return out


def check_b(m, pipe, data, enc_s, lmap, row):
    """One training step vs one sampling step, captured at the DiT's input."""
    import torch
    from diffsynth.diffusion import FlowMatchScheduler
    shared_t, posi_t, nega_t = data
    n_lat = int(shared_t["input_latents"].shape[2])
    rec, h = capture_dit(pipe.dit)
    outs = {}
    orig_fn = pipe.model_fn

    def wrapped(*a, **k):
        r = orig_fn(*a, **k)
        outs.setdefault("noise_pred_shapes", []).append([list(x.shape) for x in r])
        return r
    pipe.model_fn = wrapped
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            m({}, inputs=data)
        train_caps = list(rec)            # [uncond, cond] at cfg 4
        rec.clear()
        sv, sa = FlowMatchScheduler("MiniMax-H3"), FlowMatchScheduler("MiniMax-H3")
        sv.set_timesteps(C.INFERENCE["num_inference_steps"], shift=C.INFERENCE["flow_shift"])
        sa.set_timesteps(C.INFERENCE["num_inference_steps"], shift=C.INFERENCE["audio_flow_shift"])
        shared = L.to_device(enc_s["shared"], pipe.device)
        posi = L.to_device(enc_s["posi"], pipe.device)
        with torch.no_grad():
            pipe.model_fn(dit=pipe.dit, **posi, **shared,
                          timestep_video=sv.timesteps[0].unsqueeze(0).float().to(pipe.device),
                          timestep_audio=sa.timesteps[0].unsqueeze(0).float().to(pipe.device))
        samp_caps = list(rec)
    finally:
        h.remove()
        pipe.model_fn = orig_fn
    lt = layout(train_caps[-1], n_lat)
    ls = layout(samp_caps[-1], n_lat)
    same_pos = bool(torch.equal(train_caps[-1]["pos_ids"], samp_caps[-1]["pos_ids"]))
    same_tags = bool(torch.equal(train_caps[-1]["token_tags"], samp_caps[-1]["token_tags"]))
    a_t = shared_t["keyframe_cond_anchor"].float()
    a_s = enc_s["shared"]["keyframe_cond_anchor"].float()
    anchor_rel = float((a_t - a_s).norm() / a_t.norm())
    vis_t = vision_tokens(train_caps[-1])
    vis_s = vision_tokens(samp_caps[-1])
    frame0 = [f for f, v in (lmap or {}).items() if v["first"] == 0]
    lat0_frames = sorted(f for f, v in (lmap or {}).items() if v["first"] == 0)
    rec_map = {f: v["first"] for f, v in (lmap or {}).items()}
    expected = {f: next(i for i, fs in enumerate(L.latent_frames(n_lat)) if f in fs) for f in range(22)}
    ok = (same_pos and same_tags and lt["cond_frames"] == 1 and ls["cond_frames"] == 1
          and lt["cond_rope_t_equals_text_len"] and ls["cond_rope_t_equals_text_len"]
          and vis_t > 0 and vis_s > 0 and lat0_frames == [0] and anchor_rel < 1e-2)
    return result("PASS" if ok else "FAIL", True, clip=Path(row["video"]).stem,
                  training=lt, sampling=ls, identical_rope_grid=same_pos, identical_token_tags=same_tags,
                  dit_calls_training_step=len(train_caps),
                  keyframe_anchor_rel_diff=anchor_rel,
                  vision_tokens={"training": vis_t, "sampling": vis_s},
                  end_frame_rows=bool(lt["cond_frames"] > 1 or ls["cond_frames"] > 1),
                  loss_covers=("every video latent incl. latent 0: FlowMatchSFTMiniMaxH3AudioVideoLoss takes "
                               "mse over the full noise_pred (diffusion/loss.py:112)"),
                  model_fn_output_shapes=outs.get("noise_pred_shapes"),
                  frame_to_latent_measured=rec_map, frame_to_latent_expected=expected,
                  frame0_latent_index=[0] if lat0_frames == [0] else lat0_frames,
                  frames_first_in_latent0=frame0,
                  cache_keys={"shared": sorted(shared_t), "posi": sorted(posi_t), "nega": sorted(nega_t)})


def check_c(m, pipe, data):
    import torch
    shared_t, posi_t, nega_t = data
    rec, h = capture_dit(pipe.dit)
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            m({}, inputs=data)
    finally:
        h.remove()
    n_lat = int(shared_t["input_latents"].shape[2])
    calls = [layout(c, n_lat) for c in rec]
    rep = {
        "dit_calls_per_step": len(rec),
        "uncond_call": calls[0] if len(calls) > 1 else None, "cond_call": calls[-1] if calls else None,
        "nega_keys": sorted(nega_t), "posi_keys": sorted(posi_t),
        "text_len": {"cond": rec[-1]["text_len"] if rec else None,
                     "uncond": rec[0]["text_len"] if len(rec) > 1 else None},
        "vision_tokens": {"cond": vision_tokens(rec[-1]) if rec else None,
                          "uncond": vision_tokens(rec[0]) if len(rec) > 1 else None},
        "uncond_keeps_keyframe_rows": bool(len(calls) > 1 and calls[0]["cond_rows"] == calls[-1]["cond_rows"] > 0),
        "uncond_drops": "the caption only: inputs_uncond = {**inputs, **inputs_nega} swaps prompt_embeds/"
                        "text_token_tags/packed for the negative prompt ' ' (diffusion/loss.py:91); the keyframe "
                        "anchor, noisy latents and timesteps are shared",
    }
    return result("RECORD", False, **rep)


def check_d(m, pipe, data):
    import torch
    shared_t = data[0]
    calls, weights = [], []
    orig_mse = torch.nn.functional.mse_loss

    def mse(a, b, *k, **kw):
        v = orig_mse(a, b, *k, **kw)
        calls.append(float(v))
        return v
    w_orig = (pipe.scheduler.training_weight, pipe.scheduler_audio.training_weight)

    def tw(fn, tag):
        def f(t):
            v = fn(t)
            weights.append((tag, float(v)))
            return v
        return f
    out = {}
    torch.nn.functional.mse_loss = mse
    pipe.scheduler.training_weight = tw(w_orig[0], "video")
    pipe.scheduler_audio.training_weight = tw(w_orig[1], "audio")
    try:
        for w in (0.0, 1.0):
            calls.clear()
            weights.clear()
            m.audio_loss_weight = w
            with torch.no_grad():
                torch.manual_seed(0)
                loss = float(m({}, inputs=data))
            out[w] = {"loss": loss, "mse_calls": list(calls), "weights": list(weights)}
    finally:
        torch.nn.functional.mse_loss = orig_mse
        pipe.scheduler.training_weight, pipe.scheduler_audio.training_weight = w_orig
        m.audio_loss_weight = 0.0
    r0 = out[0.0]
    wv = next((v for t, v in r0["weights"] if t == "video"), None)
    video_term = r0["mse_calls"][0] * wv if r0["mse_calls"] and wv is not None else None
    audio_contrib = None if video_term is None else r0["loss"] - video_term
    has_audio = "audio_input_latents" in shared_t
    a = shared_t.get("audio_input_latents")
    ok = has_audio and audio_contrib is not None and abs(audio_contrib) <= 1e-6 * max(1.0, abs(r0["loss"])) \
        and len(r0["mse_calls"]) == 2
    return result("PASS" if ok else "FAIL", True, audio_stream_present=has_audio,
                  silent_latents=L.tensor_summary(a) if a is not None else None,
                  silent_latents_std=float(a.float().std()) if a is not None else None,
                  weight0=r0, weight1=out[1.0], audio_contribution_at_weight0=audio_contrib,
                  audio_term_at_weight1=(out[1.0]["loss"] - r0["loss"]))


def check_f(m, caches):
    import torch
    from diffsynth.core import UnifiedDataset
    if len(caches) < 2:
        return result("FAIL", True, error=f"only {sorted(caches)} probe caches; need f22 + f39 (+ f56)")
    ds = UnifiedDataset(base_path=str(L.PROBE), metadata_path=None, repeat=7)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    opt = torch.optim.AdamW([p for p in m.trainable_modules()], lr=1e-4, weight_decay=1e-3)
    shapes, losses, err = [], [], None
    try:
        for i, data in enumerate(dl):
            if i >= 20:
                break
            loss = m({}, inputs=data) / C.GRAD_ACCUM
            loss.backward()
            if (i + 1) % C.GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            shapes.append(int(data[0]["input_latents"].shape[2]))
            losses.append(round(float(loss) * C.GRAD_ACCUM, 5))
            print(f"  f: sample {i:2d} latents T={shapes[-1]} loss {losses[-1]}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    opt.zero_grad(set_to_none=True)
    ok = err is None and len(shapes) == 20 and len(set(shapes)) >= 2
    return result("PASS" if ok else "FAIL", True, cache_files=len(ds.cached_data), latent_T_seen=shapes,
                  distinct=sorted(set(shapes)), losses=losses, error=err,
                  on_fail="drop the 39/56 rungs and train f22 only (PLAN G0.2f, pre-declared)")


def finish(res):
    rep = C.load_json(REPORT, {}) or {}
    allc = rep.get("checks", {})
    hard_fail = [k for k, v in allc.items() if isinstance(v, dict) and v.get("hard") and v.get("status") != "PASS"]
    print("\nG0.2 summary:")
    for k in sorted(allc):
        v = allc[k]
        if isinstance(v, dict) and "status" in v:
            print(f"  {k}: {v['status']}{'' if v.get('hard') else ' (record)'}")
    L.merge_report(REPORT, {"hard_failures": hard_fail})
    print(f"report: {REPORT}")
    return 2 if any(k in res for k in hard_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
