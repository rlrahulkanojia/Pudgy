#!/usr/bin/env python3
"""G0.6 - memory and throughput on the actual box (PLAN 4, 10; C5, C6).

Runs real training steps in-process with DiffSynth's own MiniMaxH3TrainingModule
(examples/minimax_h3/model_training/train.py:14-80) on tiny stage-1 caches (one training
clip per bucket, built by DiffSynth stage 1 if absent: h3lib.build_tiny_caches), with the
arm-A recipe: pruned bf16 DiT, LoRA rank 32 on C.LORA_TARGETS, guidance loss
(--training_cfg_scale 4, so every sample also runs the no-grad unconditional pass),
--audio_loss_weight 0, gradient checkpointing, batch 1, AdamW (lr 1e-4, wd 1e-3) stepping
every sample so optimizer state is resident. Records, per bucket f22/f39/f56: peak
allocated / reserved GB, allocator retries, s/sample (median of --steps after one warmup).

The reported anchor is ~53 GB for training the pruned bf16 build (R, DiffSynth #1624).
Writes phase0/mem_probe.json, including the PLAN 10 re-plan from the measured s/sample and
the corpus mix in manifest.json.

    python eval/mem_probe.py --dry
    python eval/mem_probe.py --frames 22,39,56
    python eval/mem_probe.py --arm d                # cfg 1 + DeCFG adapter (arm D)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3lib as L  # noqa: E402
C = L.C

OUT = C.PHASE0 / "mem_probe.json"
ADAPTER_PATH = C.MODELS / C.ADAPTER_REPO / "model_for_comfy_dit.safetensors"
PLAN_SAMPLES = {"pilot": 980, "main": 3920}


def corpus_mix():
    man = C.load_json(C.CORPUS / "manifest.json") or {}
    mix = {}
    for c in man.get("clips", []):
        if c.get("split") == "train":
            mix[c["frames"]] = mix.get(c["frames"], 0) + 1
    return mix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="22,39,56")
    ap.add_argument("--steps", type=int, default=4, help="timed samples per bucket (after 1 warmup)")
    ap.add_argument("--arm", choices=["a", "d"], default="a")
    ap.add_argument("--no-gc", action="store_true", help="gradient checkpointing OFF (for the record)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    buckets = [int(f) for f in args.frames.split(",") if f]

    plan = L.tiny_cache_plan(buckets)
    if args.dry:
        probs = L.check_dry_inputs([("DiffSynth train.py", L.train_py())])
        for F in buckets:
            if F not in plan:
                probs.append(f"no metadata_f{F}.csv row to build the f{F} probe cache from")
                continue
            row, d = plan[F]
            have = sorted(d.rglob("*.pth")) if d.exists() else []
            vid = C.CORPUS / row["video"]
            print(f"  f{F}: {row['video']}  cache {'present' if have else 'to build'} at {d}")
            if not vid.exists():
                probs.append(f"f{F}: {vid} missing")
        if args.arm == "d" and not ADAPTER_PATH.exists():
            print(f"  (arm d adapter not on this machine: {ADAPTER_PATH})")
        print(f"  mix (train clips by length): {corpus_mix() or 'manifest.json not found'}")
        print("DRY: problems:\n  - " + "\n  - ".join(probs) if probs else "DRY: inputs OK")
        return 1 if probs else 0

    import torch
    log = {}
    caches = L.build_tiny_caches(buckets, log)
    cfg = 4.0 if args.arm == "a" else 1.0
    t = time.time()
    m = L.load_train_module(cfg_scale=cfg, audio_loss_weight=0.0, gradient_checkpointing=not args.no_gc,
                            preset_lora=ADAPTER_PATH if args.arm == "d" else None)
    log["module_load_s"] = round(time.time() - t, 1)
    params = [p for p in m.trainable_modules()]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-3)
    base_alloc = torch.cuda.memory_allocated() / 1024 ** 3

    def one(data):
        loss = m({}, inputs=data)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    results = {}
    for F in buckets:
        if F not in caches:
            results[f"f{F}"] = {"error": "no probe cache"}
            continue
        data = L.load_cache_item(caches[F][0])
        lat = list(data[0]["input_latents"].shape)
        try:
            torch.manual_seed(0)
            one(data)                                    # warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            r0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
            times, losses = [], []
            for _ in range(args.steps):
                t = time.time()
                losses.append(one(data))
                torch.cuda.synchronize()
                times.append(time.time() - t)
            results[f"f{F}"] = {
                "input_latents": lat,
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
                "alloc_retries": int(torch.cuda.memory_stats().get("num_alloc_retries", 0) - r0),
                "s_per_sample": round(statistics.median(times), 3),
                "s_per_sample_all": [round(x, 3) for x in times],
                "loss": losses,
            }
        except torch.cuda.OutOfMemoryError as e:
            results[f"f{F}"] = {"input_latents": lat, "error": f"OOM: {e}"}
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(f"  f{F}: {L.dumps(results[f'f{F}'])}")

    mix = corpus_mix()
    replan = {}
    sp = {int(k[1:]): v["s_per_sample"] for k, v in results.items() if "s_per_sample" in v}
    if mix and all(F in sp for F in mix):
        tot = sum(mix.values())
        s_mix = sum(sp[F] * n for F, n in mix.items()) / tot
        replan = {"mix": mix, "s_per_sample_mix": round(s_mix, 3),
                  "pilot_h_each": round(PLAN_SAMPLES["pilot"] * s_mix / 3600, 2),
                  "main_h": round(PLAN_SAMPLES["main"] * s_mix / 3600, 2),
                  "plan_estimate_s_per_sample_mix": 5.7}
    rep = {**L.report_header("G0.6 memory + throughput"), "gpu": L.gpu_info(), "arm": args.arm,
           "training_cfg_scale": cfg, "gradient_checkpointing": not args.no_gc, "lora_params": n_params,
           "lora_params_expected": 155_100_000, "model_resident_gb": round(base_alloc, 2),
           "anchor_reported_gb": 53, "results": results, "replan": replan, "log": log}
    C.write_json(OUT, rep)
    print(f"wrote {OUT}")
    print(L.dumps({"lora_params": n_params, "replan": replan}))
    return 0 if all("error" not in v for v in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
