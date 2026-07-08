#!/usr/bin/env python
"""
Generate + evaluate several checkpoints with ONE model load, for golden-checkpoint picking.

Loads the base CogVideoX1.5-5B-I2V pipeline once, then for each requested checkpoint swaps in
its LoRA (at scale lora_alpha/rank), generates a video from the SAME conditioning frame / prompt
/ seed / resolution (so results are comparable), extracts every frame, computes consistency
metrics, and writes a per-checkpoint montage. Finally emits a comparison JSON + Markdown table.

Reuses helpers from eval_pudgy_lora.py and training_report.py so the generation and metrics match
the single-checkpoint path exactly.

    python inference/compare_checkpoints.py --checkpoints 750 1000 1250 1500 1750 2000
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pudgy_lora as E
import training_report as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Step numbers or 'final'")
    ap.add_argument("--lora_root", default=E.DEFAULT_LORA_ROOT)
    ap.add_argument("--model_path", default=E.DEFAULT_MODEL_PATH)
    ap.add_argument("--dataset_dir", default="/workspace/training_dataset")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--num_frames", type=int, default=33)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    report_dir = os.path.join(args.lora_root, "report")
    os.makedirs(report_dir, exist_ok=True)
    lora_scale = args.lora_alpha / args.rank

    # Same conditioning frame + prompt for every checkpoint.
    image, prompt = E.default_conditioning_from_dataset(args.dataset_dir)
    if image is None:
        sys.exit("Could not load a conditioning frame from the dataset.")
    max_h, max_w = E.model_resolution_caps(args.model_path)
    height, width = E.fit_resolution(image.width, image.height, max_h, max_w)
    print(f"Conditioning: dataset frame 0 | {width}x{height} | {args.num_frames}f @ {args.fps} | "
          f"scale {lora_scale} | seed {args.seed}")

    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
    from diffusers.utils import export_to_video

    print("Loading base pipeline once...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    results = []
    for ck in args.checkpoints:
        lora_dir, label = E.resolve_lora_dir(args.lora_root, ck)
        tag = "final" if label == "final" else f"ckpt{ck}"
        print(f"\n=== {label} -> {tag} ===")
        pipe.load_lora_weights(lora_dir, weight_name=E.LORA_WEIGHT_NAME, adapter_name="cmp")
        pipe.set_adapters(["cmp"], [lora_scale])
        frames = pipe(
            image=image, prompt=prompt, height=height, width=width,
            num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale, use_dynamic_cfg=True, num_videos_per_prompt=1,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        ).frames[0]
        pipe.unload_lora_weights()

        video_path = os.path.join(report_dir, f"{tag}.mp4")
        export_to_video(frames, video_path, fps=args.fps)
        # numpy RGB frames for metrics/montage
        import numpy as np
        np_frames = [np.array(f) for f in frames]
        metrics = R.consistency_metrics(np_frames)
        R.build_montage(np_frames, os.path.join(report_dir, f"{tag}_montage.png"))
        s = metrics.get("summary", {})
        results.append({"checkpoint": label, "tag": tag, "video": os.path.basename(video_path),
                        "summary": s, "vs_frame0_hist_last": metrics["vs_frame0"][-1]["hist_corr"],
                        "vs_frame0_ssim_last": metrics["vs_frame0"][-1]["ssim"]})
        print(f"  saved {video_path} | adj_ssim_mean={s.get('adjacent_ssim_mean')} "
              f"min={s.get('adjacent_ssim_min')} palette_drift={s.get('palette_drift_frame0_to_last')}")

    with open(os.path.join(report_dir, "compare_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Markdown comparison table.
    lines = ["## 5. Checkpoint comparison (golden-checkpoint sweep)\n",
             "Same conditioning frame, prompt, seed, resolution across all checkpoints.\n",
             "| Checkpoint | adj-SSIM mean | adj-SSIM min | palette drift f0→last | hist-corr@last | subject-retention* |",
             "|---|---|---|---|---|---|"]
    for r in results:
        s = r["summary"]
        # subject retention heuristic: high hist-corr at last frame => scene/subject still present
        retain = r["vs_frame0_hist_last"]
        flag = "good" if retain > 0.7 else ("fading" if retain > 0.5 else "LOST")
        lines.append(f"| {r['checkpoint']} | {s.get('adjacent_ssim_mean')} | {s.get('adjacent_ssim_min')} "
                     f"| {s.get('palette_drift_frame0_to_last')} | {round(retain,3)} | {flag} |")
    lines += ["", "*subject-retention = color-histogram correlation of the LAST frame vs the "
              "conditioning frame; low = the character/scene has drifted away or vanished by the end.\n",
              "Per-checkpoint montages: " + ", ".join(f"`{r['tag']}_montage.png`" for r in results) + "\n"]
    table = "\n".join(lines)
    with open(os.path.join(report_dir, "compare_table.md"), "w") as f:
        f.write(table)
    print("\n" + table)
    print(f"\nWrote compare_metrics.json + compare_table.md to {report_dir}")


if __name__ == "__main__":
    main()
