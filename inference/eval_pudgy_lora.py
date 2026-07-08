#!/usr/bin/env python
"""
Evaluate / run inference with a trained Pudgy Penguins CogVideoX1.5-5B-I2V LoRA.

This finds the LATEST checkpoint in the training output folder (the one the launch
script writes to, `finetune/output_dir/pudgy-lora-v1/`), loads the base model, applies
the LoRA at the CORRECT scale (lora_alpha / rank), and generates a video from a single
conditioning image (image-to-video).

Why a dedicated script (vs. inference/predict_i2v.py):
  * auto-discovers the newest `checkpoint-<step>/` (or the final root weights) — no need
    to hand-copy a path;
  * applies the LoRA at scale = lora_alpha/rank (the training-time scale). The old
    predict_i2v.py used `1/lora_rank` (~0.008), which under-applied the LoRA ~rank x so it
    had almost no effect. Use --lora_scale to override / sweep;
  * clamps the output resolution to what CogVideoX1.5's rotary grid actually supports for
    this portrait data (max 768px tall for a portrait clip), avoiding the
    "Expected size 63 but got 48" rotary crash.

Examples
--------
# List the checkpoints found in the training folder, then exit:
python inference/eval_pudgy_lora.py --list

# Generate from the latest checkpoint, using frame 0 of a training clip as the
# conditioning image and that clip's caption as the prompt:
python inference/eval_pudgy_lora.py --dataset_dir /workspace/training_dataset

# Generate from a specific checkpoint + your own image/prompt:
python inference/eval_pudgy_lora.py \
    --checkpoint 1000 \
    --image /path/to/pax.png \
    --prompt "Pax the blue penguin waddles across a snowy field, gentle bouncy motion" \
    --output_path pax_1000.mp4
"""

import argparse
import glob
import json
import os
import re
import sys

import torch


# --- paths, relative to this file: <repo>/inference/eval_pudgy_lora.py ---
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(INFERENCE_DIR)
DEFAULT_LORA_ROOT = os.path.join(REPO_ROOT, "finetune", "output_dir", "pudgy-lora-v1")
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "finetune", "models", "CogVideoX1.5-5B-I2V")
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def find_checkpoints(lora_root):
    """Return [(step:int, path)] for every checkpoint-<step> dir that holds LoRA weights,
    sorted ascending by step. Also probes the root dir (the trainer's final save)."""
    found = []
    if not os.path.isdir(lora_root):
        return found
    for d in sorted(glob.glob(os.path.join(lora_root, "checkpoint-*"))):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        if m and os.path.isfile(os.path.join(d, LORA_WEIGHT_NAME)):
            found.append((int(m.group(1)), d))
    found.sort(key=lambda x: x[0])
    # The trainer also writes the FINAL LoRA directly into the root dir.
    if os.path.isfile(os.path.join(lora_root, LORA_WEIGHT_NAME)):
        found.append((-1, lora_root))  # -1 sorts first; label it "final" below
    return found


def resolve_lora_dir(lora_root, checkpoint):
    """checkpoint: 'latest' | 'final' | '<int>' | an explicit dir path -> a dir with LoRA."""
    if checkpoint and os.path.isdir(checkpoint) and os.path.isfile(
        os.path.join(checkpoint, LORA_WEIGHT_NAME)
    ):
        return checkpoint, os.path.basename(checkpoint.rstrip("/"))

    ckpts = find_checkpoints(lora_root)
    if not ckpts:
        sys.exit(
            f"ERROR: no LoRA weights found under {lora_root!r}.\n"
            f"       Expected `{LORA_WEIGHT_NAME}` inside `checkpoint-<step>/` or the root.\n"
            f"       Train first (finetune/scripts/train_pudgy_lora.sh) or pass --lora_root."
        )

    step_ckpts = [c for c in ckpts if c[0] >= 0]
    root_ckpt = next((c for c in ckpts if c[0] == -1), None)

    if checkpoint in (None, "latest"):
        # Prefer the highest-step checkpoint; fall back to the final root save.
        chosen = step_ckpts[-1] if step_ckpts else root_ckpt
    elif checkpoint == "final":
        chosen = root_ckpt or step_ckpts[-1]
    elif str(checkpoint).isdigit():
        chosen = next((c for c in step_ckpts if c[0] == int(checkpoint)), None)
        if chosen is None:
            avail = ", ".join(str(s) for s, _ in step_ckpts) or "(none)"
            sys.exit(f"ERROR: checkpoint-{checkpoint} not found. Available steps: {avail}")
    else:
        sys.exit(f"ERROR: --checkpoint must be 'latest', 'final', a step number, or a dir path")

    step, path = chosen
    label = "final" if step == -1 else f"checkpoint-{step}"
    return path, label


def model_resolution_caps(model_path):
    """Max (height_px, width_px) the transformer's rotary grid supports, from its config.
    grid = pixels // (vae_spatial * patch) must be <= sample_dim // patch. Falls back to
    the known CogVideoX1.5-5B-I2V values (768 x 1360) if the config can't be read."""
    try:
        with open(os.path.join(model_path, "transformer", "config.json")) as f:
            cfg = json.load(f)
        p = cfg.get("patch_size", 2)
        vae_spatial = 8  # 2**(len(vae block_out_channels)-1); 4 blocks -> 8
        max_h = (cfg["sample_height"] // p) * (vae_spatial * p)  # 96//2 * 16 = 768
        max_w = (cfg["sample_width"] // p) * (vae_spatial * p)   # 170//2 * 16 = 1360
        return max_h, max_w
    except Exception:
        return 768, 1360


def fit_resolution(img_w, img_h, max_h, max_w, mult=16):
    """Largest aspect-preserving H,W that fits within (max_h, max_w) and is a multiple of
    `mult` (VAE spatial factor 8 x patch 2). For a portrait 768x1360 image -> 432x768."""
    scale = min(max_h / img_h, max_w / img_w)
    h = max(mult, (int(round(img_h * scale)) // mult) * mult)
    w = max(mult, (int(round(img_w * scale)) // mult) * mult)
    # Clamp to the grid ceiling (guards rounding at the boundary).
    h = min(h, (max_h // mult) * mult)
    w = min(w, (max_w // mult) * mult)
    return h, w


def default_conditioning_from_dataset(dataset_dir):
    """Extract frame 0 of the first training clip + its caption, so the script is runnable
    out of the box. Returns (PIL.Image, prompt) or (None, None) if unavailable."""
    meta_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    entries = meta if isinstance(meta, list) else meta.get("data", [])
    entries = [e for e in entries if e.get("type", "video") == "video"]
    if not entries:
        return None, None
    entry = entries[0]
    clip = os.path.join(dataset_dir, entry["file_path"])
    prompt = entry.get("text")
    try:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(clip)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, prompt
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return image, prompt
    except Exception as e:
        print(f"WARN: could not read frame 0 of {clip}: {e}")
        return None, prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora_root", default=DEFAULT_LORA_ROOT,
                    help="Training output folder that contains checkpoint-<step>/ dirs")
    ap.add_argument("--checkpoint", default="latest",
                    help="'latest' (default), 'final', a step number, or an explicit checkpoint dir")
    ap.add_argument("--list", action="store_true", help="List discovered checkpoints and exit")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                    help="Base CogVideoX1.5-5B-I2V model path or HF id")
    ap.add_argument("--image", default=None, help="Conditioning image (I2V). If omitted, uses a dataset frame")
    ap.add_argument("--prompt", default=None, help="Text prompt. If omitted, uses the dataset caption")
    ap.add_argument("--dataset_dir", default=os.path.join(os.path.dirname(REPO_ROOT), "training_dataset"),
                    help="Used only to auto-pick a conditioning image + prompt when --image/--prompt are omitted")
    ap.add_argument("--output_path", default=None, help="Output mp4 (default: <label>_<seed>.mp4)")
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank used in training")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha used in training")
    ap.add_argument("--lora_scale", type=float, default=None,
                    help="Override the adapter scale. Default = lora_alpha/rank (the training scale)")
    ap.add_argument("--num_frames", type=int, default=33, help="Frames to generate (training used 33)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--height", type=int, default=None, help="Override output height (must fit the grid)")
    ap.add_argument("--width", type=int, default=None, help="Override output width")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--cpu_offload", action="store_true",
                    help="Enable sequential CPU offload (needed on <~48GB cards)")
    args = ap.parse_args()

    # --- discover checkpoints ---
    ckpts = find_checkpoints(args.lora_root)
    if args.list or not ckpts:
        print(f"LoRA root: {args.lora_root}")
        if not ckpts:
            print("  (no checkpoints found yet)")
        for step, path in ckpts:
            print(f"  {'final ' if step == -1 else f'step {step:>6}'}  {path}")
        if args.list:
            return
        sys.exit(1)

    lora_dir, label = resolve_lora_dir(args.lora_root, args.checkpoint)
    lora_scale = args.lora_scale if args.lora_scale is not None else args.lora_alpha / args.rank
    print(f"Using LoRA: {lora_dir}  ({label})")
    print(f"LoRA scale: {lora_scale:.4f}  (lora_alpha/rank = {args.lora_alpha}/{args.rank})")

    # --- conditioning image + prompt ---
    from diffusers.utils import export_to_video, load_image

    if args.image:
        image = load_image(args.image)
    else:
        image, ds_prompt = default_conditioning_from_dataset(args.dataset_dir)
        if image is None:
            sys.exit("ERROR: no --image given and could not read a frame from --dataset_dir. Pass --image.")
        if args.prompt is None:
            args.prompt = ds_prompt
        print(f"Conditioning image: frame 0 of first clip in {args.dataset_dir}")
    if not args.prompt:
        sys.exit("ERROR: no prompt. Pass --prompt (or --dataset_dir with a metadata.json caption).")

    # --- resolution: fit within the model's rotary grid ceiling ---
    max_h, max_w = model_resolution_caps(args.model_path)
    if args.height and args.width:
        height, width = args.height, args.width
    else:
        height, width = fit_resolution(image.width, image.height, max_h, max_w)
    print(f"Output resolution: {width}x{height} (WxH); model grid cap {max_w}x{max_h}. "
          f"Frames={args.num_frames} @ {args.fps}fps")

    # --- build the pipeline ---
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Apply the LoRA at the training-time scale.
    pipe.load_lora_weights(lora_dir, weight_name=LORA_WEIGHT_NAME, adapter_name="pudgy-lora")
    pipe.set_adapters(["pudgy-lora"], [lora_scale])

    # VAE tiling/slicing keeps the decode of a full 33-frame video within VRAM. The decode
    # peak is far higher than the diffusion steps; without this it OOMs on a 40GB card even
    # though the transformer fits. Negligible quality impact, so enable it unconditionally.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    # --- generate ---
    frames = pipe(
        image=image,
        prompt=args.prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_dynamic_cfg=True,
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).frames[0]

    out = args.output_path or f"{label.replace('/', '_')}_seed{args.seed}.mp4"
    export_to_video(frames, out, fps=args.fps)
    print(f"Saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
