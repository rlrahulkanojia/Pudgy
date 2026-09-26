#!/usr/bin/env python3
"""
sweep_ltx25.py - render checkpoints through ltx-pipelines, the real inference path.

STANDALONE. Shares nothing with the v1-v7 eval tooling.

WHY NOT THE TRAINER'S OWN VALIDATION SAMPLES
--------------------------------------------
LTX-2 #283 is OPEN and maintainer-acknowledged: the trainer's validation runner diverges
from training on the sigma schedule, never sets `keyframes_mask` (training omits it,
inference always sets it - and first_frame conditioning is exactly our mode), and uses the
opposite token order. So `outputs/.../samples/*.mp4` are sanity checks, never a basis for
picking a checkpoint. This renders through `ltx_pipelines` instead, which is what
production would actually run.

WHY IT BUILDS THE PIPELINE ONCE PER CHECKPOINT
----------------------------------------------
MEASURED on this box: one generation via the CLI costs 105 s, of which only 74 s is the
30-step denoising loop - the other ~31 s is loading the 39 GB transformer, the text
encoder and the VAE. One CLI call per generation pays that 31 s every time. Building the
pipeline once and looping (prompt x seed x start-frame) inside amortizes it to once per
checkpoint, which is ~30% off a sweep measured in hours.

THE SCORING RULE THIS ENFORCES
------------------------------
`--frames unseen` generates from start frames the model has NEVER seen (mid-clip frames of
the held-out SIDE_R angle). `--frames pose_library` generates from frame 0 of TRAINED
angles. Only `unseen` output may be scored; `pose_library` output is the control condition
of the memorisation probe. With a supplied start frame, a model that simply replays its
training clip satisfies "closest to the training footage" perfectly, so the two conditions
must be generated and reported separately or the headline number is meaningless.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from pathlib import Path

MODELS = Path("/workspace/models/ltx-2.5")
TRANSFORMER = MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
TEXT_ENCODER = MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = MODELS / "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = MODELS / "vae/ltx-2.5-audio-vae-bf16.safetensors"  # required even for silent output

# Must match the training bucket and the trained validation geometry.
H = W = 1024
FRAMES = 17
FPS = 24
STEPS = 30
CFG = 3.0
STG = 0.0        # off: #301 crashes DDP validation with STG, and the run trained with it off
RESCALE = 0.7
IMG_STRENGTH = 1.0   # #255: 1.0, not 0.8 - the start frame anchors the opening, nothing more


def start_frame_label(p: Path) -> tuple[str, str]:
    """('pax','angry') from 'pax_angry_SIDE_R__mustard_wide_f17_f8.png'."""
    m = re.match(r"(pax|polly)_([a-z]+)_", p.name)
    return (m.group(1), m.group(2)) if m else ("?", "?")


def build_jobs(data: Path, frames_dir: str, labels: list[str] | None,
               seeds: int, per_cell: int, seed0: int) -> list[dict]:
    """One job per (start frame, seed). Captions are taken from the corpus itself so the
    prompt a checkpoint is judged on is exactly the prompt distribution it trained on."""
    # Captions come from holdout.json FIRST, keyed by the exact clip the start frame was
    # cut from, so the caption's camera phrase MATCHES the frame being conditioned on.
    #
    # MEASURED CONFOUND: keying only on (character, label) against dataset.json pulled a
    # caption from some TRAINED angle - "front view", "turned strongly to its left" -
    # while the start frame was a SIDE_R profile. The model then dutifully rotated the
    # character to satisfy the caption, which reads in the output like identity
    # instability but is actually correct obedience to a prompt that disagreed with the
    # image. Unseen start frames are cut from held-out clips, so their true captions live
    # in holdout.json.
    cap_by_clip: dict[str, str] = {}
    for src in ("holdout.json", "dataset.json"):
        f = data / src
        if not f.exists():
            continue
        for e in json.loads(f.read_text()):
            cap_by_clip.setdefault(Path(e["video"]).stem, e["caption"])

    entries = json.loads((data / "dataset.json").read_text())
    cap_by_key: dict[tuple[str, str], str] = {}
    for e in entries:
        stem = Path(e["video"]).stem
        m = re.match(r"(pax|polly)_([a-z]+)_", stem)
        if m and (m.group(1), m.group(2)) not in cap_by_key:
            cap_by_key[(m.group(1), m.group(2))] = e["caption"]

    frames = sorted((data / "start_frames" / frames_dir).glob("*.png"))
    if not frames:
        sys.exit(f"no start frames in {data / 'start_frames' / frames_dir}")

    by_cell: dict[tuple[str, str], list[Path]] = {}
    for f in frames:
        by_cell.setdefault(start_frame_label(f), []).append(f)

    rng = random.Random(42)
    jobs = []
    for cell, fs in sorted(by_cell.items()):
        if labels and cell[1] not in labels:
            continue
        cap = cap_by_key.get(cell)
        if cap is None:
            continue
        rng.shuffle(fs)
        for f in fs[:per_cell]:
            # 'pax_angry_SIDE_R__olive_medium_f17_f8.png' -> clip stem without the _f<N>
            # frame-index suffix the start-frame extractor appended.
            clip_stem = re.sub(r"_f\d+$", "", f.stem)
            caption = cap_by_clip.get(clip_stem, cap)
            matched = clip_stem in cap_by_clip
            for s in range(seeds):
                jobs.append({"character": cell[0], "label": cell[1], "image": str(f),
                             "caption": caption, "caption_matches_frame": matched,
                             "seed": seed0 + s})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/workspace/Pudgy/Data/processed/ltx25_experiment"))
    ap.add_argument("--checkpoints", nargs="+", required=True, help="LoRA .safetensors paths")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="unseen", choices=["unseen", "pose_library"],
                    help="unseen = scorable; pose_library = memorisation control ONLY")
    ap.add_argument("--labels", nargs="*", default=None, help="restrict to these labels")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per start frame")
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--per-cell", type=int, default=1, help="start frames per (character,label)")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.data, args.frames, args.labels, args.seeds, args.per_cell, args.seed0)
    total = len(jobs) * len(args.checkpoints)
    print(f"{len(args.checkpoints)} checkpoints x {len(jobs)} generations = {total}")
    print(f"at ~75 s/generation after load: ~{total * 75 / 3600:.1f} h\n")
    if args.dry_run:
        for j in jobs[:8]:
            print(f"  {j['character']:5s} {j['label']:10s} seed {j['seed']}  {Path(j['image']).name}")
        print(f"  ... {len(jobs)} total per checkpoint")
        return

    import torch
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.denoisers import MultiModalGuiderParams
    from ltx_pipelines.utils.media_io.encode import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    # loras wants typed entries, not bare tuples: the loader reads .strength/.sd_ops.
    # LTXV_LORA_COMFY_RENAMING_MAP is the rename map the CLI's own LoraAction applies.
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    args.out.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    t_start = time.time()
    done = 0

    # ONE inference_mode around construction AND generation. The CLI decorates its whole
    # main() with @torch.inference_mode(); building the pipeline outside it and calling
    # inside mixes normal and inference tensors and raises "Inference tensors cannot be
    # saved for backward".
    ctx = torch.inference_mode()
    ctx.__enter__()

    for ckpt in args.checkpoints:
        step = re.search(r"step_(\d+)", Path(ckpt).stem)
        tag = f"step{int(step.group(1)):05d}" if step else Path(ckpt).stem
        cdir = args.out / tag
        cdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tag} :: building pipeline (paid once, not per generation) ===")
        t0 = time.time()
        pipeline = TI2VidOneStagePipeline(
            model_paths=ModelPaths(
                # ModelPaths is a frozen dataclass with no defaults: every slot must be
                # named. mode is descriptive only. embeddings_weight_paths is
                # (transformer, text_encoder) for a split pack with a separate TE.
                mode="split",
                transformer_path=str(TRANSFORMER), text_encoder_path=str(TEXT_ENCODER),
                video_vae_path=str(VIDEO_VAE), audio_vae_path=str(AUDIO_VAE),
                duration_head_path=None,
                embeddings_weight_paths=(str(TRANSFORMER), str(TEXT_ENCODER)),
            ),
            loras=(LoraPathStrengthAndSDOps(ckpt, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP),),
        )
        print(f"    pipeline ready in {time.time() - t0:.0f} s")

        for j in jobs:
            name = f"{j['character']}_{j['label']}_s{j['seed']}_{Path(j['image']).stem[:44]}.mp4"
            dest = cdir / name
            if dest.exists():
                done += 1
                continue
            t1 = time.time()
            if True:
                r = pipeline(
                    prompt=j["caption"], negative_prompt="", seed=j["seed"],   # "" not None: the pipeline .strip()s it
                    height=H, width=W, num_frames=FRAMES, frame_rate=FPS,
                    num_inference_steps=STEPS,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=CFG, stg_scale=STG, rescale_scale=RESCALE),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=CFG, stg_scale=0.0, rescale_scale=0.0),
                    images=[ImageConditioningInput(j["image"], 0, IMG_STRENGTH)],
                    vae_dtype=torch.bfloat16, tiling_config=AUTO_TILING,
                )
            encode_video(video=r.video, fps=FPS, audio=None, output_path=str(dest),
                         video_chunks_number=get_video_chunks_number(r.num_frames, r.tiling_config))
            done += 1
            dt = time.time() - t1
            eta = (total - done) * dt / 3600
            print(f"  [{done}/{total}] {dt:5.1f}s  ETA {eta:4.1f}h  {name}")
            index.append({"checkpoint": tag, "condition": args.frames, **j, "file": str(dest)})

        del pipeline
        torch.cuda.empty_cache()
        (args.out / "index.json").write_text(json.dumps(index, indent=1))

    ctx.__exit__(None, None, None)
    print(f"\ndone: {done} generations in {(time.time() - t_start)/3600:.2f} h -> {args.out}")
    print(f"condition = {args.frames}"
          + ("  (SCORABLE)" if args.frames == "unseen" else "  (memorisation CONTROL - never score)"))


if __name__ == "__main__":
    main()
