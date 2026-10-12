"""expv2 GPU helpers shared by vae_gate, pipeline_checks, dcls, generate and mem_probe.

Everything that touches DiffSynth is written against DiffSynth-Studio at
C.DIFFSYNTH_SHA (acf2ad2, 2026-10-09). Line references below are to that tree.

Nothing here imports torch or diffsynth at module level, so every script's `--dry` mode
runs on a machine with neither (the Mac). Heavy imports happen inside functions.

Two-pass evaluation (MODEL_NOTES 2.7, PLAN 7.1). Qwen3-VL-32B (~66 GB) and the DiT
(~40 GB) never sit on the GPU together:
  pass 1  `load_pipe(("te", "vae"[, "audio_vae"]))` runs the pipeline's own unit chain
          (pipelines/minimax_h3_audio_video.py:40-52) and caches the result on disk;
  pass 2  `load_pipe(("dit", "vae"))` replays MiniMaxH3Pipeline.__call__ steps 4-5
          (minimax_h3_audio_video.py:177-204) on the cached inputs.
G0.2g checks that pass 1 + pass 2 equals one `pipe(...)` call.
"""
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import expv2_common as C  # noqa: E402

# DiffSynth reads these when it resolves a ModelConfig (core/loader/config.py:81-90,
# 132-137). configs/env.sh exports them too; setdefault keeps an explicit export.
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "HuggingFace")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(C.MODELS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUN_NAMES = ("expv2-h3-s12", "expv2-h3-s222", "expv2-h3-main", "expv2-h3-d")
STEP_RE = re.compile(r"^step-(\d+)\.safetensors$")

# ---------------------------------------------------------------------------------------
# Checkpoint specs (INTERFACES section 2)
# ---------------------------------------------------------------------------------------


def run_ckpts(run):
    """[(opt_step, samples, path)] for every step-N file of a run, sorted by samples.
    DiffSynth names checkpoints by SAMPLES (diffusion/logger.py:104-109 increments per
    dataloader item), so opt step = samples / GRAD_ACCUM."""
    d = C.RUNS / run
    out = []
    if d.is_dir():
        for p in d.iterdir():
            m = STEP_RE.match(p.name)
            if m:
                s = int(m.group(1))
                out.append((C.opt_step(s), s, p))
    return sorted(out, key=lambda t: t[1])


def ckpt_name_for_path(p):
    p = Path(p)
    m = STEP_RE.match(p.name)
    if m and p.parent.parent == C.RUNS:
        return f"{p.parent.name}@{C.opt_step(int(m.group(1)))}"
    return p.stem


def resolve_ckpts(specs, must_exist=True):
    """'base,expv2-h3-main@250,expv2-h3-s12@last,/x/y.safetensors' -> [(name, path|None)].

    base               no LoRA
    <run>@<optstep>    output/<run>/step-<optstep*8>.safetensors
    <run>@all          every saved step of the run
    <run>@last         the highest step
    <path>             any .safetensors (name from the path)
    """
    out = []
    for spec in [s.strip() for s in str(specs).split(",") if s.strip()]:
        if spec == "base":
            out.append(("base", None))
            continue
        if "@" in spec and not spec.endswith(".safetensors"):
            run, what = spec.split("@", 1)
            have = run_ckpts(run)
            if what == "all":
                if not have and must_exist:
                    raise FileNotFoundError(f"no checkpoints under {C.RUNS / run}")
                out += [(f"{run}@{o}", p) for o, _, p in have]
            elif what == "last":
                if not have:
                    if must_exist:
                        raise FileNotFoundError(f"no checkpoints under {C.RUNS / run}")
                    out.append((f"{run}@last", None))
                else:
                    o, _, p = have[-1]
                    out.append((f"{run}@{o}", p))
            else:
                o = int(what)
                p = C.RUNS / run / f"step-{o * C.GRAD_ACCUM}.safetensors"
                if must_exist and not p.exists():
                    raise FileNotFoundError(f"{spec}: {p} missing (opt step {o} = {o * C.GRAD_ACCUM} samples)")
                out.append((f"{run}@{o}", p))
            continue
        p = Path(spec)
        if must_exist and not p.exists():
            raise FileNotFoundError(spec)
        out.append((ckpt_name_for_path(p), p))
    return out


# ---------------------------------------------------------------------------------------
# Corpus access (no torch)
# ---------------------------------------------------------------------------------------


def load_eval_items(suites=None, items=None):
    data = C.load_json(C.CORPUS / "eval_items.json")
    if data is None:
        raise FileNotFoundError(f"{C.CORPUS / 'eval_items.json'} missing - run prep/prep_expv2.py")
    if suites:
        want = set(suites)
        data = [d for d in data if d["suite"] in want]
    if items:
        keys = [k for k in items if k]
        data = [d for d in data if any(k == d["item_id"] or k in d["item_id"] for k in keys)]
    return data


def load_tier1_items(sets=("holdout",)):
    data = C.load_json(C.CORPUS / "tier1_items.json")
    if data is None:
        raise FileNotFoundError(f"{C.CORPUS / 'tier1_items.json'} missing - run prep/prep_expv2.py")
    return [d for d in data if d["split"] in set(sets)]


def png_files(d):
    d = Path(d)
    return sorted(p for p in d.glob("*.png"))


def read_frames(d, n=None):
    """PNG directory -> list of RGB PIL images (exactly what DiffSynth's loader yields:
    core/data/operators.py:182-195 builds Image.fromarray(rgb uint8) per frame)."""
    from PIL import Image
    files = png_files(d)
    if n is not None:
        if len(files) < n:
            raise ValueError(f"{d}: {len(files)} frames < {n}")
        files = files[:n]
    return [Image.open(f).convert("RGB") for f in files]


def save_frames(frames, d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(d / f"{i:04d}.png")


def frames_to_u8(frames):
    import numpy as np
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames])


def psnr_u8(a, b):
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def write_preview_mp4(png_dir, out_mp4, fps=C.FPS):
    """Viewing copy only (MODEL_NOTES 2.11: nothing is ever scored from an mp4)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(Path(png_dir) / "%04d.png"),
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:  # preview is optional
        print(f"  (preview skipped: {e})")
        return False


def stable_seed(*parts):
    """Deterministic 31-bit seed from arbitrary parts (identical across checkpoints)."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def text_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def diffsynth_sha():
    try:
        return subprocess.run(["git", "-C", str(C.DIFFSYNTH), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------------------

PARTS = {
    "te": C.TEXT_ENCODER,
    "dit": C.DIT_PRUNED,
    "dit_full": C.DIT_FULL,
    "vae": C.VIDEO_VAE,
    "audio_vae": C.AUDIO_VAE,
}


def vram_config():
    """The low-VRAM config of DiffSynth's own H3 examples
    (examples/minimax_h3/model_inference_low_vram/*, validate_lora/*): weights live on CPU
    and each layer moves to the GPU on first use while used VRAM < vram_limit
    (core/vram/layers.py:435-438). It also turns VRAM management ON
    (models/model_loader.py:20-21), which is what makes LoRA HOT-loading possible:
    pipe.load_lora hot-loads when vram management is enabled (diffusion/base_pipeline.py:
    270-271) and pipe.clear_lora removes it again (:291-303). Without it, load_lora fuses
    into the base weights and clear_lora is a no-op."""
    import torch
    return {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }


def model_config(spec, **vram):
    """'repo:pattern' -> ModelConfig (same split rule as training_module.py:244-253)."""
    from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
    i = spec.rfind(":")
    return ModelConfig(model_id=spec[:i], origin_file_pattern=spec[i + 1:], **vram)


def load_pipe(parts, vram_limit_gb=None):
    """MiniMaxH3Pipeline with only the named parts loaded (missing ones are None:
    fetch_model prints 'not an error', models/model_loader.py:103-111)."""
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
    vc = vram_config()
    cfgs = [model_config(PARTS[p], **vc) for p in parts]
    if vram_limit_gb is None:
        vram_limit_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=cfgs,
        processor_config=model_config(C.PROCESSOR), vram_limit=vram_limit_gb,
    )
    return pipe


def free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def attach_lora(pipe, path):
    """Hot-load a trained LoRA at strength 1.0 (PLAN 7.1). Keys are DiffSynth's own export
    (training_module.py:119-129) and go through MiniMaxH3LoRALoader
    (utils/lora/minimax_h3.py:5-18), exactly as validate_lora/MiniMax-H3-Pruned-FL2VA.py."""
    if path is None:
        return
    if not getattr(pipe.dit, "vram_management_enabled", False):
        raise RuntimeError("DiT loaded without VRAM management: load_lora would FUSE and "
                           "clear_lora could not undo it. Load with h3lib.load_pipe.")
    pipe.load_lora(pipe.dit, str(path), alpha=1, hotload=True)
    # Hot-loading matches keys silently (base_pipeline.py:273-284 only PRINTS the count): a
    # key mismatch would leave every "LoRA" evaluation scoring the base model. Refuse that.
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        n_file = sum(1 for k in f.keys() if "lora_A" in k)
    n_patched = sum(1 for m in pipe.dit.modules() if len(getattr(m, "lora_A_weights", None) or []) > 0)
    if n_patched == 0 or n_patched != n_file:
        raise RuntimeError(f"LoRA {path}: {n_patched} DiT layers patched, file holds {n_file} lora_A "
                           f"tensors - key layout mismatch (MODEL_NOTES 2.6); refusing to score")


def detach_lora(pipe):
    pipe.clear_lora(verbose=0)


def onload(pipe, names):
    """Mark models 'onload' so their layers migrate to the GPU on first use and stay there
    (state 0 -> 1, core/vram/layers.py:395-402). Without this a direct model_fn call
    would re-copy every weight from CPU on every forward. Units and __call__ do this
    themselves; direct DiT calls (dcls, parity) must call it once."""
    pipe.load_models_to_device(list(names))


def to_device(obj, device):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device(v, device) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj


def to_cpu(obj):
    return to_device(obj, "cpu")


def strip_pil(d):
    """Drop PIL images (keyframes, input_video) before torch.save; nothing downstream of
    the unit chain reads them (model_fn_minimax_h3, minimax_h3_audio_video.py:944+)."""
    from PIL import Image
    out = {}
    for k, v in d.items():
        if isinstance(v, Image.Image):
            continue
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], Image.Image):
            continue
        out[k] = v
    return out


# ---------------------------------------------------------------------------------------
# Pass 1 - generation inputs, exactly as MiniMaxH3Pipeline.__call__ builds them
# ---------------------------------------------------------------------------------------


def generation_inputs(pipe, caption, start_image, num_frames, seed):
    """inputs_shared / posi / nega as __call__ builds them (minimax_h3_audio_video.py:
    155-172), with the PLAN 7.1 settings, first frame only."""
    I = C.INFERENCE
    shared = {
        "cfg_scale": I["cfg_scale"],
        "height": I["height"], "width": I["width"], "num_frames": num_frames,
        "seed": seed, "rand_device": I["rand_device"],
        "tiled": I["tiled"], "tile_size": I["tile_size"], "tile_overlap": I["tile_overlap"],
        "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
        "keyframes": [start_image], "keyframe_indices": [0],
        "references": None, "ref_image_short_edge": 2048, "ref_video_short_edge": 768,
        "ref_video_max_pixels": 768 * 1344,
        "control_video": None, "control_scale": 1.0,
        "retake_video": None, "frame_regions_to_retake": None,
        "retake_audio": None, "seconds_regions_to_retake": None,
        "imgvid_cond_noise_aug": pipe.imgvid_cond_noise_aug,
        "audio_cond_noise_aug": pipe.audio_cond_noise_aug,
        "text_embedding": None,
    }
    return shared, {"prompt": caption}, {"negative_prompt": " "}


def run_units(pipe, shared, posi, nega, memo=None, memo_key=None):
    """The pipeline's unit chain (minimax_h3_audio_video.py:174-175), optionally reusing a
    PromptEmbedder result: Qwen3-VL's output depends only on (prompt, keyframes, height,
    width) (:293-325), not on the seed, so one encode serves all three seeds."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    for unit in pipe.units:
        if memo is not None and isinstance(unit, MiniMaxH3Unit_PromptEmbedder) and memo_key in memo:
            posi.update(memo[memo_key])
            if shared["cfg_scale"] == 1:
                nega.update(memo[memo_key])      # PipelineUnitRunner does the same at cfg 1 (base_pipeline.py:536-537)
            continue
        shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
        if memo is not None and isinstance(unit, MiniMaxH3Unit_PromptEmbedder):
            memo[memo_key] = {k: posi[k] for k in ("prompt_embeds", "text_token_tags")}
    return shared, posi, nega


def encode_generation(pipe, caption, start_image, num_frames, seed, memo=None, memo_key=None):
    """Pass 1 for one (caption, start frame, length, seed) -> CPU dict for torch.save."""
    import torch
    I = C.INFERENCE
    with torch.no_grad():
        # __call__ sets the inference schedule before the units (:152-153); no unit reads it
        # except the training flag, which must be False here.
        pipe.scheduler.set_timesteps(I["num_inference_steps"], shift=I["flow_shift"])
        pipe.scheduler_audio.set_timesteps(I["num_inference_steps"], shift=I["audio_flow_shift"])
        shared, posi, nega = generation_inputs(pipe, caption, start_image, num_frames, seed)
        shared, posi, nega = run_units(pipe, shared, posi, nega, memo, memo_key)
    return {"shared": to_cpu(strip_pil(shared)), "posi": to_cpu(strip_pil(posi)),
            "caption_hash": text_hash(caption), "num_frames": num_frames, "seed": seed}


def denoise(pipe, enc, steps=None, flow_shift=None, audio_flow_shift=None, progress=False):
    """Pass 2: __call__ steps 4-5 (minimax_h3_audio_video.py:177-204) on cached inputs.
    cfg is 1.0 (the model is CFG-distilled), so only the positive branch runs. The audio
    stream is denoised (it is part of the joint sequence) but not decoded - discarded."""
    import torch
    I = C.INFERENCE
    steps = steps or I["num_inference_steps"]
    flow_shift = I["flow_shift"] if flow_shift is None else flow_shift
    audio_flow_shift = I["audio_flow_shift"] if audio_flow_shift is None else audio_flow_shift
    with torch.no_grad():
        pipe.scheduler.set_timesteps(steps, shift=flow_shift)
        pipe.scheduler_audio.set_timesteps(steps, shift=audio_flow_shift)
        shared = to_device(enc["shared"], pipe.device)
        posi = to_device(enc["posi"], pipe.device)
        assert shared["cfg_scale"] == 1.0, "expv2 samples at cfg 1 only (PLAN 7.1)"
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        it = pipe.scheduler.timesteps
        if progress:
            from tqdm import tqdm
            it = tqdm(it)
        for progress_id, timestep_video in enumerate(it):
            timestep_video = timestep_video.unsqueeze(0).to(dtype=torch.float32, device=pipe.device)
            timestep_audio = pipe.scheduler_audio.timesteps[progress_id].unsqueeze(0).to(dtype=torch.float32, device=pipe.device)
            noise_pred_video, noise_pred_audio = pipe.cfg_guided_model_fn(
                pipe.model_fn, shared["cfg_scale"], shared, posi, posi,
                **models, timestep_video=timestep_video, timestep_audio=timestep_audio,
            )
            shared["video_latents"] = pipe.step(
                pipe.scheduler, shared["video_latents"], progress_id, noise_pred=noise_pred_video,
                inpaint_mask=shared.get("denoise_mask_video"), input_latents=shared.get("input_latents_video"),
            )
            shared["audio_latents"] = pipe.step(
                pipe.scheduler_audio, shared["audio_latents"], progress_id, noise_pred=noise_pred_audio,
                inpaint_mask=shared.get("denoise_mask_audio"), input_latents=shared.get("input_latents_audio"),
            )
        pipe.load_models_to_device(["video_vae"])
        frames = pipe.video_vae.decode_video(shared["video_latents"], dtype=pipe.torch_dtype,
                                             tiled=shared["tiled"], tile_size=shared["tile_size"],
                                             tile_overlap=shared["tile_overlap"])
        return pipe.vae_output_to_video(frames, min_value=0, max_value=1)


# ---------------------------------------------------------------------------------------
# Training-path inputs (Tier 1, G0.2): the unit chain as train.py's stage 1 runs it
# ---------------------------------------------------------------------------------------


def silent_audio(num_frames):
    """train.py --silent_on_missing_audio (examples/minimax_h3/model_training/train.py:101-103):
    zeros of 800*round(n/24*40) samples at 32 kHz, stereo."""
    import torch
    return (torch.zeros((2, 800 * round(num_frames / 24 * 40))), 32000)


def training_inputs(pipe, frames, caption, cfg_scale=1.0):
    """inputs as MiniMaxH3TrainingModule.get_pipeline_inputs + parse_extra_inputs build them
    for `--extra_inputs input_audio,input_image` (train.py:83-137): keyframe = frame 0,
    keyframe_indices [0], seed 42, silent audio. cfg_scale 1 skips the negative branch."""
    n = len(frames)
    shared = {
        "input_video": frames,
        "height": frames[0].size[1], "width": frames[0].size[0], "num_frames": n,
        "keyframes": [frames[0]], "keyframe_indices": [0],
        "references": None, "ref_image_short_edge": 2048,
        "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
        "imgvid_cond_noise_aug": pipe.imgvid_cond_noise_aug,
        "audio_cond_noise_aug": pipe.audio_cond_noise_aug,
        "tiled": True, "tile_size": 256, "tile_overlap": 64,
        "cfg_scale": cfg_scale, "seed": 42, "rand_device": "cpu",
        "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
        "input_audio": silent_audio(n),
    }
    return shared, {"prompt": caption}, {"negative_prompt": " "}


def encode_training_path(pipe, frames, captions):
    """Run the training unit chain once (true caption first) and re-run only the prompt
    and packing units for every other caption. Returns a CPU dict:
      x0, a0          clean video / audio latents (InputVideoEmbedder :325-340,
                      InputAudioEmbedder :342-357)
      anchor          keyframe_cond_anchor (KeyframeEncoder :428-463, seed 42 as training)
      cands[label]    {prompt_embeds, text_token_tags, packed}
    `captions` is an ordered dict label -> caption."""
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder)
    labels = list(captions)
    with torch.no_grad():
        # InputVideo/AudioEmbedder only run when scheduler.training (minimax_h3_audio_video.py:334,351)
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.scheduler_audio.set_timesteps(1000, training=True)
        shared, posi, nega = training_inputs(pipe, frames, captions[labels[0]])
        for unit in pipe.units:
            shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
        prompt_unit = next(u for u in pipe.units if isinstance(u, MiniMaxH3Unit_PromptEmbedder))
        pack_unit = next(u for u in pipe.units if isinstance(u, MiniMaxH3Unit_PackedSequenceBuilder))
        cands = {labels[0]: {k: posi[k] for k in ("prompt_embeds", "text_token_tags", "packed")}}
        for lab in labels[1:]:
            p, ng = {"prompt": captions[lab]}, {"negative_prompt": " "}
            _, p, ng = pipe.unit_runner(prompt_unit, pipe, shared, p, ng)
            _, p, ng = pipe.unit_runner(pack_unit, pipe, shared, p, ng)
            cands[lab] = {k: p[k] for k in ("prompt_embeds", "text_token_tags", "packed")}
        out = {
            "x0": shared["input_latents"], "a0": shared["audio_input_latents"],
            "anchor": shared["keyframe_cond_anchor"],
            "imgvid_cond_noise_aug": float(shared["imgvid_cond_noise_aug"]),
            "audio_cond_noise_aug": float(shared["audio_cond_noise_aug"]),
            "cands": cands, "num_frames": len(frames),
            "caption_hashes": {k: text_hash(v) for k, v in captions.items()},
        }
        pipe.scheduler.set_timesteps(1000, training=False)
        pipe.scheduler_audio.set_timesteps(1000, training=False)
        return to_cpu(out)


def velocity_at(pipe, enc, label, sigma, noise_v, noise_a):
    """One DiT forward on the training path at noise level sigma (1 = pure noise).
    Mirrors FlowMatchSFTMiniMaxH3AudioVideoLoss (diffusion/loss.py:66-118): noisy =
    (1-s)x0 + s*eps (flow_match.py:412-418), target = eps - x0 (:420-422), the model's
    'noise_pred' is model_fn's first output. Audio is noised at the same sigma.
    Returns (noise_pred_video fp32, target fp32)."""
    import torch
    dev = pipe.device
    x0 = enc["x0"].to(dev)
    a0 = enc["a0"].to(dev)
    s = torch.tensor(float(sigma), dtype=torch.float32)
    xt = (1 - s) * x0 + s * noise_v.to(dev, x0.dtype)
    at = (1 - s) * a0 + s * noise_a.to(dev, a0.dtype)
    target = noise_v.to(dev, x0.dtype) - x0
    cand = to_device(enc["cands"][label], dev)
    tv = torch.tensor([float(sigma) * 1000.0], dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred_v, _ = pipe.model_fn(
            dit=pipe.dit, video_latents=xt, audio_latents=at,
            packed=cand["packed"], prompt_embeds=cand["prompt_embeds"],
            timestep_video=tv, timestep_audio=tv,
            keyframe_cond_anchor=enc["anchor"].to(dev),
            imgvid_cond_noise_aug=enc["imgvid_cond_noise_aug"],
            audio_cond_noise_aug=enc["audio_cond_noise_aug"],
            controlnet=None,
        )
    return pred_v.float(), target.float()


def fixed_noise(enc, *seed_parts):
    """Noise shared by every checkpoint for one (clip, sigma, draw)."""
    import torch
    g = torch.Generator("cpu").manual_seed(stable_seed(C.TIER1_NOISE_SEED, *seed_parts))
    nv = torch.randn(tuple(enc["x0"].shape), generator=g, dtype=torch.float32).to(enc["x0"].dtype)
    na = torch.randn(tuple(enc["a0"].shape), generator=g, dtype=torch.float32).to(enc["a0"].dtype)
    return nv, na


def latent_frames(num_latents):
    """Frames covered by each video latent, from the H3 VAE's actual structure
    (models/minimax_h3_video_vae.py:439-443, 512-519): the clip is cut into 17-frame chunks
    FROM FRAME 0 (the last one padded by repeating its final frame, then 3 tokens dropped),
    and each chunk gives 5 tokens covering (1,4,4,4,4) frames - the same pattern the
    packer's RoPE grid uses (_FRAME_PER_TOKEN, minimax_h3_audio_video.py:591).
    22 frames -> {0},{1-4},{5-8},{9-12},{13-16},{17},{18-21}. Latent 0 = frame 0 alone."""
    out = []
    for i in range(num_latents):
        c, k = divmod(i, 5)
        base = 17 * c
        out.append([base] if k == 0 else list(range(base + 1 + 4 * (k - 1), base + 1 + 4 * k)))
    return out


def chunk_starts(num_frames, clip_length=17):
    """First frame of every VAE chunk after the first: 17, 34, 51."""
    return [k for k in range(clip_length, num_frames, clip_length)]


# ---------------------------------------------------------------------------------------
# Training module + tiny stage-1 caches (G0.2 b,c,d,f and G0.6)
# ---------------------------------------------------------------------------------------

TRAIN_PY = "examples/minimax_h3/model_training/train.py"


def train_py():
    return C.DIFFSYNTH / TRAIN_PY


def train_module_class():
    """Import DiffSynth's H3 train.py as a module (its CLI is guarded by __main__)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("expv2_diffsynth_h3_train", str(train_py()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MiniMaxH3TrainingModule, mod


def load_train_module(cfg_scale=4.0, audio_loss_weight=0.0, gradient_checkpointing=True,
                      preset_lora=None, device="cuda"):
    """The stage-2 ('sft:train') module exactly as configs/train.sh builds it (PLAN 6.2):
    pruned DiT, LoRA rank 32 on C.LORA_TARGETS, guidance loss, video term only."""
    TM, _ = train_module_class()
    m = TM(
        model_id_with_origin_paths=C.DIT_PRUNED, processor_path=C.PROCESSOR,
        lora_base_model="dit", lora_target_modules=C.LORA_TARGETS, lora_rank=C.LORA_RANK,
        preset_lora_path=str(preset_lora) if preset_lora else None,
        preset_lora_model="dit" if preset_lora else None,
        use_gradient_checkpointing=gradient_checkpointing,
        extra_inputs="input_audio,input_image", silent_on_missing_audio=True,
        training_cfg_scale=cfg_scale, audio_loss_weight=audio_loss_weight,
        remove_prefix_in_ckpt="pipe.dit.", task="sft:train", device=device,
    )
    m.to(device=device)
    return m


def accelerate_cmd():
    import shutil
    a = shutil.which("accelerate")
    return [a, "launch", "--num_processes", "1"] if a else [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1"]


def stage1_cmd(metadata_csv, num_frames, out_dir, base_path=None, cfg_scale=4.0):
    """DiffSynth stage 1 ('sft:data_process'), as configs/cache.sh runs it. No DiT (the
    pruned example's stage 1 loads TE + VAEs only); --training_cfg_scale 4 so the
    unconditional embeddings are cached (diffusion/loss.py:84-90 raises otherwise)."""
    return accelerate_cmd() + [
        str(train_py()),
        "--dataset_base_path", str(base_path or C.CORPUS),
        "--dataset_metadata_path", str(metadata_csv),
        "--data_file_keys", "video,input_audio",
        "--extra_inputs", "input_audio,input_image",
        "--silent_on_missing_audio",
        "--height", str(C.SIZE), "--width", str(C.SIZE), "--num_frames", str(num_frames),
        "--dataset_repeat", "1",
        "--model_id_with_origin_paths", ",".join([C.TEXT_ENCODER, C.VIDEO_VAE, C.AUDIO_VAE]),
        "--processor_path", C.PROCESSOR,
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", str(out_dir),
        "--lora_base_model", "dit", "--lora_target_modules", C.LORA_TARGETS,
        "--lora_rank", str(C.LORA_RANK),
        "--use_gradient_checkpointing",
        "--training_cfg_scale", str(cfg_scale),
        "--task", "sft:data_process",
    ]


PROBE = C.CACHE / "probe"


def tiny_cache_plan(buckets=C.BUCKETS):
    """One training clip per bucket for the tiny caches: {F: (metadata row dict, out dir)}.
    f56 is neutral by construction (the only label with a 56 rung)."""
    import csv
    plan = {}
    for F in buckets:
        meta = C.CORPUS / f"metadata_f{F}.csv"
        if not meta.exists():
            continue
        with open(meta, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            plan[F] = (rows[0], PROBE / f"f{F}")
    return plan


def build_tiny_caches(buckets=C.BUCKETS, log=None):
    """Stage-1 caches of one clip per bucket under CACHE/probe/f{F}/0/0.pth (reused if
    present). These are the real DiffSynth stage-1 outputs, so b/c/d/f/G0.6 see exactly
    what training sees."""
    import csv
    out = {}
    PROBE.mkdir(parents=True, exist_ok=True)
    for F, (row, d) in tiny_cache_plan(buckets).items():
        have = sorted(d.rglob("*.pth")) if d.exists() else []
        if have:
            out[F] = have
            continue
        meta = PROBE / f"meta_f{F}.csv"
        with open(meta, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "prompt", "input_audio"])
            w.writeheader()
            w.writerow({k: row[k] for k in ("video", "prompt", "input_audio")})
        cmd = stage1_cmd(meta, F, d)
        print("  $ " + " ".join(cmd[:4]) + " ... --num_frames %d --task sft:data_process" % F)
        t = time.time()
        with open(PROBE / f"stage1_f{F}.log", "w") as lf:
            r = subprocess.run(cmd, cwd=str(C.DIFFSYNTH), stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"stage 1 f{F} failed (rc {r.returncode}); see {PROBE / f'stage1_f{F}.log'}")
        out[F] = sorted(d.rglob("*.pth"))
        if log is not None:
            log[f"stage1_f{F}_s"] = round(time.time() - t, 1)
    return out


def load_cache_item(path):
    """A stage-1 .pth: the (inputs_shared, inputs_posi, inputs_nega) tuple the
    data_process task saves (diffusion/runner.py:145-151; core/data/operators.py:259-264)."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


def tensor_summary(t):
    import torch
    if isinstance(t, torch.Tensor):
        return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    if isinstance(t, dict):
        return {k: tensor_summary(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [tensor_summary(v) for v in t][:8]
    if isinstance(t, (int, float, str, bool)) or t is None:
        return t
    return type(t).__name__


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": p.name, "total_gb": round(p.total_memory / 1024 ** 3, 1),
            "torch": torch.__version__, "cuda_version": torch.version.cuda}


def report_header(what):
    return {"what": what, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_sha": git_sha(),
            "diffsynth_sha": diffsynth_sha(), "diffsynth_pinned": C.DIFFSYNTH_SHA}


def check_dry_inputs(paths):
    """[(label, path)] -> list of problems (missing files)."""
    probs = []
    for label, p in paths:
        if not Path(p).exists():
            probs.append(f"missing {label}: {p}")
    return probs


def merge_report(path, update):
    """Merge `update` into a JSON report so checks can run in separate invocations."""
    rep = C.load_json(path, {}) or {}
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(rep.get(k), dict):
            rep[k].update(v)
        else:
            rep[k] = v
    C.write_json(path, rep)
    return rep


def dumps(obj):
    return json.dumps(obj, indent=1, default=str)
