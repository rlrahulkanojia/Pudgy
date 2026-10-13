#!/usr/bin/env python3
"""Generate expv1's Tier-2 `core` suite (PLAN §7) with musubi's generator code, loading
each model ONCE (C4) instead of once per clip.

Per clip this is kandinsky5_generate_video.main() step for step (the same functions, in
the same order, with the same arguments): the caption and the negative are encoded with the
task's text config (Qwen max length 256, video template, packed by the mask); the start
image goes through get_first_frame_from_image (its own encode, as the generator does);
generate_sample_latents_only samples with the task's steps / CFG / scheduler_scale (50, 5,
10 at SD) in first-frame layout A (G0.3: what the checkpoint expects; the musubi patch adds
i2v_layout to the generator, and G0.2's gen_A check proves it matches trainer layout A); decode_latents and
save_videos_grid write the mp4. musubi's --frames counts LATENT frames (MODEL_NOTES §2.11),
so a clip of n video frames samples (n - 1) / 4 + 1 latent frames; the output frame count
and the 512^2 size are asserted on every clip (MODEL_NOTES §2.7).

Suite: 13 labels x 2 characters x {FRONT (trained angle), QF2_R (held out)} x seeds, at the
H2H window (f13 motion / f17 expression), start frame = neutral frame 0 on white
(build_refs.py). Output: EVAL/gen/<config>/core/<char>_<label>_<view>_f<N>_s<seed>.mp4 —
the naming of v8's core clips, so compare_v8.py pairs them directly.

    python eval/gen_core.py --config base [--seeds 42 43 44] [--limit 2]
    python eval/gen_core.py --config expv1@550
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expv1_common as C  # noqa: E402


def items(seeds, views=("FRONT", C.HOLDOUT_ANGLE)):
    out = []
    for char in C.CHARS:
        for label in C.LABELS:
            n = C.H2H_WINDOW[C.kind_of(label)]
            for view in views:
                for s in seeds:
                    out.append({"char": char, "label": label, "kind": C.kind_of(label), "view": view,
                                "frames": n, "seed": s, "caption": C.caption(char, label, view, n),
                                "start": str(C.REFS / "start" / f"{char.lower()}_{view}_white.png"),
                                "name": f"{char.lower()}_{label}_{view}_f{n}_s{s}.mp4"})
    return out


def packed(enc_out, attention_mask):
    """kandinsky5_generate_video.main: drop padding by the mask."""
    import torch
    te = enc_out["text_embeds"].to("cpu")
    pe = enc_out["pooled_embed"].to("cpu")
    m = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
    m = m.bool().flatten().cpu()
    if m.shape[0] != te.shape[0]:
        m = m[m] if m.sum().item() == te.shape[0] else torch.ones((te.shape[0],), dtype=torch.bool)
    return te[m], pe


def probe_frames(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,width,height", "-of", "json", str(p)],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["nb_read_frames"]), int(s["width"]), int(s["height"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="base", help="base | expv1@<step> | file:<lora>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests only)")
    a = ap.parse_args()

    import torch
    sys.path.insert(0, str(C.MUSUBI / "src"))
    from safetensors.torch import load_file
    from musubi_tuner.kandinsky5.configs import TASK_CONFIGS
    from musubi_tuner.kandinsky5.generation_utils import (generate_sample_latents_only, decode_latents,
                                                          get_first_frame_from_image)
    from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
    from musubi_tuner.kandinsky5_train_network import Kandinsky5NetworkTrainer
    from musubi_tuner.hv_train_network import clean_memory_on_device
    from musubi_tuner.hv_generate_video import save_videos_grid
    from musubi_tuner.networks import lora_kandinsky
    from dcls import lora_path

    out_dir = C.EVAL / "gen" / a.config.replace("/", "_").replace(":", "_") / "core"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items(a.seeds) if not (out_dir / it["name"]).exists()]
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        print(f"nothing to do in {out_dir}"); return
    task = TASK_CONFIGS[C.TASK]
    device = torch.device("cuda")
    steps = a.steps or task.num_steps
    guidance = task.guidance_weight
    sched = task.scheduler_scale or 1.0

    # stage 1: every prompt + the negative, text encoder loaded once
    te = get_text_embedder(SimpleNamespace(
        qwen=SimpleNamespace(checkpoint_path=str(C.QWEN), max_length=task.text.qwen_max_length),
        clip=SimpleNamespace(checkpoint_path=str(C.CLIP), max_length=task.text.clip_max_length)),
        device=device, quantized_qwen=False)
    emb = {}
    for t in sorted({it["caption"] for it in todo} | {C.NEG}):
        enc, _, am = te.encode([t], type_of_content="video")
        emb[t] = packed(enc, am)
    te.to("cpu") if hasattr(te, "to") else None
    del te
    gc.collect(); clean_memory_on_device(device)

    # stage 2: DiT (+ LoRA) and the VAE, once
    trainer = Kandinsky5NetworkTrainer()
    trainer.task_conf = task
    trainer.blocks_to_swap = 0
    trainer._vae_checkpoint_path = str(C.VAE_DIR)
    loader_args = SimpleNamespace(fp8_base=False, fp8_scaled=False, fp8_fast=False, blocks_to_swap=0,
                                  disable_numpy_memmap=False, override_dit=None, sdpa=True, flash_attn=False,
                                  flash3=False, sage_attn=False, xformers=False)
    dit = trainer.load_transformer(accelerator=SimpleNamespace(device=device), args=loader_args,
                                   dit_path=str(C.DIT), attn_mode=task.attention.type, split_attn=False,
                                   loading_device=device, dit_weight_dtype=None)
    dit.eval(); dit.requires_grad_(False)
    lora = None
    if a.config != "base":
        lora = lora_path(a.config)
        sd = load_file(str(lora))
        net = lora_kandinsky.create_arch_network_from_weights(1.0, sd, unet=dit, for_inference=True)
        net.merge_to(None, dit, sd, device=device, non_blocking=True)
        clean_memory_on_device(device)
    vae = trainer._load_vae_for_sampling(SimpleNamespace(), device=device)
    conf_ns = SimpleNamespace(model=task, metrics=SimpleNamespace(scale_factor=task.scale_factor))
    neg_te, neg_pe = emb[C.NEG]
    log = []
    for i, it in enumerate(todo, 1):
        t0 = time.time()
        lat_frames = C.latent_frames(it["frames"])
        with torch.no_grad():
            _, ff, _ = get_first_frame_from_image(it["start"], vae, device, max_area=512 * 768, divisibility=16)
            first = ff[:1]
            shape = (1, lat_frames, int(first.shape[1]), int(first.shape[2]), task.dit_params.in_visual_dim)
            te_, pe_ = emb[it["caption"]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lat = generate_sample_latents_only(
                    shape=shape, dit=dit, text_embeds=te_, pooled_embed=pe_, attention_mask=None,
                    null_text_embeds=neg_te, null_pooled_embed=neg_pe, null_attention_mask=None,
                    first_frames=first, num_steps=steps, guidance_weight=guidance, scheduler_scale=sched,
                    seed=it["seed"], device=device, conf=conf_ns, progress=False, i2v_mode="first",
                    i2v_layout="A")
            images = decode_latents(lat, vae, device=device, batch_size=1, num_frames=lat_frames)
        dest = out_dir / it["name"]
        save_videos_grid(images.permute(0, 4, 1, 2, 3).float().cpu() / 255.0, str(dest), rescale=False, n_rows=1)
        nf, w, h = probe_frames(dest)
        if (nf, w, h) != (it["frames"], C.SIZE, C.SIZE):
            dest.rename(dest.with_suffix(".bad.mp4"))
            sys.exit(f"{it['name']}: got {nf} frames {w}x{h}, expected {it['frames']} at {C.SIZE}^2")
        log.append({**it, "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(todo)}] {it['name']}  {time.time() - t0:.0f}s", flush=True)
    meta_p = out_dir / "meta.json"
    meta = C.load_json(meta_p, {}) or {}
    meta.setdefault("clips", {}).update({x["name"]: x for x in log})
    meta.update({"config": a.config, "lora": str(lora) if lora else None, "dit": str(C.DIT), "task": C.TASK,
                 "steps": steps, "guidance": guidance, "scheduler_scale": sched, "negative": C.NEG,
                 "layout": "A (G0.3)", "text_max_length": task.text.qwen_max_length})
    C.write_json(meta_p, meta)


if __name__ == "__main__":
    main()
