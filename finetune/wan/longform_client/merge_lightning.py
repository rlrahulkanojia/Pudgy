"""Bake the lightx2v Wan2.2 I2V 4-step distill LoRA (Wan2.2-Distill-Loras, "1022") into copies of
the base DiTs, so the speed test applies it in full.

musubi merges only lora_down/lora_up pairs. This LoRA also carries full-weight deltas (.diff:
norms, patch embedding, head), bias deltas (.diff_b) and modulation deltas (.diff_m), which
ComfyUI applies on top. Here everything is merged at strength 1.0 (no alpha keys -> scale 1)
in fp32 and stored back as fp16. The v8 LoRAs then load on top through musubi as before.
Keys for layers Wan2.2 doesn't have (img_emb, *_img: Wan2.1's CLIP branch) are skipped.

python merge_lightning.py   -> /workspace/wan_models/lightx2v/wan2.2_i2v_<expert>_noise_14B_fp16_lightx2v4step1022.safetensors
"""
import collections
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(32)
SRC = Path("/workspace/wan_models/comfy22/split_files/diffusion_models")
LX = Path("/workspace/wan_models/lightx2v")


def merged_path(expert):
    return LX / f"wan2.2_i2v_{expert}_noise_14B_fp16_lightx2v4step1022.safetensors"


def main():
    for expert in ("high", "low"):
        out = merged_path(expert)
        if out.exists():
            print("exists", out.name)
            continue
        with safe_open(SRC / f"wan2.2_i2v_{expert}_noise_14B_fp16.safetensors", "pt") as b:
            sd, meta = {k: b.get_tensor(k) for k in b.keys()}, b.metadata()
        with safe_open(LX / f"wan2.2_i2v_A14b_{expert}_noise_lora_rank64_lightx2v_4step_1022.safetensors", "pt") as f:
            lora = {k.removeprefix("diffusion_model."): f.get_tensor(k) for k in f.keys()}
        done, skipped, rel = collections.Counter(), [], {}

        def add(target, delta, kind):
            if target not in sd:
                skipped.append(target)
                return
            w = sd[target]
            assert w.shape == delta.shape, (target, w.shape, delta.shape)
            rel[target] = (delta.float().norm() / w.float().norm()).item()
            sd[target] = (w.float() + delta.float()).to(w.dtype)
            done[kind] += 1

        for k, t in lora.items():
            if k.endswith(".lora_down.weight"):
                m = k.removesuffix(".lora_down.weight")
                add(m + ".weight", lora[m + ".lora_up.weight"].float() @ t.float(), "lora")
            elif k.endswith(".lora_up.weight"):
                continue
            elif k.endswith(".diff_b"):
                add(k.removesuffix(".diff_b") + ".bias", t, "diff_b")
            elif k.endswith(".diff_m"):
                add(k.removesuffix(".diff_m") + ".modulation", t, "diff_m")
            elif k.endswith(".diff"):
                add(k.removesuffix(".diff") + ".weight", t, "diff")
            else:
                raise SystemExit(f"unknown LoRA key {k}")
        print(expert, dict(done), "skipped", len(skipped), sorted({s.split(".")[0] + ".." + s.split(".")[-2] for s in skipped}))
        for t in ("blocks.0.self_attn.q.weight", "blocks.0.self_attn.q.bias", "blocks.0.norm3.weight",
                  "blocks.0.modulation", "patch_embedding.weight", "head.head.weight"):
            print(f"   |delta|/|W| {t}: {rel.get(t, float('nan')):.4f}")
        save_file(sd, str(out), metadata=meta)
        print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
