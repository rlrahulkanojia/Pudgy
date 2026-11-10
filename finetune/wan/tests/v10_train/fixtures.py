"""A synthetic /workspace-like training set for cache_mark_v10.py / train_pudgy_v10.sh tests.

Cache files are written with musubi v0.3.4's own writers (save_latent_cache_wan,
save_text_encoder_output_cache_wan), so names, keys and metadata are what the box produces. T5 rows
match the real umT5 token count of each caption. Captions are real scene captions from the repo
(finetune/wan/scenes/v9b_535_scene_windows.json): 3 naming the phone, 5 not, plus 3 replay-style.
Layout: scene_f21 (4 plain + 2 phone), scene_f81 (1 plain + 1 phone), replay_f57 at 1024^2 (3).
"""
import json
import os
from pathlib import Path

import common
import torch
import asset_embed_v9 as AE
from musubi_tuner.dataset.cache_io import save_latent_cache_wan, save_text_encoder_output_cache_wan
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from safetensors.torch import save_file

TOK = AE.tokenizer()
PH = AE.phrase()
PH_N = len(AE.token_ids(TOK, PH)) - 1
WINDOWS = json.loads((common.FW / "scenes/v9b_535_scene_windows.json").read_text())["windows"]
PHONE_CAPS = [w["caption"] for w in WINDOWS if PH in w["caption"]][:3]
PLAIN_CAPS = [w["caption"] for w in WINDOWS if "ifin" not in w["caption"].lower()][:5]
REPLAY_CAPS = ["A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel "
               f"colors, showing Pax, a short round blue penguin, {a} on a plain white background."
               for a in ("waving one flipper", "jumping up and down", "walking to the right")]


def write_te(path, video_path, caption, gen=None):
    it = ItemInfo(video_path, caption, (0, 0))
    it.text_encoder_output_cache_path = str(path)
    save_text_encoder_output_cache_wan(it, torch.randn(len(AE.token_ids(TOK, caption)), 32, generator=gen).to(torch.bfloat16))


def make_ws(root):
    """Returns (dataset toml, cache root, E5-shaped zero embedding file)."""
    root = Path(root)
    data, cache = root / "data_v10", root / "wan_cache/latents_v10"
    data.mkdir(parents=True)
    plan = {("scene", 21): PLAIN_CAPS[:4] + PHONE_CAPS[:2], ("scene", 81): PLAIN_CAPS[4:] + PHONE_CAPS[2:],
            ("replay", 57): REPLAY_CAPS}
    lines = ["[general]", "resolution = [848, 480]", "batch_size = 1", "enable_bucket = true", "bucket_no_upscale = true", ""]
    g = torch.Generator().manual_seed(0)
    for (kind, n), caps in plan.items():
        jl = data / f"dataset_v10_{kind}_f{n}.workspace.jsonl"
        cd = cache / f"{kind}_f{n}"
        reso = (1024, 1024) if kind == "replay" else (848, 480)
        with open(jl, "w") as f:
            for i, c in enumerate(caps):
                vp = f"/workspace/data_v10/clips/{kind}_{i:03d}_f{n}.mp4"
                f.write(json.dumps({"video_path": vp, "caption": c}) + "\n")
                stem = Path(vp).stem
                it = ItemInfo(f"{stem}_00000-{n:03d}.mp4", c, reso, frame_count=n,
                              latent_cache_path=str(cd / f"{stem}_00000-{n:03d}_{reso[0]:04d}x{reso[1]:04d}_wan.safetensors"))
                F = (n - 1) // 4 + 1
                os.makedirs(cd, exist_ok=True)
                save_latent_cache_wan(it, torch.randn(16, F, 4, 4, generator=g), None,
                                      torch.randn(20, F, 4, 4, generator=g), None)
                write_te(cd / f"{stem}_wan_te.safetensors", vp, c, g)
        lines += ["[[datasets]]", f'video_jsonl_file = "{jl}"', f'cache_directory = "{cd}"', f"target_frames = [{n}]",
                  'frame_extraction = "head"', "num_repeats = 1"]
        if kind == "replay":
            lines.append("resolution = [1024, 1024]")
        lines.append("")
    cfg = data / "dataset_config_v10.workspace.toml"
    cfg.write_text("\n".join(lines))
    e5 = root / "e5.asset.safetensors"
    save_file({"E": torch.zeros(PH_N, 32)}, str(e5))
    return cfg, cache, e5
