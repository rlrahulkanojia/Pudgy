#!/usr/bin/env python3
"""expv1 G0.2 — conditioning check (PLAN §4). One training step per first-frame layout plus
one sampling step of the generator, with the DiT's input dumped per latent frame.

    python eval/cond_check.py build      # tiny corpus: 2 clips x {17, 21, 25} frames, 512^2
    python eval/cond_check.py cache      # musubi latents (VAE as directory AND as file) + text
    python eval/cond_check.py latents    # G0.2-L: frame counts, musubi vs Kandinsky first latent
    python eval/cond_check.py train --layout stock|A|A1|B     # 1 step, dump
    python eval/cond_check.py gen        # 1 sampling step of kandinsky5_generate_video.py, dump
    python eval/cond_check.py report     # signatures + pass/fail -> report.json

Layouts (MODEL_NOTES §2.3-2.4, corrected on the box 2026-10-12):
  stock  musubi trainer as shipped (--i2v_mode first)
  A      official inference: clean first-frame latent in the noisy input at frame 0,
         visual_cond all zero, mask 1 at frame 0
  A1     musubi generator: A plus visual_cond[0] = the first-frame latent
  B      noisy input untouched, visual_cond[0] = first frame, mask 1 at frame 0 only
A, A1 and B come from patches/expv1-k5-i2v-conditioning.patch (--i2v_layout).

Pass (PLAN §4 G0.2): the patched trainer's layout matches the generator's exactly — A1 vs
`gen` — frame 0 is out of the loss, and the musubi latents match Kandinsky's own encode.
The stock dump confirms or refutes MODEL_NOTES §2.4 (last frame at latent index 1, mask on
every frame); it is recorded, not gated.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("EXPV1_WORK", "/workspace/expv1"))
G = WORK / "eval" / "phase0" / "cond_check"
DATA = WORK / "g02"
RAW = Path(os.environ.get("PUDGY_DATA", "/workspace/Data")) / "raw"
MUSUBI = Path(os.environ.get("MUSUBI_DIR", "/workspace/musubi-tuner"))
K5_REPO = Path(os.environ.get("K5_REPO", "/workspace/kandinsky-5"))
M = Path(os.environ.get("MODELS", "/workspace/models"))
DIT = M / "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s/model/kandinsky5pro_i2v_sft_5s.safetensors"
VAE_DIR = M / "hunyuanvideo-community/HunyuanVideo"
VAE_FILE = VAE_DIR / "vae/diffusion_pytorch_model.safetensors"
QWEN = M / "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP = M / "openai/clip-vit-large-patch14"
TASK = "k5-pro-i2v-5s-sd"
SIZE = 512
LENGTHS = (17, 21, 25)   # 21 must now be REFUSED by the patched cacher (MODEL_NOTES 2.8)
CLIPS = {
    "pax_angry": ("iteration_5/EXPRESSIONS/ANGRY/PAX/PAX_ANGRY_FRONT.mov",
                  "Pax, a short round blue penguin, frowns with lowered brows; angry."),
    "polly_waving": ("iteration_4/LSLTTT-Project/MOTION_CLIPS/MOTION_WAVING/POLLY/POLLY_MOTION_WAVING_FRONT.mov",
                     "Polly, a short round pink penguin, raises one flipper and waves; waving."),
}
STYLE = "2D cartoon animation in the Pudgy Penguins style, thick clean black outlines, flat pastel colors."
SHOT = "Medium shot, eye level, front view. Fixed camera, locked-off shot, the character stays the same size. Plain white background."
NEG = ("photorealistic, 3D render, realistic lighting, gradient shading, soft shadows, textured background, "
       "blurry, distorted, deformed, low quality, worst quality")
PY = sys.executable


def sh(cmd, log):
    print("+", " ".join(map(str, cmd)), flush=True)
    with open(log, "w") as f:
        p = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=MUSUBI)
    if p.returncode:
        sys.exit(f"failed ({p.returncode}); see {log}")


# --------------------------------------------------------------------------------- build
def build():
    for n in LENGTHS:
        d = DATA / f"f{n}" / "video"
        d.mkdir(parents=True, exist_ok=True)
        for cid, (rel, cap) in CLIPS.items():
            flt = (f"color=c=white:s={SIZE}x{SIZE}:r=24[bg];[0:v]scale={SIZE}:{SIZE}:flags=lanczos,"
                   f"format=rgba[fg];[bg][fg]overlay=shortest=1")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(RAW / rel), "-filter_complex", flt,
                            "-frames:v", str(n), "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
                            str(d / f"{cid}.mp4")], check=True)
            (d / f"{cid}.txt").write_text(f"{STYLE}\n{cap}\n{SHOT}")
            png = DATA / "start" / f"{cid}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            if not png.exists():
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(d / f"{cid}.mp4"), "-frames:v", "1",
                                str(png)], check=True)
        for vae in ("dir", "file"):
            toml = DATA / f"f{n}_{vae}.toml"
            toml.write_text(f'''[general]
resolution = [{SIZE}, {SIZE}]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
video_directory = "{d}"
cache_directory = "{DATA / f'cache_f{n}_{vae}'}"
target_frames = [{n}]
frame_extraction = "head"
source_fps = 24.0
''')
    print("built", DATA)


# --------------------------------------------------------------------------------- cache
def cache():
    for n in LENGTHS:
        for vae, path in (("dir", VAE_DIR), ("file", VAE_FILE)):
            toml = DATA / f"f{n}_{vae}.toml"
            cmd = [PY, "src/musubi_tuner/kandinsky5_cache_latents.py", "--dataset_config", toml, "--vae", path]
            if vae == "dir":   # the expv1 path: Kandinsky's class + frame 0 encoded as the generator does
                cmd.append("--first_frame_image_encode")
            log = G / f"cache_latents_f{n}_{vae}.log"
            if vae == "dir" and n == 21:
                print("+", " ".join(map(str, cmd)), "(expected to be refused)", flush=True)
                with open(log, "w") as f:
                    rc = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=MUSUBI).returncode
                refused = rc != 0 and "latent frames for 21 frames" in log.read_text()
                (G / "f21_refused.json").write_text(json.dumps({"returncode": rc, "refused": refused}))
                continue
            sh(cmd, log)
        sh([PY, "src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py", "--dataset_config",
            DATA / f"f{n}_dir.toml", "--text_encoder_qwen", QWEN, "--text_encoder_clip", CLIP, "--batch_size", 2],
           G / f"cache_text_f{n}.log")
        # the "file" caches share the text outputs
        for f in (DATA / f"cache_f{n}_dir").glob("*_te.safetensors"):
            dst = DATA / f"cache_f{n}_file" / f.name
            if not dst.exists():
                dst.symlink_to(f)


def load_cache(n, vae, cid):
    from safetensors.torch import load_file
    fs = sorted((DATA / f"cache_f{n}_{vae}").glob(f"{cid}*.safetensors"))
    fs = [f for f in fs if not f.name.endswith("_te.safetensors")]
    if len(fs) != 1:
        raise FileNotFoundError(f"{n} {vae} {cid}: {fs}")
    return load_file(str(fs[0]))


# --------------------------------------------------------------------------------- latents
def latents():
    """G0.2-L: (1) does musubi's cache keep every 4k+1 length whole (G0.1 found Kandinsky's
    encode keeps only 8k+1 above 17)? (2) is musubi's cached first-frame latent the one the
    official pipeline would give the model (i2v_pipeline: image encode, opt_tiling=False)?"""
    import importlib.util
    import torch
    from PIL import Image
    spec = importlib.util.spec_from_file_location("k5_vae", K5_REPO / "kandinsky" / "models" / "vae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vae = mod.AutoencoderKLHunyuanVideo.from_pretrained(str(VAE_DIR), subfolder="vae",
                                                         torch_dtype=torch.float16).cuda().eval()
    sf = vae.config.scaling_factor
    out = {"scaling_factor": sf, "items": []}
    for cid in CLIPS:
        img = np.asarray(Image.open(DATA / "start" / f"{cid}.png").convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1)[None, :, None].to("cuda", torch.float16) / 127.5 - 1
        with torch.no_grad():
            ref = (vae.encode(x, opt_tiling=False).latent_dist.mode() * sf)[0, :, 0].float().cpu()
        for n in LENGTHS:
            for v in ("dir", "file"):
                if v == "dir" and n == 21:
                    continue   # refused by the patched cacher
                c = load_cache(n, v, cid)
                keys = {k: list(t.shape) for k, t in c.items()}
                lat = next(t for k, t in c.items() if k.startswith("latents_") and "image" not in k).float()
                img_lat = next((t for k, t in c.items() if k.startswith("latents_image")), None)
                first = (img_lat if img_lat is not None else lat)[:, 0].float()
                rel = float((first - ref).norm() / ref.norm())
                rel_mean = float((lat[:, 0] - ref).norm() / ref.norm())
                out["items"].append({
                    "clip": cid, "frames": n, "vae": v, "keys": keys,
                    "latent_frames": int(lat.shape[1]), "expected_latent_frames": (n - 1) // 4 + 1,
                    "rel_err_first_vs_k5_image_encode": round(rel, 4),
                    "rel_err_video_frame0_vs_k5": round(rel_mean, 4),
                    "latent_std": round(float(lat.std()), 4), "k5_ref_std": round(float(ref.std()), 4)})
                print(out["items"][-1], flush=True)
    G.mkdir(parents=True, exist_ok=True)
    (G / "latents.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------------- dumps
def install_dit_hook(dest, extra=None):
    """Wrap DiffusionTransformer3D.forward: save the first call's input x (F, H, W, 33)."""
    import torch
    from musubi_tuner.kandinsky5.models import dit as D
    orig = D.DiffusionTransformer3D.forward
    state = {"done": False}

    def fwd(self, x, *a, **k):
        if not state["done"]:
            state["done"] = True
            payload = {"x": x.detach().float().cpu()}
            if extra:
                payload.update({kk: (vv.detach().float().cpu() if torch.is_tensor(vv) else vv)
                                for kk, vv in extra.items()})
            torch.save(payload, dest)
            print(f"[g02] dumped DiT input {tuple(x.shape)} -> {dest}", flush=True)
        return orig(self, x, *a, **k)

    D.DiffusionTransformer3D.forward = fwd


def train(layout, n=17):
    out = G / f"train_{layout}"
    out.mkdir(parents=True, exist_ok=True)
    extra = {}
    from musubi_tuner import kandinsky5_train_network as T
    orig_call = T.Kandinsky5NetworkTrainer.call_dit

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps,
                 network_dtype, **kw):
        extra.update({"latents": latents[0], "noise": noise[0], "noisy": noisy_model_input[0],
                      "timesteps": timesteps[0], "latents_image": batch.get("latents_image")[0]})
        res = orig_call(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input,
                        timesteps, network_dtype, **kw)
        import torch
        torch.save({"pred_shape": list(res.pred.shape), "target_shape": list(res.target.shape)},
                   out / "loss_frames.pt")
        return res

    T.Kandinsky5NetworkTrainer.call_dit = call_dit
    install_dit_hook(out / "dit_input.pt", extra)
    sys.argv = ["kandinsky5_train_network.py", "--task", TASK, "--dit", str(DIT), "--vae", str(VAE_DIR),
                "--text_encoder_qwen", str(QWEN), "--text_encoder_clip", str(CLIP),
                "--dataset_config", str(DATA / f"f{n}_dir.toml"), "--mixed_precision", "bf16", "--sdpa",
                "--gradient_checkpointing", "--network_module", "networks.lora_kandinsky",
                "--network_dim", "32", "--network_alpha", "32", "--optimizer_type", "AdamW8Bit",
                "--learning_rate", "1e-4", "--max_train_steps", "1", "--timestep_sampling", "shift",
                "--discrete_flow_shift", "5.0", "--seed", "42", "--output_dir", str(out / "out"),
                "--output_name", f"g02-{layout}", "--max_data_loader_n_workers", "1"]
    if layout != "stock":
        sys.argv += ["--i2v_layout", layout]
    os.chdir(MUSUBI)
    T.main()   # not runpy: run_module re-executes the module, bypassing the patched class


def gen(cid="pax_angry", n=17):
    out = G / "gen"
    out.mkdir(parents=True, exist_ok=True)
    install_dit_hook(out / "dit_input.pt")
    cap = (DATA / f"f{n}" / "video" / f"{cid}.txt").read_text()
    sys.argv = ["kandinsky5_generate_video.py", "--task", TASK, "--dit", str(DIT), "--vae", str(VAE_DIR),
                "--text_encoder_qwen", str(QWEN), "--text_encoder_clip", str(CLIP), "--prompt", cap,
                "--negative_prompt", NEG, "--image", str(DATA / "start" / f"{cid}.png"),
                # musubi's --frames counts LATENT frames (video = 4(L-1)+1): --frames 17 gave 65 frames
                "--width", str(SIZE), "--height", str(SIZE), "--frames", str((n - 1) // 4 + 1), "--steps", "1",
                "--seed", "42",
                "--sdpa", "--output", str(out / "gen.mp4")]
    os.chdir(MUSUBI)
    runpy.run_module("musubi_tuner.kandinsky5_generate_video", run_name="__main__")


# --------------------------------------------------------------------------------- report
def signature(x, clean=None):
    """Per latent frame: what the DiT is handed. x is (F, H, W, 33) = noisy 16 | cond 16 | mask 1."""
    noisy, cond, mask = x[..., :16], x[..., 16:32], x[..., 32]
    F = x.shape[0]
    sig = {"frames": F,
           "noisy_std": [round(float(noisy[f].std()), 3) for f in range(F)],
           "cond_nonzero": [bool(cond[f].abs().max() > 0) for f in range(F)],
           "mask": [round(float(mask[f].mean()), 3) for f in range(F)]}
    if clean is not None:
        c = clean.permute(1, 2, 0) if clean.shape[0] == 16 else clean
        sig["noisy_f0_is_clean_first"] = bool(np.allclose(noisy[0].numpy(), c.numpy(), atol=2e-2))
        sig["cond_f0_is_clean_first"] = bool(np.allclose(cond[0].numpy(), c.numpy(), atol=2e-2))
    else:
        sig["noisy_f0_equals_cond_f0"] = bool(np.allclose(noisy[0].numpy(), cond[0].numpy(), atol=2e-2))
    return sig


def report():
    import torch
    rep = {"layouts": {}}
    for d in sorted(G.glob("train_*")):
        dump = torch.load(d / "dit_input.pt")
        lf = torch.load(d / "loss_frames.pt")
        li = dump["latents_image"]               # (16, 2, h, w): cached [first, last]
        s = signature(dump["x"], clean=li[:, 0])
        cond = dump["x"][..., 16:32]
        s["cond_f1_is_last_frame"] = bool(cond.shape[0] > 1 and np.allclose(
            cond[1].numpy(), li[:, 1].permute(1, 2, 0).numpy(), atol=2e-2))
        s["loss_frames"] = lf["target_shape"][1]
        s["timestep"] = round(float(dump["timesteps"].flatten()[0]), 4)
        rep["layouts"][d.name.removeprefix("train_")] = s
    if (G / "gen" / "dit_input.pt").exists():
        rep["gen"] = signature(torch.load(G / "gen" / "dit_input.pt")["x"])
        nf = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", str(G / "gen" / "gen.mp4")],
                            capture_output=True, text=True).stdout.strip()
        rep["gen"]["video_frames_out"] = int(nf) if nf.isdigit() else None
    lat = json.loads((G / "latents.json").read_text()) if (G / "latents.json").exists() else None
    rep["latents"] = lat

    def layout_key(s):
        return (s["cond_nonzero"], s["mask"])

    checks = {}
    a1, g = rep["layouts"].get("A1"), rep.get("gen")
    if a1 and g:
        checks["generator_outputs_17_frames"] = g.get("video_frames_out") == 17
        checks["A1_matches_generator"] = (layout_key(a1) == layout_key(g) and a1["noisy_f0_is_clean_first"]
                                          and a1["cond_f0_is_clean_first"] and g["noisy_f0_equals_cond_f0"])
    for k in ("A", "A1", "B"):
        s = rep["layouts"].get(k)
        if s:
            F = s["frames"]
            want_cond = [True] + [False] * (F - 1) if k in ("A1", "B") else [False] * F
            ok = (s["mask"] == [1.0] + [0.0] * (F - 1) and s["cond_nonzero"] == want_cond
                  and s["loss_frames"] == F - 1 and s["noisy_f0_is_clean_first"] == (k in ("A", "A1")))
            checks[f"{k}_layout_as_designed"] = ok
    st = rep["layouts"].get("stock")
    if st:
        rep["stock_confirms_notes_2_4"] = {"last_frame_at_index_1": st["cond_f1_is_last_frame"],
                                           "mask_on_every_frame": all(m == 1.0 for m in st["mask"]),
                                           "frame0_in_loss": st["loss_frames"] == st["frames"]}
    if lat:
        whole = all(i["latent_frames"] == i["expected_latent_frames"] for i in lat["items"] if i["vae"] == "dir")
        # bf16 (musubi) vs fp16 (Kandinsky) image encode differ by ~0.8%; whole-clip frame 0 by ~4.4%
        close = all(i["rel_err_first_vs_k5_image_encode"] <= 0.02 for i in lat["items"] if i["vae"] == "dir")
        ref = json.loads((G / "f21_refused.json").read_text()) if (G / "f21_refused.json").exists() else {}
        checks["musubi_dir_vae_keeps_all_cached_lengths"] = whole
        checks["musubi_refuses_truncating_length_f21"] = bool(ref.get("refused"))
        checks["musubi_first_latent_matches_k5"] = close
    rep["checks"] = checks
    rep["pass"] = bool(checks) and all(checks.values())
    (G / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("checks", "pass")} | {"stock": rep.get("stock_confirms_notes_2_4")},
                     indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "cache", "latents", "train", "gen", "report"])
    ap.add_argument("--layout", default="stock", choices=["stock", "A", "A1", "B"])
    a = ap.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    {"build": build, "cache": cache, "latents": latents, "gen": gen, "report": report,
     "train": lambda: train(a.layout)}[a.cmd]()


if __name__ == "__main__":
    main()
