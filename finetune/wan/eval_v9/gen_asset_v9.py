#!/usr/bin/env python3
"""musubi wan_generate_video.py with two optional in-memory hooks (musubi itself is unchanged).

  --asset <E.safetensors> [--asset-offset] [--asset-phrase "<text>"]
        Learned asset embedding (asset_embed_v9.py): every encoded prompt containing the registry
        phrase gets E in place of (offset: added to) the phrase rows. Negative prompts and prompts
        without the phrase are untouched. --asset-phrase pins the phrase E was trained on (e.g. the
        pre-2026-10-27 wording) instead of reading the live registry; E's row count must equal its
        token span.
  --x0-probe <dir>
        At the last high-noise step (t >= --timestep_boundary), records the model's clean-video
        estimate x0 = x_t - sigma * v and decodes it next to the final clip, as
        <dir>/probe_<k>.mp4 (k = generation order, i.e. the order of the prompts file). Answers
        "is the phone already turned when the low-noise expert takes over?" (RESEARCH_v9c_fixes E1a).

  --t5-fp32
        Encode prompts with T5 in fp32 (drop --fp8_t5 from the musubi args too). The v9c2 training cache is
        fp32-computed; GPU bf16/fp8 T5 has occasional large per-token errors (RESEARCH_v9c_fixes 6b).

    python gen_asset_v9.py [--asset E [--asset-offset]] [--x0-probe DIR] [--t5-fp32] -- <wan_generate_video args>
"""
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asset_embed_v9 as A  # noqa: E402

MUSUBI = Path(os.environ.get("MUSUBI", "/workspace/musubi-tuner"))


def opt(own, name):
    return own[own.index(name) + 1] if name in own else None


def write_mp4(video, path, fps=24):
    """video: [C, F, H, W] float in [-1, 1]."""
    v = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()
    f, h, w, _ = v.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
                         stdin=subprocess.PIPE)
    p.communicate(v.tobytes())


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    own, rest = argv[:i], argv[i + 1:]
    sys.path.insert(0, str(MUSUBI / "src"))

    asset = opt(own, "--asset")
    if asset:
        from musubi_tuner.wan.modules import t5 as T
        E = load_file(asset)["E"].float()
        offset = "--asset-offset" in own
        tok = A.tokenizer()
        ph_ids = A.token_ids(tok, opt(own, "--asset-phrase") or A.phrase())[:-1]
        assert not offset or E.shape[0] == len(ph_ids), f"E has {E.shape[0]} rows, phrase spans {len(ph_ids)} tokens"
        orig_call = T.T5EncoderModel.__call__

        def call(self, texts, device):
            res = []
            for text, ctx in zip(texts, orig_call(self, texts, device)):
                s, e = A.find_span(A.token_ids(tok, text), ph_ids)
                if e > s:
                    ctx = A.splice([ctx], [(s, e)], E.to(ctx.device), offset)[0]
                    print(f"[asset] spliced E into rows {s}:{e} of a {ctx.shape[0]}-token prompt", flush=True)
                res.append(ctx)
            return res

        T.T5EncoderModel.__call__ = call

    if "--t5-fp32" in own:
        assert "--fp8_t5" not in rest, "--t5-fp32 needs --fp8_t5 removed from the musubi args"
        from musubi_tuner.wan.configs import WAN_CONFIGS
        for cfg in WAN_CONFIGS.values():
            cfg.t5_dtype = torch.float32
        print("[t5] prompts encoded in fp32", flush=True)
    sys.argv = [str(MUSUBI / "src/musubi_tuner/wan_generate_video.py"), *rest]
    import musubi_tuner.wan_generate_video as G

    probe_dir = opt(own, "--x0-probe")
    if probe_dir:
        Path(probe_dir).mkdir(parents=True, exist_ok=True)
        last = max(i for i, x in enumerate(rest) if x == "--timestep_boundary")   # later flags win in argparse
        boundary = float(rest[last + 1])
        boundary = boundary / 1000.0 if boundary > 1 else boundary
        state = {"cur": None, "queue": [], "k": 0}
        S = G.FlowUniPCMultistepScheduler
        orig_step = S.step

        def step(self, model_output, timestep, sample, *a, **kw):
            sigma = float(timestep) / 1000.0
            if sigma >= boundary:                          # overwrite until the last high-noise step
                state["cur"] = (sample - sigma * model_output).squeeze(0).detach().to("cpu", torch.float32)
            elif state["cur"] is not None:                 # first low-noise step: hand-over reached
                state["queue"].append(state["cur"])
                state["cur"] = None
            return orig_step(self, model_output, timestep, sample, *a, **kw)

        S.step = step
        orig_decode = G.decode_latent

        def decode(latent, args, cfg):
            video = orig_decode(latent, args, cfg)
            if state["queue"]:
                x0 = state["queue"].pop(0)
                if latent.dim() == x0.dim() + 1:
                    x0 = x0.unsqueeze(0)
                pv = orig_decode(x0.to(latent.dtype), args, cfg)
                write_mp4(pv, Path(probe_dir) / f"probe_{state['k']:03d}.mp4")
                state["k"] += 1
            return video

        G.decode_latent = decode

    G.main()


if __name__ == "__main__":
    main()
