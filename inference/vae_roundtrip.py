#!/usr/bin/env python
"""
Phase 0.1 diagnostic — VAE round-trip test (Training_Approach_v2 §0.1).

Encodes then decodes a real Pax/Polly clip through a video VAE and measures how much the
2D art (thick outlines, flat pastel fills) survives. If the VAE can't reconstruct the style,
that's a hard image-quality ceiling no amount of training fixes -> it forces the base choice.

    python inference/vae_roundtrip.py --clip /workspace/training_dataset/train/00000001.mp4
"""
import argparse, os, sys
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_report as R  # _ssim, build helpers


def read_clip(path, max_frames, size):
    cap = cv2.VideoCapture(path); frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok: break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if size: f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model_path", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "finetune/models/CogVideoX1.5-5B-I2V"))
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1360)
    ap.add_argument("--out", default="finetune/output_dir/pudgy-lora-v1/report/vae_roundtrip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = read_clip(args.clip, args.frames, (args.width, args.height))
    print(f"loaded {len(src)} frames at {args.width}x{args.height} from {os.path.basename(args.clip)}")

    from diffusers import AutoencoderKLCogVideoX
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae",
                                                 torch_dtype=torch.bfloat16).to("cuda")
    vae.enable_slicing(); vae.enable_tiling()

    x = torch.from_numpy(np.stack(src)).float() / 127.5 - 1.0     # T,H,W,3 in [-1,1]
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to("cuda", torch.bfloat16)  # 1,3,T,H,W
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode()
        recon = vae.decode(lat).sample
    print(f"latent shape {tuple(lat.shape)}  (spatial 8x, temporal 4x compression)")
    recon = ((recon.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    recon = recon.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()    # T,H,W,3

    n = min(len(src), recon.shape[0])
    psnrs = [psnr(src[i], recon[i]) for i in range(n)]
    ssims = [R._ssim(src[i], recon[i]) for i in range(n)]
    print(f"PSNR mean {np.mean(psnrs):.2f} dB (min {np.min(psnrs):.2f})  |  "
          f"SSIM mean {np.mean(ssims):.4f} (min {np.min(ssims):.4f})")

    # Side-by-side montage: for a few frames, source | recon | 4x-abs-diff, plus a zoom crop
    picks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    tiles = []
    for i in picks:
        s, r = src[i], recon[i]
        diff = np.clip(np.abs(s.astype(int) - r.astype(int)) * 4, 0, 255).astype(np.uint8)
        # zoom a face-region crop (top-center where the penguin usually is)
        H, W = s.shape[:2]; cy, cx, ch, cw = int(H*0.18), int(W*0.5), int(H*0.22), int(W*0.35)
        crop_s = cv2.resize(s[cy:cy+ch, cx-cw//2:cx+cw//2], (W, int(W*ch/cw)))
        crop_r = cv2.resize(r[cy:cy+ch, cx-cw//2:cx+cw//2], (W, int(W*ch/cw)))
        row = np.concatenate([s, r, diff], axis=1)                      # full frames
        zoom = np.concatenate([crop_s, crop_r, np.zeros_like(crop_s)], axis=1)  # outline zoom
        tile = np.concatenate([row, zoom], axis=0)
        cv2.putText(tile, f"f{i}  src | recon | 4x-diff  (bottom: outline zoom)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=0)
    montage = os.path.join(args.out, "vae_roundtrip.png")
    cv2.imwrite(montage, cv2.cvtColor(cv2.resize(sheet, (sheet.shape[1]//2, sheet.shape[0]//2)),
                                      cv2.COLOR_RGB2BGR))
    # also write the reconstructed clip
    from diffusers.utils import export_to_video
    from PIL import Image
    export_to_video([Image.fromarray(recon[i]) for i in range(n)],
                    os.path.join(args.out, "recon.mp4"), fps=16)
    print(f"wrote {montage} and recon.mp4")


if __name__ == "__main__":
    main()
