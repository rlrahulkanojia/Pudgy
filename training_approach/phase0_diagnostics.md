# Phase 0 diagnostics — results log

Runs against `Training_Approach_v2.md` §0. Artifacts (videos/montages) are under
`finetune/output_dir/pudgy-lora-v1/report/` (gitignored; local only).

## 0.1 — VAE round-trip test ✅ DONE

**Question:** is the 2D image quality (thick outlines, flat pastel fills) limited by the VAE, or by training/generation?

**Method:** encode→decode a real clip (`train/00000001.mp4`, Pax+door+Polly) through the
CogVideoX1.5 VAE at full **768×1360**, no diffusion. Tool: `inference/vae_roundtrip.py`.

**Result (two clips — one Pax-led, one Polly-led):**

| Clip | Latent | PSNR mean (min) | SSIM mean (min) |
|---|---|---|---|
| Pax `00000001` | `1×16×9×170×96` | **38.9 dB** (32.5) | **0.996** (0.993) |
| Polly `00000065` | `1×16×9×170×96` | **38.3 dB** (33.7) | **0.995** (0.991) |

- 8× spatial, 4× temporal compression.
- Visual (`report/vae_roundtrip*/vae_roundtrip.png`, mirrored to
  `assets/vae_roundtrip_{pax,polly}.png`): outlines crisp, flat fills clean, **pink Polly + rosy
  cheeks preserved**; the 4×-amplified diff is nearly black — only faint edge softening.

**Conclusion:** **the VAE is NOT the image-quality ceiling.** A CogVideoX-class 8× VAE
represents the Pudgy style near-losslessly. Therefore the poor output is a **generation /
diffusion** problem — the 432×768 portrait cap (CogVideoX rotary-grid limit) + the diffusion
prior + attention-only recipe — **not** representation.

**Implications for the plan:**
- Base migration is justified by the **resolution cap + diffusion quality**, not a VAE ceiling.
- We're free to keep an **8× VAE** (Gate G0's "prefer 8× spatial" is satisfied by CogVideoX/Wan-2.1
  class; the 16× Wan-2.2-5B VAE remains the one to be wary of for outline softening).

## 0.3 — LoRA-targeting fix (all-linear + MLP, α=2r) — READY TO RUN

Runnable on the current base/data with no changes needed beyond the LoRA config:
`get_linear_layers()` already exists (`train_cogvideox_image_to_video_lora.py:70`) and the
all-linear `LoraConfig` path is present but commented (~line 1011). Switch target_modules to
all-linear, set `lora_alpha = 2*rank`, retrain on the same 75 clips → apples-to-apples vs the
attention-only run. ~10 h on the A100-40GB.

## 0.2 — Caption A/B — READY TO RUN

Rebuild `metadata.json` captions as *rare-token identity + variable-only action* (drop the dense
200-char identity descriptions), retrain, compare. ~10 h.
