# v2 training report — low-noise LoRA @ epoch 10 (interim probe)

**Base:** Wan2.2-I2V-A14B (low-noise expert) · **LoRA:** rank16/α32, all-linear+MLP,
lr 5e-5, timesteps 0–900, fp16 · **Checkpoint:** epoch 10 / step 750 (of a planned 40 ep).
This is an **early probe** (¼ through training), stopped deliberately to validate the
v2 architecture before committing more compute.

## Artifacts in this folder
| File | What |
|---|---|
| `inference_ep10_flf2v.mp4` / `_montage.png` | Generated video — FLF2V, start(f0)+end(f32) keyframes pinned, low-noise LoRA @ scale 1.0 |
| `training_source_00000001.mp4` / `_montage.png` | Ground-truth source clip (the conditioning clip) |
| `keyframe_start_f0.png` / `keyframe_end_f32.png` | The two FLF2V endpoints (extracted from the source) |
| `lora_lownoise_epoch10.safetensors` | The epoch-10 LoRA weights (293 MB) |

## How it was generated
`eval_flf2v.sh`, task `i2v-A14B`, 768×1360×33, 25 steps, flow-shift 5.0, seed 42.
Inference ran **fp8_scaled + `--fp8_t5` + `--blocks_to_swap 20` + `--vae_cache_cpu`**
(full-res fp16 inference OOMs at load — `model.to(dtype)` transiently doubles the 28 GB
DiT). Quality tier fp8_scaled ≈ just below fp16, so the archived video slightly
understates the true fp16 quality. ~14 min on the A100-80GB.

## Result vs the v1 baseline (the whole point of v2)

| Dimension | v1 CogVideoX (FINDINGS.md) | **v2 low-noise @ ep10** |
|---|---|---|
| **Subject persistence** | ❌ vanished by f24–f32 → empty room | ✅ **present in every frame**; ends on the pinned end keyframe |
| **Style / outlines** | (soft, but VAE not the ceiling) | ✅ **flat pastel + thick black outlines preserved** — WAN's photorealistic bias did NOT 3D-ify it |
| **Pax identity** | drifted, black-patched, off-color | ✅ on-model blue Pax, stable across the clip |
| **Motion / composition** | melting; door = oversized slab | ⚠️ door geometry morphs frame-to-frame; slight body/flipper wobble |
| **Background** | drifted (pink), emptied out | ⚠️ grayish tint vs the source's clean light-blue tile |

**Verdict:** the two failures that sank v1 — **mid-clip subject vanish** and the
**photorealistic-bias risk** flagged as the research's #1 concern — are **both resolved**
at only epoch 10. This validates the core v2 bet: FLF2V endpoint pinning bounds identity
drift to the interpolation, and it holds.

## Remaining issues → expected fixes
- **Door morphing / proportion wobble** = motion/composition → the domain of the
  **high-noise (motion) LoRA**, not yet trained. Low-noise alone handles identity/texture.
- **Background tint / softness** → more low-noise epochs (this is ¼-trained) + **full-res
  fp16 inference** instead of the fp8_scaled probe.
- **Polly (pink) barely emerges** → expected: both FLF2V endpoints are Pax-blue keyframes,
  so the interpolation correctly stays Pax. Two-character shots need reference-conditioning
  (v2 §3.4), tested later.

## Next
1. Full low-noise run to 40 epochs (restarted, wandb-tracked `rlrahulkanojia/pudgy`, `--save_state`).
2. Pick the golden checkpoint by eye across the run.
3. Train the **high-noise motion LoRA** (fp16 first, fp8_base fallback), then full two-expert FLF2V.
4. Re-eval at full-res fp16 and score vs v1 at **Gate G1** (v2 §5 rubric).
