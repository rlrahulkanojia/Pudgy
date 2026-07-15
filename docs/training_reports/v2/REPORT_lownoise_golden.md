# v2 low-noise LoRA — final report & golden checkpoint

**Run:** `pudgy-wan22-a14b-lownoise` (wandb `rlrahulkanojia/pudgy/runs/gqbz87zg`) ·
40 epochs / 3000 steps · 44.5 h on A100-80GB · final loss **0.00709** (monotonic
decline, no overfit blowup) · 20 checkpoints (epochs 2–40).

**Base:** Wan2.2-I2V-A14B low-noise expert · **LoRA:** rank16/α32, all-linear+MLP,
lr 5e-5, timesteps 0–900, fp16, flow-shift 5.0.

## 🏆 Golden checkpoint: **epoch 40 (final)** — `lora_lownoise_GOLDEN_ep40.safetensors`
(saved by musubi as `pudgy-wan22-a14b-lownoise.safetensors`, no epoch suffix.)

## Golden-window comparison (FLF2V eval, clip 00000001, 768×1360×33, fp8_scaled inference)

| Ckpt | Door / scene geometry | Polly (2nd char) | Outlines | Verdict |
|---|---|---|---|---|
| ep24 | good (hinged door) | ✅ peeks in | clean | balanced |
| ep30 | weaker (floating slab) | ✅✅ clearest | clean | best 2-char, weak door |
| ep36 | ✅ best arched door + depth | ✗ (pink light only) | crisp | best single-char scene |
| **ep40 🏆** | ✅ **best** arched door + doorway | ✅ **clear Pax + Polly** | **crispest** | **most complete** |

Epoch 40 combines ep36's clean scene geometry with ep24/30's two-character rendering,
at the sharpest outlines — the loss decline to the end sharpened the model rather than
collapsing it. Montages for all four are in this folder (`montage_ep{24,30,36}.png`,
`montage_GOLDEN_ep40.png`).

## Result vs v1 (Gate G1 evidence)
| Dimension | v1 CogVideoX | **v2 low-noise golden (ep40)** |
|---|---|---|
| Subject persistence | ❌ vanished by f24–f32 | ✅ **present every frame**, lands on end keyframe |
| Style / outlines | soft | ✅ **flat pastel + thick outlines preserved** (no WAN 3D-ify) |
| Pax identity | drifted / black-patched | ✅ on-model, stable |
| Two characters | n/a | ✅ Pax + Polly both render on-model |
| Door / scene | oversized morphing slab | ✅ proper arched door + doorway depth |

**The core v2 bet is validated:** FLF2V endpoint-pinning eliminates v1's mid-clip vanish,
and WAN's photorealistic bias did not damage the flat 2D style.

## Known limitations (this is the identity LoRA only)
- **Background:** a slightly darker teal "cinematic" tint vs the source's bright tile —
  mild WAN atmosphere creep, consistent across checkpoints.
- **Motion:** door open/close and body motion are still low-noise-only; the **high-noise
  (motion) LoRA** is the next piece (training next, fp16-first).
- **FLF2V + Polly:** both eval endpoints are Pax-alone, so the model is pinned to Pax at
  t=0 and t=1; Polly appearing mid-clip is a bonus, not guaranteed. True two-character
  shots need Polly pinned at an endpoint (or reference-conditioning, v2 §3.4).
- **Inference is fp8_scaled** (full-res fp16 OOMs at load) — the real quality is a notch
  above what these montages show.

## Artifacts (this folder) + Azure
`inference_GOLDEN_ep40_flf2v.mp4`, `montage_GOLDEN_ep40.png`, `lora_lownoise_GOLDEN_ep40.safetensors`,
the three spread montages, plus the epoch-10 set. All weights (epochs 2–40) + logs + this
output are mirrored to Azure `v2-decoupled-identity-motion` (account `pudgytraining`).

## Next
1. **High-noise (motion) LoRA** — fp16 first, fp8_base fallback. Cleans up door/body motion.
2. Full two-expert FLF2V (low identity + high motion) → re-eval at full-res fp16.
3. Formal **Gate G1** scoring vs v1 on the §5 rubric.
