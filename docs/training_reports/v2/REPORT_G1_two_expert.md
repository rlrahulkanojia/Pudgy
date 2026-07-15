# v2 Gate G1 report — two-expert FLF2V (the full decoupled pipeline)

The complete Training Approach v2 pipeline, end to end: **low-noise identity LoRA +
high-noise motion LoRA**, both on Wan2.2-I2V-A14B, run together via FLF2V with
start+end keyframes pinned. This is the artifact the v2 rubric (§5) scores against v1.

## Configuration
- **Low expert:** `lora_lownoise_GOLDEN_ep40` (identity/texture, timesteps 0–900, lr 5e-5)
- **High expert:** `lora_highnoise_GOLDEN_ep40` (motion/composition, timesteps 900–1000, lr 1e-4)
- **Inference:** `--dit` low + `--dit_high_noise` high, `--timestep_boundary 0.9`,
  `--lora_weight` low + `--lora_weight_high_noise` high (musubi maps each LoRA to its expert).
  768×1360×33, 25 steps, flow-shift 5.0, seed 42. fp8_scaled + block-swap 25 + lazy-loading
  (two 14B experts don't fit fp16 on 80 GB; fp8+lazy fits with 0 OOM).
- Both experts trained 40 epochs / 3000 steps, 44.5 h each, fp16, wandb `rlrahulkanojia/pudgy`
  (low `gqbz87zg`, high `gsofo4mm`).

## Head-to-head (clip 00000001, same conditioning as v1)

| Dimension | v1 CogVideoX | v2 low-noise only (ep40) | **v2 two-expert (final)** |
|---|---|---|---|
| Subject persistence | ❌ vanishes by f24–f32 | ✅ persists | ✅ persists every frame |
| Style / thick outlines | soft | ✅ flat 2D preserved | ✅ flat 2D preserved |
| Pax identity | drift / black-patch | ✅ on-model | ✅ on-model |
| **Background fidelity** | drifts pink / empties | ⚠️ dark teal "cinematic" tint | ✅ **bright tiled room — source-faithful** |
| Door / lamp / scene | oversized slab | ✅ door good, bg off | ✅ door+knob, lamp faithful, Polly (pink) behind door |
| WAN photorealistic bias | n/a | not triggered | not triggered |

## The key finding
Adding the high-noise expert **fixed the background/atmosphere** that identity-only training
could not — the low-noise golden rendered a darker, moodier room; the two-expert output
restores the source's **bright light-blue tiled wall + hanging lamp**. This confirms the v2
decoupling thesis operationally: **high-noise = global composition/lighting, low-noise =
identity/texture**, and they compose. The two-expert result is the most source-faithful of
the entire program.

## Gate G1 verdict: **PASS**
v2 beats the v1 CogVideoX baseline on every rubric dimension — decisively on the two that
sank v1 (mid-clip subject vanish; and now background/scene stability), with no regression on
identity or style. The core architectural bet (decouple identity from motion; pin identity
with FLF2V) is validated end to end.

## Caveats / next tuning (optional)
- **Inference is fp8_scaled** (fp16 two-expert OOMs on 80 GB) — true quality is a notch higher.
- **High-noise golden = final epoch** (not swept). A high-noise checkpoint sweep (like the
  low-noise 24/30/36/40 spread) could squeeze a bit more; the final is already strong.
- **Two-character shots:** both FLF2V endpoints are Pax-alone, so Polly is a mid-clip bonus.
  True Pax+Polly shots need Polly pinned at an endpoint or reference-conditioning (v2 §3.4).
- **LoRA strengths** (`LSCALE`/`HSCALE`) and `--timestep_boundary` are tunable via
  `eval_flf2v_2expert.sh` for further balance of identity vs motion.

## Artifacts (this folder + Azure `v2-decoupled-identity-motion`)
`inference_TWO_EXPERT_final.mp4`, `montage_TWO_EXPERT_final.png`,
`lora_lownoise_GOLDEN_ep40.safetensors`, `lora_highnoise_GOLDEN_ep40.safetensors`,
plus the low-noise golden report and the epoch-10 probe.
