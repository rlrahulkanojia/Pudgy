# Base-model exploration (Gate G0 → Phase 1)

Research toward the base migration in [`Training_Approach_v2.md`](v2/Training_Approach_v2.md) §1.1 / §4.
Goal: a base that (a) handles **768×1360 portrait natively**, (b) has an **8× VAE** (our round-trip
proved outline fidelity matters — see [`phase0_diagnostics.md`](phase0_diagnostics.md)), (c) supports
**identity-pinning** (FLF2V / keyframe interpolation), and (d) is trainable with mature LoRA tooling.

## Candidates

### Wan2.2-I2V-A14B  ★ primary
- **MoE:** ~27B total, **14B active/step** (two experts: low-noise + high-noise). Apache-2.0.
- **VAE:** Wan2.1-class **8× spatial** (good for thick outlines / flat fills).
- **Resolution:** 480/720P; for I2V the generated size follows the **input image aspect** → portrait
  768×1360 supported (no CogVideoX-style rotary cap).
- **Identity-pinning:** FLF2V (first-last-frame) via community ComfyUI workflows; Wan2.1 shipped an
  official FLF2V-14B.
- **Two-expert LoRA (per v2 §1.3):** low-noise expert = identity/texture, high-noise = motion.
- **VRAM:** model card cites **≥80 GB** for comfortable single-GPU inference; community musubi-tuner
  LoRA training fits **16–24 GB** with **fp8_base + blocks_to_swap**.

### AniSora V3 (Bilibili Index-anisora)  ★ strong alternative (best style fit)
- Built on **Wan2.1-14B + CogVideoX-5B**, RLHF-tuned; Apache-2.0; HF weights + a 948-clip dataset.
- **Anime/2D-native** — closest to our flat-pastel, thick-outline target.
- **Native keyframe interpolation** (generation conditioned on one/multiple frames at arbitrary
  positions) — this *is* the v2 plan's identity-decoupling primitive, built in. Also single-image
  I2V + lip-sync.
- Caveat: some reports cite 360p/5s throughput — **max resolution to verify** before committing.

### Wan2.2-TI2V-5B  — low-VRAM, but wrong VAE
- 5B, single model; 720P@24fps; runs on **8–12 GB** (RTX 4090-class).
- **VAE is 16× (4×16×16)** — the v2 plan flags this as softening outlines/flat fills, and our
  round-trip shows outline fidelity is exactly what our style needs. **Avoid for final quality**;
  usable only as a fast prototyping base.

## Tooling
- **musubi-tuner** (kohya) — the standard Wan2.2 LoRA trainer; supports `fp8_base`, `fp8_scaled`,
  `blocks_to_swap` for memory. **diffusion-pipe** is the alternative. CogVideoX is not first-class in
  either → migration also means switching trainer.

## Feasibility on this box (A100-40 GB)
- **A14B / AniSora (14B):** LoRA-trainable only with **fp8 + block-swapping** — feasible but tight
  and slow. An **80 GB H100** is strongly preferred.
- **5B:** trains comfortably here, but the 16× VAE trades away outline quality.

## Recommendation
Primary **Wan2.2-I2V-A14B** (8× VAE, portrait, mature tooling, two-expert LoRA); run **AniSora V3**
in parallel through Gate G1 for its anime-native style + built-in keyframe interpolation. Keep the
5B only as a fast prototyping base, not for final delivery.

**Next concrete step:** download *just the VAE weights* for Wan 8× and Wan2.2 16× and extend
`inference/vae_roundtrip.py` to compare them against CogVideoX on our clips — cheap, and it decides
the VAE question before any multi-hour setup.

## Sources
- Wan2.2-I2V-A14B — https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- Wan2.2-TI2V-5B — https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Wan 2.2 VRAM guide — https://willitrunai.com/blog/wan-2-2-vram-requirements
- Wan2.2 FLF2V workflow — https://www.mimicpc.com/workflows/wan22-14b-flf2v-first-and-last-frame-img2video
- musubi-tuner Wan2.2 training — https://github.com/kohya-ss/musubi-tuner/discussions/455
- AniSora (Index-anisora) — https://huggingface.co/IndexTeam/Index-anisora · https://github.com/bilibili/Index-anisora
- AniSora paper — https://arxiv.org/html/2412.10255v4
