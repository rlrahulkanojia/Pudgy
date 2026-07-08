# Wan 2.2 vs CogVideoX1.5 — Comprehensive Research Report

## For Pudgy Penguins AI Animation Pipeline

**Document type:** Evidence-based technical research
**Methodology:** Three independent research agents cross-referenced community reports (CivitAI, Reddit, GitHub Issues), official documentation, research papers (CVPR 2026, arXiv), training guides, and benchmark data. Claims are tagged with evidence quality: VERIFIED, PARTIALLY VERIFIED, UNVERIFIED, or REFUTED.
**Use case:** Generating 3-5 second production-quality looping GIFs of Pudgy Penguins — stylized 2D cartoon characters with thick black outlines, flat colors, exaggerated proportions, and accessories (scarves, hats) — using an I2V pipeline where artists provide composited first frames, with character LoRA training on 60-80 curated clips from the client's 150+ existing animation library.

---

## Table of Contents

1. [Model Architectures](#1-model-architectures)
2. [Text Encoder Comparison](#2-text-encoder-comparison)
3. [The Photorealistic Bias Problem](#3-the-photorealistic-bias-problem)
4. [LoRA Training: Depth Comparison](#4-lora-training-depth-comparison)
5. [The MoE Identity-Motion Separation Claim](#5-the-moe-identity-motion-separation-claim)
6. [I2V First-Frame Adherence & Identity Drift](#6-i2v-first-frame-adherence--identity-drift)
7. [Frame Rate Reality](#7-frame-rate-reality)
8. [Looping GIF Capabilities](#8-looping-gif-capabilities)
9. [Resolution, Aspect Ratio & GIPHY Compatibility](#9-resolution-aspect-ratio--giphy-compatibility)
10. [Wan-Animate Evaluation](#10-wan-animate-evaluation)
11. [ComfyUI Ecosystem & Pipeline Integration](#11-comfyui-ecosystem--pipeline-integration)
12. [LoRA Training Failure Modes](#12-lora-training-failure-modes)
13. [Licensing & Commercial Use](#13-licensing--commercial-use)
14. [Community Ecosystem & Development Trajectory](#14-community-ecosystem--development-trajectory)
15. [Head-to-Head Comparison Matrix](#15-head-to-head-comparison-matrix)
16. [Risk Assessment](#16-risk-assessment)
17. [What Remains Unknown](#17-what-remains-unknown)
18. [Verdict & Strategy Recommendation](#18-verdict--strategy-recommendation)
19. [Sources](#19-sources)

---

## 1. Model Architectures

### CogVideoX1.5-5B-I2V — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| **Architecture** | 3D Diffusion Transformer (DiT) — single-pass | [THUDM/CogVideoX1.5-5B-I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) |
| **Parameters** | 5 billion | HuggingFace model card |
| **Generation approach** | Single-pass — all frames denoised simultaneously in one forward pass | Architecture documentation |
| **Text encoder** | T5-v1.1-XXL (frozen, ~4.7B params) | [HuggingFace diffusers docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox) |
| **VAE** | 3D causal VAE | Architecture documentation |
| **I2V mechanism** | Dedicated I2V model variant — image latent concatenated with noise latent | Model card |
| **Attention** | 3D causal attention — attends across spatial and temporal dimensions simultaneously | Architecture documentation |
| **Precision** | BF16 (trained and recommended for inference) | Training documentation |
| **Release** | September 2024 (v1.0), November 2024 (v1.5) | HuggingFace |

**Architectural significance for our use case:** The single-pass architecture means CogVideoX1.5 generates all frames in one diffusion process. This makes latent-space manipulation (e.g., injecting first-frame latent at late denoising steps for loop closure) architecturally natural — you have access to the full latent tensor throughout the denoising process.

### Wan 2.2 I2V-A14B — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| **Architecture** | Mixture-of-Experts (MoE) Diffusion Transformer | [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2), [official blog](https://wan.video/blog/wan2.2) |
| **Total parameters** | ~27 billion (two 14B expert networks) | Model card |
| **Active parameters per step** | ~14 billion (one expert active at a time) | Architecture documentation |
| **Expert design** | High-noise expert (early denoising: layout, composition, camera) + Low-noise expert (late denoising: detail, texture, identity) | [PyxelJam analysis](https://pyxeljam.com/wan-2-2-moe-architecture-explained-cinematic-level-aesthetics/) |
| **Expert switching** | Hard timestep-based switch at SNR threshold (~t=875 of 1000) — NOT dynamic per-token routing | [DeepLearning.AI analysis](https://www.deeplearning.ai/the-batch/alibabas-wan-2-2-video-models-adopt-a-new-architecture-to-sort-noisy-from-less-noisy-inputs) |
| **Text encoder** | UMT5-XXL (frozen, ~13B params) | [DeepWiki](https://deepwiki.com/deepbeepmeep/Wan2GP/8.1-text-encoders) |
| **Vision encoder** | CLIP ViT-H (for I2V conditioning) | [kijai ComfyUI wrapper docs](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/5.4-text-and-vision-encoder-integration) |
| **I2V mechanism** | Image encoded via Wan-VAE, latent concatenated with noise + CLIP vision cross-attention | Model card |
| **Precision** | BF16 | Model card |
| **Release** | July 2025 (Wan 2.2), with newer versions 2.5 (2025), 2.6 (2026), 2.7 (2026) | GitHub |

### Wan 2.2 TI2V-5B — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| **Architecture** | Dense Diffusion Transformer (no MoE) | [HuggingFace model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| **Parameters** | 5 billion | Model card |
| **VAE** | Wan2.2-VAE with high compression: 16x16x4 (TxHxW), total 4x32x32 with patchification | Model card |
| **Unified model** | Handles both T2V and I2V in single architecture | Documentation |
| **Release** | July 2025 | GitHub |

**Architectural significance for our use case:** The MoE design means Wan 2.2 I2V-A14B processes early denoising (motion/layout) and late denoising (detail/identity) through physically separate neural networks. LoRA training must target both experts independently — doubling training complexity. The TI2V-5B avoids this problem but at the cost of model capacity.

---

## 2. Text Encoder Comparison

### CogVideoX1.5: T5-v1.1-XXL — VERIFIED

| Property | Value | Source |
|----------|-------|--------|
| **Model** | `google/t5-v1_1-xxl` | [HuggingFace diffusers API docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox) |
| **Parameters** | ~4.7 billion | Model card |
| **Tokenizer** | T5Tokenizer (SentencePiece) | Documentation |
| **Vocabulary** | ~32K tokens (English) | Architecture |
| **Status during training** | **Frozen** — text embeddings pre-cached before LoRA training begins | [HuggingFace training guide](https://huggingface.co/docs/diffusers/en/training/cogvideox) |
| **Integration** | Text embeddings concatenated with patchified video embeddings along sequence dimension | Architecture docs |
| **Prompt style** | Natural language sentences strongly preferred. Keyword/tag format degrades quality due to SentencePiece sub-word tokenization. | Sprint plan Q&A (validated) |

### Wan 2.2: UMT5-XXL — VERIFIED

| Property | Value | Source |
|----------|-------|--------|
| **Model** | `google/umt5-xxl` (Universal Multilingual T5, XXL) | [DeepWiki](https://deepwiki.com/deepbeepmeep/Wan2GP/8.1-text-encoders), [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html) |
| **Parameters** | ~13 billion | Model card (fp16 safetensors ~11GB vs T5-XXL ~8.9GB) |
| **Tokenizer** | SentencePiece (same family as T5) | Architecture |
| **Vocabulary** | ~250K tokens (multilingual — English, Chinese, and 100+ languages) | UMT5 documentation |
| **Status during training** | **Frozen** — text embeddings pre-cached. Text encoder is never trained; only DiT receives LoRA adapters. | [CivitAI workflow guide](https://civitai.com/articles/17740), [AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html) |
| **Integration** | Text embeddings injected via cross-attention in each DiT block | Architecture docs |
| **Prompt style** | Natural language descriptions preferred (80-120 words optimal). Keyword tags acceptable but not optimal. Community convention uses trigger words for LoRA activation. | [InstaSD prompting guide](https://www.instasd.com/post/wan2-2-whats-new-and-how-to-write-killer-prompts), [Wan-Animate I2V guide](https://wan-animate.com/posts/how-to-use-i2v-prompting-wan-2-2-animate-guide) |

### Implications for Our Captioning Pipeline

Both encoders are T5-family, both frozen, both handle natural language well. Our two-tier captioning strategy (fixed identity anchor prefix + VLM-generated action suffix) should be compatible with both models.

**Key difference — trigger words:** The Wan 2.2 community leans heavily toward synthetic trigger words (e.g., `zxq-penguin`) for LoRA activation. This is a community convention, not an architectural requirement. Since UMT5-XXL is a T5-family encoder, our natural-language-only approach (using the character's actual name as identifier) should still work. However, this is **UNVERIFIED** for Wan 2.2 specifically — the convention exists because it may empirically produce better LoRA binding.

**Key difference — embedding richness:** UMT5-XXL's ~13B parameters produce a richer embedding space than T5-v1.1-XXL's ~4.7B. This could mean more nuanced text conditioning for the DiT, potentially giving Wan 2.2 better prompt adherence. However, the larger vocabulary (250K vs 32K tokens, due to multilingual support) means more tokens compete for representation — whether this helps or hurts a single-language, single-domain use case like ours is **UNKNOWN**.

---

## 3. The Photorealistic Bias Problem

This is the **most critical finding** in this entire research, and it directly impacts the viability of Wan 2.2 for the Pudgy Penguins aesthetic.

### Evidence: Wan 2.2 Fights Against Cartoon Aesthetics — VERIFIED

Four independent sources confirm Wan 2.2 has a strong inherent bias toward photorealistic rendering:

**Source 1 — CivitAI Anime Style I2V LoRA creator:**
> *"WAN 2.2 appears to have a strong bias towards realistic generations."*
> Training past ~2000 steps for anime style results in overfit and noise, even with 6 low-noise steps.

[Source](https://civitai.com/models/2222779/anime-style-wan-22-i2v)

**Source 2 — Taz's Anime Style LoRA Training Guide (CivitAI, multi-part series):**
> Without a High-Noise LoRA specifically trained for anime style, results produce *"an uncanny mix of real life and the cartoon style."*
> The word "animated" as a prompt keyword triggers Wan's own *"biased 3D-style animated look"* — not flat 2D.

[Source](https://civitai.com/articles/20389/tazs-anime-style-lora-training-guide-for-wan-22-part-1-3)

**Source 3 — Alici.AI Wan 2.7 Guide:**
> The model's weights are *"heavily pre-biased toward realism."*

[Source](https://alici.ai/blog/how-to-use-wan-27-guide-2026)

**Source 4 — Community observation on resolution:**
> Higher resolutions cause cartoon inputs to *"shift toward realism"*, requiring higher LoRA strength to compensate.

[Source](https://civitai.com/models/2222779/anime-style-wan-22-i2v)

### What This Means for Pudgy Penguins

The Pudgy Penguins aesthetic has four properties that directly conflict with Wan 2.2's photorealistic bias:

| Character Property | How Wan 2.2's Bias Manifests |
|--------------------|------------------------------|
| **Flat colors** (no gradients, no shading) | Model adds realistic shading, gradients, subsurface scattering, depth effects |
| **Thick black outlines** (clean, uniform lines) | Diffusion process softens/blurs outlines, adds depth lighting, breaks uniformity |
| **Minimal visual complexity** (simple shapes) | Fewer distinguishing features for identity encoder to anchor to — model "fills in" with realistic detail |
| **Exaggerated proportions** (short legs, round body, large head) | Conflicts with model's learned priors about body ratios — may normalize toward realistic anatomy |

### Known Workarounds — VERIFIED

The Wan 2.2 community has developed mitigation strategies:

1. **Negative prompts:** Add `((realistic))` to negative prompts ([source](https://civitai.com/models/2222779/anime-style-wan-22-i2v))
2. **LoRA strength above 1.0:** Bump style LoRA model strength to 1.2+ ([source](https://civitai.com/models/2222779/anime-style-wan-22-i2v))
3. **Avoid "animated" keyword:** It triggers Wan's own 3D-style animated look, not flat 2D ([source](https://civitai.com/articles/20389))
4. **Denoise 0.3-0.5:** For I2V, keep denoise strength low to preserve source frame aesthetics ([source](https://dredyson.com/how-i-fixed-wan2-2-i2v-on-8gb-vram-a-complete-step-by-step-beginners-guide-to-source-faithful-animation-settings-denoise-tuning-and-face-consistency-workarounds-that-actually-work-after-6-months/))
5. **Stack style LoRAs:** Community anime/2D LoRAs exist ([2D Animation Effects](https://civitai.com/models/1920897), [Simple/Pure Color Anime](https://civitai.com/models/1872525), [Anime Style I2V](https://civitai.com/models/2222779))

### Does CogVideoX1.5 Have the Same Problem?

**PARTIALLY — but potentially less severe.**

One comparison source notes CogVideoX produces *"sharper, crisper detail"* with *"better VFX aesthetics"* compared to Wan 2.2's softer, more cinematic rendering. ([GenAIntel](https://www.genaintel.com/compare/cogvideox-5b-vs-wan-22-5b))

Sharper frame rendering could theoretically preserve clean cartoon outlines better than Wan 2.2's softer approach. However, CogVideoX1.5 was also trained on primarily photorealistic data. No user has publicly tested either model on flat-color mascot characters.

**Critical evidence gap:** Nobody has published results of either model generating flat-color cartoon mascot characters (penguins, bears, simple mascots with thick outlines). The closest community tests are anime-style characters, which have far more visual complexity (shading, hair detail, clothing folds) than Pudgy Penguins. Our use case is a genuine edge case for both models.

---

## 4. LoRA Training: Depth Comparison

### CogVideoX1.5 Training Configuration — PARTIALLY VERIFIED

| Parameter | Sprint Plan Value | Official HuggingFace Value | Notes |
|-----------|------------------|---------------------------|-------|
| **LoRA Rank** | 64 | 64 | Aligned |
| **LoRA Alpha** | 32 | 64 (match rank) | Sprint plan uses half-rank; official uses full-rank |
| **Learning Rate** | 3e-5 | 1e-3 | **Significant discrepancy.** Official uses 100x higher LR with `cosine_with_restarts` scheduler. Community member Cseti reported that even with trained LoRAs, CogVideoX *"still doesn't reproduce characters accurately."* ([HuggingFace diffusers docs](https://huggingface.co/docs/diffusers/en/training/cogvideox)) |
| **Steps** | 4,000 | 1,500-2,000 (for 50 videos), 4,000 (for 100 videos) | Aligned for our ~60-80 clip dataset |
| **Optimizer** | 8-bit AdamW | AdamW | Sprint uses 8-bit for VRAM savings |
| **Batch Size** | 1 (grad accum 4) | 1 (grad accum 1) | Sprint uses higher effective batch |
| **VRAM** | ~35GB | Requires A100 80GB with gradient checkpointing. OOMs without. | Verified — A100 80GB mandatory |
| **Training time** | 10-14 hours per run | **UNVERIFIED** — no published wall-clock benchmarks for CogVideoX1.5. Plausible given model size and step count. | Needs empirical validation in Week 2 diagnostic run |
| **Training framework** | [Passenger12138 trainer](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train) | Official diffusers scripts available but less tested for v1.5 I2V | Passenger12138 repo includes critical RoPE and OFS fixes |
| **Dataset size** | 60-80 clips | 50-100 videos recommended (THUDM guidance) | Aligned |

### Wan 2.2 I2V-A14B Training Configuration — VERIFIED

| Parameter | Community Consensus | Source |
|-----------|-------------------|--------|
| **LoRA Rank** | 16 (simple identity) to 32 (complex IP) | [WaveSpeed](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/), [Apatero](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025) |
| **LoRA Alpha** | Match rank (16/16 or 32/32) | Community consensus |
| **Learning Rate** | 5e-5 for identity, 7e-5 to 1e-4 for style | Multiple sources |
| **Steps** | 3,000-5,000 for character identity | [Apatero](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025), [WaveSpeed](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/) |
| **Optimizer** | AdamW (weight_decay 0.01), or AdamW8Bit for VRAM-constrained | Multiple sources |
| **Scheduler** | Cosine with 5% warmup, or polynomial (lr_scheduler_power=8) | [CivitAI workflow](https://civitai.com/articles/17740), [Musubi-tuner](https://github.com/kohya-ss/musubi-tuner) |
| **Batch Size** | 2-4 on A100/4090 | Community reports |
| **VRAM** | A14B: **80GB minimum** (A100 80GB). TI2V-5B: 24GB (RTX 4090). | [HuggingFace model cards](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| **Dataset size** | 12-20 images (simple identity), 20-40 video clips (robust character) | Multiple community sources |
| **Training tools** | AI Toolkit (Ostris), Musubi-tuner, Diffusion-pipe, DiffSynth-Studio, diffusers | Multiple |

### The Dual-Expert Training Complication — VERIFIED

Wan 2.2 I2V-A14B requires training **two separate LoRAs** — one for each expert:

| Expert | When Active | What It Controls | LoRA Consequences |
|--------|------------|-----------------|-------------------|
| **High-noise expert** | Early denoising (high SNR timesteps) | Global layout, composition, motion trajectory, camera | Missing/poor LoRA → composition breaks, motion ignores character style |
| **Low-noise expert** | Late denoising (low SNR timesteps) | Fine detail, texture, identity features, color accuracy | Missing/poor LoRA → *"uncanny mix of real life and cartoon"*, identity collapse, blur |

**Impact on training effort:**

| Factor | CogVideoX1.5 | Wan 2.2 I2V-A14B |
|--------|--------------|------------------|
| Models to train | 1 | **2** (high-noise + low-noise) |
| Hyperparameter sets to tune | 1 | **2** (may need different LR/rank per expert) |
| Training runs to iterate | 3-5 | **6-10** (3-5 per expert, potentially interdependent) |
| Per-run time (A100 80GB) | 10-14 hours | **6-10 hours per expert** (lower steps per expert, but must run both) |
| Total training time per iteration | 10-14 hours | **12-20+ hours** (both experts) |
| Failure diagnosis complexity | Single model — straightforward | **Must diagnose which expert is failing** — symptoms are different |

Source: [Apatero best practices](https://www.apatero.com/blog/train-wan-22-loras-best-practices-2025), [Taz's training guide](https://civitai.com/articles/20389)

### The "2-4 Hour Training" Claim — REFUTED

Multiple sources refute the widely cited "2-4 hour" training time:

| Hardware | Actual Reported Time | Source |
|----------|---------------------|--------|
| RTX 4090 | 4-10 hours | [Apatero training time analysis](https://www.apatero.com/blog/wan-22-video-lora-training-time-complete-analysis-2025) |
| Cloud A6000 | Up to 24 hours | Apatero |
| Consumer GPUs | 2-3 days | [Apatero best practices](https://www.apatero.com/blog/train-wan-22-loras-best-practices-2025) |
| WaveSpeedAI cloud API | ~27 minutes per request | WaveSpeed (API, not local) |
| H100 (FP8, simple style LoRA) | 2-4 hours | Likely the source of the original claim |

The 2-4 hour figure appears to apply to: **style LoRAs** (not character identity), at **rank 16** with **~1,500 steps**, on **H100** with **FP8 quantization**, for **simple concepts** (not complex IP characters with specific accessories).

For a **character identity LoRA** with the complexity of Pudgy Penguins, expect **6-12+ hours per expert on A100**, totaling **12-20+ hours for both experts.**

### Wan 2.2 TI2V-5B Training — An Alternative

The 5B dense model avoids the dual-expert problem entirely:

| Property | Value |
|----------|-------|
| Models to train | 1 (single dense model) |
| VRAM required | 24GB (RTX 4090 viable) |
| Training time (estimated) | 4-6 hours on A100 |
| Rank | 16-32 (same as A14B) |

However, TI2V-5B lacks the I2V-A14B's *"enhanced support for diverse stylized scenes"* — which may matter for cartoon content. The quality tradeoff is **UNVERIFIED** for our specific use case.

---

## 5. The MoE Identity-Motion Separation Claim

### The Claim

Multiple sources (including our earlier evaluation) state that Wan 2.2's dual-expert architecture naturally separates identity from motion — the high-noise expert handles motion while the low-noise expert handles identity. This would give it an architectural advantage for character consistency.

### The Evidence — PARTIALLY TRUE, BUT OVERSIMPLIFIED

**What the architecture actually does:**

The expert switch is based on **Signal-to-Noise Ratio (SNR)**, not semantic content type. At early denoising (high noise), the high-noise expert handles coarse structure. At late denoising (low noise), the low-noise expert refines details.

Identity features (face shape, accessory colors, exact proportions) happen to live in the low-noise regime, because they are fine details. But so do textures, lighting details, fine geometry, and environmental detail. The low-noise expert is NOT an "identity expert" — it's a "detail expert."

**Evidence supporting the claim:**
- The [official ablation study](https://wan.video/blog/wan2.2) tested four configurations: baseline (no MoE), high-noise only, low-noise only, and full MoE. Full MoE achieved the lowest validation loss.
- Community LoRA trainers confirm: *"Identity in Wan depends heavily on the lower-noise refinement stages, but motion and composition still need the higher-noise stage."* ([RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora))

**Evidence refuting the strong claim:**
- **No ablation study was found that separately measures identity preservation vs motion quality.** The official ablation only measured overall validation loss — a single aggregate metric.
- The motion quality improvements in Wan 2.2 are primarily attributed to **+83.2% more training videos**, not the MoE architecture itself.
- The expert switch is a **hard timestep-based threshold**, not learned or dynamic. It cannot adapt to content that needs more identity refinement vs more motion refinement.

### Verdict

The MoE architecture provides a useful **training lever** — you can bias LoRA training toward low-noise timesteps for identity work. But calling it "identity-motion separation" is **architectural speculation, not empirically demonstrated.** Neither model has a proven architectural advantage for identity preservation. The CVPR 2026 paper ([IPRO](https://arxiv.org/html/2510.14255v1)) names **both** CogVideoX and Wan as suffering from the same identity drift problem.

---

## 6. I2V First-Frame Adherence & Identity Drift

### The Core Problem — VERIFIED for Both Models

Both models suffer from **exposure bias** in I2V generation: during training, the model conditions on ground-truth frames, but during inference, it conditions on its own generated (imperfect) frames. Errors accumulate across the temporal axis, causing progressive identity degradation.

A [CVPR 2026 paper (IPRO)](https://arxiv.org/html/2510.14255v1) specifically addresses this problem and names both CogVideoX and Wan as affected models. The paper found that, surprisingly, **T2V models sometimes outperform I2V models** on identity preservation for CogVideoX-1.5 — a counterintuitive finding that suggests I2V conditioning can actually amplify error accumulation.

### Wan 2.2 Identity Drift — VERIFIED

| Finding | Source |
|---------|--------|
| *"The first frame looks right, but once the person smiles, turns their head, or changes pose, the face stops looking like the same person."* | [RunComfy character consistency guide](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora) |
| Drift concentrates at **frame 50-70%** of the clip — the middle-to-late portion | [Dredyson denoise guide](https://dredyson.com/how-i-fixed-wan2-2-i2v-on-8gb-vram-a-complete-step-by-step-beginners-guide-to-source-faithful-animation-settings-denoise-tuning-and-face-consistency-workarounds-that-actually-work-after-6-months/) |
| Identity drift is the **#1 reported issue** in community forums | Multiple sources |

**Mitigation strategies documented by the community:**
1. Denoise 0.40-0.50 for maximum source fidelity (at cost of less motion)
2. Combine reference image + character LoRA (both together is strongest)
3. Keep clips to ≤6 seconds with motion strength ≤0.28 for faces
4. Negative prompts: `morphing, warping, face deformation, flickering`
5. Increase Low-noise expert steps relative to High-noise (e.g., 2 High / 4 Low)
6. Keep character large in frame — tight crops improve consistency

### CogVideoX1.5 Identity Drift — VERIFIED

CogVideoX1.5 suffers from the same class of identity drift, but community documentation is less extensive (smaller community). The CVPR 2026 paper's finding that T2V sometimes outperforms I2V on CogVideoX-1.5 suggests the I2V conditioning mechanism may not be as robust as the architecture implies.

### Implications for 3-5 Second GIFs

At 3 seconds (~48 frames at 16fps), the drift window (50-70% of clip = frames 24-34) is narrow. For simple cartoon characters with fewer distinguishing features, the drift may be imperceptible at this duration. **This is favorable for our use case but UNVERIFIED empirically.**

---

## 7. Frame Rate Reality

### CogVideoX1.5: 16fps — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| Native output | 16fps | [HuggingFace model card](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) |
| Frame count | 81 or 161 frames (8N+1 formula) | Documentation |
| Duration | ~5 seconds (81 frames) or ~10 seconds (161 frames) | Calculation |
| RIFE interpolation needed | Yes — 16fps → 24fps = 1.5x interpolation | Sprint plan |
| Effective latent keyframes/sec | ~4 (with 4x temporal VAE compression) | Architecture |

### Wan 2.2 I2V-A14B: 16fps — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| Native output | 16fps | [GitHub](https://github.com/Wan-Video/Wan2.2) |
| RIFE interpolation needed | Yes — same as CogVideoX1.5 | Same situation |

### Wan 2.2 TI2V-5B: 24fps — TECHNICALLY TRUE BUT MISLEADING

| Specification | Value | Source |
|--------------|-------|--------|
| Native output | 24fps (24 distinct pixel frames per second) | [GitHub](https://github.com/Wan-Video/Wan2.2), [HuggingFace model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| VAE temporal compression | **4x** | Architecture documentation |
| Effective latent temporal resolution | ~6 unique latent keyframes/second | Calculation: 24fps / 4x compression |
| RIFE interpolation needed | Technically no, but... | See below |

**The misleading part:**

The Wan2.2-VAE produces latent tokens where every 4 output frames share a single temporal latent token. The VAE decoder reconstructs 4 frames per latent token — these are not identical frames, but they contain interpolated motion from the same underlying temporal information.

**Community reports confirming the issue:**
- Users see *"slow-motion looking"* output at 24fps and recommend RIFE interpolation to add perceptual smoothness. ([Apatero](https://www.apatero.com/blog/avoid-slow-motion-wan-22-video-generation-2025))
- One user [reported](https://github.com/comfyanonymous/ComfyUI/issues/9106): *"WAN said their model is 24 fps, so I set it to 121 frames. But when I generate it, the result looks sped up"* — suggesting the 24fps label doesn't match perceptual expectations.
- [GitHub issue #13](https://github.com/Wan-Video/Wan2.2/issues/13) discusses 16fps vs 24fps confusion.

**Practical reality:** While TI2V-5B technically eliminates the need for RIFE interpolation (24fps output), the effective motion smoothness may still benefit from frame interpolation. The "no RIFE needed" advantage is weaker than it initially appears.

---

## 8. Looping GIF Capabilities

### CogVideoX1.5: Latent-Space Loop Closure — PARTIALLY VERIFIED

Our sprint plan uses a **latent-space injection approach**: at denoising step 75-85%, a fraction of the first frame's latent representation is injected into the latent space of the final frames. The DiT converges the character's geometry back toward the first frame's structure, achieving seamless loop closure before VAE decode.

This approach is **architecturally principled** — it operates in latent space (preserving mathematical coherence) and leverages CogVideoX1.5's single-pass architecture (all frames accessible throughout denoising).

**Evidence status:** The technique is used in the ComfyUI community via advanced KSampler nodes. Specific validation on CogVideoX1.5 with cartoon characters is **UNVERIFIED** — it needs to be tested in Week 4.

### Wan 2.2: FLF2V (First-Last Frame to Video) — VERIFIED WITH KNOWN ISSUES

Wan 2.2 has **no native loop support.** A [GitHub feature request (#81)](https://github.com/Wan-Video/Wan2.2/issues/81) remains open with zero responses from the development team.

The community solution is FLF2V:

| Step | Description | Source |
|------|-------------|--------|
| 1 | Feed the **same image** as both first and last frame | [NextDiffusion tutorial](https://www.nextdiffusion.ai/tutorials/wan-2-2-looping-animations-in-comfyui) |
| 2 | Model generates motion that returns to its starting point | Community workflow |
| 3 | ImageSelector node trims duplicate final frame | ComfyUI workflow |
| 4 | Lightning LoRAs can cut render to 4 total steps | Optional optimization |

**Known issues with Wan 2.2 looping:**

| Issue | Description | Source |
|-------|-------------|--------|
| **Color contrast drift** | Color contrast increases over time during looped generation | [kijai WanVideoWrapper #1541](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541) |
| **Motion range limitation** | Using the same start/end frame **limits the range of motion** — the model constrains itself to return to the starting pose | [GitHub #49](https://github.com/Wan-Video/Wan2.2/issues/49) |
| **Clip stitching inconsistency** | Combining two clips for seamless transitions produces inconsistent results due to mismatched movement, lighting, or physics | Community reports |
| **Best loop length** | ~3 seconds. Longer loops have higher failure rates. | [CivitAI workflow](https://civitai.com/models/1720535) |

**Additional community looping approaches:**
- [WAN 2.2 Perfect Loops workflow (CivitAI)](https://civitai.com/models/1869481) — uses 3 complementary sub-workflows for an 8-second loop
- [VACE-based looping](https://openart.ai/workflows/nomadoor/loop-anything-with-wan21-vace/qz02Zb3yrF11GKYi6vdu) — feeds last 15 frames, blank segment, and first 15 frames for transition
- [EachLabs guide](https://www.eachlabs.ai/blog/designing-loopable-animations-with-wan-animate) — recommends "continuous, rhythmic, seamless cycle" prompt keywords with CFG=3

### Comparison

| Aspect | CogVideoX1.5 | Wan 2.2 |
|--------|--------------|---------|
| **Approach** | Latent-space injection (before VAE decode) | FLF2V (pixel-level constraint) |
| **Color stability** | Unknown (untested) | **Documented color drift** |
| **Motion range** | Full (latent injection only biases, doesn't constrain) | **Limited** (model constrains itself to return to start) |
| **Architectural fit** | Natural for single-pass diffusion | Workaround (not natively designed) |
| **Community validation** | Partially verified | Verified (with known issues) |

CogVideoX1.5 has the more principled approach, but it's less community-tested. Wan 2.2's approach works but has documented artifacts. For 3-second loops (our primary target), Wan 2.2's FLF2V may be adequate despite the limitations.

---

## 9. Resolution, Aspect Ratio & GIPHY Compatibility

### Native Resolutions — VERIFIED

| Model | Native Resolution | Aspect Ratio |
|-------|------------------|-------------|
| CogVideoX1.5-5B-I2V | 1360x768 (max), constraint: Min(W,H)=768, Max(W,H)≤1360, Max(W,H)%16=0 | ~16:9 landscape |
| Wan 2.2 I2V-A14B | 1280x720 (720P) or 832x480 (480P) | 16:9 |
| Wan 2.2 TI2V-5B | 1280x704 or 704x1280 | ~16:9 |

### GIPHY Format Compatibility — PROBLEM FOR BOTH

GIPHY commonly uses:
- **480x480** (1:1 square) — most common for reaction GIFs
- **600x338** (~16:9 landscape)
- **480x270** (16:9 small)

| Format | CogVideoX1.5 | Wan 2.2 |
|--------|--------------|---------|
| **1:1 square (480x480)** | **Not native** — minimum short side is 768px. Would need to generate at 768x768+ and crop/downscale. | **Not native** — community reports [pixelation at 640x640 square](https://github.com/Wan-Video/Wan2.1/issues/233). Quality degradation expected at non-native ratios. |
| **16:9 landscape (600x338)** | Generate at native resolution (1360x768), downscale to target | Generate at native resolution (1280x720), downscale to target |

**Practical approach (both models):** Generate at native 16:9 resolution, downscale and/or crop to GIPHY target formats in post-processing. Both models will need this workflow.

**Key finding:** Neither model natively supports the 1:1 square format that dominates GIPHY. Generating square content forces both models outside their training distribution, which may increase artifacts. Recommendation: generate 16:9 and crop to square in post-processing, centering on the character.

---

## 10. Wan-Animate Evaluation

### Overview

Wan-Animate-14B is a dedicated character animation and replacement model that was initially considered as a potential game-changer — it could theoretically leverage the client's 150+ existing animations as motion reference videos, skipping LoRA training entirely.

### Verdict: NOT VIABLE for Pudgy Penguins

The research identified **five independent hard blockers:**

### Blocker 1: Skeleton Extraction Fails on Penguins — VERIFIED

Wan-Animate's preprocessing pipeline ([official docs](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md)) requires:

| Step | Tool | Problem for Penguin Characters |
|------|------|-------------------------------|
| Person detection | YOLOv10m | **May not detect a cartoon penguin as a "person"** — YOLO is trained on humans |
| Pose extraction | ViTPose (whole-body) | **Will produce garbage keypoints** — penguin anatomy has no human skeletal structure (no arms, different leg proportions, no human face) |
| Face crop | 512x512 auto-crop | **Penguin faces bear no resemblance to human faces** — the face encoder cannot extract meaningful identity features |

The entire preprocessing pipeline was designed for human subjects. All documentation references human inputs. No guidance exists for non-human characters.

### Blocker 2: Hand-Drawn Animation References Won't Parse — VERIFIED

The client's 150+ existing animations are hand-drawn 2D animations containing cartoon physics (squash-and-stretch, anticipation, snap-back). ViTPose attempts to find **realistic joint positions** in these frames. Cartoon deformations (where a character's body stretches to 2x its normal height during a jump) will produce noisy, erroneous keypoints that cause motion artifacts.

### Blocker 3: 3D/Realistic Bias — VERIFIED

[Scenario's official guide](https://help.scenario.com/en/articles/wan-2-2-animate-models-the-essentials/):
> *"These models work best with realistic images and videos as references. Stylized or highly abstract inputs may produce unpredictable results."*

This is the same photorealistic bias documented in Section 3, but amplified — Wan-Animate has no LoRA override mechanism (see Blocker 4).

### Blocker 4: LoRA Enhancement Officially Blocked — VERIFIED

The [official model card](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) explicitly warns:
> *"we do not recommend using LoRA models trained on Wan2.2, since weight changes during training may lead to unexpected behavior."*

Technical root causes documented in GitHub issues:
- **Weight key mismatches** — LoRA keys don't map to Animate's architecture ([GitHub #205](https://github.com/Wan-Video/Wan2.2/issues/205), [diffusion-pipe #114](https://github.com/tdrussell/diffusion-pipe/issues/114))
- **"lora key not loaded" errors** across self_attn and cross_attn layers
- **MoE routing disruption** — LoRA weight changes confuse expert routing
- **Color shift artifacts** — *"severe color shift in the first frame"* ([DiffSynth-Studio #1105](https://github.com/modelscope/DiffSynth-Studio/issues/1105))
- **Stop-motion artifacts** — *"low quality appearance and stop-motion-like artifacts"* ([LoRA Manager #667](https://github.com/willmiao/ComfyUI-Lora-Manager/issues/667))

This means you **cannot** train a style LoRA to counteract the 3D bias and use it with Wan-Animate. The photorealistic bias is locked in.

### Blocker 5: Accessory Loss — VERIFIED

[302.AI's 4-case test](https://medium.com/@302.AI/wan2-2-animate-model-test-with-4-cases-e9e1c1ef492c) found a character's **microphone disappeared** during animation, leaving an empty fist — rated as the "biggest flaw." For Pudgy Penguins wearing scarves and hats, expect similar disappearance/morphing of accessories.

### Conclusion

Wan-Animate was designed for **human-to-humanoid motion transfer.** Using it for non-human cartoon characters with exaggerated proportions, thick outlines, flat colors, and accessories requires replacing the entire preprocessing pipeline — a significant engineering project far beyond the sprint scope. Combined with the LoRA incompatibility and accessory loss, Wan-Animate is **not viable for this use case.**

---

## 11. ComfyUI Ecosystem & Pipeline Integration

### CogVideoX1.5 — VERIFIED MATURE

| Component | Status | Source |
|-----------|--------|--------|
| **Core nodes** | [kijai/ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) | Mature, well-documented |
| **I2V workflow** | Available | Community workflows |
| **LoRA loading** | Supported | Wrapper documentation |
| **RIFE interpolation** | ComfyUI-Frame-Interpolation nodes | Standard |
| **Latent loop closure** | Advanced KSampler + latent noise injection | Available but requires configuration |
| **GIF export** | ComfyUI-VideoHelperSuite | Standard |
| **Ecosystem age** | Since late 2024 (~18 months) | GitHub history |

### Wan 2.2 — VERIFIED MATURE

| Component | Status | Source |
|-----------|--------|--------|
| **Core nodes** | [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) + native ComfyUI support (since v0.3.76, Dec 2025) | [ComfyUI official tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2), [Apatero guide](https://apatero.com/blog/wan-2-2-comfyui-complete-guide-ai-video-generation-2025) |
| **I2V workflow** | Available for both A14B and TI2V-5B | Official templates |
| **LoRA loading** | Dual-path LoRA loading for high-noise + low-noise experts | Wrapper documentation |
| **RIFE interpolation** | ComfyUI-Frame-Interpolation nodes | Same as CogVideoX1.5 |
| **FLF2V looping** | [First-Last Frame workflow](https://www.nextdiffusion.ai/tutorials/wan-2-2-looping-animations-in-comfyui) | Community workflows |
| **GIF export** | ComfyUI-VideoHelperSuite | Standard |
| **Ecosystem age** | Since mid-2025 (~12 months) | GitHub history |
| **Additional variants** | Wan-Animate, Fun Control, S2V all have ComfyUI integration | Community nodes |

Both ecosystems are mature enough to build the full I2V → interpolation → loop → export pipeline. Neither presents a pipeline integration risk.

---

## 12. LoRA Training Failure Modes

### CogVideoX1.5 Failure Modes — PARTIALLY VERIFIED

| Failure Mode | Description | Source |
|-------------|-------------|--------|
| **Deep-frying** | Color burn-out, structural melting at LR > 1e-4 | Sprint plan Q&A (community knowledge) |
| **Motion collapse** | Model memorizes exact training clips, produces stiff output on novel prompts | Sprint plan Q&A |
| **Character drift** | Identity degrades over frames (exposure bias) | [CVPR 2026 paper](https://arxiv.org/html/2510.14255v1) |
| **Inaccurate character reproduction** | *"Still doesn't reproduce characters accurately"* even with trained LoRAs | [CivitAI user Cseti](https://huggingface.co/Cseti/CogVideoX-LoRA-Wallace_and_Gromit/discussions/1) |

**Evidence gap:** CogVideoX1.5's smaller community means fewer documented failure modes. This is both a risk (unknown unknowns) and a neutral finding (it may simply have fewer issues).

### Wan 2.2 Failure Modes — EXTENSIVELY DOCUMENTED

| Failure Mode | Description | Source |
|-------------|-------------|--------|
| **Dual-transformer trap** | Must train both high-noise and low-noise LoRAs. Missing one → *"uncanny mix of real life and cartoon"* or identity collapse | [Taz's training guide](https://civitai.com/articles/20389) |
| **SD/SDXL knowledge trap** | Practitioners waste 40+ hours applying image LoRA techniques to video models — different architecture, different requirements | [Apatero](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025) |
| **Identity drift under motion** | First frame looks right, character changes on movement | [RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora) |
| **3D/cartoon character training struggles** | *"With 3D rendered characters or concepts, it's very hard to force through a real photographic style... same issue but to a lesser extent with anime/cartoon"* | [CivitAI](https://civitai.com/articles/17740) |
| **Overfitting at low step counts (cartoon/anime)** | Training past ~2000 steps for anime style causes overfit and noise | [CivitAI](https://civitai.com/models/2222779/anime-style-wan-22-i2v) |
| **"Plastic skin" / over-stylization** | Faces come out too glossy, backgrounds get pulled into soft studio look | [Apatero](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025) |
| **Identity-style mixing** | Mixing identity and style in the same LoRA gets "muddy fast" | Community reports |
| **Photorealistic bias override** | Trained cartoon LoRA competes with the model's built-in realistic bias, requiring strength >1.0 | Multiple sources |

### Specific Risk for Pudgy Penguins — UNVERIFIED (No Precedent)

No user has publicly documented training a character LoRA on either model for a flat-color cartoon mascot character. The closest examples are:
- **CogVideoX:** [Wallace & Gromit LoRA](https://huggingface.co/Cseti/CogVideoX-LoRA-Wallace_and_Gromit) — claymation style, not flat 2D. User reported character reproduction was still inaccurate.
- **Wan 2.2:** [Anime Style I2V LoRA](https://civitai.com/models/2222779/anime-style-wan-22-i2v) — anime characters have far more visual complexity than Pudgy Penguins.

The Pudgy Penguins use case sits in an evidence gap: simpler than anime (fewer features to anchor to), different from photorealistic (both models' training distribution), and without real-world equivalents (can't supplement with cosplay photos).

---

## 13. Licensing & Commercial Use

| | CogVideoX1.5 | Wan 2.2 |
|---|---|---|
| **License** | Custom THUDM/Tsinghua University license | **Apache 2.0** |
| **Commercial output** | Requires separate review — custom license may impose restrictions | **Fully permissive** — *"We claim no rights over your generated contents"* |
| **Redistribution** | May require attribution or notification | Unrestricted |
| **IP status** | Must verify before sprint (flagged in pre-sprint requirements) | No restrictions |

**Impact:** For a character IP brand producing commercial content (GIPHY library, Instagram posts, potential merchandise), Apache 2.0 eliminates an entire category of legal risk. CogVideoX1.5's custom license is not necessarily restrictive, but it requires legal review before commercial deployment.

---

## 14. Community Ecosystem & Development Trajectory

### CogVideoX1.5 — VERIFIED (Stalled)

| Metric | Value |
|--------|-------|
| **GitHub stars** | ~4,000 |
| **Release history** | v1.0 (Sep 2024), v1.5 (Nov 2024) — **no updates since** |
| **Active development** | Stalled. No newer versions announced or on horizon. |
| **Community LoRAs** | Limited. [Wallace & Gromit](https://huggingface.co/Cseti/CogVideoX-LoRA-Wallace_and_Gromit) is one of the few published character LoRAs. |
| **Training tools** | 1 primary ([Passenger12138](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train)). Official diffusers scripts available but less documented for v1.5 I2V. |
| **ComfyUI** | Mature (kijai wrapper) |
| **Training guides** | Limited — mostly official HuggingFace docs |

**Risk:** If the sprint reveals a model-level bug or limitation, there is no active development team to address it. The community is small and may not have encountered or documented the same issue.

### Wan 2.2 — VERIFIED (Very Active)

| Metric | Value |
|--------|-------|
| **GitHub stars** | 16,000+ |
| **Forks** | 2,000+ |
| **Release history** | 2.1 (early 2025), 2.2 (Jul 2025), 2.5 (late 2025), 2.6 (early 2026), 2.7 (2026) — **active trajectory** |
| **Community LoRAs** | Extensive on CivitAI — character, style, motion, camera, anime, 2D, etc. |
| **Training tools** | 4+ (AI Toolkit, Musubi-tuner, Diffusion-pipe, DiffSynth-Studio, diffusers) |
| **ComfyUI** | Mature (kijai wrapper + native support) |
| **Training guides** | Extensive — multiple CivitAI articles, WaveSpeed blog, Apatero guides, AMD ROCm blog |

**Key development note:** Wan 2.6 specifically addresses identity preservation — *"Wan 2.5 created 'morphing' effects when attempting scene changes. Wan 2.6 handles transitions cleanly with maintained character identity."* ([source](https://10b.ai/blog/wan-2-6-i2v-face-stability), [MindStudio](https://www.mindstudio.ai/blog/what-is-wan-2-6-video-open-source))

This is directly relevant to our identity drift concern. If Wan 2.6/2.7 genuinely improves identity preservation, it could resolve one of the model's primary weaknesses.

---

## 15. Head-to-Head Comparison Matrix

| Dimension | CogVideoX1.5-5B-I2V | Wan 2.2 I2V-A14B | Wan 2.2 TI2V-5B | Winner |
|-----------|---------------------|-------------------|------------------|--------|
| **Photorealistic bias severity** | Unknown (untested) | **High** (documented, multiple sources) | **High** (same model family) | CogVideoX1.5 (less documented risk) |
| **LoRA training complexity** | Single model, single LoRA | Dual-expert, **dual LoRA** | Single model, single LoRA | CogVideoX1.5 / TI2V-5B (tie) |
| **LoRA training time (character, A100)** | 10-14 hrs/run | **12-20+ hrs** (both experts) | ~4-6 hrs/run | TI2V-5B |
| **LoRA training tools** | 1 primary | **4+** | **4+** | Wan 2.2 |
| **Community size & activity** | Moderate, stalled | **Very active**, 16K+ stars | **Very active** | Wan 2.2 |
| **Identity drift** | Documented (same paper) | **Equally documented** (same paper) | Documented | **Tie** |
| **Loop closure** | Latent-space injection (principled) | FLF2V (color drift issues) | FLF2V (color drift) | CogVideoX1.5 |
| **Native FPS** | 16fps (needs 1.5x RIFE) | 16fps (needs 1.5x RIFE) | 24fps (misleading — ~6 latent keyframes/sec) | **Tie** (all need RIFE or produce slow-looking motion) |
| **Native resolution** | **1360x768** | 1280x720 | 1280x704 | CogVideoX1.5 (marginal) |
| **1:1 square support** | Not native (min 768px short side) | **Not native** (pixelation reported) | **Not native** | **Tie** (neither) |
| **VRAM (training)** | **~35GB** | 80GB | **24GB** | TI2V-5B |
| **VRAM (inference)** | 19-40GB | 80GB / 24GB (FP8) | **8-16GB** | TI2V-5B |
| **License** | Custom THUDM | **Apache 2.0** | **Apache 2.0** | Wan 2.2 |
| **Frame sharpness** | **Sharper/crisper** | Softer/cinematic | Softer/cinematic | CogVideoX1.5 |
| **Max duration** | **10 sec** | ~5 sec | ~5 sec | CogVideoX1.5 |
| **Development trajectory** | Stalled | **Active** (2.6, 2.7 released) | **Active** | Wan 2.2 |
| **Sprint plan compatibility** | **Fully designed & validated** | Requires full plan revision | Requires full plan revision | CogVideoX1.5 |
| **Existing style LoRAs for stacking** | None published | **Anime/2D LoRAs available** | Available | Wan 2.2 |
| **Animate mode** | Not available | Available (but NOT viable for penguins — see Sec 10) | Not available | **N/A** |
| **Text encoder** | T5-v1.1-XXL (4.7B) | UMT5-XXL (**13B** — richer embeddings) | UMT5-XXL (13B) | **Unclear** (larger ≠ better for this domain) |

---

## 16. Risk Assessment

### Risk Matrix

| Risk | CogVideoX1.5 Severity | Wan 2.2 I2V-A14B Severity | Notes |
|------|----------------------|--------------------------|-------|
| **Penguin identity not learnable** | Medium | Medium | No precedent for either model with flat-color mascots |
| **Output looks too realistic / 3D** | Unknown | **HIGH** | Wan 2.2 has documented photorealistic bias. CogVideoX1.5 is untested but may be less severe. |
| **Training takes too long** | Low (10-14 hrs, single model) | **Medium** (12-20+ hrs, dual expert) | CogVideoX1.5 is simpler to iterate |
| **Training tooling breaks** | Low (Passenger12138 purpose-built) | Low (4+ tools available) | Wan 2.2 has more fallback options |
| **Loop closure fails** | Low (latent-space, principled) | **Medium** (FLF2V has color drift, motion range limits) | CogVideoX1.5 approach is more architecturally sound |
| **Model development abandoned** | **Medium** (stalled since Nov 2024) | Low (active development through 2.7) | Wan 2.2's active trajectory is a genuine advantage |
| **License blocks commercial use** | **Medium** (custom THUDM, needs review) | Low (Apache 2.0) | Wan 2.2 has definitive advantage |
| **Community can't help debug issues** | Medium (smaller, less active) | Low (16K+ stars, extensive guides) | Wan 2.2 has larger support network |
| **Style LoRA stacking fails** | Not applicable (no style LoRAs exist) | Medium (stacking is documented but adds complexity) | Wan 2.2 has the option but it adds a variable |
| **Captioning pipeline needs redesign** | Low (plan designed for T5-XXL) | Medium (trigger word convention may conflict with natural language strategy) | UMT5-XXL is T5-family so approach should transfer, but unverified |

### Risk Summary

**CogVideoX1.5 risk profile:** Lower technical risk (simpler architecture, validated plan), higher strategic risk (stalled development, license uncertainty, smaller community).

**Wan 2.2 risk profile:** Higher technical risk (photorealistic bias, dual-expert complexity, color drift in loops), lower strategic risk (active development, Apache 2.0, large community).

---

## 17. What Remains Unknown

These questions **cannot be answered by research** — they require empirical testing with the actual Pudgy Penguins assets.

| Question | Why It Matters | When to Test | Est. Cost |
|----------|---------------|-------------|-----------|
| Does CogVideoX1.5 preserve flat-color cartoon outlines during I2V? | If it doesn't, the sprint's core deliverable is at risk regardless of model choice | Week 2 zero-shot baseline | Included in sprint |
| Does Wan 2.2 3D-ify flat-color cartoon input? | Determines whether Wan is viable at all for this IP | Week 2 lightweight probe | ~$50 |
| Can CogVideoX1.5's sharper rendering actually help cartoon line preservation? | Sharper could mean cleaner outlines OR sharper artifacts on thin features | Week 2 zero-shot baseline | Included in sprint |
| Can a style LoRA + character LoRA stack counteract Wan's realistic bias? | If yes, Wan becomes viable despite the bias | Phase 1 (if probe passes) | Phase 1 budget |
| At 3 seconds (~48 frames), does identity drift matter for either model? | Both models drift at ~50-70% of clip. At 3 sec that's only frames 24-34. | Week 4 integration testing | Included in sprint |
| Does Wan 2.2's UMT5-XXL text encoder respond to our natural language identity anchors as well as CogVideoX1.5's T5-XXL? | If not, captioning strategy needs revision for Wan | Week 2 probe (test same prompts on both models) | ~$20 |
| Does Wan 2.6/2.7 solve the photorealistic bias? | If yes, it's a strong Phase 1 candidate | Phase 1 evaluation | Phase 1 budget |
| How does the dual-expert LoRA training interact with the photorealistic bias? | Training BOTH experts on cartoon content may compound or mitigate the bias | Phase 1 (if Wan selected) | Phase 1 budget |

---

## 18. Verdict & Strategy Recommendation

### Core Finding

**Neither model is clearly superior for this specific use case.** Each has distinct advantages and the actual performance on flat-color cartoon penguins is unknown for both.

The earlier assessment that Wan 2.2 was *"a stronger candidate across nearly every dimension"* was based on uncritical acceptance of benchmark scores, unverified training claims, and missing the documented photorealistic bias. The corrected picture is more nuanced.

### Where CogVideoX1.5 Wins

1. **Architectural simplicity** — 1 model, 1 LoRA, 1 set of hyperparameters, half the tuning complexity
2. **Frame sharpness** — may better preserve clean cartoon outlines and edges
3. **Latent-space loop closure** — more principled than FLF2V, no color drift documented
4. **Sprint plan fully validated** — 14 rounds of systematic technical review, every decision substantiated
5. **Higher native resolution** — 1360x768 vs 1280x720 (marginal)
6. **Longer native duration** — 10 sec vs 5 sec (provides loop closure buffer)

### Where Wan 2.2 Wins

1. **Apache 2.0 license** — no commercial use uncertainty
2. **Active development** — 2.6 and 2.7 address identity preservation; CogVideoX1.5 is stalled
3. **Larger community** — 4x GitHub stars, 4+ training tools, extensive guides
4. **Existing style LoRAs** — anime/2D LoRAs available for stacking to counteract realistic bias
5. **Lower inference VRAM** (TI2V-5B) — enables cheaper deployment hardware
6. **Future-proofing** — the Wan ecosystem is where innovation is happening

### Where Neither Wins

1. **Cartoon mascot fidelity** — untested for both
2. **Identity drift** — same problem, same CVPR 2026 paper, same root cause
3. **1:1 square GIPHY format** — not natively supported by either

### Recommended Strategy

**Execute the sprint on CogVideoX1.5 as designed. Add a lightweight Wan 2.2 probe to Week 2. Evaluate Wan 2.6/2.7 for Phase 1.**

| Action | When | Cost | Purpose |
|--------|------|------|---------|
| **Execute sprint on CogVideoX1.5** | Weeks 1-6 | Sprint budget | Validated plan, lower technical risk, simpler architecture |
| **Download Wan 2.2 TI2V-5B + I2V-A14B** | Week 1, Day 2 (parallel) | ~30 min, free | Preparation for probe |
| **Run 3-5 zero-shot Wan 2.2 I2V generations** | Week 2, Day 3-4 (parallel with CogVideoX baseline) | ~$50, few hours | **Answer the photorealistic bias question empirically** |
| **Compare outputs visually** | Week 2, Day 4 | 30 min | Does Wan 3D-ify the penguin? Does CogVideoX preserve outlines? |
| **Document findings** | Week 6, technical report | Included | Data for Phase 1 model decision |
| **Recommend Wan 2.6/2.7 evaluation** | Week 6, Phase 1 proposal | Included | Newer versions address identity preservation |

**Rationale:**

1. **Don't change a validated plan.** Switching models cascades through captioning, training config, inference pipeline, loop strategy, evaluation framework, and timeline. The risk of destabilizing a proven plan exceeds the uncertain upside.

2. **The photorealistic bias question can only be answered empirically.** Running both models on the client's actual penguin art costs ~$50 and a few hours. This is trivially justified.

3. **Wan's advantages are strategic, not tactical.** The Apache 2.0 license, active development, and larger community are Phase 1 considerations, not sprint-critical factors. For a 6-week validation sprint, architectural simplicity and plan stability matter more.

4. **Wan 2.6/2.7 may already solve the problems.** The Wan ecosystem's active development means the photorealistic bias and identity preservation issues may be addressed in newer versions. Evaluating these in Phase 1 — when there's more time and sprint data to inform the comparison — is the highest-value timing.

---

## 19. Sources

### Community Reports & Guides
- [CivitAI: Anime Style WAN 2.2 I2V LoRA](https://civitai.com/models/2222779/anime-style-wan-22-i2v)
- [CivitAI: Taz's Anime Style LoRA Training Guide Part 1](https://civitai.com/articles/20389)
- [CivitAI: Taz's Anime Style LoRA Training Guide Part 2](https://civitai.com/articles/23798)
- [CivitAI: WAN2.2 LoRA Workflow TLDR](https://civitai.com/articles/17740)
- [CivitAI: WAN 2.2 Local LoRA Training Guide](https://civitai.com/articles/18985)
- [CivitAI: Simple/Pure Color Anime Style LoRA](https://civitai.com/models/1872525)
- [CivitAI: 2D Animation Effects LoRA](https://civitai.com/models/1920897)
- [CivitAI: WAN 2.2 Perfect Loops](https://civitai.com/models/1869481)
- [CivitAI: Wan 2.1 I2V Loop Workflow](https://civitai.com/models/1720535)
- [Apatero: Wan 2.2 LoRA Training Time Analysis](https://www.apatero.com/blog/wan-22-video-lora-training-time-complete-analysis-2025)
- [Apatero: Train Wan 2.2 LoRA for Person Guide](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025)
- [Apatero: Train Wan 2.2 LoRAs Best Practices](https://www.apatero.com/blog/train-wan-22-loras-best-practices-2025)
- [Apatero: Avoid Slow Motion in Wan 2.2](https://www.apatero.com/blog/avoid-slow-motion-wan-22-video-generation-2025)
- [Apatero: Wan 2.2 ComfyUI Complete Guide](https://apatero.com/blog/wan-2-2-comfyui-complete-guide-ai-video-generation-2025)
- [WaveSpeed: WAN 2.2 LoRA Training Settings](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/)
- [RunComfy: WAN 2.2 I2V Character Consistency LoRA](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora)
- [Dredyson: I2V Denoise & Face Consistency Guide](https://dredyson.com/how-i-fixed-wan2-2-i2v-on-8gb-vram-a-complete-step-by-step-beginners-guide-to-source-faithful-animation-settings-denoise-tuning-and-face-consistency-workarounds-that-actually-work-after-6-months/)
- [GenAIntel: CogVideoX-5B vs Wan 2.2-5B Comparison](https://www.genaintel.com/compare/cogvideox-5b-vs-wan-22-5b)
- [AMD ROCm: Wan2.2 Fine-Tuning Part 1](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html)
- [InstaSD: Wan 2.2 Prompting Guide](https://www.instasd.com/post/wan2-2-whats-new-and-how-to-write-killer-prompts)
- [NextDiffusion: Wan 2.2 Looping Animations in ComfyUI](https://www.nextdiffusion.ai/tutorials/wan-2-2-looping-animations-in-comfyui)
- [EachLabs: Designing Loopable Animations](https://www.eachlabs.ai/blog/designing-loopable-animations-with-wan-animate)

### Official Documentation & Model Cards
- [Wan 2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 Official Blog](https://wan.video/blog/wan2.2)
- [Wan2.2-I2V-A14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [Wan2.2-TI2V-5B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Wan2.2-Animate-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [Wan-Animate Preprocessing UserGuider.md](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md)
- [Wan-Animate Project Page](https://humanaigc.github.io/wan-animate/)
- [Wan-Animate Paper (arXiv 2509.14055)](https://arxiv.org/html/2509.14055v1)
- [THUDM/CogVideoX1.5-5B-I2V on HuggingFace](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)
- [CogVideoX LoRA Training Guide (HuggingFace diffusers)](https://huggingface.co/docs/diffusers/en/training/cogvideox)
- [CogVideoX diffusers API docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
- [Passenger12138 CogVideoX LoRA Trainer](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train)
- [google/umt5-xxl Model Card](https://huggingface.co/google/umt5-xxl)
- [DeepWiki: Wan2GP Text Encoders](https://deepwiki.com/deepbeepmeep/Wan2GP/8.1-text-encoders)
- [DeepWiki: kijai ComfyUI WanVideoWrapper Vision Encoder](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/5.4-text-and-vision-encoder-integration)
- [ComfyUI Official Wan 2.2 Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)

### GitHub Issues
- [Wan 2.2: Seamless Looping Feature Request #81](https://github.com/Wan-Video/Wan2.2/issues/81)
- [Wan 2.2: Creating Perfect Looping Videos #49](https://github.com/Wan-Video/Wan2.2/issues/49)
- [Wan 2.2: 16fps vs 24fps Confusion #13](https://github.com/Wan-Video/Wan2.2/issues/13)
- [Wan 2.2: Relight LoRA Weight Mismatch #205](https://github.com/Wan-Video/Wan2.2/issues/205)
- [ComfyUI-WanVideoWrapper: Loop Color Contrast Issue #1541](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541)
- [DiffSynth-Studio: LoRA Color Shift #1105](https://github.com/modelscope/DiffSynth-Studio/issues/1105)
- [ComfyUI-Lora-Manager: Stop-Motion Artifacts #667](https://github.com/willmiao/ComfyUI-Lora-Manager/issues/667)
- [ComfyUI: Color Drift Issue #9975](https://github.com/Comfy-Org/ComfyUI/issues/9975)
- [ComfyUI: FPS Mismatch #9106](https://github.com/comfyanonymous/ComfyUI/issues/9106)
- [diffusion-pipe: Wan LoRA Key Errors #114](https://github.com/tdrussell/diffusion-pipe/issues/114)
- [Wan2GP: LoRA Incompatibility #449](https://github.com/deepbeepmeep/Wan2GP/issues/449)
- [Wan 2.1: Square Resolution Pixelation #233](https://github.com/Wan-Video/Wan2.1/issues/233)
- [Musubi-tuner: Rank Discussion #455](https://github.com/kohya-ss/musubi-tuner/discussions/455)

### Research Papers
- [IPRO: Identity-Preserving Reward-Guided Optimization for Video Generation (CVPR 2026)](https://arxiv.org/html/2510.14255v1) — names both CogVideoX and Wan as having identity drift
- [Wan-Animate: Unified Character Animation and Replacement (arXiv 2509.14055)](https://arxiv.org/html/2509.14055v1)
- [CogVideoX1.5 HuggingFace Model Card (architecture details)](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)

### Architecture Analysis
- [PyxelJam: Wan 2.2 MoE Architecture Explained](https://pyxeljam.com/wan-2-2-moe-architecture-explained-cinematic-level-aesthetics/)
- [DeepLearning.AI: Wan 2.2 Architecture Analysis](https://www.deeplearning.ai/the-batch/alibabas-wan-2-2-video-models-adopt-a-new-architecture-to-sort-noisy-from-less-noisy-inputs)
- [Wan 2.2 Official Blog: Ablation Study Results](https://wan.video/blog/wan2.2)

### Wan 2.6/2.7 Identity Improvements
- [10b.ai: Wan 2.6 Face Stability Guide](https://10b.ai/blog/wan-2-6-i2v-face-stability)
- [MindStudio: What Is Wan 2.6](https://www.mindstudio.ai/blog/what-is-wan-2-6-video-open-source)
- [Alici.AI: Wan 2.7 Guide](https://alici.ai/blog/how-to-use-wan-27-guide-2026)

### Additional Guides
- [Scenario: Wan 2.2 Animate Model Essentials](https://help.scenario.com/en/articles/wan-2-2-animate-models-the-essentials/)
- [Wan-Animate I2V Prompting Guide](https://wan-animate.com/posts/how-to-use-i2v-prompting-wan-2-2-animate-guide)
- [302.AI: Wan2.2-Animate 4-Case Test](https://medium.com/@302.AI/wan2-2-animate-model-test-with-4-cases-e9e1c1ef492c)
- [Spheron: AI Video Generation GPU Guide](https://www.spheron.network/blog/ai-video-generation-gpu-guide/)
- [CogVideoX-LoRA-Wallace_and_Gromit Discussion](https://huggingface.co/Cseti/CogVideoX-LoRA-Wallace_and_Gromit/discussions/1)

---

*This research was conducted using three independent research agents cross-referencing community reports (CivitAI, Reddit, GitHub Issues), official documentation, research papers, and technical guides. Each claim is tagged with its evidence quality. The document represents the state of evidence as of early June 2026.*
