# Wan 2.2 & Competing Models — Evaluation for Pudgy Penguins Animation Pipeline

## Comparative Assessment of Open-Source Video Generation Models

**Companion document to:** `Final_Sprint_Plan.md` (CogVideoX1.5 sprint) and `Helios_Experimentation_Plan.md` (long-form exploration)
**Purpose:** Evaluate whether Wan 2.2 or any other open-source model should replace, augment, or serve as an alternative to CogVideoX1.5-5B-I2V for the Pudgy Penguins animation pipeline.
**Priority:** Best solution with best consistency. Budget is not a constraint.

---

## Table of Contents

1. [Landscape Overview](#1-landscape-overview)
2. [Wan 2.2 Deep Evaluation](#2-wan-22-deep-evaluation)
3. [LTX-Video 2.3 Evaluation](#3-ltx-video-23-evaluation)
4. [HunyuanVideo-I2V Evaluation](#4-hunyuanvideo-i2v-evaluation)
5. [Head-to-Head Comparison Matrix](#5-head-to-head-comparison-matrix)
6. [Verdict & Recommendations](#6-verdict--recommendations)
7. [Revised Model Strategy](#7-revised-model-strategy)

---

## 1. Landscape Overview

As of June 2026, five open-source models are viable candidates for I2V character animation with LoRA fine-tuning:

| Model | Params | I2V | LoRA Training | LoRA Maturity | Max Duration | Native FPS | License |
|-------|--------|-----|---------------|---------------|-------------|------------|---------|
| **CogVideoX1.5-5B-I2V** | 5B | Native variant | Passenger12138 + diffusers | Mature (18+ months) | 10 sec | 16fps | Custom THUDM |
| **Wan 2.2 I2V-A14B** | 14B active (27B MoE) | Native variant | AI Toolkit + Musubi + diffusion-pipe | **Very mature** | ~5 sec | 16fps | Apache 2.0 |
| **Wan 2.2 TI2V-5B** | 5B (dense) | Unified T2V+I2V | DiffSynth-Studio + community | Mature | ~5 sec | 24fps | Apache 2.0 |
| **Helios 14B** | 14B | Supported | **Not available** | None | 60+ sec | 16-24fps | Apache 2.0 |
| **LTX-Video 2.3** | 22B | Supported | Official trainer (ltx-trainer) | Mature | ~5 sec | 24fps | Apache 2.0* |
| **HunyuanVideo-I2V** | ~13B | Native variant | Official + Musubi + diffusion-pipe | Mature | ~5 sec | 24fps | Tencent Open |

*LTX-Video 2.3: Apache 2.0 for orgs under $10M annual revenue.

---

## 2. Wan 2.2 Deep Evaluation

### 2.1 Architecture

Wan 2.2 uses a **Mixture-of-Experts (MoE)** architecture with two specialized 14B-parameter transformers:

| Expert | Active During | Responsibility |
|--------|--------------|----------------|
| **High-noise expert** | Early denoising (high SNR steps) | Global composition, motion trajectory, camera movement |
| **Low-noise expert** | Late denoising (low SNR steps) | Fine detail, identity preservation, texture |

Total parameters: ~27B. Active per step: ~14B. The expert switch occurs at a threshold step corresponding to half of the minimum SNR.

**Why this matters for Pudgy Penguins:** The dual-expert architecture naturally separates "what moves" (high-noise) from "what the character looks like" (low-noise). This is architecturally aligned with our problem — we need the penguin to move fluidly (high-noise expert) while keeping its exact beak shape, scarf color, and proportions perfectly stable (low-noise expert). LoRA training can target these experts independently.

### 2.2 The TI2V-5B Variant

The TI2V-5B is a **dense 5B parameter** model using a new high-compression VAE:

| Spec | Value |
|------|-------|
| VAE compression | 16x16x4 (T x H x W) = 64x overall |
| With patchification | 4x32x32 total compression |
| Resolution | 720P (1280x704) |
| Native FPS | **24fps** (no interpolation needed) |
| Consumer GPU | RTX 4090 (24GB VRAM) with offloading |
| Generation speed | 5-sec 720P video in under 9 minutes on consumer GPU |

**Why this matters:** The TI2V-5B generates at **24fps natively**, eliminating our entire RIFE interpolation + LoRA cleanup pipeline (Steps 5-6 of the sprint inference chain). This removes the ghosting/artifact risk zone entirely and simplifies the ComfyUI workflow.

### 2.3 Wan-Animate-14B — The "Animation Engine" Model

This is the most relevant Wan 2.2 variant for the Pudgy Penguins use case. It's a **purpose-built character animation and replacement model:**

| Feature | Description |
|---------|-------------|
| **Character Animation** | Takes a reference character image + a motion reference video. Transfers the motion onto the character while maintaining visual identity. |
| **Character Replacement** | Replaces a character in an existing video with a new reference character. Preserves motion, lighting, and camera. |
| **Relighting LoRA** | Built-in LoRA that automatically adjusts lighting/color tone when placing a character into a new scene. |
| **Non-human support** | Explicitly supports cartoon figures, fictional characters, stylized designs, mascots, and anime characters. |

**Critical detail:** The Wan-Animate developers **explicitly warn against using LoRAs trained on Wan 2.2 with Wan-Animate** — "weight changes during training may lead to unexpected behavior." This means Wan-Animate relies on its built-in generalization to handle custom characters from reference images alone, without fine-tuning.

### 2.4 LoRA Training Ecosystem

Wan 2.2 has the **most mature LoRA training ecosystem** of any current open-source video model:

| Tool | I2V Support | Notes |
|------|-------------|-------|
| **AI Toolkit (Ostris)** | T2V and I2V 14B | GUI-based, RunPod/Modal recipes, consumer GPU support |
| **Musubi-tuner** | T2V and I2V | CLI-focused, dataset configs, pre-caching, FP8, block-swapping |
| **Diffusion-pipe** | T2V and I2V | Pipeline-parallel architecture, efficient on multi-GPU |
| **DiffSynth-Studio** | T2V and I2V, TI2V-5B | Low-memory offload, FP8 quantization, LoRA + full training |
| **Diffusers** | T2V and I2V | Official HuggingFace integration |

**Community-validated hyperparameters (I2V character LoRA):**

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| LoRA Rank | 16 (identity) / 32 (style) | Lower than CogVideoX1.5 (64) — Wan is more parameter-efficient |
| Alpha | Match rank (16 or 32) | |
| Learning Rate | 5e-5 (identity) / 7e-5 to 1e-4 (style) | Higher than CogVideoX1.5 (3e-5) — different model sensitivity |
| Optimizer | AdamW (wd=0.01) | AdamW8Bit for VRAM-constrained |
| Scheduler | Cosine + 5% warmup | |
| Steps | 1500-2500 (identity) / 2000-3000 (style) | Lower step count than CogVideoX1.5 (4000) |
| Dataset | 12-20 images (identity) / 30-50 (style) | Smaller dataset requirements |
| Batch Size | 2-4 (A100/4090) | Higher than CogVideoX1.5 (1) |

**Key advantage:** Wan 2.2 LoRA training requires **fewer steps, smaller datasets, and lower rank** than CogVideoX1.5 for comparable character fidelity. Community reports suggest training converges in ~2-4 hours on A100 vs 10-14 hours for CogVideoX1.5.

### 2.5 Merits for Pudgy Penguins

| Merit | Detail | Impact |
|-------|--------|--------|
| **MoE architecture separates identity from motion** | Low-noise expert handles identity preservation. Can be targeted independently during LoRA training. | Potentially stronger character consistency than single-DiT approaches |
| **24fps native (TI2V-5B)** | Eliminates RIFE interpolation entirely | Removes ghosting/artifact risk, simplifies pipeline, reduces inference time |
| **Wan-Animate for motion transfer** | Reference image + motion video = animated character | Could skip LoRA entirely for some use cases — direct motion transfer from existing animation library |
| **Mature LoRA ecosystem** | 4+ training tools, validated hyperparameters, active community (16K+ GitHub stars) | Lowest risk for LoRA training success |
| **Lower training compute** | Rank 16, 1500-2500 steps, 2-4 hour training runs | More iteration cycles per week — 6+ runs vs 3 for CogVideoX1.5 |
| **Apache 2.0 license** | Fully permissive commercial use | Eliminates the THUDM license risk for commercial output |
| **Consumer GPU inference (TI2V-5B)** | 8-12GB VRAM with FP8 | Glass box deployment could run on vastly cheaper hardware |
| **Community-proven character consistency** | Dedicated character consistency LoRA training guides exist | The specific problem we're solving has been addressed by the community |

### 2.6 Demerits for Pudgy Penguins

| Demerit | Detail | Impact | Mitigation |
|---------|--------|--------|------------|
| **Max 5-second clips** | Neither I2V-A14B nor TI2V-5B supports long-form generation | Same limitation as CogVideoX1.5. Helios remains the long-form candidate. | GIF pipeline (sprint target) is 3-5 seconds — within capability |
| **MoE requires 80GB VRAM (14B)** | I2V-A14B needs A100 80GB or H100 for training | Training cost similar to CogVideoX1.5 | TI2V-5B variant only needs 24GB — use 5B for inference, 14B for training if needed |
| **Trigger word approach** | Community LoRA training uses synthetic trigger words (e.g., "zxq-person") | Conflicts with our natural language identity anchor strategy | **Needs investigation** — test natural language anchors vs trigger words on Wan 2.2's text encoder |
| **Wan-Animate warns against custom LoRAs** | Cannot combine character LoRA + Wan-Animate motion transfer | Limits the "best of both worlds" approach | Use Wan-Animate with reference images only (no LoRA). Use LoRA-based I2V pipeline separately. |
| **Different text encoder** | Wan 2.2 does NOT use T5-XXL — uses a different encoder architecture | Our entire captioning pipeline was designed for T5-XXL. Identity anchors, pruning scripts, and natural language strategy would need re-validation. | Re-validate captioning approach on Wan 2.2's encoder. May work as-is or may require keyword-based approach. |
| **720P max resolution (TI2V-5B)** | 1280x704 vs CogVideoX1.5's 1360x768 | Slightly lower resolution, different aspect ratio | Marginal — both are ~720P. Upscale to 4K with Real-ESRGAN if needed. |
| **Newer release (July 2025)** | Less battle-tested than CogVideoX1.5 | More unknown failure modes | Mitigated by large community (16K stars) and multiple training tools |

---

## 3. LTX-Video 2.3 Evaluation

### 3.1 Overview

| Spec | Value |
|------|-------|
| Parameters | 22B (DiT-based) |
| I2V support | Yes |
| Audio generation | **Native** — generates video + audio in single pass |
| LoRA training | Official trainer (`ltx-trainer`) included in repo |
| Max duration | ~5 seconds |
| Native FPS | 24fps |
| License | Apache 2.0 (orgs under $10M revenue) |
| VRAM | 32GB recommended (RTX 3090/4090) |
| GitHub stars | Moderate (~5K) |

### 3.2 Merits

| Merit | Detail |
|-------|--------|
| **Official LoRA trainer** | Built into the repo — no third-party tools needed. Supports LoRA, full fine-tuning, and IC-LoRA. |
| **Audio + video in one pass** | Generates synchronized audio with video. Future Phase 1 consideration for narrative clips. |
| **24fps native** | No frame interpolation needed |
| **Multi-LoRA stacking** | Supports up to 3 LoRAs simultaneously at inference |
| **Community LoRA ecosystem** | Camera movement LoRAs, transition LoRAs, style LoRAs published on HuggingFace |
| **Training speed** | "Less than an hour" for motion/style LoRAs |

### 3.3 Demerits

| Demerit | Detail | Severity |
|---------|--------|----------|
| **I2V stability bugs** | "Known instability bugs" — model occasionally freezes or over-applies Ken Burns effect | **High** — unreliable I2V is a dealbreaker for our I2V-first pipeline |
| **22B parameters** | Largest model in the comparison. Training requires significant compute. | Medium — offset by efficient LoRA training |
| **Revenue-gated license** | Apache 2.0 only for orgs under $10M. Pudgy Penguins may exceed this. | **High** — must verify client's revenue against threshold |
| **Primarily photorealistic** | Community LoRAs and showcases skew toward live-action/photorealistic content. Stylized 2D cartoon animation is underrepresented. | Medium — may not generalize well to Pudgy Penguins aesthetic |
| **Rank 384 LoRA issues** | Official distilled LoRA at rank 384 "can actively dampen conditioning signals in I2V workflows" | Medium — use lower rank (72 and below) for I2V |
| **Less character consistency focus** | Community focus is on transitions, camera movement, and audio — less on character identity | Medium — fewer proven character LoRA examples |

### 3.4 Verdict

**Not recommended as primary model.** The I2V instability bugs alone disqualify it for a production sprint where I2V first-frame conditioning is the core mechanism. The revenue-gated license adds legal uncertainty. The audio generation capability is interesting for Phase 1's 30-second narrative clips but doesn't help the GIF sprint.

**Watch for:** If I2V stability is fixed in a future release and the client's revenue falls under the license threshold, LTX-2.3 becomes a strong contender for audio-visual narrative clips in Phase 1.

---

## 4. HunyuanVideo-I2V Evaluation

### 4.1 Overview

| Spec | Value |
|------|-------|
| Parameters | ~13B |
| I2V support | **Dedicated I2V variant** (`HunyuanVideo-I2V`) |
| LoRA training | Official scripts + Musubi Tuner + diffusion-pipe |
| Max duration | ~5 seconds |
| Native FPS | 24fps |
| License | Tencent Open Source |
| VRAM (training) | 79-80GB minimum (A100 80GB) |
| Organization | Tencent |

### 4.2 Merits

| Merit | Detail |
|-------|--------|
| **Dedicated I2V model** | Separate model optimized specifically for I2V — not a unified model with I2V as secondary |
| **Official LoRA training scripts** | `run_train_image2video_lora.sh` included in repo |
| **Mature training ecosystem** | Musubi Tuner, diffusion-pipe, diffusers all support it |
| **24fps native** | No frame interpolation needed |
| **Strong temporal consistency** | Known for good motion coherence in community benchmarks |
| **Trigger word training** | Supports trigger word injection in captions for concept LoRAs |

### 4.3 Demerits

| Demerit | Detail | Severity |
|---------|--------|----------|
| **80GB VRAM minimum for training** | "Tested on a single 80G GPU, with a minimum GPU memory requirement of 79GB for 360p" | High — most expensive training requirements in this comparison |
| **Tencent license** | Not Apache 2.0. Requires review for commercial output permissions. | Medium — same category of risk as CogVideoX1.5's THUDM license |
| **360P training minimum** | Training at higher resolutions requires even more VRAM or multi-GPU | High — training at 720P may require 2+ A100s |
| **Less community momentum than Wan 2.2** | Wan 2.2 has 16K stars vs HunyuanVideo's ~8K. Fewer training guides and published LoRAs. | Medium |
| **Tencent organizational risk** | Chinese tech company — potential export control / sanction risk for some clients | Low (but worth noting) |
| **Stylized 2D unknown** | Community focus is primarily photorealistic. Few examples of cartoon/stylized LoRAs. | Medium |

### 4.4 Verdict

**Not recommended.** HunyuanVideo-I2V is a capable model, but it offers no clear advantage over Wan 2.2 while having higher VRAM requirements, a less mature LoRA ecosystem, and a more restrictive license. The 80GB minimum for LoRA training at 360P is a significant compute constraint.

---

## 5. Head-to-Head Comparison Matrix

### For the Pudgy Penguins Use Case (Stylized 2D Character Animation, I2V-First Pipeline)

| Dimension | CogVideoX1.5-5B | Wan 2.2 I2V-A14B | Wan 2.2 TI2V-5B | Helios 14B | LTX-2.3 | HunyuanVideo-I2V |
|-----------|-----------------|-------------------|------------------|------------|---------|-------------------|
| **I2V Quality** | Good | Excellent | Very Good | Good (unproven with LoRA) | Unstable (known bugs) | Good |
| **Character Consistency** | Good (with LoRA) | **Excellent** (MoE identity expert) | Very Good | Unknown | Moderate | Good |
| **LoRA Training Maturity** | Mature | **Most Mature** | Mature | None | Mature | Mature |
| **Training Speed** | 10-14 hrs/run | **2-4 hrs/run** | 2-4 hrs/run | N/A | <1 hr/run | 8-14 hrs/run |
| **Native FPS** | 16fps (needs RIFE) | 16fps (needs RIFE) | **24fps (no RIFE)** | 16-24fps | **24fps** | **24fps** |
| **Max Duration** | 10 sec | 5 sec | 5 sec | **60+ sec** | 5 sec | 5 sec |
| **Max Resolution** | 1360x768 | 720P | 1280x704 | 1360x768+ | Flexible | 720P+ |
| **Inference VRAM** | 19-40GB | 80GB (14B) | **8-12GB (FP8)** | 6-19GB | 32GB | 24-80GB |
| **Training VRAM** | ~35GB | 80GB (14B) | **24GB** | N/A | 32GB | **79-80GB** |
| **License** | Custom THUDM | **Apache 2.0** | **Apache 2.0** | **Apache 2.0** | Apache 2.0* | Tencent Open |
| **2D Stylized Support** | Untested | **Explicitly supports cartoon/stylized** | Supported | Untested | Photorealistic-biased | Untested |
| **Animate Mode** | No | **Yes (Wan-Animate)** | No | No | No | No |
| **Community Size** | Moderate | **16K+ stars** | **16K+ stars** | 1.9K stars | ~5K stars | ~8K stars |
| **Long-Form Potential** | Low | Low | Low | **Excellent** | Low | Low |

### Winner by Category

| Category | Winner | Runner-Up |
|----------|--------|-----------|
| **Character consistency (LoRA)** | **Wan 2.2 I2V-A14B** | CogVideoX1.5 |
| **Production simplicity (no RIFE needed)** | **Wan 2.2 TI2V-5B** | LTX-2.3 |
| **Training speed & iteration** | **Wan 2.2** (any variant) | LTX-2.3 |
| **Long-form content (30+ sec)** | **Helios 14B** | None viable |
| **License safety** | **Wan 2.2 / Helios** (Apache 2.0) | LTX-2.3 (revenue-gated) |
| **Consumer GPU deployment** | **Wan 2.2 TI2V-5B** (8-12GB) | Helios (6GB) |
| **2D stylized animation** | **Wan 2.2** (explicit support) | CogVideoX1.5 |
| **Audio + video** | **LTX-2.3** (native) | None |

---

## 6. Verdict & Recommendations

### The Uncomfortable Truth

Based on this comprehensive analysis, **Wan 2.2 is a stronger candidate than CogVideoX1.5-5B-I2V for the Pudgy Penguins sprint** across nearly every dimension:

| Advantage | CogVideoX1.5 | Wan 2.2 |
|-----------|--------------|---------|
| Character consistency architecture | Single DiT — identity and motion entangled | MoE — identity expert separated from motion expert |
| LoRA training ecosystem | 1 specialized tool (Passenger12138) | **4+ tools** (AI Toolkit, Musubi, diffusion-pipe, DiffSynth) |
| Training iteration speed | 10-14 hours per run (3 runs/week) | **2-4 hours per run (6+ runs/week)** |
| Native FPS (TI2V-5B) | 16fps — requires RIFE pipeline | **24fps — no interpolation needed** |
| License | Custom THUDM (requires review) | **Apache 2.0 (fully permissive)** |
| 2D stylized support | Not explicitly documented | **Explicitly supports cartoon/stylized characters** |
| Community validation | Moderate | **Character consistency LoRA guides published** |
| Consumer deployment | Requires A100 40GB+ | **TI2V-5B runs on RTX 4090 (8-12GB FP8)** |

CogVideoX1.5's only advantages are:
1. **Longer max duration (10 sec vs 5 sec)** — marginal, since our target is 3-5 sec GIFs
2. **Higher max resolution (1360x768 vs 1280x704)** — marginal, both ~720P
3. **Our sprint plan was designed around it** — significant, but a sunk planning cost, not a technical advantage

### However — Three Critical Unknowns Remain

Before recommending a model switch, three things must be tested:

**1. Text Encoder Compatibility**
Our entire captioning pipeline — natural language identity anchors, two-tier structure, pruning scripts, anti-entanglement strategy — was designed for CogVideoX1.5's frozen T5-XXL encoder. Wan 2.2 uses a different text encoder. We must verify:
- Does the natural language approach work, or does Wan 2.2 prefer keyword tags + trigger words?
- Can the identity anchor strategy transfer, or does it need to be redesigned?

**2. Zero-Shot I2V Quality on Pudgy Penguins**
Wan 2.2's quality superiority is documented on photorealistic and general content. We need to verify it holds for highly stylized 2D cartoon penguins with thick outlines, flat colors, and exaggerated proportions.

**3. Wan-Animate vs I2V LoRA — Which Path is Better?**
Two viable approaches exist for Wan 2.2:
- **Path A:** Train an I2V LoRA on Wan 2.2 (same strategy as CogVideoX1.5 sprint, adapted)
- **Path B:** Use Wan-Animate with reference character images + motion reference videos from the client's existing 150+ animation library (no LoRA needed — the model generalizes to cartoon characters)

Path B is potentially revolutionary — it would skip LoRA training entirely and use the client's existing animations as motion templates. But it's unproven for this specific IP.

---

## 7. Revised Model Strategy

### Recommendation: Dual-Track Validation in Week 2

Rather than committing the entire sprint to a single model, we use the existing Week 2 zero-shot baseline testing to evaluate **both** CogVideoX1.5 and Wan 2.2 side-by-side. This adds ~1 day to Week 2 but provides definitive data for the most important decision of the sprint.

### Proposed Week 2 Dual-Track Test

| Test | CogVideoX1.5-5B-I2V | Wan 2.2 TI2V-5B (I2V mode) | Wan 2.2-Animate-14B |
|------|---------------------|---------------------------|---------------------|
| **A: Single penguin, zero-shot** | Feed hand-drawn layout, generate 5 sec | Same layout, same prompt | Reference image + motion video from library |
| **B: Multi-character, zero-shot** | Tier 1 layout | Same layout, same prompt | Two reference images + interaction video |
| **C: Complex background** | Kitchen scene layout | Same layout | Same layout |
| **D: Motion quality** | "Penguin waving" prompt | Same prompt | Motion reference: existing wave animation |

**Evaluation:** Run all outputs through core metrics (FID, LPIPS, SSIM) + HeliosBench drifting + qualitative rubric. **Blind test:** Show animation director outputs from all three approaches without labels.

### Decision Gate (End of Week 2)

| Outcome | Decision |
|---------|----------|
| Wan 2.2 clearly outperforms CogVideoX1.5 on zero-shot quality | **Switch to Wan 2.2** for Weeks 3-6. Re-validate captioning strategy for Wan's encoder. Use faster training cycles (2-4 hrs vs 10-14 hrs) to iterate more aggressively. |
| CogVideoX1.5 and Wan 2.2 roughly equivalent | **Stay on CogVideoX1.5.** Plan is already built around it. Lower migration risk. Note Wan 2.2 as Phase 1 alternative. |
| Wan-Animate produces usable output without LoRA | **Major finding.** Test further in Week 3 alongside LoRA training. If Wan-Animate works for 80%+ of GIF use cases, it becomes the primary pipeline (no training needed). |
| Both models fail on stylized 2D penguins | **Escalate.** Re-evaluate model selection. Consider HunyuanVideo-I2V or LTX-2.3 as fallbacks. |

### What This Costs

| Addition | Time | Compute |
|----------|------|---------|
| Download Wan 2.2 TI2V-5B + Animate-14B | Week 1, Day 2 (parallel with Docker build) | ~30 min download |
| Run Wan 2.2 zero-shot tests (12-16 generations) | Week 2, Day 3-4 (parallel with CogVideoX tests) | ~1 day A100 time (~$30-50) |
| Animation director blind review | Week 2, Day 4-5 (15 min session) | None |

**Total added cost: ~$50 and 1 day.** For the potential to discover a significantly better model, this is trivially justified.

---

## Appendix: Key Links

### Wan 2.2
- [GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan-Animate Project Page](https://wan-animate.org/)
- [TI2V-5B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [I2V-A14B on HuggingFace (Diffusers)](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- [Animate-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [WAN 2.2 LoRA Training Settings Guide](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/)
- [Character Consistency LoRA Training (RunComfy)](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora)
- [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

### LTX-Video 2.3
- [GitHub Repository](https://github.com/Lightricks/LTX-2)
- [LTX-2.3 on HuggingFace](https://huggingface.co/Lightricks/LTX-2.3)
- [LoRA Training Documentation](https://ltx.io/model/capabilities/lora-training)

### HunyuanVideo-I2V
- [GitHub Repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)
- [HuggingFace Model](https://huggingface.co/tencent/HunyuanVideo-I2V)
- [Civitai LoRA Training Guide](https://civitai.com/articles/12954/hunyuan-video-lora-training)

### CogVideoX1.5 (Current Sprint Model)
- [THUDM/CogVideoX1.5-5B-I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)
- [Passenger12138 LoRA Trainer](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train)

### Helios (Long-Form Candidate)
- [GitHub Repository](https://github.com/PKU-YuanGroup/Helios)
- [Experimentation Plan](./Helios_Experimentation_Plan.md)

---

*This evaluation was conducted to ensure the sprint uses the best available model, not just the first model selected. The dual-track validation in Week 2 costs ~$50 and 1 day but could prevent 4 weeks of training on a suboptimal model.*
