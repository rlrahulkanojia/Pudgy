# Pudgy Penguins — AI Animation Engine: 6-Week Validation Sprint Plan

## Final Comprehensive Plan

**Prepared for:** Saksham / Pudgy Penguins IP Team
**Sprint Duration:** 6 Weeks
**Objective:** Validate the feasibility of fine-tuning an open-source video diffusion model to augment the in-house 2D animation pipeline, beginning with the GIPHY looping GIF channel.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Decisions (Locked In)](#2-architecture-decisions-locked-in)
3. [Pre-Sprint Requirements](#3-pre-sprint-requirements)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Captioning Pipeline](#5-captioning-pipeline)
6. [Training Configuration](#6-training-configuration)
7. [Inference & Post-Processing Pipeline](#7-inference--post-processing-pipeline)
8. [Multi-Character Technical Strategy](#8-multi-character-technical-strategy)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Week-by-Week Execution Plan](#10-week-by-week-execution-plan)
11. [Handoff & Phase 1 Transition](#11-handoff--phase-1-transition)
12. [Risk Register](#12-risk-register)
13. [Appendix: Glossary of Terms](#13-appendix-glossary-of-terms)

---

## 1. Executive Summary

This sprint validates whether a fine-tuned open-source video diffusion model can produce production-quality looping GIFs that match the Pudgy Penguins brand aesthetic — with sufficient consistency, continuity, and speed to augment (not replace) the existing manual animation pipeline.

**Primary Target:** GIPHY channel — single-character looping reaction GIFs (3-5 seconds).
**Secondary Target (Committed):** Tier 1 multi-character scenes (side-by-side, non-interacting).
**Stretch Goal:** Tier 2 multi-character scenes (proximate interaction, no physical contact).
**Excluded from Sprint:** Tier 3 multi-character physical interaction/occlusion, 30-second+ narrative clips.

**Output Specifications:** All generated content is **silent** (no audio). Audio integration is out of scope for this validation sprint and is flagged as a Phase 1 consideration when scaling to 30-second narrative clips.

**Output Resolution:** Native generation at **1360x768** (CogVideoX1.5-5B maximum), with post-generation upscaling to 4K available via Real-ESRGAN if needed for specific delivery channels.

**Success Criteria:** Average qualitative rubric score >= 3.0/4.0 across all evaluation dimensions, with no dimension scoring 1. Quantitative metrics (FID, LPIPS, SSIM, temporal consistency) calibrated against and matching the client's own hand-animated baselines.

---

## 2. Architecture Decisions (Locked In)

The following decisions were validated through systematic technical review and are final for this sprint.

### 2.1 Base Model: CogVideoX1.5-5B-I2V (Single-Track)

| Decision | Rationale |
|----------|-----------|
| **Model:** `THUDM/CogVideoX1.5-5B-I2V` | Latest and best CogVideoX I2V variant. Max resolution **1360x768** (vs 720x480 on v1.0). Native **16fps** output (vs 8fps on v1.0). Up to **81 frames / 5 seconds** or **161 frames / 10 seconds**. LoRA training requires only ~35GB VRAM on A100 80GB — ample headroom. |
| **AnimateDiff dropped** | Older SD1.5-era architecture. Splitting Week 3 across two architectures would prevent adequate iteration on either. ControlNet advantage mitigated by I2V first-frame conditioning. |
| **Single-track execution** | Reclaims iteration time in Week 3. If CogVideoX fails catastrophically by mid-Week 3, AnimateDiff remains a documented fallback. |
| **Training framework:** [Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train) | Purpose-built for CogVideoX1.5 I2V LoRA training. Includes critical fixes: bucket-based multi-resolution training, corrected RoPE (Relative Position Encoding), and fixed OFS embedding. Preferred over legacy `cogvideox-factory`/`finetrainers` scripts which do not explicitly support the 1.5 I2V variant. |

**Resolution & Aspect Ratio:**

| Parameter | Value |
|-----------|-------|
| **Native max resolution** | 1360x768 (landscape) or 768x1360 (portrait) |
| **Resolution constraints** | Min(W, H) = 768, 768 <= Max(W, H) <= 1360, Max(W, H) % 16 = 0 |
| **Frame count formula** | 8N + 1 (e.g., 49, 81, 161 frames) |
| **Sprint default** | 81 frames @ 16fps = ~5 seconds @ 1360x768 |
| **Post-generation upscaling** | Real-ESRGAN to 4K if required for specific delivery channels |
| **Training resolution** | Must match inference resolution — dataset clips rendered/resized to 1360x768 |

### 2.2 No Separate Image LoRA Phase

| Decision | Rationale |
|----------|-----------|
| **Flux.1 / SDXL Image LoRA phase eliminated** | Latent spaces do not transfer between architectures. CogVideoX's 3D DiT has its own latent representation — a Flux LoRA provides zero mathematical benefit. |
| **Client's artists provide first frames** | The team already produces production-quality composited layouts. These serve as ground-truth I2V input frames, guaranteed 100% on-brand. |
| **Character volume taught natively** | T-poses and multi-angle turnarounds are packed into the CogVideoX training dataset via joint image-video training, teaching the DiT the character's 3D volume directly. |

### 2.3 Natural Language Character Identification

| Decision | Rationale |
|----------|-----------|
| **No rare/synthetic trigger tokens (no `sks`, `ohwx`)** | CogVideoX uses a frozen T5-XXL text encoder (4.7B params). T5's SentencePiece tokenizer fragments rare tokens into meaningless sub-word chunks. The frozen encoder cannot learn new token associations. |
| **Dense natural language identity anchors** | T5-XXL is a language model — it already understands "penguin," "red scarf," "thick outlines." Semantically rich descriptions exploit the frozen encoder rather than fighting it. |
| **Character's actual name as identifier** | e.g., "Pogo" — used as a natural language proper noun, not an out-of-vocabulary injection. |

### 2.4 Multi-Character Scope

| Tier | Description | Sprint Status |
|------|-------------|---------------|
| **Tier 1:** Side-by-side, non-interacting (>30% canvas gap) | Two penguins performing independent actions, no physical contact | **Included — committed deliverable** |
| **Tier 2:** Proximate interaction (close, reacting, no contact) | Two penguins near each other, slight overlap possible | **Included — stretch goal** |
| **Tier 3:** Physical interaction with occlusion | Touching, overlapping, passing objects, limb crossing | **Excluded — Phase 1 roadmap** |

---

## 3. Pre-Sprint Requirements

**CRITICAL: The following must be completed BEFORE Week 1 begins.**

### 3.1 GPU / Compute Procurement

GPU resources must be requested and provisioned in advance. The following is the detailed compute specification:

#### Training Requirements

| Resource | Specification | Purpose |
|----------|--------------|---------|
| **GPU** | 1x NVIDIA A100 80GB | LoRA training on CogVideoX1.5-5B-I2V |
| **VRAM** | ~35GB (Rank 128 DDP bf16 at full 81x768x1360 — Rank 64 will use less). Ample headroom on 80GB card. | Fits Rank 64 LoRA + 5B model comfortably |
| **Training Duration** | 10-14 hours per run | ~4,000 steps at batch_size=1, grad_accum=4 |
| **Estimated Runs** | 3-5 runs minimum | Hyperparameter iteration |
| **Total Training GPU-Hours** | 40-80 hours | Across all runs in Weeks 2-3 |

#### Inference Requirements

| Resource | Specification | Purpose |
|----------|--------------|---------|
| **GPU** | 1x A100 40GB or H100 | I2V generation + RIFE + cleanup |
| **VRAM** | 19-40GB (19GB at max res 81 frames; less at lower res) | Full inference pipeline |
| **Time per GIF** | ~10-15 minutes (full pipeline) | Generation + loop closure + decode + interpolation + cleanup |
| **Batch mode** | 4 variations per input | ~50-60 minutes per batch |

#### Ancillary Compute

| Resource | Specification | Purpose |
|----------|--------------|---------|
| **VLM Captioning** | GPT-4o API or 1x A6000 (24GB) for local MiniCPM-V | Automated caption generation |
| **Evaluation Scripts** | CPU-only or lightweight GPU | FID, LPIPS, SSIM, RAFT scoring |

#### Cost Estimates (Cloud — RunPod Pricing)

Estimates include a **50% experimentation buffer** on top of baseline projections. Cost is not a constraint — the priority is quality and iteration freedom.

| Item | Rate | Baseline Estimate | With 50% Buffer |
|------|------|-------------------|-----------------|
| Training (A100 80GB) — 40-80 GPU-hrs | ~$1.50-2.00/hr | $60-160 | **$90-240** |
| Inference (A100 40GB) — Weeks 2-5, includes idle pod time | ~$1.00-1.50/hr | $100-200 | **$150-300** |
| Experimentation / failed runs / exploratory generations | ~$1.00-2.00/hr | $50-100 | **$75-150** |
| Persistent Storage (2 months) | ~$0.10/GB/month | $10-20 | **$15-30** |
| VLM Captioning (GPT-4o API) | ~$0.01/image | $5-10 | **$8-15** |
| **Total Sprint Compute Budget** | | **$225-490** | **$340-735** |

**Recommended budget allocation: $750.** This provides full freedom for additional training runs, extended inference testing, multi-character experimentation, and resolution upscaling trials without budget anxiety slowing iteration speed.

#### Infrastructure Specifications

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Cloud Provider** | RunPod (preferred) over Vast.ai | More reliable uptime, better networking, native Docker support, persistent network volumes |
| **Docker Image** | Custom, built Week 1 Day 1-2 | CogVideoX1.5-5B-I2V (`THUDM/CogVideoX1.5-5B-I2V`) + diffusers (source build from branch) + Passenger12138 LoRA trainer + ComfyUI + RIFE + all dependencies frozen. Pushed to Docker Hub. Every subsequent instance spin-up takes minutes |
| **Persistent Storage** | RunPod Network Volume | Dataset uploaded once, mounted to any instance. No re-uploading |
| **Checkpointing** | Every 500 training steps | If instance dies, resume from last checkpoint. Spot instance eviction protection |

### 3.2 Client Asset Delivery

The following assets must be delivered by the client's team before or during Week 1:

| Asset | Quantity | Format | Notes |
|-------|----------|--------|-------|
| Multi-angle character turnarounds / T-poses | All available per character | PNG, transparent background | Used for volume training |
| Master Control rules / style guides | All available | PDF / image | Reference for captioning |
| Raw After Effects project files | 10-15 representative skits | .aep files | For extracting isolated character layers |
| Character names + accessory descriptions | Per character | Text document | For identity anchor construction |

### 3.3 Legal / IP Review

> *Note: IP and licensing matters are being handled separately from this technical plan. CogVideoX1.5-5B license must be reviewed for commercial output permissions before Week 1.*

---

## 4. Dataset Strategy

### 4.1 Total Dataset Composition

**Target: 60-80 video clips + T-poses/turnarounds**

| Category | Clip Count | Percentage | Purpose |
|----------|-----------|------------|---------|
| Single-character actions | 36-48 | 60% | Core volume + motion learning |
| Tier 1 multi-character (side-by-side) | 15-20 | 25% | Spatial separation + dual-identity learning |
| Tier 2 multi-character (proximate) | 9-12 | 15% | Stretch goal proximity training |
| Static T-poses / turnarounds | 50-100 images | Joint dataset | 3D volume learning via joint image-video training |

### 4.2 Clip Curation Rules

All video clips must adhere to these strict parameters:

**Micro-Actions, Not Scenes:**
- Each clip isolates ONE action: a wave, a walk cycle, a head turn, a jump, an idle breathing loop.
- Do NOT include clips where the character performs a sequence (walk → stop → wave). Split into separate clips.

**Duration:**
- 2-5 seconds per clip at the client's native frame rate (24fps).
- CogVideoX1.5 will internally resample to its native 16fps during training.
- Frame count must conform to the 8N+1 formula (e.g., 49 or 81 frames at 16fps).

**Resolution:**
- All clips must be rendered or resized to **1360x768** (landscape) or **768x1360** (portrait) to match the training resolution.
- Mismatched aspect ratios between training and inference will cause distortion. The training framework supports bucket-based multi-resolution, but pinning to a single resolution maximizes consistency for this sprint.

**Action Parity:**
- Maintain roughly even distribution across core action categories.
- If the dataset has 15 walk cycles but only 2 jumps, the model will heavily bias toward walking.
- Target: minimum 4-5 clips per action type.

**Suggested action categories:**
- Walking / running
- Waving / gesturing
- Idle breathing / blinking
- Jumping / bouncing
- Head turns / looking around
- Emotional expressions (happy, sad, surprised)
- Object holding (if applicable)

### 4.3 Background Strategy: 70/30 Mixed Split

| Background Type | Percentage | Purpose |
|-----------------|-----------|---------|
| **Neutral flat color** (50% gray) | 70% | Forces DiT to focus on character physics and edge boundaries. Eliminates background entanglement. |
| **Actual show backgrounds** | 30% | Teaches the model environmental isolation — "backgrounds exist and stay still." Prevents hallucination of static elements during inference. |

**Rendering instructions for the client's team:**
- The 70% neutral clips: render the character layer from After Effects against a solid 50% gray (#808080) background.
- The 30% environment clips: render using actual show backgrounds/props from the existing library.
- Both sets: ensure the character layer is cleanly separated (no background bleed at edges).

---

## 5. Captioning Pipeline

### 5.1 Architecture: Two-Tier Natural Language Captions

CogVideoX uses a frozen T5-XXL text encoder. T5 thrives on dense, natural-language sentences — NOT comma-separated keyword tags. Every caption follows a strict two-tier structure.

#### Tier 1: The Fixed Identity Anchor (Hardcoded Prefix)

A manually crafted, 15-20 word natural language block describing the character's core identity. This block is **programmatically injected** into the beginning of every caption across the entire dataset.

**Example (single character):**
> "A stylized 2D cartoon animation of Pogo, a pudgy penguin with thick black outlines, a white belly, and a red scarf,"

**Example (multi-character — Tier 1 scene):**
> "A stylized 2D cartoon animation featuring Pogo, a pudgy penguin with thick black outlines and a red scarf, positioned on the left third of the frame, and Bubbles, a pudgy penguin with a blue hat, positioned on the right third of the frame,"

**Critical:** Multi-character captions include explicit spatial anchoring ("positioned on the left/right third of the frame") to give T5 a geometric scaffold for maintaining character separation.

#### Tier 2: The Variable Action/Pose Suffix (Automated VLM + Manual Pruning)

Generated by a Vision-Language Model (GPT-4o-Vision or MiniCPM-V), with different prompting strategies per asset type:

| Asset Type | VLM Instruction | Suffix Focus |
|------------|-----------------|--------------|
| **Static T-poses / turnarounds** | "Describe ONLY spatial relationships, camera angle, and limb placement. Do NOT describe what the character looks like." | Angle, pose geometry |
| **Video clips (neutral BG)** | "Describe ONLY the temporal dynamics, speed, direction, and movement style. Do NOT describe the character's appearance." | Motion, timing, physics |
| **Video clips (show BG)** | Same as above, PLUS: "Explicitly state that the background environment and all props remain completely stationary and static." | Motion + environmental lock |

#### Tier 3: The Pruning Phase (Anti-Entanglement)

After VLM caption generation, an automated pruning pipeline removes any visual/identity descriptions from the suffix that would conflict with the Fixed Identity Anchor. This prevents the model from receiving contradictory or redundant identity signals.

**Pruning Implementation (built in Week 2, Day 2-3, alongside captioning pipeline):**

The pruning script uses a **two-stage approach** — a fast keyword blocklist for obvious matches, followed by a lightweight NLP pass for semantic deduplication:

1. **Stage 1 — Keyword blocklist:** A manually curated list of terms that appear in the Identity Anchor (e.g., "penguin," "pudgy," "red scarf," "thick outlines," "white belly") and their synonyms/variants (e.g., "azure scarf," "crimson scarf," "bird," "avian," "plump"). Any sentence or clause in the VLM suffix containing a blocklist term is removed.

2. **Stage 2 — Semantic similarity check:** Each remaining suffix sentence is compared against the Identity Anchor using a lightweight sentence-embedding model (e.g., `all-MiniLM-L6-v2` via `sentence-transformers`). Sentences with cosine similarity > 0.7 to any anchor clause are flagged for removal. This catches rephrasings the keyword blocklist misses (e.g., "a rotund bird wearing neckwear" → semantically similar to "a pudgy penguin with a red scarf").

3. **Stage 3 — Manual spot-check:** 10% of pruned captions are manually reviewed in Week 1 to calibrate the blocklist and similarity threshold before full dataset processing.

**Pruning targets:** Color descriptions, body part descriptions, accessory mentions, species identification, art style descriptions — anything already covered by the Anchor.

**Built by:** The sprint lead (Saksham's contractor), not the client's team. Delivered as a reusable Python script included in the handoff package.

### 5.2 Caption Examples (Final Format)

**Static T-Pose:**
> "A stylized 2D cartoon animation of Pogo, a pudgy penguin with thick black outlines, a white belly, and a red scarf, standing stationary in a perfect profile view, with its left flipper resting flat against its side against a neutral gray background."

**Video Clip (Neutral Background):**
> "A stylized 2D cartoon animation of Pogo, a pudgy penguin with thick black outlines, a white belly, and a red scarf, walking forward in a continuous loop, utilizing a bouncy squash-and-stretch motion, while waving its right flipper enthusiastically."

**Video Clip (Show Background):**
> "A stylized 2D cartoon animation of Pogo, a pudgy penguin with thick black outlines, a white belly, and a red scarf, jumping excitedly with exaggerated vertical squash-and-stretch, while the background kitchen environment and surrounding props remain completely stationary and static."

**Multi-Character Tier 1 (Neutral Background):**
> "A stylized 2D cartoon animation featuring Pogo, a pudgy penguin with thick black outlines and a red scarf, positioned on the left third of the frame, waving its right flipper, and Bubbles, a pudgy penguin with a blue hat, positioned on the right third of the frame, dancing with a rhythmic side-to-side motion, against a neutral gray background."

---

## 6. Training Configuration

### 6.1 LoRA Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LoRA Rank** | 64 | Sweet spot for single-character + motion. Rank 16 too weak for volume encoding. Rank 128 overfits on 60-80 clips. |
| **LoRA Alpha** | 32 or 64 | Must be equal to or half of Rank. THUDM default Alpha=1 with Rank 64 produces a LoRA that learns nothing in diffusers (scaling factor = Alpha/Rank). |
| **Learning Rate** | 3e-5 | Conservative but safe. CogVideoX1.5-5B is highly saturated — 1e-4 or higher causes color burn-out and structural melting by step 1,500. |
| **LR Scheduler** | Cosine annealing to 1e-6 | Aggressive identity mapping in early steps, smooth refinement of edges and temporal physics in later steps. |
| **Optimizer** | 8-bit AdamW | Crucial for fitting 5B model + Rank 64 LoRA into single A100 80GB VRAM. |
| **Batch Size** | 1 | VRAM constraint. |
| **Gradient Accumulation** | 4 | Simulates effective batch size of 4. Smooths gradient updates without VRAM crash. |
| **Training Steps** | 4,000 (with early stopping evaluation) | ~80 epochs over ~50 training windows. Golden checkpoint expected between steps 1,500-2,500. |
| **Checkpointing** | Every 500 steps | 8 checkpoints total. Visual evaluation at each. Overfitting signal expected after step 2,500-3,000. |
| **Training Framework** | [Passenger12138/CogVideoX1.5-5B-I2V-v1.5-lora-train](https://github.com/Passenger12138/CogVideoX1.5-5B-I2V-v1.5-lora-train) + HuggingFace `diffusers` (source branch) | Purpose-built for CogVideoX1.5 I2V LoRA. Includes bucket-based multi-resolution, corrected RoPE, and fixed OFS embedding. |
| **Model Checkpoint** | `THUDM/CogVideoX1.5-5B-I2V` | Pinned version — do not substitute with v1.0 (`THUDM/CogVideoX-5b-I2V`). |
| **Precision** | BF16 | CogVideoX1.5-5B models were trained in BF16. Training and inference must match. |
| **Training Resolution** | 81 frames x 768 x 1360 | Matches inference resolution. VRAM ~35GB on A100 80GB. |

### 6.2 Training Execution Strategy

**Per-run duration:** 10-14 hours on a single A100 80GB.
**Planned iterations:**

| Run | Purpose | Variables Changed |
|-----|---------|-------------------|
| Run 1 | Baseline | Default config above |
| Run 2 | Learning rate sweep | Test 1e-5 and 5e-5 if Run 1 shows under/overfitting |
| Run 3 | Dataset composition | Adjust 70/30 background ratio if environmental artifacts appear |
| Run 4 (if needed) | Caption refinement | Adjust identity anchor wording if character drift persists |
| Run 5 (if needed) | Rank adjustment | Test Rank 32 or 128 if capacity issues identified |

**Golden checkpoint selection:** Visual evaluation of outputs at each 500-step checkpoint. Select the checkpoint that maximizes character fidelity + motion smoothness before overfitting onset.

---

## 7. Inference & Post-Processing Pipeline

### 7.1 Full Pipeline (ComfyUI Node Chain)

The entire inference pipeline is built as an automated ComfyUI workflow:

```
[Artist's First Frame] 
    → [CogVideoX1.5-5B I2V Generation + Latent Loop Closure] (81 frames @ 16fps, loop closure integrated into denoising at step 75-85%)
    → [VAE Decode] (latent → pixel space)
    → [RIFE Frame Interpolation] (16fps → 24fps)
    → [LoRA-Guided Cleanup Pass] (denoise 0.15-0.20 on interpolated frames only)
    → [GIF Export]
```

**Critical pipeline ordering note:** Latent loop closure MUST occur before VAE decode — it operates in latent space. Once decoded to pixels, latent-space operations are no longer possible. The loop closure is integrated directly into the CogVideoX denoising process (injecting the first frame's latent at step 75-85%), making it part of the generation step rather than a separate post-processing step.

### 7.2 Step-by-Step Breakdown

| Step | Operation | Tool/Node | Time | Output |
|------|-----------|-----------|------|--------|
| 1 | Artist provides composited layout as first frame | Manual (existing pipeline) | N/A | PNG first frame (1360x768) |
| 2 | CogVideoX1.5 I2V generation + latent loop closure | CogVideoX1.5-5B-I2V + trained LoRA + latent wrapping (integrated at denoising step 75-85%) | ~5-8 min | 81 frames @ 16fps (~5 sec), seamlessly looping in latent space |
| 3 | VAE decode | CogVideoX VAE decoder | ~15-30 sec | 81 pixel-space frames @ 16fps |
| 4 | Loop quality check | SSIM comparison: frame 1 vs final frame | ~5 sec | If SSIM < 0.92 → trigger fallback (Step 4b) |
| 4b | Loop closure fallback (if needed) | FILM/RIFE VFI cross-fade on decoded pixel frames | ~1-2 min | Alternative seamless loop |
| 5 | Frame interpolation | RIFE VFI ComfyUI node (16fps → 24fps) | ~30-60 sec | ~121 frames @ 24fps (~5 sec) |
| 6 | Artifact cleanup | img2img with LoRA @ 0.15-0.20 denoise on interpolated frames only | ~2-3 min | Cleaned 24fps sequence |
| 7 | Export | GIF encoder | ~15-30 sec | Final production GIF (1360x768 @ 24fps) |

**Total per-GIF (single generation):** ~10-15 minutes

### 7.3 Frame Rate Bridge: 16fps → 24fps

| Component | Detail |
|-----------|--------|
| **Native CogVideoX1.5 output** | 16fps @ 1360x768 (81 frames = ~5 seconds) |
| **Target output** | 24fps (client's production standard) |
| **Interpolation ratio** | 1.5x (significantly easier than 3x from 8fps) |
| **Interpolation model** | RIFE (ComfyUI-Frame-Interpolation node suite) |
| **Artifact risk zones** | Fast flipper motion, scarf physics, thin outlines |
| **Artifact mitigation** | LoRA-guided img2img cleanup pass at denoise 0.15-0.20 |

### 7.4 Looping GIF Strategy

**Primary: Latent-Space Loop Closure (Option D)**

The CogVideoX1.5 I2V model begins denoising from the artist's first frame. At approximately step 75-85% of the denoising schedule, a fraction of the initial frame's latent representation is injected back into the latent space of the final 6-10 frames. During the remaining 15% of denoising, the DiT naturally converges the character's geometry back toward the first frame's structure — achieving seamless loop closure entirely in latent space with zero ghosting.

Implemented via ComfyUI advanced sampling nodes (KSampler_Select + custom latent noise injection). This occurs BEFORE VAE decode — once decoded to pixel space, latent operations are no longer possible.

**Fallback: Cross-Fade + VFI (Option C)**

For complex actions where latent injection causes structural artifacts (detected when SSIM between first and last decoded frame drops below 0.92):

1. The full 81-frame generation is already decoded to pixels
2. Overlap the final 12 frames with the first 12 frames (~15% of total sequence)
3. Pass this overlap window through the FILM/RIFE VFI model
4. VFI calculates optical flow vectors and creates morphing intermediate frames
5. Result: seamless loop without transparency artifacts

The ComfyUI pipeline auto-selects the fallback if the primary method produces a loop SSIM below 0.92 after VAE decode.

### 7.5 Batch Generation & Curation Workflow

The pipeline is NOT a one-click-one-result tool. It is a **batch curation system:**

1. Layout artist drops completed first frame (1360x768 PNG) into a designated watch folder
2. ComfyUI pipeline automatically generates **4 variations** using different random latent seeds
3. A100 runs unattended for ~50-60 minutes
4. Animation director reviews 4 variations, selects the one scoring >= 3.0 on the rubric
5. Selected GIF is exported to the delivery folder

**At a conservative 25% hit rate (1 in 4):** effective human time per GIF = minutes of review. Compute cost per approved GIF = ~$1.50-2.00.

---

## 8. Multi-Character Technical Strategy

### 8.1 How Single-LoRA Handles Multiple Characters

The trained LoRA primarily encodes the **primary character's volume and the animation style/physics.** For multi-character scenes:

- The hand-drawn first frame establishes both characters' appearances via I2V conditioning
- The LoRA handles motion style and temporal physics for both subjects
- Spatial captioning ("positioned on the left/right third") provides T5 with geometric scaffolding

### 8.2 Known Failure Mode: Spatial Token Diffusion

CogVideoX's 3D causal attention applies globally across all visual tokens. Over time, patches containing Character A's features mathematically influence patches containing Character B's features. This causes:

- **Identity collapse:** Accessories merge, colors average across characters
- **Motion entanglement:** Attention heads fail to separate limb ownership between characters

**Mitigation for Tier 1 (committed):** Spatial separation (>30% canvas width) prevents meaningful cross-attention interference for 3-5 second GIFs.

**Mitigation for Tier 2 (stretch):** Stronger spatial captioning + shorter generation duration (2-3 seconds) to limit temporal drift.

### 8.3 Multi-Character Stress Test (Week 5, Days 4-5)

- Test 3-5 multi-character layouts from the client's animatics
- Measure frame-by-frame identity drift of the secondary character
- Document **proximity-to-bleed threshold** (e.g., "characters interact safely if separated by >30% canvas width, cross-contaminate upon occlusion")
- Results feed directly into the Phase 1 Multi-Character Roadmap

---

## 9. Evaluation Framework

### 9.1 Quantitative Metrics (Automated)

All thresholds are **calibrated against the client's own hand-animated content** in Week 1. The client's existing 150+ animations are run through the same scoring scripts to establish the "human animator baseline."

| Metric | What It Measures | Tool | Calibration Method |
|--------|-----------------|------|--------------------|
| **FID** (Frechet Inception Distance) | Statistical similarity between AI frames and client's hand-drawn frames | `clean-fid` library | Run against client's animation library; AI must match or beat |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Per-frame perceptual drift from first frame — character identity stability | `lpips` library | Calibrate against client's own squash-and-stretch deformation (e.g., if human animators hit 0.35 during a wave, AI threshold = 0.35) |
| **Temporal Consistency** | Frame-to-frame optical flow stability — detects flickering, jitter, jumps | RAFT optical flow + variance | Calibrate flow variance against client's hand-animated clips |
| **Loop SSIM** | Structural similarity between first and last frame of looping GIF | SSIM library | Target: >0.92 |

#### HeliosBench Drifting Metrics (Temporal Degradation — Phase 1 Critical)

Adopted from [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios/tree/main/eval). These metrics measure **start-to-end quality degradation** — how much a video deteriorates over its duration. Computed as `|M(first 15% of frames) - M(last 15% of frames)|`. Lower = better (less drift). Scored 1-10.

| Metric | What It Measures | Tool | Calibration Method |
|--------|-----------------|------|--------------------|
| **Drifting Aesthetic** | Does the art style / visual quality degrade by the end of the clip? | CLIP + LAION Aesthetic (first 15% vs last 15%) | Calibrate against client's hand-animated clips — expect near-zero drift in human animation |
| **Drifting Motion Smoothness** | Does motion become jittery, stiff, or unnatural over time? | AMT (first 15% vs last 15%) | Calibrate against client's clips |
| **Drifting Semantic** | Does the character / scene lose meaning or coherence by the end? | ViCLIP (first 15% vs last 15%) | Calibrate against client's clips |
| **Drifting Naturalness** | Does the overall naturalness of motion degrade temporally? | VLM-based scoring (first 15% vs last 15%) | Calibrate against client's clips |

**Why these matter:** Our core metrics (FID, LPIPS, SSIM, RAFT) measure per-frame or frame-to-frame quality but do NOT detect gradual temporal degradation. A GIF could score perfectly on LPIPS frame-by-frame yet still show the character's scarf slowly changing color from start to end. The drifting metrics catch this directly. For 3-5 second GIFs, drifting is a minor concern. For Phase 1's 30-second clips, it becomes the primary failure mode.

**Sprint integration:** HeliosBench evaluation scripts are installed in Week 1 alongside the core evaluation pipeline. Drifting baselines are established against the client's hand-animated content in Week 2. All Week 5 deliverables include drifting scores alongside the core metrics.

**Critical calibration note:** Because this is stylized 2D animation with intentional squash-and-stretch deformation, raw LPIPS/RAFT thresholds from photorealistic benchmarks will produce **false negatives** — flagging beautiful cartoon physics as "identity drift." The Week 1 calibration against the client's own content eliminates this trap. The HeliosBench drifting metrics are similarly calibrated — human-animated clips establish what "acceptable" temporal variation looks like in this art style.

### 9.2 Qualitative Rubric (Human Review — Structured)

Presented to the client in Week 1 as the **Sprint Acceptance Criteria.** Signed off before any training begins.

| Dimension | 1 (Fail) | 2 (Weak) | 3 (Acceptable) | 4 (Production-Ready) |
|-----------|----------|----------|-----------------|---------------------|
| **Character Identity** | Unrecognizable | Recognizable but proportions wrong | Correct proportions, minor accessory drift | Indistinguishable from hand-drawn |
| **Motion Quality** | Static or chaotic | Moves but stiff/robotic | Smooth with minor artifacts | Natural squash-and-stretch |
| **Loop Closure** | Visible hard cut | Noticeable stutter | Slight hesitation | Perfectly seamless |
| **Background Stability** | Warping/melting | Subtle drift | Mostly static, minor shimmer | Rock solid |
| **Overall Aesthetic** | Clearly AI-generated | AI-visible upon inspection | Blends with existing library | Could ship today |

**Pass Bar:** Average score >= 3.0 across all dimensions. No single dimension may score 1.

### 9.3 Production Throughput Metrics (Week 5 Deliverable)

| Metric | Description |
|--------|-------------|
| **Generation success rate** | % of raw outputs passing >= 3.0 rubric without re-generation |
| **Time-per-GIF** | Average wall-clock from first-frame input to approved looping GIF |
| **Cost-per-GIF** | GPU compute cost per approved output |
| **Human-vs-AI comparison** | Side-by-side pipeline throughput comparison (framed as system throughput, not individual performance) |

---

## 10. Week-by-Week Execution Plan

### Week 1: Infrastructure Setup & Client Asset Intake

**Goal:** Production-ready environment operational. Client assets received and organized.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1-2** | Build and push Docker image: CogVideoX1.5-5B-I2V (`THUDM/CogVideoX1.5-5B-I2V`) + diffusers (source branch) + Passenger12138 LoRA trainer + ComfyUI + RIFE + HeliosBench evaluation scripts + all deps, version-pinned. Provision RunPod A100 80GB instance. Create persistent network volume. | Working Docker image on Docker Hub. RunPod instance operational. |
| **Day 2-3** | Client asset intake: T-poses, turnarounds, AE project files, character name/accessory documents. Begin organizing raw asset library. | Raw asset library organized and accessible on persistent volume. |
| **Day 3-5** | Begin clip extraction from client's 150+ animation library. Apply micro-action extraction rules. **Client's team renders clips**: 70% against neutral gray (#808080), 30% against show backgrounds. All clips at 1360x768 resolution. Target 60/25/15 single/Tier1/Tier2 split. | Clip extraction in progress. First batch of clips rendered and delivered. |
| **Day 5** | Present qualitative evaluation rubric to client for sign-off as Sprint Acceptance Criteria. | Rubric approved before any model outputs are generated. |

**Buffer note:** Clip rendering by the client's team may extend into Week 2. This is acceptable — captioning cannot begin until clips are delivered, but infrastructure and baseline testing can proceed in parallel.

### Week 2: Data Curation, Captioning, Zero-Shot Baseline & Pipeline Validation

**Goal:** Complete the curated dataset. Establish baselines. Validate training pipeline.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1-2** | Finalize clip curation: receive remaining clips from client. Validate action parity across categories. Verify 60-80 total clips at 1360x768. Resize/reformat any non-conforming clips. | Complete curated clip library with action parity confirmed. |
| **Day 2-3** | Run captioning pipeline: construct identity anchors per character, run VLM (GPT-4o) on all clips with type-specific prompts, build and execute pruning script (keyword blocklist + semantic similarity), manual spot-check on 10% sample. | Fully captioned dataset ready for training. |
| **Day 3-4** | **Zero-shot baseline testing:** Feed client's raw hand-drawn layouts into the untrained base CogVideoX1.5-5B-I2V model. Generate 10-15 test clips. Analyze failure modes (beak shrinkage, scarf color drift, background warping, melt on rotation). | Zero-shot baseline outputs + failure mode taxonomy. |
| **Day 4** | Run client's existing hand-animated content through FID/LPIPS/RAFT/SSIM evaluation scripts AND HeliosBench drifting metrics (Drifting Aesthetic, Drifting Motion Smoothness, Drifting Semantic, Drifting Naturalness). Establish human-animator quantitative baselines across all metrics (especially LPIPS thresholds during squash-and-stretch, and drifting baselines for temporal consistency). | Quantitative baselines documented and calibrated — core metrics + drifting metrics. |
| **Day 4-5** | Compile joint image-video dataset: pack T-poses alongside video clips in training-ready format. Upload to persistent volume. Run **diagnostic training run (500 steps)** to verify end-to-end pipeline: VRAM usage, gradient flow, checkpoint saving, no crashes. | Training-ready dataset uploaded. Pipeline validated on hardware. |

### Week 3: CogVideoX LoRA Training (Dedicated — Full Week)

**Goal:** Produce a high-quality, validated LoRA checkpoint with sufficient iteration buffer.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1** | **Run 1 launch (Baseline):** Full training with locked-in config (Rank 64, LR 3e-5, Alpha 32, 4000 steps, bf16, 81x768x1360). Set checkpointing at every 500 steps. Monitor loss curve remotely. | Run 1 in progress (10-14 hours). |
| **Day 2** | **Run 1 completes.** Visual evaluation of outputs at all 8 checkpoints (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000). Identify preliminary golden checkpoint. Diagnose issues: deep-frying, underfitting, motion collapse, character drift. | Run 1 checkpoint series + visual evaluation grid + analysis document. |
| **Day 2 (evening)** | **Run 2 launch:** Adjust based on Run 1 findings. Typical adjustments: LR sweep (1e-5 or 5e-5), Alpha adjustment (32 vs 64), or dataset caption refinement. | Run 2 in progress (10-14 hours, runs overnight). |
| **Day 3** | **Run 2 completes.** Comparative evaluation against Run 1. Identify best checkpoint across both runs. Decide if Run 3 is needed. | Run 2 checkpoint series + comparative evaluation. |
| **Day 3 (evening)** | **Run 3 launch (if needed):** Final refinement — dataset composition adjustment, rank change, or caption revision. | Run 3 in progress (runs overnight). |
| **Day 4** | **Run 3 completes (if run).** Select overall golden checkpoint across all runs. Run automated quantitative evaluation (FID, LPIPS) against Week 2 baselines. | **Final trained LoRA weights (.safetensors).** |
| **Day 5** | **Buffer day.** Emergency re-run if all prior checkpoints underperform. Or: begin preliminary ComfyUI workflow assembly for Week 4. | Slack absorbed or Week 4 head start. |

**Training runs execute overnight** — a 10-14 hour run launched in the evening completes by morning. This allows analysis and the next run to launch the same day, fitting 3 full runs into the week with a buffer day.

### Week 4: Workflow Integration, Frame Rate Bridge & Control

**Goal:** Build the complete ComfyUI production pipeline. Solve the 16fps→24fps gap. Validate at 24fps. First demo to client.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1** | Build the master ComfyUI node workflow with corrected pipeline ordering: I2V input → CogVideoX1.5 generation with latent loop closure → VAE decode → SSIM loop check → RIFE interpolation (16fps→24fps) → LoRA cleanup → GIF export. | Working ComfyUI workflow file (.json). |
| **Day 2** | Implement and test the frame rate bridge: 16fps → 24fps via RIFE + LoRA cleanup at 0.15-0.20 denoise. Tune denoise strength on 5-10 test clips. Verify no ghosting on fast flipper/scarf motion. | Validated 24fps output. Denoise strength calibrated. |
| **Day 3** | Implement latent loop closure nodes (integrated into KSampler denoising at step 75-85%). Test primary strategy. Build auto-fallback: if SSIM < 0.92 after VAE decode, trigger FILM/RIFE VFI cross-fade on decoded pixel frames. | Seamless looping pipeline with dual-path closure + auto-fallback. |
| **Day 4** | **Integration test:** Process 5-8 of the client's actual art layouts through the full pipeline. Test single-character and Tier 1 multi-character scenes. Test against both neutral and complex backgrounds. | End-to-end validated pipeline producing looping 24fps GIFs at 1360x768. |
| **Day 5** | Build batch generation system: watch-folder automation, 4-seed variation generation, output organization. **First demo to the animation director** — must see 24fps, fully looped output. Capture initial feedback. | Batch pipeline operational. Initial client feedback captured. |

**Rollback strategy:** If the golden checkpoint from Week 3 reveals subtle flaws at 24fps (e.g., interpolated frames expose jitter invisible at 16fps), the fallback path is: (1) Try adjacent checkpoints (+-500 steps) from the existing 8 checkpoints per run. (2) Adjust the cleanup pass denoise strength (0.10-0.25 range). (3) If all checkpoints fail, compress Week 5 by 2 days and run one emergency training iteration with adjusted dataset/hyperparameters in the first 2 days of Week 5.

### Week 5: GIF Production Sprint, Multi-Character Testing & Evaluation

**Goal:** Produce the deliverable GIF library. Execute multi-character stress test. Run full evaluation. Compile economic benchmark.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1-2** | **Single-character production run:** Process 15-25 GIF requests through the batch pipeline (4 seeds each). Animation director reviews and scores each using the qualitative rubric. | First production GIF batch + completed scorecards. |
| **Day 3** | Iterate based on feedback: adjust ComfyUI parameters (denoise strength, motion scale, loop closure timing, latent injection step) for any outputs scoring below 3.0. Re-generate failed GIFs. | Refined GIF batch — all single-character outputs meeting >= 3.0 average. |
| **Day 4** | **Multi-character production + stress test:** Process 5-8 Tier 1 multi-character layouts through the batch pipeline. For Tier 2 (stretch), process 2-3 proximate interaction layouts. Measure frame-by-frame identity drift on secondary character. Document proximity-to-bleed thresholds. | Multi-character GIF batch + bleed metrics document. |
| **Day 5** | Run full automated evaluation suite on ALL approved GIFs (single + multi-character): FID, LPIPS, SSIM loop score, temporal consistency, AND HeliosBench drifting metrics (Drifting Aesthetic, Drifting Motion Smoothness, Drifting Semantic, Drifting Naturalness). Compile production throughput metrics: success rate, time-per-GIF, cost-per-GIF, human-vs-AI comparison. | **Complete evaluation dashboard** — core metrics + drifting metrics + qual rubric + economic benchmark for every deliverable. |

### Week 6: Documentation, Handoff & Phase 1 Proposal

**Goal:** Deliver the "glass box" system. Train the client's team. Present Phase 1 roadmap.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1-2** | Write technical report: architecture decisions, training methodology, hyperparameters, dataset composition, evaluation results, blockers encountered, solutions found. Include multi-character roadmap with measured bleed thresholds. Include audio integration note for Phase 1 (30-sec narrative clips). | Technical report document. |
| **Day 2-3** | Build the "glass box" deployment: frozen Docker image with pinned ComfyUI state + trained LoRA + `THUDM/CogVideoX1.5-5B-I2V` weights, deployed on RunPod with 1-click launch (web URL access, no command line). Verify the image reproduces identical outputs to the development environment. | Deployed, accessible production environment. |
| **Day 3-4** | **Managed handoff sessions** (recorded): Train the animation director and technical directors on two flows: (1) **Happy Path** — drop layout into watch folder, run batch, review 4 variations, export approved GIF. (2) **Recovery Path** — VRAM restart (stop/start pod), denoise slider adjustment, "generation melted" troubleshooting. Build FAQ document from their real questions during training. | Recorded training sessions + troubleshooting FAQ. |
| **Day 5** | **Phase 1 proposal presentation:** ML-Ops-as-a-Service retainer model. Present multi-character roadmap (Tier 2→3 progression via regional latent masking). Present 30-second narrative clip scaling roadmap — including Helios 14B as candidate long-form model (conditional on LoRA ecosystem maturity, see companion document: `Helios_Experimentation_Plan.md`). Present audio integration considerations. Present economic impact analysis (Human-vs-AI throughput comparison). | Phase 1 proposal deck + economic analysis + Helios assessment. |

---

## 11. Handoff & Phase 1 Transition

### 11.1 Sprint Deliverables (What the Client Owns)

| Deliverable | Format | Status |
|-------------|--------|--------|
| Trained LoRA weights (golden checkpoint) | .safetensors file | Client-owned IP |
| Complete ComfyUI workflow | .json workflow file | Client-owned |
| Captioned training dataset | Video clips + caption files | Client-owned |
| Docker image (frozen environment) | Docker Hub image | Client-accessible |
| Evaluation scripts + baselines (FID/LPIPS/SSIM/RAFT + HeliosBench drifting) | Python scripts + baseline data | Client-owned |
| Technical report | Document | Client-owned |
| Recorded training sessions | Video files | Client-owned |
| Evaluation dashboard (Week 5 results) | Spreadsheet / web dashboard | Client-owned |
| Phase 1 proposal | Presentation deck | Shared |

### 11.2 The "Glass Box" Deployment Model

The client's creative team operates the pipeline via a simplified interface:

- **1-Click Launch:** Start the RunPod pod → open web URL → ComfyUI loads with all nodes pre-configured
- **Input:** Drop 1360x768 PNG first frame into designated watch folder
- **Output:** 4 GIF variations appear in output folder within ~50-60 minutes
- **Review:** Select best variation using the qualitative rubric
- **No command line required** for standard operation

### 11.3 Phase 1: ML-Ops as a Service (Proposed)

| Service | Description | Cadence |
|---------|-------------|---------|
| **Infrastructure uptime** | Maintain cloud deployment, update node versions safely, manage storage | Ongoing |
| **Continuous retraining** | New characters, new props, new actions → updated LoRA versions | As needed |
| **Workflow iteration** | Scale from 3-sec GIFs → 30-sec narrative clips. Build context-window scaling. | Phased |
| **Multi-character progression** | Tier 2 → Tier 3 engineering (regional latent masking, multi-LoRA switching) | Phased |
| **Troubleshooting** | Debug pipeline failures, dependency conflicts, quality regressions | As needed |

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CogVideoX1.5-5B-I2V fails to learn the character's volume | Low | Critical | Joint image-video dataset with T-poses. If catastrophic by mid-Week 3, pivot to AnimateDiff. |
| Training overfits (memorizes clips rather than generalizing) | Medium | High | 500-step checkpoints. Golden checkpoint selection. Rank 64 cap. Dataset diversity requirements. |
| LoRA "deep-frying" (color burn-out, melting) | Medium | High | Conservative LR (3e-5). Cosine annealing. Visual evaluation at every checkpoint. |
| Spot instance eviction during training | Medium | Medium | 500-step checkpointing. Resume from last checkpoint. Consider reserved instance for critical Week 3 runs. |
| Background hallucination during inference | Medium | High | 70/30 mixed background training. Environmental captioning with explicit static-background instructions. |
| Frame interpolation artifacts (ghosting, splitting) | Medium | Medium | LoRA-guided cleanup pass at 0.15-0.20 denoise. FILM/RIFE fallback. |
| Loop closure failure on complex actions | Medium | Medium | Dual-path strategy (latent wrapping primary, VFI cross-fade fallback). Auto-fallback logic. |
| Multi-character identity collapse | High (Tier 2+) | Medium | Sprint scoped to Tier 1 (committed). Spatial captioning. Stress test in Week 5 for data collection. |
| Client evaluates subjectively without rubric | Medium | High | Rubric signed off in Week 1 as Sprint Acceptance Criteria before any outputs are shown. |
| Pipeline abandoned post-handoff (no ML engineer on team) | High | Critical | Glass box deployment (no CLI). Recorded training sessions. Phase 1 ML-Ops retainer proposal. |
| GPU procurement delayed | Medium | Critical | **Must be requested and provisioned BEFORE Week 1.** See Section 3.1 for detailed specs. |
| CogVideoX license restricts commercial output | Unknown | Critical | **Legal review handled separately.** Must be resolved before sprint begins. |
| Golden checkpoint fails at 24fps (jitter invisible at 16fps) | Medium | High | Try adjacent checkpoints (+-500 steps). Adjust cleanup denoise (0.10-0.25). If all fail, compress Week 5 by 2 days for emergency retraining. |
| Client clip rendering delays (neutral BG renders extend past Week 1) | Medium | Medium | Infrastructure and baseline testing proceed in parallel. Captioning begins on delivered clips immediately. Full dataset not required until end of Week 2. |
| Training resolution mismatch with inference | Low | High | All training clips pinned to 1360x768. Verified during dataset compilation in Week 2. Bucket-based training in Passenger12138 framework handles minor variations. |
| Pruning script misses identity leakage in captions | Medium | Medium | Two-stage pruning (keyword + semantic similarity). 10% manual spot-check. False negatives caught during training via character drift diagnosis. |

---

## 13. Appendix: Glossary of Terms

| Term | Definition |
|------|-----------|
| **CogVideoX1.5-5B-I2V** | Open-source video diffusion model by THUDM/Tsinghua University. 5 billion parameter 3D Diffusion Transformer (DiT). Version 1.5 supports 1360x768 resolution, 16fps, and up to 10 seconds of video. HuggingFace ID: `THUDM/CogVideoX1.5-5B-I2V`. |
| **LoRA** | Low-Rank Adaptation — a lightweight fine-tuning method that trains a small set of additional weights on top of a frozen base model. |
| **I2V (Image-to-Video)** | A generation mode where the model takes a single image as input and produces a video sequence starting from that image. |
| **DiT (Diffusion Transformer)** | A transformer-based architecture for diffusion models, replacing the older UNet backbone. |
| **T5-XXL** | A 4.7 billion parameter text encoder used by CogVideoX to process text prompts into embeddings. |
| **ComfyUI** | A node-based graphical interface for building diffusion model workflows. |
| **RIFE / FILM** | Video Frame Interpolation (VFI) models that synthesize intermediate frames between keyframes to increase frame rate. |
| **FID** | Frechet Inception Distance — measures statistical similarity between generated and reference image distributions. Lower is better. |
| **LPIPS** | Learned Perceptual Image Patch Similarity — measures perceptual difference between images. Lower means more similar. |
| **SSIM** | Structural Similarity Index — measures structural similarity between images. Higher (closer to 1.0) means more similar. |
| **RAFT** | Recurrent All-Pairs Field Transforms — an optical flow estimation model used to measure temporal consistency. |
| **Latent space** | The compressed mathematical representation of images/video inside a diffusion model, before decoding to pixels. |
| **Cosine annealing** | A learning rate schedule that gradually reduces the learning rate following a cosine curve. |
| **Gradient checkpointing** | A memory optimization technique that trades compute time for VRAM by recomputing activations during backpropagation. |
| **VAE (Variational Autoencoder)** | The component that encodes images into latent space and decodes latents back into pixel space. |
| **Squash-and-stretch** | A fundamental principle of 2D animation where characters deform during motion for expressive, cartoon-like movement. |
| **Identity anchor** | The fixed natural-language prefix in every training caption that encodes the character's visual identity. |
| **Spatial Token Diffusion** | The phenomenon where CogVideoX's global attention causes visual features of one character to bleed into another over time. |
| **Golden checkpoint** | The specific training checkpoint that produces the best balance of character fidelity and motion quality before overfitting. |
| **RoPE** | Rotary Position Encoding — a position encoding mechanism used in transformers. The CogVideoX1.5 training framework includes a corrected RoPE implementation. |
| **OFS Embedding** | Optical Flow Score embedding — a conditioning signal in CogVideoX that guides motion intensity. The v1.5 training framework fixes an issue where this was incorrectly set to None. |
| **Real-ESRGAN** | A super-resolution upscaling model capable of upscaling video frames to 4K resolution. Used as optional post-processing. |
| **BF16 (bfloat16)** | A 16-bit floating-point format used for training and inference. CogVideoX1.5-5B models were trained in BF16 and must use BF16 for inference. |
| **Bucket-based training** | A training technique that groups samples by resolution into "buckets," allowing the model to train on multiple aspect ratios without distortion from resizing. |
| **HeliosBench** | An evaluation benchmark from PKU-YuanGroup specifically designed for assessing temporal degradation in video generation. Measures drifting across aesthetic, motion smoothness, semantic, and naturalness dimensions. |
| **Drifting Metrics** | Evaluation metrics that compare the quality of the first 15% of frames against the last 15% of frames, detecting gradual temporal degradation that per-frame metrics miss. |
| **ViCLIP** | A video-language model used in HeliosBench for measuring semantic consistency between generated video and its text description. |
| **Helios** | A 14B parameter autoregressive video diffusion model from PKU-YuanGroup. Achieves real-time long-video generation (19.5 FPS on H100). Candidate for Phase 1 long-form content generation. Apache 2.0 license. |

---

*This plan was developed through systematic technical review covering: base model selection, training architecture, dataset strategy, captioning methodology, hyperparameter configuration, inference pipeline design, frame rate bridging, loop closure mechanics, multi-character handling, evaluation frameworks, production throughput economics, and operational handoff strategy.*
