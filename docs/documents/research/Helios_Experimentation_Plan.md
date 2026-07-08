# Helios 14B — Phase 1 Experimentation Plan

## Long-Form Animation Engine: Feasibility Assessment

**Companion document to:** `Final_Sprint_Plan.md`
**Timeline:** Post-Sprint (Phase 1) — begins after successful 6-week CogVideoX1.5 validation
**Objective:** Determine whether Helios 14B can serve as the production model for scaling from 3-5 second GIFs to 30-60 second narrative animation clips with superior consistency and anti-drifting performance.
**Budget constraint:** None — the goal is the best possible solution with the best consistency.

---

## Table of Contents

1. [Why Helios](#1-why-helios)
2. [Current Blockers & Maturity Timeline](#2-current-blockers--maturity-timeline)
3. [Experimentation Phases](#3-experimentation-phases)
4. [Phase A: Zero-Shot Capability Assessment](#4-phase-a-zero-shot-capability-assessment)
5. [Phase B: LoRA Training Feasibility](#5-phase-b-lora-training-feasibility)
6. [Phase C: Long-Form Pipeline Engineering](#6-phase-c-long-form-pipeline-engineering)
7. [Phase D: Head-to-Head Comparison vs CogVideoX1.5](#7-phase-d-head-to-head-comparison-vs-cogvideox15)
8. [Compute Requirements](#8-compute-requirements)
9. [Decision Framework](#9-decision-framework)
10. [Risk Register](#10-risk-register)

---

## 1. Why Helios

### The Problem Helios Solves

The 6-week sprint validates the AI animation pipeline on **3-5 second looping GIFs.** The client's stated long-term goal is **30-60 second narrative clips** for Instagram short-form content. This is where CogVideoX1.5-5B-I2V hits its architectural ceiling:

| Capability | CogVideoX1.5-5B-I2V | Helios 14B |
|-----------|---------------------|------------|
| **Max native duration** | ~5-10 seconds (81-161 frames) | **60+ seconds** (1452 frames) |
| **Architecture** | Single-pass diffusion (all frames denoised simultaneously) | **Autoregressive chunked** (33 frames/chunk, each conditioned on previous) |
| **Anti-drifting** | None — quality degrades beyond ~5 seconds | **Built-in Easy Anti-Drifting** — explicitly simulates and corrects temporal degradation during training |
| **Inference speed** | ~5-8 min for 5 seconds on A100 | **~4 seconds for 5 seconds** on H100 (19.5 FPS) |
| **VRAM (inference)** | 19-40 GB | **~6 GB** (with group offloading) |
| **Parameters** | 5B | 14B |
| **License** | Custom THUDM (requires commercial review) | **Apache 2.0** (fully permissive) |

### The Core Thesis

For 30-60 second character animation, the problem is NOT generating good individual frames — it's maintaining **temporal consistency over hundreds of frames.** Helios was specifically engineered to solve this exact problem. Its autoregressive architecture with anti-drifting training is a fundamentally different approach than extending CogVideoX1.5's single-pass window.

If Helios's I2V quality matches CogVideoX1.5 AND its anti-drifting holds for stylized 2D animation, it becomes the strictly superior choice for long-form content.

---

## 2. Current Blockers & Maturity Timeline

### As of Sprint Start (Current State)

| Capability | Status | Blocker Severity |
|-----------|--------|-----------------|
| **I2V inference** | Fully supported | None |
| **T2V inference** | Fully supported | None |
| **V2V inference** | Fully supported | None |
| **LoRA training (official)** | Not supported | **CRITICAL** — cannot fine-tune for custom characters |
| **LoRA training (community)** | Not available | **CRITICAL** — no equivalent of Passenger12138's trainer |
| **ComfyUI nodes** | Early/partial | Medium — inference possible but workflow integration immature |
| **diffusers integration** | Supported (inference only) | Medium — training scripts not available |
| **Character LoRAs published** | None | Informational — no community validation of fine-tuning quality |

### Projected Maturity (3-6 Months Post-Sprint)

Based on the trajectory of comparable models (CogVideoX went from release to mature LoRA ecosystem in ~4 months, Wan 2.1 in ~3 months):

| Milestone | Estimated Timeline | Confidence |
|-----------|-------------------|------------|
| Community LoRA training scripts appear | 1-3 months | High — Apache 2.0 license accelerates community adoption |
| diffusers LoRA training integration | 2-4 months | Medium — depends on HuggingFace prioritization |
| ComfyUI nodes mature | 2-3 months | High — active community |
| First published character LoRAs | 2-4 months | Medium |
| Validated I2V LoRA training pipeline | 3-6 months | Medium — I2V LoRA is always slower to mature than T2V |

**Trigger for beginning Phase B:** The first of the following events:
1. Official THUDM/PKU-YuanGroup LoRA training scripts released
2. A community trainer (equivalent to Passenger12138) publishes a validated Helios I2V LoRA pipeline with at least 3 public examples
3. HuggingFace `finetrainers` adds official Helios LoRA support

### Monitoring Strategy

Set a bi-weekly check during Phase 1:
- Monitor [Helios GitHub Issues](https://github.com/PKU-YuanGroup/Helios/issues) for LoRA-related discussions
- Monitor [HuggingFace finetrainers](https://github.com/huggingface/finetrainers) for Helios support PRs
- Monitor CivitAI / HuggingFace for published Helios LoRAs
- Subscribe to PKU-YuanGroup release announcements

---

## 3. Experimentation Phases

The experimentation is structured in four sequential phases. Each phase has explicit **go/no-go criteria** — if a phase fails, subsequent phases are skipped and the findings are documented for future reassessment.

```
Phase A: Zero-Shot Assessment (no training)
    ↓ [Go if I2V quality is promising]
Phase B: LoRA Training Feasibility (first fine-tune)
    ↓ [Go if character fidelity matches CogVideoX1.5 LoRA]
Phase C: Long-Form Pipeline Engineering (30-60 sec)
    ↓ [Go if anti-drifting holds for stylized 2D]
Phase D: Head-to-Head Comparison (production readiness)
    ↓ [Decision: migrate, hybrid, or stay on CogVideoX1.5]
```

**Estimated total duration:** 4-6 weeks (can overlap with Phase 1 operational support)
**Not gated by:** Budget — the priority is finding the best solution for consistency.

---

## 4. Phase A: Zero-Shot Capability Assessment

**Duration:** 3-5 days
**Prerequisite:** None — can begin immediately, even during the sprint (recommended: Week 2 of sprint, alongside CogVideoX1.5 zero-shot baseline)
**Compute:** 1x H100 80GB (or A100 80GB) for inference only
**Goal:** Establish how well base Helios handles the Pudgy Penguins IP without any fine-tuning.

### Test Matrix

| Test | Input | Helios Mode | Duration | What We're Measuring |
|------|-------|------------|----------|---------------------|
| **A1: Single character, short** | Client's hand-drawn penguin layout (1360x768 PNG) | I2V (Helios-Base) | 5 sec (132 frames) | Does Helios preserve the penguin's identity from the first frame? How does it compare to CogVideoX1.5 zero-shot? |
| **A2: Single character, medium** | Same layout | I2V (Helios-Base) | 15 sec (396 frames) | At what point does the character start drifting without fine-tuning? |
| **A3: Single character, long** | Same layout | I2V (Helios-Base) | 30 sec (726 frames) | Does the anti-drifting architecture maintain character identity at the 30-second mark? |
| **A4: Single character, full minute** | Same layout | I2V (Helios-Base) | 60 sec (1452 frames) | Stress test — what does failure look like at maximum duration? |
| **A5: Multi-character, short** | Client's Tier 1 multi-character layout | I2V (Helios-Base) | 5 sec (132 frames) | How does the 14B model handle dual-character identity preservation vs CogVideoX1.5's 5B? |
| **A6: Multi-character, medium** | Same layout | I2V (Helios-Base) | 15 sec (396 frames) | Spatial token diffusion rate — does the larger model resist character bleed longer? |
| **A7: Complex background** | Client's full-scene layout (kitchen, props) | I2V (Helios-Base) | 10 sec (264 frames) | Background stability — warping, prop drift, hallucination |
| **A8: Motion style match** | Layout showing penguin in pre-wave pose | I2V with prompt: "waving flipper in bouncy cartoon style" | 5 sec (132 frames) | Can text prompting guide the model toward the client's specific animation style without LoRA? |

**Total generations:** 8 tests x 3 prompt variations x 2 models (Base + Distilled) = **48 generations**

### Evaluation

Run all outputs through:
1. **Core metrics:** FID, LPIPS, SSIM (against client's hand-drawn frames)
2. **HeliosBench drifting metrics:** All 4 drifting dimensions
3. **Qualitative rubric:** Same 5-dimension rubric from the sprint
4. **Frame-by-frame identity drift tracking:** LPIPS plotted per frame to visualize the degradation curve

### Go/No-Go Criteria for Phase B

| Criterion | Go | No-Go |
|-----------|-----|-------|
| Character recognizable at 5 sec (A1) | Rubric Character Identity >= 2 | Score 1 — model fundamentally cannot interpret 2D stylized characters |
| Character recognizable at 30 sec (A3) | Rubric >= 2 AND drifting metrics within 2x of human baseline | Character unrecognizable by frame 300 — anti-drifting doesn't help |
| Background stability (A7) | Background Stability >= 2 | Severe warping/melting of static elements |
| Motion plausibility (A8) | Motion Quality >= 2 | Motion is frozen, chaotic, or physically impossible |

**Note:** These are intentionally LOW bars (>= 2, not >= 3). We're testing whether the architecture can *fundamentally* handle stylized 2D animation, not whether it's production-ready. Training will close the quality gap — if the architecture can't even approximate the task zero-shot, fine-tuning won't save it.

---

## 5. Phase B: LoRA Training Feasibility

**Duration:** 2-3 weeks
**Prerequisite:** Phase A passes go/no-go AND LoRA training tooling is available (see Section 2 triggers)
**Compute:** 1x H100 80GB (or 2x A100 80GB) for training + inference
**Goal:** Train a Helios LoRA on the Pudgy Penguins dataset and achieve character fidelity comparable to the CogVideoX1.5 sprint LoRA.

### Dataset

Re-use the curated sprint dataset (60-80 clips + T-poses) with the following adaptations:

| Sprint Dataset Element | Helios Adaptation |
|-----------------------|-------------------|
| **Identity anchors** | Re-validate against Helios's text encoder. Helios is built on Wan/Open-Sora Plan lineage — may use a different text encoder than T5-XXL. If so, re-test keyword vs natural language approach. |
| **Caption pruning** | Re-calibrate semantic similarity thresholds against Helios's encoder behavior. |
| **Resolution** | Helios supports flexible resolution. Test at 384x640 (Helios default benchmark res), 768x1360 (CogVideoX1.5 match), and 1280x720 (16:9 standard). |
| **Frame count** | Helios uses 33-frame chunks (8N+1 no longer applies). Adjust clips to multiples of 33. |
| **Background split** | Maintain 70/30 neutral/show background ratio. |

### Training Configuration (Provisional — Subject to Tooling Availability)

| Parameter | Starting Value | Rationale |
|-----------|---------------|-----------|
| **LoRA Rank** | 64 (same as sprint) | Start with validated sprint config. Adjust if 14B model requires different capacity. |
| **LoRA Alpha** | 32 or 64 | Match rank or half-rank. Validate against Helios-specific scaling behavior. |
| **Learning Rate** | 1e-5 to 3e-5 | Start more conservative than sprint — 14B models are more sensitive to LR. |
| **Training Steps** | 4000 | Validate — 14B model may converge faster or slower than 5B. |
| **Checkpointing** | Every 500 steps | Same strategy — golden checkpoint between 1500-2500. |
| **Precision** | BF16 | Standard for diffusion transformers. |

**Important:** These hyperparameters are provisional. The actual training framework may impose different constraints or expose different parameters. This configuration will be adjusted based on whatever tooling becomes available.

### Training Experiments

| Run | Purpose | Key Variable |
|-----|---------|-------------|
| Run B1 | Baseline character training | Default config, 5-sec generations (132 frames) |
| Run B2 | LR sweep | Test 1e-5, 3e-5, 5e-5 |
| Run B3 | Long-form validation | Same LoRA, but generate 15-30 sec (396-726 frames) — does character fidelity hold? |
| Run B4 | Multi-character | Generate Tier 1 dual-character scenes — compare spatial token diffusion rate vs CogVideoX1.5 |
| Run B5 | Motion quality | Test specific action prompts (walk, wave, dance) — compare motion naturalness vs sprint LoRA |

### Evaluation

All outputs evaluated with:
- Core metrics (FID, LPIPS, SSIM, RAFT)
- HeliosBench drifting metrics (all 4 dimensions)
- Qualitative rubric (5 dimensions)
- **Direct comparison against CogVideoX1.5 sprint LoRA outputs** on identical prompts/layouts

### Go/No-Go Criteria for Phase C

| Criterion | Go | No-Go |
|-----------|-----|-------|
| Character fidelity at 5 sec | Rubric Character Identity >= 3 (matches sprint LoRA quality) | Score <= 2 — LoRA training doesn't effectively teach character identity |
| Character fidelity at 15 sec | Rubric >= 2.5 AND drifting metrics within 1.5x of human baseline | Severe drift by 15 seconds — anti-drifting advantage doesn't materialize |
| Motion quality | Rubric Motion Quality >= 3 | Stiff, robotic, or generic motion that LoRA couldn't fix |
| Training stability | Converges within 4000 steps, no deep-frying | Training diverges or produces unusable outputs across all LR values |

---

## 6. Phase C: Long-Form Pipeline Engineering

**Duration:** 2-3 weeks
**Prerequisite:** Phase B passes go/no-go
**Compute:** 1x H100 80GB for inference + pipeline development
**Goal:** Build a production pipeline for 30-60 second animated clips using Helios + trained LoRA.

### Key Engineering Challenges

**1. Chunk Boundary Continuity**

Helios generates in 33-frame chunks. Each chunk is conditioned on the previous chunk. The critical question: **are there visible seams at chunk boundaries?**

| Test | Method | Pass Criteria |
|------|--------|--------------|
| Generate 30 sec clip (22 chunks) | Visual inspection at every chunk boundary (frame 33, 66, 99...) | No visible motion stutter, color shift, or structural pop at boundaries |
| Optical flow analysis at boundaries | RAFT flow magnitude at chunk transitions vs mid-chunk | Flow variance at boundaries within 1.5x of mid-chunk variance |
| Character identity at boundaries | LPIPS frame 32 vs frame 34, frame 65 vs frame 67, etc. | LPIPS at boundaries no worse than LPIPS between any other adjacent frames |

**2. Looping Strategy for Long-Form**

The sprint's latent loop closure technique (injecting first-frame latent at denoising step 75-85%) was designed for single-pass generation. For Helios's autoregressive chunks, looping requires a different approach:

| Strategy | Description | Feasibility |
|----------|-------------|-------------|
| **Last-chunk conditioning** | Feed the first frame as a conditioning signal to the final chunk's denoising process | High — aligns with Helios's I2V input mechanism |
| **Circular chunking** | Treat the video as circular — the last chunk's next-chunk conditioning is the first chunk | Medium — requires modification to the autoregressive loop |
| **Ping-pong + transition** | Generate half the duration forward, reverse for the second half, use VFI to smooth the reversal point | High — works for non-directional motions |
| **Post-hoc cross-fade** | Standard FILM/RIFE overlap blend on decoded pixel frames | High — fallback, same as sprint |

**3. Scene Transitions (New Capability)**

30-second narrative clips require **multiple shots** — something the 3-5 second GIF pipeline never addresses. Helios's autoregressive architecture naturally supports this:

- Chunk 1-5: Wide shot of penguin in kitchen
- Chunk 6-10: Close-up of penguin's face reacting
- Chunk 11-15: Wide shot of second penguin entering

Each "shot" can be initiated by providing a new first-frame layout at the start of the corresponding chunk. This would leverage the client's existing storyboard/animatic pipeline directly.

**Test plan for scene transitions:**
1. Prepare 3-shot storyboard from client's animatics
2. Generate 30-second clip with manual shot breaks at chunk boundaries
3. Evaluate continuity across shot transitions using drifting metrics

**4. Frame Rate Handling**

Helios generates at 16fps (matching CogVideoX1.5). The same RIFE interpolation pipeline (16fps → 24fps) from the sprint applies. However, at 30-60 seconds of content, the interpolation compute time scales linearly:

| Duration | Native Frames (16fps) | After RIFE (24fps) | Interpolation Time (est.) |
|----------|-----------------------|---------------------|--------------------------|
| 5 sec | 80 | 120 | ~30-60 sec |
| 15 sec | 240 | 360 | ~2-3 min |
| 30 sec | 480 | 720 | ~4-6 min |
| 60 sec | 960 | 1440 | ~8-12 min |

### Deliverable

A **ComfyUI workflow for 30-second narrative clip generation** including:
- Multi-shot storyboard input (first frames per shot)
- Helios autoregressive generation with LoRA
- RIFE frame interpolation to 24fps
- LoRA cleanup pass
- Final export

---

## 7. Phase D: Head-to-Head Comparison vs CogVideoX1.5

**Duration:** 1 week
**Prerequisite:** Phase C produces usable 30-second clips
**Goal:** Definitive production readiness assessment — should the pipeline migrate to Helios, run hybrid, or stay on CogVideoX1.5?

### Comparison Matrix

Generate the **exact same outputs** from both models using identical inputs:

| Test | Input | CogVideoX1.5 LoRA | Helios LoRA | Evaluation |
|------|-------|--------------------|-------------|------------|
| **D1: 5-sec GIF (sprint equivalent)** | Client layout, single character | Full sprint pipeline | Helios pipeline | Side-by-side quality at the sprint's own benchmark |
| **D2: 5-sec GIF, multi-character** | Tier 1 layout | Sprint pipeline | Helios pipeline | Character bleed comparison |
| **D3: 15-sec clip** | Client layout | CogVideoX1.5 with extended generation (likely degrades) | Helios | Where CogVideoX1.5 breaks and whether Helios holds |
| **D4: 30-sec clip** | 3-shot storyboard | CogVideoX1.5 (likely unviable) | Helios | Helios's target use case |
| **D5: Speed benchmark** | Same input as D1 | Time full pipeline | Time full pipeline | Wall-clock comparison |
| **D6: Cost benchmark** | Same input as D1 | GPU cost per approved GIF | GPU cost per approved GIF | Economic comparison |

### Evaluation Dimensions

For every test, run:
1. **Core metrics:** FID, LPIPS, SSIM, RAFT
2. **HeliosBench drifting metrics:** All 4 dimensions
3. **Qualitative rubric:** 5-dimension scorecard from the sprint
4. **Animation director blind test:** Present outputs from both models without labels. Director scores each. Reveal labels after scoring.

### Decision Framework

| Outcome | Decision |
|---------|----------|
| Helios >= CogVideoX1.5 on GIFs AND significantly better on 15-30 sec clips | **Migrate to Helios** — it's the superior model at all durations |
| Helios < CogVideoX1.5 on GIFs BUT significantly better on 15+ sec clips | **Hybrid pipeline** — CogVideoX1.5 for GIFs, Helios for long-form narrative |
| Helios ~= CogVideoX1.5 on GIFs AND ~= on longer clips | **Stay on CogVideoX1.5** — migration cost not justified by marginal gains |
| Helios < CogVideoX1.5 at all durations | **Stay on CogVideoX1.5** — Helios not ready for this use case |

---

## 8. Compute Requirements

Budget is not a constraint. The priority is thoroughness and finding the best solution.

### Phase A (Zero-Shot Assessment)

| Resource | Specification | Duration | Estimated Cost |
|----------|--------------|----------|---------------|
| 1x H100 80GB | Inference only | 3-5 days | $500-800 |
| Storage | Network volume for outputs | 1 month | $10 |

### Phase B (LoRA Training)

| Resource | Specification | Duration | Estimated Cost |
|----------|--------------|----------|---------------|
| 1x H100 80GB (or 2x A100 80GB) | Training + inference | 2-3 weeks | $2,000-4,000 |
| Training runs | 5 runs x 14-20 hours each | ~70-100 GPU-hours | Included above |
| Storage | Network volume for checkpoints + outputs | 2 months | $20 |

### Phase C (Pipeline Engineering)

| Resource | Specification | Duration | Estimated Cost |
|----------|--------------|----------|---------------|
| 1x H100 80GB | Inference + pipeline testing | 2-3 weeks | $1,500-3,000 |
| Long-form generation | 30-60 sec clips — higher compute per clip | ~200 test generations | Included above |

### Phase D (Head-to-Head)

| Resource | Specification | Duration | Estimated Cost |
|----------|--------------|----------|---------------|
| 1x H100 80GB (Helios) + 1x A100 80GB (CogVideoX1.5) | Parallel inference | 1 week | $500-1,000 |

### Total Estimated Budget

| Phase | Duration | Cost Range |
|-------|----------|------------|
| Phase A | 3-5 days | $500-800 |
| Phase B | 2-3 weeks | $2,000-4,000 |
| Phase C | 2-3 weeks | $1,500-3,000 |
| Phase D | 1 week | $500-1,000 |
| **Total** | **6-8 weeks** | **$4,500-8,800** |

With 50% experimentation buffer: **$6,750-13,200**

**Recommended allocation: $15,000** — provides full freedom for extended training runs, additional long-form experiments, and resolution/duration scaling tests.

---

## 9. Decision Framework

### Phase Gating Summary

```
Phase A: Zero-Shot
├── PASS → Phase B (when LoRA tooling available)
├── FAIL → Document findings. Reassess in 6 months.
└── PARTIAL → Document specific failure modes. Monitor ecosystem.

Phase B: LoRA Training
├── PASS → Phase C
├── FAIL → Stay on CogVideoX1.5. Document why.
└── PARTIAL → If character fidelity < CogVideoX but motion/drifting better,
              explore hybrid approach.

Phase C: Long-Form Pipeline
├── PASS → Phase D
├── FAIL → Helios not ready for long-form production.
│          Use CogVideoX1.5 for GIFs, wait for Helios maturation.
└── PARTIAL → If chunk boundaries visible, explore mitigation.

Phase D: Head-to-Head
├── Helios wins all → MIGRATE to Helios
├── Helios wins long-form only → HYBRID pipeline
├── Tie → Stay on CogVideoX1.5 (lower migration cost)
└── CogVideoX wins → Stay on CogVideoX1.5
```

### Success Definition

The Helios experimentation is a success if ANY of the following are achieved:

1. **Full migration:** Helios replaces CogVideoX1.5 as the production model for all durations
2. **Hybrid deployment:** Helios handles 15-60 sec content, CogVideoX1.5 handles GIFs
3. **Informed rejection:** Quantitative data proves Helios is not ready, preventing wasted Phase 1 engineering time on the wrong model
4. **Knowledge asset:** Drifting metrics, benchmark methodology, and comparison data become reusable for evaluating future models (Wan 3.x, LTX-3.x, etc.)

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LoRA tooling never matures for Helios | Low (Apache 2.0 attracts community) | Critical — blocks Phases B-D | Monitor bi-weekly. If no tooling after 6 months, pivot to Wan 2.2 (which has mature LoRA support and similar architecture lineage). |
| Helios I2V quality genuinely inferior to T2V | Medium (acknowledged by authors) | High — our pipeline is I2V-first | Detected in Phase A. If I2V is fundamentally weak, no amount of LoRA training fixes architectural limitations. |
| 14B model requires multi-GPU training | Medium | Medium — increases compute cost | Use H100 80GB instead of A100. If still insufficient, explore DeepSpeed ZeRO-2 with 2x A100. |
| Autoregressive chunk boundaries create visible seams in stylized 2D | Medium | High — undermines the anti-drifting advantage | Detected in Phase C. Mitigate with chunk-boundary smoothing (VFI on boundary frames). If severe, may disqualify Helios for this art style. |
| Helios's text encoder behaves differently from T5-XXL | High | Medium — invalidates sprint captioning pipeline | Detected in Phase B dataset prep. May require re-engineering identity anchors and pruning scripts. Budget includes time for this. |
| Animation director prefers CogVideoX1.5 output "feel" despite Helios scoring higher on metrics | Medium | Medium — subjective preference overrides objective data | Phase D includes blind testing. If director consistently prefers CogVideoX1.5, that's a valid signal — artistic judgment matters. |
| Helios project abandoned or development stalls | Low (active as of June 2026, 1891 stars) | Critical — blocks entire plan | Monitor GitHub activity. If no commits for 3+ months, deprioritize. Wan 2.2 serves as backup candidate with similar architecture. |
| Long-form RIFE interpolation introduces cumulative artifacts | Medium | Medium — 30-60 sec = 720-1440 interpolated frames | Test at full duration in Phase C. If cumulative, explore generating at 24fps natively (shorter chunks) or per-chunk interpolation with boundary alignment. |

---

## Appendix: Helios vs CogVideoX1.5 — Architecture Comparison

| Dimension | CogVideoX1.5-5B-I2V | Helios 14B |
|-----------|---------------------|------------|
| **Architecture** | 3D Diffusion Transformer (DiT) — single-pass | Autoregressive DiT — chunked (33 frames/chunk) |
| **Parameters** | 5B | 14B (2.8x) |
| **Text encoder** | T5-XXL (4.7B, frozen) | TBD — likely different (Wan/Open-Sora lineage) |
| **Max frames** | 161 (10 sec @ 16fps) | 1452+ (60+ sec @ 24fps) |
| **Native fps** | 16fps | 16-24fps |
| **Max resolution** | 1360x768 | 1360x768+ (flexible) |
| **I2V support** | Native variant (`THUDM/CogVideoX1.5-5B-I2V`) | Unified model (T2V/I2V/V2V in one) |
| **Anti-drifting** | None | Easy Anti-Drifting (training-integrated) |
| **LoRA training** | Mature (Passenger12138 trainer + diffusers) | Not yet available |
| **ComfyUI** | Mature node ecosystem | Early/developing |
| **Inference VRAM** | 19-40 GB | ~6 GB (with offloading) — 19+ GB standard |
| **Inference speed** | ~5-8 min / 5 sec (A100) | ~4 sec / 5 sec (H100) — ~75x faster |
| **License** | Custom THUDM | Apache 2.0 |
| **Training data** | Established dataset practices | Built on Open-Sora Plan / Wan foundations |
| **Community maturity** | 18+ months, well-documented | 3 months, rapidly growing |

---

## Appendix: Key Links

- [Helios GitHub Repository](https://github.com/PKU-YuanGroup/Helios)
- [Helios Paper (arXiv:2603.04379)](https://arxiv.org/abs/2603.04379)
- [Helios Project Page](https://pku-yuangroup.github.io/Helios-Page/)
- [HeliosBench Evaluation Scripts](https://github.com/PKU-YuanGroup/Helios/tree/main/eval)
- [HuggingFace Model Collection](https://huggingface.co/collections/BestWishYsh/helios)
- [CogVideoX1.5-5B-I2V (Sprint Model)](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)
- [Passenger12138 CogVideoX LoRA Trainer](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train)

---

*This experimentation plan is a companion to the 6-Week Validation Sprint Plan (`Final_Sprint_Plan.md`). The sprint uses CogVideoX1.5-5B-I2V as the production model for the immediate GIF pipeline. This document explores Helios 14B as the long-form successor for Phase 1 and beyond, conditioned on LoRA ecosystem maturity.*
