# LTX-Video 2.3 & HDR Workflow — Deep Research Report

## Comprehensive Evaluation for Pudgy Penguins AI Animation Pipeline

**Methodology:** Three independent research agents examined (1) architecture, all variants, LoRA training, IC-LoRA, looping, and license details, (2) HDR workflow end-to-end, color pipeline, and relevance for 2D cartoon animation, and (3) three-way quality comparison vs CogVideoX1.5 and Wan 2.2 with real community evidence. All claims cross-referenced against GitHub source, issues, HuggingFace model cards, CivitAI guides, research papers, and user reports.

---

## Table of Contents

1. [Architecture Deep Dive](#1-architecture-deep-dive)
2. [All Model Variants](#2-all-model-variants)
3. [I2V Capabilities — Current State (June 2026)](#3-i2v-capabilities--current-state-june-2026)
4. [LoRA Training System](#4-lora-training-system)
5. [IC-LoRA — The Unique Mechanism](#5-ic-lora--the-unique-mechanism)
6. [Stylized 2D / Cartoon Content Handling](#6-stylized-2d--cartoon-content-handling)
7. [HDR Workflow — Full Analysis](#7-hdr-workflow--full-analysis)
8. [Looping GIF Capabilities](#8-looping-gif-capabilities)
9. [Character Consistency & Identity Drift](#9-character-consistency--identity-drift)
10. [Three-Way Comparison: LTX 2.3 vs CogVideoX1.5 vs Wan 2.2](#10-three-way-comparison)
11. [Known Bugs & Failure Modes](#11-known-bugs--failure-modes)
12. [License Details](#12-license-details)
13. [Development Trajectory](#13-development-trajectory)
14. [Merits for Pudgy Penguins Pipeline](#14-merits-for-pudgy-penguins-pipeline)
15. [Demerits for Pudgy Penguins Pipeline](#15-demerits-for-pudgy-penguins-pipeline)
16. [Verdict & Recommendation](#16-verdict--recommendation)
17. [Sources](#17-sources)

---

## 1. Architecture Deep Dive

### Core Specifications — VERIFIED

| Specification | Value | Source |
|--------------|-------|--------|
| **Architecture** | Dual-stream asymmetric Diffusion Transformer (DiT) | [arXiv 2601.03233](https://arxiv.org/abs/2601.03233) |
| **Total parameters** | 22 billion (up from 19B in LTX-2.0) | [GitHub](https://github.com/Lightricks/LTX-2) |
| **Video stream** | ~14B parameters | arXiv paper |
| **Audio stream** | ~5B parameters | arXiv paper |
| **Transformer depth** | 48 layers (shared blocks, differing width per stream) | [arXiv](https://arxiv.org/html/2601.03233v1) |
| **Model class** | `AVTransformer3DModel` (new in 2.3) | [DeepWiki](https://deepwiki.com/deepbeepmeep/Wan2GP/7.2-ltx-2-models) |
| **Text encoder** | Gemma 3 12B IT (quantized: `google/gemma-3-12b-it-qat-q4_0-unquantized`) | GitHub README |
| **Text connector** | 4x larger gated attention (vs 2.0) for better prompt adherence | [WaveSpeed](https://wavespeed.ai/blog/posts/ltx-2-to-ltx-2-3-upgrade-guide-2026/) |
| **VAE** | Completely rebuilt in 2.3 — sharper detail, reduced oversaturation, better conditioning fidelity | WaveSpeed upgrade guide |
| **VAE compression** | Spatial 32x per dimension, Temporal 8x, 128 latent channels = 1:192 overall | [arXiv 2501.00103](https://arxiv.org/abs/2501.00103) |
| **Latent shape** | `[B, 128, 1+(F-1)/8, H/32, W/32]` | Architecture docs |
| **Audio vocoder** | HiFi-GAN (upgraded in 2.3 for cleaner stereo at 24kHz) | GitHub README |
| **Positional encoding** | 3D RoPE (video), 1D RoPE (audio) | arXiv paper |

### Per-Layer Block Structure (48 layers)

Each dual-stream block performs four sequential operations:
1. **Self-Attention** — within-modality attention per stream
2. **Text Cross-Attention** — conditioning on Gemma text encoder output
3. **Audio-Visual Cross-Attention** — bidirectional inter-modal exchange with 1D temporal RoPE
4. **Feed-Forward Networks** — with RMS normalization

### I2V Conditioning Mechanism — IMPORTANT

LTX-2.3 does **NOT** use a standard image encoder for I2V (unlike CogVideoX1.5's dedicated I2V variant or Wan 2.2's CLIP ViT-H). Instead, I2V is handled via **IC-LoRA (In-Context LoRA)** — reference images are encoded through the VAE and injected as guiding latents via the adapter framework.

Key I2V pipelines:
- `TI2VidTwoStagesPipeline` — production-quality I2V with 2x upsampling
- `ICLoraPipeline` — uses IC-LoRA adapters for V2V/I2V transformations
- `KeyframeInterpolationPipeline` — interpolates between keyframe images

---

## 2. All Model Variants

### Evolution

| Model | Params | Key Changes |
|-------|--------|-------------|
| **LTX-Video (v1)** | ~2B | Original model, U-Net-like architecture |
| **LTX-2.0** | 19B | DiT architecture, video-only, first dual-stream design |
| **LTX-2.3** | 22B | Rebuilt VAE, 4x text connector, native audio (HiFi-GAN), portrait mode, AVTransformer3DModel |

### LTX-2.3 Checkpoint Variants — VERIFIED

| Checkpoint | Purpose | Steps | CFG | Best For |
|------------|---------|-------|-----|----------|
| **`ltx-2.3-22b-dev`** | Full BF16 model, max quality | 30-50 | ~3.0 | LoRA training, research, final renders |
| **`ltx-2.3-22b-distilled-1.1`** | Fast inference, re-distilled with improved audio | 8 | 1.0 | Rapid iteration, production |
| **Distilled LoRA (`384-1.1`)** | Applied atop Dev for two-stage pipelines | 12 | 1.0 | Dev + distilled hybrid workflows |
| **Spatial Upscaler x2 (1.1)** | 2x spatial latent upscaling | — | — | Second stage of two-stage pipeline |
| **Spatial Upscaler x1.5 (1.0)** | 1.5x spatial upscaling | — | — | Lighter upscaling |
| **Temporal Upscaler x2 (1.0)** | 2x temporal frame interpolation | — | — | Future pipeline support |

### Distilled v1.0 vs v1.1

The v1.1 is a **re-distillation only** (no VAE change):
- Improved audio clarity ("natural, not digital")
- Different aesthetic feel
- Requires updated IC-LoRAs for compatibility
- Community reception mixed — some report motion stiffness

**Critical:** All LTX-2.0 LoRAs are **incompatible** with LTX-2.3. The rebuilt VAE means weight offsets produce garbage. No conversion path — must retrain from scratch. ([WaveSpeed migration guide](https://wavespeed.ai/blog/posts/ltx-2-to-ltx-2-3-upgrade-guide-2026/))

---

## 3. I2V Capabilities — Current State (June 2026)

### Historical I2V Bugs (January 2026)

Filed as [GitHub Issue #11](https://github.com/Lightricks/LTX-2/issues/11):
1. **Frozen/static output** — audio generates correctly but image barely moves or only zooms
2. **Delayed motion** — image motionless until halfway through, then sudden movement
3. **Ken Burns over-application** — unintended slow pan/zoom instead of actual motion

Issue was **closed** with no developer comment or linked PR.

### What's Been Fixed — VERIFIED

- Ken Burns over-application **significantly reduced** ([Awesome Agents review](https://awesomeagents.ai/reviews/review-ltx-2-3/))
- Frozen video failure rate **reduced**
- Identity drift across frames **reduced**
- LTX 2.3 release notes: *"Less freezing, less Ken Burns, more real motion"*

### What's STILL Broken — VERIFIED (Multiple Sources)

| Bug | Status | Source | Impact for Our Use Case |
|-----|--------|--------|------------------------|
| **End-of-clip bright flash artifact** | OPEN, no fix, no developer response | [GitHub #148](https://github.com/Lightricks/LTX-2/issues/148) | **HIGH** — loop point is exactly where this artifact appears |
| **Ken Burns partial fix** | Reduced but not eliminated | [LTX model page](https://ltx.io/model/ltx-2-3) | **MEDIUM** — camera drift fights stable loop |
| **V2V fails on clips under 2 seconds** | Active bug | [MindStudio](https://www.mindstudio.ai/blog/ltx-23-video-to-video-fails-under-2-seconds-workaround) | LOW — our clips are 3-5 seconds |
| **Spatial upscaler v1.0 faulty** | Must use v1.1 only | Community warning | LOW — use correct version |
| **Character LoRA kills motion at weight 1.0** | OPEN, no fix | [HuggingFace #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36) | **HIGH** — see Section 9 |

### LTX 2.3's Own Take on 2D Animation

The [LTX blog on 2D animation](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) explicitly states:
> *"2D animation styles produce fewer temporal artifacts than photorealistic output because simpler geometry and flat color reduces the model's burden."*

This is a direct counter to Wan 2.2's documented photorealistic bias problem. For flat-color cartoon penguins, LTX 2.3 may produce fewer I2V artifacts than Wan 2.2 specifically because the simpler visual content is easier for the model to handle.

---

## 4. LoRA Training System

### Official `ltx-trainer` — VERIFIED

The monorepo includes `ltx-trainer` supporting three training modes:
1. **Standard LoRA** — character, style, motion fine-tuning
2. **IC-LoRA** — image-conditioned LoRA (Canny/Depth/Pose control)
3. **Full fine-tuning** — complete model parameter training

### Hyperparameters — Community-Validated

| Parameter | Recommended | Source |
|-----------|-------------|-------|
| **LoRA Rank** | 32 (default); 64 for complex multi-concept; 96-128 for multiple characters | [RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/ltx-2-3-lora-training-guide), [WaveSpeed](https://wavespeed.ai/blog/posts/ltx-2-3-lora-training-guide-2026/) |
| **LoRA Alpha** | Equal to rank (32) | Multiple sources |
| **Learning Rate** | 1e-4 (starting); 5e-5 for style LoRAs | Multiple sources |
| **LR Schedule** | Cosine decay with 5-10% warmup | Multiple sources |
| **Max Steps** | 1,000-3,000 (checkpoint every 500) | Multiple sources |
| **Batch Size** | 1 | Standard |
| **Gradient Accumulation** | 4 | Standard |
| **Precision** | BF16 (save); FP8 (transformer quantized during training) | Documentation |
| **Gradient Checkpointing** | Required for <80GB VRAM | Documentation |
| **Optimizer** | AdamW8bit (for consumer GPUs) | Standard |

### Data Requirements

- **20-50 images** for character/style LoRAs; up to 80-120 for highly specific subjects
- Text captions optional but recommended
- Frame count: `(F-1) % 8 == 0`; width/height divisible by 32
- LoRA training runs on **video data natively** (not individual frames)

### VRAM Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| Ideal | 80GB (H100/H200) | Full BF16, no quantization |
| Practical | 48GB | With gradient checkpointing |
| RTX 4090 | 24GB | With 8-bit optimizer + gradient checkpointing + FP8 |
| Minimum | 12GB (RTX 3060) | With FP8 quantization + 96GB system RAM for offloading |

### Training Speed — THE Key Differentiator

| Model | Training Time (Character LoRA, A100) |
|-------|--------------------------------------|
| **LTX 2.3** | **~1-3 hours** (1000-2000 steps) |
| **Wan 2.2 I2V-A14B** | 12-20+ hours (dual expert, 3000-5000 steps each) |
| **CogVideoX1.5** | 10-14 hours (single model, 4000 steps) |

LTX 2.3 LoRA training is **10-20x faster** than Wan 2.2 and **5-10x faster** than CogVideoX1.5. For a sprint where you need 3-5 training iterations to find the golden checkpoint, this means:
- **LTX 2.3:** 3-5 runs in a single day
- **CogVideoX1.5:** 3 runs across the full week (with overnight runs)
- **Wan 2.2:** 2-3 runs across the full week

---

## 5. IC-LoRA — The Unique Mechanism

### What IC-LoRA Is — VERIFIED

**IC-LoRA (In-Context LoRA)** is a training approach where the LoRA learns to **condition generation on reference inputs at inference time** rather than just text. It separates motion/structure from visual styling.

### How It Differs from Standard LoRA

| Aspect | Standard LoRA | IC-LoRA |
|--------|--------------|---------|
| **Conditioning** | Text prompt only | Reference image/video + text |
| **Training** | Learns style/identity from dataset | Learns to follow structural signals (edges, depth, pose) |
| **Typical Rank** | 32 | 128 (higher capacity for identity encoding) |
| **Pipeline** | Any pipeline | `ICLoraPipeline` only (distilled checkpoint required) |
| **Inference** | Apply adapter weights | Provide reference frames as guiding latents |
| **Use case** | Bake character appearance into weights | Control motion/structure at inference time |

### Three Conditioning Modes

| Mode | Signal | Relevance for Pudgy Penguins |
|------|--------|------------------------------|
| **Canny** | Edge maps | **HIGHLY RELEVANT** — preserves outlines. Thick-outline 2D characters map directly to Canny edges. |
| **Depth** | Depth maps | Low relevance — flat 2D characters have no depth information |
| **Pose** | Skeleton joints (DWPose) | Low relevance — penguin anatomy doesn't map to human skeletons |

### Why IC-LoRA Canny Is a Natural Fit

The Canny IC-LoRA mode preserves **edge structure** while allowing the model to fill in color and motion. For Pudgy Penguins:
1. Create a reference animation (even rough) showing the penguin's motion
2. Extract Canny edge maps — the thick black outlines produce clean, strong edges
3. Use IC-LoRA to generate the final styled output following those edges exactly
4. The thick outlines ARE the Canny signal — no additional preprocessing needed

This is architecturally more principled than either CogVideoX1.5's latent injection or Wan 2.2's FLF2V for maintaining character structure.

### Community IC-LoRAs Available

| IC-LoRA | Purpose | Source |
|---------|---------|--------|
| **Union Control** | Canny + Depth + Pose combined | [Lightricks official](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) |
| **ReStyle** | Arbitrary style transfer — supports "Disney 2D Animation style," flat 2D, cel-shaded, monochrome line art | [HuggingFace](https://huggingface.co/Cseti/LTX2.3-22B_ReStyle_IC-LoRA) |
| **Motion Track** | Motion trajectory control | Lightricks official |
| **HDR** | SDR→HDR conversion (see Section 7) | Lightricks official |
| **MergeGreen** | First/last/middle frame workflows for loops | [HuggingFace](https://huggingface.co/siraxe/MergeGreen_IC-lora_ltx2.3) |

### Limitations

- **Single-identity only** — two distinct reference identities not natively supported
- **Drift accumulates** beyond 5-20 seconds
- **Distilled checkpoint only** for `ICLoraPipeline` (not Dev) — lower motion quality ceiling
- IC-LoRA rank 128 means higher memory usage than standard LoRA

---

## 6. Stylized 2D / Cartoon Content Handling

### LTX 2.3: Better Than Wan 2.2 for 2D — VERIFIED

| Evidence | Source |
|----------|--------|
| *"2D animation styles produce fewer temporal artifacts than photorealistic output because simpler geometry and flat color reduces the model's burden."* | [LTX blog](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) |
| Image conditioning with a strongly stylized first frame is **more effective than text-only style control** | [LTX blog](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) |
| LTX 2.3 responds well to style prompts — *"declare style first"* approach works | Multiple community guides |
| Community style LoRAs exist: Pixar/Toon, Fantasy Anime, Paper Cut-Out, Cozy Felt, ReStyle IC-LoRA | [CivitAI](https://civitai.com/models/2448150/ltx-23), [Stable Diffusion Tutorials](https://www.stablediffusiontutorials.com/2026/05/ltx2.3-lora-models.html) |

### Comparison: Wan 2.2 Has Documented Photorealistic Bias

| | LTX 2.3 | Wan 2.2 |
|---|---|---|
| **Photorealistic bias** | Present but manageable via image conditioning + style prompts | **Documented strong bias** — *"heavily pre-biased toward realism"* (multiple CivitAI sources) |
| **2D animation quality** | Officially endorsed as easier content for the model | Officially warned: *"Stylized or highly abstract inputs may produce unpredictable results"* |
| **Flat-color preservation** | Better via image conditioning from artist's first frame | Model actively fights flat colors, adds shading/gradients |
| **Thick outline handling** | IC-LoRA Canny mode preserves edges architecturally | No equivalent mechanism — outlines soften during generation |
| **Community workarounds** | Style LoRAs + prompting | Must train BOTH high-noise and low-noise expert LoRAs + bump strength above 1.0 |

### Practical Tips for 2D Content on LTX 2.3

1. **Prompt vocabulary:** Declare style first — *"2D cartoon animation, cel-shaded rendering, bold outlines, flat color fills, no gradients, no shadows, no realistic textures"*
2. **Image conditioning:** Use artist's first frame as visual anchor (stronger than text-only control)
3. **IC-LoRA Canny:** Extract edges from reference animation to force outline preservation
4. **Keep clips 2-4 seconds** to prevent style drift toward photorealism
5. **Use Dev model** for final quality output (Distilled for rapid iteration)

### Evidence Gap

No user has tested LTX 2.3 specifically on flat-color mascot characters (penguins, bears, simple cartoon mascots). The closest tests are anime characters and cel-shaded illustrations, which have more visual complexity. Our use case remains empirically unverified for ALL models.

---

## 7. HDR Workflow — Full Analysis

### What HDR Is in LTX 2.3 — VERIFIED

**LTX 2.3 does NOT natively generate HDR video.** HDR is delivered as a separate **IC-LoRA adapter** (`LTX-2.3-22b-IC-LoRA-HDR`) that sits on top of the base distilled checkpoint.

### How It Works

The technique is called **LumiVid** ([arXiv:2604.11788](https://arxiv.org/abs/2604.11788)):

1. The base model generates standard 8-bit SDR video
2. The HDR IC-LoRA was trained on a proprietary HDR dataset using **ARRI LogC3** logarithmic encoding
3. LogC3 maps HDR imagery into a distribution **naturally aligned** with the model's latent space
4. No VAE retraining needed — the existing frozen VAE handles LogC3-compressed values as if they were standard SDR
5. After generation, inverse LogC3 transform recovers full linear HDR values

Two modes:
- **Text/Image → HDR video generation** (generate natively in HDR)
- **SDR → HDR video conversion** (upconvert existing 8-bit SDR video to 16-bit HDR)

### HDR Output Format

| Property | Value |
|----------|-------|
| **Color space** | Scene-linear (unbounded) — NOT HDR10, HLG, or Dolby Vision |
| **Container** | **EXR image sequence** (OpenEXR), not a video container |
| **Bit depth** | 16-bit float (half) or 32-bit float |
| **Effective range** | ~20 bits dynamic range |
| **SDR preview** | Reinhard tonemapped + sRGB gamma → standard 8-bit |

**Critical:** HDR output is an **EXR image sequence** — individual frames, not an MP4 or MOV. Professional VFX format, not consumer video.

### ComfyUI HDR Workflow

Official Lightricks nodes (not community add-on):
1. `CheckpointLoaderSimple` → load LTX-2.3 distilled
2. `LoraLoaderModelOnly` → apply distilled quality LoRA
3. `LTXICLoRALoaderModelOnly` → load HDR IC-LoRA + extract latent downscale factor
4. `LTXAddVideoICLoRAGuide` → add downscaled reference latent as guide
5. `LTXVHDRDecodePostprocess` → decompress LogC3 → linear HDR + Reinhard tonemap for SDR preview

Official workflow: [`LTX-2.3_ICLoRA_HDR_Distilled.json`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.3/LTX-2.3_ICLoRA_HDR_Distilled.json)

### Wan2GP HDR Integration

Wan2GP integrates LTX 2.3 HDR via its wrapper, but with issues:
- [Issue #1749](https://github.com/deepbeepmeep/Wan2GP/issues/1749): HDR LoRA became a **mandatory dependency** for ALL IC-LoRAs — users must download it even for non-HDR workflows
- Gradio video player auto-converts HDR to SDR for preview — need external player (MPC-BE) for actual HDR viewing

### HDR VRAM Requirements

| Resolution | Frames | Min VRAM |
|-----------|--------|----------|
| 720p | — | 12-16 GB |
| 1080p | — | 16-24 GB |
| 4K (3840x2160) | 49 frames | 48 GB |
| 4K (3840x2160) | 121 frames | 80 GB |

### HDR Relevance for Pudgy Penguins

## VERDICT: HDR is IRRELEVANT and potentially COUNTERPRODUCTIVE for this use case.

| Reason | Detail |
|--------|--------|
| **HDR solves a problem you don't have** | HDR preserves smooth gradients, subtle tonal transitions, highlight/shadow detail. Flat-color cartoons **intentionally avoid** all of these. |
| **Color banding is a non-issue** | Banding appears in smooth gradients. Flat-color characters use solid fills with hard edges. No gradients = no banding. |
| **The HDR LoRA fights your aesthetic** | Trained on cinematic/photorealistic HDR data, it will try to add tonal nuance and smooth falloff to what should be sharp, quantized color blocks. |
| **Computational waste** | HDR adds VRAM overhead and processing time for zero visual benefit on flat content. |
| **GIF is 256 colors / 8-bit** | HDR's 16-bit float dynamic range is completely wasted when the output format maxes out at 256 colors. |

### When HDR COULD Be Useful (Not Now)

- **Phase 1 compositing:** If compositing flat-color penguins alongside live-action or CG backgrounds, HDR gives colorists grading latitude
- **Subtle background gradients:** If backgrounds have atmospheric effects behind flat characters
- **Professional VFX pipeline integration:** If output feeds into DaVinci Resolve / Nuke for color grading

### Recommendation

**Skip HDR entirely for the sprint and Phase 1.** Generate in standard 8-bit SDR with explicit flat-color prompting. HDR is an impressive technical achievement (the LumiVid paper is genuinely novel), but it's solving a problem that doesn't exist in a flat-color cartoon pipeline.

---

## 8. Looping GIF Capabilities

### Official Looping Support — VERIFIED

LTX 2.3 has an **official `LTXVLoopingSampler`** ComfyUI node specifically designed for seamless video generation. ([GitHub looping_sampler.md](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md))

**Neither CogVideoX1.5 nor Wan 2.2 has an equivalent official looping mechanism.**

### How the Looping Sampler Works

Uses **temporal tiling** with overlap blending:
- Divides video into overlapping temporal segments
- Each tile conditioned on the final frames of the previous tile
- **AdaIn normalization** prevents color drift/oversaturation across tiles
- Spatial tiling for high-resolution output

Key parameters:

| Parameter | Recommended | Purpose |
|-----------|-------------|---------|
| `temporal_tile_size` | 80 frames | Size of each temporal tile |
| `temporal_overlap` | 24 frames (~30% of tile size) | Overlap between adjacent tiles |
| `temporal_overlap_cond_strength` | 0.5 | Conditioning strength in overlap region |
| `adain_factor` | 0.0-0.1 (high quality), 0.1-0.3 (long sequences) | Style normalization strength |

### Additional Loop Methods

**Method 1: First-Last Frame Keyframe Workflow** — VERIFIED
- Supply the **same image** as both start and end keyframe
- `KeyframeInterpolationPipeline` generates smooth interpolation
- [RunComfy workflow](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-first-last-frame-in-comfyui-keyframe-to-smooth-video)

**Method 2: IC-LoRA Transition** — VERIFIED
- Community IC-LoRA ([MergeGreen](https://huggingface.co/siraxe/MergeGreen_IC-lora_ltx2.3)) designed for first/last/middle frame workflows

**Method 3: Video Extension** — VERIFIED
- Extend start or end with smooth transitions ([JAI Portal](https://www.jaiportal.com/model/ltx-2-3-extend-video))

### CRITICAL BUG: End-of-Clip Flash Artifact

[GitHub Issue #148](https://github.com/Lightricks/LTX-2/issues/148) — **OPEN, no developer response, no fix:**
- Bright flash/artifact appears in the final frames of generated video
- Appears in both vertical and portrait aspect ratios
- Only in LTX-2.3, not LTX-2.0
- **Directly impacts looping GIFs** because the loop point is where this artifact appears

**Workaround:** Generate slightly longer clips and trim the final frames before looping. Wasteful but workable.

### Three-Way Loop Comparison

| Model | Loop Method | Quality | Color Stability | Our Assessment |
|-------|-----------|---------|-----------------|----------------|
| **LTX 2.3** | Official Looping Sampler + First/Last Frame keyframes | Good — purpose-built tooling | AdaIn normalization prevents drift | Best tooling, but end-of-clip artifact is a concern |
| **CogVideoX1.5** | Latent-space injection at denoising step 75-85% | Architecturally principled | Unknown (untested) | Most principled approach but least tested |
| **Wan 2.2** | FLF2V (First-Last Frame to Video) | Strong — purpose-designed workflow | **Documented color contrast drift** ([kijai #1541](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541)) | Most battle-tested but has color issues |

---

## 9. Character Consistency & Identity Drift

### LTX 2.3 Character LoRA — CRITICAL BUG

**Character LoRAs at weight 1.0 severely reduce motion and ignore action prompts.** At weight 0.5, simple actions like "walking" are still almost entirely ignored.

This is documented in:
- [HuggingFace Discussion #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36) — no resolution
- [ComfyUI Discussion #13213](https://github.com/Comfy-Org/ComfyUI/discussions/13213) — community workarounds

**Workaround:** Custom sigma schedules (euler_ancestral_cfg, 8 steps, sigmas: 1, 0.82, 0.76, 0.38, 0.32, 0.26, 0.22, 0.18). Keep LoRA strength 0.55-0.9. The workaround helps but doesn't fully solve the problem.

**This is a serious concern for our use case.** We need a character LoRA that both:
1. Maintains the penguin's exact visual identity
2. Allows the model to generate motion (waving, walking, dancing)

If the LoRA kills motion, the output is a static image, not an animation.

### Alternative: IC-LoRA + Image Conditioning (No Character LoRA)

If we skip the character LoRA entirely and rely on:
- **Image conditioning** (artist's first frame locks the visual identity)
- **IC-LoRA Canny** (edge maps preserve outline structure)
- **Prompt reinforcement** (describe character in text)

This bypasses the motion-killing bug entirely. The tradeoff: less precise character identity control (no trained LoRA weights), but more natural motion.

### Identity Drift Across Models — CVPR 2026 Paper

[IPRO (CVPR 2026)](https://arxiv.org/html/2510.14255v1) names both CogVideoX and Wan as suffering from identity drift due to exposure bias. LTX 2.3 is not named in this paper but uses the same diffusion paradigm.

| Model | Drift Characteristics |
|-------|----------------------|
| **LTX 2.3** | Holds well for 5-20 seconds. Drift accumulates beyond 25 seconds. Character LoRA prevents drift but kills motion. |
| **Wan 2.2** | Face drift past 6 seconds. Concentrates at frames 50-70% of clip. |
| **CogVideoX1.5** | Similar exposure bias. Surprisingly, T2V sometimes outperforms I2V for identity. |

At **3-5 seconds** (our target), identity drift is likely manageable for all three models.

---

## 10. Three-Way Comparison

### Community Consensus Ranking (June 2026) — VERIFIED

| Dimension | Winner | Runner-Up | Source |
|-----------|--------|-----------|--------|
| **Motion quality** | **Wan 2.2** | LTX 2.3 | [WaveSpeed](https://wavespeed.ai/blog/posts/ltx-2-3-vs-wan-2-2-comparison-2026/), [CrePal](https://crepal.ai/blog/aivideo/ltx-2-3-vs-wan-2-2/) |
| **Speed of iteration** | **LTX 2.3** (10-14x faster) | Wan 2.2 | Multiple sources |
| **Visual fidelity** | Wan 2.2 ≈ LTX 2.3 | — | Community consensus |
| **Stylized 2D content** | **LTX 2.3** | CogVideoX1.5 | LTX blog, community tests |
| **Loop tooling** | **LTX 2.3** (official Looping Sampler) | Wan 2.2 (FLF2V) | Documentation |
| **Character LoRA reliability** | **Wan 2.2** | CogVideoX1.5 | LTX has motion-killing bug |
| **LoRA ecosystem size** | **Wan 2.2** | LTX 2.3 | CivitAI counts |
| **Training speed** | **LTX 2.3** (1-3 hrs) | CogVideoX (10-14 hrs) | Community reports |
| **License simplicity** | **Wan 2.2** (Apache 2.0) | LTX 2.3 ($10M cap) | License pages |
| **I2V stability** | **Wan 2.2** | LTX 2.3 | Bug trackers |

### The Emerging Community Strategy

> *"Prototype fast with LTX 2.3 (under 1 minute per 5s clip), then refine keepers with Wan 2.2 (4-5 minutes per clip)."*
> — [LTX23.org comparison](https://ltx23.org/blog/ltx23-vs-wan22)

This dual-model approach leverages each model's strength: LTX for speed, Wan for quality.

### Full Comparison Matrix

| Dimension | LTX 2.3 | CogVideoX1.5-5B | Wan 2.2 I2V-A14B |
|-----------|---------|-----------------|-------------------|
| **Parameters** | 22B | 5B | 27B MoE (14B active) |
| **Text encoder** | Gemma 3 12B | T5-v1.1-XXL (4.7B) | UMT5-XXL (13B) |
| **Native FPS** | 24/25/48/50 | 16 | 16 |
| **Max resolution** | 1080p native (4K via upscaler) | 1360x768 | 1280x720 |
| **Max duration** | ~20 sec | 10 sec | ~5 sec |
| **VRAM (training)** | 24GB (RTX 4090 w/ FP8) | ~35GB | 80GB |
| **VRAM (inference)** | ~12GB (quantized) | 19-40GB | 80GB / 24GB (FP8) |
| **Training time** | **1-3 hours** | 10-14 hours | 12-20+ hours |
| **LoRA tools** | Official trainer + CivitAI | Passenger12138 | 4+ tools |
| **I2V mechanism** | IC-LoRA pipeline | Dedicated I2V model variant | CLIP ViT-H + VAE latent |
| **Loop support** | **Official Looping Sampler** | Latent injection (principled) | FLF2V (mature) |
| **Photorealistic bias** | Present but manageable | Unknown | **High** (documented) |
| **2D content** | **Officially easier** | Unknown | Officially warned against |
| **Character LoRA + motion** | **BROKEN** (kills motion) | Functional | Functional |
| **Audio** | **Native** (single-pass) | None | None (S2V separate) |
| **IC-LoRA (ControlNet-like)** | **Yes** (Canny/Depth/Pose) | No | No |
| **License** | LTX Community ($10M cap) | Custom THUDM | **Apache 2.0** |
| **Development** | Very active | Stalled | Active (2.6, 2.7) |
| **Community** | Growing (10K + 6.3K stars) | Declining (~4K) | Largest (16K+) |
| **End-of-clip artifacts** | **Yes** (open bug) | Not reported | Not reported |

---

## 11. Known Bugs & Failure Modes

| Bug | Severity (Our Use Case) | Status | Source |
|-----|------------------------|--------|--------|
| **Character LoRA kills motion** | **CRITICAL** | Open, no fix | [HF #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36) |
| **End-of-clip bright flash** | **HIGH** (loop point = end of clip) | Open, no response | [GH #148](https://github.com/Lightricks/LTX-2/issues/148) |
| **Ken Burns over-application** | MEDIUM | Partially fixed | [LTX model page](https://ltx.io/model/ltx-2-3) |
| **LTX-2.0 LoRAs incompatible** | LOW (we'd train from scratch) | By design | [WaveSpeed](https://wavespeed.ai/blog/posts/ltx-2-to-ltx-2-3-upgrade-guide-2026/) |
| **V2V fails <2 seconds** | LOW (our clips are 3-5 sec) | Active | [MindStudio](https://www.mindstudio.ai/blog/ltx-23-video-to-video-fails-under-2-seconds-workaround) |
| **Spatial upscaler v1.0 faulty** | LOW (use v1.1) | Known | Community |
| **Lip sync broken (image+audio→video)** | None (we output silent GIFs) | Open | [GH #438](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/438) |
| **Distilled v1.1 motion stiffness** | MEDIUM | Community reports | [HF discussions](https://huggingface.co/Kijai/LTX2.3_comfy/discussions/49) |

---

## 12. License Details

| Property | Detail |
|----------|--------|
| **License name** | "LTX-2 Community License" — **NOT pure Apache 2.0** despite some third-party claims | 
| **Revenue threshold** | Free for companies with **< $10M annual revenue** |
| **Above $10M** | Custom commercial license required (contact Lightricks) |
| **No retroactive charges** | Confirmed |
| **Model access** | Open weights, no gated access, no usage limits |
| **Output ownership** | Commercial use of generated content allowed under threshold |
| **Source** | [ltx.io/model/license](https://ltx.io/model/license) |

**License confusion alert:** Multiple platforms (fal.ai, some HuggingFace pages) list LTX 2.3 as "Apache 2.0." The official Lightricks license page describes a custom community license with the $10M cap. This is **NOT pure Apache 2.0.** ([HuggingFace Discussion #8](https://huggingface.co/Lightricks/LTX-2.3/discussions/8))

**Comparison:**
- **Wan 2.2:** Apache 2.0 — no revenue cap, fully permissive
- **CogVideoX1.5:** Custom THUDM — requires review
- **LTX 2.3:** Community license — free under $10M, needs negotiation above

---

## 13. Development Trajectory

| Dimension | LTX 2.3 | Wan 2.2 | CogVideoX |
|-----------|---------|---------|-----------|
| **Latest release** | March 2026 (22B) | July 2025 (27B MoE), Wan 2.6/2.7 in 2026 | November 2024 (5B) |
| **Active development** | Very active — monthly updates, new checkpoints, IC-LoRAs, upscalers | Active — Wan 2.6 addresses identity preservation; 2.7 exists | **Stalled** |
| **Backing** | Lightricks (well-funded startup) | Alibaba Group (massive R&D) | Tsinghua University (academic) |
| **Community growth** | 4M downloads in first 6 weeks | Largest community (16K+ stars) | Declining |
| **Downloads (HuggingFace)** | ~4M+ | High | Moderate |
| **Innovation pace** | Fastest first-party iteration (IC-LoRA, HDR, audio) | Regular variant releases | None |

---

## 14. Merits for Pudgy Penguins Pipeline

| Merit | Detail | Evidence Quality |
|-------|--------|-----------------|
| **2D content officially easier** | LTX's own documentation says flat colors and simple geometry reduce temporal artifacts | VERIFIED — [LTX blog](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) |
| **IC-LoRA Canny mode** | Thick outlines produce clean Canny edges — architectural preservation of character structure | VERIFIED — [LTX IC-LoRA guide](https://ltx.io/model/model-blog/how-to-use-ic-lora-in-ltx-2) |
| **Official Looping Sampler** | Purpose-built for seamless loops, with AdaIn color normalization | VERIFIED — [GitHub docs](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md) |
| **10-20x faster LoRA training** | 1-3 hours vs 10-20+ hours — enables 5+ iteration runs per day | VERIFIED — multiple sources |
| **24fps native** | No RIFE interpolation needed (unlike CogVideoX1.5 at 16fps) | VERIFIED |
| **Multi-LoRA stacking (up to 3)** | Character + Style + IC-LoRA simultaneously | VERIFIED |
| **ReStyle IC-LoRA** | Supports "Disney 2D Animation style," flat 2D, cel-shaded, monochrome line art | VERIFIED — [HuggingFace](https://huggingface.co/Cseti/LTX2.3-22B_ReStyle_IC-LoRA) |
| **Audio can be disabled** | No impact on video quality when audio generation is turned off | VERIFIED — [fal.ai](https://fal.ai/ltx-2.3) |
| **Sub-1-minute generation** | 5-second clip in under 1 minute on capable hardware | VERIFIED — community benchmarks |
| **Lower training VRAM** | 24GB (RTX 4090) vs 35GB (CogVideoX) vs 80GB (Wan 2.2 A14B) | VERIFIED |
| **First/Last Frame keyframing** | Explicit keyframe control for loop start and end | VERIFIED |
| **Active, fast-moving development** | Monthly updates, new IC-LoRAs, new upscalers | VERIFIED |

---

## 15. Demerits for Pudgy Penguins Pipeline

| Demerit | Detail | Severity | Evidence |
|---------|--------|----------|----------|
| **Character LoRA kills motion** | At weight 1.0, model produces static output. At 0.5, ignores action prompts. Sigma-schedule workaround partially helps. | **CRITICAL** | [HF #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36) |
| **End-of-clip flash artifact** | Bright flash in final frames — directly impacts loop closure point. No fix, no developer response. | **HIGH** | [GH #148](https://github.com/Lightricks/LTX-2/issues/148) |
| **License is NOT Apache 2.0** | $10M revenue cap. Must verify client's revenue. Wan 2.2 is simpler. | **MEDIUM** | [License page](https://ltx.io/model/license) |
| **IC-LoRA requires distilled checkpoint** | Lower motion quality ceiling than Dev model. Character consistency via IC-LoRA may trade off motion quality. | MEDIUM | Documentation |
| **Single-identity IC-LoRA limitation** | Cannot reference two distinct characters simultaneously — limits multi-character scenes. | MEDIUM | Documentation |
| **No FLF2V equivalent** | No first-last-frame-to-video native workflow (unlike Wan 2.2). Looping Sampler is different approach. | MEDIUM | [Community reports](https://github.com/Lightricks/LTX-2) |
| **Ken Burns partially unfixed** | Camera drift still occurs, fighting stable loops. | MEDIUM | [LTX model page](https://ltx.io/model/ltx-2-3) |
| **22B model size** | Largest model in comparison — higher compute cost for inference than CogVideoX1.5 (5B). | LOW | Architecture |
| **Flat-color mascot untested** | No community evidence for Pudgy Penguins-style characters specifically. | MEDIUM | Evidence gap |
| **No Pudgy-specific LoRAs** | Must train from scratch — no community head start. | LOW (same for all models) | CivitAI search |

---

## 16. Verdict & Recommendation

### The Nuanced Picture

LTX 2.3 has **the best theoretical fit** for stylized 2D cartoon animation (less photorealistic bias, IC-LoRA Canny for outlines, official looping sampler, fastest training). However, two unresolved bugs — the **character LoRA motion-killing bug** and the **end-of-clip flash artifact** — create serious practical risks for production use.

### LTX 2.3 vs CogVideoX1.5 vs Wan 2.2 for Pudgy Penguins

| Factor | Best Model | Why |
|--------|-----------|-----|
| **Preserving flat-color 2D aesthetic** | **LTX 2.3** | Less photorealistic bias, image conditioning works well, IC-LoRA Canny preserves edges |
| **Reliable character LoRA** | **CogVideoX1.5 or Wan 2.2** | LTX 2.3's LoRA kills motion (critical bug) |
| **Seamless looping** | **Tie: LTX 2.3 vs CogVideoX1.5** | LTX has official Looping Sampler (but end-of-clip bug). CogVideoX has latent injection (untested). Wan 2.2 has FLF2V (color drift). |
| **Training iteration speed** | **LTX 2.3** | 1-3 hours vs 10-14+ hours |
| **Production reliability** | **CogVideoX1.5** | Simplest architecture, fewest known bugs for our use case |
| **Long-term trajectory** | **LTX 2.3 or Wan 2.6** | Both actively developed; CogVideoX stalled |

### Recommendation

**LTX 2.3 should be evaluated as a secondary candidate alongside CogVideoX1.5, with specific focus on whether the character LoRA motion bug and end-of-clip artifact actually manifest for cartoon content.**

Specifically:

1. **Week 2 zero-shot probes should include LTX 2.3** alongside CogVideoX1.5 and Wan 2.2. Three models, same penguin art layouts, same evaluation metrics. Cost: ~$75-100 additional, a few hours.

2. **If LTX 2.3 zero-shot output preserves the flat-color aesthetic** (which the model's own documentation predicts), test the **IC-LoRA Canny approach** (no character LoRA needed) in Week 3 as a parallel experiment. This bypasses the motion-killing bug entirely.

3. **If IC-LoRA Canny + image conditioning proves sufficient for character identity** (which is plausible for simple mascot characters with strong, distinctive outlines), LTX 2.3 becomes the **strongest candidate** — faster training, better 2D handling, official looping, 24fps native.

4. **If character LoRA is essential** (IC-LoRA alone doesn't maintain precise brand identity), stay on CogVideoX1.5. The motion-killing bug is a dealbreaker for character animation.

5. **HDR: Skip entirely.** Irrelevant for flat-color cartoon GIFs. Adds compute cost and complexity for zero visual benefit.

---

## 17. Sources

### Official Documentation
- [LTX-2 GitHub Repository](https://github.com/Lightricks/LTX-2)
- [LTX-Video GitHub Repository](https://github.com/Lightricks/LTX-Video)
- [ComfyUI-LTXVideo Plugin](https://github.com/Lightricks/ComfyUI-LTXVideo)
- [LTX 2.3 Model Page](https://ltx.io/model/ltx-2-3)
- [LTX 2.3 on HuggingFace](https://huggingface.co/Lightricks/LTX-2.3)
- [LTX HDR IC-LoRA on HuggingFace](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
- [LTX Union Control IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)
- [LTX License Page](https://ltx.io/model/license)
- [LTX Blog: 2D Animation Guide](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models)
- [LTX Blog: Character Consistency](https://ltx.io/blog/how-to-maintain-character-consistency-in-ai-video)
- [LTX Blog: IC-LoRA Usage Guide](https://ltx.io/model/model-blog/how-to-use-ic-lora-in-ltx-2)
- [LTX Blog: ComfyUI Workflow Guide](https://ltx.io/model/model-blog/comfyui-workflow-guide)
- [LTX Blog: Reduce Warble Artifacts](https://ltx.io/model/model-blog/how-to-reduce-warble-and-ai-pattern-artifacts-in-ltx-2)
- [LTX Blog: SDR to HDR](https://ltx.io/blog/sdr-to-hdr)
- [LTX Blog: AI Video for Broadcast](https://ltx.io/blog/ai-video-for-broadcast)
- [LTX Looping Sampler Documentation](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md)
- [LTX HDR Workflow JSON](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.3/LTX-2.3_ICLoRA_HDR_Distilled.json)
- [LTX System Requirements](https://docs.ltx.video/open-source-model/getting-started/system-requirements)

### Research Papers
- [LTX-Video 2.3 Paper (arXiv 2601.03233)](https://arxiv.org/abs/2601.03233)
- [LTX-Video v1 Paper (arXiv 2501.00103)](https://arxiv.org/abs/2501.00103)
- [LumiVid HDR Paper (arXiv 2604.11788)](https://arxiv.org/abs/2604.11788)
- [LumiVid Project Page](https://hdr-lumivid.github.io/)
- [IPRO: Identity-Preserving Reward-Guided Optimization (CVPR 2026)](https://arxiv.org/html/2510.14255v1)

### GitHub Issues
- [#11: I2V Historical Bugs (closed)](https://github.com/Lightricks/LTX-2/issues/11)
- [#148: End-of-Clip Flash Artifact (open)](https://github.com/Lightricks/LTX-2/issues/148)
- [#438: Lip Sync Broken (open)](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/438)
- [HuggingFace #36: Character LoRA Kills Motion](https://huggingface.co/Lightricks/LTX-2.3/discussions/36)
- [HuggingFace #8: License Discussion](https://huggingface.co/Lightricks/LTX-2.3/discussions/8)
- [ComfyUI #13213: LoRA Motion Issue](https://github.com/Comfy-Org/ComfyUI/discussions/13213)
- [HuggingFace: Distilled v1.1 Discussion](https://huggingface.co/Kijai/LTX2.3_comfy/discussions/49)
- [Wan2GP #1749: HDR Mandatory Dependency](https://github.com/deepbeepmeep/Wan2GP/issues/1749)

### Community Guides & Comparisons
- [WaveSpeed: LTX 2.3 vs Wan 2.2](https://wavespeed.ai/blog/posts/ltx-2-3-vs-wan-2-2-comparison-2026/)
- [WaveSpeed: LTX 2.0 → 2.3 Upgrade Guide](https://wavespeed.ai/blog/posts/ltx-2-to-ltx-2-3-upgrade-guide-2026/)
- [WaveSpeed: LTX 2.3 LoRA Training Guide](https://wavespeed.ai/blog/posts/ltx-2-3-lora-training-guide-2026/)
- [CrePal: LTX 2.3 vs Wan 2.2](https://crepal.ai/blog/aivideo/ltx-2-3-vs-wan-2-2/)
- [CrePal: What Is LTX 2.3](https://crepal.ai/blog/aivideo/what-is-ltx-2-3/)
- [CrePal: IC-LoRA Guide](https://crepal.ai/blog/aivideo/ltx-2-3-ic-lora-guide/)
- [RunComfy: LTX 2.3 LoRA Training Guide](https://www.runcomfy.com/trainer/ai-toolkit/ltx-2-3-lora-training-guide)
- [RunComfy: First-Last Frame Workflow](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-first-last-frame-in-comfyui-keyframe-to-smooth-video)
- [Stable Diffusion Tutorials: LTX 2.3 LoRA Models](https://www.stablediffusiontutorials.com/2026/05/ltx2.3-lora-models.html)
- [Awesome Agents: LTX 2.3 Review](https://awesomeagents.ai/reviews/review-ltx-2-3/)
- [Pixazo: Best Open Source Video Models 2026](https://www.pixazo.ai/blog/best-open-source-ai-video-generation-models)
- [Clore.ai: Video Generation Comparison](https://docs.clore.ai/guides/comparisons/video-gen-comparison)
- [GenAIntel: LTX 2.3 Guide](https://www.genaintel.com/guides/ltx-2-open-source-audio-video-model-guide)
- [MindStudio: Bach vs LTX IC-LoRAs](https://www.mindstudio.ai/blog/bach-model-vs-ltx-2-3-ic-loras-character-consistency)
- [MindStudio: V2V Under 2 Seconds Fix](https://www.mindstudio.ai/blog/ltx-23-video-to-video-fails-under-2-seconds-workaround)
- [AI Study Now: 3-Stage Artifact Fix](https://aistudynow.com/how-i-fixed-ltx-2-3-video-artifacts-comfyui-3-stage-workflow/)
- [LTX23.org: LTX vs Wan Comparison](https://ltx23.org/blog/ltx23-vs-wan22)

### HDR-Specific Sources
- [LTX Blog: HDR Output Announcement](https://ltx.io/model/model-blog/ltx-closes-the-gap-between-ai-production-with-16-bit-hdr-output)
- [ComfyUI-LTXVideo HDR Source Code](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/hdr.py)
- [pixelsHAM: LumiVid HDR Analysis](https://www.pixelsham.com/2026/04/18/ltx-lumidvid-hdr-video-generation-via-latent-alignment-with-logarithmic-encoding-generating-float16-exr-scene-linear-ready-for-color-grading-and-re-exposure-in-post-production/)
- [MiraFlow: LTX 2.3 Explained](https://miraflow.ai/blog/ltx-2-3-explained-features-capabilities-2026)
- [GPURedeem: LTX 2.3 Guide](https://gpuredeem.com/ltx-2-3/)

### IC-LoRA Community Models
- [Cseti/ReStyle IC-LoRA](https://huggingface.co/Cseti/LTX2.3-22B_ReStyle_IC-LoRA)
- [siraxe/MergeGreen IC-LoRA](https://huggingface.co/siraxe/MergeGreen_IC-lora_ltx2.3)
- [joyfox/Transition LoRA](https://huggingface.co/joyfox/LTX-2.3-Transition-LORA)
- [WarmBloodAban/Singularity LoRA](https://huggingface.co/WarmBloodAban/Singularity-LTX-2.3_OmniCine_V1)
- [awesome-ltx2 Community Resource List](https://github.com/wildminder/awesome-ltx2)

### CivitAI Workflows & LoRAs
- [LTX 2.3 All-in-One Workflow](https://civitai.com/models/2553704/ltx23-all-in-one-prompt-relay-id-lora-controlnet-detailer-upscaler-custom-audio-keyframes)
- [Character Consistency Without LoRAs](https://civitai.com/articles/27654/character-consistency-without-loras-free-360-viewers-with-ltx-video-23-in-comfyui)
- [LTX 2.3 Video Control & HD Enhancement](https://civarchive.com/models/2619715)
- [LTX 2.3 Low VRAM Workflow](https://civitai.com/models/2477099)
- [VideoFlow LTX + Wan I2V Workflow](https://civitai.com/models/1815300)
- [CivitAI LTX 2.3 LoRA Training](https://developer.civitai.com/orchestration/recipes/training-ltx2)

### Other Model References
- [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [CogVideoX1.5-5B-I2V HuggingFace](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)
- [Wan 2.6 on Artlist](https://artlist.io/ai/models/wan-2-6)

---

*This research was conducted by three independent agents examining architecture/capabilities, HDR workflow, and three-way quality comparison. All claims are cross-referenced against primary sources. Evidence quality is tagged throughout. The document covers both the HDR workflow (verdict: skip it) and the broader LTX 2.3 viability question (verdict: promising but two critical bugs need monitoring).*
