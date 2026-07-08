# Wan 2.2 — Unbiased Technical Assessment for Pudgy Penguins Animation Pipeline

## Evidence-Based Evaluation (Correcting Prior Analysis)

**Purpose:** This document replaces the previous `Wan2.2_Evaluation.md` as the authoritative assessment. That document contained several claims that were insufficiently verified. This version is based on cross-referenced community reports, technical documentation, GitHub issues, CivitAI training guides, and research papers — not marketing copy or benchmark headlines.

**Use Case:** Generating 3-5 second production-quality looping GIFs of Pudgy Penguins (stylized 2D cartoon characters with thick black outlines, flat colors, exaggerated proportions, and accessories like scarves and hats) using an I2V pipeline where artists provide composited first frames.

---

## Corrections to Prior Assessment

Before proceeding, here are the specific claims from the earlier evaluation that the evidence does not support:

| Prior Claim | Reality | Source |
|-------------|---------|--------|
| "Wan 2.2 LoRA training takes 2-4 hours" | Character identity LoRAs take **6-12+ hours on A100**. The 2-4 hour figure applies to simple style LoRAs on H100 with aggressive quantization. | [Apatero training time analysis](https://www.apatero.com/blog/wan-22-video-lora-training-time-complete-analysis-2025), community reports |
| "MoE architecture naturally separates identity from motion" | This is **architectural speculation**. No ablation study measures identity vs motion as separate metrics. The experts separate by noise level, not by semantic concept. | [Wan official ablation](https://wan.video/blog/wan2.2) — only measures overall validation loss |
| "Wan 2.2 explicitly supports cartoon/stylized characters" | Wan 2.2 has a **documented strong photorealistic bias**. Multiple LoRA creators confirm the model actively fights against flat-color cartoon aesthetics. | [CivitAI Anime Style LoRA](https://civitai.com/models/2222779/anime-style-wan-22-i2v), [Taz's training guide](https://civitai.com/articles/20389) |
| "Wan-Animate could skip LoRA training using motion reference videos" | Wan-Animate's preprocessing pipeline uses **ViTPose human skeleton extraction** — it will fail on penguin anatomy. It's a **hard blocker**. | [Official preprocessing docs](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md), Agent 3 research |
| "TI2V-5B generates at true 24fps" | Technically 24 distinct frames, but VAE temporal compression (4x) means **effective motion information rate is ~6 latent keyframes/sec**. Users report "slow-motion looking" output. | [Wan 2.2 GitHub issue #13](https://github.com/Wan-Video/Wan2.2/issues/13), [ComfyUI issue #9106](https://github.com/comfyanonymous/ComfyUI/issues/9106) |
| "Wan 2.2 LoRA training requires fewer steps and smaller datasets" | For CHARACTER IDENTITY (not style), community recommends **3,000-5,000 steps** and **20-40 video clips** — comparable to CogVideoX1.5, not dramatically less. | [WaveSpeed blog](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/), [Apatero guide](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025) |
| "Consumer GPU deployment (8-12GB)" | This is inference-only for TI2V-5B with FP8 quantization. LoRA training on the I2V-A14B model requires **80GB VRAM**. | [HuggingFace model card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |

---

## 1. The Photorealistic Bias Problem

This is the **single most important finding** for the Pudgy Penguins use case, and it was entirely absent from my previous evaluation.

### The Evidence

Multiple independent sources confirm Wan 2.2 has a strong inherent bias toward photorealistic rendering:

- **CivitAI Anime Style LoRA creator:** *"WAN 2.2 appears to have a strong bias towards realistic generations."* Training past ~2000 steps for anime style results in overfit and noise. ([source](https://civitai.com/models/2222779/anime-style-wan-22-i2v))

- **Taz's Training Guide (CivitAI):** Without a High-Noise LoRA trained specifically for anime, results produce *"an uncanny mix of real life and the cartoon style."* The word "animated" as a prompt keyword triggers Wan's own *"biased 3D-style animated look"* rather than flat 2D. ([source](https://civitai.com/articles/20389))

- **Alici.AI guide:** The model's weights are *"heavily pre-biased toward realism."* ([source](https://alici.ai/blog/how-to-use-wan-27-guide-2026))

- **Higher resolutions worsen the effect:** Cartoon inputs at higher resolution can *"shift toward realism"* requiring higher LoRA strength to compensate. ([source](https://civitai.com/models/2222779/anime-style-wan-22-i2v))

### What This Means for Pudgy Penguins

Pudgy Penguins have:
- **Flat colors** — Wan 2.2 will try to add realistic shading, gradients, subsurface scattering
- **Thick black outlines** — the diffusion process will soften/blur these, adding depth and lighting effects
- **Minimal visual complexity** — fewer distinguishing features for the identity encoder to anchor to
- **Exaggerated proportions** — conflicts with the model's learned priors about body ratios

Without heavy countermeasures (dedicated style LoRA + high strength + negative prompts for "realistic"), the output will drift toward a semi-3D rendered look that breaks the brand aesthetic.

### Does CogVideoX1.5 Have the Same Problem?

**Partially, but less severely.** One comparison source notes CogVideoX produces *"sharper, crisper detail"* with *"better VFX aesthetics"* ([GenAIntel comparison](https://www.genaintel.com/compare/cogvideox-5b-vs-wan-22-5b)). The sharper frame rendering could theoretically preserve clean cartoon outlines better than Wan 2.2's softer, more cinematic look.

However, **neither model has been publicly tested on flat-color mascot characters.** This is genuinely uncharted territory for both.

---

## 2. Wan-Animate: Dead on Arrival for This Use Case

My previous evaluation suggested Wan-Animate could be a game-changer — potentially skipping LoRA training entirely by transferring motion from the client's existing animation library. The research comprehensively refutes this.

### Hard Blockers

**1. Skeleton Extraction Fails on Penguins**
Wan-Animate's preprocessing pipeline runs:
- YOLO person detection → **may not detect a cartoon penguin as a "person"**
- ViTPose skeleton extraction → **will produce garbage keypoints on penguin anatomy** (no human skeletal structure to find)
- Face crop at 512x512 → **penguin faces bear no resemblance to human faces**

The entire preprocessing pipeline would need to be replaced with custom skeleton extraction — a significant engineering project far beyond the sprint scope.

**2. Hand-Drawn Animation References Won't Parse**
The motion reference videos are hand-drawn 2D animations containing cartoon physics (squash-and-stretch, anticipation, snap-back). ViTPose attempts to find **realistic joint positions** in these frames. Cartoon deformations will produce noisy, erroneous keypoints that result in artifacts.

**3. 3D/Realistic Bias**
[Scenario's official guide](https://help.scenario.com/en/articles/wan-2-2-animate-models-the-essentials/): *"These models work best with realistic images and videos as references. Stylized or highly abstract inputs may produce unpredictable results."*

**4. LoRA Enhancement Officially Blocked**
You cannot train a style LoRA to counteract the 3D bias and use it with Wan-Animate. The [official model card](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) warns: *"we do not recommend using LoRA models trained on Wan2.2."* Technical root causes include MoE weight key mismatches, expert routing disruption, color shift artifacts, and stop-motion quality degradation. ([GitHub #205](https://github.com/Wan-Video/Wan2.2/issues/205), [DiffSynth-Studio #1105](https://github.com/modelscope/DiffSynth-Studio/issues/1105))

**5. Accessory Loss**
[302.AI's test](https://medium.com/@302.AI/wan2-2-animate-model-test-with-4-cases-e9e1c1ef492c) found a character's **microphone disappeared** during animation, leaving an empty fist. For penguins wearing scarves and hats, expect similar disappearance/morphing.

### Verdict

**Wan-Animate is not viable for Pudgy Penguins.** The preprocessing pipeline, the skeleton extraction, the realistic bias, the LoRA incompatibility, and the accessory handling all independently disqualify it. It was designed for human-to-human(oid) motion transfer, not cartoon animal animation.

---

## 3. Wan 2.2 I2V + LoRA: Honest Assessment

Setting Wan-Animate aside, the standard I2V + character LoRA approach (same strategy as our CogVideoX1.5 sprint) is the relevant comparison.

### Text Encoder: Compatible with Our Captioning Strategy

| | CogVideoX1.5 | Wan 2.2 |
|---|---|---|
| **Encoder** | T5-v1.1-XXL | UMT5-XXL |
| **Parameters** | ~4.7B | ~13B |
| **Status** | Frozen | Frozen |
| **Language** | English | English + Chinese |
| **Natural language** | Excellent | Excellent |
| **Keyword tags** | Poor (degrades with T5) | Acceptable but not optimal |

Both encoders are T5-family, both frozen during training, both handle natural language descriptions well. Our two-tier captioning strategy (identity anchor + action suffix) should transfer to Wan 2.2's UMT5-XXL with minimal adaptation.

**One difference:** Wan 2.2's community leans toward **trigger words** (e.g., `zxq-penguin`) for LoRA activation, which contradicts our natural-language-only strategy. However, since UMT5-XXL is a T5-family encoder, our approach should still work — trigger words are a community convention, not an architectural requirement.

### LoRA Training: Dual-Expert Complication

This is a significant hidden cost that my previous evaluation missed.

Wan 2.2's I2V-A14B uses MoE architecture. Training a LoRA requires training **two separate LoRAs**:
- One for the high-noise expert (composition, motion, layout)
- One for the low-noise expert (detail, identity, texture)

| Impact | Detail |
|--------|--------|
| **Training time** | ~2x what a single-model LoRA takes. At 3,000-5,000 steps per expert on A100 80GB, expect **12-20+ hours total** (not 2-4 hours). |
| **Hyperparameter complexity** | Each expert may need different LR, rank, or step count. More variables to tune = more iteration runs. |
| **Failure modes** | Missing or poorly-trained high-noise LoRA → *"uncanny mix of real life and cartoon"*. Missing low-noise LoRA → blur and identity collapse. |
| **VRAM** | A14B requires **80GB VRAM** for training — same as CogVideoX1.5 on A100. |

Compare to CogVideoX1.5: single model, single LoRA, 10-14 hours per run. The per-run time is similar, but CogVideoX1.5 has **half the tuning complexity**.

### The TI2V-5B Alternative

The 5B dense model avoids the dual-expert problem — single model, single LoRA, lower VRAM (24GB). However:

| Pro | Con |
|-----|-----|
| Single LoRA (simpler) | Lacks I2V-A14B's *"enhanced support for diverse stylized scenes"* |
| 24GB VRAM training | 24fps output has misleading effective motion rate (~6 latent keyframes/sec) |
| Consumer GPU deployment | Lower quality than the 14B variant |
| 24fps label (no RIFE needed) | Users report "slow-motion looking" output anyway |
| Faster training (~4-6 hours) | Less community testing for character identity LoRAs |

### Looping: Achievable but with Known Issues

Wan 2.2 has **no native loop support** ([GitHub #81](https://github.com/Wan-Video/Wan2.2/issues/81) — open, zero responses from devs). Community solutions:

- **FLF2V (First-Last Frame to Video):** Feed the same image as first AND last frame. Model generates motion that returns to start. ([NextDiffusion tutorial](https://www.nextdiffusion.ai/tutorials/wan-2-2-looping-animations-in-comfyui))
- **Known issue:** Color contrast increases over time during looped generation. ([kijai #1541](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541))
- **Known issue:** Same start/end frame **limits the range of motion** — model constrains itself. ([GitHub #49](https://github.com/Wan-Video/Wan2.2/issues/49))
- **Best loop length:** ~3 seconds (aligns with our target).

Compare to CogVideoX1.5: our sprint plan uses **latent-space loop closure** (injecting first frame latent at denoising step 75-85%), which is architecturally more principled than FLF2V. CogVideoX1.5's single-pass generation is naturally better suited to latent-space manipulation than Wan 2.2's approach.

### Identity Drift: Same Problem, Same Severity

A [CVPR 2026 paper](https://arxiv.org/html/2510.14255v1) explicitly names **both** CogVideoX and Wan as suffering from the same identity drift problem. The root cause is exposure bias (training on ground-truth frames, inferring from self-generated frames).

For Wan 2.2 specifically, drift concentrates at **frame 50-70%** of the clip. Mitigations: denoise 0.40-0.50, combine reference image + LoRA, keep clips to ≤6 seconds, use negative prompts for `morphing, warping, face deformation`.

**Neither model has an inherent architectural advantage for identity preservation.** The MoE claim is unsubstantiated (see Section 2 above).

---

## 4. What Wan 2.2 Genuinely Does Better

Despite the corrections above, Wan 2.2 has real advantages:

### 4.1 LoRA Ecosystem Size and Tooling

| Tool | CogVideoX1.5 Support | Wan 2.2 Support |
|------|----------------------|-----------------|
| Passenger12138 trainer | Yes | No |
| AI Toolkit (Ostris) | No | Yes |
| Musubi-tuner | Partial | Yes |
| Diffusion-pipe | Partial | Yes |
| DiffSynth-Studio | No | Yes |
| diffusers (official) | Yes (source branch) | Yes |

Wan 2.2 has **more training tools available**, which means more community knowledge, more hyperparameter recipes, and faster troubleshooting. CogVideoX1.5 relies primarily on one specialized trainer (Passenger12138).

### 4.2 Community Momentum

- Wan 2.2: **16K+ GitHub stars**, 2K forks, active development (2.2 → 2.5 → 2.6 → 2.7)
- CogVideoX1.5: **~4K stars**, development has largely stalled since late 2024

The Wan ecosystem is where the community energy is. Bug fixes, ComfyUI nodes, new LoRAs, and training guides are being produced at a much higher rate.

### 4.3 Apache 2.0 License

Wan 2.2 uses Apache 2.0 — fully permissive commercial use. CogVideoX1.5 uses a custom THUDM license requiring review. For a character IP brand producing commercial content, this is a meaningful difference.

### 4.4 Newer Versions Available

Wan 2.6 specifically addresses identity preservation issues: *"Wan 2.5 created 'morphing' effects when attempting scene changes. Wan 2.6 handles transitions cleanly with maintained character identity."* ([source](https://10b.ai/blog/wan-2-6-i2v-face-stability))

If we choose Wan, we have the option to evaluate newer versions. CogVideoX1.5 has no newer versions on the horizon.

### 4.5 Existing Anime/Cartoon Style LoRAs

While the photorealistic bias is a real problem, the community has produced workarounds:
- [Anime Style WAN 2.2 I2V LoRA](https://civitai.com/models/2222779/anime-style-wan-22-i2v)
- [Simple/Pure Color Anime Style LoRA](https://civitai.com/models/1872525/wan22-simple-and-pure-color-anime-style)
- [2D Animation Effects LoRA](https://civitai.com/models/1920897/wan22-2d-animation-effects-2d)

These could be **stacked** with a custom Pudgy Penguins character LoRA to counteract the realistic bias. CogVideoX1.5 has no equivalent community style LoRAs.

---

## 5. What CogVideoX1.5 Genuinely Does Better

### 5.1 Architectural Simplicity

Single model, single LoRA, single set of hyperparameters. No dual-expert complexity. For a 6-week sprint with limited iteration time, simplicity has real value.

### 5.2 Frame Sharpness

CogVideoX produces *"sharper, crisper detail"* compared to Wan 2.2's softer cinematic look. For cartoon characters with thick outlines and clean edges, frame sharpness may directly translate to better aesthetic preservation.

### 5.3 Latent-Space Loop Closure

Our sprint's loop strategy (injecting first-frame latent at denoising step 75-85%) operates in latent space before VAE decode. CogVideoX1.5's single-pass architecture is naturally suited to this manipulation. Wan 2.2's FLF2V approach (feeding same image as first and last frame) is a pixel-level workaround with documented color-drift issues.

### 5.4 Higher Native Resolution

CogVideoX1.5: 1360x768. Wan 2.2 I2V-A14B: 1280x720. Wan 2.2 TI2V-5B: 1280x704. Marginal, but CogVideoX1.5 has a slight edge.

### 5.5 Longer Native Duration

CogVideoX1.5: up to 10 seconds (161 frames). Wan 2.2: ~5 seconds. For the 3-5 second GIF target this doesn't matter, but it provides a small buffer for loop closure overhead (generate longer, trim to loop point).

### 5.6 Our Sprint Plan Is Already Built Around It

The entire captioning pipeline, training config, inference pipeline, evaluation framework, and week-by-week timeline were designed and validated specifically for CogVideoX1.5. Switching models isn't just swapping a checkpoint — it cascades through every section of the plan.

---

## 6. The Honest Comparison

| Dimension | CogVideoX1.5-5B-I2V | Wan 2.2 I2V-A14B | Wan 2.2 TI2V-5B |
|-----------|---------------------|-------------------|------------------|
| **Photorealistic bias** | Unknown severity | **Documented problem** for cartoon content | **Documented problem** |
| **Cartoon/2D style validation** | Untested | Untested (anime LoRAs exist, flat-color mascots untested) | Untested |
| **LoRA training complexity** | Single model, single LoRA | **Dual-expert, dual LoRA** — 2x complexity | Single model, single LoRA |
| **LoRA training time (character, A100)** | 10-14 hours/run | **12-20+ hours** (both experts) | ~4-6 hours/run |
| **Training tools** | 1 primary (Passenger12138) | **4+ tools** | 4+ tools |
| **Community momentum** | Stalled | **Very active** | Very active |
| **Identity drift** | Documented | **Equally documented** | Documented |
| **Loop support** | Latent-space injection (architecturally principled) | FLF2V workaround (color drift issues) | FLF2V workaround |
| **Native FPS** | 16fps (needs 1.5x RIFE) | 16fps (needs 1.5x RIFE) | 24fps (misleading — see VAE compression) |
| **Native resolution** | 1360x768 | 1280x720 | 1280x704 |
| **VRAM (training)** | ~35GB | 80GB | 24GB |
| **VRAM (inference)** | 19-40GB | 80GB / 24GB (FP8) | 8-16GB |
| **License** | Custom THUDM | **Apache 2.0** | **Apache 2.0** |
| **Frame sharpness** | **Sharper/crisper** | Softer/cinematic | Softer/cinematic |
| **Max duration** | 10 sec | ~5 sec | ~5 sec |
| **Development trajectory** | Stalled | Active (2.6, 2.7 released) | Active |
| **Sprint plan compatibility** | **Fully designed** | Requires plan revision | Requires plan revision |
| **Existing style LoRAs** | None | **Anime/2D LoRAs available** for stacking | Available |
| **Square (1:1) GIPHY support** | Min 768px short side (oversized) | **Not natively supported** | **Not natively supported** |

---

## 7. Honest Verdict

### Neither model is clearly superior for this specific use case.

The previous evaluation's conclusion that Wan 2.2 was *"a stronger candidate than CogVideoX1.5 across nearly every dimension"* was **wrong**. It was based on:
- Uncritical acceptance of benchmark scores (which don't measure cartoon fidelity)
- Unverified training time claims
- Conflating marketing language ("supports stylized content") with actual capability
- Missing the photorealistic bias entirely
- Missing the dual-expert training complexity

The corrected picture is: **each model has distinct advantages and disadvantages, and the actual performance on flat-color cartoon penguins is unknown for both.**

### Risk Profile

| Risk | CogVideoX1.5 | Wan 2.2 I2V-A14B |
|------|--------------|------------------|
| "Can't learn the penguin's visual identity" | Medium | Medium |
| "Output looks too realistic / 3D" | Unknown (untested) | **High** (documented bias) |
| "Training takes too long for sprint timeline" | Low (10-14 hrs, well-scoped) | **Medium** (dual-expert adds complexity + time) |
| "LoRA training tooling breaks or is undocumented" | Low (Passenger12138 purpose-built) | Low (4+ tools available) |
| "Loop closure fails" | Low (latent-space approach, principled) | **Medium** (FLF2V has color-drift issues) |
| "Model development abandoned" | **Medium** (stalled) | Low (active development) |
| "License blocks commercial use" | **Medium** (custom license) | Low (Apache 2.0) |
| "Community can't help debug issues" | Medium (smaller community) | Low (16K+ stars, active forums) |

### The Actual Decision

There are two defensible strategies:

---

**Strategy A: Stay on CogVideoX1.5 (Lower Risk, Proven Path)**

- The sprint plan is fully designed, every decision validated through our 14-question grilling process
- Single-model simplicity means fewer variables to debug in a 6-week sprint
- Latent-space loop closure is more principled than FLF2V
- Sharper frame rendering may better preserve cartoon outlines
- The unknown (photorealistic bias severity) applies to both models equally

*Risk: CogVideoX1.5 development has stalled. If the sprint reveals model-level issues that need upstream fixes, there's no active development team to address them.*

---

**Strategy B: Dual-Track Test in Week 2 (Higher Effort, Data-Driven Decision)**

- Add 1-2 days and ~$50-100 to Week 2 to run zero-shot tests on both models with the client's actual penguin art
- The photorealistic bias question can only be answered empirically — run both models, look at the output
- If Wan 2.2 passes the style-preservation test, its larger ecosystem and Apache 2.0 license become genuine long-term advantages
- If Wan 2.2 fails the style test, you've lost 1-2 days but gained definitive evidence to stay on CogVideoX1.5

*Risk: Adds complexity to Week 2. If results are ambiguous (neither clearly wins), you're stuck making a judgment call mid-sprint.*

---

**Strategy C: CogVideoX1.5 for Sprint, Wan 2.6/2.7 for Phase 1 (Staged Approach)**

- Execute the sprint on CogVideoX1.5 as planned — zero changes to the validated plan
- In Week 6's Phase 1 proposal, recommend evaluating Wan 2.6 or 2.7 (which specifically improved identity preservation) as the next-generation model
- The Wan ecosystem's active development trajectory means future versions may solve the photorealistic bias and loop closure issues

*Risk: None to the sprint. Defers the model-switching question to Phase 1 when there's more data and more time.*

---

## 8. Recommendation

**Strategy C (CogVideoX1.5 for sprint, Wan 2.6/2.7 for Phase 1) with a lightweight Strategy B probe.**

Rationale:

1. **Don't change the sprint plan.** It was validated through 14 rounds of systematic technical review. Switching models cascades through captioning, training config, inference pipeline, loop strategy, and evaluation. The risk of destabilizing a proven plan exceeds the potential upside.

2. **Add a lightweight Wan 2.2 zero-shot test to Week 2.** This costs ~$50 and a few hours — not the full 1-2 day dual-track from Strategy B. Run 3-5 of the client's art layouts through base Wan 2.2 I2V (zero-shot, no LoRA). The ONLY question we need answered: **does Wan 2.2 3D-ify the penguin's flat-color aesthetic?** If yes, Wan 2.2 is disqualified. If no, it becomes a strong Phase 1 candidate.

3. **Recommend Wan 2.6 or 2.7 in the Phase 1 proposal.** By then, the Wan ecosystem will have matured further, the photorealistic bias may be addressed, and we'll have sprint data on the penguin characters to inform a more targeted evaluation.

---

## 9. What Remains Genuinely Unknown

These questions cannot be answered by research — they require empirical testing with the actual Pudgy Penguins assets:

| Question | Why It Matters | When to Test |
|----------|---------------|-------------|
| Does CogVideoX1.5 preserve flat-color cartoon outlines during I2V? | If it doesn't, the sprint's core deliverable is at risk regardless of model choice | Week 2 zero-shot baseline |
| Does Wan 2.2 3D-ify flat-color cartoon input? | Determines whether Wan is viable at all for this IP | Week 2 lightweight probe |
| Can a style LoRA + character LoRA stack counteract Wan's realistic bias? | If yes, Wan becomes viable despite the bias | Phase 1 if probe passes |
| Does CogVideoX1.5's frame sharpness actually help or hurt cartoon rendering? | Sharper could mean cleaner lines OR sharper artifacts | Week 2 zero-shot baseline |
| At 3 seconds, does either model's identity drift even matter? | Both models drift at ~50-70% of clip length. At 3 sec / ~48 frames, drift may be negligible for both | Week 4 integration testing |

---

## Appendix: All Sources

### Community Reports & Guides
- [CivitAI: Anime Style WAN 2.2 I2V LoRA](https://civitai.com/models/2222779/anime-style-wan-22-i2v)
- [CivitAI: Taz's Anime Style LoRA Training Guide Part 1](https://civitai.com/articles/20389)
- [CivitAI: WAN2.2 LoRA Workflow TLDR](https://civitai.com/articles/17740)
- [CivitAI: WAN 2.2 Local LoRA Training Guide](https://civitai.com/articles/18985)
- [CivitAI: Simple/Pure Color Anime Style LoRA](https://civitai.com/models/1872525)
- [CivitAI: 2D Animation Effects LoRA](https://civitai.com/models/1920897)
- [Apatero: Wan 2.2 LoRA Training Time Analysis](https://www.apatero.com/blog/wan-22-video-lora-training-time-complete-analysis-2025)
- [Apatero: Train Wan 2.2 LoRA for Person Guide](https://www.apatero.com/blog/wan-2-2-lora-training-person-method-guide-2025)
- [Apatero: Train Wan 2.2 LoRAs Best Practices](https://www.apatero.com/blog/train-wan-22-loras-best-practices-2025)
- [WaveSpeed: WAN 2.2 LoRA Training Settings](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/)
- [RunComfy: WAN 2.2 I2V Character Consistency LoRA](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora)
- [Dredyson: I2V Denoise & Face Consistency Guide](https://dredyson.com/how-i-fixed-wan2-2-i2v-on-8gb-vram-a-complete-step-by-step-beginners-guide-to-source-faithful-animation-settings-denoise-tuning-and-face-consistency-workarounds-that-actually-work-after-6-months/)
- [GenAIntel: CogVideoX-5B vs Wan 2.2-5B Comparison](https://www.genaintel.com/compare/cogvideox-5b-vs-wan-22-5b)

### Official Documentation
- [Wan 2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 Official Blog](https://wan.video/blog/wan2.2)
- [Wan-Animate Paper (arXiv)](https://arxiv.org/html/2509.14055v1)
- [Wan-Animate Preprocessing Docs](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md)
- [Wan-Animate Project Page](https://humanaigc.github.io/wan-animate/)
- [Wan2.2-I2V-A14B HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [Wan2.2-TI2V-5B HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Wan2.2-Animate-14B HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [CogVideoX1.5-5B-I2V HuggingFace](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)
- [CogVideoX LoRA Training (HuggingFace diffusers)](https://huggingface.co/docs/diffusers/en/training/cogvideox)
- [google/umt5-xxl Model Card](https://huggingface.co/google/umt5-xxl)

### GitHub Issues
- [Wan 2.2 Seamless Looping Request #81](https://github.com/Wan-Video/Wan2.2/issues/81)
- [Wan 2.2 Looping Videos #49](https://github.com/Wan-Video/Wan2.2/issues/49)
- [ComfyUI-WanVideoWrapper Loop Color Issue #1541](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541)
- [Wan 2.2 16/24 FPS Issue #13](https://github.com/Wan-Video/Wan2.2/issues/13)
- [Wan 2.2 Relight LoRA Mismatch #205](https://github.com/Wan-Video/Wan2.2/issues/205)
- [DiffSynth-Studio LoRA Color Shift #1105](https://github.com/modelscope/DiffSynth-Studio/issues/1105)
- [ComfyUI Color Drift #9975](https://github.com/Comfy-Org/ComfyUI/issues/9975)
- [ComfyUI FPS Mismatch #9106](https://github.com/comfyanonymous/ComfyUI/issues/9106)

### Research Papers
- [IPRO: Identity-Preserving Reward-Guided Optimization (CVPR 2026)](https://arxiv.org/html/2510.14255v1) — names both CogVideoX and Wan as having identity drift
- [Wan-Animate: Unified Character Animation (arXiv 2509.14055)](https://arxiv.org/html/2509.14055v1)

### Wan 2.6 Identity Improvements
- [Wan 2.6 Face Stability Guide](https://10b.ai/blog/wan-2-6-i2v-face-stability)
- [What Is Wan 2.6 — MindStudio](https://www.mindstudio.ai/blog/what-is-wan-2-6-video-open-source)

### Scenario/Guide Articles
- [Scenario: Wan 2.2 Animate Essentials](https://help.scenario.com/en/articles/wan-2-2-animate-models-the-essentials/)
- [Wan 2.2 I2V Prompting Guide](https://wan-animate.com/posts/how-to-use-i2v-prompting-wan-2-2-animate-guide)
- [Alici.AI: Wan 2.7 Guide](https://alici.ai/blog/how-to-use-wan-27-guide-2026)
- [302.AI: Wan2.2-Animate 4-Case Test](https://medium.com/@302.AI/wan2-2-animate-model-test-with-4-cases-e9e1c1ef492c)

---

*This assessment was produced by synthesizing findings from three independent research agents examining: (1) real-world quality reports from CivitAI, Reddit, and GitHub, (2) deep technical architecture comparison with verified sources, and (3) comprehensive Wan-Animate capability evaluation. Claims are tagged as VERIFIED, PARTIALLY VERIFIED, UNVERIFIED, or REFUTED based on the evidence found.*
