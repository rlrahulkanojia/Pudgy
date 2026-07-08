# Pudgy Penguins — Training Requirements

**Purpose:** Concrete hardware, data, and infrastructure required to begin LoRA fine-tuning of **CogVideoX1.5-5B-I2V** for the Pudgy Penguins animation engine.

**Source docs:** `Final_Sprint_Plan.md`, `solution_1_training.md`, `Sprint_Plan_Overview.md`, `research/VIDEO_DIRECTION_DESCRIPTIONS.md`, `research/CHARACTER_SHEETS.md`.

---

## 1. Base Model & Training Stack

| Component | Choice | Why |
|---|---|---|
| **Base model** | `THUDM/CogVideoX1.5-5B-I2V` | Best CogVideoX I2V variant. 1360×768 max res, native 16 fps, up to 81 frames / 5 s. Trainable on a single A100 80 GB. |
| **Training framework** | [`Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train`](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train) | Purpose-built for v1.5 I2V LoRA. Includes bucket-based multi-res, corrected RoPE, fixed OFS embedding. |
| **Pipeline runtime** | ComfyUI | I2V → loop closure → RIFE interp → cleanup → export |
| **Frame interpolation** | RIFE | 16 fps → 24 fps bridge (1.5× ratio) |
| **No separate image LoRA** | Eliminated | Latent spaces don't transfer between Flux/SDXL and CogVideoX. T-poses get folded into joint image-video training instead. |

---

## 2. Hardware

### 2.1 Training

| Item | Spec | Notes |
|---|---|---|
| **GPU** | 1× **NVIDIA A100 80 GB** | LoRA Rank 64, bf16, batch 1, grad-accum 4 → ~35 GB VRAM. Headroom for Rank 128 experiments. |
| **Per-run duration** | 10–14 h | ~4 000 steps |
| **Planned runs** | 3–5 | 40–80 total GPU-hours across the sprint |

### 2.2 Inference

| Item | Spec | Notes |
|---|---|---|
| **GPU** | 1× **A100 40 GB** or **H100** | 19–40 GB VRAM for 81-frame I2V + RIFE + cleanup |
| **Batch generation** | 4 variations per input | ~50–60 min per batch on A100 40 GB |

### 2.3 Supporting Compute

| Item | Spec | Notes |
|---|---|---|
| **VLM captioning** | GPT-4o API **or** 1× A6000 24 GB (local MiniCPM-V) | Auto-caption pipeline for the dataset |
| **Evaluation** | CPU or light GPU | FID, LPIPS, SSIM, RAFT, HeliosBench drift metrics |

### 2.4 Cloud & Storage

| Item | Choice | Notes |
|---|---|---|
| **Provider** | **RunPod** | Preferred over Vast.ai — better uptime, native Docker, network volumes |
| **Persistent storage** | RunPod Network Volume, ~100–200 GB | Dataset + checkpoints; mounted once across instances |
| **Container** | Custom Docker image, built Week 1 Day 1–2 | CogVideoX1.5-5B-I2V weights + `diffusers` (source branch) + Passenger12138 trainer + ComfyUI + RIFE + pinned deps. Pushed to Docker Hub. |

### 2.5 Budget

**Total recommended: ~$750** for the 6-week training phase.

| Line item | Rate | Hours/Qty | Cost |
|---|---|---|---|
| Training (A100 80 GB) | ~$1.50–2.00/hr | 40–80 hr | $90–240 |
| Inference (A100 40 GB) | ~$1.00–1.50/hr | weeks 2–5 + idle | $150–300 |
| Experimentation / failed runs | ~$1.00–2.00/hr | — | $75–150 |
| Persistent storage (2 mo) | ~$0.10/GB/mo | — | $15–30 |
| VLM captioning (GPT-4o) | ~$0.01/image | — | $8–15 |

---

## 3. Data

### 3.1 Dataset Composition

**Total target: 60–80 video clips + 50–100 character images.**

| Bucket | Count | Purpose |
|---|---|---|
| Single-character micro-action clips | ~60 | Walks, idles, eats, waves, blinks — one atomic action per clip |
| Tier-1 two-character clips | ~25 | Couple side-by-side, no occlusion |
| Tier-2 two-character clips | ~15 | Light interaction |
| Static T-poses / multi-angle turnarounds (PNGs) | 50–100 | Joint image-video training for 3D volume |
| Master Control rules / style guides | All available | Reference for captioning anchors |

### 3.2 Per-Clip Requirements (strict)

| Property | Spec |
|---|---|
| **Resolution** | 1360 × 768 (landscape) or 768 × 1360 (portrait) |
| **Resolution constraints** | min(W,H) = 768, 768 ≤ max(W,H) ≤ 1360, max(W,H) % 16 = 0 |
| **Frame count** | Must obey **8N + 1** (e.g., 49 or 81 frames) |
| **Duration** | 2–5 s per clip at 24 fps source (CogVideoX1.5 resamples to 16 fps internally) |
| **Action atomicity** | One micro-action per clip. Split "walk → stop → wave" into 3 clips. |
| **Action parity** | ≥ 4–5 clips per action type — otherwise the model biases toward the over-represented action |
| **Background mix** | **70 %** neutral gray (#808080) rendered from AE; **30 %** real show backgrounds |

### 3.3 Captioning

CogVideoX uses a frozen T5-XXL encoder — needs dense natural-language sentences, not tag lists. Two-tier structure:

1. **Fixed Identity Anchor** (15–20 words, manually authored per character, programmatically injected at the start of every caption).
2. **Auto-generated motion suffix** from VLM (GPT-4o or MiniCPM-V) describing temporal dynamics only — appearance is pruned out via keyword blocklist + semantic similarity check.

Multi-character captions must include explicit spatial anchors ("positioned on the left/right third of the frame") to give T5 geometric scaffolding.

Manual spot-check 10 % of pruned captions before full-dataset processing.

---

## 4. LoRA Hyperparameters (starting point)

| Param | Value | Rationale |
|---|---|---|
| **LoRA Rank** | 64 | Sweet spot for single-character + motion. R=16 too weak for volume; R=128 overfits on 60–80 clips. |
| **LoRA Alpha** | 32 or 64 | Must equal or be half of Rank. THUDM default Alpha=1 with R=64 in diffusers learns nothing (scale = α/R). |
| **Optimizer** | 8-bit AdamW | Required to fit 5 B model + R=64 LoRA into one A100 80 GB. |
| **Batch size** | 1 | VRAM constraint |
| **Gradient accumulation** | 4 | Effective batch 4 without OOM |
| **Training resolution** | 81 × 768 × 1360 | Must match inference resolution |
| **Precision** | bf16 | — |

---

## 5. Readiness Checklist

### Hardware / infra
- [ ] RunPod account funded (~$750 budget)
- [ ] A100 80 GB pod reservation confirmed
- [ ] Network volume created (~100–200 GB)
- [ ] Docker image built, pinned, pushed to Docker Hub
- [ ] HuggingFace token + model access for `THUDM/CogVideoX1.5-5B-I2V`

### Data
- [ ] 60–80 source clips identified from client's 150+ animation library
- [ ] Clips re-rendered/resized to 1360 × 768
- [ ] Each clip conforms to 8N + 1 frame count
- [ ] 70/30 neutral-gray vs. show-background split rendered
- [ ] ≥ 4–5 clips per action type (action parity verified)
- [ ] 50–100 T-poses / multi-angle turnarounds delivered
- [ ] Master Control rules and style guides on the network volume

### Captioning
- [ ] Identity anchors written for Blue (PP-001) and Pink (PP-002)
- [ ] GPT-4o API key provisioned (or MiniCPM-V deployed locally)
- [ ] Caption-pruning script built (keyword blocklist + similarity threshold)
- [ ] 10 % manual spot-check completed and threshold calibrated

### Evaluation
- [ ] Rubric signed off by client's animation director
- [ ] FID / LPIPS / SSIM / RAFT scripts deployed
- [ ] HeliosBench drifting metrics calibrated against client's hand-animated clips

### Currently in `Data/`
- [x] 5 reference reels (`Video-6, -425, -435, -439, -510`)
- [x] Extracted PNG keyframes per reel (`frames/Video-*`)
- [ ] **Missing:** T-poses, AE source files, character guides, prop / background library, Figma assets, design guidelines (noted in `NOTES.md` as "client provided" but not yet in the repo)

---

*Generated 2026-06-23 from the Pudgy sprint planning documents.*
