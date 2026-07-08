# Pudgy Penguins — AI Animation Engine: Sprint Plan Overview

**Objective:** Build an AI animation engine that produces production-quality, brand-faithful Pudgy Penguins content — starting with GIPHY looping GIFs, scaling to 30-second Instagram shorts.

**Approach:** Explore first, then build. Use paid platforms to establish what's possible out-of-the-box, learn the data, and set a quality baseline — then train a custom model to exceed it.

---

## Phase 1: Exploration & Baseline (Weeks 1–2)

**Objective:** Test commercial platforms with real Pudgy assets to understand what works, what breaks, and why — before committing to custom training.

### Week 1 — Platform Testing & Asset Understanding

**Goal:** Run the same Pudgy Penguin content through multiple platforms and document what each produces.

- Onboard to client's Slack, receive assets (T-poses, AE files, character guides, sample skits)
- Catalog available assets: characters, props, backgrounds, existing animations
- Select 5 representative test cases (1 walk cycle, 1 wave, 1 idle loop, 1 multi-character, 1 with background)
- Run all 5 through **3 platforms** using their free/paid tiers:
  - **Google Flow** (free) — test Ingredients system for character reuse + clip chaining
  - **Higgsfield** (Plus $49) — test Soul ID for character consistency + multi-model comparison
  - **Runway** (Pro $28) — test Extend for shot continuity + camera controls
- Document results per platform: character fidelity, motion quality, loop closure, background stability

**Deliverable:** Platform comparison grid scored against the qualitative rubric (same 4-point scale from the Sprint Plan).

### Week 2 — Prompting Strategy & Baseline Lock

**Goal:** Develop the optimal prompting structure for Pudgy content and lock the quality baseline that custom training must beat.

- Iterate on prompts: test natural language descriptions, style keywords, negative prompts, reference images
- Map which prompt patterns produce the best character consistency across platforms
- Test the I2V (image-to-video) workflow: artist-drawn first frame → platform animation
- Test clip chaining: generate 3–5 sequential shots and stitch into a 15-second sequence
- Run best outputs through the evaluation framework (FID, LPIPS, SSIM) to set quantitative baselines
- Present findings to client's animation director — calibrate expectations

**Deliverables:**
- Best-case baseline outputs (the ceiling of what paid platforms can do with Pudgy IP)
- Prompting playbook (what works, what doesn't, exact prompt templates)
- Gap analysis: where platforms fail and what custom training needs to solve
- Go/no-go recommendation for Phase 2

---

## Phase 2: Custom Model Training (Weeks 3–8)

**Objective:** Fine-tune CogVideoX1.5-5B-I2V with a LoRA adapter to produce brand-faithful Pudgy Penguin animations that exceed the Phase 1 baseline.

*This phase follows the detailed Final Sprint Plan. Below is the high-level week-by-week summary.*

### Week 3 — Infrastructure & Data Curation

**Goal:** Set up the training environment and prepare the dataset using everything learned in Phase 1.

- Provision cloud GPU (A100 80GB on RunPod)
- Build Docker image with all dependencies
- Curate 60–80 video clips from client's library (micro-actions, not full scenes)
- Apply prompting insights from Phase 1 to build the captioning pipeline
- Render clips: 70% neutral gray background, 30% show backgrounds
- Present evaluation rubric to client for sign-off

### Week 4 — Captioning & Zero-Shot Baseline

**Goal:** Caption the dataset and establish the untrained model's baseline (to measure training improvement).

- Construct identity anchors per character using Phase 1 prompt learnings
- Run VLM captioning + pruning pipeline on all clips
- Test the base CogVideoX1.5 model with no fine-tuning against client layouts
- Calibrate evaluation metrics against client's hand-animated content
- Compile training-ready dataset

### Week 5 — LoRA Training (Full Week)

**Goal:** Train the LoRA adapter and find the best checkpoint.

- Run 3–5 training iterations (10–14 hours each, overnight)
- Evaluate outputs at every 500-step checkpoint
- Select golden checkpoint: best balance of character fidelity + motion quality
- Run automated evaluation against Phase 1 baseline and hand-animated baseline

### Week 6 — Pipeline Integration

**Goal:** Build the full ComfyUI production pipeline and solve the frame rate gap.

- Build master ComfyUI workflow: I2V → generation → loop closure → frame interpolation → export
- Implement 16fps → 24fps frame rate bridge via RIFE
- Implement seamless loop closure (latent-space primary, VFI fallback)
- Integration test with 5–8 real client layouts
- First demo to animation director at 24fps

### Week 7 — Production Sprint & Evaluation

**Goal:** Produce the deliverable GIF library and run full evaluation.

- Batch-generate 15–25 single-character GIFs (4 variations each)
- Animation director reviews and scores using the rubric
- Multi-character stress test (5–8 Tier 1 layouts)
- Full automated evaluation: FID, LPIPS, SSIM, temporal consistency, HeliosBench drifting
- Compile production metrics: success rate, time-per-GIF, cost-per-GIF

### Week 8 — Documentation & Handoff

**Goal:** Deliver the complete system and propose Phase 1 roadmap.

- Write technical report
- Deploy "glass box" system (1-click RunPod launch, no CLI needed)
- Train client's animation director and technical directors (recorded sessions)
- Present Phase 1 proposal: ML-Ops retainer, multi-character roadmap, 30-second scaling plan

---

## Timeline at a Glance

```
Week  1  ███░░░░░░░░░░░░░  Platform testing + asset onboarding
Week  2  ███░░░░░░░░░░░░░  Prompting strategy + baseline lock
         ─── Go/No-Go Decision ───
Week  3  ░░░███░░░░░░░░░░  Infrastructure + data curation
Week  4  ░░░███░░░░░░░░░░  Captioning + zero-shot baseline
Week  5  ░░░░░░███░░░░░░░  LoRA training (full week)
Week  6  ░░░░░░░░░███░░░░  Pipeline integration + frame rate bridge
Week  7  ░░░░░░░░░░░░███░  Production sprint + evaluation
Week  8  ░░░░░░░░░░░░░░██  Documentation + handoff
```

| Phase | Duration | Cost |
|-------|----------|------|
| **Phase 1: Exploration** | 2 weeks | ~$77/mo (Flow free + Higgsfield $49 + Runway $28) |
| **Phase 2: Training** | 6 weeks | ~$750 (GPU compute) + platform subs |
| **Total** | 8 weeks | ~$830 |

---

## Why Explore First?

| Without Phase 1 | With Phase 1 |
|----------------|-------------|
| Guess at prompting structure | Proven prompt templates from real testing |
| Assume platforms can't work | Know exactly where and why they fail |
| Train blind — hope the dataset is right | Dataset informed by what models respond to |
| No baseline to measure against | Clear quality bar the custom model must beat |
| Client sees results only at Week 5 | Client sees AI output in Week 1 |
| Risk: 6 weeks spent, unclear if better than paid | Risk reduced: go/no-go gate at Week 2 |

---

## Success Criteria

| Milestone | When | Pass Condition |
|-----------|------|---------------|
| Platform baseline established | End of Week 2 | Best paid platform output scored on rubric; gap analysis complete |
| Go/no-go for custom training | End of Week 2 | Client agrees paid platforms can't meet their quality bar → proceed to Phase 2 |
| Golden checkpoint trained | End of Week 5 | LoRA outputs exceed Phase 1 baseline on all rubric dimensions |
| Production GIFs delivered | End of Week 7 | Avg rubric score >= 3.0/4.0, no dimension scoring 1 |
| System handed off | End of Week 8 | Client's team can operate pipeline independently |

---

*Phase 2 follows the detailed [Final Sprint Plan](./Final_Sprint_Plan.md) with all training configurations, hyperparameters, evaluation frameworks, and risk mitigations.*
