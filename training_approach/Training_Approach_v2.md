# Pudgy Penguins — Training Approach v2

> **Refined by [Training_Approach_v3.md](./Training_Approach_v3.md)** — v3 keeps this plan's thesis, rubric, and phases but commits the open base A/B to **AniSora V3.2** (anime-native Wan2.2-A14B) and makes the identity-pinning native. Read v3 for the current base decision.

**Supersedes** the CogVideoX1.5-5B-I2V single-LoRA plan. Built directly on the *Model & Architecture Reassessment* memo and the three research threads behind it (SOTA I2V models, character-consistency methods, training-objective math).

**Goal (unchanged):** generate 2D cartoon animation of **Pax** (blue) and **Polly** (pink) — flat pastel, thick black outlines — with **exacting character consistency, robustness, and high image quality**. Budget is uncapped.

**Core thesis:** the underwhelming first run is a *pipeline-shape* problem, not a tuning problem. On ~75 clips, a bigger model or full fine-tune won't fix consistency — it overfits. The wins are **architectural** (decouple identity from motion) and in the **data + objective**, not in parameter count.

---

## 1. Guiding principles

1. **Decouple identity from motion.** Lock Pax/Polly in the *image domain* (high-res, human-QC'd keyframes); use the *video model* only to in-between locked frames. Identity error can't accumulate across a clip when both endpoints are verified art.
2. **Verify before you train.** Two cheap gates come first — the VAE reconstruction test and a caption A/B — because both can invalidate expensive training before it starts.
3. **Data and objective beat parameters.** Fix captions, layer-targeting, and the timestep shift; add turnarounds and stills. Don't reach for full fine-tune.
4. **Keep humans on identity, automate the in-betweening.** This mirrors the studio's existing Concept → Animatic → **Posing** → Animation flow. The video model replaces the labor between key poses, not the identity decisions.

---

## 2. Target end-state pipeline

```
                IMAGE DOMAIN  (identity — human-controlled)
   ┌───────────────────────────────────────────────────────────┐
   │  turnarounds / character sheet + image-domain character     │
   │  method  ──►  key-pose KEYFRAMES per shot  ──►  HUMAN QC    │
   └───────────────────────────────────────────────────────────┘
                              │  locked frames
                              ▼
                VIDEO DOMAIN  (motion — automated)
   ┌───────────────────────────────────────────────────────────┐
   │  FLF2V / keyframe interpolation on a flow-matching base     │
   │  (+ optional motion/style LoRA for Pudgy timing & bounce)   │
   │  ──►  boundary-anchored in-between  ──►  shot clip           │
   └───────────────────────────────────────────────────────────┘
                              │
                              ▼   optional for 2-character shots
              reference-conditioning (Phantom / VACE) to inject
              both Pax & Polly refs and prevent identity blending
```

**Why this shape:** free I2V lets identity drift because nothing pins the character mid-clip. FLF2V pins it at `t=0` and `t=1`, so drift is bounded to the interpolation. AniSora supports keyframe interpolation natively; Wan 2.2 supports FLF2V. For two characters in one frame, single-character LoRAs tend to *blend* Pax and Polly — reference-conditioning (Phantom is explicitly multi-subject) addresses that.

---

## 3. Phased plan

### Phase 0 — Diagnostics & de-risking · *days, no new data needed*

The highest-ROI work in the whole plan. Nothing here needs client deliverables.

| # | Task | Output |
|---|---|---|
| 0.1 | **VAE round-trip test.** Encode→decode a handful of real Pax/Polly clips through each candidate VAE (Wan 2.1 & A14B `8×`, Wan 2.2-5B `16×`, Hunyuan `8×`, CogVideoX). Diff round-trip vs source; inspect outlines + flat fills. | Chooses the base; confirms whether image quality is a VAE ceiling or a training problem. |
| 0.2 | **Caption A/B on current CogVideoX.** Re-run with rare-token identity + variable-only descriptions vs the current dense identity captions. | Isolates how much of the bad result was caption strategy vs model. |
| 0.3 | **LoRA-targeting fix.** Quick re-run targeting all-linear incl. MLP (not attention-only), α=2r. | Confirms the cheap layer-targeting win before migrating. |

**→ Gate G0:** base model chosen · image-quality root cause identified · caption/targeting deltas measured.

---

### Phase 1 — Base migration & corrected baseline · *~week 1–2*

| # | Task | Notes |
|---|---|---|
| 1.1 | Stand up the chosen base env (**Wan 2.2 A14B** primary; **AniSora** in parallel) with **musubi-tuner** or **diffusion-pipe**. | These two trainers are the current standard; CogVideoX is not first-class in either. |
| 1.2 | Repackage the existing 75-clip dataset to the new trainer's format (metadata/caption layout, resolution/fps bucket). | Keep 768×1360 portrait; Wan native 16 fps, AniSora 24 fps — resample as needed (source is 24 fps). |
| 1.3 | Train a corrected **style/character LoRA** on the existing 75 clips: all-linear incl. MLP, α=2r, rank 16–32, timestep shift matched to resolution. On A14B train **two LoRAs** (low-noise = identity/texture, high-noise = motion). | This is the apples-to-apples "did the base + recipe help" run. |

**→ Gate G1:** corrected baseline beats the CogVideoX run on the §5 rubric. If not, diagnose before scaling.

---

### Phase 2 — Decoupled pipeline stand-up · *~week 2–4*

| # | Task | Notes |
|---|---|---|
| 2.1 | **Image identity stage.** Build consistent-character keyframe generation: an image-domain LoRA (or consistent-character method) for Pax/Polly. Interim identity anchors = clean frames extracted from the 30 videos until turnarounds arrive. | Where possible, human-QC every keyframe. |
| 2.2 | **Motion stage.** Wire FLF2V / keyframe interpolation on the chosen base; optionally attach the motion/style LoRA from Phase 1. | AniSora = native keyframe interp; Wan 2.2 = FLF2V workflow. |
| 2.3 | **Integrate end-to-end:** key-pose keyframes → interpolate → assemble a full shot. | First real test of the decoupled architecture. |

**→ Gate G2:** an end-to-end shot generated with locked identity across the clip, no mid-clip drift.

---

### Phase 3 — Data upgrade & scale · *parallel + ongoing*

Driven by the evidence-ranked client asks (§6).

| # | Task | Unblocks |
|---|---|---|
| 3.1 | Client delivers **turnarounds + expression sheets w/ color codes**. | Stronger keyframe identity + reference-conditioning. |
| 3.2 | Add **high-quality stills** → **joint image+video training**. | Cuts gradient variance, sharpens per-frame identity; cheapest way past the 75-clip bottleneck. |
| 3.3 | Add **clean single-action clips** + **20–40 more skits** from the 150+ library. | Breaks motion-redundancy of the tiled windows; widens diversity. |
| 3.4 | Evaluate **Phantom / VACE reference-conditioning** for two-character shots. | Prevents Pax/Polly identity blending. |

---

### Phase 4 — Evaluation & selection · *continuous*

- Score every checkpoint against the §5 rubric on a fixed held-out prompt set.
- Pick the **golden checkpoint** by eye — fidelity usually peaks mid-run before overfitting; don't assume the last is best.
- Monitor for overfitting/burn-in (mode collapse to training backgrounds, style bake-in).

---

## 4. Recommended starting config (chosen base — confirm at G0)

Starting points from the research; treat exact numbers as A/B candidates, not gospel.

| Knob | Starting value | Rationale |
|---|---|---|
| Base | **Wan 2.2 A14B** (or AniSora) | Flow matching, Apache, mature LoRA tooling; AniSora if VAE/style eval favors the anime-native model. |
| Method | **LoRA** (not full FT) | Full FT overfits on ~75 clips; LoRA is itself a regularizer. |
| Target layers | **all linear incl. MLP/FFN** | Biggest cheap win; attention-only is the classic mistake. |
| Rank / α | **r = 16–32, α = 2r** | Rank barely matters on small data; regularization > capacity. |
| Two-expert (A14B) | low-noise LoRA = identity, high-noise = motion | Tune identity independently. |
| LR / warmup | ~8e-5–1e-4, ~100 warmup steps | Community-standard for Wan/Hunyuan LoRA. |
| Timestep shift | matched to training resolution | Governs whether global temporal structure is learned at all. |
| VAE | prefer **8× spatial** | 16× (Wan 2.2-5B) softens outlines/flat fills — a hard ceiling. |
| Captions | rare-token identity + describe variables | Dense identity descriptions kill the identity loss signal. |
| Dataset spec | 768×1360 · base-native fps · joint image+video once stills arrive | Match the base's final-stage distribution. |

---

## 5. Evaluation rubric

Make the three goals measurable. Score each generated shot 1–5; track per-checkpoint.

| Dimension | What to check |
|---|---|
| **Character identity** | Pax/Polly shape, proportions, color codes correct and stable across the whole clip; no blending in 2-shots. |
| **Line & color quality** | Outlines crisp (not soft/doubled), flat fills clean (no banding), on-model. |
| **Motion robustness** | Coherent bouncy motion; no melting/warping under larger motion; correct timing feel. |
| **Prompt adherence** | Action, staging, camera match the brief. |
| **Temporal stability** | No flicker, no identity wobble frame-to-frame. |

Hold a fixed prompt set (mix of single-character, 2-shot, and an action-primitive) so checkpoints are comparable.

---

## 6. What we need from the client — evidence-ranked

| Priority | Asset | Why |
|---|---|---|
| **1** | Turnarounds + expression sheets for Pax & Polly, **exact color codes** | Feeds keyframe generation, reference-conditioning, cross-angle consistency. |
| **2** | High-quality character **stills** | Joint image+video training; escapes the 75-clip bottleneck. |
| **3** | Clean **single-action clips** (static cam, ~3–5 s, one action) | Breaks tiled-window redundancy; guarantees motion-primitive coverage. |
| **4** | **20–40 more skits** from the 150+ Drive library | Diversity of environments/actions. |
| **Dropped** | ~~Neutral-gray background plates~~ | Folklore; alpha/transparent passes are strictly better if isolation is ever needed. |

---

## 7. Decision gates (summary)

| Gate | Passes when | If it fails |
|---|---|---|
| **G0** | Base chosen; image-quality root cause known; caption/targeting deltas measured | If VAE is the ceiling → the base choice is forced; if captions were the cause → cheap fix, re-baseline |
| **G1** | Corrected baseline > CogVideoX run on the rubric | Diagnose recipe before scaling |
| **G2** | End-to-end decoupled shot with locked identity, no drift | Fall back to reference-conditioning-heavy or tighten keyframe density |

---

## 8. Risks & mitigations

- **VAE ceiling on 2D art** → tested in Phase 0; pick 8× VAE; escalate to a base whose VAE round-trips outlines cleanly.
- **Two-character blending** → reference-conditioning (Phantom/VACE); separate identity keyframes per character.
- **Overfitting / burn-in on 75 clips** → LoRA over full FT, modest rank, joint image+video data, checkpoint monitoring.
- **AniSora/Wan tooling immaturity for our exact style** → run both in parallel through G1; keep the fallback base warm.
- **Client deliverables slip** → Phases 0–2 are designed to run entirely on what we already have; data upgrades (Phase 3) are additive, not blocking.

---

## 9. Immediate next actions

1. **Write & run the VAE round-trip diagnostic** (0.1) — one day, decides the base.
2. **Caption A/B + LoRA-targeting re-run** on current CogVideoX (0.2, 0.3) — isolates model vs method.
3. **Send the tightened client asset request** (§6) — starts the long-lead deliverables now.
4. Stand up the Wan 2.2 (+ AniSora) training env in parallel (1.1). — ✅ **A14B env done**, see [`actions_done.md`](./actions_done.md) (built, pre-cached, smoke-tested on the A100-80GB; not yet trained).

---

*References: see the Model & Architecture Reassessment memo and `Data_Readiness_Gap_Analysis.md`. Base/consistency/math sources cited there (Wan, AniSora, HunyuanVideo, SD3 rectified-flow, LoRA-Without-Regret, Improved-Video-VAE, Phantom, VACE).*
