# expv1-kandinsky-50-video-pro — plan

**Status:** plan, written 2026-10-10, nothing run. **Isolated experiment**: see
[`../README.md`](../README.md) for what that means. Every inherited lesson below is cited
by its id in [`LEARNINGS_CARRIED.md`](LEARNINGS_CARRIED.md) (A1–E). Model facts and code
gotchas are in [`MODEL_NOTES.md`](MODEL_NOTES.md) (§2.1–2.7).

---

## 1. The question

> Can **Kandinsky 5.0 Video Pro I2V**, with **one LoRA** trained from its public checkpoint
> on the client's alpha clips, generate **on-model, flat-2D** Pax and Polly clips in which
> the **named action or expression** plays out, from a **start frame it never trained on**,
> at 512×512, up to 2.4 s?

The bars that make this a yes are fixed in §9 before any data exists (B19).

- **A yes** means the base is worth a second isolated experiment (`expv2-…`: 1024², end-frame
  anchoring, two characters).
- **A no** at the VAE or style gate means this base does not suit the art.
- **A no** on motion only means nothing until it is read against the reference (§8): every
  base so far has been data-limited on fast motion (A13).

## 2. One variable at a time

| Held fixed | Why |
|---|---|
| Base checkpoint, as published | The question is about this base |
| Resolution 512×512 (the SD tier) | musubi defines SD tasks only; full attention (no sparse-attention approximation); ~4× cheaper than 1024². 1024² is an `expv2` question |
| One LoRA, one recipe (plus one pre-declared fallback arm, §6.4) | Attributable results |
| Input layout identical in training and inference (§4, G0.2–G0.3) | MODEL_NOTES §2.3–2.4 |
| Caption wording, frozen once written | Train and evaluation share the distribution (B12) |

**Not in expv1:** 1024²/HD, first+last-frame anchoring, two characters, motion × expression
in one prompt, long-form chaining, props, loop tiling, distilled models, text-to-video.

---

## 3. Inputs

| Input | Source | Pin |
|---|---|---|
| DiT | HF `kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s` → `model/kandinsky5pro_i2v_sft_5s.safetensors` | record the HF commit SHA at download |
| VAE | HF `hunyuanvideo-community/HunyuanVideo` → `vae/` | record SHA |
| Text encoders | `Qwen/Qwen2.5-VL-7B-Instruct`, `openai/clip-vit-large-patch14` | record SHA |
| Trainer | musubi-tuner **v0.3.4** (tag) | + `patches/musubi-v0.3.4-resume-step.patch` (copied from `finetune/wan/patches/`, C2) + `patches/expv1-k5-i2v-conditioning.patch` (new, §4) |
| Reference code (read-only) | `kandinskylab/kandinsky-5` @ `3a74261` | the official inference path is the authority on the I2V input layout |
| Raw data | `Data/raw/iteration_3/`, `_4/`, `_5/` (Azure `pudgy/raw/`) | MD5-deduplicated (A8) |
| Hardware | 1× A100-80GB or H100-80GB; disk is not persistent (C1) | — |

---

## 4. Phase 0 — measure before training

No LoRA is trained until G0.1–G0.3 have run. Each step writes a JSON report and a contact
sheet to `eval/phase0/`, mirrored to Azure.

### G0.1 — VAE gate (stop if it fails)

Round-trip real art through **Kandinsky's own VAE code path** (`kandinsky.models.vae.build_vae`
with the pipeline's tiling settings), not diffusers. Kandinsky changed the tiling, and
temporal tile blending changes the result (report §7.1). Use the training size, 512×512,
plus 1024×1024 for the record.

- **Clips:** Pax and Polly × {one expression, one in-place motion, `running`}, plus one
  constant flat-fill clip (the flat-region grid test).
- **Metrics:** PSNR, SSIM, edge-PSNR over **positive** gradients with edge coverage reported,
  flat-band, absolute shimmer (B7).
- **By eye, at 4× zoom:** beak outline, eye highlights, flipper outline.

**Pass:** outlines and highlights intact at 512², and no grid or banding on flat fills.
**Fail → stop expv1 and report.** No LoRA recovers what the VAE discards. Also record
whether the running stride survives the 4× temporal compression; that is informational
(A13).

### G0.2 — Conditioning check

Run 1 training step each with stock musubi and with the expv1 patch, plus 1 sampling step
of the generator. Dump, per latent frame:

- statistics of the noisy-input channels;
- which frames carry non-zero `visual_cond`;
- the mask;
- which frames enter the loss.

**Pass:** the patched trainer's tensors are **identical in layout** to the generator's.
Frame 0 holds the clean start-frame latent in the noisy-input channels, `visual_cond` is
all zero, the mask is 1 at frame 0 only, and frame 0 is excluded from the loss. The stock
dump also confirms or refutes MODEL_NOTES §2.4 (the last frame at latent index 1, the mask
on every frame); record which.

### G0.3 — Which layout does the checkpoint expect?

Base model, no LoRA, on the 26 held-out `QF2_R` clips. Measure the flow-matching loss on
frames ≥ 1, change-masked, at t ∈ {0.65, 0.85}, with k = 2 noise draws shared across
layouts (the Tier-1 machinery, §7). Compare:

| Layout | First frame |
|---|---|
| **A** (inference) | written into the noisy input at frame 0 |
| **A1** (musubi generator; added after G0.2) | as A, plus the same latent in `visual_cond[0]` |
| **B** (stock trainers) | in the `visual_cond` channels, mask at frame 0 only |
| **B′** (stock musubi, `stock`) | as B, plus the last frame at latent index 1 and the mask on every frame |

Implemented as `eval/dcls.py score --configs base --layouts A A1 B stock --true-only`, then
`eval/dcls.py g03` (true-caption loss only, paired per clip; Wilcoxon reported).

**Result (2026-10-13): A.** The mean true-caption loss was A 0.0342, A1 0.0395 (+15.6%),
B 0.0423 (+23.6%) and stock 0.2495 (+630%). A was lowest on **26/26** clips against every
other layout (Wilcoxon p = 3·10⁻⁸ each).
- musubi's own generator (A1) therefore does **not** feed the checkpoint what it was trained
  on.
- The expv1 patch adds `--i2v_layout A` to the generator. G0.2's `gen_A` check shows that its
  input now matches trainer layout A.
- **expv1 trains, generates and scores Tier 1 in A.** `eval/phase0/g03_layout.json`

**Rule:** the layout with the lower paired loss is the one the checkpoint was trained on.
The report (§6.3.1) predicts A.

- If A wins, the patch trains in A.
- If B wins, patch the **generator** to B instead.

Either way, training and inference end up identical. This replaces an argument about the
code with a measurement.

### G0.4 — Zero-shot probe, and the control

Base model, no LoRA, Pudgy negative prompt (§5.5), prompt rewriting off:

- **Main grid:** 13 labels × 2 characters, `QF2_R` start frames, 3 seeds (78 clips).
- **Negative-prompt ablation:** 2 labels × 2 characters × 3 seeds with the **stock**
  negative, to measure MODEL_NOTES §2.1 (12 clips).
- **Optional:** 6 clips at 1024² (HD), for the record only.

Scored by every Tier-1 and Tier-2 metric. This is the **control** (B17) that every trained
checkpoint is compared against. It is not a gate: a style LoRA fixed off-style zero-shot
output on LTX-2.3 (v4).

### G0.5 — Source baseline

Measure the real clips at 512², per label:

- motion energy (mean absolute frame difference inside the character mask);
- frozen fraction;
- centroid travel;
- shadow width and head line (the zoom reference);
- ground flatness;
- DINOv2 similarity of the character crop, last vs first frame.

These are the denominators for §9. A bar borrowed from another run's motion would be
meaningless (B6).

### G0.6 — Memory and throughput

On the actual box, longest bucket (f57) at 512², batch 1:

- bf16 base with no block swap; then the smallest block swap with 0 allocator retries;
  `--fp8_base` only if nothing else fits (musubi warns it costs quality).
- Record peak GB, retries and s/clip, and re-plan §10 from them (C5–C6).

---

## 5. Corpus — `Data/processed/expv1_k5pro/`

Built by `prep/prep_expv1.py` (to write) from raw. It may copy decoding helpers from
`finetune/wan/prep_v8.py` with provenance noted; it never imports them.

### 5.1 Format

- **Size:** 512×512, re-decoded from the 1080² source straight to target size (one
  resample; the flat ground is regenerated, not resampled).
- **Frame rate:** 24 fps (A2). **Silent.**
- **Lengths:** n = 4k + 1 frames. Each label has **two rungs**:
  - a **shared floor** that every label of both kinds reaches;
  - its **native length**, capped at f57.
- **The floor is computed, not chosen:** the largest 4k + 1 length that every label reaches
  with at least half its sources, after cutting at the first missing frame (A6). Expected
  f13, because sitting sources have gaps from frame 13. Prep **asserts every label is present
  at the floor** and exits otherwise (B11).
- **Truncated rungs** are captioned `"<label>, opening frames only"`: a truncated clip shows
  only the opening of the arc, and the caption has to say so.
- Rungs are measured against **active** length (A5). The dead tail is trimmed, but never
  below the floor.

### 5.2 Content

All labels delivered in iteration_3–5:

| Kind | Labels |
|---|---|
| Expressions (8) | happy, surprised, angry, confused, crying, neutral, laughing, blushing |
| Motions (5) | walking, running, waving, sitting, jumping |

Both characters. Expected ≈ 190 training sources, i.e. **26 performances** (A1).

**Excluded:**
- the turnaround clip (an identity reference, not a performance);
- the 12 two-character clips (A14);
- files that tear on all 24 decode attempts (A7);
- files with a gap below the floor (A6).

Every exclusion is listed in `manifest.json` with its reason.

### 5.3 Holdout — fixed now (B9)

- **Angle `QF2_R`, every label, both characters (26 sources), never trained.**
  - It has trained neighbours on both sides (`QF1_R`, `QF3_R`), so it tests generalisation
    to a camera angle the model never trained on rather than extrapolation.
  - It is the start frame v8's `core` suite already used, so v8@550's existing clips can be
    re-scored for §8 without running Wan again.
- **Grounds:** lavender and sky-blue never trained.

The corpus is video-only (no stills), so nothing else can leak a held-out view.

### 5.4 Augmentation

| Applied | Rule |
|---|---|
| Alpha compositing onto 4 flat grounds | white, pastel blue, peach, mint (B15) |
| Shot-size ladder 1.00× / 0.75× / 0.55×, zoom-out only (A4) | **one** zoom per (clip, ground), round-robin, asserted balanced against label, character, ground and angle (prep exits on > 34% deviation) |

**Not applied:**
- colour jitter (brand colour is the deliverable);
- flips (near-useless on symmetric characters filmed both ways);
- random crops, rotation, temporal jitter (they mislabel the arc);
- loop tiling (A11; one variable).

Expected size: **≈ 1,300–1,500 clips**. The exposure budget is set per performance in §6.2,
so the corpus size does not drive memorisation.

### 5.5 Captions

The content rules carry over (A10, B11–B13). The wording is new, and frozen once written.
One template; the **label clause is the only part that changes between labels**:

```text
2D cartoon animation in the Pudgy Penguins style, thick clean black outlines, flat pastel colors.
{Pax, a short round blue penguin | Polly, a short round pink penguin}, {action description}; {label}[, opening frames only].
{close-up | medium | wide} shot, eye level, {angle clause}. Fixed camera, locked-off shot, the character stays the same size.
Plain {ground} background.
```

- **Colour:** "blue" in every Pax caption, "pink" in every Polly caption, never "pink" in a
  Pax caption ("rosy" for the blush). Asserted.
- **Length:** ≤ 200 Qwen tokens per caption (the encoder truncates at 256). Asserted.
- **Negative prompt, everywhere:** `photorealistic, 3D render, realistic lighting, gradient
  shading, soft shadows, textured background, blurry, distorted, deformed, low quality,
  worst quality`.
  - It contains **no** "static", "still" or camera words (B13; the v8 zoom experiments).
  - It contains **no** "cartoon" or "2D" (MODEL_NOTES §2.1).
- **Prompt rewriting:** off everywhere (MODEL_NOTES §2.2).

### 5.6 Assertions (prep exits non-zero)

- frame count exact (ffprobe on every file);
- no padded duplicate frames;
- floor bucket contains every label;
- shot-size balance;
- colour anchors;
- caption token count;
- holdout sources absent from training;
- MD5 uniqueness;
- naming disagreements all resolved (A9).

---

## 6. Recipe — `configs/train_expv1.sh` (to write)

### 6.1 Arm A

| Knob | Value | Why |
|---|---|---|
| Task | `k5-pro-i2v-5s-sd` | the SD I2V preset, full attention |
| Input layout | the G0.3 winner, same in training and inference; frame 0 out of the loss | MODEL_NOTES §2.3 |
| Network | `networks.lora_kandinsky`, **rank 32 / α 32**, default targets: self- and cross-attention q/k/v/out + FFN, modulation excluded | the same targets as the official trainer; D5 |
| Optimiser | AdamW8bit, **LR 1e-4**, betas (0.9, 0.95), weight decay 1e-3, grad clip 1.0 | a fresh LoRA at α/r = 1: the official trainer's and musubi's documented value. Not inherited from v8 (E) |
| Schedule | warmup 20, cosine → 10% | D4 (the official 1,000-step warmup would exceed the whole run) |
| Batch | 1 × **grad-accum 8** | D4 |
| Timesteps | `--timestep_sampling shift --discrete_flow_shift 5.0` | the same logit-normal + shift-5 schedule as the official trainer (checked in both codebases) |
| Budget | **≤ 600 optimizer steps** = 4,800 clips | §6.2 |
| Saving | every 50 steps + state, keep all | D3 |
| Precision | bf16; base in bf16 unless G0.6 forces fp8 | — |
| In-training sampling | off | B8; it would also need a second model in memory |
| Seed | 42 | — |

### 6.2 Exposure budget

4,800 clips ÷ 26 performances ≈ **185 exposures per performance**:

- LTX-2.5 memorised at ~500 exposures per performance;
- v8 plateaued at ~90 (D6).

The gates are expected to plateau inside this budget. If Tier 1 is still improving at
step 600, report it, and extend only by a stated number.

### 6.3 Gate breaks — stop, measure, resume (resume patch, C2)

| Step | Measure | Rule (fixed now) |
|---|---|---|
| **100** | Tier 1 on the holdout | Both kinds' losses below the base (G0.4) on ≥ 75% of clips, paired. **If not, stop and re-check the input layout before touching LR**: a setup error is likelier than a learning-rate problem on a fresh LoRA. If the loss is *above* the base, restart at LR 5e-5 |
| **300** | Tier 1, plus a motion probe (5 actions × 2 characters × 1 seed, `QF2_R`) | Margins above the base; the motion probe is recorded for the D1 curve |

### 6.4 Pre-declared fallback — arm B (D1)

After training, the motion probe runs on every 100-step checkpoint. **Trigger:** median
motion energy falls by ≥ 25% from its best checkpoint across ≥ 3 consecutive probed
checkpoints while Tier-1 expression accuracy holds. That is the LTX-2.5 pattern.

**Then:** train **arm B**, the same recipe at **rank 16 / α 16**. LTX-2.5's own
recommendation was a lower rank for motion. The verdict (§9) is computed for each arm;
expv1 passes if either does, and both are reported.

---

## 7. Evaluation

All generation goes through `kandinsky5_generate_video.py` (patched layout if G0.3 chose B),
with:

- start frames pre-sized to **exactly 512×512**, and the output size asserted
  (MODEL_NOTES §2.7);
- the Pudgy negative prompt, rewriting off, 50 steps, CFG 5, the task's `scheduler_scale`;
- one model load per configuration (C4).

| Tier | What | When |
|---|---|---|
| **0** | LoRA norm, consecutive-update cosine (B18) | every checkpoint, CPU |
| **1** | **Diffusion classifier** (`eval/dcls.py`, layout A, clips f13 motion / **f17** expression: f21 is truncated by the VAE, MODEL_NOTES §2.8) on the 26 held-out `QF2_R` clips: is the true caption the best denoiser among all 13 label captions? Loss on frames ≥ 1, change-masked, t ∈ {0.65, 0.85}, k = 2 noise draws identical across checkpoints (B4). Paired comparisons with bootstrap CIs | base (G0.4), gate breaks, every 50-step checkpoint after training |
| **2** | Generated suites, 3 seeds (B2), scored with CLIP/DINOv2 1-NN against the **real source clips** (B3), plus the G0.5 measures | winner + base |

**Winner rule (Tier 1):**
1. Among checkpoints whose expression *and* motion held-out losses beat the base on ≥ 90%
   of clips, take the highest expression + motion margin.
2. If several are within 0.001, take the **earliest** (less memorisation).
3. **Motion veto:** if that checkpoint's probed motion energy is < 0.8× the best probed
   checkpoint's, take the best-motion checkpoint within 0.002 of the top score.

**Tier-2 suites:**

| Suite | Content | Clips |
|---|---|---|
| `core` | 13 labels × 2 characters × {FRONT (trained angle), `QF2_R` (held out)} × 3 seeds, at **f13 motion / f17 expression** (amended 2026-10-13 from f21, which Kandinsky's VAE truncates; v8's f21 clips are trimmed to f17 for §8b). `eval/gen_core.py` | 156 |
| `native` | each label at its native length, `QF2_R`, 3 seeds: does the action complete? | 78 |
| `gb` | 3 labels × 2 characters × {lavender, sky-blue} × 3 seeds | 36 |

The base runs `core` only (the control). Total: **≈ 426 clips**.

---

## 8. Reference read — context, not a gate

*(Superseded as the comparison with v8 by §8b, added 2026-10-13 before any expv1 training.
The read below stays as context for §9.)*

v8@550's `core` clips already exist (Azure `pudgy/v8/eval/suites/core/v8@550/`: 52 clips,
seed 42, FRONT + `QF2_R`, f13/f21). Downscale them to 512² and score them with **expv1's**
scorer. Compare them with expv1's seed-42 `core` subset: same start frames, same labels,
same lengths.

**Confounds, stated:**
- v8 ran at 1024² and saw `QF2_R` through its v7 base;
- the prompts are worded differently;
- n is small (±0.1–0.2 per cell).

**Use:** it answers *"does expv1 fail where v8 also fails?"* If yes, the shortfall is
probably the data (A13). If no, it is probably the base. It never changes a §9 outcome.

---

## 8b. Is expv1 better than v8@550? — pre-registered (2026-10-13), `eval/compare_v8.py`

Fixed before any expv1 checkpoint exists. Every comparison is **paired** on the same held-out
material, and the verdict follows from the numbers alone.

**Material.**
- **Tier 1** (does the model *know* the label?): the 26 `QF2_R` source clips. Each model scores
  every label caption of the clip's kind; correct = the true caption has the lowest loss.
  v8: `v8/eval/dcls/v8@550_k2.json`. expv1: `eval/dcls/<ckpt>__A_k2.json`. Paired by
  (character, label).
- **Tier 2** (does it *show* the label?): `core` seed 42, 13 labels × 2 characters ×
  {FRONT, `QF2_R`} = **52 pairs**. Same start frame for both models, scored by the same code at
  the same window (f13 / f17) against the same real-clip bank (leave-angle-out CLIP 1-NN, same
  character, the kind's labels).

**Primary endpoints** (superiority; exact one-sided McNemar on the discordant pairs, Holm over
the two): **P1** Tier-1 held-out accuracy (26 pairs); **P2** Tier-2 CLIP 1-NN accuracy (52 pairs).
The posterior P(expv1 better) = P(θ > ½), θ ~ Beta(b+1, c+1), is reported next to each p.

**Non-inferiority gates** (all must hold for "better"; 90% paired bootstrap CIs):

| | Gate | Holds when |
|---|---|---|
| N1 | Style (blinded rubric, `--rubric-template`) | lower bound of (expv1 − v8) pass rate ≥ −0.10, and expv1 colour correct on 100% |
| N2 | Motion | lower bound of mean log(expv1 / v8 motion ratio) ≥ log 0.8 |
| N3 | Camera | upper bound of mean (\|zoom\| expv1 − \|zoom\| v8) ≤ +0.01 |
| N4 | Late-frame identity | lower bound of mean (expv1 − v8) DINOv2 similarity ≥ −0.05 |
| N5 | Ground | expv1 median corner drift ≤ 5/255 (V6) |

**Verdict:**

| Outcome | When |
|---|---|
| **BETTER** | every N holds, ≥ 1 primary significant (Holm p < 0.05), neither primary's estimate favours v8 |
| **WORSE** | v8 significantly better on a primary |
| **NOT BETTER** | a non-inferiority gate fails |
| **PROBABLY BETTER** | every N holds, both primaries favour expv1 with P(better) ≥ 0.80, nothing significant |
| **NO DETECTABLE DIFFERENCE** | otherwise |

"Replacement candidate" = BETTER **and** §9 PASS.

**Validated on v8's own runs (2026-10-13):**
- **v8 against itself:** b = c = 0 everywhere, so no difference is found.
- **v2 golden against v8:** **WORSE**. v8 is significantly better on P1 (Holm p = 0.0004).
- **v7fixed against v8:** **no detectable difference** (P1 b = 1, c = 6: one-sided p = 0.06, Holm 0.125).
- **Exact test, false positives:** under a simulated null it fires 2.9% of the time at
  α = 0.05 (n = 52).

**Power, stated up front:**
- **P2:** 52 pairs detect 0.80 vs 0.60 with power ≈ 0.80.
- **P1:** sits near the ceiling, because v8 is already at 0.81. 0.95 vs 0.81 is detected only
  ≈ 28% of the time.
- **So** a "no detectable difference" on P1 is not evidence that the two are equal.

**Confounds:** resolution (v8 1024² resized to 512²), captions, v8 saw `QF2_R` through v7,
Tier-1 lengths f21 vs f17, one seed per model.

---

## 9. Pre-registered verdict

All bars are on the **winner**, from **unseen `QF2_R` start frames**, over 3 seeds. Bars
marked † sit near v8's measured level; they are not derived from it and are not a
comparison with it.

| # | Bar | Pass when |
|---|---|---|
| V1 | VAE | G0.1 passed |
| V2 | Style and identity | ≥ 90% of `core` `QF2_R` clips pass the manual rubric (flat fills, intact black outlines, no 3D shading, on-model shape); colour correct in **100%** |
| V3 | Tier 1 | held-out accuracy ≥ 0.75 expression (chance 0.125) and ≥ 0.70 motion (chance 0.20)†; both losses below the base on ≥ 90% of clips |
| V4 | Controllability | CLIP 1-NN label accuracy ≥ 0.45 expression and ≥ 0.40 motion† |
| V5 | Motion | per action: median motion energy ≥ 0.8× source, and frozen fraction ≤ source + 15 points (G0.5) |
| V6 | Camera and ground | \|zoom\| ≤ 3%; corner drift ≤ 5/255 on unseen grounds; ground pixel spread ≤ 2× the source's (no gradients, B16) |
| V7 | Late-frame identity | last-vs-first DINOv2 similarity of the character ≥ 0.9× the source's (B16) |
| V8 | Memorisation | trained-angle (FRONT) minus held-out-angle (`QF2_R`) 1-NN accuracy ≤ 0.15 (B10) |

| Outcome | Meaning | Next |
|---|---|---|
| **PASS** (V1–V8) | the base learns Pudgy and stays controllable on unseen views | propose `expv2` (1024², end-frame anchoring) |
| **PARTIAL** (V1–V3 pass; some of V4–V8 fail) | it learns the labels, but generation misses bars | read §8: data-limited or base-limited, then decide |
| **FAIL at V1 or V2** | the VAE or the style prior does not suit this art at 512² | stop; record why |

---

## 10. Cost — estimates, to be replaced by G0.6

Tokens per latent frame at 512² = 32 × 32 = 1,024; f57 = 15 latent frames ≈ 15k tokens, a
quarter of v8's f57 at 1024².

| Item | Estimate | Basis |
|---|---|---|
| Training | ≈ **12 s/clip** on an A100 → 4,800 clips ≈ **16–25 h** | 19B forward ≈ 2 × 19e9 × 10k tokens; fwd + bwd + recompute ≈ 4×; ~40% of A100 bf16 peak |
| Generation | ≈ 2–3 min/clip on an A100 (f21, 100 forward passes) | the same arithmetic at 6k tokens |
| Phase 0 | G0.4 90 clips ≈ 4 h; G0.3 + Tier-1 baselines ≈ 2 h; G0.1/G0.5 < 1 h | — |
| Tier 1 sweep | 12 checkpoints × 26 clips × 13 captions × 2 draws × 2 t | measure in Phase 0 |
| Tier 2 | ≈ 426 clips ≈ **15–20 h**; motion probe on 6 checkpoints ≈ 2–3 h | — |
| **Total GPU** | **≈ 45–60 h** on one A100-80GB, less on an H100 | — |

Memory: the bf16 DiT is ~38 GB. Latents and text are cached, so neither the VAE nor Qwen
is resident while training. bf16 is expected to fit on 80 GB; G0.6 decides.

---

## 11. Risks and open questions

| Risk | Handling |
|---|---|
| The conditioning patch is wrong | G0.2 dumps prove the layout; G0.3 lets the checkpoint choose |
| musubi's Kandinsky support is experimental | pinned tag; the G0.2 smoke test; every flag checked against `--help` before launch |
| Prompt following is the vendor-admitted weakness | V3 vs V4 separates "knows the label" (likelihood) from "shows the label" (generation) |
| 1080 → 512 halves outline width in pixels | G0.1 is judged at 512² by eye |
| The VAE's licence (Tencent) | check before any client-facing use; it does not block the experiment |
| FlashAttention 3 is Hopper-only | SDPA / FlashAttention 2 on an A100 |
| Box loss | Azure mirror on a timer from the first hour (C1) |

**Open, and not answered by expv1:** 1024² behaviour, two characters, end-frame anchoring,
composition, clips longer than the delivered footage.

---

## 12. Runbook (scripts marked † are still to write)

```bash
# 0. Environment
git clone https://github.com/kohya-ss/musubi-tuner /workspace/musubi-tuner
git -C /workspace/musubi-tuner checkout v0.3.4
git -C /workspace/musubi-tuner apply expv1-kandinsky-50-video-pro/patches/musubi-v0.3.4-resume-step.patch      # †copy
git -C /workspace/musubi-tuner apply expv1-kandinsky-50-video-pro/patches/expv1-k5-i2v-conditioning.patch      # †
#    download the DiT, VAE and encoders (section 3); write their HF SHAs to eval/phase0/pins.json

# 1. Phase 0
python expv1-kandinsky-50-video-pro/eval/vae_gate.py                 # † G0.1  -> stop on fail
python expv1-kandinsky-50-video-pro/eval/cond_check.py               # G0.2
python expv1-kandinsky-50-video-pro/eval/build_refs.py               # Tier-1 clips, real-clip bank, start frames
#   cache: kandinsky5_cache_latents.py --dataset_config /workspace/expv1/refs/tier1/tier1.toml --vae <VAE dir> --first_frame_image_encode
python expv1-kandinsky-50-video-pro/eval/dcls.py score --configs base --layouts A A1 B stock --true-only
python expv1-kandinsky-50-video-pro/eval/dcls.py g03                 # G0.3
python expv1-kandinsky-50-video-pro/eval/dcls.py score --configs base --split holdout seen   # Tier-1 control
python expv1-kandinsky-50-video-pro/eval/gen_core.py --config base   # Tier-2 core, seed 42
python expv1-kandinsky-50-video-pro/eval/probe.py --config base      # † G0.4
python expv1-kandinsky-50-video-pro/eval/baseline.py                 # † G0.5

# 2. Corpus
python expv1-kandinsky-50-video-pro/prep/prep_expv1.py --dry-run     # † plan only
python expv1-kandinsky-50-video-pro/prep/prep_expv1.py --workers 16  # † -> Data/processed/expv1_k5pro/
python src/musubi_tuner/kandinsky5_cache_latents.py --dataset_config <expv1.toml> --vae <vae>
python src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py --dataset_config <expv1.toml> \
    --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct --text_encoder_clip openai/clip-vit-large-patch14

# 3. Train (arm A) — the flags configs/train_expv1.sh† will wrap
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/kandinsky5_train_network.py \
  --task k5-pro-i2v-5s-sd --dit <kandinsky5pro_i2v_sft_5s.safetensors> --vae <vae> \
  --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct --text_encoder_clip openai/clip-vit-large-patch14 \
  --dataset_config <expv1.toml> --mixed_precision bf16 --sdpa --gradient_checkpointing \
  --network_module networks.lora_kandinsky --network_dim 32 --network_alpha 32 \
  --optimizer_type AdamW8Bit --learning_rate 1e-4 \
  --optimizer_args "weight_decay=0.001" "betas=(0.9,0.95)" --max_grad_norm 1.0 \
  --lr_scheduler cosine_with_min_lr --lr_warmup_steps 20 --lr_scheduler_min_lr_ratio 0.1 \
  --gradient_accumulation_steps 8 --max_train_steps 600 \
  --timestep_sampling shift --discrete_flow_shift 5.0 --i2v_layout A \
  --save_every_n_steps 50 --save_state --seed 42 \
  --output_dir /workspace/expv1_output --output_name expv1-k5pro-a \
  --log_with all --log_tracker_name pudgy --wandb_run_name expv1-k5pro-a

# 4. Gate breaks at 100 and 300, then Tier 1 on every checkpoint, the motion probe, the winner rule
# 5. Tier 2 on the winner and the base; section 9 verdict -> docs/REPORT.md
# 6. Section 8b: is expv1 better than v8@550? (v8 refs: az pudgy/v8/eval/{dcls,suites/core} -> /workspace/v8_ref)
python expv1-kandinsky-50-video-pro/eval/compare_v8.py --a-name expv1@<step> \
    --a-dcls /workspace/expv1/eval/dcls/expv1@<step>__A_k2.json \
    --a-gen /workspace/expv1/eval/gen/expv1@<step>/core --rubric-template     # rate blind, then rerun without the flag
```
