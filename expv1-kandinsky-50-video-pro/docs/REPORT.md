# expv1 — findings report (Kandinsky 5.0 Video Pro I2V + one LoRA)

**Status (2026-10-14, 20:40Z):** both arms trained and compared with v8@550.
**Arm A** (rank 32): winner **expv1@550**, NO DETECTABLE DIFFERENCE if the style rubric passes,
NOT BETTER if it fails (§11). **Arm B** (rank 16, §12): winner **expv1b@500**, **NOT BETTER**
(fails N2 motion: front-view jumps do not happen in the window). **Neither arm is better than
v8@550; v8 stays the production model.** Arm A remains expv1's best checkpoint.

Every number below comes from a file under `/workspace/expv1/eval/` (mirrored to Azure
`pudgy/expv1-kandinsky-50-video-pro/eval/`). Each section names its source file and the
script that produced it.

---

## 0. Summary so far

| Question | Answer | Evidence |
|---|---|---|
| Does Kandinsky's VAE keep Pudgy art? | **Yes** (G0.1 PASS) | §2.1 |
| Which I2V input layout does the checkpoint expect? | **A** (official inference), by 15.6% loss, 26/26 clips | §2.3 |
| Does the LoRA learn the corpus? | **Yes**: held-out true-caption loss below the base model's on **26/26** clips at every checkpoint | §5 |
| Does it learn to tell **expressions** apart? | **Yes, mostly**: held-out accuracy 0.375 (base) → **0.75** at step 550 (v8: 0.81) | §5 |
| Does it learn to tell **motions** apart? | **No**: held-out accuracy 0.30 (base) → 0.30 at step 550, 0.20 at 600. It peaks at 0.50 at step 350 and fades | §5 |
| Does it move? | Yes, too much: probed motion is 1.4–2.1× the real clips' energy, and up to 4.5× on `sitting` | §6 |
| Winner (PLAN §7 rule) | **expv1@550**: step 600 scored highest but the motion veto applied | §7 |
| Better than v8@550 on Tier 1? | **No**: 0.58 vs 0.81 pooled (b = 4, c = 10, not significant, p = 0.09 for v8). Expression is close (0.75 vs 0.81); motion is far behind (0.30 vs 0.80) | §8 |
| Better than v8@550 overall? | **No.** Both primaries favour v8 (Tier 1 0.58 vs 0.81; Tier-2 CLIP 1-NN 0.385 vs 0.423), neither significantly. Non-inferiority N2-N5 hold (after a zoom-measurement fix, §11.2). Outcome: NO DETECTABLE DIFFERENCE if the rubric passes, NOT BETTER if it fails. BETTER is ruled out either way | §11 |
| Where expv1 is ahead | Steadier camera (\|zoom\| −0.033 vs v8, p = 0.005), less motion overshoot (1.62× vs 1.79× source), DINOv2 1-NN expression 0.53 vs 0.25 (secondary, not gated) | §11 |
| Where expv1 is behind | Motion labels (Tier 1 0.30 vs 0.80; Tier-2 CLIP 0.45 vs 0.65), late-frame identity (−0.011), ground drift (4.1 vs 2.0 /255, under the 5/255 bar), and a visible frame-0 → frame-1 shadow/tint pop | §11 |
| Does a lower rank (arm B, 16/16) help? | **No.** Motion labels a little up (Tier 1 0.40 vs 0.30), expression down (0.625 vs 0.75), DINOv2 1-NN 0.27 vs 0.56 (p = 0.0007 for A), still 1.7× source motion | §12 |
| Arm B vs v8@550 | **NOT BETTER**: P1 0.54 vs 0.81 (unadjusted p = 0.046 for v8), P2 0.35 vs 0.42; **N2 fails** (−0.237, CI lower −0.351 < −0.223), driven by FRONT jumps at 0.55× source vs v8's 1.35× | §12.6 |
| expv1 overall (PLAN §6.4) | **Not better than v8 with either arm** | §12.8 |

---

## 1. Setup

| | |
|---|---|
| Base | `kandinsky5pro_i2v_sft_5s.safetensors` (19B, single DiT), task `k5-pro-i2v-5s-sd`, 512×512 |
| Trainer | musubi-tuner v0.3.4 + `patches/` (resume-step continuation; `--i2v_layout` in trainer **and** generator) |
| LoRA | arm A: rank 32 / α 32, from zero, `networks.lora_kandinsky` |
| Optimiser | AdamW8bit, LR 1e-4, cosine to 0.1×, warmup 20, grad accum 8, weight decay 0.001, betas (0.9, 0.95), clip 1.0 |
| Timesteps | shift sampling, discrete flow shift 5.0 |
| Steps | 600, save every 50, gate breaks at 100 and 300 |
| Box | 1× A100-80GB (Vast.ai). Peak 57.5 GB at f57, bf16, **no block swap** (G0.6). ≈ 38 s/optimizer step at accum 8 (≈ 3 h 10 m for steps 300 → 600) |
| Config / services | `configs/train_expv1.sh`; supervisor `expv1-train` + `expv1-mirror` (Azure every 15 min) via `scripts/install_services.sh` |

---

## 2. Phase 0 (measure before training)

### 2.1 G0.1: VAE gate. **PASS**
`eval/vae_gate.py` → `eval/phase0/vae_gate/report.json`. Six clips (Pax/Polly × angry, waving,
running) go through Kandinsky's HunyuanVideo-VAE at 512², with Wan2.1's VAE as the reference.
Bars: centreline ≥ 0.95, within 0.03 of the reference, ≤ 2 broken outlines, highlight ≥ 0.95,
ground drift ≤ 2, fill std ≤ 1, fill MAE ≤ 2. All pass. Example (`pax_angry`, white): PSNR
37.7, SSIM 0.994, centreline min 0.985, broken outlines 0 (Wan2.1: 3).

### 2.2 G0.2: conditioning check. **PASS**
`eval/cond_check.py` → `eval/phase0/cond_check/report.json`. Dumps the I2V tensors the trainer
builds under each layout and compares them with the generator's:

| Layout | Noisy frame 0 = clean first frame | Cond frame 0 = first frame | Loss frames (of 5) |
|---|---|---|---|
| A (official inference) | yes | no (zeros) | 4 |
| A1 (musubi generator) | yes | yes | 4 |
| B | no | yes | 4 |
| stock trainer | no | yes, plus frame 1 = **last** frame | 5 (mask all ones) |

The stock trainer leaks the last frame into the conditioning and puts frame 0 in the loss
(MODEL_NOTES). After the patch, `gen_A` (generator in layout A) matches trainer layout A on 9/9
checks. f21 is refused by the VAE path (`f21_refused.json`), so the expression length is f17
(MODEL_NOTES §2.8).

### 2.3 G0.3: which layout does the checkpoint expect? **A**
`eval/dcls.py g03` → `eval/phase0/g03_layout.json`. The base model, no LoRA, scores the true
caption's denoising loss on the 26 held-out `QF2_R` clips (t ∈ {0.65, 0.85}, k = 2 identical
noise draws). Lower = the model recognises its own input format.

| Layout | Mean true-caption loss | vs A | Clips where A is lower | Wilcoxon p | 95% CI of the gap |
|---|---|---|---|---|---|
| **A** | **0.0342** | — | — | — | — |
| A1 | 0.0395 | +15.6% | 26/26 | 3.0e-8 | [0.0044, 0.0063] |
| B | 0.0423 | +23.6% | 26/26 | 3.0e-8 | [0.0066, 0.0096] |
| stock | 0.2495 | +630% | 26/26 | 3.0e-8 | [0.211, 0.220] |

**Decision:** train **and** generate in A. musubi's generator used A1, so it was patched
(`--i2v_layout A`, MODEL_NOTES §2.12). That way training and inference see identical inputs.

### 2.4 G0.4: base model zero-shot (the control)
- **Tier 1** (layout A, `eval/dcls/base__A_k2.json`): held-out expression acc **0.375**
  (chance 0.125), motion **0.300** (chance 0.20); seen angle (`QF2_L`) 0.312 / 0.111.
- **Tier 2 `core`:** 26/52 seed-42 clips generated (`eval/gen/base/core`), paused to free
  the GPU for training, and resumable (`gen_core.py --config base`).

### 2.5 G0.6: memory
f57 at 512², bf16, no block swap: 57.5 GB peak, no allocator retries.

### 2.6 Not yet run
- G0.5 as a standalone per-label source baseline (frozen fraction, twos index, zoom
  reference). §6 uses the real `QF2_R` f13 clips' motion energy as the denominator, which
  covers what the probe needs. The full baseline is still needed for §9 V5–V7.

---

## 3. Corpus (`prep/prep_expv1.py` → `Data/processed/expv1_k5pro/`)

| | |
|---|---|
| Clips | **1,860** from **195** sources, 26 performances |
| Lengths | floor f13; rungs = floor + eval length + "top" (the shortest legal length covering the active arc) |
| Per length | f13 784 · f17 568 · f25 276 · f33 180 · f57 52 |
| Grounds | white, pastel blue, peach, mint (lavender and sky blue never trained: the `gb` suite) |
| Zooms | 1.0, 0.75, 0.55 |
| Captions | frozen §5.5 grammar, 1,380 distinct, ≤ 124 Qwen tokens. A *complete-action* caption only if the arc is held to the end (≤ 3 frames slack) and not gap-cut |
| Holdout | the `QF2_R` camera angle, every character and label, never trained |
| Exclusions | 35 (listed in `manifest.json`), incl. `PAX_MOTION_SITTING_QF2_L` (torn) |
| Natives updated | angry 25, blushing 33, laughing 41. Two Polly `confused` sources honestly truncated at f17 |
| Caches | 1,860 latents + 1,860 text (`/workspace/expv1/cache`) |

---

## 4. Gate breaks (PLAN §6.3), automated by `scripts/gate_breaks.sh`

Each break: wait for checkpoint + state → stop training → Tier 1 on held-out + seen →
`eval/gate_rule.py` → restart → **verify** the step count continued (`continuing at global step S`
and a progress line at S+1..S+3).

### Step 100: **PASS → continue** (`eval/reports/gate_100.json`)
Rule: both kinds' held-out true-caption loss below the base's on ≥ 75% of clips.

| | Below base | Loss drop | 95% CI | Wilcoxon p | Acc (base) | Margin (base) |
|---|---|---|---|---|---|---|
| Expression (16) | 16/16 | −16.6% | [0.0045, 0.0065] | 1.5e-5 | 0.375 (0.375) | −0.0148 (−0.0070) |
| Motion (10) | 10/10 | −22.4% | [0.0056, 0.0108] | 0.00098 | 0.300 (0.300) | −0.0017 (−0.0084) |

Resume verified at step 100. Reading: the LoRA lowered **every** caption's loss (style first),
without yet separating labels.

### Step 300: margins **not** above base on motion (reported, not a stop rule) (`gate_300.json`)

| | Below base | Loss drop | 95% CI | Wilcoxon p | Acc (base) | Margin (base) |
|---|---|---|---|---|---|---|
| Expression | 16/16 | −26.5% | [0.0068, 0.0108] | 1.5e-5 | **0.625** (0.375) | −0.0002 (−0.0070) ✓ |
| Motion | 10/10 | −32.6% | [0.0082, 0.0156] | 0.00098 | 0.200 (0.300) | −0.0145 (−0.0084) ✗ |

Resume verified at step 300. **Deviation:** the step-300 motion probe was not run at the break,
because the gate script omitted it. It was run after training instead (§6), and
`gate_breaks.sh` is now fixed to include it.

### Training loss (running average per segment, from the tqdm logs)

| Step | 50 | 100 | 150 | 200 | 300 | 350 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|---|---|
| avr_loss | 0.0173 | 0.0150 | 0.0114 | 0.0115 | 0.0111 | 0.0093 | 0.0095 | 0.0093 | 0.0089 |

(The average resets at each restart (steps 100 and 300), so the values are comparable only
within a segment.)

---

## 5. Tier 1 on every checkpoint (diffusion classifier, `eval/dcls.py`)

**Logic.** For each held-out `QF2_R` source clip (16 expression at f17, 10 motion at f13), the
model denoises the clip under **every** label caption of that kind (8 expressions or 5 motions).
Loss is on frames ≥ 1, change-masked, t ∈ {0.65, 0.85}, k = 2 noise draws **identical across
checkpoints**. Correct = the true caption has the lowest loss. Margin = mean(other captions'
loss) − true loss; positive = the true caption wins on average. No sampler is involved, so this
measures what the model *knows*.

Held-out (`QF2_R`, unseen angle), accuracy / mean margin:

| Config | Expression (n = 16) | Motion (n = 10) | Sum of margins | Both kinds below base on |
|---|---|---|---|---|
| base | 0.375 / −0.0070 | 0.300 / −0.0084 | −0.0154 | — |
| @50 | 0.562 / −0.0011 | 0.300 / −0.0196 | −0.0207 | 100% / 100% |
| @100 | 0.375 / −0.0148 | 0.300 / −0.0017 | −0.0166 | 100% / 100% |
| @150 | 0.500 / −0.0125 | 0.300 / +0.0006 | −0.0119 | 100% / 100% |
| @200 | 0.375 / −0.0081 | 0.400 / −0.0053 | −0.0135 | 100% / 100% |
| @250 | 0.500 / −0.0062 | 0.400 / +0.0005 | −0.0057 | 100% / 100% |
| @300 | 0.625 / −0.0002 | 0.200 / −0.0145 | −0.0148 | 100% / 100% |
| @350 | 0.500 / −0.0033 | **0.500 / +0.0090** | +0.0058 | 100% / 100% |
| @400 | 0.438 / −0.0016 | 0.400 / +0.0054 | +0.0039 | 100% / 100% |
| @450 | 0.500 / +0.0011 | 0.400 / +0.0016 | +0.0027 | 100% / 100% |
| @500 | 0.500 / +0.0070 | 0.300 / −0.0025 | +0.0045 | 100% / 100% |
| **@550** | **0.750 / +0.0130** | 0.300 / +0.0032 | +0.0161 | 100% / 100% |
| @600 | 0.688 / +0.0148 | 0.200 / +0.0026 | **+0.0174** | 100% / 100% |
| *v8@550 (its own scorer)* | *0.81* | *0.80* | | |

Seen angle (`QF2_L`, trained), accuracy / margin:

| Config | Expression (16) | Motion (9) |
|---|---|---|
| base | 0.312 / −0.0154 | 0.111 / −0.0096 |
| @300 | 0.562 / −0.0011 | 0.333 / −0.0122 |
| @350 | 0.688 / +0.0018 | 0.556 / −0.0123 |
| @550 | 0.562 / +0.0107 | 0.333 / −0.0184 |
| @600 | 0.625 / +0.0100 | 0.222 / −0.0263 |

**Findings.**
1. **Expression is learned.** The margin rises roughly monotonically from −0.015 to +0.015,
   and accuracy doubles (0.375 → 0.75). Held-out (unseen angle) ≥ seen accuracy at 550, so
   this is not angle memorisation.
2. **Motion is not learned as a label.** The margin hovers around zero, and accuracy peaks
   at step 350 (0.50) and returns to chance by 600. Even on the **trained** angle, motion
   accuracy never exceeds 0.56, and its margin is negative at every checkpoint.
3. Errors at step 550 (held-out): expression misses are Pax angry → crying, Pax happy →
   crying, Pax surprised → laughing, Polly confused → angry. Motion misses are Pax jumping →
   walking, Polly jumping → sitting, Pax running → jumping, Polly running → walking, Pax sitting
   → waving, Pax walking → sitting, Polly walking → running. The model confuses the locomotion
   cycles with each other, and `sitting` with `waving`.

---

## 6. Motion probe (PLAN §6.3–6.4), `eval/gen_core.py --suite probe` + `eval/motion_probe.py`

**Logic.** 5 motion labels × 2 characters × `QF2_R` start frame × seed 42, f13, 50 steps, CFG 5,
Pudgy negative, layout A (10 clips per config; the same files are part of the `core` suite).
Motion energy = mean |grey frame difference| inside the character mask (`metrics.clip_measures`).
Ratio = clip energy / the real `QF2_R` source clip's energy for the same character and label
at f13. Run on base + every 100-step checkpoint, plus @550 (the only checkpoint within 0.002
of the top Tier-1 score, which the veto could need). Runner: `scripts/run_probe.sh`.
Output: `eval/reports/motion_probe.json`.

| Config | Median energy | Median ratio to source |
|---|---|---|
| real `QF2_R` sources | 7.59 | 1.00 |
| base | 9.25 | 1.22 |
| @100 | **20.14** | 2.06 |
| @200 | 10.91 | 1.44 |
| @300 | 16.61 | 2.11 |
| @400 | 13.39 | 1.70 |
| @500 | 12.95 | 1.69 |
| @550 | 15.35 | 1.88 |
| @600 | 14.10 | 1.66 |

Per clip, ratio to source (Pax | Polly):

| Config | walk | run | wave | sit | jump | walk | run | wave | sit | jump |
|---|---|---|---|---|---|---|---|---|---|---|
| source energy | 7.5 | 20.5 | 6.8 | 3.5 | 16.7 | 7.6 | 18.7 | 7.1 | 3.6 | 15.1 |
| base | 1.22 | **0.45** | 1.73 | 2.30 | **0.46** | 1.22 | 0.72 | 1.99 | 2.52 | 0.59 |
| @100 | 2.45 | 1.31 | 2.06 | 3.18 | 1.70 | 2.86 | 1.36 | 2.07 | 2.72 | 1.46 |
| @200 | 1.44 | 0.90 | 1.47 | 2.95 | 1.11 | 1.44 | 1.10 | 1.53 | 2.63 | 1.11 |
| @300 | 2.35 | 1.18 | 2.25 | 2.94 | 1.19 | 2.61 | 1.24 | 1.97 | 2.72 | 1.02 |
| @400 | 1.80 | 1.00 | 1.93 | 2.92 | 0.99 | 1.69 | 1.12 | 1.70 | 3.01 | 1.01 |
| @500 | 1.72 | 0.97 | 1.80 | 2.81 | 0.92 | 1.69 | 1.05 | 1.70 | 2.72 | 0.94 |
| @550 | 1.91 | 1.04 | 2.01 | **4.51** | 1.06 | 1.86 | 1.10 | 1.94 | **4.25** | 1.02 |
| @600 | 1.66 | 0.99 | 2.01 | 2.88 | 0.94 | 1.66 | 1.10 | 1.77 | 4.07 | 0.95 |

**Findings.**
1. **Training fixed under-motion on the big actions.** The base model runs and jumps at
   0.45–0.72× the source. Every trained checkpoint is at ≈ 0.9–1.3×.
2. **Small actions over-move.** `walking`/`waving` sit at 1.5–2.9× and `sitting` at 2.6–4.5×
   the source (sitting's source energy is tiny, 3.5, so any body bob inflates the ratio). Above
   1.0, more energy is **not** better. It is extra movement the source does not have, and the
   rubric/§9 V5 frozen-fraction checks will show whether it is jitter or real action.
3. **D1 trigger (arm B): no.** The rule is median energy ≤ 0.75× its best-so-far on ≥ 3
   consecutive probed checkpoints. Energy drops after step 100 (20.1 → 10.9) but recovers
   (16.6, 13.4, 12.9, 14.1): no run of 3.

---

## 7. Winner rule (PLAN §7), `eval/winner.py` → `eval/reports/winner.json`

1. **Eligible:** both kinds' held-out losses below the base on ≥ 90% of clips → all 12
   checkpoints (100% / 100% each).
2. **Highest expression + motion margin:** @600 = +0.0174. Nothing else is within 0.001
   (@550 = +0.0161, gap 0.0013), so the earliest-within-0.001 rule does not move it.
3. **Motion veto:** @600's probed median energy 14.10 < 0.8 × the best probed (20.14 at @100)
   = 16.11 → **veto applies**. Take the best-motion checkpoint within 0.002 of the top score:
   candidates @600 (14.10) and @550 (15.35) → **@550**.

**Winner: expv1@550.**

**Caveat on the veto.** The veto takes "best motion" to mean "highest energy". Every checkpoint
is above the source's energy, and the best probed one (@100, 2.06× source) is the most
over-moving, so here the veto rewards overshoot. It was followed as pre-registered. The cost is
small: @550 trails @600 by 0.0013 in Tier-1 score, but has **higher expression accuracy (0.75 vs
0.69) and higher motion accuracy (0.30 vs 0.20)**. A future plan should cap the veto's
reference at the source's energy (e.g. |log ratio| instead of raw energy).

---

## 8. Comparison with v8@550 (PLAN §8b), `eval/compare_v8.py`

### 8.1 Method (pre-registered before any expv1 checkpoint existed)
- **P1 (Tier 1):** held-out accuracy on the same 26 `QF2_R` source clips, pooled over both kinds,
  paired by (character, label). v8's rows: `v8/eval/dcls/v8@550_k2.json` (its own scorer, f13 /
  f21). expv1's: `eval/dcls/expv1@550__A_k2.json` (f13 / f17).
- **P2 (Tier 2):** CLIP 1-NN label accuracy of generated `core` clips, seed 42, 13 labels × 2
  characters × {FRONT, `QF2_R`} = 52 pairs, same start frame, same window (f13 / f17; v8's
  1024² f21 clips resized and trimmed), leave-angle-out against the same real-clip bank.
- **Tests:** exact one-sided McNemar on discordant pairs (b = expv1 right & v8 wrong, c = the
  reverse); Holm over P1 and P2; posterior P(expv1 better) = P(θ > ½), θ ~ Beta(b + 1, c + 1).
- **Non-inferiority** (90% paired bootstrap): N1 blinded style rubric (lower bound of the
  difference ≥ −0.10, colour 100%); N2 motion log-ratio ≥ log 0.8; N3 |zoom| ≤ +0.01; N4
  late-frame identity ≥ −0.05; N5 median corner drift ≤ 5/255.
- **Verdict:** BETTER / WORSE / NOT BETTER / PROBABLY BETTER / NO DETECTABLE DIFFERENCE (PLAN §8b).

### 8.2 Validation of the comparison (on v8's own runs; `eval/stats.py`)
| Run | Expected | Got |
|---|---|---|
| v8 vs itself | no difference | b = c = 0 ✓ |
| v2 golden vs v8 | WORSE | WORSE, Holm p = 0.0004 ✓ |
| v7fixed vs v8 | close | no detectable difference (b = 1, c = 6, p = 0.06, Holm 0.125) ✓ |
| Simulated null, n = 52 | ≤ 5% false positives | 2.9% ✓ |

Power: P2 detects 0.80 vs 0.60 with ≈ 0.80; P1 detects 0.95 vs 0.81 only ≈ 28% of the time (v8
is near the ceiling), so "no detectable difference" on P1 is **not** evidence of equality.

### 8.3 Tier-1 read so far (interim: Tier 2 not yet in)

| A vs v8@550 | Pooled acc | Expression | Motion | b / c | p (v8 better) | P(A better) |
|---|---|---|---|---|---|---|
| base | 0.346 vs 0.808 | 0.375 vs 0.812 | 0.300 vs 0.800 | 0 / 12 | **0.00024** | 0.0001 |
| **expv1@550** | 0.577 vs 0.808 | 0.750 vs 0.812 | 0.300 vs 0.800 | 4 / 10 | 0.090 | 0.06 |

Files: `eval/reports/compare_base_vs_v8-550.json`,
`eval/reports/compare_expv1@550-tier1only_vs_v8-550.json`.

**Reading.** Training closed most of the expression gap (expression b = 3, c = 4, P(better) =
0.36: indistinguishable at n = 16). The motion gap is untouched (b = 1, c = 6, one-sided p =
0.06 for v8). Pooled, v8 is ahead and the evidence is suggestive, not significant (p = 0.09).
On Tier 1 expv1@550 is **not better** than v8. Whether it *shows* labels better in generation
is P2, below.

---

## 9. Code added for this phase

| File | Purpose |
|---|---|
| `eval/gate_rule.py` | step-100 / step-300 gate rules → `gate_<S>.json` |
| `scripts/gate_breaks.sh` | stop → Tier 1 → gate → restart → verify resume; Tier 1 on every checkpoint after training. **Now also runs the probe at 300** |
| `eval/gen_core.py --suite probe` | the motion probe (subset of `core`, same filenames, reused by the suite) |
| `eval/motion_probe.py` | probe energies vs real sources, D1 trigger → `motion_probe.json` |
| `scripts/run_probe.sh` | probe over base + every 100-step checkpoint (+ extras) |
| `eval/winner.py` | PLAN §7 winner rule incl. motion veto → `winner.json` |
| `scripts/after_probe.sh` | winner → `core` suite for the winner → `compare_v8.py` |

---

## 10. Deviations from the plan (stated, not hidden)

1. Step-300 motion probe: run after training, not at the break (§4).
2. Tier 2 uses **seed 42 only** for the v8 comparison (that is all v8 has). PLAN §7's 3-seed
   `core`, `native` and `gb` suites are still to run for §9.
3. Base `core` is at 26/52 clips.
4. G0.5 full source baseline not yet computed (§2.6).
5. `zoom_estimate` reference fixed after the first expv1 vs v8 run (§11.2); validation runs
   unchanged by it.

---

## 11. Tier 2 and the verdict vs v8@550 (`eval/reports/compare_expv1-550_vs_v8-550.json`)

`core` suite: `eval/gen_core.py --config expv1@550 --seeds 42`, which gives 52 clips at 512²:
13 labels × 2 characters × {FRONT, `QF2_R`}, f13 motion / f17 expression, 50 steps, CFG 5,
layout A, start frame = neutral frame 0 on white. Every clip's frame count and size are
asserted. 10 of the 52 are the probe clips (same files). v8@550's 52 clips come from Azure
`pudgy/v8/eval/suites/core/v8@550`, resized 1024² → 512² and trimmed to the same window. Both
sets are scored by the same code.

### 11.1 Results

**Primary endpoints** (one-sided exact McNemar; b = expv1 right & v8 wrong, c = the reverse):

| | expv1@550 | v8@550 | b / c | p (expv1 better) | p (v8 better) | Holm p (v8 better) | P(expv1 better) |
|---|---|---|---|---|---|---|---|
| **P1** Tier-1 held-out acc (26) | 0.577 | 0.808 | 4 / 10 | 0.97 | 0.090 | 0.18 | 0.06 |
| **P2** Tier-2 CLIP 1-NN acc (52) | 0.385 | 0.423 | 7 / 9 | 0.77 | 0.40 | 0.40 | 0.31 |

Per kind (secondary):

| | Expression expv1 / v8 | b / c | Motion expv1 / v8 | b / c |
|---|---|---|---|---|
| Tier 1 | 0.750 / 0.812 | 3 / 4 | 0.300 / 0.800 | 1 / 6 (p = 0.06 for v8) |
| Tier-2 CLIP 1-NN | 0.344 / 0.281 | 5 / 3 | 0.450 / 0.650 | 2 / 6 |
| Tier-2 DINOv2 1-NN | **0.531 / 0.250** | 15 / 6 (p = 0.039, unadjusted) | 0.600 / 0.600 | 5 / 5 |
| Tier-2 all, DINOv2 | 0.558 / 0.385 | 20 / 11 (p = 0.075) | | |

CLIP 1-NN predictions collapse towards `laughing` for **both** models (expv1 18/32 expression
clips, v8 14/32), the weak expression sensitivity of CLIP at f17 already noted in validation.
DINOv2 separates expressions better and favours expv1 there. That is a secondary measure,
reported and not gated, so it cannot change the verdict.

**Non-inferiority** (90% paired bootstrap CI):

| Gate | Measured | Bar | Holds |
|---|---|---|---|
| N1 style (blinded rubric) | not rated yet: 104-clip blinded sheet `eval/reports/rubric_expv1-550_vs_v8-550.json` (key in `..._key.json`, do not open while rating) | lower bound ≥ −0.10, colour 100% | **pending** |
| N2 motion, mean log(expv1 / v8 ratio) | −0.047, CI [−0.129, +0.033]; median ratio to source expv1 1.62, v8 1.79 | lower bound ≥ log 0.8 = −0.223 | yes |
| N3 camera, mean \|zoom\| expv1 − v8 | **−0.033**, CI [−0.049, −0.017], Wilcoxon p = 0.005 (expv1 steadier) | upper bound ≤ +0.01 | yes (after fix, §11.2) |
| N4 late-frame identity, DINOv2 | −0.011, CI [−0.021, −0.001] | lower bound ≥ −0.05 | yes |
| N5 ground, median corner drift | expv1 4.08 /255 (v8 2.02) | ≤ 5 /255 | yes |

### 11.2 A measurement bug found and fixed: the zoom reference (N3)

**First run:** NOT BETTER on N3, with a mean \|zoom\| difference of **+4.18** (CI [2.6, 5.8]). The
zoom is a fraction, so that claimed a ≈ 400% scale change. Implausible, so it was checked
before being accepted. The raw output is kept: `compare_expv1-550_vs_v8-550.raw_zoom.json`.

**Cause.** `zoom_estimate` takes the ground-shadow width in the last third of the clip relative
to **frame 0**. In expv1's clips the VAE-decoded frame 0 has a slightly lighter, bluish shadow:
RGB (209, 213, 215), against a target of ground × 0.8 = 202 ± 14. The real start frame has 203,
and v8's frame 0 has 202. At the edge of the tolerance the shadow mask fragments and is opened
away, so frame 0 measured **24–176 px** against **≈ 293 px** on every later frame. The
character's height did not change (444 → 428 px, −4%). The contact sheets confirm no zoom.

**Fix** (`eval/metrics.py: zoom_estimate`). Frame 0 is the reference only if its shadow
width is within 20% of the median of frames 1–3, since no camera changes scale by 20% in 1/24 s.
Otherwise frame 1 is the reference. This is applied to both models.

**Checks that the fix is not goal-post moving:**
- v8 vs itself, golden vs v8 and v7fixed vs v8 give **identical** numbers before and after (zoom
  diffs 0.0 / −0.0076 / −0.0102). The fix only changes clips whose frame-0 mask failed.
- After the fix, expv1 is as steady or steadier than v8 on 16/20 motion pairs, worse by at most
  +0.014 on the other 4.
- A first, looser version of the fix (frame 0 valid if ≥ 0.5× the clip median) still let two
  clips through at 173 / 176 px (zoom +0.64 / +0.69), and N3 still failed. Those were inspected:
  same artifact. Hence the 20% rule.
- The fix was made **after** seeing the result. It is stated here so a reader can weigh it.
  Without it the formal label would be NOT BETTER; with it, PENDING: N1. Neither is BETTER.

**What it does reveal:** expv1's frame 0 → frame 1 has a visible **shadow/tint pop**: the shadow
darkens and widens, and the background picks up a slight grey tint on some frames. That is a
real (small) defect of expv1's I2V output, also reflected in its higher corner drift (4.1 vs
2.0 /255), and the rubric rater should look for it.

### 11.3 Verdict

| Condition | Status |
|---|---|
| Every N holds | N2–N5 yes; N1 pending |
| ≥ 1 primary significant for expv1 (Holm) | **no**: Holm p = 1.0 for both |
| Neither primary favours v8 | **no**: both point estimates favour v8 |
| v8 significantly better on a primary | no (Holm 0.18 / 0.40) |

- If N1 holds, the result is **NO DETECTABLE DIFFERENCE**. If N1 fails, it is **NOT BETTER**.
  BETTER and PROBABLY BETTER are impossible because both primaries favour v8.
- **Replacement candidate: no.**
- **Plain reading.** expv1@550 has caught up with v8 on expressions: Tier 1 is close, and
  generation is level on CLIP and ahead on DINOv2. It has a steadier camera and less motion
  overshoot. It does **not** know or show motion labels as well as v8, and that is where the gap is.
- **Power caveat.** With 26 and 52 pairs, P1 detects 0.95 vs 0.81 only ≈ 28% of the time, so
  "no detectable difference" is not proof of equality. v8 is not proven better either: Holm
  p = 0.18 on P1.

### 11.4 Not done (PLAN §9 needs these; not part of the §8b comparison)
- N1 / §9 V2: human rating of the blinded rubric.
- 3-seed `core`, `native` and `gb` suites for the winner; base `core` (26/52); the G0.5 full
  source baseline. §9 V3 already **fails** on Tier 1: motion 0.30 < 0.70 (expression 0.75
  meets its 0.75 bar). So the best §9 outcome available is PARTIAL.

---

## 12. Arm B: rank 16 / α 16 (PLAN §6.4)

**Why it was trained.** The pre-declared D1 trigger (motion energy ≤ 0.75× its best on ≥ 3
consecutive probed checkpoints) did **not** fire for arm A (§6). Arm B was trained anyway, on
request, as a direct test of the LTX-2.5 lesson that a lower rank keeps motion. Everything
except rank and α is identical to arm A (§6.1 recipe, data, seed 42, 600 steps, same gates and
scorers). PLAN §6.4: the verdict is computed per arm; expv1 passes if either arm does.

**Wiring.** Config prefix `expv1b@<step>` → `output/expv1-k5pro-b-step*.safetensors`; reports
carry an `_expv1b` suffix (`gate_expv1b_*.json`, `motion_probe_expv1b.json`,
`winner_expv1b.json`). Orchestration: `scripts/gate_breaks.sh` (PFX=expv1b) then
`scripts/after_training_b.sh` (probe → winner → `core` → compare vs v8@550 → compare vs arm A).
Training logs: `output/train_b_segment_to_step{100,300,600}.log`.

### 12.1 Training

Started 2026-10-14 06:57Z, finished 14:41Z (gate breaks included). ≈ 38 s per optimizer step,
the same as arm A (rank does not change the cost; the DiT dominates). The two resumes (steps 100,
300) were verified to continue at the right step. The traceback at the end of the step-300
segment log is the `KeyboardInterrupt` from `supervisorctl stop`, not a crash.

| Segment end | tqdm running avr_loss |
|---|---|
| step 100 | 0.0141 |
| step 300 | 0.0118 |
| step 600 | 0.0094 (arm A: 0.0089) |

### 12.2 Gate breaks

| | Step 100 B | Step 100 A | Step 300 B | Step 300 A | base |
|---|---|---|---|---|---|
| Expression held-out clips below base loss | 16/16 | 16/16 | 16/16 | 16/16 | – |
| Expression rel. loss drop | 14.8% | 16.6% | 25.0% | 26.5% | – |
| Expression acc / margin | **0.50** / −0.0035 | 0.375 / −0.0148 | 0.375 / −0.0009 | **0.625** / −0.0002 | 0.375 / −0.0070 |
| Motion held-out clips below base loss | 10/10 | 10/10 | 10/10 | 10/10 | – |
| Motion rel. loss drop | 19.4% | 22.4% | 28.9% | 32.6% | – |
| Motion acc / margin | 0.30 / −0.0174 | 0.30 / −0.0017 | 0.30 / **+0.0163** | 0.20 / −0.0145 | 0.30 / −0.0084 |
| Rule result | **PASS** | PASS | **PASS** (margins above base) | expression only | |

Lower rank learns slightly less in loss terms (as expected), and at step 300 it is the first
run of either arm with a motion margin above the base.

### 12.3 Tier 1 on every checkpoint (held-out `QF2_R`; every checkpoint below base loss on 100% of clips in both kinds)

| config | expr acc | expr margin | motion acc | motion margin | winner score |
|---|---|---|---|---|---|
| base | 0.375 | −0.0070 | 0.30 | −0.0084 | |
| expv1b@50 | 0.500 | −0.0037 | 0.30 | −0.0154 | −0.0191 |
| expv1b@100 | 0.500 | −0.0035 | 0.30 | −0.0174 | −0.0209 |
| expv1b@150 | 0.562 | −0.0065 | 0.20 | −0.0125 | −0.0190 |
| expv1b@200 | 0.562 | +0.0035 | 0.30 | −0.0114 | −0.0079 |
| expv1b@250 | 0.625 | +0.0009 | 0.30 | +0.0020 | +0.0029 |
| expv1b@300 | 0.375 | −0.0009 | 0.30 | +0.0163 | +0.0153 |
| expv1b@350 | 0.625 | +0.0036 | 0.20 | +0.0052 | +0.0089 |
| expv1b@400 | 0.438 | +0.0021 | 0.30 | +0.0094 | +0.0115 |
| expv1b@450 | 0.562 | +0.0103 | 0.40 | +0.0135 | **+0.0237** |
| expv1b@500 | 0.625 | +0.0087 | 0.40 | +0.0147 | +0.0234 |
| expv1b@550 | 0.562 | +0.0120 | 0.40 | +0.0066 | +0.0186 |
| expv1b@600 | 0.500 | +0.0103 | **0.50** | +0.0034 | +0.0138 |
| *arm A winner expv1@550* | *0.750* | *+0.0130* | *0.30* | *+0.0032* | *+0.0161* |

Reading: arm B keeps motion — margin positive from step 250 on and accuracy 0.40–0.50 late,
where arm A fell to 0.30 / 0.20. It pays on expression: peak 0.625 against arm A's 0.75. Tier-1
expression is noisy step to step (0.375 ↔ 0.625 between neighbouring checkpoints: 16 clips, one
clip = 0.0625).

### 12.4 Motion probe (`eval/reports/motion_probe_expv1b.json`; source median energy 7.59)

| | base | @100 | @200 | @300 | @400 | @450 | @500 | @600 |
|---|---|---|---|---|---|---|---|---|
| median energy | 9.25 | 13.52 | **18.52** | 17.26 | 15.74 | 12.53 | 13.40 | 15.88 |
| median ratio to source | 1.22 | 1.86 | 2.24 | 2.18 | 1.91 | 1.71 | 1.71 | 1.96 |

D1 trigger: no. Per clip at @500: run 1.05–1.08×, jump 0.98–1.25× (fixed from the base's
0.46–0.59×), walk/wave 1.6–1.9×, sitting 3.2–3.3× (arm A@550: 4.3–4.5×). At @450 the
QF2_R jumps fall back to 0.52–0.55× (under-move). Lower rank did **not** reduce overall
over-motion: medians are in the same 1.7–2.2× band as arm A (1.6–2.1×).

### 12.5 Winner (`eval/reports/winner_expv1b.json`)

- Steps 1–2: **expv1b@450** (+0.0237); @500 (+0.0234) is within 0.001, and the rule takes the earliest.
- Step 3, motion veto: @450's energy 12.53 < 0.8 × the best probed (@200, 18.52) → the best-motion
  checkpoint within 0.002 of the top: **expv1b@500** (13.40). **Winner: expv1b@500.**
- Same caveat as §7: the veto treats more energy as better, and @200's 18.5 is a 2.2× overshoot.
  Here it changed little (the scores differ by 0.0003), and it happened to avoid @450's
  under-moving QF2_R jumps, but the rule should use |log ratio to source| in a future plan.

### 12.6 Verdict vs v8@550 (`eval/reports/compare_expv1b-500_vs_v8-550.json`)

Same method as §8/§11: 52-clip `core` suite, seed 42, scored by the same code.

**Primary endpoints:**

| | expv1b@500 | v8@550 | b / c | p (v8 better) | Holm p (v8 better) | P(expv1b better) |
|---|---|---|---|---|---|---|
| **P1** Tier-1 held-out acc (26) | 0.538 | 0.808 | 3 / 10 | **0.046** | 0.092 | 0.03 |
| **P2** Tier-2 CLIP 1-NN acc (52) | 0.346 | 0.423 | 2 / 6 | 0.145 | 0.145 | 0.09 |

Per kind (secondary):

| | Expression B / v8 | b / c | Motion B / v8 | b / c |
|---|---|---|---|---|
| Tier 1 | 0.625 / 0.812 | 2 / 5 | 0.400 / 0.800 | 1 / 5 |
| Tier-2 CLIP 1-NN | 0.281 / 0.281 | 1 / 1 | 0.450 / 0.650 | 1 / 5 |
| Tier-2 DINOv2 1-NN | 0.281 / 0.250 | 4 / 3 | 0.250 / 0.600 | 2 / 9 (p = 0.033 for v8) |

**Non-inferiority** (90% paired bootstrap CI):

| Gate | Measured | Bar | Holds |
|---|---|---|---|
| N1 style | blinded sheet `rubric_expv1b-500_vs_v8-550.json` (+ `_key.json`), not rated | | pending |
| N2 motion, mean log(B / v8 ratio) | **−0.237**, CI [−0.351, −0.128]; median ratio to source B 1.69, v8 1.79 | lower bound ≥ −0.223 | **NO** |
| N3 camera, mean \|zoom\| B − v8 | −0.034, CI [−0.057, −0.010] (B steadier) | upper ≤ +0.01 | yes |
| N4 late-frame identity | +0.007, CI [−0.001, +0.016] | lower ≥ −0.05 | yes |
| N5 ground, median corner drift | 3.76 /255 (v8 2.02) | ≤ 5 /255 | yes |

**N2 checked before accepting it (as N3 was in §11.2).** The mean is driven by the FRONT jumps:
log ratio −0.91 (Pax) and −0.89 (Polly). B's FRONT jump clips move at 0.54× / 0.56× source
against v8's 1.35× / 1.36×. A frame strip (Pax, FRONT, f13) shows it is real, not a metric
artifact: v8@550 is airborne from frame 8 and arm A@550 from frame 10 (of 13); expv1b@500 only crouches with
its feet tucked in the last frame. B also moves less than v8 on the sitting clips (−0.47 to
−0.57). Without the two jumps the mean is −0.16 (ratio 0.85, which would pass), but dropping
them would be goal-post moving: a jump that does not happen in the window is exactly the
failure N2 exists to catch. The visible frame-0 → frame-1 shadow/tint pop (§11.2) is present
in arm B too.

**Verdict: NOT BETTER** (N2 fails). Replacement candidate: no. Even setting N2 aside, both
primaries favour v8 and P1 is close to significant for v8 (unadjusted p = 0.046, Holm 0.092),
so the best possible reading would have been NO DETECTABLE DIFFERENCE.

### 12.7 Arm B vs arm A (`eval/reports/compare_expv1b-500_vs_expv1-550.json`, same scorer)

| | expv1b@500 | expv1@550 | b / c | note |
|---|---|---|---|---|
| P1 Tier 1 | 0.538 | 0.577 | 1 / 2 | ns |
| Tier 1 expression / motion | 0.625 / **0.400** | **0.750** / 0.300 | 0/2, 1/0 | ns |
| P2 CLIP 1-NN | 0.346 | 0.385 | 5 / 7 | ns |
| DINOv2 1-NN all | 0.269 | **0.558** | 3 / 18 | **p = 0.0007 for A** |
| DINOv2 expression / motion | 0.281 / 0.250 | **0.531 / 0.600** | 3/11, 0/7 | p = 0.029 / 0.008 for A |
| motion ratio to source (median) | 1.69 | 1.62 | | B moves less on matched clips: mean log −0.19, CI [−0.31, −0.08] |
| late-frame identity | +0.018 vs A | | | CI [+0.011, +0.027], B keeps identity better |
| corner drift | 3.76 | 4.08 | | |

Arm B does not beat arm A: the primaries are level-to-worse, arm A's one clear advantage over
v8 (DINOv2 expression 0.53) **disappears** in arm B (0.28), and B under-moves on the FRONT jumps.
B's real gains are Tier-1 motion (0.40 vs 0.30, one clip), identity (+0.018) and slightly less
drift. **Arm A (expv1@550) remains expv1's best checkpoint.**

### 12.8 What arm B settles

- **Rank is not the lever.** Halving the rank moved the trade-off (motion labels a little up,
  expression and DINOv2 similarity down) without closing the gap to v8 on any primary.
- **The LTX-2.5 lesson does not transfer as stated.** Arm A never showed the LTX-2.5 motion
  collapse (D1 did not fire), and the lower rank did not reduce over-motion overall; it
  under-moved on two jumps instead.
- **expv1 outcome (PLAN §6.4: passes if either arm does): neither arm is better than v8@550.**
  Arm A: NO DETECTABLE DIFFERENCE / NOT BETTER (pending N1). Arm B: NOT BETTER. v8@550 stays
  the production model.
