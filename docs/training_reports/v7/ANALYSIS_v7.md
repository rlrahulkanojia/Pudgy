# v7 — post-hoc analysis: what the weights and the eval clips actually say

**Date** 2026-09-26 · **Inputs** 19 checkpoints from Azure (`v2/weights/curated`,
`v7/weights/*`), the 84 G-C clips at steps 500 / 2500 (`v7/eval/refine/`), and the gate
JSON. No retraining — every number here is computed from artifacts that already exist.
Scripts: [`analysis/`](analysis/). Companion to [`REPORT_v7.md`](REPORT_v7.md), whose
conclusions this revises in three places (§1, §3, §4).

Notation: base weights W, v2 low-noise golden G, expression LoRA L_e = G + τ_e, motion
LoRA L_m = G + τ_m. A LoRA module contributes ΔW = (α/r)·B·A. Every inner product,
norm and spectrum below is computed **exactly** from the r×r cores —
⟨B₁A₁, B₂A₂⟩_F = tr((B₁ᵀB₂)(A₂A₁ᵀ)), σ(UV) = σ(R_U R_Vᵀ) — no out×in matrix is formed.

---

## 1. Loading the two v7 LoRAs together counts the golden twice

Both v7 LoRAs were continue-trained **from the same golden**, so each file contains G:

    W + L_e + L_m = W + 2G + τ_e + τ_m          intended: W + G + τ_e + τ_m

| | Frobenius norm (all 400 modules) |
|---|---|
| ‖G‖ (v2 low golden) | 31.48 |
| ‖τ_e‖ (expr step 2500 − G) | 17.60 |
| ‖τ_m‖ (motion final − G) | 14.65 |
| **error / intended update** | **‖G‖ / ‖G+τ_e+τ_m‖ = 0.76** |

Naive stacking is off by 76% of the intended update. **REPORT_v7 §3.3's "G-S is
unnecessary" does not hold**: with both kinds on the low-noise expert, composing them is
now the untested case, and the obvious way to do it is wrong.

**Zero-training fix** (musubi merges every `--lora_weight` additively):
`--lora_weight expr.safetensors motion.safetensors v2_low_GOLDEN.safetensors --lora_multiplier 1 1 -1`
— wired up as config `v7fixed` in `finetune/wan/eval_v8/common.py`.

Second-order overlap — the two task vectors are far from independent:

| | value | random baseline |
|---|---|---|
| cos(τ_e, τ_m), all modules | 0.33 | ~0 |
| … self-attn q / k | 0.54 / 0.58 | |
| … cross-attn q/k/v | 0.10–0.13 | |
| top-8 input-subspace overlap | 0.67 | 8/5120 = 0.0016 |
| top-8 output-subspace overlap | 0.24 | |

Both LoRAs learned a common "v7 data" component (flat pastel grounds, 1024², the shot
grammar) — mostly in self-attention q/k. Summing them counts it twice as well.

## 2. After ~500 steps the updates were mostly noise

Batch 1, constant LR 3e-5, Adam (musubi's default scheduler is `constant`):

| step | 250 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|---|
| ‖τ_e(t)‖/‖G‖ | 0.15 | 0.22 | 0.33 | 0.42 | 0.49 | 0.56 | 0.63 | 0.68 |
| cos(τ_e(t), τ_e(3600)) | 0.38 | 0.51 | 0.68 | 0.78 | 0.86 | 0.92 | 0.96 | 1.00 |

- ‖τ(t)‖ ∝ t^0.58 (expression), t^0.53 (motion). Pure drift gives 1.0, a random walk 0.5.
- Consecutive 500-step increments have cosine **0.24, 0.19, 0.16, 0.15, 0.10** — each
  block of updates points somewhere nearly unrelated to the last.
- Fitting ‖τ‖² = D²t² + σ²t: the systematic (drift) share of ‖τ‖² is **5% at step 500,
  19% at 2500, 25% at 3500**.

So the LoRA kept moving (to 69% of the golden's own norm) mostly along directions the
loss is flat in. That is the mechanism behind REPORT §3.1's flat sweep: **choosing step
2500 over step 500 was choosing between two noise draws.** Also: effective rank of τ
≈ 8.9 of 32, 97% of energy in the top 16 — rank was never the constraint; 44% (expr) /
37% (motion) of the update energy sits in the last 8 of 40 blocks.

## 3. G-C's face-SSIM measures the seed, not the label

On the 84 G-C clips (7 labels × 3 seeds × 2 characters × 2 checkpoints), 1 − face-SSIM:

| Pax @ 2500 | distance |
|---|---|
| different label, **same** seed (what G-C reports) | 0.198 |
| **same** label, different seed | 0.208 |
| different label, different seed | 0.223 |

**Two different expressions from one seed are more alike than one expression from two
seeds.** 1-NN on SSIM recovers the seed 67–81% and the label 5–19% (chance 14%). The label
effect is real (permutation p < 0.001) but smaller than the seed effect (0.014 vs
0.016–0.025). The 0.004 checkpoint "spread" and the `confused|neutral @ seed 42` failure
are inside that noise.

A semantic embedding tells a different story — CLIP ViT-L/14 on the back half of each clip:

| | CLIP 1-NN label | CLIP zero-shot | DINOv2 1-NN | seed recovered (CLIP) |
|---|---|---|---|---|
| step 500 Pax / Polly | 0.62 / 0.62 | 0.52 / 0.48 | 0.57 / 0.52 | 0.19 / 0.14 |
| step 2500 Pax / Polly | 0.57 / 0.71 | 0.52 / 0.43 | 0.38 / 0.48 | 0.29 / 0.24 |

**The model does control the expression; the metric could not see it.** And there is no
semantic difference between 500 and 2500 either (SE ≈ ±0.11 at n = 21). Power: telling two
checkpoints apart by 10 accuracy points needs ≈ 350 clips each — impossible at 6.4
min/clip, which is why v8 adds a likelihood-based classifier (Tier 1).

## 4. G-X measured the data more than the expert

In flow matching, x_t = (1−t)x₀ + tε and at t ≥ 0.9 the SNR ((1−t)/t)² ≤ 0.012: the optimal
prediction is ≈ E[x₀ | first frame, caption]. The high-noise expert can therefore only
learn a label through the **between-label variance of the conditional mean**,
Var_y E[x₀ | c, y]. Every v7 action is in place from the same start pose (centroid drift
≤ 15 px), so the *mean* jump and the *mean* wave are nearly the same image — that
variance is ≈ 0 by construction. M-high losing is what the maths predicts for this
dataset, not evidence that motion in general belongs on the low-noise expert.

## 5. Data defects found while building v8

- **Missing frames padded with duplicates.** 21 sources (all 18 iteration_4 sitting, 4
  confused, 1 walking, 1 blushing) are short of real frames — e.g. `PAX_MOTION_SITTING_QF1_R`
  has no frames 15–17. ffmpeg's constant-rate output fills each gap with a duplicate, the
  alpha tear check passes (a duplicate is not torn), and v7 trained f21/f25 sitting rungs
  across such gaps (a frozen stutter mid-motion). "Pixel-verified clean" in REPORT_v7
  was checking tearing, not completeness. Detected in v8 by frame timestamps.
- **`EXPRESSIONS/HAPPY/Pax` skipped** as "unknown character" (case-sensitive lookup vs
  `PAX`). Harmless while it was an md5 re-send; 7 files were re-uploaded 2026-09-26.
- **Colour anchor hazard**: a Pax caption must never contain "pink" (Polly's anchor).

## Reproduce

```bash
# weights: download pudgy/v2/weights/curated/*GOLDEN_ep40* and pudgy/v7/weights/** into
# one folder, flattening "/" -> "__"; clips: pudgy/v7/eval/refine/step0000{0500,2500}/
V7_WEIGHTS=/path/to/w python docs/training_reports/v7/analysis/run_geom.py   # §1, §2
python docs/training_reports/v7/analysis/ctrl.py                             # §3 SSIM
python docs/training_reports/v7/analysis/sem.py                              # §3 CLIP/DINO
```
