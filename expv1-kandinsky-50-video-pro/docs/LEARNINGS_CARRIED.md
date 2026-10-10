# What expv1 carries over from earlier runs, and what it does not

expv1 shares no weights, data, code or recipe with v1–v9 or with `ltx25/`
([`../README.md`](../README.md)). What it does take is **knowledge**: lessons measured on
this footage, this client and this GPU workflow. Each lesson below is sorted into one of
four kinds, so nobody later mistakes an inherited assumption for an expv1 result:

| Kind | Meaning for expv1 |
|---|---|
| **A. Fact about the footage** | Measured on the client's files; true whatever the base model. Applied as-is |
| **B. Fact about measurement** | How to evaluate without fooling ourselves. Applied as-is |
| **C. Operational fact** | How the box, the trainer and the storage fail. Applied as-is |
| **D. Hypothesis from another base** | Measured on Wan, LTX or CogVideoX. **Re-tested** here, never assumed |

Sources (links relative to the repo root):

| Tag | Document |
|---|---|
| v1 | [`training_approach/FINDINGS.md`](../../training_approach/FINDINGS.md) |
| v4 | [`training_approach/v4/README.md`](../../training_approach/v4/README.md) |
| v5 | [`docs/training_reports/v5/REPORT_happy_pilot.md`](../../docs/training_reports/v5/REPORT_happy_pilot.md) |
| v6 | [`docs/training_reports/v6/FINDINGS_v6.md`](../../docs/training_reports/v6/FINDINGS_v6.md), [`REPORT_v6.md`](../../docs/training_reports/v6/REPORT_v6.md), [`training_approach/v6/…`](../../training_approach/v6/Training_Approach_v6.md) |
| v7 | [`docs/training_reports/v7/REPORT_v7.md`](../../docs/training_reports/v7/REPORT_v7.md), [`GATES_v7.md`](../../docs/training_reports/v7/GATES_v7.md), [`ANALYSIS_v7.md`](../../docs/training_reports/v7/ANALYSIS_v7.md), [`training_approach/v7/…`](../../training_approach/v7/Training_Approach_v7.md) |
| v8 | [`docs/training_reports/v8/REPORT_v8.md`](../../docs/training_reports/v8/REPORT_v8.md), [`training_approach/v8/…`](../../training_approach/v8/Training_Approach_v8.md), [`NOTES.md`](../../training_approach/v8/NOTES.md) |
| v9 | [`training_approach/v9/Training_Approach_v9.md`](../../training_approach/v9/Training_Approach_v9.md) (plan; nothing measured) |
| ltx25 | [`ltx25/README.md`](../../ltx25/README.md), [`ltx25/docs/FINDINGS_run1.md`](../../ltx25/docs/FINDINGS_run1.md) |
| raw | `Data/raw/README.md` (outside the repo) |

---

## A. Facts about the footage — applied as-is

| # | Lesson | Source | How expv1 applies it |
|---|---|---|---|
| A1 | Each (character, label) cell is **one performance filmed from up to 9 cameras**. ~200 source files are ~26 distinct performances | ltx25 §4.1; v9 §1.1 | Exposure budget counted **per performance** (PLAN §6.2). The holdout is a camera angle; holding out a performance or an action is impossible |
| A2 | Hand-drawn at **24 fps, on ones, with eased in-betweens**. Resampling to 16 fps judders or ghosts the outlines | v8 plan §1 | Stay at 24 fps, which is also Kandinsky's native rate |
| A3 | Sources are ProRes 4444 with **real alpha**. A naive decode composites onto black | v5 §2 | Composite explicitly onto flat grounds |
| A4 | The character fills **~91–92% of frame height**. A real zoom-in would crop the body | v6 plan §4.3; ltx25 #3 | Shot sizes are made by **zooming out** (padding with flat ground) only |
| A5 | Source is **37.9% frozen at the median**; `NEUTRAL` is 96.6% frozen | ltx25 #2 | Duration rungs measured against **active** length; frozen holds inside a clip are kept |
| A6 | **21 files are missing frames**. ffmpeg pads each gap with a duplicate, and a tear check passes duplicates | v7 analysis §5; v8 NOTES | Read frame timestamps; cut every rung **before the first missing frame** |
| A7 | ProRes decoding tears non-deterministically | v7 plan; ltx25 | Tear check on every written frame, up to 24 decode attempts |
| A8 | Deliveries from iteration_4 on **re-send earlier files** (iteration_5: 216 of 242) | raw | De-duplicate by MD5 |
| A9 | Naming is unreliable: `Pax` vs `PAX`, 5 angle vocabularies, a suffix-offset QF convention, a Polly clip in Pax's folder, walking clips named `WAVING` | v7 plan §3; v8 NOTES; ltx25 | Case-insensitive matching; angle normalisation; **disagreements are listed, never silently resolved**. Character is settled by measured body hue (blue vs pink); label by a recorded manual call. Prep exits while any disagreement is open |
| A10 | **Polly renders blue when the caption omits "pink"**. Fixed by colour words in every caption, on Wan and LTX-2.5 alike | v4 §5.3; v6 #3; ltx25 §5.4 | Every caption names the colour. Pax's blush is **"rosy"**, never "pink". Asserted in prep |
| A11 | `running` does not loop (seam 2.3–2.7× the median frame step) | v8 §2 | No loop tiling in expv1 (one variable at a time) |
| A12 | `confused` and `happy` were delivered at one short length; `confused` is the thinnest class | v7 §3.2; v8 §2 | Expect them weakest; read their scores in that light; no upweighting |
| A13 | Fast motion is the least accurate on **every** base so far, and the cause is data (11 s of unique running footage, longest clip 0.71 s) | v8 §9; v9 §1 | A motion shortfall is read against the v8 reference (PLAN §8) before it is blamed on Kandinsky |
| A14 | Not delivered: idle, sad, scared, affectionate, turning. The 12 two-character clips need their own caption design | v8 NOTES | Out of scope |

## B. Facts about measurement — applied as-is

| # | Lesson | Source | How expv1 applies it |
|---|---|---|---|
| B1 | **Loss and weight drift said "fine" while the model ignored its prompts** | v5 §4.3 | Every gate is behavioural or likelihood-based; loss is logged, never gating |
| B2 | **Seed 42 alone hid 40–66% failure rates** | v4 §5.1 | ≥ 3 seeds on every generated gate |
| B3 | Face- and whole-frame SSIM recover the **seed** 67–81% of the time and the **label** at chance | v7 analysis §3 | Generated clips scored with CLIP/DINOv2 1-NN, never SSIM |
| B4 | A **diffusion classifier** on held-out clips (conditioning frame excluded, change-mask weighting, identical noise across checkpoints) separates a trained model from an untrained one on 8 clips | v8 plan §5–6 | Tier 1 selects the checkpoint; the design is re-implemented for Kandinsky's I2V layout |
| B5 | Telling two checkpoints apart by 10 accuracy points takes **~350 generated clips each** | v7 analysis §3 | Checkpoint choice comes from Tier 1; Tier 2 confirms, it does not rank |
| B6 | **A threshold is only valid for the motion it was calibrated on** (the >200 px bar falsely failed in-place actions) | v7 defect 5; v6 #7 | Motion bars are relative to **this corpus's own source clips** (G0.5), never borrowed |
| B7 | **Metrics picked the VAE that dissolves outlines**; the edge mask and shimmer ratio degenerated on flat art | ltx25 §1, §1.2 | VAE gate = 4× outline zoom by eye, plus positive-gradient edge metrics with coverage reported and absolute shimmer |
| B8 | Trainer-side validation samples are not the inference path | ltx25 README | Generate only through the inference generator, with the same conditioning layout |
| B9 | A holdout must be **decided before launch** and cover every sample type (LTX-2.5 leaked held-out-angle stills) | v7 §9; ltx25 §2.1 | `QF2_R` fixed now, for every label; video-only corpus, so nothing else can leak |
| B10 | Scoring from training start frames rewards memorisation | ltx25 #5 | Verdict scored on **unseen** start frames; the seen–unseen gap is reported |
| B11 | **Length must not predict the label** | v6 §4.1; v7 §3; ltx25 #1 | One shared floor length containing every label, asserted by prep |
| B12 | Contrastive labels (same start frame, many outcomes, with `neutral` as a real class) make the label promptable | v6 §3, #2 | Same caption grammar for all labels; the label is the only thing that changes |
| B13 | **A default negative prompt can fight the captions** ("static" in the negative vs "static … shot" in captions → zoom-out) | v8 §8 | Pudgy negative prompt everywhere; camera clause "fixed camera, locked-off shot, the character stays the same size" (the fix proposed in v9 §1.3) |
| B14 | Prompt enhancers inject off-style vocabulary | v4 plan §7 | Kandinsky's prompt rewriter off |
| B15 | Flat grounds composited from alpha generalise to unseen grounds (1–2/255 corner drift) | v5 §4.2; v8 §3.4 | Train on white / pastel blue / peach / mint; lavender and sky-blue reserved |
| B16 | A base can render the ground as a **soft gradient** despite a flat-ground caption, and identity can **drift in late frames** under first-frame-only conditioning | ltx25 §5.1–5.2 | Ground flatness and last-frame identity are measured gates |
| B17 | A control is needed to see a trend | v6 | The untrained base (G0.4) is scored by every tier |
| B18 | The drift-share fit assumes a constant LR | v8 §3.3 | Tier 0 reports the consecutive-update cosine |
| B19 | **Pre-register** the decision rule before the data exists | v8 plan §1.2 | PLAN §9 is fixed in this commit |

## C. Operational facts — applied as-is

| # | Lesson | Source | How expv1 applies it |
|---|---|---|---|
| C1 | The box disk is not persistent; the v5 training data was lost with a box | v5; v6 | Mirror weights, logs and eval output to Azure **during** the run, on a timer |
| C2 | musubi v0.3.4 `--resume` restarts the step counter at 0 (overwrites checkpoints, overruns the schedule) | v8 §7 | The resume patch is copied into `patches/`. It edits `NetworkTrainer` in `training/trainer_base.py`, which `Kandinsky5NetworkTrainer` subclasses (checked), so it applies unchanged |
| C3 | musubi writes the final epoch without a number, and `--save_path` is always a directory | v6 §5 | Step-numbered saves only; generate into a scratch directory and rename |
| C4 | **Model loading dominated inference** (501 clips → 15 loads, saving ~33 h) | v6; v7 defect 6 | One load per configuration, prompts batched |
| C5 | Block-swap is a throughput tax when memory is free, and memory must be **re-probed whenever clip length changes** | v5 §4.7; v6 §3.1; v7 defect 1 | G0.6 probes the longest bucket before launch; no inherited swap value |
| C6 | Single-bucket probes bound memory reliably and throughput loosely | v6 | Re-plan the schedule from the first hour's measured s/it |
| C7 | Dry runs wrote result files that looked like real gates; an uploader flattened subdirectories | v7 defects 7–8 | `--dry` persists nothing; uploads keep relative paths |
| C8 | Download paths ≠ the path the trainer reads; `az download-batch` needs the destination to exist | v7 defects 2–3 | Verify against the path each script reads |
| C9 | `pkill -f` killed its own shell; a W&B cleanup by name regex deleted a real run | v8 §4 | Kill by PID; delete W&B runs by exact id only |

## D. Hypotheses from other bases — re-tested, not assumed

| # | Observed on | What was seen | What expv1 does with it |
|---|---|---|---|
| D1 | LTX-2.5 run 1 | **One rank-32 LoRA could not carry motion and expression**: `waving` decayed from 1.48× to ~0.53× source motion while expressions held | **The central risk.** Tier 2 tracks motion energy per checkpoint; a pre-declared fallback arm exists (PLAN §6.4) |
| D2 | Wan v8 | One joint LoRA *did* carry both, and beat two specialist LoRAs | The counter-example to D1. Which way Kandinsky falls is the finding |
| D3 | LTX-2.5, v8, v5 | The best checkpoint came early (LTX 500–1000 of 3000; v8 plateau ≈ 300 of 600; v5 ep4 > ep18). v7 found it flat | Save every 50 steps, keep all, let Tier 1 decide |
| D4 | Wan v7 → v8 | Batch 1 + constant LR gave mostly-noise updates (consecutive cosine 0.10–0.24); accumulation 8 + cosine decay fixed it (0.55–0.63) | Adopted: accumulation 8, cosine decay. Checked again by Tier 0 |
| D5 | Wan v7 | Effective rank ≈ 9 of 32 per kind | Rank 32 is not expected to be the constraint |
| D6 | LTX-2.5; v8 | ~500 exposures per performance went with memorisation; v8 saw ~185 and plateaued at ~90 | Budget ≈ 185 per performance (PLAN §6.2) |
| D7 | Wan v8 `compose`; LTX-2.5 | Expressions are learned more easily than motion; mixed prompts favour the face (94% vs 44%) | Expect it; motion and expression bars are reported separately |
| D8 | CogVideoX, Wan | CogVideoX's 8× video VAE round-tripped this art at ~38 dB with outlines intact; Wan's 8× VAE showed no outline problems in v2–v8 output | **Not assumed** for HunyuanVideo's VAE: G0.1 measures it |
| D9 | LTX-2.3 | A character LoRA can freeze motion (it did not on cartoon content there) | Covered by D1's tracking |

## E. Does not carry over

| Item | Why not |
|---|---|
| Wan's high-/low-noise expert routing (G-X, "expression belongs on low-noise", the frozen high-noise golden) | Kandinsky is one transformer with no expert split |
| v2 goldens, `v7fixed`, `v8-merged`, any LoRA file | Different architecture, and the isolation rule |
| Rank 16 / α 32 and LR 3e-5 | Forced by continuing training from the rank-16 golden; expv1 trains a fresh LoRA |
| v7/v8 LoRA-merge algebra | Nothing is merged here |
| Numeric bars from Wan gates (face-SSIM 0.92, x-range 200 px, …) | Metric- and motion-specific (B3, B6) |
| `blocks_to_swap` values | Architecture-specific memory; re-probed (C5) |
| LTX specifics (Gemma encoder, `frames % 8 == 1`, ÷32 sizes, IC-LoRA) | Different base |
| The exact v8 caption wording | Tuned for UMT5. The **content rules** carry over (A10, B11–B13); the wording is re-authored for Qwen2.5-VL's 256-token encoder, then frozen between training and evaluation |
| v8's 1024² resolution | expv1 runs at 512² (PLAN §5.1). 1024² would be a separate experiment |
