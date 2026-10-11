# What expv2 carries over from earlier runs, and what it does not

The lessons themselves, with their sources, are indexed in expv1's register:
[`../../expv1-kandinsky-50-video-pro/docs/LEARNINGS_CARRIED.md`](../../expv1-kandinsky-50-video-pro/docs/LEARNINGS_CARRIED.md).
That register is *read*, not inherited as a dependency. Every tag below (A1–E) means the
lesson with that tag there. This file says **how each one applies to H3**, where it differs
from expv1, and adds the facts that are new (N1–N5).

| Kind | Meaning for expv2 |
|---|---|
| **A, B, C** | Facts about the footage, measurement and operations. Applied as-is unless a row says otherwise |
| **D** | Hypotheses measured on another base. **Re-tested** here, never assumed |
| **E** | Does not carry over |
| **N** | New in expv2, measured on 2026-10-11 from the raw files or found while writing this plan |

---

## A. Facts about the footage

| # | Lesson (short) | How expv2 applies it |
|---|---|---|
| A1 | One performance filmed from up to 9 cameras | 12 trained labels × 2 characters = **24 performances**. Budget counted per performance (PLAN §6.4); the holdout is a camera angle |
| A2 | 24 fps, on ones | H3 is native 24 fps, but its anime prior is on twos (MODEL_NOTES 2.12). Cadence is measured (bar V10) |
| A3 | ProRes 4444 with real alpha | Composite onto flat grounds explicitly |
| A4 | Character fills ~91–92% of frame height | Shot sizes by **zooming out** only: 1.00 / 0.75 / 0.55 |
| A5 | 37.9% frozen at the median; `neutral` 96.6% | Upper rungs (39/56) are built only where the source's **active** length reaches them. `neutral` is exempt, because holding is its content |
| A6 | Files are missing frames; ffmpeg pads them with duplicates | Cut before the first missing frame. Re-counted for expv2 (N1, N2): **sitting loses 13 of 18 sources** at the 22-frame floor |
| A7 | ProRes decodes tear non-deterministically | Tear check on every written frame, up to 24 decode attempts |
| A8 | Later deliveries re-send earlier files | MD5 de-duplication: 526 raw `.mov` → 240 unique |
| A9 | Naming is unreliable | Same handling as expv1. Known cases this time: two clips in `MOTION_WALKING/PAX/` named `WAVING`, and iteration_3's `QF_L`/`QF_R`/`Right` angle names. Prep exits while any disagreement is open |
| A10 | Polly renders blue without "pink" | Colour word in every caption; Pax's blush is "rosy". Asserted |
| A11 | `running` does not loop | **Running is excluded** from training (17 frames, below the floor). **Walking is loop-tiled** 16 → 22: a deliberate departure from expv1, because walking otherwise has no legal length. v8 measured walking's seam at 0.62–1.04 |
| A12 | `confused` and `happy` delivered short; `confused` thinnest | Both are 21 frames, padded by **one** held frame to 22. **Sitting** is now the thinnest class (N2). No upweighting |
| A13 | Fast motion is data-limited on every base | Running is not tested at all until Round 5. Motion shortfalls are read against v8@550 (PLAN §8) |
| A14 | Not delivered: idle, sad, scared, …; two-character clips | Out of scope |

## B. Facts about measurement

| # | Lesson (short) | How expv2 applies it |
|---|---|---|
| B1 | Loss and weight drift said "fine" while prompts were ignored | Gates are behavioural or likelihood-based; loss is logged, never gating |
| B2 | Seed 42 alone hid 40–66% failure rates | ≥ 3 seeds on every generated gate |
| B3 | SSIM recovers the seed, not the label | CLIP/DINOv2 1-NN against the real source clips |
| B4 | Diffusion classifier on held-out clips | **Re-implemented for H3** (PLAN §7.2): in-context start-frame rows, frame-0 latent excluded, within-kind captions. The t pair is calibrated on **trained-angle** clips at the pilots, then frozen, so the holdout stays clean. The model is distilled, so its conditional prediction is the guided one; margins are not comparable to v8's in absolute terms |
| B5 | ~350 clips per checkpoint to separate 10 points | Checkpoint choice from Tier 1; Tier 2 confirms |
| B6 | A threshold is only valid for the motion it was calibrated on | Motion bars relative to **this** corpus's sources (G0.5) |
| B7 | Metrics picked the VAE that dissolves outlines | G0.1: the eye check at 4× decides; metrics are recorded. Positive-gradient edge mask with coverage reported |
| B8 | Trainer-side samples are not the inference path | Generate only through DiffSynth's `MiniMaxH3Pipeline`, the same code that defines the training layout |
| B9 | Decide the holdout before launch | `QF2_R` for every label, fixed now: **23 clips** at 22 frames (Pax sitting `QF2_R` has 17 usable frames, so it is a start frame only) |
| B10 | Training start frames reward memorisation | Verdict on unseen start frames; the seen–unseen gap is bar V8 |
| B11 | Length must not predict the label | The 22-frame floor contains **every trained label**, asserted. The 39/56 rungs exist for four expressions only, which B11 allows (N4) |
| B12 | Contrastive labels make the label promptable | One template; the label clause is the only part that changes |
| B13 | A default negative prompt can fight the captions | H3 has no negative prompt, so the failure cannot happen, and the lever is gone too. The camera clause "fixed camera, locked-off shot, the character stays the same size" stays |
| B14 | Prompt enhancers inject off-style vocabulary | H3's rewriter (Context-IR) is not released; nothing rewrites prompts |
| B15 | Flat grounds from alpha generalise | White, pastel blue, peach, mint trained; lavender and sky-blue held out |
| B16 | Soft-gradient grounds; late-frame identity drift | Both are measured bars (V6, V7) |
| B17 | A control is needed | The untrained base runs the `core` and `native` suites (G0.4) and Tier 1 (G0.3) |
| B18 | Drift-share fit assumes constant LR | Tier 0 reports the consecutive-update cosine |
| B19 | Pre-register the decision rule | PLAN §8 and §9 are fixed in this commit |

## C. Operational facts

| # | Lesson (short) | How expv2 applies it |
|---|---|---|
| C1 | Box disk is not persistent | Azure mirror on a timer from the first hour |
| C2 | musubi v0.3.4 `--resume` resets the step counter | **Not applicable** (DiffSynth). DiffSynth's own resume is undocumented (MODEL_NOTES 2.15), so the plan does not depend on resuming |
| C3 | Trainer naming quirks | Record DiffSynth's checkpoint names in G0.2 and rename into `step-NNNN` |
| C4 | Model loading dominated inference | Two-pass evaluation: Qwen3-VL encodes everything once, then one DiT load per checkpoint |
| C5 | Re-probe memory whenever clip length changes | G0.6 probes the **f56** bucket with guidance loss on |
| C6 | Re-plan from the first hour's s/it | Applied |
| C7 | Dry runs that look like real gates | `--dry` persists nothing; uploads keep relative paths |
| C8 | Download path ≠ the path the trainer reads | Verify against the path each script reads |
| C9 | `pkill -f` killed its own shell; W&B regex delete | Kill by PID; delete runs by exact id |

## D. Hypotheses from other bases — re-tested

| # | Observed on | What expv2 does with it |
|---|---|---|
| D1 | LTX-2.5: one LoRA could not carry motion and expression | Tracked by the motion probe on every checkpoint. No rank fallback: expv2's pre-declared fallback is for **distillation** (arm D), the risk specific to H3 |
| D2 | Wan v8: one joint LoRA did carry both | The counter-example; which way H3 falls is a finding |
| D3 | Best checkpoint came early | Save every 50; H3 reports suggest best ≤ ~2,000 samples (Eval §6.1) |
| D4 | Batch 1 + constant LR = noisy updates; accum 8 + cosine fixed it | Accumulation 8 + cosine, patched in if DiffSynth has no scheduler (MODEL_NOTES 2.15). Tier 0 checks |
| D5 | Effective rank ≈ 9 of 32 on Wan | Rank 32 not expected to bind |
| D6 | ~500 exposures/performance memorised; v8 plateaued at ~90 of ~185 | 3,920 samples ≈ **163 per performance** |
| D7 | Expressions learned more easily than motion | Expected; reported separately |
| D8 | CogVideoX's and Wan's 8× VAEs kept the outlines | Not assumed for H3's 16× VAE. G0.1 measures it, with Wan's VAE on the same clips as a reference |
| D9 | A character LoRA can freeze motion | Covered by D1's tracking |

## E. Does not carry over

| Item | Why not |
|---|---|
| Wan's expert routing, goldens, `v7fixed`, `v8-merged`, merge algebra, any LoRA file | Different architecture; the isolation rule |
| Rank 16 / α 32, LR 3e-5 | Artefacts of continuing from a rank-16 golden |
| Numeric Wan gate bars (face-SSIM 0.92, x-range 200 px) | Metric- and motion-specific (B3, B6) |
| `blocks_to_swap` values | No block swap here; memory re-probed (C5) |
| LTX specifics; expv1's Kandinsky specifics (negative prompt, prompt expansion, `visual_cond` patch) | Different bases |
| v8's caption wording | Re-authored into H3's three-field format; the content rules carry over |
| v8's 1024² and f13–f57 ladder | expv2 trains at 768² on the 17n+5 grid |

---

## N. New in expv2

| # | Fact | Source | Effect |
|---|---|---|---|
| N1 | **The v7 manifest's `source_frames` are not source lengths.** They are v7's 4k+1 rungs (angry 37 vs a true 40, neutral 57 vs 60), and v7 never detected missing frames | Raw re-count, 2026-10-11 (ffprobe decoded timestamps, the `prep_v8.real_frame_gap` method; 21 damaged files, matching v8 NOTES) | Prep measures every length from the raw file. Eval §4.1 was corrected the same day |
| N2 | **Sitting has gaps in 13 of 18 sources**, at frames 13–25. Only 5 reach 22 frames (Pax FRONT, Pax QF3_R, Polly QF3_R, SIDE_L, SIDE_R), or 7 with one held frame (Polly QF1_R, QF2_R at 21). Pax sitting `QF2_R` has 17 | Same | Sitting trains on 6 sources and is held out on 1. It is the class to watch, ahead of confused |
| N3 | **A training shift cannot be chosen by base-model likelihood.** The per-timestep loss does not depend on the shift; only the weighting over timesteps does | Reasoning (I), correcting Eval §6.3 and §6.5 | Shift chosen by two pilot LoRAs (PLAN §6.3) |
| N4 | **Every motion trains at exactly 22 frames on today's data.** Only angry, laughing, blushing and neutral reach 39, and only neutral reaches 56 | Raw re-count | Motion results speak for 0.92 s only. The `native` suite tests 39/56 frames for those four expressions alone |
| N5 | **Pruned-build LoRAs are not portable** to the full build, and the DeCFG adapter has a separate file per layout | DiffSynth example script | One build end to end; parity checked once (G0.2) |
