# expv2-minimax-h3 — plan

**Status:** plan, written 2026-10-11, nothing run. **Isolated experiment**: see
[`../README.md`](../README.md). Lessons are cited by tag (A1–E from expv1's register, N1–N5
new), as applied in [`LEARNINGS_CARRIED.md`](LEARNINGS_CARRIED.md). Model facts and gotchas
are in [`MODEL_NOTES.md`](MODEL_NOTES.md) (§1, §2.1–2.16). The reasons for choosing H3 are in
[`MiniMax_H3_Evaluation.md`](../../docs/documents/research/MiniMax_H3_Evaluation.md) (Eval §n).
The licence is set aside at the user's request and is not a gate here.

---

## 1. The question

> Can **MiniMax H3 FL2VA**, with **one LoRA** trained from the public pruned checkpoint on
> the client's alpha clips, generate **on-model, flat-2D** Pax and Polly clips in which the
> **named action or expression** plays out, from a **start frame it never trained on**, at
> 768×768 and 22 frames, while keeping its first frame, its cadence and its colours?

The bars that make this a yes are fixed in §9 before any data exists (B19). A second,
separate rule (§8) says whether a yes also makes H3 a **replacement candidate** for v8@550.

| Outcome | Meaning |
|---|---|
| **PASS** | H3 learns Pudgy at 0.92 s on unseen views. Read §8, then wait for Round 5 (§12) |
| **FAIL at V1** | The VAE cannot represent the art. Stop; no LoRA recovers it |
| **FAIL at V2/V11** | The style prior or the distillation wins. If the failure is wash-out, arm D (§6.6) is the one pre-declared second try |
| **PARTIAL** | It learns the labels but misses generation bars. Read §8 to see whether v8 fails the same way (data-limited, A13) or not (base-limited) |

## 2. One variable at a time

| Held fixed | Why |
|---|---|
| Base build: pruned bf16 FL2VA, as published | The question is about this base. LoRAs are build-specific (MODEL_NOTES 2.6) |
| 768×768 | H3's documented short edge, inside its area cap, square like our source. 9:16 needs a new shot-size design (later) |
| First-frame-only conditioning | Our scope ceiling is frame control. End-frame anchoring and Ref2VA are later questions |
| One LoRA, one recipe, plus **two pre-declared forks**: the shift pilots (§6.3) and arm D (§6.6) | Attributable results |
| Inference = DiffSynth's `MiniMaxH3Pipeline`, the code that defines the training layout | B8 |
| Caption template, frozen once written | B12 |

**Not in expv2:** 9:16, end-frame anchoring, Ref2VA, two characters, props, running,
clips longer than the delivered footage, audio, motion × expression in one prompt, full
fine-tuning, int8/FP8 bases.

---

## 3. Inputs

| Input | Source | Pin |
|---|---|---|
| DiT (trained and evaluated) | `Comfy-Org/MiniMax-H3: diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` (40.2 GB) | revision + MD5 into `eval/phase0/pins.json` |
| DiT (G0.2 parity only) | `Comfy-Org/MiniMax-H3: diffusion_models/minimax_h3_fl2va_bf16.safetensors`: same Comfy layout, AdaLN not pruned | revision + MD5; deleted after G0.2 |
| Text encoder | `MiniMax/MiniMax-H3: FL2VA/text_encoder/model*.safetensors` (Qwen3-VL-32B) | revision |
| Video VAE, audio VAE | `MiniMax/MiniMax-H3: FL2VA/video_vae/source/model.safetensors`, `FL2VA/audio_vae/model.safetensors` | revision |
| DeCFG adapter (arm D only) | `DiffSynth-Studio/MiniMax-H3-TrainingAdapter: model_for_comfy_dit.safetensors` | revision |
| Trainer and pipeline | DiffSynth-Studio at one commit that has `--training_cfg_scale` and `--audio_loss_weight` | SHA. Patches in `patches/` (§6.1) |
| G0.1 references | Wan2.1 VAE (the one Wan2.2-A14B uses; 8×) and Wan2.2-TI2V-5B VAE (16×), via the registry in `training_approach/scripts/vae_roundtrip.py` (copied, provenance noted) | repo ids as in that registry |
| Raw data | `Data/raw/iteration_3/`, `_4/`, `_5/` (Azure `pudgy/raw/`) | MD5-deduplicated (A8): 526 files → 240 unique |
| Hardware | 1× A100-80GB (no FP8), ≥ 128 GB host RAM (Qwen3-VL stage), ≥ 300 GB disk. Disk is not persistent (C1) | — |

---

## 4. Phase 0 — measure before training

No LoRA is trained until G0.1 and G0.2 pass. Each step writes a JSON report and a contact
sheet to `eval/phase0/`, mirrored to Azure.

### G0.1 — VAE gate (stop if it fails)

Round-trip real art through **H3's own VAE code path** (DiffSynth's loader and tiling
defaults), not diffusers. Run the **same clips** through the two reference VAEs.

- **Clips:** Pax and Polly × {`angry` at 39 frames, `waving` at 22, `walking` tiled to 22},
  composited on white and on mint, plus one constant flat-fill clip of 22 frames.
- **Sizes:** 768² (training) and 512² (for the record).
- **Metrics** (ltx25 gate's definitions, copied): PSNR, SSIM, edge-PSNR over positive
  gradients with edge coverage, flat-band, absolute shimmer. Added for H3:
  - **ground drift:** mean absolute per-channel error on ground pixels, /255;
  - **chunk-boundary dip:** mean luminance per frame, the largest step at frames 5, 22 and
    39 (chunk starts) minus the largest step elsewhere (Eval §4.4, #85).
- **By eye at 4× zoom:** beak outline, eye highlights, flipper outline, blush shapes.
- All comparisons on decoded arrays or PNG; no mp4 anywhere (MODEL_NOTES 2.11).

**Pass, all of:**
1. outlines and highlights intact at 768² by eye, judged against the Wan2.1 reference
   (B7: the eye decides; metrics are recorded);
2. ground drift ≤ 2/255 on every clip (the v5/v8 bar; HF #78's colour-shift risk);
3. chunk-boundary dip ≤ 1/255;
4. no grid or banding on the flat-fill clip.

**Fail → stop expv2 and report.** Informational: H3 vs the Wan2.2-5B 16× VAE, the only other
16× VAE we can test, says whether a failure is about the compression ratio or about H3's
decoder.

### G0.2 — Pipeline checks (one GPU session, no training beyond single steps)

| # | Check | Pass |
|---|---|---|
| a | **Parity, pruned vs full DiT:** one clip, cached embeddings, fixed noise, t ∈ {0.25, 0.5, 0.875}; compare velocity outputs | relative L2 ≤ 1% at every t. If not, stop: the build question must be answered first (MODEL_NOTES 2.6) |
| b | **Conditioning dump**, one training step vs one sampling step, first-frame-only: which rows carry the keyframe; their t and RoPE time; whether Qwen3-VL saw the keyframe; whether any end-frame rows exist; which latents enter the loss; the frame→latent map (22 → 7) | layouts identical; keyframe in **both** paths (2.8); **no** end-frame rows (2.4); frame 0's latent index recorded for Tier 1 |
| c | **Guidance-loss branch:** with `--training_cfg_scale 4`, which inputs the no-grad unconditional pass drops | recorded (2.16); no pass/fail |
| d | **Audio:** silent fallback active; audio loss term logged | audio term exactly 0 (2.9) |
| e | **Loader:** frames the loader reads, hashed, vs the 22/39/56 PNGs prep wrote | bit-identical, no padding or looping (2.3) |
| f | **Mixed lengths:** three stage-1 caches (f22, f39, f56) in one stage-2 run at batch 1 | first 20 steps print mixed shapes, no error. If it fails, **drop the 39/56 rungs** and train f22 only (pre-declared) |
| g | **Two-pass evaluation:** cached-embedding generation vs the one-pass pipeline on 2 clips | identical frames (PSNR ≥ 50 dB) |
| h | **Trainer flags:** `--help` and code for an LR scheduler, max steps, save semantics, resume | record; apply `patches/diffsynth-cosine-lr.patch` if no scheduler; confirm `patches/diffsynth-h3-train-shift.patch` changes the training sigma schedule (logged sigmas) |

### G0.3 — Base Tier-1 baseline

Base model, no LoRA, Tier 1 (§7.2) on the 23 held-out clips at all five calibration t
values, plus the 24 trained-angle `FRONT` clips used for calibration. This is the control's
likelihood: every "below the base" in §6 and §9 refers to it.

### G0.4 — Zero-shot probe = the control's generations

Base model, no LoRA, the exact inference settings of §7.1:
- `core` (§7.4): 13 labels × 2 characters × {FRONT, `QF2_R`} × 3 seeds, f22 = **156 clips**;
- `native`: angry, laughing, blushing at f39 and neutral at f56, `QF2_R`, 2 characters ×
  3 seeds = **24 clips**.

Scored by every Tier-2 metric, including the H3-specific ones: twos index, first-frame
fidelity, saturation, fill flatness. **Not a gate** (v4: a style LoRA fixed off-style
zero-shot output). It does answer three questions early, for the record: does the base
animate Pudgy on twos, drift to 3D, or lose frame 0?

### G0.5 — Source baseline

The real clips at 768², per label, on the frames each suite uses (first 22; 39/56 for
`native`):

| Measure | Definition |
|---|---|
| motion energy | mean absolute frame difference inside the character mask |
| frozen fraction | share of frame pairs below the hold threshold |
| hold threshold *h* | 25% of the label's median moving-pair difference (sources hold exactly; generations hold approximately) |
| **twos index** | share of frame pairs (2j, 2j+1) that are holds while (2j+1, 2j+2) are not, maximised over the phase |
| centroid travel, shadow width, head line | as v8 (the zoom reference) |
| ground flatness | pixel spread of the ground |
| late-frame identity | DINOv2 similarity of the character crop, last vs first frame |
| saturation | mean HSV saturation inside the character mask |
| fill flatness | luminance std inside the character mask minus the outline mask |

These are the denominators for §9 (B6).

### G0.6 — Memory and throughput

On the actual box: f56 at 768², batch 1, guidance loss on, gradient checkpointing. Record
peak GB, allocator retries and s/sample for f22, f39 and f56, then re-plan §10 (C5–C6).
The reported anchor is ≈ 53 GB for training the pruned bf16 build (R, DiffSynth #1624).

---

## 5. Corpus — `Data/processed/expv2_h3/`

Built by `prep/prep_expv2.py` (to write). It may copy decoding, gap, tear and seam helpers
from `finetune/wan/prep_v8.py` with provenance noted; it never imports them.

### 5.1 Format

- **768×768**, re-decoded from the 1080² source straight to target size (one resample; the
  flat ground is regenerated, not resampled).
- **24 fps, silent.** Lossless H.264 (`-qp 0`) in a pixel format DiffSynth's loader decodes
  bit-exactly (G0.2e). A PNG copy of every clip is kept for scoring.
- **Lengths on the 17n+5 grid:**
  - **f22, the floor:** every trained label, asserted (B11);
  - **f39:** sources whose **active** length reaches 39 (A5): angry, laughing, blushing,
    neutral at most;
  - **f56:** neutral only (exempt from the active-length rule, A5).
- **Cut before the first missing frame** (A6), measured from decoded timestamps (N1).
- **Two departures, both pre-declared:**
  - **one held frame** for 21-frame sources (happy, confused, two sitting): 21 → 22. Prep
    asserts that this is the only duplicate in each file;
  - **walking loop-tiled** 16 → 22 (frames 0–15, then 0–5), seam checked ≤ 1.5× the median
    step (A11).
- **Truncation captions** follow v8's rule exactly: a clip shorter than its source's usable
  length gets `"<label>, opening frames only"`.

### 5.2 Content (counted from raw on 2026-10-11; prep re-counts and must agree)

| Label | Kind | Usable at f22 | Held out (`QF2_R`) | Train sources | f22 clips | f39 sources (≤) | f56 sources |
|---|---|---|---|---|---|---|---|
| angry | expr | 18 | 2 | 16 | 64 | 16 | — |
| laughing | expr | 18 | 2 | 16 | 64 | 16 | — |
| blushing | expr | 14 | 2 | 12 | 48 | 11 | — |
| neutral | expr | 18 | 2 | 16 | 64 | 16 | 16 |
| crying | expr | 18 | 2 | 16 | 64 | — | — |
| surprised | expr | 18 | 2 | 16 | 64 | — | — |
| happy | expr | 14 (padded) | 2 | 12 | 48 | — | — |
| confused | expr | 14 (padded; 4 damaged dropped) | 2 | 12 | 48 | — | — |
| waving | motion | 18 | 2 | 16 | 64 | — | — |
| jumping | motion | 18 | 2 | 16 | 64 | — | — |
| walking | motion | 18 (tiled; 1 undecodable dropped) | 2 | 16 | 64 | — | — |
| sitting | motion | **7** (5 + 2 padded; 11 gapped dropped) | **1** (Polly) | **6** | 24 | — | — |
| running | motion | 0 (17 f) | start frames only | 0 | 0 | — | — |
| **Total** | | **193** | **23** | **170** | **680** | **≤ 59 → ≤ 236 clips** | **16 → 64 clips** |

Corpus ≈ **980 clips**, 24 performances (A1). Each clip is one (source, ground) pair; 4
grounds per source. Excluded and listed in `manifest.json` with reasons: the turnaround,
the 12 two-character clips (A14), running, every file that tears on all 24 decode attempts
(A7), and every file whose first missing frame falls below 21.

### 5.3 Holdout — fixed now (B9)

- **Angle `QF2_R`, every trained label, both characters: 23 clips.** Pax sitting `QF2_R`
  has 17 usable frames; it serves as a start frame only, as do both running `QF2_R` files.
  `QF2_R` has trained neighbours on both sides, so the test is generalisation to an unseen
  angle, not extrapolation. It is also v8's `core` start frame, which makes §8 possible.
- **Grounds:** lavender and sky-blue never trained (B15).
- The corpus is video-only, so nothing else can leak a held-out view.

### 5.4 Augmentation

| Applied | Rule |
|---|---|
| Alpha compositing onto 4 flat grounds | white, pastel blue, peach, mint |
| Shot-size ladder 1.00× / 0.75× / 0.55×, zoom-out only (A4) | one zoom per (clip, ground), round-robin; balance asserted against label, character, ground and angle (prep exits on > 34% deviation) |

Not applied: colour jitter, flips, random crops, rotation, temporal jitter, offset windows.

### 5.5 Captions

One template in H3's official I2V format (MODEL_NOTES §1). The **label clause is the only
part that changes between labels** (B12):

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] 2D cartoon animation in the Pudgy Penguins style, hand-drawn and animated on ones at 24 frames per second, thick clean black outlines, flat pastel colors, no shading. {Pax, a short round blue penguin | Polly, a short round pink penguin}, {action description}; {label}[, opening frames only]. {close-up | medium | wide} shot, eye level, {angle clause}. Fixed camera, locked-off shot, the character stays the same size. Plain flat {ground} background.

overall_soundscape: Silence.

non_diegetic_music: None.
```

- **Action descriptions and angle clauses:** copied from `prep_v8.py`'s tables with
  provenance, reworded only where H3's format needs it.
- **Colour:** "blue" in every Pax caption, "pink" in every Polly caption, never "pink" in a
  Pax caption ("rosy" for the blush). Asserted (A10).
- **Length:** ≤ 220 Qwen3-VL text tokens. Asserted.
- **"animated on ones at 24 frames per second"** is a deliberate counter to the on-twos prior
  (2.12). It is part of the frozen template, so it applies identically in training and
  evaluation.
- No negative prompt exists (2.1); nothing rewrites prompts (B14).

### 5.6 Assertions (prep exits non-zero)

- frame count exact (ffprobe on every file) and on the 17n+5 grid;
- duplicates only where declared (one held end frame; the walking tile seam);
- the f22 floor contains every trained label;
- shot-size balance;
- colour anchors;
- caption token count;
- holdout sources absent from training, by MD5 and by (character, label, angle);
- MD5 uniqueness;
- naming disagreements all resolved (A9);
- the per-label counts equal §5.2, or prep prints the difference and exits.

---

## 6. Recipe — `configs/` (to write)

### 6.1 Patches (written before Phase 2, checked in G0.2h)

| Patch | Why |
|---|---|
| `diffsynth-h3-train-shift.patch` | Exposes the training shift, hard-coded at 2.22 (2.5). Needed for the shift-12 pilot |
| `diffsynth-cosine-lr.patch` | Only if G0.2h finds no scheduler: warmup + cosine to 10% (D4) |

### 6.2 Arm A

| Knob | Value | Why |
|---|---|---|
| Base | pruned bf16 FL2VA, unquantised | A100 has no FP8; int8 changes the base |
| Conditioning | `--extra_inputs "input_audio,input_image"`, `--data_file_keys "video,input_audio"`, `--silent_on_missing_audio` | first frame only (2.4); silent audio (2.9) |
| Network | LoRA **rank 32** on `attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2`, AdaLN untouched: **155.1M** params | DiffSynth's H3 example targets; D5 |
| Distillation | `--training_cfg_scale 4` | 2.1; the documented value across trainers (Eval §6.1) |
| Audio | `--audio_loss_weight 0` | the video term only |
| Optimiser | AdamW (DiffSynth default), **LR 1e-4**, weight decay 1e-3 | DiffSynth's documented H3 value and two public H3 LoRAs; a fresh LoRA |
| Schedule | warmup 20, cosine → 10% | D4 |
| Batch | 1 × **grad-accum 8** | D4 |
| Training shift | the pilot winner (§6.3) | 2.5 |
| Budget | `--dataset_repeat 4`, 1 epoch = **3,920 samples = 490 optimizer steps** | §6.4 |
| Saving | every 50 optimizer steps, keep all | D3 |
| Precision | bf16, gradient checkpointing | — |
| Seed | 42 | — |

### 6.3 Pilots — the training shift, and the first sanity gate

Two runs, identical except the training shift: **S12** (12, the inference value) and **S2**
(2.22, DiffSynth's default). Each is **1 epoch, 980 samples, 122 optimizer steps**, with the
cosine schedule over its own length.

1. **Calibrate Tier 1** (B4): on the 24 trained-angle `FRONT` clips, compute each pilot's
   and the base's true-caption margin at t ∈ {0.25, 0.5, 0.75, 0.875, 0.9375} (multiples of
   1/1024, where the pruned build's 1,025-entry table is exact if it is indexed linearly in
   t; G0.2a checks). Fix the **two** t values where (pilot − base) margin, averaged over both
   pilots, is largest. Frozen for the rest of expv2.
2. **Sanity gate:** in at least one pilot, both kinds' held-out losses must be below the base
   on ≥ 75% of clips (expression ≥ 12/16, motion ≥ 5/7). If neither pilot does, **stop and
   re-check the conditioning (G0.2b) before touching LR**. If both are *above* the base on
   most clips, re-run the pilots at LR 5e-5.
3. **Choose the shift:** the pilot with the higher held-out Tier-1 margin (expression +
   motion). If the difference lies inside the paired bootstrap 95% CI, take **S12**: it
   matches inference, while 2.22 is an unexplained default.

The main run then starts **from zero** with the winning shift. DiffSynth's resume is
undocumented (2.15), so nothing depends on continuing a pilot.

### 6.4 Exposure budget

3,920 samples ÷ 24 performances ≈ **163 exposures per performance**:
- LTX-2.5 memorised at ~500;
- v8 plateaued at ~90 of ~185 (D6);
- public H3 LoRAs improved by ~2,000 samples and degraded by ~12,000 (Eval §6.1).

The best checkpoint is expected well inside the budget. If Tier 1 still improves at step 490,
report it, and extend only by a stated number of steps in a new run.

### 6.5 No mid-run gate breaks

expv1 stopped at steps 100 and 300 to measure. Here the pilots play that role, because the
GPU cannot hold training (≈ 53 GB) and a scoring DiT (40 GB) at once, and resuming is
unverified. During the main run, **Tier 0** runs on CPU on each saved checkpoint, and the
motion probe runs after training on every checkpoint.

### 6.6 Pre-declared fallback — arm D (distillation)

**Trigger, any of:**
- arm A's winner fails **V11**;
- arm A's winner fails **V2** with "washed out / foggy" recorded as the reason on ≥ 20% of
  `core` `QF2_R` clips;
- the motion probe shows character saturation < 0.9× source on ≥ 3 consecutive probed
  checkpoints before step 300.

**Then:** train **arm D**, identical to arm A except `--training_cfg_scale 1` and the DeCFG
adapter `--preset_lora_path …/model_for_comfy_dit.safetensors --preset_lora_model dit`
during training only (never at inference; never stacked with the guidance loss, 2.16). The
verdict (§9) is computed for each arm. expv2 passes if either does; both are reported.

---

## 7. Evaluation

### 7.1 Generation settings (every generated clip, base and LoRA)

- DiffSynth `MiniMaxH3Pipeline` on the pruned build, LoRA loaded at strength 1.0.
- `cfg_scale` 1.0, `flow_shift` 12, `audio_flow_shift` 3, 50 steps.
- 768×768, `num_frames` 22 (39/56 for `native`), seeds {42, 43, 44}.
- Start frames pre-sized to **exactly 768×768**; output size asserted.
- **Two-pass** (2.7, C4): Qwen3-VL encodes every (caption, start frame) pair once, then the
  DiT loads once per checkpoint.
- Frames saved as PNG; audio discarded; mp4 for viewing only (2.11).

### 7.2 Tiers

| Tier | What | When |
|---|---|---|
| **0** | LoRA norm, consecutive-update cosine (B18), on CPU | every checkpoint |
| **1** | **Diffusion classifier** on the 23 held-out clips at f22: is the true caption the best denoiser among the captions of its kind? Expression clips (16) against 8 expression captions (chance 0.125); motion clips (7) against 4 motion captions (chance 0.25). Loss on every latent except those covering frame 0 (index from G0.2b), change-masked (\|z[f] − z[0]\|), at the two calibrated t values, k = 2 noise draws identical across checkpoints. Only the label clause differs between candidates; truncation suffixes are held at the true clip's value. Paired comparisons, bootstrap CIs | base (G0.3), pilots, every checkpoint |
| **2** | Generated suites, 3 seeds (B2), scored with CLIP/DINOv2 1-NN against the real source clips at the same frame window (B3), plus the G0.5 measures and first-frame fidelity | base (G0.4), the winner, arm D's winner if trained |

**Distilled-model note:** the conditional prediction of a CFG-distilled model approximates the
*guided* velocity. Tier 1 compares captions under shared noise, so ranking still works, but
its absolute margins are not comparable with v8's.

### 7.3 Winner rule (Tier 1)

1. Among checkpoints whose held-out losses beat the base on ≥ 15/16 expression and
   ≥ 6/7 motion clips, take the highest expression + motion margin.
2. If several are within 0.001, take the **earliest** (less memorisation, and less
   distillation drift).
3. **Motion veto:** if that checkpoint's probed motion energy is < 0.8× the best probed
   checkpoint's, take the best-motion checkpoint within 0.002 of the top score.

### 7.4 Suites

| Suite | Content | Clips |
|---|---|---|
| `core` | 13 labels × 2 characters × {FRONT, `QF2_R`} × 3 seeds at f22. Running is an untrained label, reported apart from the bars | 156 |
| `native` | angry, laughing, blushing at f39, neutral at f56; `QF2_R`; 2 characters × 3 seeds. Does the arc complete where the footage allows? | 24 |
| `gb` | angry, waving, jumping × 2 characters × {lavender, sky-blue} × 3 seeds | 36 |
| motion probe | 4 trained motions × 2 characters × `QF2_R` × seed 42, on every checkpoint | 8 per checkpoint |

The base runs `core` and `native` (G0.4). Total generation ≈ **490 clips** with arm A only.

---

## 8. Head-to-head with v8@550 — the replacement read

v8@550's `core` clips already exist: Azure `pudgy/v8/eval/suites/core/v8@550/`, 52 clips,
seed 42, FRONT + `QF2_R`, f13 motion / f21 expression, 1024². No Wan GPU time is needed.

- **Comparable set:** the 12 labels both models trained × 2 characters × 2 angles, seed 42 =
  **48 paired clips**. The H3 winner's clips are trimmed to the first 13 (motion) or 21
  (expression) frames, and v8's are downscaled to 768². Both are scored by expv2's scorer
  against the same real-clip bank at the same windows.
- **Measures:** V2 rubric pass rate; V4 1-NN correctness per clip; V5 motion ratio; V6 drift.
- **Statistic:** per clip, correct/incorrect under 1-NN for each model. Count *b* = H3-only
  correct and *c* = v8-only correct, and report the exact two-sided binomial *p* on b vs c
  (McNemar).

**Rule, fixed now:**

| Result | Label |
|---|---|
| H3 PASSES §9, b > c, and H3's rubric pass rate ≥ v8@550's | **replacement candidate** (*p* < 0.05: **clear** candidate) |
| H3 PASSES §9, but b ≤ c or its rubric rate is lower | **not a replacement on today's data**; H3 stays a candidate only for its long-clip ability, tested in §12 |
| H3 does not pass | §8 is read only to separate data-limited from base-limited failures (A13) |

**Stated confounds:** different resolutions and prompt wording; v8 saw `QF2_R` through its v7
parent; trimming H3's 22-frame clips to 13 frames; n = 1 seed per model. Optional, and
outside the budget: two more v8@550 seeds (≈ 104 Wan clips ≈ 11 A100-h) to make n = 3.

**Never a replacement before §12:** H3 does not cover running on today's data. The actual
replacement decision is made on the Round 5 re-run.

---

## 9. Pre-registered verdict

All bars are on the **winner**, from **unseen `QF2_R` start frames**, over 3 seeds, at f22,
excluding running.

| # | Bar | Pass when |
|---|---|---|
| V1 | VAE | G0.1 passed |
| V2 | Style and identity | ≥ 90% of `core` `QF2_R` clips pass the manual rubric (flat fills, intact black outlines, no 3D shading, on-model shape); colour correct in **100%** |
| V3 | Tier 1 | held-out accuracy ≥ 0.75 expression (≥ 12/16; chance 0.125) and ≥ 0.70 motion (≥ 5/7; chance 0.25); losses below the base on ≥ 15/16 expression and ≥ 6/7 motion clips |
| V4 | Controllability | CLIP 1-NN label accuracy ≥ 0.45 expression and ≥ 0.40 motion |
| V5 | Motion | per trained motion: median motion energy ≥ 0.8× source, frozen fraction ≤ source + 15 points |
| V6 | Camera and ground | \|zoom\| ≤ 3%; corner drift ≤ 5/255 on unseen grounds (`gb`); ground spread ≤ 2× the source's |
| V7 | Late-frame identity | last-vs-first DINOv2 similarity of the character ≥ 0.9× the source's |
| V8 | Memorisation | FRONT minus `QF2_R` 1-NN accuracy ≤ 0.15 |
| **V9** | **First-frame fidelity** (2.2) | on ≥ 95% of `core` `QF2_R` clips: generated frame 0 vs the supplied start frame, PSNR ≥ (G0.1's mean 768² round-trip PSNR − 3 dB) and character-crop DINOv2 similarity ≥ 0.95 |
| **V10** | **Cadence** (2.12) | per-label median twos index ≤ the source's + 0.10, for ≥ 10 of 12 labels |
| **V11** | **Colour and flatness** (2.1, 2.13) | median character saturation ≥ 0.9× source; median fill flatness (luminance std) ≤ 1.5× source |

| Outcome | Meaning | Next |
|---|---|---|
| **PASS** (V1–V11, arm A or D) | H3 learns Pudgy at 0.92 s and stays controllable on unseen views | §8 label; §12 when Round 5 lands |
| **PARTIAL** (V1–V3 pass; some of V4–V11 fail) | it learns the labels, but generation misses bars | §8 read; decide whether Round 5 data is likely to fix it |
| **FAIL at V1, V2 or V11** (after arm D if triggered) | the VAE, the style prior or the distillation does not suit this art | stop; record why |

V9–V11 are new for H3 and set before any data. If the base (G0.4) already misses one by a
wide margin, that is recorded; the bar does not move.

---

## 10. Cost — estimates, to be replaced by G0.6

Anchor: 12.5 s/step at 768²×73 frames on 1× A100 with guidance loss (pablodawson card, R),
scaled by sequence length (I). The sequence at 768² is 576 tokens per latent frame, plus 576
start-frame rows, ≈ 200 text tokens and ~3% audio rows: about 5.3k tokens at f22, 8.2k at
f39 and 11.1k at f56. That gives ≈ 4.7 / 7.3 / 9.9 s per sample, and ≈ 5.7 s on the corpus mix.

| Item | Estimate |
|---|---|
| G0.1 VAE gate (VAE only, ~5 GB) | ≤ 1 h |
| G0.2 checks (incl. the full-DiT parity load) | ≈ 2 h |
| G0.6 memory probe | ≈ 0.5 h |
| Stage-1 cache: Qwen3-VL over ~980 clips + ~400 evaluation pairs, VAE encode | ≈ 1.5–2 h |
| Pilots: 2 × 980 samples × 5.7 s | ≈ 3.1 h |
| Main: 3,920 samples × 5.7 s | ≈ 6.2 h |
| Tier 1: base + 2 pilots + 10 checkpoints, ~15 min each, + calibration | ≈ 4–5 h |
| Generation: ≈ 490 clips × 45–90 s (≈ 1.7 s/step × 50 steps at f22, from HF #35's A100 timing scaled by tokens) | ≈ 6–12 h |
| **Total, arm A** | **≈ 25–32 A100-h** |
| Arm D, if triggered: train + Tier 1 + Tier 2 | + 11–13 h |

For scale: v8's training alone took 57.5 h, and expv1's plan budgets 45–60 h.

---

## 11. Risks and open questions

| Risk | Handling |
|---|---|
| The VAE dissolves outlines or tints flat grounds | G0.1 stops the experiment before any training spend |
| Distillation drift (wash-out, lost adherence) | guidance loss; early checkpoints; V11; arm D |
| The trainer silently trains first + last frame, or pads a short clip | G0.2b, G0.2e |
| The shift is wrong for our data | pilots (§6.3) |
| 22 frames is 5.6× below H3's trained floor (124 frames): an untested regime | Phase 0's base probe shows it at no training cost; `native` measures 39/56 where possible |
| Sitting has 6 training sources; confused and happy 12 | read as the watch cells; no upweighting (A12) |
| On-twos and 3D priors | caption counter-wording; V10, V2, V11 |
| DiffSynth is young; flags or behaviour change between commits | one pinned SHA; G0.2 on that SHA |
| 1080 → 768 thins outlines to 0.71× | G0.1 judged at 768² by eye |
| Box loss | Azure mirror on a timer from the first hour (C1) |

**Open, not answered by expv2:** running; 9:16; end-frame anchoring; clips longer than 2.5 s
of footage; two characters; audio.

---

## 12. Phase 4 — Round 5 re-run (trigger only; written as an amendment)

**Trigger:** Round 5 delivers clips of ≥ 39 frames for **every** label, including running
with ≥ 3 cycles (the ask: `docs/documents/Client_Data_Request_Round5.md`).

**Then:** a `PLAN_round5.md` amendment adds running and builds 39/56/73-frame rungs for every
label. It re-runs Phase 0's G0.4 on the new holdout, re-uses this plan's bars, and makes the
**replacement decision** against v8@550 on the shared start frames. It only starts if expv2
reached PASS or PARTIAL.

---

## 13. Runbook (scripts marked † are still to write)

```bash
# 0. Environment
git clone https://github.com/modelscope/DiffSynth-Studio /workspace/DiffSynth-Studio
git -C /workspace/DiffSynth-Studio checkout <SHA>             # record in eval/phase0/pins.json
pip install -e /workspace/DiffSynth-Studio
git -C /workspace/DiffSynth-Studio apply expv2-minimax-h3/patches/diffsynth-h3-train-shift.patch   # †
#   download the inputs of section 3; write revisions + MD5s to eval/phase0/pins.json

# 1. Phase 0
python expv2-minimax-h3/eval/vae_gate.py                      # † G0.1 -> stop on fail
python expv2-minimax-h3/eval/pipeline_checks.py               # † G0.2a-h (applies the cosine patch if needed)
python expv2-minimax-h3/eval/source_baseline.py               # † G0.5 (CPU)
python expv2-minimax-h3/eval/dcls.py --ckpt base              # † G0.3
python expv2-minimax-h3/eval/generate.py --ckpt base --suites core,native   # † G0.4
python expv2-minimax-h3/eval/mem_probe.py --frames 22,39,56   # † G0.6

# 2. Corpus
python expv2-minimax-h3/prep/prep_expv2.py --dry-run          # † plan only; prints the section 5.2 table
python expv2-minimax-h3/prep/prep_expv2.py --workers 16       # † -> Data/processed/expv2_h3/{clips,png,metadata_f22.csv,metadata_f39.csv,metadata_f56.csv}

CACHE_MODELS="MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors"
LORA="--lora_base_model dit --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 --lora_rank 32"
cd /workspace/DiffSynth-Studio
for F in 22 39 56; do                                         # stage 1, once per bucket
  accelerate launch examples/minimax_h3/model_training/train.py \
    --dataset_base_path <repo>/Data/processed/expv2_h3 \
    --dataset_metadata_path <repo>/Data/processed/expv2_h3/metadata_f${F}.csv \
    --data_file_keys "video,input_audio" --extra_inputs "input_audio,input_image" \
    --silent_on_missing_audio --height 768 --width 768 --num_frames $F --dataset_repeat 1 \
    --model_id_with_origin_paths "$CACHE_MODELS" --remove_prefix_in_ckpt "pipe.dit." \
    --output_path /workspace/expv2_cache/f${F} $LORA --use_gradient_checkpointing \
    --task "sft:data_process"
done
#   merge the three caches as G0.2f established

# 3. Pilots, then main (the flags configs/train_expv2.sh† will wrap)
DIT="Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
for S in 12 2.22; do
  accelerate launch examples/minimax_h3/model_training/train.py \
    --dataset_base_path /workspace/expv2_cache/all --dataset_repeat 1 --num_epochs 1 \
    --model_id_with_origin_paths "$DIT" --remove_prefix_in_ckpt "pipe.dit." \
    --learning_rate 1e-4 --weight_decay 1e-3 --gradient_accumulation_steps 8 \
    --training_cfg_scale 4 --audio_loss_weight 0 \
    --train_flow_shift $S \
    $LORA --use_gradient_checkpointing --find_unused_parameters --save_steps 50 \
    --output_path /workspace/expv2_output/pilot_s$S --task "sft:train"
done
#   --train_flow_shift comes from the shift patch; scheduler flags from the cosine patch if applied
python expv2-minimax-h3/eval/dcls.py --calibrate --ckpt pilot_s12,pilot_s2.22   # † section 6.3 -> shift
#   main: same command, --train_flow_shift <winner> --dataset_repeat 4, output .../main

# 4. Tier 1 on every checkpoint, motion probe, winner rule (section 7.3)
# 5. Tier 2 on the winner (and the base from G0.4); section 8 head-to-head; section 9 verdict -> docs/REPORT.md
# 6. Arm D only if section 6.6 triggers: main command with --training_cfg_scale 1 \
#      --preset_lora_path <adapter>/model_for_comfy_dit.safetensors --preset_lora_model dit
```
