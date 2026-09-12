# LTX-2.5 run 1 — findings

**Status:** measured · **Base:** LTX-2.5-22B dev transformer · **Run:** 2026-09-11/12
**Box:** 1 × A100 80GB (Ampere, SM 8.0), driver CUDA 13.2, torch 2.13.0+cu132
**Artifacts:** Azure `pudgytraining` / container `pudgy` / prefix `ltx25/run1/`

> **One line.** The base is fit for this art and the pipeline works end to end, but the run
> proved its own central assumption wrong: **one rank-32 LoRA cannot carry both motion and
> expression.** Expressions hold flat across 3,000 steps while motion degrades monotonically
> to roughly half of source, and the cause is measured — the corpus contains **24 distinct
> performances**, each seen ~500 times.

Everything here was measured on this box. Nothing is inherited from the v1–v7 Wan line, and
nothing here shares code, data, config or recipe with it.

---

## 1. Gate zero — PASSED, and the metrics lied

LTX-2.5 can represent flat 2D art with thin black outlines. Both 2.5 video VAEs round-tripped
Pax and Polly at 1024²; the **diffusion VAE** was selected.

| decoder | psnr | ssim | edge_psnr | edge_cov | flat_band | shimmer_abs |
|---|---|---|---|---|---|---|
| diffusion | 37.69 | 0.9955 | 20.31 | 0.014 | 0.087 | 0.099 |
| conv | **40.30** | **0.9957** | **22.88** | 0.014 | 0.098 | 0.134 |

⚠️ **The decoder was chosen by eye-check, OVERRULING every metric.** conv scores higher on
all of them and **visibly dissolves outlines**: at 4× zoom Polly's black beak outline degrades
into an orange blur and the eye highlights smear, while diffusion holds them. Reading the
table alone would have selected the decoder that destroys the one feature this art depends on.
This is PIPELINE.md §8.3's *"no metric sees identity"* reproducing at the VAE stage, one layer
below where it was predicted. Evidence: `ltx25/eval/vae_gate_frames/outline_zoom*.png`.

**Two secondary results, both favourable:**

- **Temporal shimmer is a non-issue.** 0.099 grey levels of flicker added on pixels the source
  holds perfectly still. LTX-2 [#223](https://github.com/Lightricks/LTX-2/issues/223) — the
  VAE-level flicker the maintainer says 2.5's DiffVAE "does not fully remove" — **does not
  bite this corpus.** PIPELINE.md §10.2 called this the single largest technical risk to this
  art style; it is retired.
- **No banding or flat-region grid artefact** (`flat_band` 0.087).

### 1.1 Gate zero could not be answered by the existing tool

`training_approach/scripts/vae_roundtrip.py` loads VAEs the diffusers way
(`from_pretrained(repo, subfolder="vae")`). **LTX-2.5 ships a split pack**: bare
`.safetensors` whose config lives in file metadata, loaded through
`ltx_core.model.video_vae.model_configurator`. There is no diffusers layout to point at, so
adding a registry entry — which both LTX-2.5 planning docs instructed — **cannot work**.
`ltx25/eval/vae_gate_ltx25.py` is the purpose-built replacement.

### 1.2 Both art-specific metrics degenerated before they worked

Stated because the numbers above are only trustworthy after two fixes:

- **Edge mask selected the whole frame.** On this art **92.8% of pixels have exactly zero
  gradient** (flat fills on flat synthetic ground), so `np.percentile(mags, 90)` returns 0.0
  and `mags >= 0` is every pixel. `edge_psnr` silently equalled global `psnr` to two decimals.
  Fixed: percentiles over positive gradients only, and `edge_cov` is now reported so the
  degeneration cannot hide again.
- **Shimmer as a ratio divided by ~0.** The source holds those pixels perfectly still, so the
  quotient exploded (15–41 on real clips, 1.9e6 on a synthetic check). Fixed: reported as an
  absolute grey-level difference, which is the honest question anyway.

---

## 2. Corpus — built, invariants verified

900 clips + 211 PNG stills → **1,018 train / 93 holdout** (angle `SIDE_R`), 12 labels,
1024×1024 @ 24 fps, silent. Both asserted invariants passed:

- **length balance OK — all 12 labels present at the 17f floor.** The anti-confound guarantee
  holds: frame count cannot predict the label.
- **shot-size balance OK** across label / character / ground / angle.

Real bucket string: **`1024x1024x1;1024x1024x17;1024x1024x25;1024x1024x33`**. This matches
*neither* of the two hand-written bucket strings in the planning docs (no `x57`, because
`NEUTRAL` trims to 27 active frames), which is why every downstream script reads
`manifest.json` instead of a literal.

Documented source defects reproduced exactly: `PAX_MOTION_WALKING_QF1_R` decodes 11f of a
claimed 16; two `POLLY…CONFUSED` files truncated at 18f against the cell's 21f.

### 2.1 ⚠️ The viewpoint holdout leaked

`prep_ltx25.py` split **clips** by angle but appended **stills unfiltered**:

```python
# Stills join the training split only. The holdout stays video-only ...
train += [dict(r, file=r["file"]) for r in still_records]
```

So **24 `SIDE_R` stills were in the training split** while `SIDE_R` clips were held out. The
model had seen the held-out viewpoint's **appearance**; only its **motion** was genuinely
novel. That silently weakens the one generalisation axis this corpus can support (PIPELINE.md
§5: a viewpoint holdout and a start-frame holdout, and no novel-action holdout at all).

- **Magnitude:** 24 of 1,018 samples = **2.4%**.
- **Honest claim for run 1:** *"novel-viewpoint motion"*, **not** *"unseen viewpoint"*.
- **Fixed in prep** for the next run (stills now filtered by `HOLDOUT_ANGLE`, with a count
  printed). Run 1 was **not** retrained.

---

## 3. Training

| | |
|---|---|
| Duration | **8 h 21 m** (501.1 min), 3,000/3,000 steps at 0.10 steps/s |
| Final loss | 0.0213 (lr 1e-4 → 1e-5, linear) |
| Peak GPU memory | **53.97 GB** of 80 — the 3,072-token geometry left 26 GB unused |
| Trainable params | 163,577,856 (rank 32 / alpha 32, attn+FFN) |
| Checkpoints | **12 retained**, 312 MB each, steps 250 → 3000 |

`keep_last_n: -1` earned its place: the default of 3 would have deleted steps 250–2500, and
**the best checkpoints turned out to be 500 and 1000.**

Attention backend, as the trainer reported it:
`SDPA[CUDNN_ATTENTION>FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH]` — flash in the chain, exactly
as predicted for Ampere.

---

## 4. ⭐ The headline result: motion degrades, expressions do not

All 12 checkpoints rendered through `ltx-pipelines` (never the trainer's own validation
sampler — see §6.1), scored against the **measured source baseline**. `frozen% / motion as
×source`:

| ckpt | WAVING | WALKING | ANGRY | HAPPY |
|---|---|---|---|---|
| 250 | 28% / **1.48×** | 12% / 1.93× | 25% / 1.26× | 25% / 9.98× |
| **500** | **12% / 1.10×** | 25% / 1.47× | 38% / 1.25× | 41% / 3.27× |
| 750 | 31% / 0.75× | 0% / 3.57× | 34% / 1.40× | 47% / 3.09× |
| **1000** | **16% / 1.62×** | 19% / 1.60× | 34% / 1.68× | 69% / 1.23× |
| 1250 | 53% / 0.59× | 38% / 1.30× | 28% / 1.49× | 44% / 2.10× |
| 1500 | 62% / **0.23×** | 31% / 1.24× | 28% / 1.05× | 41% / 3.74× |
| 1750 | 50% / 0.60× | 31% / 1.60× | 19% / 1.87× | 50% / 2.88× |
| 2000 | 41% / 0.60× | 31% / 1.43× | 28% / 1.43× | 59% / 2.01× |
| 2250 | 50% / 0.52× | 38% / 1.35× | 25% / 1.53× | 78% / 1.74× |
| 2500 | 50% / 0.52× | 38% / 1.38× | 28% / 1.46× | 53% / 2.33× |
| 2750 | 56% / 0.54× | 38% / 1.20× | 38% / 1.32× | 69% / 2.01× |
| 3000 | 53% / 0.54× | 38% / 1.29× | 31% / 1.31× | 50% / 2.28× |

*(source frozen% medians: WAVING 24.0 · WALKING 0.0 · ANGRY 46.2 · HAPPY 40.0)*

1. **`waving` decays monotonically then plateaus at ~0.52–0.54× source** from step 2250,
   frozen frames climbing 12% → 56%. Eleven checkpoints in one direction: not noise.
   Confirmed by eye — **`waving` never raises a flipper at any checkpoint**.
2. **`ANGRY` is flat and healthy across the entire run** (1.05–1.87×, frozen 19–38% against a
   source median of 46%). Expressions do not care how long you train; motion does.
3. **`HAPPY` at step 250 is 9.98× source** — the earliest checkpoint over-moves and is
   unstable, settling by 500–1000. So the usable window is genuinely **500–1000**, not earlier.

**Caveat, stated plainly:** this sweep is **n = 1–2 per label**. The trend across eleven
checkpoints is consistent enough to act on directionally; the 8-seed deep eval on steps 500
and 1000 is what converts it from signal to finding.

### 4.1 Root cause — measured, and not what was first suspected

| | |
|---|---|
| Training samples | 1,018 |
| Distinct source **files** | 203 |
| **Distinct performances** | **24** — one take per (character, label) cell, shot from 9 cameras |
| Samples seen | 12,000 (3,000 steps × accum 4 = 11.8 epochs) |
| **Exposures per performance** | **~500×** |

The nine camera angles are **one take from nine cameras**, so the corpus holds **24 motion
patterns, not 203**. At rank 32 with ~500 exposures each, the model does exactly what the
vendor's mode selector warns of — *"high ranks just memorise frame content"* — and when
conditioned on an unseen profile start frame it cannot replay the memorised trajectory, so it
falls back to near-static. Expressions survive because they are a bigger, slower, more legible
signal that generalises from fewer distinct examples.

**A hypothesis that was checked and is WRONG:** stills are not the cause. By sample count they
look dominant (211/1,018 = 20.7%), but **by token mass they are only 6.5%**, and total
near-static signal is **19%**, not the third that sample counting suggests. Recorded because it
is the obvious wrong turn.

### 4.2 This contradicts assumption A2

PIPELINE.md §0 assumption **A2** was: *"one LoRA, rank 32, not two — v7's split exists because
Wan2.2 is a mixture of experts; LTX has no expert split, so that argument does not transfer."*

The reasoning about experts is still correct. **The conclusion is not.** LTX's own guidance
splits rank by *signal type* (motion 8–16, character/style 32–64) for a reason independent of
expert architecture, and this run measured the consequence of ignoring it. **A2 is retired.**

---

## 5. Other defects visible in the output

1. **Backgrounds render as soft radial gradients, not the flat grounds the corpus was built
   from** — at every checkpoint. The ground is a named caption slot, so this is a style
   adherence failure, not randomness.
2. **Identity drifts in late frames.** Pax's face distorts, Polly's beak reshapes toward the
   end of a clip. Exactly what LTX-2 [#255](https://github.com/Lightricks/LTX-2/issues/255)
   predicts: *"`--image` is first-frame / keyframe conditioning, **not an identity encoder**"*.
   Whatever identity survives to the last frame comes from the LoRA.
3. **One clip changed background colour mid-shot** (rust → sage green).
4. **Colour identity is correct throughout** — Pax blue, Polly pink, at every checkpoint. v4's
   single worst failure (the same conditioning rendering Polly **blue**) is **fixed**, and the
   fix is the two words of colour in every caption. Cheapest high-value intervention in the
   programme, reconfirmed on a new base.

---

## 6. Tooling findings worth keeping

### 6.1 Do not pick checkpoints from the trainer's validation samples

LTX-2 [#283](https://github.com/Lightricks/LTX-2/issues/283) is open and acknowledged: the
validation runner diverges from training on the sigma schedule, never sets `keyframes_mask`
(training omits it, inference always sets it — **and first_frame conditioning is our exact
mode**), and uses the opposite token order. All 84 + 368 evaluation renders go through
`ltx_pipelines`. The trainer's own 26 samples are shipped in the Azure bundle labelled as
sanity checks only.

### 6.2 Inference throughput: 105 s → 61 s per generation

One CLI call costs **105 s**, of which only **74 s** is the 30-step denoising loop; the other
~31 s loads the 39 GB transformer, text encoder and VAE. `sweep_ltx25.py` builds the pipeline
**once per checkpoint** and loops (prompt × seed × start-frame) inside, giving **60.7 s** per
generation — ~40% off any sweep measured in hours.

### 6.3 `ltx-pipelines` API gotchas (each cost a cycle)

| Symptom | Cause |
|---|---|
| `ModelPaths.audio_vae_path is required but missing` | `--audio-vae-path` is mandatory **even for silent output** |
| `ModelPaths.__init__() missing 3 required positional arguments` | frozen dataclass, no defaults: `mode`, `duration_head_path`, `embeddings_weight_paths` must all be named; the last is `(transformer, text_encoder)` on a split pack |
| `'tuple' object has no attribute 'strength'` | `loras=` wants `LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP)`, not a bare tuple |
| `'NoneType' object has no attribute 'strip'` | `negative_prompt=""`, not `None` |
| `Inference tensors cannot be saved for backward` | the pipeline must be **constructed and called inside the same `inference_mode`**; the CLI decorates all of `main()` |

### 6.4 Three prep bugs, all fixed

1. **Deadlock.** 16 ffmpeg decoders wedged 28 min at 0% CPU on the corrupt ProRes files; none
   of five subprocess calls passed a `timeout`, so a hung decode blocked forever instead of
   failing. A timeout now feeds the existing 24-attempt retry loop. It stalled at 851/924 with
   `manifest.json` unwritten — i.e. it blocked the whole pipeline permanently.
2. **x264 thread explosion.** x264 defaults to one thread per core; 32 workers requested ~1024
   threads, `pthread_create` failed and ffmpeg died `SIGABRT` at clip 898/924. Bounded to
   `-threads 2`. Per-clip threading buys nothing on 17–33 frame clips.
3. **Stills as single-frame mp4 killed preprocessing.** The trainer dispatches on **file
   extension** (`process_videos.py:159`): only `.png/.jpg/.jpeg` reach `_preprocess_image`,
   which handles F=1. A single-frame mp4 goes to `_preprocess_video`, whose `_resize_and_crop`
   *"returns [C, H, W] for single-frame input (squeeze removes dim 0)"*, so
   `for frame in frames_resized` iterates **channels** and hands torchvision a 2-D tensor.
   Stills are now **lossless PNG** — which also dodges yuv420p chroma subsampling on thin
   outlines.

### 6.5 A confound in the evaluation itself, fixed before the deep eval

The sweep first keyed captions on `(character, label)` against `dataset.json`, so a **`SIDE_R`
profile** start frame was paired with a caption from some *trained* angle — *"front view"*,
*"turned strongly to its left"*. The model dutifully rotated the character to obey the prompt,
which reads in the output as identity instability but is **correct behaviour on a prompt that
contradicted the image**. Captions now come from `holdout.json` keyed to the exact source clip.
The coarse sweep was re-run from scratch rather than reinterpreted.

### 6.6 LTX-2.5's custom kernels buy nothing on an A100

Not on the HF Hub — Lightricks publishes models and IC-LoRAs there, no kernel repos, and the
repo has no `kernels`-hub integration. They ship in-repo as `packages/ltx-kernels`, and almost
all are hardware-gated off on SM 8.0:

| Kernel | Requires | A100 |
|---|---|---|
| `blockwise_cpp` — blockwise FP8 GEMM | SM89 Ada / SM90 Hopper | ❌ |
| `nvfp4_cpp` — NVFP4 quant + GEMM | Blackwell SM ≥ 10.0 | ❌ |
| `vae` CuTe DSL (`na_attn_dsl`, `block_fna_dsl`) | datacenter Blackwell sm_100/103 or Jetson Thor — needs tcgen05 UMMA **and** TMEM, which *"consumer Blackwell, Hopper and Ada do not have"* | ❌ |
| `all2all_cpp` — tensor-parallel comms | multi-GPU | ❌ (1 card) |

This is a **second, independent** reason for `quantization: null`, beyond the fp8-cast LoRA
crash in [#296](https://github.com/Lightricks/LTX-2/issues/296) /
[#253](https://github.com/Lightricks/LTX-2/issues/253): **the FP8 kernel does not even build
for SM 8.0.** What does apply: **natten** (real extra; pins torch 2.13.0+cu132, compatible with
this CUDA 13.2 driver) and torch SDPA's flash path, which is already the default. Nothing to
install, and no kernel-level speedup available without different hardware.

---

## 7. Recommendations for run 2

### 7.1 Motion consistency, by expected value

1. **Split the LoRA and split the rank.** Motion at **8–16**, expression/appearance at
   **32–64**, `alpha == rank`. Evidence-backed by §4, not just vendor guidance. Lower rank has
   less capacity to memorise frames.
2. **Stop at ~500–1000 steps, not 3,000.** Motion peaks early and degrades monotonically. Also
   cuts training from 8.4 h to ~1.5–2.8 h — better *and* cheaper.
3. **Cut the angle multiplication.** Nine angles of one take inflates apparent data 9× while
   adding zero new motion patterns and tripling exposure count. 3–4 angles per cell costs no
   information.
4. **Cut the shot × ground multiplication** (~4.4 samples per source today). With #3 that is
   ~40 exposures per performance instead of 500.
5. **Lower `first_frame` probability below 0.5**, so more steps are text-only and the model
   learns motion priors instead of frame completion.
6. **Fix the background-gradient failure** — worth one ablation on whether the flat-ground
   caption slot is being learned at all.

### 7.2 Training throughput

Baseline **501 min, 0.10 steps/s, peak 53.97 GB of 80**.

1. **Test disabling gradient checkpointing.** 26 GB of headroom exists. The v4-era "~78 GB
   needed" warning was measured at 768×1344×49 = **7,056 tokens**; this geometry is
   1024²×17 = **3,072 tokens**, under half. Typically ~30%. **Measure it, don't assume it.**
2. **Separate the stills run to unlock `batch_size > 1`.** `batch_size: 1` is forced *only*
   because the dataset mixes F=1 stills with video. Splitting appearance from motion serves
   §7.1 #1 anyway, since they want different ranks.
3. **Fewer steps** (§7.1 #2).

### 7.3 The data ask this run justifies

68.3% of clips sit at the **17-frame floor = 0.71 s**, and `walking` had to be **cycle-wrapped**
to reach the floor at all. The two asks already standing in the docs are now evidence-backed
rather than speculative:

- **Multiple distinct performances per action** — not more camera angles. This is the direct
  fix for §4.1 and the highest-value item in the programme.
- **Motion takes of ≥ 2 s.** A 0.71 s clip cannot contain a walk cycle.

### 7.4 Still unanswered by this corpus, regardless of run 2 config

1. **No novel-action generalisation claim** — every label is trained; there is no held-out action.
2. **No rear views exist** for either character, so any away-facing generation is unconstrained.
3. **Two-character shots are untested** — the corpus is single-character throughout.

---

## 8. Artifact index

| Artifact | Location |
|---|---|
| Sample output + evidence bundle | Azure `pudgytraining` / `pudgy` / `ltx25/run1/` (README at prefix root) |
| All 12 checkpoints, 7 cells each | `ltx25/run1/sweep_coarse/step*/` |
| 8-seed deep eval (steps 500, 1000) | `ltx25/run1/deep/step*/` |
| Gate-zero verdict + outline zooms | `ltx25/run1/gate_zero/`, `ltx25/eval/vae_gate.json` |
| Trained LoRAs (not uploaded, 312 MB × 12) | `/workspace/ltx25_output/pudgy_i2v/checkpoints/` |
| Corpus | `Data/processed/ltx25_experiment/` (900 clips + 211 stills, `manifest.json` authoritative) |
| Gate-zero tool | `ltx25/eval/vae_gate_ltx25.py` |
| Preprocessing validation gate | `ltx25/eval/validate_precomputed.py` |
| Checkpoint sweep / inference | `ltx25/eval/sweep_ltx25.py` |
| Runner (gate / preprocess / train) | `ltx25/run_ltx25.sh` |
| Box-resolved training config | `ltx25/configs/i2v_lora_ltx25.workspace.yaml` |
