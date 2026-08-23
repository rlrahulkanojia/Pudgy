# Experiment **alpha v-alpha** — LTX 2.5 base migration + primitives data contract

**Status:** plan · **Base:** LTX-2.5-22B (Lightricks/LTX-2, `packages/ltx-trainer`) · **Predecessor:** [v4 / LTX-2.3](../v4/README.md) (G1 ✅, G2 ✅)
**Run-name prefix:** `ava_` · **Hardware assumed:** 1–2 × 80 GB (A100/H100), CUDA 13+, Linux

> **One line.** `alpha v-alpha` ports the *proven* v4 LTX track onto **LTX 2.5**, and in the same
> harness runs the one experiment that v4 left unanswered: **does a cleaner data contract beat a
> bigger dirty one?** The base swap buys quality; the data contract buys controllability. They are
> measured separately so we can attribute the gain.

---

## 0. Why this experiment exists

v4 (LTX-2.3) shipped a working system and, more valuably, a set of *measured negatives*
([v4 README §5.2](../v4/README.md)): LoRA strength, guided dev inference with CFG+STG, negative
prompts, and LoRA stacking **all failed** to fix the two identity failure basins. Only two things
worked, and both were **data/labelling or structure** interventions:

1. **IC-LoRA edge conditioning** → fixed species/construction (structure supplied outside the text).
2. **Two words of colour in the caption** → fixed Polly rendering blue (labelling supplied a signal
   that was absent from all 33 solo-Polly captions).

That is the thesis `alpha v-alpha` is built on: **inference knobs are exhausted; the remaining
levers are the base model and the data contract.** LTX 2.5 is the base lever. The v5 primitive
taxonomy ([Training_Approach_v5.md](../v5/Training_Approach_v5.md) §3) is the data lever.

---

## 1. What LTX 2.5 actually is

### 1.1 Component list (split pack, ~66 GiB)

LTX 2.5 abandons 2.3's single bundled `.safetensors`. Weights ship **one file per component**, so
you download only what a pipeline needs.

| Component | File | Needed for |
|---|---|---|
| Transformer (full) | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | guided two-stage pipelines, **training** |
| Transformer (distilled) | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `DistilledPipeline`, **`ICLoraPipeline`**, `DubItPipeline` |
| Text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | everything — **Gemma 4**, LTX-finetuned, projection bundled |
| Video VAE (diffusion) | `vae/ltx-2.5-video-vae-bf16.safetensors` | better decode quality; `NADiffusionDecoder`, wants `natten` |
| Video VAE (conv) | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | lighter decode, no extra deps |
| Audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | any audio path (we skip audio, but preprocessing wants it named on a split pack) |
| Spatial upscaler ×2 | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | two-stage pipelines; our 768×1344 → 1536×2688 path |
| **Temporal upscaler ×2** | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | **new** — `DFRPipeline` temporal refine rounds |
| Distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | two-stage pipelines running dev in stage 1 |
| **Detailing IC-LoRA** | `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` | **new** — `DFRPipeline` refinement stage |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | optional; `--auto-duration`, length predicted from prompt |

### 1.2 What changed vs LTX-2.3 — and what it costs us

| Change | Consequence for this project |
|---|---|
| **Text encoder Gemma 3 → Gemma 4** | ⚠️ **Every cached `conditions/*.pt` from v4 is invalid.** Re-run `process_dataset.py` into a **fresh `.precomputed/`** (or `--overwrite`). The loader validates the checkpoint↔Gemma pair, but does **not** re-encode stale `.pt` files — it silently skips them. |
| **LoRAs are not portable across model versions** | ⚠️ **`pudgy_lora_B_768`, `pudgy_ic_768` and `pudgy_p3_768` must all be retrained.** This is the single largest cost of the migration, and the reason A0 below is a gate, not a formality. |
| Different VAE weights | Video latents must be recomputed too (same geometry, different encoder). |
| **Latent geometry unchanged** | ✅ 128 channels, **32× spatial, 8× temporal** (`AGENTS.md`: "LTX 2.5 RC checkpoints use the default … layout"). So **W,H ÷ 32** and **frames % 8 == 1** still hold → **the v4 `768×1344×49` clip spec carries over verbatim**. `prep_ltx.py` needs no change. |
| **Trainer validation default fps 25 → 24** | ✅ Our source is **native 24 fps**. v4 had to re-derive 24 → 25; on 2.5 we can **drop the resample entirely**. (Verify on box — it is a config default, not a hard model constraint.) |
| **`DFRPipeline`** (detail-fidelity rendering) | Generated interior keyframes + full-res detailing pass + up to 2 rounds of **temporal ×2** refine. Aims squarely at v4's measured weak spot: jerk (0.20 at the golden IC step) and held/frozen frames (8–21%). |
| **`RetakePipeline`** | Regenerate one time-region of an existing clip — beat-level fixes without re-rolling the whole shot. |
| Duration head / `--auto-duration` | Length inferred from the prompt; fits the Claude prompt subsystem (§7 of the v4 plan). |
| NVFP4 (Blackwell) + `fp8-cast`/`fp8-scaled-mm`, natten VAE decode | Memory/throughput headroom. Relevant if we ever drop below 80 GB. |
| Per-modality validation guidance | `video_cfg_scale` / `audio_cfg_scale`, `video_stg_scale` / `audio_stg_scale`, `guidance_rescale`. Old `guidance_scale`/`stg_scale` fields are auto-migrated. |
| License re-dated | `LTX-2.x Community License Agreement`, **license date 2026-08-11**. v4's clearance (2026-07-12) was against the earlier LTX-2 text. ⚠️ **Re-read and re-confirm before delivery** — expected to be a non-issue, but it is a different dated document. |

### 1.3 What does *not* change

- **IC-LoRA is still distilled-only at inference** (`ICLoraPipeline` expects the distilled
  transformer) and still has **no CFG / negative prompt**. The flat-2D look must keep coming from
  the LoRA + positive prompt. v4 limitation #5 stands.
- **Video models still can't render legible text.** Speech-bubble source clips stay excluded.
- The v4 architectural bet — **identity via conditioning (edge/IC-LoRA + keyframes), style via a
  light LoRA** — is unchanged. Nothing in 2.5 supersedes it.

---

## 2. Honest verdict: will LTX 2.5 get us better results?

**Partly — and not on the axis that is actually capping us.**

| Axis | 2.5 helps? | Why |
|---|---|---|
| Decode fidelity (thin outlines, flat fills) | **Likely** | New diffusion video VAE; must be *measured*, not assumed (gate A1). |
| Motion smoothness / frozen frames | **Likely, and this is the best single win** | DFR + temporal ×2 upsampler are purpose-built for it, and it is our measured weakness. |
| Prompt adherence | **Likely** | Gemma 4 encoder on a base the v4 doc already called "extremely prompt-sensitive". Our Claude prompt subsystem gets more to work with. |
| Workflow / iteration speed | **Yes** | Split checkpoints (download only what you need), Retake, duration head, fp8/NVFP4. |
| **Seed-fragile identity in text-only generation** | **No** | Structural (v4 §5.1–5.2). Fixed by edge conditioning, not by a base version. |
| **Polly under-representation, coverage gaps, unaddressable failures** | **No** | Pure data-contract problems. A newer base cannot invent a signal the corpus never contained. |

**Ranked levers, by expected effect on our actual goal:**

> **data contract  >  IC-LoRA / edge control  >  base version (2.3 → 2.5)  >  inference knobs (exhausted)**

That ranking is why `alpha v-alpha` is *two* experiments in one harness rather than a migration.

---

## 3. Experiment design

Five arms. A1 → A0 are sequential gates; A2 and A3 run off A0's output; A4 was data-gated and is now largely unblocked.
Every arm names its **decision**, so no arm can end in "interesting but inconclusive".

| Arm | Name | Needs new data? | Decision it produces |
|---|---|---|---|
| **A1** | VAE gate | no | Which 2.5 video VAE (diffusion vs conv); is 2.5 representation-safe for flat 2D? |
| **A0** | Port gate | no | **Does 2.5 ≥ 2.3 on our rubric?** If no → stay on 2.3 and stop here. |
| **A2** | Free-wins gate | no | Do DFR + temporal ×2 beat the v4 jerk/frozen-frame baseline at inference only? |
| **A3** | **Data-contract ablation** | no (uses a *filtered* subset) | **Does a small clean corpus beat the big dirty one?** → justifies (or kills) the client data ask. |
| **A4** | Primitives slice | **mostly landed** — 68 clips in (2026-08-20); 16 stills + a `sad` cell still open | Are primitives separately addressable? Does Polly reach parity? |

### A1 — VAE gate *(hours, cheap, do first)*

Add both 2.5 video VAEs to `training_approach/scripts/vae_roundtrip.py`'s `REGISTRY` and round-trip
real Pax/Polly art at **768×1344**.

- Measure PSNR/SSIM **and** look specifically for: thin-outline softening, flat-region banding, and
  the flat-region grid artefact that affected 2.3 ([#202](https://github.com/Lightricks/LTX-2/issues/202)).
- CogVideoX's 8× VAE scored PSNR 38.9 / SSIM 0.996 on this art ([FINDINGS §4](../FINDINGS.md)) — that
  is the bar.
- **Gate:** pick the decoder; if *both* soften outlines materially vs 2.3, escalate before spending
  on A0.

### A0 — Port gate *(the expensive one; ~2 training runs)*

Rebuild the v4 production stack on 2.5, changing **nothing but the base**.

- Data: the existing **`Data/processed/v4_ltx_249clip/`** — 249 clips, 1080×1920 @ 24 fps, 825.8 s, captions
  already colour-grounded and schema-conformant. Re-encode with the *unchanged*
  `prep_ltx.py --bucket 768x1344x49`.
- Re-preprocess into a **fresh `.precomputed/`** (new VAE + Gemma 4).
- Retrain two LoRAs with the v4 recipes:
  - `ava_style_768` — style LoRA, **attention + FFN** (v4's config B was the golden), rank 32.
  - `ava_ic_768` — IC-LoRA on Canny edge pairs, rank 32 (v4 used 128; the trainer's own
    guidance is 16–32 for structural control — sweep both if time allows).
- Evaluate with the **existing `eval_v4/` tooling**, the **same held-out prompt set**, and the
  **same v2 §5 rubric**, so the numbers are directly comparable to the v4 goldens.
- ⚠️ **≥ 8 seeds per prompt.** v4 §5.1: G1 used seed 42 and hid the true failure-basin rates.
  Report the *rate* of raccoon-basin and 3D-claymation-basin, not a best-of.
- ⚠️ **Checkpoint sweep, not final-step.** v4's IC-LoRA golden was **step 1000**, with 2.4× lower
  jerk than step 2000 at identical identity. Keep `keep_last_n: -1` and sweep.

**Gate A0:** 2.5 must be **≥ 2.3** on identity rate and **≥** on smoothness. Anything less and the
migration is not worth retraining every LoRA — say so and stay on 2.3.

### A2 — Free-wins gate *(inference only, no training)*

On A0's outputs, compare:

| Path | Against |
|---|---|
| `ICLoraPipeline` (as v4) | v4 baseline: jerk **0.20**, frozen frames **8.3–20.8%** |
| `DFRPipeline` + detailing IC-LoRA, `--temporal-upsample-rounds 1` and `2` | same metrics |
| `RetakePipeline` on a known-bad beat | does it fix locally without breaking identity? |

Use `eval_v4/micro_anim_eval.py` verbatim — it already measures jerk, frozen %, flicker, loop-seam
spikes. Heed its documented caveats: **read ratios alongside absolute values**, and
**always eye-check before recommending** (motion metrics miss identity — `reunion_walk` scored best
and was completely off-model).

**Gate A2:** if DFR reduces jerk without degrading identity, it becomes the default render path and
`--temporal-upsample-rounds` becomes a delivery knob.

### A3 — Data-contract ablation ⭐ *(the arm that answers the user's question)*

**This is the cheapest possible test of whether the client data ask is worth making.**

Train the identical recipe on a **filtered** subset of the same 249 clips — one that satisfies the
v5 intake rules — and compare it to the full-249 A0 run.

Filter (all four, using `eval_v4/source_scan.py` which already scores these):

1. **No speech bubbles / on-screen text** — text renders as gibberish.
2. **`char_frac ≥ 0.14`** — character large in frame. v4 §5.4 measured this as a *driver* of identity
   quality: small characters → thin Canny edges → off-model output.
3. **No internal cuts** — and verify visually; `report.csv`'s `has_internal_cuts` flag missed one
   (26 of the 249 are merged clips that contain internal cuts by construction).
4. **One action per clip** — reject anything needing two primitive phrases (v5 §4 rule 3).

Then **rebalance by construction**: oversample solo-Polly to parity with solo-Pax (today 33 : 59,
with 69% of clips two-character and therefore diluting both).

**Two outcomes, both useful:**

- **Clean-subset wins** → the primitives contract is validated on data we already own. The client
  ask in §5 becomes evidence-backed, and we can quantify the expected gain per clip delivered.
- **Full-249 wins** → volume dominates curation at this scale; the ask shifts from *"clean primitive
  clips"* to *"more footage"*, which is far cheaper for the client. Equally decision-changing.

⚠️ **Power caveat, stated up front.** v4 measured that only **17 of 206** two-character clips are
text-free with both characters. If the full filter leaves **< ~60 clips**, this arm is underpowered
and its own honest result is *"we cannot answer this without new data"* — which is itself the
argument for §5. Run `source_scan.py` **before** committing GPU time and report the surviving count.

### A4 — Primitives slice *(data-gated)*

Runs when the §5 **T1 slice** lands. Trains the same recipe on primitive-labelled clips and scores
**per primitive**, so a failure reads as *"`motion.wave.polly` is weak"* rather than *"the clip is
bad"*. This is the v5 promise; `alpha v-alpha` is where it gets tested on the LTX track instead of
the Wan track.

Today's intake is **68 expression clips** across `Data/raw/iteration_3/03_expression_clips/` — Pax and
Polly × happy (7 angles each), surprised, angry and neutral (9 angles each). Batch 1 (2026-08-07) was
the 7 Pax/happy angles; Batch 2 (2026-08-20) added the other 61 and the first Polly footage in the
programme. Prepared to **272 training clips** (× 4 composited grounds) in
`Data/processed/v6_expressions_272/`.

That is **4 of the 9 expression primitives, both characters, evenly balanced** — enough to test
separate addressability and Polly parity, which is what A4 asks. It is still an angle spread of a
*fixed* performance per cell, not 10 distinct performances, so within-cell variety remains the thin
axis. `sad` — one of the three T1 primitives — has not arrived; `angry` and `neutral` came instead.

---

## 4. Dataset requirements

Two separate things, and conflating them is how datasets get rebuilt twice: **format** (what the
trainer will accept) and **content** (what actually makes the output good).

### 4.1 Format — hard requirements enforced by `ltx-trainer`

| Requirement | Value | Note |
|---|---|---|
| Metadata file | **CSV / JSON / JSONL** with `caption` + `video` columns | `media_path` is a legacy alias for `video` |
| Path resolution | **relative to the metadata file's own directory** | staging media out-of-tree is the #1 preprocessing failure |
| Spatial dims | **W and H divisible by 32** | 2.5 uses the default 32× spatial VAE |
| Frame count | **`frames % 8 == 1`** → 25, 33, 49, 81, 97, 121 | 2.5 uses the default 8× temporal VAE |
| Bucket | preprocess bucket **==** intended generation size | LTX-2 [#155](https://github.com/Lightricks/LTX-2/issues/155); v4 hit this |
| fps | **24** (native source; 2.5 trainer default) | no resample needed any more |
| Audio | silent → `--skip-audio`, and keep `audio: null` in the strategy | audio branch untrained; a future upsell, not scope |
| Images | allowed at `F=1`; mixing stills + video needs **multi-bucket** (`"768x1344x1;768x1344x49"`) **and `batch_size: 1`** | docs' officially-supported alternative: train two LoRAs and stack |
| IC-LoRA pairs | `reference_video` column, **same frame count** as target; may be spatially downscaled via `--reference-downscale-factor` | validation `reference` conditions must repeat the same factors |
| Trigger token | **inline in captions** (v4's choice) — *not* `--lora-trigger` | correct here: per-clip character presence differs, so a blanket prepend would wrongly stamp Polly-only clips |
| Text embeddings | recompute for **Gemma 4** — fresh `.precomputed/` or `--overwrite` | stale `.pt` files are silently skipped |

**Chosen bucket for `alpha v-alpha`: `768×1344×49` @ 24 fps, silent.** Unchanged from v4, deliberately —
so the base swap is the only variable.

```
sequence_length = (H/32) · (W/32) · ((F−1)/8 + 1)
768×1344×49  → 24 · 42 · 7  =  7,056 tokens     ← chosen (v4-measured: ~45 GB style, ~62 GB IC-LoRA)
768×1344×97  → 24 · 42 · 13 = 13,104 tokens     ← only if A2 shows DFR can't cover the length gap
768×1344×121 → 24 · 42 · 16 = 16,128 tokens
```

> **Train short, extend at inference.** The ×2 temporal upscaler plus DFR means length and smoothness
> are now partly an *inference* concern. Training at 49 frames keeps the sequence — and the VRAM —
> at v4's proven point. v4 §5.5 already showed edge-conditioned identity holds to **249 frames**;
> length was never the training bucket's job.

⚠️ **Gradient checkpointing is mandatory at this bucket** — v4 measured that disabling it needs
~78 GB for the backward pass and OOMs. And **2 GPUs buy experiment throughput, not resolution**:
run two configs concurrently, never data-parallel one run.

### 4.2 Content — the contract that actually drives quality

Every rule below is *earned from a measured v4 or v1 result*, not style preference.

| # | Rule | Evidence |
|---|---|---|
| 1 | **Colour-ground every character in every caption** — `pxngn0, a blue penguin` / `plngn0, a pink penguin` | v4 §5.3: the same IC-LoRA renders Polly **blue** without it and **pink** with it. Zero of 33 solo-Polly captions contained "pink". |
| 2 | **Characters large in frame** (`char_frac ≥ 0.14`) | v4 §5.4: character size in frame drives identity quality; small → thin edges → off-model. Keep wide shots a deliberate minority. |
| 3 | **No speech bubbles / on-screen text** | v4 §8.1: renders as gibberish ("xngno", "angea"). |
| 4 | **No internal cuts** — verify by eye, not by flag | v4 §6: the `has_internal_cuts` flag missed a clip and the model faithfully reproduced the cut. |
| 5 | **One primitive per clip** — one action, one intent | v5 §4 rule 3. If a clip needs two action phrases, reject it at intake. |
| 6 | **Character balance enforced by construction**, not emergent | today 33 solo-Polly : 59 solo-Pax, 69% two-character → both signals diluted. |
| 7 | **Camera is a caption slot, never a primitive** — vary zoom/angle/facing *across* clips of the same primitive | v5 §3.5: making it a primitive entangles framing with action. |
| 8 | **Backgrounds deliberately varied per primitive** | decorrelates content from background; v1–v4 had them correlated. |
| 9 | **Expression clips are `neutral → expression → hold`**, hold **animated** not frozen | makes an expression a temporal primitive, and lets a composition cut into and out of it cleanly. |
| 10 | **Style block verbatim-constant across the whole corpus** | prevents slot vocabulary drift; captions are rendered programmatically from structured records, never hand-written. |
| 11 | **Long-form = concatenate *different* consecutive clips, never loop one** | v4 §5.6: looping one 49-frame edge gives the same 2 s action 5× with hard cuts every 49 frames (seam motion up to 528× median). |

### 4.3 Structure on disk

```
Data/LTX-2.5/
├── dataset.json               # [{caption, video, reference_video?}, …]  paths RELATIVE to this file
├── holdout.jsonl              # never preprocessed, never trained — Phase-9 generalisation set
├── clips/                     # 768×1344, 24 fps, 49 frames, H.264 yuv420p, SILENT
│   └── <primitive_id>__<source>__<nn>.mp4
├── edges/                     # Canny reference videos, SAME frame count as their target
│   └── <same basename>.mp4
├── stills/                    # design.turn.* / design.sheet.*  (F=1 bucket, if the image arm runs)
├── prompts/                   # structured prompt records — the source of truth
│   └── <clip_id>.json         # v4 §7.1 schema; dataset.json captions are a RENDERING of these
├── character_bible.json       # canonical Pax/Polly: shape, proportions, exact colour codes, tokens
├── catalog.json               # primitive coverage matrix — auditable, fillable
└── .precomputed/              # ← data.preprocessed_data_root  (FRESH for 2.5 — Gemma 4)
    ├── latents/               #   video latents
    ├── conditions/            #   Gemma-4 text embeddings
    └── reference_latents/     #   IC-LoRA reference latents
```

**Primitive ID grammar** (v5 §3), one per clip, and the join key between `catalog.json`, the caption
record, and the eval report:

```
<class>.<primitive>.<character>
class     ∈ {design, motion, expression, moment}
character ∈ {pax, polly, both}

e.g.  motion.wave.polly · expression.happy.pax · moment.relative_size.both
```

**Caption render template** (v5 §4) — one flowing paragraph, ≤200 words, LTX-native ordering:

```
<token>, a <colour> penguin. 2d cartoon animation in the Pudgy Penguins style, thick clean
black outlines, flat pastel colors, cel shading; <primitive phrase>; <expression phrase>;
<camera phrase>; <background phrase>.
```

**Holdout.** Reserve ~10% (or a handful) *before* preprocessing, seed 42, and remove those entries
from `dataset.json`. v4 had no formal holdout; without one, "it generalises" is unfalsifiable.

### 4.4 Volume — what exists, what's needed

| Class | Primitives | × chars | Clips/cell | **Target** | **Delivered** |
|---|---|---|---|---|---|
| `design.turn` | 8 angles | 2 | stills | **16 stills** + colour swatches | 0 stills (1 Pax turnaround *clip*, 240 f) |
| `design.sheet` | 11 emotions | 2 | stills | **100–200 stills** | 0 |
| `motion` | 9 (walk, run, turn, sit, idle, jump, wave, head_turn, bounce) | 2 | 4–5 | **~90 clips** | 0 |
| `expression` | 9 (happy, sad, angry, surprised, scared, confused, laughing, crying, affectionate) | 2 | 10 | **~180 clips** | **68** (Pax+Polly × happy/surprised/angry/neutral) |
| `moment` | 7 (hug, hold_flippers, high_five, back_to_back, piggyback, sync_action, relative_size) | both | ~8 | **~56 clips** | 0 |
| | | | | **≈ 326 clips + 116–216 stills** | **68 clips (21%), 0 stills** |
| *carry-over* | `Data/processed/v4_ltx_249clip/` skit-cut corpus | | | 249 clips / 825.8 s | ✅ on hand |

### 4.5 ⭐ The T1 slice — the bounded ask that unblocks A4

**Do not wait for 326 clips.** The minimum that *tests the hypothesis* is a vertical slice: enough
clips of one primitive to learn it, ≥3 primitives to test separate addressability, and both
characters to test balance.

> **T1 = 3 expressions (happy, sad, surprised) × 2 characters × 10 clips = 60 clips**
> **+ both 8-angle turnarounds = 16 stills.**
>
> Each clip: `neutral → expression → animated hold`, one character, simple varied background,
> static camera unless camera is the variable, character large in frame, no text, ≥2 s at 24 fps,
> native resolution (short edge ≥ 1080), clean base render — **no watermark, no burned-in text, no
> timecode**. Camera variety spread *across* the 10 clips of each cell, not within them.

**Status after the 2026-08-20 delivery: T1 is substantially met, but not on the requested axis.**
68 clips arrived against the 60-clip ask, both characters, evenly balanced — so *addressability* and
*Polly parity* are now testable. What is still missing: **`sad`** (the delivery brought `angry` and
`neutral` instead), and **the 16 turnaround stills** — a single 240-frame Pax turnaround clip came
instead, with no Polly turnaround and no colour swatches.

A4 can therefore run on `happy / surprised / angry / neutral`, treating `sad` as a deferred cell.
The full §4.4 table stays the standing ask; the stills and one `sad` cell are what to chase now.

---

## 5. Runbook

```bash
# 0 — env + models (~66 GiB, split pack)
git clone https://github.com/Lightricks/LTX-2 && cd LTX-2 && uv sync --extra natten
hf auth login
hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
    --local-dir models/ltx-2.5

# A1 — VAE gate  (add both 2.5 VAEs to REGISTRY first)
python training_approach/scripts/vae_roundtrip.py --vae ltx-2.5-diffusion --vae ltx-2.5-conv \
    --size 768x1344 --clips <pax.mp4> <polly.mp4>

# A0.1 — clips  (unchanged spec; prep_ltx.py needs no edit)
python prep_ltx.py --bucket 768x1344x49          # from Data/processed/v4_ltx_249clip/

# A0.2 — Canny edge pairs for the IC-LoRA arm  (use ABSOLUTE paths — v4 gotcha)
uv run python scripts/compute_reference.py <clips_dir> -o <dataset.json> --override

# A0.3 — preprocess into a FRESH .precomputed  (split pack ⇒ VAEs must be named)
uv run accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  scripts/process_dataset.py <dataset.json> \
    --resolution-buckets "768x1344x49" \
    --model-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    --text-encoder-path models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path   models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path   models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --skip-audio --decode
#   --decode writes .precomputed/decoded_videos — EYEBALL THESE before training.
#   Omit --lora-trigger: our captions already carry pxngn0/plngn0 per clip.

# A0.4 — train
uv run python scripts/train.py configs/ava_style_768.yaml    # style LoRA, attn+FFN, rank 32
uv run python scripts/train.py configs/ava_ic_768.yaml       # IC-LoRA, reference + first_frame 0.2
```

**Config deltas from the shipped examples** (start from `i2v_lora.yaml` for style, `v2v_ic_lora.yaml`
for IC-LoRA — the trainer's own guidance makes I2V the default for a concept/style LoRA because
`first_frame` at `probability: 0.5` yields a T2V/I2V superset from one run):

```yaml
model:
  model_path:         ".../diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  text_encoder_path:  ".../text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  video_vae_path:     ".../vae/ltx-2.5-video-vae-bf16.safetensors"     # required on a split pack
  audio_vae_path:     ".../vae/ltx-2.5-audio-vae-bf16.safetensors"
lora:
  rank: 32
  alpha: 32
  target_modules: [attn1.*, attn2.*, ff.net.0.proj, ff.net.2]   # attn+FFN — v4's golden was config B
training_strategy:
  video: { is_generated: true, latents_dir: latents, conditions: [...] }
  # audio block OMITTED — silent corpus (v4's `audio: null`)
optimization:
  enable_gradient_checkpointing: true    # MANDATORY at 768×1344 — off OOMs (~78 GB)
validation:
  video_dims: [768, 1344, 49]            # MUST equal the preprocess bucket (#155)
  frame_rate: 24.0
  generate_audio: false
checkpoints:
  interval: 250
  keep_last_n: -1                        # fidelity peaked at step 1000 in v4 — sweep, don't take the last
```

---

## 6. Gates, and what each one buys

| Gate | Pass condition | If it fails |
|---|---|---|
| **G-A1** | 2.5 VAE round-trips Pudgy art with no outline softening or flat-region banding vs the 38.9 dB / 0.996 bar | pick the other decoder; if both fail, stop the migration |
| **G-A0** | On ≥8 seeds × the held-out set, 2.5 ≥ 2.3 on identity rate **and** smoothness | stay on 2.3; the retraining cost isn't repaid |
| **G-A2** | DFR + temporal ×2 lowers jerk below 0.20 without degrading identity (eye-checked) | keep the v4 `ICLoraPipeline` render path |
| **G-A3** | Clean subset ≥ full 249 on identity rate | volume beats curation → re-aim the client ask at *more footage*, not *cleaner clips* |
| **G-A4** | Each trained primitive is separately addressable; Polly reaches Pax parity | localise the failure to a named primitive and issue a bounded data ask for it |

---

## 7. Risks and open verifications

1. **Every LoRA must be retrained.** Non-negotiable cost; A0 is a gate precisely because of it.
2. **Stale precomputed data fails silently.** Gemma-3 `conditions/*.pt` are skipped, not rejected.
   Always a fresh `.precomputed/`; `--decode` and eyeball before training.
3. **Underpowered A3.** If `source_scan.py` leaves < ~60 clean clips, report that as the result.
4. **License re-dated 2026-08-11** — re-confirm; v4's clearance was against the older text.
5. **Verify on box:** the trainer's 24 fps default against our 24 fps clips (no resample); IC-LoRA
   rank 32 vs v4's 128; whether `--reference-downscale-factor 2` costs identity on thin outlines
   (it halves reference tokens, which is real VRAM on the 62 GB IC-LoRA run).
6. **Two-character shots remain the weakest axis.** IC-LoRA conditions one identity at a time;
   `moment.*` clips plus colour-grounding are the mitigation, not a solution.
7. **Metrics lie about identity.** `reunion_walk` scored best on smoothness and was completely
   off-model. Every gate needs a human or vision-judge eye-check.

---

## 8. Sequencing

| Step | Blocked on | Output |
|---|---|---|
| A1 VAE gate | nothing | decoder choice; go/no-go |
| `source_scan.py` over the 249 | nothing | the A3 clean-subset count — **run this before booking GPU time** |
| **T1 client ask** (§4.5) | client | 53 more clips + 16 stills → unblocks A4 |
| A0 port gate | A1 | `ava_style_768`, `ava_ic_768` + comparison vs v4 goldens |
| A2 free-wins | A0 | render-path decision |
| A3 data-contract ablation | A0 + scan count | **the evidence behind the data ask** |
| A4 primitives slice | T1 delivery | per-primitive scores |

---

## 9. Artifact index

| Artifact | Location |
|---|---|
| This plan | `training_approach/LTX-2.5/Experiment_alpha_v-alpha.md` |
| Carry-over dataset | `Data/processed/v4_ltx_249clip/` (249 clips, captions, `prep_ltx.py`, `character_bible.json`) |
| New-data intake tree | `Data/raw/iteration_3/` (68 of ~326 clips delivered; see its `CHANGELOG.md`) |
| Prepared expression set | `Data/processed/v6_expressions_272/` (272 clips — built 2026-08-20, not yet trained) |
| v4 measured results | [`training_approach/v4/README.md`](../v4/README.md) |
| Primitive taxonomy + caption schema | [`training_approach/v5/Training_Approach_v5.md`](../v5/Training_Approach_v5.md) §3–4 |
| Eval tooling (carries over unchanged) | `eval_v4/{g2_generate,g2_batch,micro_anim_eval,metrics,source_scan,make_montages}.py` |
| VAE round-trip tool | `training_approach/scripts/vae_roundtrip.py` |
| Client data request | `docs/documents/final_docs/Data_Requirements.md`, `Client_Data_Request_Round3.md` |
