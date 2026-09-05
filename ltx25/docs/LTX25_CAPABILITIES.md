# LTX-2.5 feature survey — what's actually available, and what v7 can use

**Status:** survey · **Companion to:** [`Experiment_alpha_v-alpha.md`](Experiment_alpha_v-alpha.md)
**Reference corpus:** `raw/iteration_4` → `processed/v7_primitives_1776` · **Written:** 2026-09-05

> **One line.** The plan doc was written against a partial view of LTX-2.5: it knows one IC-LoRA
> and assumes every LoRA must be retrained. There are **17 IC-LoRA adapters**, the vendor says
> 2.3 LoRAs mostly **run on 2.5 unchanged**, and one adapter — **Ingredients** — does
> training-free character identity, which is the single failure basin v4 never closed. The trainer
> is also broader than assumed: **extension and outpainting are training modes**, and the vendor's
> own mode selector prescribes **different ranks for motion vs character/style** — which v7's
> shared recipe does not do. Against all that, v7 has **one hard blocker**: half its frame buckets
> are illegal on LTX, including both floors that carry its anti-confound design.

---

## 0. Corrections to the plan doc

Two load-bearing claims in `Experiment_alpha_v-alpha.md` are contradicted by current vendor
documentation. Both change cost, so both are worth resolving before booking GPU time.

| Plan says | Vendor says | Consequence |
|---|---|---|
| §1.2 / Risk 1: "**LoRAs are not portable across model versions** — `pudgy_lora_B_768`, `pudgy_ic_768` and `pudgy_p3_768` must all be retrained. This is the single largest cost of the migration, and the reason A0 is a gate." | "The large majority of LoRAs and IC-LoRAs trained on LTX-2.3 run on **LTX-2.5 without changes**" — a small number of exceptions exist; validate before production. ([HF model card](https://huggingface.co/Lightricks/LTX-2.5)) | **A0 may collapse from ~2 training runs to a load-and-eval.** Test first: load the existing v4 LoRAs on the 2.5 transformer and render the held-out set. If they load and hold identity, the port gate is answered for the price of inference. |
| §1.2: only the **Detailing IC-LoRA** (`Pixel-Spatial-Upscaler`) is named as new. | **17 IC-LoRA adapters** are published, 13 of them explicitly 2.5-compatible. ([IC-LoRA adapters](https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters)) | Several are directly on-target for our failure modes — see §2. |

**On the retraining claim specifically.** The plan cites `AGENTS.md` for latent geometry, so it is
worth noting what that file does and does not say. It confirms the geometry verbatim — "Video
latent channels … 128", "Spatial compression: 32×", "Temporal compression: 8×" — but it makes
**no statement at all about LoRA portability across versions**. The plan's "LoRAs are not portable"
is therefore unsourced, while the 2.5 model card asserts the opposite. Treat it as an open
question with a cheap test attached, not as a settled cost.

**Not contradicted, still true:** the Gemma 3 → Gemma 4 text-encoder swap really does invalidate
every cached `conditions/*.pt`. `AGENTS.md` is explicit: "Caption features are produced by the
selected Gemma model and are **not interchangeable across model versions** … preprocess into a
fresh `conditions/` directory or pass `--overwrite`. Existing `.pt` files are **skipped by
default**." A fresh `.precomputed/` remains mandatory. That is a *preprocessing* invalidation, and
it is independent of whether the LoRA weights themselves port.

`AGENTS.md` adds two gotchas the plan does not carry: the Gemma root must be **LTX's fine-tuned
Gemma 4, not Google's vanilla Gemma 4** (validated against the checkpoint's `gemma_source_checkpoint`
metadata), and **model version detection is fully automatic** — "do not add an explicit version
flag."

---

## 1. The v7 corpus, as it actually is

`raw/iteration_4/LSLTTT-Project` — 180 files, 1.3 GB, of which 70 are byte-identical re-sends of
`iteration_3`. Delivered 2026-09-03.

| Property | Value |
|---|---|
| Codec / pixel format | ProRes 4444, **`yuva444p12le` — real 12-bit alpha** |
| Geometry | **1080×1080 square**, 24 fps |
| Structure | `<CATEGORY>/<ACTION\|EXPRESSION>/<CHAR>/` — **9 fixed camera angles** per cell |
| Angles | `FRONT`, `QF1_L/R`, `QF2_L/R`, `QF3_L/R`, `SIDE_L/R` |
| Motion (first in programme) | walking, running, waving, sitting, jumping × Pax/Polly |
| Expressions | angry, happy, laughing, neutral, surprise × Pax/Polly |
| Turnaround | **`PAX_TURNAROUND.mov` only** — 240 frames. No Polly turnaround. |
| Source durations | **0.67 s – 1.63 s** (16–39 frames) |
| Prepared to | `processed/v7_primitives_1776` — 1024², 1776 clips (motion 856 / expression 920) |

Two structural facts drive everything below:

1. **The nine angles are one performance rendered from nine cameras**, not nine takes. Within-cell
   variety is camera, not performance.
2. **The nine angles are all front-hemisphere.** `FRONT` through `SIDE_L/R` spans roughly 180°.
   Nothing looks at the character from behind.

---

## 2. Capability inventory, with verdicts

### 2.1 IC-LoRA adapters

All are published against 2.3 and documented as 2.5-compatible unless noted.

| Adapter | What it does | Verdict for Pudgy |
|---|---|---|
| **Ingredients** | Character / prop / location consistency from a **reference sheet** image | ⭐ **The headline.** Training-free identity. See §3. |
| **Union Control** | Depth + Canny + pose **in one checkpoint**, multiple simultaneous signals | ⭐ **Supersedes v4's Canny-only IC-LoRA.** Alpha gives us all three signals for free (§4). |
| **Motion Track Control** | Sparse spline motion trajectories | **Promising for `motion.*`** — trajectory control that does *not* lock silhouette the way an edge video does. |
| **Pixel Spatial Upscaler** (= plan's "Detailing IC-LoRA") | Creative 2×/4× upscale with synthesised detail | **Yes** — 1080² source, delivery wants more. Already in the plan's A2. |
| In-Outpainting | Canvas extension, masked fill | Useful for reframing square → portrait without re-render. |
| Clean Plate | Subject removal + background reconstruction | Marginal — alpha already gives us clean plates. |
| Colorization / Decompression / Deblur | Restoration | Not applicable — source is clean vector render. |
| Day-to-Night / Water / Relight / Instant Shave / Cross-Eyed | Effects | Not applicable. |
| HDR, Dub-It | Beta | **2.3 only**, 2.5 support in development. Out of scope (we are silent, SDR). |

**Two IC-LoRA parameters the plan does not mention**, and both are structural rather than
strength knobs — which matters, because v4 §5.2 declared *inference knobs exhausted*:

- **`attention_strength` (0.0–1.0)** — global IC-LoRA influence. 0.5 blends control against free
  generation. This one *is* a strength knob and should be assumed exhausted-adjacent.
- **`attention_mask`** — optional **spatial or spatiotemporal** mask giving *region-level* control
  over where the IC-LoRA takes effect. This is genuinely new territory, and it is the only
  mechanism found that could address plan Risk 6 (two-character shots). See §3.3.

### 2.2 Pipelines

| Pipeline | Relevance |
|---|---|
| `DFRPipeline` | Generated keyframes + detailing + temporal refine. Plan A2 already targets it — correctly, it aims at our measured jerk/frozen-frame weakness. |
| `RetakePipeline` | Regenerate one time-region. Beat-level fixes. |
| `ICLoraPipeline` | **Distilled-only, no CFG, no negative prompt** — plan §1.3 stands, and it now gates Ingredients too. |
| `KeyframeInterpolationPipeline` | Transitions between keyframe images. Pairs with our per-angle stills. |
| `TI2VidTwoStagesHQPipeline` | `res_2s` sampler, fewer steps. |
| `A2VidPipeline`, `DubItPipeline` | Audio. Out of scope (silent corpus), but a named upsell. |

### 2.3 Native multishot — worth calling out separately

LTX-2.5 generates **a connected sequence with cuts inside one generation**, holding character,
environment, lighting and style across those cuts, rather than requiring clips be assembled
afterwards.

This directly supersedes plan §4.2 **rule 11** ("long-form = concatenate *different* consecutive
clips, never loop one"), which exists because v4 §5.6 measured hard seams — up to **528× median
seam motion** — when looping one 49-frame edge. Multishot is the native fix for exactly that
defect, and it should be tested before any more effort goes into manual concatenation.

### 2.4 Trainer — from the primary source

`packages/ltx-trainer` is richer than the plan assumes. Everything below is from the trainer's own
docs, not blog write-ups.

**One strategy, many modes.** `training_strategy.name: "flexible"` is the only supported strategy
for new configs. Each modality is marked `is_generated: true` (denoised, contributes to loss) or
`false` (frozen, acts as conditioning); at least one must be generative. Modes are therefore
*configuration*, not separate code paths.

| Mode | Conditions | Relevance to us |
|---|---|---|
| **T2V** | none | baseline |
| **I2V** | `first_frame`, `probability: 0.5` | plan's choice — yields a T2V/I2V superset from one run |
| **Video Extension** | `prefix` / `suffix`, `temporal_boundary` in **latent** frames | ⭐ **see §4.4** |
| **Video Outpainting** | `spatial_crop: [y1,x1,y2,x2]` in pixels | ⭐ **see §4.5** |
| **Video Inpainting** | `mask`, thresholded at 0.5 | region fixes |
| **V2V IC-LoRA** | `reference` (`latents_dir`) | v4's path; Union Control / Ingredients live here |
| A2V / V2A / T2A / A2A / AV2AV | audio | out of scope — silent corpus |

**Conditions compose.** The docs state `reference` + `first_frame` can combine on a V2V IC-LoRA.
v4 treated style-LoRA and IC-LoRA as two separate runs; one run can carry both signals.

**`target_modules` — the plan's "config B" is the documented default.** For video-only IC-LoRA the
docs prescribe exactly `attn1.{to_k,to_q,to_v,to_out.0}`, `attn2.{…}`, `ff.net.0.proj`, `ff.net.2`
— which is the attn+FFN set v4 measured as its golden. Cross-modal modes instead use the short
patterns `to_k`/`to_q`/`to_v`/`to_out.0`, which match all branches.

**Config values worth copying** (defaults unless marked):

```yaml
lora:        { rank: 32, alpha: 32, dropout: 0.0 }   # docs: typical rank range 8-128
optimization:
  learning_rate: 1e-4          # typical range 1e-5 .. 1e-3
  batch_size: 1                # per GPU; 1 is REQUIRED for mixed image+video
  enable_gradient_checkpointing: true     # default true; plan is right that it is mandatory
  optimizer_type: adamw        # or adamw8bit
acceleration:
  mixed_precision_mode: bf16
  quantization: null           # int8-quanto / int4-quanto / fp8-quanto available
  load_text_encoder_in_8bit: false        # saves ~8 GiB
validation:
  frame_rate: 24.0             # already our native rate
  inference_steps: 30
  video_cfg_scale: 3.0         # LTX-2.5 default
  video_stg_scale: 1.0         # 0.0 disables STG
  stg_blocks: [28]
  guidance_rescale: 0.7
  video_modality_guidance_scale: 3.0      # "isolation guidance" - not in the plan
  generate_audio: true         # ⚠️ DEFAULT TRUE - must be set false for our silent corpus
checkpoints:
  keep_last_n: 3               # ⚠️ default 3; plan correctly wants -1 (v4 golden was step 1000)
flow_matching:
  timestep_sampling_mode: shifted_logit_normal   # or uniform
```

⚠️ **Two defaults that will bite silently.** `generate_audio` defaults to **true** — the plan
omits the audio block but never says to set this false, and validation will try to generate audio
on a silent corpus. And `keep_last_n` defaults to **3**, which would discard exactly the early
checkpoint that was v4's golden.

**Hardware.** 80 GB+ is the documented standard; there is a `configs/t2v_lora_low_vram.yaml` for
**32 GB** GPUs using INT8 quantization. Full fine-tuning (`training_mode: "full"`) needs 4–8×
H100 80 GB with FSDP — out of scope, but worth knowing it exists.

**Preprocessing flags the plan does not list:** `--reference-temporal-scale-factor` (alongside
`--reference-downscale-factor`), `--audio-durations`, and `--lora-trigger`, which **prepends to all
captions** — confirming the plan's decision to keep tokens inline per clip, since a blanket prepend
would stamp Pax's token onto Polly-only clips.

### 2.5 ⭐ The vendor's own mode selector — and what it says about v7's rank

LTX-2 ships a Claude Code skill at `.claude/skills/train-model/`, with phase guides
(`prepare-dataset`, `preprocess-dataset`, `launch-and-monitor`, `post-train-validate`) and
references including `mode-selector.md`, `hardware-profiles.md`, `config-patching.md` and a
`plan-template.md`. **It is directly installable into this repo** and encodes the vendor's own
decision procedure — worth adopting rather than re-deriving.

Its mode recommendations answer the character-training question outright:

| Goal | Mode / config | **Rank** | Vendor's rationale |
|---|---|---|---|
| **Character LoRA** | I2V — `configs/i2v_lora.yaml` | **32–64** | focused visual concept; `first_frame` at `probability: 0.5` covers both image-conditioned and text-only inference |
| **Style LoRA** | I2V — `configs/i2v_lora.yaml` | **32–64** | "the same checkpoint loads in both T2V and I2V inference … at no extra cost" |
| **Motion / behavioural LoRA** | I2V — `configs/i2v_lora.yaml` | **8–16** | motion is a "thin signal"; **"high ranks just memorise frame content"** |
| **Control / IC-LoRA (V2V)** | `configs/v2v_ic_lora.yaml` | **16–32** | 16 for structural control; 24–32 if the reference carries richer style/texture |
| Multi-concept complexity | — | 96–128 | higher capacity to keep concepts distinct |
| Default / uncertain | — | 32 | "safe baseline" |

Guidance is to **default to I2V** unless text-only inference is certain, and to keep
**`alpha == rank`** unless there is a specific reason not to.

> ⚠️ **This has a direct implication for v7.** v7 trains motion and expression with a **single
> recipe — rank 16 / α 32** — inherited unchanged from v5 so that results stay attributable to the
> data. On LTX that single rank is wrong at both ends: motion wants **8–16** (v7's 16 sits at the
> ceiling of the range, where the vendor warns about frame memorisation — and v5's *measured*
> failure was exactly start-frame memorisation), while expression is a visual concept closer to a
> character LoRA and wants **32–64**. And `alpha == rank` conflicts with v7's α = 2 × rank.
>
> The v7 expert split already argues that motion and expression are different *kinds* of signal
> and must be separate LoRAs. LTX's own guidance says the same thing about **rank**, independently.
> Porting v7 to LTX is therefore the natural moment to make rank part of the split rather than a
> shared constant — but note it breaks recipe-parity with the Wan line, so it must be a *declared*
> change, not a silent one.

**Hardware tiers**, for planning: **32 GB is the hard floor** ("training is unsupported" below it).
The 32 GB tier is `t2v_lora_low_vram.yaml` — `int8-quanto`, `adamw8bit`, 8-bit text encoder,
rank/α 16. The 80 GB+ tier is `t2v_lora.yaml` — no quantization, `adamw`, rank/α 32. For 40–60 GB
(A40, A6000, L40) the guidance is to start from the low-VRAM config and let autotune relax
settings empirically. The plan assumes 1–2 × 80 GB; a 48 GB box is a supported fallback.

### 2.6 Inference conditioning knobs

From `ltx-pipelines/docs/conditioning.md`:

- **`image_conditionings_by_replacing_latent`** — substitutes the latent at a specific frame with
  an encoded image. Precise frame control.
- **`image_conditionings_by_adding_guiding_latent`** — applies the image as *guidance* rather than
  replacement; documented as "better for smooth interpolation between keyframes."
- **`VideoConditionByKeyframeIndex`** — full reference-video conditioning, **`ICLoraPipeline` only**.
- **`generated_keyframes` / `--num-generated-keyframes N`** (default 0) — interior frames, first
  stage only on TI2Vid and Distilled variants; `DFRPipeline` manages its own slots.

Note that `attention_strength` and `attention_mask` (§2.1) appear in the **hosted** IC-LoRA usage
guide but *not* in this repo doc — verify they are exposed in the installed version before
building the §3.3 two-identity experiment on them.

---

## 3. Character training — three routes, and why the third is the interesting one

The user's question was "character training". LTX-2.5 offers three distinct mechanisms, and they
are not substitutes.

### 3.1 Style / character LoRA (what v4 did)

Train `ava_style_768` on the corpus, identity carried in the weights plus a trigger token inline
in the caption. **Measured to be seed-fragile in text-only generation** (v4 §5.1–5.2). Necessary
for the flat-2D look; insufficient for identity.

### 3.2 Structural IC-LoRA (what v4 fell back on)

Supply identity *outside* the text as a Canny edge video. **This worked** — it was one of only two
interventions that fixed the failure basins. But it carries a cost the plan never states
explicitly:

> **An edge-conditioned generation inherits the reference's motion.** The edge video dictates the
> silhouette frame by frame, so identity and motion are supplied by the same signal. You cannot
> ask for a *new* action while holding identity this way — you get the reference's action back.

That is why v4's long-form attempt had to loop one 49-frame edge and produce the same 2 s action
five times.

### 3.3 Ingredients — identity decoupled from motion ⭐

Ingredients conditions on a **static reference sheet**, not a motion signal.

| Aspect | Value |
|---|---|
| Sheet format | Composite image, **one clean panel per element on a black background** |
| Per character | "face close-up + body turnaround" |
| Typical sheet | 6–8 panels, 2–3 rows × 2–3 columns |
| Fed as | A **static video looped from the sheet**, matched to output resolution/fps |
| Prompt format | `Reference sheet: <panels> / Generated video: <action to generate>` |
| Trained bucket | **≥121 frames**, 768×448 @ 24 fps optimal |
| Strength | 1.0–1.4, ~30 steps |
| Hard limit | "Does not reproduce identities **absent from the supplied sheet**" |

**Why this is the most promising lever in the survey.** v4's measured conclusion was that identity
must be supplied outside the text. Ingredients does that from a *still* sheet — so identity is
pinned while the action stays free for the prompt to drive. It is the first mechanism that
addresses the failure basin without also freezing the motion.

**And we can build the sheet today, with no client delivery.** The sheet wants panels on a black
background: our source is ProRes 4444 **with real alpha**, so every panel composites perfectly with
no matting work. The nine camera angles per cell are already a body turnaround; expression cells
supply the face close-ups. Plan §4.4 lists `design.turn` as **"16 stills, 0 delivered"** and treats
it as a client blocker — for Ingredients purposes it is **derivable from what we already hold**,
with one real gap (§5).

**Known limitations, stated honestly:**

- Ingredients runs on `ICLoraPipeline` → **distilled only, no CFG, no negative prompt**. Plan §1.3
  carries over intact.
- Community reports say IC-LoRA is trained on **single-identity** reference inputs and two-person
  scenes with two distinct identities are **not natively supported**
  ([HF discussion](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/discussions/3)).
  This is plan Risk 6 restated at the adapter level. **`attention_mask` is the untested mechanism**
  that might address it — mask Pax's region to Pax's panels, Polly's to Polly's. Bounded
  experiment, high value if it works, and a clean negative result if it doesn't.
- The 121-frame / 768×448 training bucket does not match our 1024² or the plan's 768×1344. Verify
  before trusting it at our geometry.

---

## 4. Porting v7 to LTX — one hard blocker

### 4.1 ⚠️ Half of v7's frame buckets are illegal on LTX

`prep_v7.py` `buckets_for()` hardcodes **`(f - 1) % 4 == 0`** — Wan's 4× temporal VAE. LTX-2.5 uses
an **8× temporal VAE and requires `frames % 8 == 1`**. The trainer's dataset docs enumerate the
legal values outright:

> valid values are **1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121**

and note that spatial dimensions must be multiples of the VAE spatial factor (32 by default).

| v7 bucket | 4n+1 (Wan) | 8n+1 (LTX) |
|---|---|---|
| f13, f21, f29, f37 | ✅ | ❌ **illegal** |
| f17, f25, f33, f57 | ✅ | ✅ |

**The damage is concentrated exactly where it hurts most.** `COMMON = {"motion": 13, "expression": 21}` —
the floor buckets that carry the entire anti-confound design, where *every label is re-emitted at a
common length so that frame count cannot predict the label* — are **both illegal on LTX**. Ported
naively, v7 loses the one defect v6 identified and shipped without fixing.

### 4.2 Re-deriving the ladder at 8n+1

Truncating each **source** length to the largest legal 8n+1 value:

| Label | Source | Wan top (4n+1) | LTX top (8n+1) | LTX seconds |
|---|---|---|---|---|
| walking | 16 | 13 | **9** | **0.38 s** |
| running | 17 | 17 | 17 | 0.71 s |
| waving | 26 | 25 | 25 | 1.04 s |
| sitting | 28 | 25 | 25 | 1.04 s |
| jumping | 33 | 33 | 33 | 1.38 s |
| happy | 21 | 21 | 17 | 0.71 s |
| surprised | 30 | 29 | 25 | 1.04 s |
| angry | 40 | 37 | 33 | 1.38 s |
| neutral | 60 | 57 | 57 | 2.38 s |
| laughing | 39 | 37 | 33 | 1.38 s |

The floor must sit at or below the shortest label in its kind, so the de-confound holds:

```
COMMON_LTX = {"motion": 9, "expression": 17}
```

**`motion` floor 9 is a real problem, and it is a data problem, not a code one.** Walking's 16
source frames admit exactly one legal LTX rung — 9 frames, **0.38 s**, which will not contain a
walk cycle. Holding the floor at 9 drags every motion label down to a 9-frame rung; raising the
floor to 17 evicts walking from the floor bucket and reintroduces the length↔label confound for
that label.

The clean fix is upstream: **ask the client for longer motion takes (≥2 s per action)**, which is
already what plan §4.5 specifies for T1 (`≥2 s at 24 fps`). Motion was delivered well under it.

### 4.3 What ports cleanly

- **1024² geometry is legal** — 1024/32 = 32, so W,H ÷ 32 holds with no re-render.
  Sequence length at 1024×1024×25 is 32·32·4 = **4,096 tokens**, comfortably under the plan's
  768×1344×49 = 7,056. Training is *cheaper* than the plan assumes.
- **24 fps native**, matching the 2.5 trainer default — no resample, as the plan already noted.
- **Captions are jsonl-only**, so `prep_v7.py --captions-only` re-renders them with no re-encode.
  The LTX caption template (§4.2 of the plan) can be applied for pennies.
- Colour-grounding (plan rule 1) is **already satisfied** — every v7 caption carries
  "a short round blue penguin" / pink for Polly. This was v4's single highest-value labelling fix
  and v7 has it by construction.

### 4.4 ⭐ Video Extension — the trained answer to short clips

§4.2 frames "our clips are too short" as a client-data problem. The trainer offers a second,
parallel attack: **Video Extension is a first-class training mode**, conditioning on `prefix` or
`suffix` with a `temporal_boundary` expressed in *latent* frames.

Train an extension LoRA on the v7 corpus and short source clips stop being a ceiling on *output*
length — the model learns to continue a Pudgy action in-style, rather than needing the client to
shoot longer takes. This is strictly better than the loop-and-cut approach v4 §5.6 measured as
producing 528× median seam motion, and it is complementary to native multishot (§2.3): multishot
handles *cuts between shots*, extension handles *length within a shot*.

It does not remove the client ask in §4.2 — walking still has no full cycle to learn from, and a
model cannot extend an action it never saw complete. It does mean the ask is about **coverage of
the action**, not about delivery length per se.

### 4.5 ⭐ Outpainting — square source, portrait delivery

v7 is **1024×1024 square** (inherited from Wan); the plan's LTX bucket is **768×1344 portrait**.
Today that mismatch means either re-rendering or letterboxing.

**Video Outpainting is also a training mode**, conditioned on `spatial_crop: [y1, x1, y2, x2]` in
pixels: the named region is held clean and excluded from loss, and the model generates beyond it.
Trained on our own alpha-composited material, it converts square masters into portrait deliveries
natively — and the alpha channel means the extension region is pure background, the easiest
possible case. The `In-Outpainting` IC-LoRA (§2.1) is the training-free version of the same idea
and should be tried first.

---

## 5. The coverage gap nobody has written down: no rear views

`PAX_TURNAROUND.mov` is 240 frames and is *called* a turnaround. **It is not a 360° turnaround.**
Sampling 20 frames evenly across the clip, **every single frame shows the face** — eyes and beak
visible throughout. The character rotates profile-left → front → profile-right and back. It never
shows its back.

The nine render angles have the same limit by construction: `FRONT`, three quarter-views per side,
and `SIDE_L/R` span the front hemisphere only.

> **The corpus therefore contains zero rear views of either character**, and Polly has no
> turnaround clip at all.

Consequences, in order of concreteness:

1. **Any generation that turns a character away from camera is unconstrained.** The model invents
   the back — there is no training signal and no reference panel to correct it.
2. **Ingredients sheets will be front-hemisphere only.** The "body turnaround" panel the adapter
   wants can be built (§3.3), but it cannot cover rear angles.
3. **This is a precise, cheap client ask** and it belongs in Round 3: *full 360° turnarounds for
   both characters — including rear and rear-quarter views — plus a Polly turnaround, which does
   not exist in any delivery.*

---

## 6. Recommended sequencing

Ordered by cost-to-information, cheapest first. The first two cost no GPU training at all.

| # | Action | Cost | Answers |
|---|---|---|---|
| 1 | **Load the v4 2.3 LoRAs on the 2.5 transformer** and render the held-out set | inference only | Whether A0 is a gate or a formality (§0). Highest-leverage test in the survey. |
| 2 | **Build an Ingredients reference sheet** from v7 alpha panels; run `ICLoraPipeline` on held-out prompts | inference only | Whether training-free identity beats v4's edge conditioning — and whether identity can be held while motion stays prompt-driven (§3.3). |
| 3 | Add both 2.5 VAEs to `vae_roundtrip.py` — plan's **A1**, unchanged | hours | Decoder choice. Already correctly specified. |
| 4 | **Fix `buckets_for()` to `% 8`** and re-derive `COMMON_LTX`; re-emit v7 for LTX | CPU re-encode | Makes v7 trainable on LTX at all (§4.1). |
| 5 | **Native multishot** vs manual concatenation on a known-bad seam | inference only | Retires plan rule 11 (§2.3). |
| 6 | **`attention_mask` two-identity test** — Pax region and Polly region masked to separate references | inference only | Plan Risk 6. Clean negative result if it fails. First confirm the parameter is exposed (§2.6). |
| 7 | **In-Outpainting IC-LoRA** on a square v7 clip → portrait | inference only | Whether square masters can be delivered portrait without re-render (§4.5). |
| 8 | **Split rank across the two experts** — motion 8–16, expression 32–64, `alpha == rank` | config only | Aligns v7 with LTX's own guidance (§2.5). Must be declared, since it breaks Wan recipe-parity. |
| 9 | **Union Control** on alpha-derived depth + pose + canny vs v4's Canny-only | 1 training run | Whether richer structure beats edges alone. |
| 10 | **Video Extension LoRA** on the v7 corpus | 1 training run | Whether short sources stop capping output length (§4.4). |

**Items 1, 2, 5, 6 and 7 are inference-only, and item 8 is a config change.** The survey's
practical conclusion is that most of the plan's open questions — including its expensive A0 port
gate — can be answered before any LoRA is retrained.

**Adopt the vendor's training skill.** `.claude/skills/train-model/` in the LTX-2 repo encodes
Lightricks' own phase-by-phase procedure and hardware autotune strategy (§2.5). Installing it into
this repo is cheaper than re-deriving the same runbook, and it keeps us aligned with upstream as
2.5 tooling moves.

### New client ask, consolidated

Both items are additive to the standing §4.4 table, and neither was previously written down:

1. **Full 360° turnarounds for Pax and Polly**, rear and rear-quarter views included. Polly has
   none at all. (§5)
2. **Motion takes of ≥2 s per action.** Current motion is 0.67–1.38 s; at LTX's 8× temporal
   compression, walking admits exactly one 9-frame rung. (§4.2)

---

## 7. Sources

- [Lightricks/LTX-2 — repo README](https://github.com/Lightricks/LTX-2)
- [`packages/ltx-trainer/README.md`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/README.md)
  and its docs: [`training-modes`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-modes.md),
  [`dataset-preparation`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md),
  [`configuration-reference`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/configuration-reference.md)
- [`packages/ltx-trainer/AGENTS.md`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/AGENTS.md) — latent geometry, Gemma invalidation
- [`.claude/skills/train-model/references/mode-selector.md`](https://github.com/Lightricks/LTX-2/blob/main/.claude/skills/train-model/references/mode-selector.md) — rank guidance per LoRA type
- [`.claude/skills/train-model/references/hardware-profiles.md`](https://github.com/Lightricks/LTX-2/blob/main/.claude/skills/train-model/references/hardware-profiles.md) — VRAM tiers
- [`packages/ltx-pipelines/docs/conditioning.md`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/conditioning.md)
- [Lightricks/LTX-2.5 — model card](https://huggingface.co/Lightricks/LTX-2.5)
- [IC-LoRA Adapters — LTX Documentation](https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters)
- [IC-LoRA usage guide — LTX Documentation](https://docs.ltx.io/open-source-model/usage-guides/ic-lo-ra)
- [Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
- [Ingredients — multi-identity discussion](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/discussions/3)
- [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)
- [Using LoRA Adapters with LTX-2.5 — LTX Blog](https://ltx.io/blog/using-lora-adapters)
- [LTX-2.3 LoRA Training Guide — WaveSpeed](https://wavespeed.ai/blog/posts/ltx-2-3-lora-training-guide-2026/)
