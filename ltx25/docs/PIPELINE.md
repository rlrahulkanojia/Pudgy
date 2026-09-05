# LTX-2.5, raw to evaluation — an LTX-native pipeline

**Status:** design · **Base:** LTX-2.5-22B · **Source:** `Data/raw/iteration_4` (+ `iteration_3`)
**Written:** 2026-09-05 · **Deliberately independent of** the v7/Wan training approach

> **One line.** Build the corpus for LTX from the **raw ProRes**, not from any prior processed set,
> because LTX's 8× temporal VAE, Gemma-4 encoder and I2V conditioning impose a different shape on
> the data than Wan did. The measured lessons carry; none of the Wan-specific structure does.

---

## 0. What is settled, and what I assumed

**Settled with the client-side owner:**

| Decision | Value |
|---|---|
| Deliverable | individual GIFs, compiled by hand afterwards |
| Duration ceiling | **2.5 s** for now (more data expected later) |
| Looping | **not required** |
| Generation mode | **I2V** — supplied first frame **plus** caption |
| Prompt surface | captions |
| Background | none in the output; **random flat colour grounds** in processing |
| Characters | Pax + Polly. Artifacts (props) are a later phase |
| Compute | 1–2 × 80 GB |
| Priority | quality over schedule |
| Success | closeness to the training footage on character similarity, consistency, caption adherence, plus client expectation |

**Assumed, because the question was left open. Each is cheap to reverse, and each is called out
again at the point it bites:**

1. **A1 — eval scores only on unseen start frames.** Clips generated from a training start frame
   are diagnostics and never count (§8.1). Without this the stated success bar cannot distinguish
   quality from memorisation.
2. **A2 — one LoRA, rank 32, not two.** v7's split exists because Wan2.2 is a mixture of experts;
   LTX has no expert split, so that argument does not transfer (§7).
3. **A3 — the walk cycle is wrapped to 17 frames** rather than emitted at a degenerate 9 (§3.1).
4. **A4 — 1024², `first_frame` probability 0.5** (§3.2, §7).

---

## 1. Gate zero: prove the VAE before touching the data

Everything downstream assumes LTX-2.5 can represent flat 2D art with thin black outlines. That is
unproven, and it is the cheapest possible thing to falsify.

Round-trip real Pax/Polly frames through **both** 2.5 video VAEs (`ltx-2.5-video-vae-bf16` and
`-conv-bf16`) at 1024². Measure PSNR/SSIM, and look specifically for outline softening, flat-region
banding, and the flat-region grid artefact reported against 2.3. The bar to beat is
**PSNR 38.9 / SSIM 0.996**, which CogVideoX's VAE achieved on this same art.

**If both decoders soften the outlines materially, stop.** No amount of data or training recovers
information the VAE discards, and that finding is worth more than a trained LoRA.

---

## 2. Raw intake

Source is `raw/iteration_4/LSLTTT-Project`, ProRes 4444 `yuva444p12le`, **1080×1080, 24 fps, real
12-bit alpha**, structured `<CATEGORY>/<ACTION|EXPRESSION>/<CHAR>/` with nine fixed camera angles
(`FRONT`, `QF1_L/R`, `QF2_L/R`, `QF3_L/R`, `SIDE_L/R`).

**215 files, 210 usable sources, 12 labels** as of 2026-09-05. `CONFUSED` and `CRYING` arrived
mid-design that afternoon, which is why the prep **discovers cells from the directory tree** rather
than hardcoding a label list: the client delivers incrementally and a new cell must not require a
code change. An unknown label is emitted with a caption warning, never silently mis-captioned.

Raw-data properties the pipeline must handle explicitly:

| Property | Handling |
|---|---|
| **16 files have corrupt ProRes streams**, tearing **non-deterministically** (measured 40%/10%/10% per decode) | A pre-pass cannot certify a file. Check tearing **on the frames about to be written**, with retries; drop what will not decode clean. `PAX_MOTION_WALKING_QF1_R` is unrecoverable (11 real frames vs 16 claimed). |
| **Naming defects** — 3 files under `MOTION_WALKING/PAX/` are named `..._WAVING_*` but contain walking, and two of those are byte-identical to each other | Normalise internally by content, never rename under `raw/`. Deduplicate by MD5. |
| **70 of the delivered files are byte-identical re-sends of `iteration_3`** | Deduplicate by MD5 before emission or the corpus is silently weighted toward re-sent cells. |
| **Frozen holds exist in the source** — measured corpus-wide at **37.9% median**, and `NEUTRAL` at **96.6%** | Holds *inside* a clip are kept: they are the client's animation intent. Dead *tails* are trimmed by laddering against active length (§3.1). They also set the eval baseline (§8.3). |
| **Three different angle naming conventions** — `QF1_L` (index first), `QF_L2` (index last, `NEUTRAL` only), `QF_L` (no index), `FR` for front | All normalised. Unparsed names are reported and skipped, never guessed. |
| **A Pax file filed under `POLLY/`** (`EXPRESSIONS/HAPPY/POLLY/PAX_HAPPY_SIDE_R.mov`) | Folder is authoritative; the disagreement is reported. It is also a byte-duplicate, so it drops out anyway. |
| **Truncated streams** — `PAX_MOTION_WALKING_QF1_R` decodes 11f of a claimed 16 | Caught by comparing against the **cell's modal length**, not against a fixed floor. Every angle in a cell renders the same performance, so they must share a frame count. |

**Removed from intake:** `EXPRESSIONS/TURNAROUND/` (deleted from `raw/` 2026-09-05 as faulty).
It was misfiled under expressions and was a front-hemisphere rotation rather than a true 360°.
See §6 for the consequence for start frames.

**Watch the footage before writing a caption.** `sitting` is a seated *idle* (the character is
already seated at frame 0), so "sitting down" would mislabel it. `jumping` is a full
crouch → airborne → land hop.

---

## 3. Clip emission

### 3.1 Frame buckets — the ladder, re-derived for 8× temporal

LTX requires `frames % 8 == 1`: **1, 9, 17, 25, 33, 41, 49, 57, …**. Truncating each raw source to
the largest legal value, with a **common floor of 17**:

| Label | Kind | Raw | Active | Ladder | Note |
|---|---|---|---|---|---|
| walking | motion | 16 | 16 | **[17]** | ⚠️ **cycle wrapped** to the floor |
| running | motion | 17 | 17 | [17] | |
| waving | motion | 26 | 26 | [17, 25] | |
| sitting | motion | 28 | 28 | [17, 25] | |
| jumping | motion | 33 | **29** | [17, 25] | dead tail trimmed |
| angry | expression | 40 | **24** | [17] | dead tail trimmed |
| confused | expression | 21 | 17 | [17] | trimmed, clamped at floor |
| crying | expression | 31 | **24** | [17] | dead tail trimmed |
| happy | expression | 21 | 17 | [17] | trimmed, clamped at floor |
| laughing | expression | 39 | 38 | [17, 25, 33] | |
| neutral | expression | 60 | **27** | [17, 25] | ⚠️ 96.6% frozen; the 57f rung was pure static |
| surprise | expression | 30 | **21** | [17] | dead tail trimmed |

**Ladder against ACTIVE length, not raw length.** This is the single most consequential thing the
baseline measurement changed. `eval_ltx25.py baseline` found the source corpus is **37.9% frozen at
the median**, and `NEUTRAL` is **96.6% frozen** — effectively a still image held for 60 frames.
Laddering against raw length would have emitted **162 clips of static video**, and training on static
video teaches the model to generate static video, which is the frozen-frame failure this project
keeps hitting. So the frozen *tail* is trimmed before rungs are chosen.

Two rules keep that honest:

- **Holds inside a clip are preserved.** Only the tail goes. A mid-clip hold is the client's
  animation intent (`LAUGHING` holds ~9 frames in the middle, deliberately).
- **Trimming never goes below the floor.** `CONFUSED` (13 active) and `HAPPY` (15) would otherwise
  ladder to 9 frames, which is 2 latent frames after 8× compression, and would break the floor
  invariant. They clamp to 17 instead, keeping a few held frames rather than losing the guarantee.

**Why 17 is the floor, and why walking is wrapped.** Walking's 16 raw frames admit only **9** under
8n+1: 0.375 s, and just **2 latent frames** after 8× temporal compression. That is a nearly
degenerate sample, and holding the common floor at 9 would drag every other label down with it.
Walking is a **verified cycle**, and a cycle has no narrative arc to mislabel, so wrapping it to 17
frames manufactures no motion that the client did not shoot. That is the trade: one wrapped cyclic
action, against a floor that would otherwise halve in length.

**Every label appears at 17.** This is not cosmetic. The trainer buckets by frame count, so a label
that occurs at exactly one length makes sequence length a perfect shortcut for that label. The
common floor guarantees one bucket carries all ten labels, so **length predicts nothing**. Assert
it; do not assume it.

### 3.2 Geometry

1080 is not divisible by 32, so the raw frame cannot be used as-is. **Resize to 1024×1024**
(32 × 32 latent), a clean downscale that preserves the square framing and leaves ample headroom
above GIPHY's recommended 480p.

| Bucket | Tokens | Note |
|---|---|---|
| 1024×1024×17 | 3,072 | the floor, and most of the corpus |
| 1024×1024×25 | 4,096 | |
| 1024×1024×33 | 5,120 | |
| 1024×1024×57 | 8,192 | `neutral` only |

All are far below the 7,056 tokens the earlier 768×1344×49 plan assumed, so **1–2 × 80 GB is
comfortable** and gradient checkpointing is a safety margin rather than a necessity.

### 3.3 Alpha, backgrounds and shot size

Video models cannot emit alpha, so the character must be composited onto a ground for training and
matted back out at delivery. The alpha channel makes both ends exact.

- **Random flat colour grounds**, named in the caption so the ground is a *controlled slot* rather
  than a hidden variable. Exclude colours that collide with the palette (Pax blue, Polly pink,
  belly white, feet orange, outline black), so the matte stays clean.
- **Shot-size ladder that zooms OUT, not in.** The client frames the character at **~92% of frame
  height** in every source clip, so there is no headroom to crop inward: cropping gave `medium` and
  `close` a **2 percentage point** difference, which would have meant three captions describing one
  image. Because the ground is synthetic, the ladder instead pads the canvas with flat ground to
  make the character *smaller*. Measured result: **45% / 66% / 90%** character height. Every
  (source, rung) is emitted at all three, so zoom cannot correlate with label by construction.
  This also matters on its own terms: v4 measured that character size in frame drives identity
  quality, so the axis has to be real rather than nominal.
- Both balances are `assert`-and-exit, not warnings. The two confounds they defend against are
  measured, not hypothetical: v5's failure was the model latching onto the single most predictive
  cue available.

**Actual volume:** 210 sources → **954 clips + 216 stills = 1,170 samples**. Fewer than a 4n+1
corpus of the same footage would give, because 8n+1 is coarser and the active-length trim removes
rungs that were mostly static. That is the real cost of LTX's temporal VAE, and it is a
length-diversity cost rather than a content one.

---

## 4. Captions

LTX is documented as extremely prompt-sensitive, and 2.5 encodes with **Gemma 4**. Captions are
rendered programmatically from structured records, never hand-written, so slot vocabulary cannot
drift.

```
<token>, a <colour> penguin. 2D cartoon animation in the Pudgy Penguins style, thick clean
black outlines, flat pastel colors, cel shading; <action phrase>; <expression phrase>;
<camera phrase>; <background phrase>.
```

Rules, each earned from a measured result:

1. **Colour-ground every character, every caption** (`pxngn0, a blue penguin` / `plngn0, a pink
   penguin`). The same conditioning renders Polly **blue** without it. This is the single
   highest-value labelling fix ever measured on this project.
2. **Trigger tokens inline, never `--lora-trigger`.** That flag prepends to *all* captions, which
   would stamp Pax's token onto Polly-only clips.
3. **Style block verbatim-constant** across the corpus.
4. Action clauses are **gerund phrases**; a truncated clip is marked by annotating the *label*
   (`"jump cycle, opening frames only"`), never by rewriting the action.
5. Captions live in the JSONL only, so they can be re-rendered without re-encoding a single frame.

---

## 5. Splits — where this corpus will silently lie to you

⚠️ **The nine camera angles are one performance rendered from nine cameras, not nine takes.**

They are near-duplicates in content. A random clip-level train/holdout split therefore **leaks**:
the holdout clip is the same performance the model trained on, viewed from 40° away, and it will
score beautifully while proving nothing. Every rung of the ladder compounds this, since the rungs
are head-truncations of the same frames.

The split has to be made on an axis that actually separates:

| Axis | Split | What it tests |
|---|---|---|
| **Angle** | hold out one angle (e.g. `SIDE_R`) across all cells | novel viewpoint of a known action |
| **Start frame** | frames that never enter training (§6) | resistance to start-frame memorisation |
| Label | *not* available — holding out a label means it is untrained and untestable | — |

So this corpus can support a **viewpoint** holdout and a **start-frame** holdout, and cannot
support a true novel-action holdout at all. Say that out loud in the report rather than implying a
generalisation claim the data cannot support.

Reserve the holdout **before** preprocessing, seed 42, and remove those entries from the dataset
JSONL. A holdout defined after the fact is not a holdout.

---

## 6. Start frames

I2V means every generation needs a first frame, and where those frames come from decides whether
the evaluation means anything.

- **Production:** a canonical pose library built from the alpha assets, one clean frame per
  character × angle × shot size, matted and composited exactly as training clips are.
- **Evaluation:** frames the model has **never seen**. Because ladder rungs are head-truncations,
  frame 0 is shared across every rung of a source, so a novel start frame must come from a
  held-out angle, a mid-clip time offset, or a different performance entirely.
⚠️ **There is no turnaround footage in the programme.** `EXPRESSIONS/TURNAROUND/PAX_TURNAROUND.mov`
was removed from `raw/` on 2026-09-05 as faulty. It was misfiled under expressions and was never a
true 360°: sampling it evenly across its 240 frames showed the face in **every** frame, so it was a
front-hemisphere rotation only. It had been the one distinct performance available as a novel-pose
source, so its removal narrows the options to:

| Source of an unseen start frame | Strength | Weakness |
|---|---|---|
| **Held-out angle** (e.g. `SIDE_R`, excluded from training) | free, available today | same performance, so it tests viewpoint rather than pose novelty |
| **Mid-clip time offsets** from held-out angles | free, genuinely different poses | still the same performance |
| **Client-rendered fresh frames** | truly novel, and cheap for the client to produce from their project | requires a delivery |

➡️ Use the held-out angle plus mid-clip offsets to start, and **add "a handful of novel first
frames, rendered from your project" to the outstanding client ask** — it is a far smaller request
than new clips and it is the only route to a genuinely unseen pose.

One incidental benefit of the removal: Pax and Polly are now **symmetric**, since neither has
turnaround footage. The evaluation no longer has to caveat that one character had a richer
start-frame pool than the other.

**A verified gap that constrains all of this:** the nine camera angles span the front hemisphere
only (`FRONT` through `SIDE_L/R`), and with the turnaround gone there is nothing else. **The corpus
contains zero rear views of either character.** Any prompt that turns a character away from camera
is unconstrained, and no start frame can fix it.

---

## 7. Preprocessing and training

```bash
# Fresh .precomputed - Gemma 4 embeddings are NOT interchangeable with Gemma 3,
# and stale conditions/*.pt are SKIPPED SILENTLY, not rejected.
uv run accelerate launch --num_processes 2 --mixed_precision bf16 \
  scripts/process_dataset.py dataset.json \
    --resolution-buckets "1024x1024x17;1024x1024x25;1024x1024x33;1024x1024x57" \
    --model-path        models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    --text-encoder-path models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path    models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path    models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --skip-audio --decode
```

`--decode` writes `.precomputed/decoded_videos`. **Eyeball those before spending a GPU-hour.** Paths
in the metadata file resolve **relative to that file's own directory**, which is the most common
preprocessing failure. Do not pass a version flag: version detection is automatic.

Training starts from `configs/i2v_lora.yaml`:

```yaml
lora:
  rank: 32
  alpha: 32                     # vendor guidance: alpha == rank
  target_modules: [attn1.to_k, attn1.to_q, attn1.to_v, attn1.to_out.0,
                   attn2.to_k, attn2.to_q, attn2.to_v, attn2.to_out.0,
                   ff.net.0.proj, ff.net.2]      # documented video-only IC/LoRA set
training_strategy:
  name: flexible
  video:
    is_generated: true
    conditions: [{type: first_frame, probability: 0.5}]
  # audio block OMITTED - silent corpus
optimization:
  learning_rate: 1.0e-4
  enable_gradient_checkpointing: true
validation:
  video_dims: [1024, 1024, 17]  # MUST equal a preprocess bucket
  frame_rate: 24.0
  generate_audio: false         # SETTLED - off. ⚠️ trainer default is TRUE
checkpoints:
  interval: 250
  keep_last_n: -1               # ⚠️ DEFAULT IS 3
```

**Three defaults that fail silently if left alone:** `generate_audio` is **true** by default and
would attempt audio on a silent corpus, so it is **explicitly set false** (settled decision, not a
preference); `keep_last_n` is **3** and would discard early checkpoints,
which is exactly where fidelity peaked in the previous LTX generation; and stale Gemma-3 embeddings
are skipped rather than erroring.

**Rank, and why one LoRA (A2).** The vendor's own mode selector prescribes rank 32–64 for a
character or style LoRA and 8–16 for motion, warning that "high ranks just memorise frame content".
That is a real argument for splitting motion from expression, but it is a *different* argument from
v7's, which rests on Wan's high-noise/low-noise experts. LTX has no expert split, so that structure
should not be imported by default. Train **one rank-32 LoRA first**, measure per-label, and split
only if motion and expression demonstrably interfere. Since schedule is not a constraint, the split
can be experiment two, justified on LTX evidence.

`first_frame` at `probability: 0.5` (A4) yields a T2V/I2V superset from one run. Production always
supplies a frame, but the text-only fallback is free and covers requests the pose library misses.

---

## 8. Evaluation

### 8.1 The stated bar, and the hole in it

The success criterion is closeness to the training footage on identity, consistency and caption
adherence. With supplied start frames that criterion has a hole: **a model that replays the
training clip scores perfectly on all three.** The prior generation measured exactly this failure,
producing 0.8 px of motion from a training start frame against 221–267 px from a novel one.

**Assumption A1, restated as the rule:** score only on clips generated from **start frames the
model has never seen**. Clips from training start frames are run deliberately, as the memorisation
probe below, and never contribute to the score.

### 8.2 Protocol

- Fixed held-out prompt set × **≥ 8 seeds per prompt**. Report the **rate** of each failure mode,
  never a best-of. A single-seed result hides the true failure rate.
- **Checkpoint sweep, not final step** (`keep_last_n: -1`). Fidelity has peaked mid-run before.
- Score **per label**, so a failure reads as "`waving` is weak", not "the model is bad".
- Score **per character**, since Polly has historically been the weaker identity.

### 8.3 What to measure

| Axis | Measure | Note |
|---|---|---|
| **Identity** | human or vision-judge eye-check, on-model yes/no | ⚠️ **No metric sees identity.** A clip previously scored best on every motion metric and was completely off-model. |
| **Style fidelity** | outline weight, flat-fill integrity, palette accuracy | this is where a weak VAE shows up |
| **Caption adherence** | does the named action/expression actually occur | per-label |
| **Motion quality** | frozen %, centroid travel, motion energy | **Baseline measured**: source is 37.9% frozen at the median (p90 65.0), median centroid travel 18.6 px. Per label it ranges from `RUNNING` 0% frozen / 71.6 px to `NEUTRAL` 96.6% frozen / 4.0 px. A naive "frozen is bad" threshold would score the client's own animation as defective. |
| **Memorisation probe** | motion magnitude from a *training* start frame vs a *novel* one | the single most important number in the run |

### 8.4 Gates

| Gate | Pass condition | If it fails |
|---|---|---|
| **G0 VAE** | round-trips Pudgy art with no outline softening or flat-region banding | stop; LTX is the wrong base for this art |
| **G1 Identity** | on-model rate on unseen start frames, ≥ 8 seeds, both characters | localise to a character or label and issue a bounded data ask |
| **G2 Memorisation** | motion from a novel start frame is within the same order as from a training frame | the anti-confound design failed; revisit the ladder before adding data |
| **G3 Adherence** | each of the ten labels separately addressable | name the weak labels |
| **G4 Delivery** | survives matte + GIF palette quantisation at 480p | fix at export, not by retraining |

---

## 9. What this pipeline cannot answer

Stated up front, so no one reads a result it does not support:

1. **No novel-action generalisation claim.** Every label is trained; there is no held-out action.
2. **No rear views exist**, and no turnaround footage remains, so any away-facing generation is
   unconstrained (§6).
3. **Within-cell variety is camera, not performance.** Nine angles of one take is one sample
   observed nine ways, which caps how much identity robustness this corpus can demonstrate.
4. **Two-character shots are untested** and unsupported by this corpus, which is single-character
   throughout.

The bounded asks that would lift 1–3, in priority order: **full 360° turnarounds for both
characters** (rear views; the one front-hemisphere turnaround that existed has been removed as
faulty, so there is now none at all), **a handful of novel first frames** rendered from the
client's project for evaluation, **multiple distinct performances per action**
rather than more camera angles, and **longer motion takes** so the ladder is not floored at 17.

---

## 10. Known upstream issues, and what to actually expect

Sourced from the `Lightricks/LTX-2` tracker, September 2026. Three of these bear directly on
decisions made above.

### 10.1 ⚠️ The start frame will not carry identity on its own

[**#255**](https://github.com/Lightricks/LTX-2/issues/255) asked exactly our question, and the
maintainer's answer is unambiguous:

> "`--image` is first-frame / keyframe conditioning, **not an identity encoder**. … That still will
> not keep identity through camera moves or large motion — that needs an IC-LoRA."

This qualifies the I2V bet in §0. The start frame **anchors the opening**, it does not hold identity
across the clip. Whatever identity survives to the last frame comes from the **LoRA**, which is
precisely what every prior generation of this project measured.

Our case is milder than the issue's: a flat 2D penguin has a rigid silhouette and a four-colour
palette, unlike a photoreal human face; clips are ≤ 2.5 s; and the camera is locked off. But the
mechanism is real, so plan for it:

- Set conditioning strength to **1.0**, not the 0.8 in the issue. Syntax is
  `--image <path> <frame_idx> <strength>`.
- Expect identity to be strongest at frame 0 and to **degrade with clip length and motion
  magnitude**. Score identity on the *last* frame, not the first, or the metric flatters itself.
- The recommended fix upstream is `ICLoraPipeline` with **Ingredients**, "which expects a character
  sheet rather than a single portrait". That is the phase-two path and it is worth reaching for as
  soon as the props work begins.

### 10.2 ⚠️ Thin outlines are the VAE's known weak spot

[**#223**](https://github.com/Lightricks/LTX-2/issues/223) reports flickering on high-frequency
texture, isolates it to a **pure VAE encode → decode round-trip** with no transformer involved, and
is closed by the maintainer as:

> "a known limitation of the current VAE's temporal compression, not a pipeline or sampler bug.
> **LTX-2.5's DiffVAE is a newer decoder; it does not fully remove this class of artifact.**"

Flat 2D cartoon art is exactly high-frequency content (thin black outlines) sitting on
low-frequency content (flat fills). **This is the single largest technical risk to this art style on
this model**, and it is why §1 is gate zero rather than a formality. The gate now has a specific
prediction to test: look for outline shimmer frame-to-frame, not just static softening.

### 10.3 ⚠️ Do not select checkpoints from the trainer's own validation samples

[**#283**](https://github.com/Lightricks/LTX-2/issues/283) is **open**, acknowledged by the
maintainer, and documents three silent train/inference divergences:

| Divergence | Effect |
|---|---|
| **σ schedule shift** derived from real sequence length in training, but fixed at the 4,096-token default in 7 of 8 inference call sites, including the trainer's own validation runner | training and validation sample from different noise distributions |
| **`keyframes_mask`** never set in training, always set at inference, and both 2.5 checkpoints ship `use_keyframes_abs_pos_embedding: True` | affects **keyframe conditioning specifically**, which is our exact mode |
| **Token order** prepended in training, appended at inference | equivalent for the model, but silent when porting logic |

Practical consequence: **render checkpoints through `ltx-pipelines` for evaluation rather than
trusting in-trainer validation output.** The §8 checkpoint sweep must run on the real inference
path.

One piece of luck: the σ divergence scales with sequence length, and our geometry is **3,072 tokens**
against the 4,096 default, so we sit much closer to it than the trainer's default 6,120-token
validation geometry. We are less exposed than most users, not unexposed.

### 10.4 ⭐ An opportunity: add a stills bucket

[**#249**](https://github.com/Lightricks/LTX-2/issues/249), from the maintainer:

> "Image-only character LoRAs still work. … **60–120 images is a reasonable appearance dataset; it
> will not teach motion.**" Mixing stills and video in one run is supported via
> `--resolution-buckets "960x544x1;960x544x49"` with `optimization.batch_size: 1`.

Given §10.1, identity is the weak axis and stills are the documented way to strengthen appearance.
We hold perfect alpha art for both characters at nine angles and three shot sizes, so **a 1024×1024×1
stills bucket is essentially free** and targets exactly the axis the video clips are weakest on.

➡️ Add `1024x1024x1` to the bucket list, emit ~60–120 stills per character, and set
`optimization.batch_size: 1` with `gradient_accumulation_steps` for effective batch. This is the
highest-value change discovered in this pass.

### 10.5 Gotchas that each cost a day if unknown

| Issue | Trap | Avoidance |
|---|---|---|
| [#165](https://github.com/Lightricks/LTX-2/issues/165) | Training from an **fp8 checkpoint outputs pure noise** at first sampling | Train from the **bf16** checkpoint |
| [#296](https://github.com/Lightricks/LTX-2/issues/296), [#253](https://github.com/Lightricks/LTX-2/issues/253) | `--quantization fp8-cast` **crashes with any LoRA on pre-Hopper GPUs** | **A100 is Ampere, pre-Hopper.** Rent **H100** if quantizing, or skip quantization (we have the VRAM headroom) |
| [#284](https://github.com/Lightricks/LTX-2/issues/284) | Single-stage pipelines need W,H divisible by **32**; **two-stage enforce 64** | 1024 satisfies both. Do not "optimise" to a /32-only size later |
| [#301](https://github.com/Lightricks/LTX-2/issues/301) | **DDP validation crashes** on `transformer.num_blocks` when **STG is enabled** | On 2 GPUs, disable STG in validation (`video_stg_scale: 0.0`) |
| [#288](https://github.com/Lightricks/LTX-2/issues/288) | i2v at 1216×832 crashes the DiffVAE decoder with an illegal memory access | Stay on known-good geometry; 1024² is not a reported case but verify at gate zero |
| [#277](https://github.com/Lightricks/LTX-2/issues/277) | DiffVAE AUTO tiling produces **gray tails** at a 2³² element boundary | Watch for it on the longest bucket (57 frames) |
| [#180](https://github.com/Lightricks/LTX-2/issues/180) | Training VRAM is widely reported as heavy | Our 3,072-token geometry is unusually small; this should not bite |

### 10.6 What to expect, honestly

**Likely outcome.** On-model short clips with promptable action and expression, identity strong at
the opening frame and softening toward the end, expression more reliable than motion (a bigger,
slower, more legible signal), and `walking` the weakest label since it is the shortest and the only
wrapped one.

**The two results that would kill the approach**, both cheap to reach:

1. Gate zero shows outline shimmer that survives both decoders (§10.2). LTX is then the wrong base
   for this art, and no amount of data fixes it.
2. The memorisation probe (§8.3) shows motion collapsing from training start frames. The
   anti-confound design failed and the ladder needs rework before any more data is added.

**What this experiment cannot tell you** regardless of outcome: whether the model generalises to
novel actions (no held-out action exists), how it handles two characters in frame (the corpus is
single-character throughout), or anything about rear views (§6).
