# Pudgy Penguins — Training Approach v5 (Wan2.2-A14B, primitives-first)

**Status:** plan · **data-gated** · current intake = **7 clips (Pax / happy only)**
**Base:** Wan2.2-I2V-A14B — the [v2](../v2/Training_Approach_v2.md) stack, Gate G1 PASS (temporal SSIM 0.949, structural stability 0.880, source fidelity 0.905)
**Sibling doc:** [`Training_Approach_v5_Happy_Expression_LoRA.md`](./Training_Approach_v5_Happy_Expression_LoRA.md) — the executable pilot on today's 7 clips. That doc is the *how* for T0; this one is the programme it sits inside.

> The v1/v2 training reports and montages cited throughout were removed from the repo in
> `87b77b2` ("add v5 and clear files"). They remain in git history at `187725f` and in
> Azure `v2-decoupled-identity-motion`.

v1–v4 all trained on **skit-cut footage**: two-second windows carved out of finished
episodes, each mixing scene changes, dialogue, camera moves, backgrounds and (69% of the
time) both characters at once. Every version learned the *look* and then fought the same
family of problems — drift, entanglement, seed lottery, an under-represented Polly.

v5 changes the **data contract**, not the base. We stop training on finished output and
start training on a **defined, enumerated set of primitives** — one action, one character,
one intent per clip — then compose finished video out of them.

> **Two assumptions, stated up front.** (1) "Moments" in the brief is read here as
> **two-character interaction beats** (§4 of the client ask) and is named `moment` in the
> taxonomy. (2) Training resolution is held at v2's **768×1360** so that the effect of the
> data pivot is measurable against a known baseline instead of confounded with a
> resolution change.

---

## 1. What actually changes

| | v1–v4 (skit-cut) | **v5 (primitives-first)** |
|---|---|---|
| Unit of data | a 2 s window of a finished episode | **one primitive**: one action, one character, one intent |
| Label | free-text scene description | **structured caption** over a closed vocabulary |
| Coverage | whatever the skits happened to contain | **a coverage matrix** we can audit and fill |
| Character balance | emergent (59 Pax : 33 Polly solo, 69% two-char) | **enforced by construction** (§7.4) |
| Background | correlated with content | **deliberately varied per primitive** (§9.1) |
| Long video | concatenate clips → hard cuts (v4 §5.6) | **chain FLF2V beats on shared keyframes** (§6, T4) |
| Failure diagnosis | "the clip is bad" | "primitive `motion.wave.polly` is weak" |

The last row is the real argument. Today a bad output is a guess; under v5, every failure
localises to a named primitive with a known clip count, and the fix is a bounded data ask.

## 2. Thesis

**Controllability is a data-schema problem.** A model can only be asked for what the data
taught it to name. v4 proved this twice over, in both directions:

- Adding two words of colour to a prompt turned Polly from blue to pink — because **zero
  of the 33 solo-Polly captions contained "pink"** (v4 §5.3). The signal was absent from
  the labels, not the pixels.
- Edge conditioning fixed species and construction where no inference knob could (v4 §5.2),
  because it supplied structure the captions never carried.

v5 generalises that lesson: **enumerate the operations first, label them consistently,
then train.** Composition into full video (§6, T4) is deferred until the primitives hold,
because a composition of unreliable parts is unreliable in a way you cannot debug.

---

## 3. The primitive taxonomy

Every training clip is assigned **exactly one** primitive ID. This is the closed vocabulary
the caption schema (§4) and the coverage matrix (§7.4) are both built on.

```
<class>.<primitive>.<character>
```

`class` ∈ `{design, motion, expression, moment}` · `character` ∈ `{pax, polly, both}`

### 3.1 `design` — identity anchors (images + short clips)
`design.turn.<char>` — the 8-angle neutral turnaround (front, front-L/R, side-L/R,
back-L/R, back)
`design.sheet.<char>` — expression sheet stills with colour codes

These are the **only** assets that define what "on-model" means. Everything downstream
inherits identity from here. This is the first time the programme has had them — v1–v4
reverse-engineered identity from footage, which is precisely why colour codes drifted.

### 3.2 `motion` — 9 primitives × 2 characters = 18
`walk` · `run` · `turn` · `sit` · `idle` · `jump` · `wave` · `head_turn` · `bounce`

One action, one character, static camera unless the camera itself is the variable.

### 3.3 `expression` — 10 primitives × 2 characters = 20
`neutral` · `happy` · `sad` · `angry` · `surprised` · `scared` · `confused` · `laughing` ·
`crying` · `affectionate`
(`exasperated` appears on the design sheet list; carry it as an 11th if clips arrive.)

**Structure matters as much as the label.** The client spec is right to require
*neutral → expression → hold*, and the hold must be **animated, not a freeze**. That
three-beat arc is what makes an expression a temporal primitive rather than a still, and
it is what lets T4 cut into and out of an expression cleanly.

### 3.4 `moment` — two-character interaction beats
`hug` · `hold_flippers` · `high_five` · `back_to_back` · `piggyback` · `sync_action` ·
`relative_size`

`moment.relative_size.both` is a **calibration primitive**, not a shot: its only job is to
teach that Polly is slightly smaller than Pax. Worth its own ID because that relationship
has never been explicitly taught and cannot be recovered from prompts.

### 3.5 Camera as an orthogonal axis, not a primitive
Zoom (`close`/`medium`/`wide`), angle (`eye`/`high`/`low`), facing (`direct`/`angled`/`profile`)
are **caption slots** (§4), varied *across* clips of the same primitive. They must never
become a primitive of their own, or the model will entangle framing with action.

⚠️ **Camera framing fights identity.** v4 §5.4 measured this: character size in frame
drives identity quality, and small/distant characters produce off-model output. So wide
shots are the framing most likely to go off-model — deliberately keep them a **minority**
of each primitive's clips, and expect the identity rubric to dip on them.

---

## 4. Caption schema

One template, filled from the closed vocabulary. Consistency here is what makes primitives
addressable at inference.

```
<token>, a <colour> penguin. 2d cartoon animation in the Pudgy Penguins style,
thick clean black outlines, flat pastel colors, cel shading;
<primitive phrase>; <expression phrase>; <camera phrase>; <background phrase>.
```

Worked example, `motion.wave.polly`:

```
plngn0, a pink penguin. 2d cartoon animation in the Pudgy Penguins style, thick clean
black outlines, flat pastel colors, cel shading; waving one flipper, gentle bouncy
motion; neutral happy expression; static medium shot, eye level, facing direct;
plain pastel interior wall.
```

**Rules, each earned from a prior version:**

1. **Colour-ground every character, every caption** — `pxngn0, a blue penguin` /
   `plngn0, a pink penguin`. Non-negotiable: v4 §5.3 traced blue-Polly directly to its
   absence. Keep the v4 trigger tokens for continuity.
2. **Describe the variable, not the constant.** Identity detail belongs in the design
   sheets and the token; dense per-clip identity description dilutes the identity signal
   (v2 §4).
3. **One primitive phrase per caption.** If a clip needs two, it is not a primitive —
   reject it at intake.
4. **Always fill the camera and background slots**, even when they are boring. An unfilled
   slot means the model learns that primitive's framing as fixed.
5. **Style block is verbatim-constant** across every caption in the corpus.

The captions are generated from `catalog.json`-style structured records, never hand-written
prose — the template is applied programmatically so slot vocabulary cannot drift.

---

## 5. Model and recipe

Reuse the v2 stack verbatim — it is documented end to end in
[`actions_done.md`](../v2/actions_done.md) §8 and reached Gate G1. Changing base and data
in the same step would make the pivot unmeasurable.

| Knob | Value | Source |
|---|---|---|
| Base | Wan2.2-I2V-A14B, fp16 DiT, Wan2.1 **8×** VAE | v2; 8× proven near-lossless (PSNR 38.9 / SSIM 0.996) |
| Trainer | musubi-tuner 0.3.4, `networks.lora_wan` | all-linear incl. FFN is the default |
| Rank / α | 16 / 32 | v2 §4 (α = 2r) |
| Timestep split | low 0–900 @ 5e-5 · high 900–1000 @ 1e-4 | `train_pudgy_wan_a14b.sh` |
| Flow shift | 5.0 | official I2V |
| Precision | fp16 (forced — Comfy-Org ships fp16-only 14B) | v2 §1 |
| Resolution | 768×1360 | held constant vs v2 |

### 5.1 Mapping the taxonomy onto the two experts

The MoE split is a natural fit for the primitive classes:

| Expert | Timesteps | Trains on | Rationale |
|---|---|---|---|
| **low-noise** | 0–900 | `design`, `expression` | identity, colour, facial detail, texture |
| **high-noise** | 900–1000 | `motion`, `moment` | trajectory, composition, staging |

This is a **hypothesis, not a given.** The repo's own research flagged that the experts
split by noise level, not by semantic concept, and that no published ablation supports the
identity/motion reading (`Wan2.2_Unbiased_Assessment.md` §Corrections). v2's G1 result is
suggestive — the high-noise expert fixed global lighting and background that the low-noise
expert could not — but it is one observation. Expressions in particular are *both*
appearance and motion and may not sit cleanly on either side.

→ **Test it at T1** with a single-expert ablation before committing the full curriculum
(§6). If expressions train better on the high-noise expert, the mapping changes; the plan
does not.

### 5.2 Joint image + video training

v2 §3.2 wanted this and never got the data. The design sheets are **images**, so it is now
possible: an image dataset (turnarounds, expression sheets) alongside the video datasets in
the same musubi config. This is the cheapest known route past a small clip count, and it is
the only mechanism that teaches **cross-angle consistency** — nothing in v1–v4 addressed
back and three-quarter views at all.

> Verify image-dataset support against `musubi-tuner/docs/wan.md` before caching — the
> flags used in this plan's scripts are proven for video datasets only.

### 5.3 Clip length and the VRAM ceiling

Wan wants `4N+1` frames at 16 fps. Motion primitives fit 33 frames (2.06 s); expression
primitives need the full neutral→expression→hold arc, so **49 frames (3.06 s)**.

⚠️ **49 frames at 768×1360 has not been measured and will probably not fit.** v2 measured
the fp16 DiT holding **~67 GB of 80** at 768×1360×33; 49 frames raises the sequence ~1.5×.
Fallback ladder, in order of preference: `--blocks_to_swap 16` (raise toward 39) →
`--fp8_base` → drop the expression bucket to 544×960. **Measure before scheduling the
tier**, and record which rung was used — it changes how expression output compares to
motion output.

---

## 6. The curriculum

Five tiers. Each has a gate; each warm-starts from the previous tier's golden weights, so
later tiers refine rather than relearn.

### T0 — Harness + pilot · *runs on the 8 clips we have now*
Rebuild the v2 env (it lived on non-persistent `/workspace` and is gone), write the
taxonomy and caption schema, QC the 8 Pax/happy clips against §7, cache latents, train a
short low-noise LoRA.

**This is not a usable model.** Eight clips of one expression on one character will
memorise, fast. T0 exists to (a) prove the harness end to end before 400 clips arrive,
(b) measure whether an atomic primitive imprints at all, and (c) produce the intake QC
report that tightens client delivery *before* the bulk lands.

**→ Gate T0:** harness reproduces; `expression.happy.pax` is legibly reproduced at ≥3
seeds versus a no-LoRA control; intake report returned to the client.

### T1 — Identity from the design sheets
Train on `design.*` (images + turnaround clips) only, both characters. First model in the
programme whose identity comes from *canonical reference* rather than inferred from footage.

**→ Gate T1:** on-model Pax **and** Polly at correct colour codes across all 8 angles, from
a text prompt alone, at ≥3 seeds. Run the §5.1 expert ablation here.

### T2 — Motion primitives
Add `motion.*`. Warm-start high-noise from T1.

**→ Gate T2 (controllability):** for each of the 18 `motion.*` IDs, prompting the primitive
produces that action ≥4 of 5 seeds, both characters, without identity loss. Per-primitive
scoring — a mean hides exactly the holes this plan exists to find.

### T3 — Expression primitives
Add `expression.*`. Warm-start low-noise from T2.

**→ Gate T3:** each emotion is legible to a blind rater, distinguishable from its
neighbours (`sad`≠`crying`, `surprised`≠`scared`), and **character-differentiated** — the
client's stated goal that an emotion reads differently on Pax than on Polly.

### T4 — Moments and composition
Add `moment.*`, then compose.

**Composition mechanism — beat chaining.** A shot is a sequence of primitive beats. Generate
each beat with FLF2V, where **beat N's end keyframe is beat N+1's start keyframe**. v2 proved
endpoint pinning bounds drift within a beat; sharing the endpoint extends that bound across
beats. This is the direct answer to v4 §5.6: concatenating independent clips produced hard
cuts and teleporting characters (seam motion up to 528× median), and looping one clip
produced the same 2 s action five times. Shared endpoints give continuity *and* new action.

**→ Gate T4:** a multi-beat shot with both characters, correct relative size, no seam
discontinuity, identity stable across the full duration.

### Tier ↔ data dependency

| Tier | Needs | Have now | Can start |
|---|---|---|---|
| T0 | any clips | ✅ 8 | **now** |
| T1 | `design.*`, both characters | ❌ | on sheet delivery |
| T2 | `motion.*` ≥4/action/char | ❌ | ~50% motion delivery |
| T3 | `expression.*` ≥6/emotion/char | 8 (1 of 20 cells) | ~50% expression delivery |
| T4 | `moment.*` + T2/T3 golden | ❌ | last |

T1 can begin the moment the design sheets land, independent of every clip tier — worth
saying to the client, because it makes the sheets the highest-leverage single delivery.

---

## 7. Data intake

### 7.1 Acceptance spec (per clip)

| Check | Requirement | Why |
|---|---|---|
| Actions | exactly one | otherwise it is not a primitive |
| Characters | one (`motion`/`expression`), two (`moment`) | isolation is the whole point |
| Duration | ≥2.1 s motion, ≥3.1 s expression, 24 fps+ native | 33 / 49 frames at 16 fps |
| Resolution | ≥1080×1920 portrait | downsample only |
| Audio | silent or stripped | v2/v4 both train silent |
| **Speech bubbles / on-screen text** | **none** | v4 §8.1 — renders as gibberish, unfixable |
| Character size | occupies ≥~14% of frame | v4 §5.4 — thin edges → off-model |
| Internal cuts | none | a window straddling a cut teaches discontinuity |
| Background | simple, and **varied across clips of the same primitive** | §9.1 |
| Camera | static unless camera is the declared variable | |

Enforce with an intake script (extend `eval_v4/source_scan.py`, which already scores text,
character fraction and motion) producing a per-clip pass/fail plus reason. Every rejected
clip goes back to the client with its reason — that loop is what makes "one thing at a
time" actually converge.

### 7.2 Volume

| Class | Target | Clips |
|---|---|---|
| `design` | 25–50 images + 4–5 clips per character | ~50–100 img + ~10 clips |
| `motion` | 4–5 × 9 actions × 2 chars | 72–90 |
| `expression` | 10 × 10 emotions × 2 chars | 200 |
| `moment` | 6–8 × 7 interactions | 42–56 |
| **Total** | | **~320–350 clips + ~100 images** |

Comparable in size to v4's 298 windows, but *structured*. Note the shape: expressions are
**~60% of the corpus** under the client's current numbers. If delivery capacity is limited,
that ratio is the first thing to negotiate — 10 per emotion per character is the single
largest ask in the brief.

### 7.3 Delivery order

The client is delivering incrementally, so the order should follow the curriculum, not the
document:

1. **Design sheets, both characters** — unblocks T1, unblocks nothing else if delayed
2. **Polly, everything** — she has been the chronic deficit since v1 (§9.3)
3. **Motion**, breadth first (all 9 actions × 2–3 clips) before depth (4–5 each)
4. **Expression**, `happy`/`sad`/`angry`/`surprised` first — the four most distinguishable
5. **Moments**, `relative_size` first (calibration), then physical-contact beats

Breadth-before-depth in (3) matters: 2 clips × 9 actions is a trainable curriculum with
known-weak cells; 5 clips × 3 actions is a model that cannot walk.

### 7.4 Coverage matrix and balance

Maintain a live `coverage.json`: every primitive ID × character → clip count, held-out
count, delivery status. It is the intake dashboard and the training gate.

**Two hard rules:**
- **No tier trains until every cell in it has ≥2 clips.** A primitive with 1 clip is
  memorised, and it inflates the mean while hiding a hole.
- **Balance is enforced in the config, not hoped for.** Per-character `num_repeats`
  oversampling equalises Pax and Polly, exactly as v4 did (solo-Polly 3×, 33→99). The
  alternative — letting the delivered ratio leak into the model — is the documented cause
  of Polly's fragility across four versions.

**Held-out:** reserve 1 clip per (primitive, character) — 2 for the 10-clip emotion cells —
never trained, used only for gate scoring.

---

## 8. Evaluation

Score the [v2 §5 rubric](../v2/Training_Approach_v2.md#5-evaluation-rubric) (identity, line
and colour, motion robustness, prompt adherence, temporal stability), plus three axes this
plan adds:

| New axis | Question |
|---|---|
| **Primitive controllability** | Asked for primitive P, did we get P? Scored **per ID**, not averaged. |
| **Cross-angle consistency** | Does identity hold across the 8 turnaround angles? (never tested before) |
| **Relative size** | Is Polly consistently slightly smaller than Pax in two-character output? |

**Protocol, non-negotiable:**

- **≥3 seeds for every judgement.** v4 §5.1 found two seed-dependent failure basins
  ("raccoon" wrong-species, "3D claymation") that a single seed hid completely — G1 was
  scored on seed 42, which happened to dodge both. Never select a checkpoint on one seed.
- **Visual verification before any recommendation.** v4's `reunion_walk` scored *best* on
  every motion metric and was completely off-model. Metrics rank; eyes decide.
- **Read ratios beside absolute values.** Flow-normalised metrics inflate on near-static
  clips (a 35× seam ratio was an absolute flow of 0.074 — i.e. fine).
- **Sweep the golden checkpoint, don't assume.** v1's golden was mid-run (2000 of 2500);
  v2's was the final epoch. Sweep the last ~40% at ≥3 seeds.
- **Note the inference precision.** All v2 evidence is fp8_scaled because full-res fp16
  inference OOMs at load — montages **understate** true quality. Keep that caveat attached
  to every v5 artifact too.

---

## 9. Risks

### 9.1 Primitive–background entanglement ⚠️ *the one that will bite*
If every `happy` clip shares one background, the LoRA learns the background, not the
emotion. Isolated primitives on simple backgrounds make this *more* likely than skit data
did, because skits at least varied their scenes incidentally.

**Mitigation:** require **≥2 distinct backgrounds and ≥2 camera framings per primitive per
character** at intake — this is why the client's "variable background preferred" note
matters more than it looks. Audit it: compute background diversity per primitive before
training and refuse to train a cell that fails. Probe it: prompt a trained primitive with
an unseen background and check the action survives.

### 9.2 Small per-cell counts
4–5 clips per motion primitive is thin, and overfitting shows up as background memorisation
and pose-freezing rather than as a loss curve. Mitigations: modest rank (16), the
design-sheet image data as a regulariser, held-out clips per cell, early checkpoint sweeps.

### 9.3 Polly, again
33 solo-Polly clips across four versions produced identity fragility that prompt-grounding
only papered over. v5's structural fix is the coverage matrix (§7.4) plus repeat balancing —
but it only works if Polly's clips actually arrive. **Treat a Polly-shaped hole in delivery
as a schedule blocker, not a gap to work around.**

### 9.4 Expression legibility on a stylised 2D face
Emotions on a flat cartoon penguin ride on small features — brow, beak, cheek, eye shape.
Wide framing may not carry them at all, and §3.5's identity/framing tension bites hardest
here. Expect expression primitives to need close/medium framing more than motion does; if
T3 stalls, framing is the first variable to test.

### 9.5 Photorealistic bias — carried, but downgraded
The research called this the #1 risk (Wan "actively fights flat-color cartoon aesthetics").
It **did not materialise** in v2 at any epoch: flat pastel and thick outlines survived
intact. Keep the negative prompt (`realistic, 3d render, soft shading, gradients`) and keep
watching, but do not architect around it.

### 9.6 Composition may expose primitive seams
T4 assumes clean primitives compose. They may not — a wave learned in isolation may not
blend into a walk. If beat chaining shows seams, the fallback is **transition primitives**
(explicit A→B beats) as a new taxonomy class, which is additive and does not invalidate
T0–T3.

---

## 10. Immediate actions — what runs on the 8 clips

1. **Rebuild the v2 environment** from [`actions_done.md`](../v2/actions_done.md) §8 —
   trainer, `.venv-wan`, ~65 GB of weights. It lived on non-persistent storage and is gone.
   *(Watch the two recorded gotchas: positional filenames for `hf download`, and the
   Wan2.1 8× VAE — not `Wan2.2_VAE.pth`.)*
2. **Write the taxonomy + caption schema** as data: `primitives.json` (the closed
   vocabulary) and `caption_template.py` (slot-filling). These outlive every tier.
3. **QC the 8 clips** against §7.1 → the first intake report. Expect some to fail; the
   failures are the most useful output, because they tighten delivery before the bulk lands.
4. **Caption and cache** the survivors as `expression.happy.pax`.
5. **T0 pilot:** short low-noise LoRA, rank 16, checkpoints every 2 epochs.
6. **Evaluate at ≥3 seeds vs a no-LoRA control** — does prompting `happy` produce a legible
   happy beat the base model does not? Sweep for the memorisation point.
7. **Report back** with: the intake QC results, a sample compliant vs non-compliant clip,
   and the §7.3 delivery order.

Deliverables from T0 are a working harness, a schema, and a client feedback loop — not a
model. That is the point of taking one thing at a time.

---

## 11. Open questions

- **Expression clip length** — is 3 s enough for neutral→expression→hold, or does the hold
  need longer? Affects the frame bucket and therefore the VRAM ladder (§5.3). Decide from
  the 8 clips we have.
- **Do the design sheets carry usable colour codes?** If yes, they can be asserted as a
  post-hoc palette check on generated output — a cheap, objective identity metric the
  programme has never had.
- **Where does `expression` sit on the expert split?** §5.1 — resolve empirically at T1.
- **Is `exasperated` in or out?** It is on the design-sheet list but not the clip list.
- **Does the client's existing 249-clip `iteration_2_v4` corpus get folded back in?** It is
  skit-cut and violates the v5 contract, but it is real data. Recommendation: keep it out of
  T0–T3 so primitives stay clean, and revisit at T4 as composition/style ballast only.

---

*Prior art: [v2](../v2/Training_Approach_v2.md) (base, recipe, FLF2V, rubric) ·
[v4 README](../v4/README.md) (failure basins, colour grounding, character size, long-form) ·
[FINDINGS](../FINDINGS.md) (VAE gate, v1 baseline) ·
[base_model_exploration](../base_model_exploration.md) (Gate G0).*
