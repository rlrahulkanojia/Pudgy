# v9c-phone: what went wrong, why, and the plan for the next run (v9c2) — revision 2

Research on the findings of [`REPORT_v9c_phone.md`](../../docs/training_reports/v9/REPORT_v9c_phone.md).
Revision 2 (2026-10-26, late) re-checked every claim of revision 1 against the data. Twelve
statements were wrong or overstated; they are listed first so nothing is silently changed.
Companion: [`RESEARCH_asset_memory.md`](RESEARCH_asset_memory.md) (store the phone as an
embedding).

---

## Corrections to revision 1

| # | Revision 1 said | What the data shows | How it was checked |
|---|---|---|---|
| C1 | Winner ≈ base: mean pixel difference ~1/255 | That was the **whole-frame** average. Inside the region that moves (phone + flippers, 2.5–9% of the frame) the difference is **11–25/255**. v9c changes the phone area; it does **not** change the defect outcome | per-pixel diff restricted to the moving region, 8 clips |
| C2 | Phone flips in 7–8 of 8 clips; becomes a pen or a generic iPhone | **5 of 8 in both models**: FRONT **4/4**, QF2_R 1/4. The "pen" is the phone **edge-on, mid-turn**. The defect is the phone **turning to show its screen**, often drawn with a call screen (green/red buttons) or "iFi…" lettering; the tuft is sometimes kept (same object turned), sometimes lost | full-resolution crops of every 5th frame; then the ID1 detector (§5), which agrees on 16/16 |
| C3 | Layout is decided above t = 0.9 by the high-noise expert, so v9c couldn't act | **Unverified hypothesis.** At the switch (t = 0.9, x_t = 0.1·x₀ + 0.9·ε) the SNR is (0.1/0.9)² ≈ 0.012, so much is still open, and C1 shows v9c does change the phone region | E1 now tests it directly (x₀ probe) |
| C4 | v9c's gain grows with noise (×0.77 at t = 0.65 → ×0.52 at 0.85), so there is signal above 0.9 | **Confounded**: shift-5 timestep sampling puts many more training samples near t = 0.9 than at 0.65, so a larger gain there is expected regardless | sampling density of `--timestep_sampling shift --discrete_flow_shift 5` |
| C5 | Motion regressed because 80% of the data is near-static ("hold still") | **Not supported.** Per-clip regression is uncorrelated with motion energy (Spearman ρ = −0.01, p = 0.94, n = 55). What is supported: the **"sitting" caption absorbs other motions** (§3), plus a uniform +2–4% loss drift | motion energy of each eval clip vs its loss ratio; competitor-gap analysis |
| C6 | "The gradient into the text path averages to zero" | Imprecise. Correct: the caption adds no information beyond the image, so **nothing favours the text route**; the solution found ignores the text (measured) | — |
| C7 | The binding check "measures nothing" | The check *can* detect binding (it will for an asset embedding); it just **cannot move with this data** | — |
| C8 | Cross-attention LoRA (k, v, q, o) gives "binding by construction" | **q and o act on every prompt.** k, v act through every token, including the shared style prefix, whose contextual T5 vectors also carry "phone" information. Only an asset embedding gives a hard guarantee | Wan block structure; T5 is bidirectional |
| C9 | Preservation loss leaves "only the phone-specific change" | Only **on the replay distribution**, and only as λ → ∞. In expectation it equals real-data replay wherever the base fits replay. Its real advantages: a zero-variance target, the base (not the data) as the anchor, and a free choice of anchor inputs | E‖v−u‖² = E‖v−E[u\|x_t]‖² + Var(u\|x_t) |
| C10 | Visual metric "to write", named V1 | Now **ID1**: implemented and validated (§5). Renamed, because V0–V2 are the embedding validation stages | `phone_id_v9.py` |
| C11 | Experiments at 21 frames (~8 min per clip) | 2 of 5 flips start **after frame 21** (onsets 2, 5, 13, 25, 27). Use 21 frames only for a fast screen on the early-flipping starts; confirm at 41 frames (~16.5 min per clip) | ID1 per-frame onset |
| C12 | E1 arm: timestep boundary 1.0 | **Confounded**: the low-noise expert was never trained above t = 0.9, so running it there tests something else. Replaced by an x₀ probe and boundary 0.95 | Wan2.2 expert ranges |

---

## 0. Summary (revised)

| # | Finding | Root cause (status) | Fix | Client? |
|---|---|---|---|---|
| F1 | 99.6% of the phone-loss gain is caption-independent; text sensitivity on phone clips fell to **×0.30–0.68** | The caption adds no information when the phone is always in frame 0 (verified) | **Asset embedding** (hard guarantee, being validated); entrance clips for text-called phones | A1 |
| F2 | The phone **turns round to show its screen**: 5/8 clips, **FRONT 4/4**, identical in base and v9c | The words "phone … black screen" push a phone that faces the camera toward its screen side; the drawn call screens and "iFi…" lettering show the text is rendered literally (strong evidence). *Where* in the denoising it happens is open (C3) | Inference-only first: negative prompt, lower CFG, rewritten phrase, FLF; then the embedding; high-noise training only if E1 points there | A2, A6 |
| F3 | Motion recognition regressed (held-out margin −0.0074, 95% CI [−0.012, −0.003]) | The **sitting caption absorbs other motions** (phone-sitting reuses the v8 sitting wording, 190 : 7), plus a mild drift (verified, small n) | Embedding route avoids it entirely; otherwise sitting anchors + preservation | A4 |
| F4 | The update is small and plateaus by step 125 | It changes the phone area (C1) but not the defect, which has another cause (F2) | Fix F2's cause; 150 steps max | — |
| F5 | The gate passed a run with no visible improvement and missed F3 | Averaged bars, no visual metric, one seed | New gate (§5): paired CIs, per-t ratios, text-attributable gain, **ID1**, 3 seeds | — |

---

## 1. F1: the phone was learned through the image, not the name

**Evidence.** For each phone clip, the denoising loss *E* under 5 captions (the registry phrase,
"a phone", "a small red book", "a white cup", "a black TV remote control"), v8merged → v9c@200:
the true-caption loss fell ×0.524, and the **text-attributable part** (Δlog *E*(true) − mean
Δlog *E*(distractors)) is **−0.0023 ± 0.0047**, i.e. 0.4% of the gain and zero within error.
Text sensitivity, measured as the spread of *E* across a clip's candidate captions (std/mean), changed
as follows:

| clips | v9c@200 vs base | 95% CI | clips lower |
|---|---|---|---|
| phone, held out | **×0.68** | [0.61, 0.73] | 4/4 |
| phone, seen | **×0.30** | [0.14, 0.61] | 4/4 |
| expression, held out | ×0.95 | [0.91, 0.99] | 12/16 |
| motion, held out | ×0.99 | [0.81, 1.19] | 6/12 |

**Why.** The flow-matching optimum is the conditional mean E[v | x_t, img, c]. The caption can only
matter if I(x₀ ; c | x_t, img) > 0. In v9c every phone clip has the phone in frame 0 and the
same phrase, so I(x₀ ; c | img) = 0. The data contains no pressure toward the text route, and the
solution found didn't take it (C6). It also became less text-sensitive on phone clips, and
slightly on expressions.

**Fixes.**
1. **Asset embedding** ([`RESEARCH_asset_memory.md`](RESEARCH_asset_memory.md)). The phrase
   rows become a stored, trainable matrix E. With the network frozen, **every** change goes through
   the name, and prompts without the name are bit-identical to the base. Being validated now (§4, V0–V2).
2. **Entrance clips** (A1): the only data where I(x₀ ; c | img) > 0, so the only way to teach
   "the text calls the phone".
3. A cross-attention k/v LoRA is a soft version of 1 (C8). Use it only as stage 2, with preservation.

---

## 2. F2: the phone turns round to show its screen

**Evidence** (full-resolution crops plus the ID1 detector, §5):

| start | angle | base | v9c@200 | onset frame |
|---|---|---|---|---|
| Pax sitting | FRONT | turns: edge-on, then screen out, tuft lost | same | 13 |
| Pax standing | FRONT | screen out (tuft kept) | same | 2 |
| Polly sitting | FRONT | turns; call screen with green/red buttons | same | 27 |
| Polly standing | FRONT | screen out with a photo on it | same | 5 |
| Pax standing | QF2_R | turns late; call screen, "iFi…" lettering | same | 25 |
| Pax / Polly sitting, Polly standing | QF2_R | phone keeps its back to the viewer | same | — |

- In **every training frame** the phone shows its grey back; the screen faces the character.
- With no phone in frame 0 (`textcall`), the base draws a generic smartphone with its screen
  toward the viewer.
- **The frontal view fails every time**, on clips whose start frames come from the **trained** FRONT
  angle. A phone that squarely faces the camera, described as "a phone … with a black screen", is
  pulled toward the base model's picture of a phone facing the viewer, which is its screen. The
  call UIs and the "iFi…" lettering are the text being drawn literally.

**Where it happens is open (C3).** E1 measures it instead of assuming.

**Fixes, cheapest first** (all measurable with ID1):
1. **Negative prompt** (inference): add "phone screen, smartphone display, call screen, text,
   letters, buttons" to the negative prompt. With classifier-free guidance,
   v = v_neg + s·(v_cond − v_neg), so this pushes away from exactly the drawn content.
2. **Lower CFG** (inference): s = 5 amplifies the text direction 5×. In I2V the image already carries
   the object, so try s = 3.
3. **Rewrite the phrase**: describe what is visible ("seen from the back, three camera lenses and a
   little tuft on top") and drop "black screen". It's a registry change, used verbatim (Assets_v9 §2).
4. **FLF** with the phone frame at both ends (plan default for held props, Assets_v9 §6).
5. **Asset embedding**: the name stops meaning "phone with a black screen". Train it through both
   experts ([`RESEARCH_asset_memory.md`](RESEARCH_asset_memory.md) §3).
6. **High-noise LoRA**: only if E1 shows the turn is set before t = 0.9 *and* 1–5 don't fix it.

---

## 3. F3: motion regressed through the sitting caption

**Evidence** (paired per clip, 5,000-sample bootstrap):

| split / kind | Δmargin v9c@200 − base | 95% CI | clips worse |
|---|---|---|---|
| held-out motion (12) | **−0.0074** | [−0.0121, −0.0026] | 9/12 |
| seen expression (16) | **−0.0088** | [−0.0156, −0.0029] | 12/16 |
| held-out expression (16) | +0.0020 | [−0.0017, +0.0056] | 8/16 |
| seen motion (11) | −0.0024 | [−0.0078, +0.0022] | 4/11 |

Change in each competitor's gap to the true caption (negative = the competitor got closer).
The **sitting** caption pulls hardest: on average −0.0078, with −0.014 on running clips and
−0.006 on walking, waving and jumping clips. Sitting clips themselves lost their lead to walking
(−0.0145) and running (−0.0085), and their rank went 1.00 → 1.43.

**Mechanism.** The phone-sitting caption repeats the v8 sitting phrase verbatim ("sitting on the
ground with its legs stretched out in front") and outnumbers the v8 sitting clips **190 : 7**. That
phrase now predicts phone-clip content, so it explains other motions better than it should.
Separately, every v8 clip's loss rose by +2–4%, at all motion levels (C5). Each label has
n = 4–7 clips with 2 noise draws, so the pattern is consistent but noisy.

**Fixes.**
1. **The embedding route avoids this by construction**: with the network frozen, v8 prompts are
   unchanged.
2. If a LoRA stage is used, **anchor the shared wording**: v8 sitting clips with their v8 caption at
   ≥ 1 per 4 phone-sitting samples (`num_repeats`), 35% replay overall, and a preservation term on
   replay (C9 for what it does and doesn't guarantee).
3. Measure it with bar **R1** (paired margin CI), not an averaged loss ratio.

---

## 4. Experiments, in order (GPU free since 18:47 UTC)

**Embedding validation (running).**

| stage | test | result |
|---|---|---|
| V0 | Span found in 380/380 phone captions and 0 others; cache rows = tokens; splice exact | ✅ PASS |
| V1a | `offset` E = 0 reproduces the base | ✅ PASS: 0.0 difference on all 40 clip × caption losses |
| V1a | `replace` from the mean phrase rows (22% row distance) | ✅ ×1.0002 mean, max 0.4%: functionally the base |
| V1b | Embedding-only training, 40 steps, short clips | ✅ |
| V1c | Tier 1 with E@40 | seen clips strong, held-out ×0.99: overfitting at 40 steps |
| V2 | Generation with E, scored by ID1 | 5/6 vs 5/6 (no effect; smoke E) |
| **E5** | Both experts, new wording, 150 steps | **ID1 0/24 vs Rb 1/24; held-out phone loss ×0.899; name binding 3/4 held-out; P1/P2 PASS** (job 015, see RESEARCH_asset_memory) |
| **Pivotal** | Cross-attention k/v LoRA (80 modules, rank 16) on frozen E5, both experts, preservation ×1 on 37% replay, 150 steps | **Gate PASS at both noise levels**. Low-noise: R1 Δmargin ±0.0001, R2 ×1.000, P1 ×0.890 (E5 ×0.899), P2 −0.186 CI [−0.288, −0.084]. High-noise: R2 ×1.000. v8 captions move by a median 0.02–0.04%; the phone gains ~1% over E5 (held-out = seen ×0.990, no overfit gap). **ID1 0/24** (= E5). Generated quality ≈ E5 on every paired metric (back-face min/steadiness, motion, drift, jumps: ~12/12 wins); flicker 20/4 for pivotal but tiny (0.026 vs 0.027) |
| **Decision (2026-10-30)** | Phone setup = **v8merged + new registry wording + E5 embedding** (step 150, offset; both experts) | E5 and pivotal are equivalent in generation, and E5 is simpler: no LoRA, a 25×4096 file, v8 prompts unchanged *by construction*. The pivotal LoRA is kept (Azure `v9c2/weights/pudgy-v9c2-pivotal`) for when more varied phone footage arrives (A1) |

V1a also shows a risk: a 22% change of the phrase rows moves phone-clip loss by only 0.4% at
t ∈ {0.65, 0.85}. The frozen model is barely text-sensitive on phone clips there, so V1c may show
a small effect even if the visual effect is real. **V2 decides.**

**Inference experiments (no training).**
- **Screen tier:** the two standing-FRONT starts (onsets at frames 2 and 5) × 3 seeds, 21 frames,
  ~7 min per clip, so ~42 min per arm. A control arm must reproduce the flip at 21 frames; if it
  doesn't, the screen tier is invalid and everything runs at 41 frames.
- **Confirm tier:** all 8 starts × 3 seeds at 41 frames (~6.6 h per arm), for arms that pass the screen.

| id | question | arm(s), all on v9c@200 unless noted |
|---|---|---|
| E0 | control | default settings (also on the base, once, to confirm the base = v9c defect rate holds per seed) |
| E1a | Is the turn already set when the low-noise expert takes over? | **x₀ probe**: decode x̂₀ = x_t − t·v at step 9 (the switch) for the E0 clips; ID1 on x̂₀. A ~20-line patch in the sampler loop |
| E1b | Does shortening the high-noise phase help? | boundary 0.95 (6 high-noise steps instead of 9) |
| E2a | Does the text cause it? | negative prompt (screen / display / call screen / text / buttons) |
| E2b | | CFG 3.0 instead of 5.0 |
| E2c | | phrase without "black screen", visible-side wording |
| E2d | Do both-end references hold it? | FLF, phone frame at both ends |
| E3 | Is v9c's direction useful? | LoRA multiplier ×2 (lower priority: v9c didn't change the defect at ×1) |

**Significance with the screen tier.** On standing FRONT the base defect rate is 2/2 (onset ≤ 5), so
6 control clips should all fail. An arm must fix ≥ 5 of 6 for an exact McNemar p ≤ 0.0625 (6 of 6:
p = 0.031). The screen tier finds large effects only. The confirm tier (24 clips) detects 0.6 → 0.2
at α = 0.05, power 0.8 (needs ~17–22 paired clips, depending on how often both models fail together).

---

## 5. New gate for v9c2 (fixed before the run)

| id | check | bar | v9c@200 would |
|---|---|---|---|
| R1 | Per kind, held out: paired Δmargin vs base, 5,000-sample bootstrap | 95% CI lower bound ≥ −0.003 **and** top-1 accuracy down by ≤ 1 clip | **fail** (motion: [−0.012, −0.003], −2 clips) |
| R2 | v8-label loss ratio **at each t** | ≤ 1.05 at every t | **fail** (motion ×1.079 at t = 0.85) |
| P1 | Held-out phone loss vs base | < 1.00 | pass (0.613) |
| P2 | Text-attributable gain on phone clips | < 0, CI excluding 0 | **fail** (−0.0023 ± 0.0047) |
| ID1 | **Phone keeps its back to the viewer** (`eval_v9/phone_id_v9.py`): per frame, pixels of the phone's back colour (pale blue-grey; the shadow is neutral, characters and backgrounds saturated or white). Defect = back face < 35% of its frame-0 area for ≥ 4 consecutive frames | Defect rate below base, exact McNemar p < 0.05 on the 24-clip confirm tier | **fail** (5/8 = base) |

**ID1 is validated:** **0 / 380** false alarms on the real phone clips (lowest back-face ratio 0.68,
from the flipper covering the phone while tapping), and **16 / 16** agreement with the
full-resolution manual labels (defects ≤ 0.11, healthy ≥ 0.64). It doesn't apply to clips without a
phone in frame 0 (`textcall`, `leak`).

Tier-2 `leak` (done): no phone appears in v8 prompts for either model (CLIP phone score 0.25 base,
0.23 v9c).

---

## 6. v9c2: decision tree

1. **If an inference fix (E2a–d) removes the turn**, it becomes the inference default, plus a
   registry phrase change for E2c, with **no training needed for F2**.
2. **Phone stage = asset embedding** if V2 shows it lowers ID1 defects. That gives zero v8 regression
   by construction (no R1/R2 risk), and P2 is satisfied by construction.
   - Train E through **both experts** (it feeds both at inference). This needs a ~30-line patch:
     musubi's dual-expert mode merges `--base_weights` into only one expert. v8merged must go into
     the low-noise expert, and the golden LoRA into the high-noise one.
3. **Only if E is not enough**: stage 2, a cross-attention k/v LoRA with E frozen, sitting anchors,
   35% replay, preservation λ = 1, 150 steps, gated by R1/R2/P1/P2/ID1.
4. **High-noise LoRA**: only if E1a shows the turn is already set at the switch and 1–3 fail.
5. Entrance clips (A1) and layered exports (A3) feed the next pass.

---

## 6b. Data pipeline for v9c2 (2026-10-27)

- **Captions recompiled** from the updated registry (`finetune/wan/recaption_v9.py`): 380 phone captions in
  `/workspace/data_v9c2/`, each checked two ways (prep's own compiler = old caption with exactly the two
  text changes). The 95 replay captions are unchanged. Same clips and latents as v9c.
- **Text cache rebuilt on CPU** (`/workspace/wan_cache/latents_v9c2`, then span-annotated in `latents_v9c2_emb`,
  25-token phrase, V0 pass). This CPU has no native bf16: emulated-bf16 T5 drifted 3–7% from the GPU cache, so
  the cache is computed in **fp32** and stored in bf16 (same file format). New files equal a fresh fp32 encode to
  1.7×10⁻³ (bf16 storage).
- **Finding: the original GPU-bf16 text caches have occasional large per-token errors.** Against fp32, the median
  difference is 2.6%, but some captions reach 16%, concentrated in a few rows that are off by up to 3.8× their norm
  (e.g. `polly_happy_QF2_L__peach_f21`, rows 46/48/60). v9c (and likely v8) trained with these. Generation encodes
  prompts with `--fp8_t5`, coarser still. **Tested (job 020, 2026-10-29):** old wording on v9c@200, T5 in fp32 vs fp8:
  13/24 vs 12/24 flips, 23/24 identical outcomes (McNemar p = 1.0). **Text-encoder precision doesn't affect the
  phone's turn**: the words do, not the encoding noise. Generation keeps `--fp8_t5`, and the fp32-trained E5 vs fp8
  inference mismatch is immaterial for ID1.

---

## 6c. Introspection (2026-10-28): what was re-checked, what was fixed, what remains open

**Verified against data**
- **The wording fix:** confirm tier, 8 starts × 3 seeds, paired on seed and start. Old → new wording: **11 fixed, 0 broken**,
  exact McNemar **p = 9.8×10⁻⁴**, on both v9c@200 and v8merged. Causal: only the text changed.
  - Not an artifact of the metric: the frozen-frame fraction is unchanged (0.013 → 0.014).
  - Motion energy is 18% lower with the new wording, and part of the old wording's motion *is* the flip.
  - The largest single-frame jumps (up to 22× the median) are pre-existing: a last-frame scale shift on Pax QF2_R that also
    appears with the old wording, and a pose turn on Polly QF2_R where the phone keeps its back.
- **The probe maths:** x̂₀ = x_t − σ·v is exact for flow matching (x_t = (1−σ)x₀ + σε, v = ε − x₀). With CFG it's the guided
  estimate, i.e. the one that drives sampling. It agreed with the final clip in 12/12 at the 0.9 handover and 5/6 at 0.95,
  so the turn settles between t = 0.95 and 0.9.
- **The two-expert merge** (CPU, real weights): exact to fp16 storage. The expert routing in the smoke test (4 of 12 samples
  high-noise) matches the predicted P(t ≥ 0.9) ≈ 0.28.

**Found and fixed before running**
1. **The preservation loss would have broken block swap.** In training mode the offloader moves blocks during a forward pass
   and relies on the *backward* pass to move them back. The extra no-grad base pass has no backward, so the next pass would have
   started with blocks in the wrong place. Fix: the base pass now runs between `switch_block_swap_for_inference()` and
   `switch_block_swap_for_training()`, musubi's own switch around sample generation. Each call drains in-flight transfers and
   resets block placement.
2. **The gate didn't score the high-noise expert.** The pivotal LoRA changes it, and that's where layout and motion are decided.
   Fix: `dcls_asset_v9.py --expert high` (t 0.95/0.92) and an R2 gate on it. R1 is reported only there, since label margins at
   t ≥ 0.9 are weak by construction.
3. **R2 pooled expression and motion,** hiding the motion regression (×1.041 pooled vs ×1.079 motion). Now it's per kind.
4. **The percentile bootstrap is too narrow at n = 4–16** (lenient R1/P2). Now the wider of the bootstrap and the Student-t interval.
5. **There was no held-out checkpoint selection for E5,** although V1c showed overfitting (seen margin +0.136, held-out ×0.99).
   New job 015 scores every E5 checkpoint (held-out and seen) and selects on held-out phone loss. It's used only if that's < 1.00.
6. **The T5-precision arm** compared fp8 with bf16, which itself has per-token errors up to 3.8×. Now it compares fp8 with
   fp32 (`--t5-fp32`), the accurate reference and the precision of the v9c2 training cache.
7. **The E5 smoke failure** (replay clips in the cache dir → a step with no embedding gradient) was caught by the smoke gate
   and fixed with phone-only cache dirs.
8. **The classifier index held stale phone captions** (compiled 2026-10-24, before the wording change). E5's first Tier 1
   swapped E into 0 captions and returned the base's numbers exactly. `dcls_asset_v9.py` now recompiles phone candidates
   from the live registry with the index builder's own call. It checks each one equals the stored caption with only the
   known revisions applied (8/8), and asserts the splice count equals the phone-clip count.

**Open caveats (not fixable without more data or compute)**
- **ID1 only measures orientation and presence.** It doesn't check the tuft, the lenses or motion quality, so a visual review
  of each winner is still needed. A crop-similarity identity metric would be the next tool.
- **Held-out phone evidence is 4 clips (QF2_R),** so the P1 and checkpoint selection are noisy. More held-out phone footage
  would fix it (client asks A1/A3).
- **The pivotal LoRA is shared by both experts** (musubi's two-expert mode). That's valid parameter sharing, but separate
  per-expert LoRAs would be the more flexible design if this one underfits.
- **E5 trains on fp32-computed text, and generation encodes in fp8.** If job 020 shows T5 precision matters, the E5 and pivotal
  ID1 arms should be re-run with `--t5-fp32`.
- **Text-called phones** (the phone appearing without being in frame 0) remain impossible to train without entrance clips (A1).

---

## 7. Client asks (unchanged)

| id | ask | why |
|---|---|---|
| **A1** | Phone pick-up / take-out / put-away animations from empty flippers (both characters, sitting and standing, FRONT + ¾), plus the same start pose with no phone event | The only data where the caption decides whether a phone appears (F1) |
| **A2** | Phone turnaround sheet: back, screen side, side, ¾, with the screen state shown | Defines every visible face; reference images |
| **A3** | Layered exports of the 32 phone clips (phone on its own alpha layer) | Exact phone masks for ID1 and an optional phone-weighted loss; matched pairs with and without the phone |
| **A4** | More sitting footage, re-exports of the sitting clips with missing frames, standing idle | The sitting caption is where motion regressed (F3) |
| A5 | 11 Polly phone clips + 2 old files (already requested) | Full-length clips |
| **A6** | Should the screen ever face the viewer? If yes, what's on it? | Decides whether the turn is ever correct; the model is currently inventing call screens |

---

## Appendix: artefacts

- `finetune/wan/eval_v9/phone_id_v9.py`: ID1. Validation outputs in
  `/workspace/research_v9/phone_id_train.txt`; crops in `/workspace/research_v9/crops/`.
- Paired statistics, text sensitivity, competitor gaps: inline Python over
  `/workspace/eval_v9/runs/pudgy-v9c-phone-lownoise/dcls/*_k2.json` (`rows[].E`, `margin`,
  `loss_true`), to be folded into `gate_v9.py` as R1, R2 and P2.
- Embedding: `finetune/wan/asset_embed_v9.py`, `eval_v9/dcls_asset_v9.py`, `eval_v9/gen_asset_v9.py`;
  V1 script and logs in `/workspace/research_v9/v1/`.
