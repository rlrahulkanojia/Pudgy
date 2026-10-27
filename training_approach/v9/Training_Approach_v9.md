# Training Approach v9 — dynamic motion, long scenes, props

**One sentence:** keep what v8 proved (one joint LoRA on an exactly-merged base, selected by
likelihood), and spend v9 on the three things v8 cannot do — **fast motion**, **long scenes
with backgrounds and props**, and **named props that stay stable** — each fed by new data and
each with its own measurement before any training starts.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B + **`pudgy-v8-merged`** = v7fixed + v8@550, one exact LoRA (`lora_tools_v8.py combine`, as v8 did for v7) |
| **Trains** | low-noise LoRA (as v8) **+ a high-noise LoRA for the first time** — layout and travel are decided at high noise, where the v2 golden has been frozen since v2 |
| **Data** | three new sets: `v9_motion` (fast motion, Round 5 + augmentation), `v9_scenes` (mined from the 99 client skits), `v9_props` (prop in-use clips + the skit shots they appear in) |
| **Order** | **v9c-phone first** (2026-10-24, §0) → **v9b** scenes, on `pudgy-v8-merged` (§0.1) → **v9a** fast motion when its data arrives → v9c2 and more prop stages; Phase 0 alongside |
| **Supersedes** | [v8](../v8/Training_Approach_v8.md) — results in [`REPORT_v8.md`](../../docs/training_reports/v8/REPORT_v8.md) |
| **Assets** | [`Assets_v9.md`](Assets_v9.md): asset registry, props and set plates in training and inference |
| **Run it** | [`RUNBOOK_v9c_phone.md`](RUNBOOK_v9c_phone.md): `finetune/wan/v9_box.sh`, step by step, on a fresh GPU box |
| **Status** | **v9c-phone trained** (2026-10-26): gate passed, winner step 200; the phone still turns to show its screen in 5/8 clips, as in v8; motion accuracy regressed; `leak` clean ([report](../../docs/training_reports/v9/REPORT_v9c_phone.md)). The redesign is v9c2, embedding-first ([RESEARCH_v9c_fixes](RESEARCH_v9c_fixes.md) rev. 2). **v9b-1**: curated, set built and verified (`processed/v9b_535`, approved phone phrase), ready to train; **moved to a separate GPU machine on 2026-10-27** ([HANDOVER_v9b](HANDOVER_v9b.md)), so this box keeps the phone track. v9a is still blocked on Round 5 Tier A (iteration_6 has no new fast motion) |

---

## 0. Update 2026-10-24: iteration_6, and the phone stage runs first

iteration_6 brought the following (changelog: `Data/raw/iteration_6/CHANGELOG.md`, outside the repo; on Azure at `pudgy/raw/iteration_6/CHANGELOG.md`):

- **the first prop**: the iFin phone, 36 texting clips plus 24 views; the client confirmed
  `PRP_PHONE` is the iFin;
- **the first set plates**: 33 unique living-room / TV-room plates;
- **a second sitting performance** (Sitting V2);
- **a clean Confused re-export** (Confused V2).

It brought **no fast motion**, so v9a's data still has not arrived. The order therefore
changes:

| run | data | base | status |
|---|---|---|---|
| **v9c-phone** | 36 phone texting clips (32 trained, 4 held out) + 20% stratified v8 replay = 475 clips | `pudgy-v8-merged` = v7fixed + v8@550 (one exact file) | **trained** 2026-10-26, [report](../../docs/training_reports/v9/REPORT_v9c_phone.md) |
| v9b | skit shots (v9b-1), then character composites on the iteration_6 plates (v9b-2) | `pudgy-v8-merged` (rule: the latest merged base that passes the R1 regression gate, §0.1 D1) | prerequisites in §0.1 |
| v9a | Round 5 Tier A fast motion | the latest merged base | blocked on the client |
| v9c-2… | the next props, about 5 per stage | the latest merged base | as the client delivers prop art + in-use clips |

**Why the phone goes first.** It is the only complete new data set, and the client is
animating phone use next. One small stage also tests the whole asset mechanism end to end
before it scales to many props: registry → captions → staged LoRA → gate → merge. The
mechanism is in [Assets_v9.md](Assets_v9.md).

**What v9c-phone changes, and what it deliberately keeps.**

| | v9c-phone | why |
|---|---|---|
| base | v8@550, merged into one file | every stage stacks on a single exact base; no LoRA stacking at inference (v7's 76% error) |
| LoRA | fresh, low-noise, rank 16, ≤ 200 steps, save every 25 | §4 (prop stages) |
| high-noise | frozen | §4; prop entry is handled by first+last-frame anchoring |
| data | phone clips at f21 / f41 / f61, from each clip's longest clean window; replay sampled per (character, label), with every label kept at its common bucket | 11 Polly clips are damaged mid-clip, and every clip ends on a 10-frame hold |
| phone stills | **not trained** | in image-to-video a 1-frame sample is conditioned on itself, so it teaches almost nothing; the views are keyframe pixels and the eval reference (corrects §3.3) |
| captions | v8's wording unchanged (incl. "static … shot"); the phone named by the registry phrase | one new variable per stage; the camera-clause fix (§1.3 #5) moves to v9a |
| gate | held-out v8-label loss ≤ 1.05× base; held-out phone loss < 1.0× base; name binding reported | §4 regression gate, fixed before the data |

## 0.1 Update 2026-10-26: what v9b (long-form) needs before it can start

v9c-phone has finished training ([`REPORT_v9c_phone.md`](../../docs/training_reports/v9/REPORT_v9c_phone.md)).
v9b is next. The skits are pulled and audited
([`SKIT_AUDIT_v9b.md`](../../docs/training_reports/v9/SKIT_AUDIT_v9b.md)). This section was
re-checked on 2026-10-26 against the footage, the v9c findings (RESEARCH_v9c_fixes rev. 2)
and the code on the box. That re-check changed the base, the buckets, and the curation and
caption rules below.

### What the footage is (measured)

| | |
|---|---|
| **Source** | 99 skits: iteration_1 has 29 (5.8 min), iteration_2 has 70 (14.1 min); 2 animatics are excluded. All decode clean, every skit has its storyboard PDF, and there are **no burned-in captions** |
| **Usable now** | 311 shots (16.4 min) after the per-shot review and its re-check; 11 shots still to check. The re-check found **10 of the 321 first-pass keeps wrong**. The curator pass will cut more: 12 missed cuts or fades and 11 camera moves were found in the long shots alone, and shorter shots hide more (Tasks#1) |
| **Two characters** | **59% of kept scene time has Pax and Polly together** (v4 curation). v8 trained one character per clip, so the two-character shot is v9b's main new skill |
| **Motion** | median drift from frame 0 is **9.7** (v8 motions 2–16, phone clips 0.83). Only 7 shots are as still as the phone clips. The skits are dynamic footage |
| **Settings** | two kinds: full sets (bedrooms, igloo village, shops, car, gym, beach) and **plain pale-blue grounds with a few pieces of furniture**. The second kind sits between v8's flat grounds and full sets, and captions name it as such |
| **Not the leads** | about 15 shots with side characters (grey penguins, crowds, gym-goers, a flower-crab). **A Polly lookalike**: Polly's mum is Polly plus a pearl necklace (MothersDay2). Outfits on the leads: hard hat, scarf, headband, tie, backpack, shower cap, hood, a long white beard. Effects: a time-lapse (Aging), a fly-in through a light burst (Jump), dark vignettes (WithHer, Without). An overlay arrow inside a kept shot (BF) |
| **Leakage** | no identical files, and no shot reused across skits (frame hashes), so holding out whole skits is clean |
| **Lengths** | 41 kept shots (0.8 min) are shorter than f33; 200 are ≥ f49; 95 are ≥ f81 |

### Decisions (D1–D4 confirmed by the user 2026-10-26; D5–D10 as planned)

Execution, commands and a dated log: [`RUNBOOK_v9b.md`](RUNBOOK_v9b.md).

| # | decision | recommendation | why |
|---|---|---|---|
| D1 | v9b's base | **`pudgy-v8-merged`**, not v9c-phone. Rule: v9b starts on the latest merged base that passes the R1 regression gate ([RESEARCH_v9c_fixes §5](RESEARCH_v9c_fixes.md)) | v9c-phone did not change the phone defect (5/8 turns in both), regressed motion (0.75 → 0.58), and is being replaced by v9c2: an asset embedding first, which leaves v8 prompts untouched by construction. Fixing it first would put ~30 h on v9b's critical path for a branch that is being replaced. v9c2 can stack on v9b's merged base later |
| D2 | resolution | **480×848** (9:16). **Confirmed by the Phase 0 VAE check, 2026-10-26** (RUNBOOK_v9b log): faces intact by eye; lead-box PSNR 32.8 dB vs 33.8 at 576×1024 and 34.9 for v8's 1024² clips | In wide shots the leads are ¼–⅓ of the frame height, ≈ 210–280 px at 848. In v9c's 1024² clips they are a median 0.63 of the frame, ≈ 650 px (measured). 720×1280 at f81 is 76k tokens per clip, ≥ 2.3× the cost, too slow for 800 steps on one A100 |
| D3 | camera | **static spans only** in v9b-1. Pans, zooms and pushes are cut out and kept, tagged, for a later camera stage | v9 captions promise a fixed camera; a hidden zoom teaches the zoom-out regression (REPORT_v8 §8). 83% of v4's scenes are static |
| D4 | who may be in a shot | the leads, plus **distinct** side characters named by a registry phrase (`kind = "character"`, generic wording, no "blue" or "pink"). **Drop lookalikes** (Polly's mum) and lead plushies, posters and photos | dupcheck and captions cannot tell Polly from Polly with a necklace; a lookalike teaches "two Pollys" |
| D5 | outfits | every outfit on a lead is named with a registry `outfit` phrase ("…, wearing a yellow hard hat"). The curator adds the phrases as it finds them | unnamed, the model learns that outfits appear on the leads at random |
| D6 | buckets | **f21 · f33 · f49 · f81** (f21 added) | the old "drop < 1 s" kept 41 shots that no bucket could use. f21 is v8/v9c's floor |
| D7 | experts | low-noise LoRA first. A **high-noise LoRA as a separate run** afterwards, on the travel and entrance spans, gated on its own and bisectable, **only if** E1a shows the layout is already set at the switch, or the low-noise run fails its travel bar | that layout is decided above t = 0.9 is an unverified hypothesis (RESEARCH_v9c_fixes C3); E1a (x₀ probe) tests it. Joint mode would share one LoRA across two experts |
| D8 | loss | the motion-weighted loss (§4) only after a 50-step pilot against plain MSE, compared on *unweighted* held-out loss. A preservation loss on replay is optional: in expectation it equals replay, and its gain is a noise-free target (RESEARCH_v9c_fixes C9) | the weighted loss **exists only in eval** (`eval_v8/dcls_v8.py`), not in training. In a wide scene shot, a nearly static background fills most of the frame and dominates plain MSE. But the weight is an untested variable |
| D9 | camera clause | scene clips get the new clause (§1.3 #5); replay keeps v8's wording, so published v8 prompts behave as before. Test the negative-prompt fix first (§5.1 #5) | v9a, which was to introduce the clause, is blocked |
| D10 | **caption overlap with v8** | scene captions describe each action in their own words and **never copy a v8 action clause verbatim** (lint against `prep_v8`'s label texts). v8 replay anchors every label at ≥ 1 replay clip per 4 scene spans that use the same motion word. R1 gates per kind and is **reported per label** (n = 4–7 clips each, too few to gate on), so a clause that absorbs others shows up by name | v9c's motion regression came from the phone-sitting caption reusing v8's sitting clause at 190 : 7, so that clause "absorbed" other motions (RESEARCH_v9c_fixes F3, rev. 2), not from static footage (C5). v9b will use "walks", "sits" and "waves" in hundreds of spans that look nothing like v8's clips |

### Our side (no client input needed)

| # | item | state |
|---|---|---|
| 1 | Pull and audit the skits | **done**: [`SKIT_AUDIT_v9b.md`](../../docs/training_reports/v9/SKIT_AUDIT_v9b.md); per-shot verdicts in `finetune/wan/scenes/audit/shot_review.json`, corrected after the re-check |
| 2 | **Curator pass, v9b rules** (D3–D5): every cut and fade at frame accuracy; static spans; which characters are present and where (left / right); outfits; side characters; prop events (enters, picked up, put away); overlays anywhere in the span; a reason for every drop. iteration_2 starts from the v4 JSONs, iteration_1 starts fresh. Output: one JSON per skit, plus a contact strip per span for spot checks | **done** 2026-10-26: 12 curators, then 3 verification passes covering every kept span (173 ok / 94 fixed / 12 dropped); 266 kept spans, 9.7 min (RUNBOOK_v9b log) |
| 3 | Registry: side characters, outfits, settings (each full set, plus the plain pale-blue ground) and skit props, each with one phrase | **deferred**: v9b-1 captions use the curators' per-skit descriptions (consistent within a skit); registry entries for recurring sets/outfits are a follow-up for the inference prompt compiler. The iFin phrase change is pending the user's decision |
| 4 | Captions: a scene sheet per skit, plus ordered actions per span (storyboard beat + v4 `action_beats`), compiled from the registry (§3.4) and linted against v8's action clauses (D10) | **written**: `scenes/captions_v9b.py` compiles from the curators' beats |
| 5 | `prep_v9b.py`: spans → windows at f21 / f33 / f49 / f81 (480×848); `longform/dupcheck.py` on every window; replay mix; holdout split; manifest | **done**: `processed/v9b_535` (348 scene + 187 replay); every clip verified; dupcheck on all windows (62 flags, all false positives by eye) |
| 6 | Holdout: 6 whole skits (3 per iteration), chosen to cover one and two characters, flat ground and full set, and at least one prop event | **done**: BackToBed, TirePressure, Drinker, BackTogether, MorningFart, BubbleWrap (7% of kept frames) |
| 7 | Trainer: motion-weighted loss (D8, optional); the high-noise run's config (D7) | `train_pudgy_v9b.sh` + `v9b_box.sh` written; plain MSE (D8); high-noise deferred (D7) |
| 8 | Eval: a scene suite (held-out shots' first frames, 3 seeds), scored with dupcheck, zoom, prop stability and motion relative to the real shot; the v9b gate (§5.4) | **written**: `eval_v9b/` (build, tier1, gate, suite) |
| 9 | Phase 0 on the GPU: scene baseline, scene-scale VAE check, negative-prompt zoom test (§5.1 #3–5) | VAE check running on CPU (`eval_v9b/phase0_vae_v9b.py`); baseline after the v9c2 GPU queue; the negative-prompt test is informational and runs after training starts |
| — | Composites on the plates: `scenes/composite_plates.py`, plus `floor` and `scale` annotations | **moved to v9b-2**; it needs the client's background answers |

### Client side

| item | why | blocking? |
|---|---|---|
| ~~Text-free skits~~ (Round 5 §5) | mined shots must have no burned-in captions | **not needed**: the `_Base` exports carry no captions (audit) |
| Background answers: consistent plate names, the door sequences' order, direction and timing, the TV room's time-of-day sequence, which skit shot each plate comes from | composites and their captions | v9b-2 only |
| Prop art (+ in-use clips) for scene objects: TV, blanket, cushion, popcorn, remote, … | props the characters handle need a v9c stage | no, can follow |
| Character-layer alpha exports of skit shots | many more distinct performances (Assets_v9 §5.5) | no, optional |

**Nothing in v9b-1 needs the client.**

**A side benefit for v9c2.** Skit shots where a phone is taken out or put away are the
entrance clips that RESEARCH_v9c_fixes asks the client for (A1). There, the caption carries
information the first frame does not, so the phone's name can bind. The curator tags these
prop events (item 2). Each skit phone is checked against the iFin (three lenses, tuft)
before it is captioned with the registry phrase.

### Order, effort and cost

- **Now, without the GPU (~1–1.5 weeks):** items 2–8. The curator pass is most of it.
- **On the GPU meanwhile:** the Phase 0 items (§5.1 #3–5) and v9c2's experiments
  ([RESEARCH_v9c_fixes §4](RESEARCH_v9c_fixes.md)). None of them needs v9b data.
- **v9b-1 training, ≈ 3.5–6 days on one A100:**
  - low-noise run, 400 steps (set from the curated data size, RUNBOOK_v9b §6), Tier 1 at
    100–400: ≈ 30–45 h. v9c ran at 61 s per clip; a v9b sample is about ⅔ of v9c's tokens, but
    block swapping does not shrink with it;
  - the high-noise run, only if D7 calls for it: ≈ 25 h;
  - eval: ≈ 20 h. A 480×848 f81 clip generates in about 12 min.

  Confirm all of this with a 20-step smoke test.
- v9b now runs before v9a (§6), because Round 5 is still pending.

## 1. What v8 showed

### 1.1 The two findings

**F1 — the faster the motion, the more data it needs; fast motion is the least accurate.**
The diffusion classifier *recognises* running and jumping (rank 1 on 3 of 4 held-out clips),
but they carry the most unexplained error of any motion — the model knows which action it is,
and reproduces it worst:

| action | unique footage | longest clip | held-out loss, v8@550 | vs golden |
|---|---|---|---|---|
| sitting | 11.8 s | 1.17 s | 0.00083 | −68% |
| waving | 16.7 s | 1.08 s | 0.00113 | −61% |
| walking | 7.6 s | 0.67 s | 0.00258 | −41% |
| **jumping** | 22.0 s | 1.38 s | **0.00436** | −54% |
| **running** | 11.3 s | 0.71 s | **0.00452** | −58% |

(n = 2 held-out clips per action; loss is change-masked, so a faster action has more to
predict — the ranking is the point, not the absolute gap.) In generated clips, long motion
doesn't cycle: every `gd` clip at f49/f57 has periodicity ≤ 0.23 against a bar of 0.5 — one
slow arc, not a repeating stride.

The cause is the data, not the recipe: every action is **one performance filmed from nine
angles**, performed **on the spot**, in clips shorter than one or two cycles. The VAE
compresses time 4× (≈ 6 latent frames per second), so a running stride (~8 frames) lands in
about 2 latent steps — the model sees very few distinct samples of what fast motion looks like.

**F2 — the model can play two animations at once, but not equally.** In `compose` (motion ×
expression in one prompt) the **expression reads in 94%** of clips and the **motion in 44%**;
motion energy drops to 78% of motion-only clips. Visually the expression takes over in the
second half (Pax waving → crying covers his face). Co-existence works; the balance favours
the face.

### 1.2 What v8's ship bars say (Tier 2)

| bar | measured (v8@550) | |
|---|---|---|
| controllability: CLIP 1-NN ≥ 0.57 | expr 0.44 / 0.50, motion 0.40 / 0.60 (Pax / Polly) | ❌ except Polly motion |
| holdout view ≤ 0.15 below FRONT (zero-shot) | expr 0 / 0; motion −0.2 / −0.4 | ❌ motion |
| composition: motion energy ≥ 80% | 78% | ❌ (borderline) |
| unseen grounds: corner drift ≤ 5/255 | 1.1–1.7 | ✅ |
| duration: periodicity ≥ 0.5 | −0.27 … 0.23 (all 12 clips) | ❌ |
| parity within 10% | expr ✅, motion ❌ (0.40 vs 0.60) | ❌ motion |

Tier 1 (likelihood on held-out clips) improved decisively; Tier 2 (what generated clips look
like) is weakest exactly on **motion** — the same message as F1. **Inherited, not a v8
regression:** on the same prompts v8 is best or tied in every cell (motion 1-NN Pax 0.20 →
0.30 → **0.40**, Polly 0.50 → 0.50 → **0.60** for golden → v7fixed → v8; `compose` motion 0.39
→ 0.44), so every model trained on this data misses the motion bars. More data, not a
different recipe, is what v9a tests.

### 1.3 → v9 decisions

| # | Evidence | v9 decision |
|---|---|---|
| 1 | F1: fast motion has the least data and the highest residual | tiered data by motion complexity (Round 5 §2); augmentation that adds *travel* and *speed* (§3.1); a data-scaling experiment to measure the curve (§5.3) |
| 2 | clips shorter than a cycle; periodicity ≤ 0.23 | clips of **3–5 s with ≥ 3 cycles** for fast actions; f81 rung |
| 3 | everything in place; layout fixed at high noise, high-noise expert frozen | **train a high-noise LoRA** on travel + scene data |
| 4 | F2: motion loses to expression in `compose` | real motion × expression clips (Round 5 §3) and a **motion-weighted loss** (§4) |
| 5 | zoom-out: "static … shot" in captions vs "static" in Wan's negative prompt | caption camera clause → "fixed camera, locked-off shot, the character stays the same size" |
| 6 | long-form: two characters + props on a table lose the layout ~0.8 s in | scene data with props (v9b) + prop library (v9c); first+last-frame anchoring stays as the fallback |

---

## 2. Next steps → workstreams

The three next steps agreed after v8, and where each lands:

| Next step | Workstream | Data | Run |
|---|---|---|---|
| **1. Longer videos** — i. with props, ii. with different backgrounds, iii. with captions for long-form | scene mining + compositing + scene captions | `v9_scenes` | **v9b** |
| **2. More samples of fast animation** (running, jumping, …) | Round 5 request + augmentation | `v9_motion` | **v9a** |
| **3. Prop images** (bus, plate, pillow, …) fed incrementally with the videos they appear in | prop library, staged training with replay | `v9_props` | **v9c** |

### 2.1 What the client has been asked for (2026-10-01) — and what it unlocks

More data has been requested: **fast-animation videos** and **longer clip durations** (full
spec: [`Client_Data_Request_Round5.md`](../../docs/documents/Client_Data_Request_Round5.md)).

| ask | unlocks | why it matters |
|---|---|---|
| **fast animation** (run, chase, jump sequences, …) | **v9a** — the run is blocked on it | running has 11.3 s of unique footage, all one performance; F1 says fast motion needs the most data |
| **longer durations** (3–5 s, ≥ 3 cycles) | **v9a** (cycles, f81 rung) and **v9b** (full shots) | longest running clip today is 0.71 s (< 2 strides): the model has never seen a stride repeat — the `gd` periodicity failure (≤ 0.23 vs 0.5) |

**What makes the delivery count** (confirm with the client):
1. **Different performances, not more angles** — 5 runs from 4 angles beat 1 run from 9. Every
   action so far is one take re-rendered from nine cameras, so more footage has not meant more
   variety.
2. **Some clips that cross the frame**, not only in-place — scenes need travel (and the
   high-noise LoRA needs it to learn anything).
3. **Same format**: 1080×1080, 24 fps, alpha (ProRes 4444) — so the travel/speed augmentation
   (§3.1) and compositing (§3.2) work on it unchanged.
4. **Transitions** (idle→run→stop, walk→run) — the joins long-form scenes need.
5. **`<TAKE>` in the filename** — so takes are told apart and some held out for evaluation.

**How long clips are used.** One generated shot is ≤ 81 frames (3.4 s at 24 fps); clips up to
that train as-is, longer clips are cut into several windows (more cycles, plus the transitions
inside them). A 15 s scene is still a chain of shots — longer data makes each shot better, it
does not remove the chaining.

**How much is enough is measured, not assumed.** The ≈ 90 s per fast action (≈ 8× today's
running) is a judgement; the data-scaling experiment (§5.3) trains on 25 / 50 / 100% of the
new takes and shows where the gains flatten, which sizes any follow-up request.

**Not covered by this ask:** prop images (v9c — client asset art preferred), keyframes for
new scenes (§7), and the **21 damaged files**, which still need a re-export.

**While the data is pending** (no client data needed): Phase 0 (§5.1), skit mining (§3.2),
travel/speed augmentation (§3.1) — so v9a can start on delivery.

---

## 3. Data

### 3.1 `v9_motion` — fast motion (v9a)

**From the client (Round 5, priority 1):** Tier A actions (run, run across the frame, chase,
jump sequences, fall/get-up, dance) at **≥ 5 different takes, ≈ 90 s per action per
character**, 3–5 s clips with ≥ 3 cycles, four angles; Tier B ≈ 45 s, Tier C ≈ 20 s;
transitions (idle→walk→idle, walk→run, stand→sit, idle→jump→land); expression changes during
motion; `<TAKE>` in the filename. Plus the **21 re-exports** (all 20 from the earlier list
came back byte-identical; one new blushing file).

**Augmentation we can build now** (the clips are ProRes 4444 with alpha, which makes these exact):

| augmentation | how | what it teaches | guard |
|---|---|---|---|
| **travel** | slide an in-place cycle across the frame at the speed its feet imply (stride length ÷ cycle frames) — how 2D animators make a walk travel | entering, crossing, leaving the frame | foot-slide check: the planted foot's ground-plane position may move ≤ 2 px per frame |
| **speed** | drop every other frame of a cycle (2× faster); label "running fast" | speed words in captions | only for cycles ≥ 16 frames, so a stride still spans ≥ 2 latent steps |
| **cycle tiling** | as v8 (seam ÷ median step ≤ 1.5) | longer clips | running was never tileable (seam 2.3–2.7) — needs the new ≥ 3-cycle takes |

Augmented clips are **≤ 30% of any label** and are tagged, so the scaling experiment (§5.3)
can separate real from synthetic data.

### 3.2 `v9_scenes` — long scenes with props and backgrounds (v9b)

**Source:** the client's finished skits. **iteration_1** has 29 skits (5.8 min, 0.57 GB) and
**iteration_2** has 70 skits plus 2 animatics (14.1 min, 1.38 GB). Every skit has a storyboard
PDF (130 PDFs in all). The skits are 1080×1920 at 24 fps (one is 23.976 fps). They are the only
footage with sets, props, two characters together, entrances and camera moves. Audit:
[`SKIT_AUDIT_v9b.md`](../../docs/training_reports/v9/SKIT_AUDIT_v9b.md).

Pipeline (`finetune/wan/scenes/`; the audit scripts are in `scenes/audit/`, the rest is to
write):
1. **Shot split.** Scene-cut detection is only a seed; it misses cuts, dissolves, fades and
   camera moves. A **curator pass** places every cut at frame accuracy and splits shots into
   static spans (§0.1 item 2). Anything shorter than f21 is dropped.
2. **Clean.** Drop graphics and overlays (title cards, meters, chat and speech bubbles,
   arrows), ads and merch, inserts with no lead, lead plushies, posters and photos,
   off-style shots, time-lapses and lookalike characters. Check the whole span for overlays,
   not only its middle frame. In-world text (signs, packaging, screens) stays and is never
   transcribed. Crop nothing. The skits carry no burned-in captions, so nothing is dropped
   for that.
3. **Identity check.** `longform/dupcheck.py` on every window (never two Pax or two Polly),
   at least one lead present. The curator also records side characters and outfits, which
   are named from the registry (§0.1 D4, D5).
4. **Caption** (§3.4): from the storyboard beat, v4's `action_beats` and a per-span visual
   pass, compiled from the registry.
5. **Buckets.** 9:16 at 480×848 (§0.1 D2), f21 · f33 · f49 · f81, 24 fps.
6. **Holdout by skit, not by shot.** 6 whole skits (3 per iteration) are never trained on,
   so the long-form test is an unseen performance. Sets repeat across skits, so held-out
   skits test new performances in known rooms; the held-out plates (Assets_v9 §5.4) test new
   rooms.

**Backgrounds (next step 1.ii):**
- (a) the sets inside the skits, both the full sets and the plain pale-blue grounds with
  furniture (v9b-1);
- (b) the single-character alpha clips **composited onto the 33 client plates** from
  iteration_6 (v9b-2, Assets_v9 §5). The contact shadow is already in the clip alpha, so no
  synthetic shadow is needed (Assets_v9 §8).

Composites stay ≤ 25% of `v9_scenes`.

### 3.3 `v9_props` — prop library (v9c)

- **Prop list** — every recurring prop named in the storyboards (bus, plate, pillow, bowl,
  phone, blanket, lamp, …), ranked by how many skits use it.
- **Images** (superseded, [Assets_v9 §4.1](Assets_v9.md)): per prop, the client's asset art if
  they can send it, else clean crops from skit frames. The art is **not trained**: in
  image-to-video a 1-frame sample is conditioned on itself. It serves as inference keyframe
  pixels and as the eval reference.
- **Videos**: the prop's in-use clips, plus the `v9_scenes` shots it appears in, with the
  registry phrase in every caption. Shots where the prop enters or is picked up are the ones
  that can bind the name (RESEARCH_v9c_fixes F1).

**What this buys, honestly:** in image-to-video the prop is usually already in the start
frame, so the gain is (1) props that **stay the same object** through motion — the
EatingStages failure — and (2) props that **enter or get picked up** mid-shot. Putting a new
prop into a *new* start frame is a keyframe problem (an image model), tracked as §7.

### 3.4 Captions for long-form (next step 1.iii)

v8's structure (`STYLE + character anchors + action + camera`) extended to scenes:

```
<STYLE>. <SETTING: place, background, props in view>. <CHARACTER anchors>.
<ACTION in order: "first …, then …, finally …">, <speed word: slowly / quickly>.
<EXPRESSION and when it changes>. Fixed camera, locked-off shot, the characters stay the same size.
```

- Prop names verbatim from the prop library (§3.3); colour anchors as v8 (no "pink" for Pax).
  Every phrase comes from the registry, so a phrase change, such as the phone's after E2,
  recompiles every caption.
- **Two characters** (59% of the scene time): both anchors, each with its screen side and its
  own actions ("Pax, …, on the left, …; Polly, …, on the right, …"). The 21 iteration_6
  interaction clips use the same wording.
- **Side characters and outfits**: a registry phrase each ("a grey penguin in sunglasses";
  "Pax, …, wearing a yellow hard hat"). Never a colour word that belongs to a lead.
- **Setting**: the set's registry phrase, or "a plain pale blue background with a cream
  couch" for the flat-ground skits. Time of day when it isn't day.
- Ordered actions ("then") so a 3 s shot with two beats is described as two beats.
- **Never copy a v8 action clause verbatim** into a scene caption (§0.1 D10). A reused clause
  gets redefined by the new footage, which is what regressed motion in v9c-phone.
- A **scene sheet** per skit (setting + props + characters) reused for every shot of that
  skit, so consecutive generated shots share wording — this is what makes chained shots stay
  consistent.
- The camera clause replaces "static … shot" (zoom-out fix, REPORT_v8 §8).

**Long means chained shots.** One generated shot tops out at 81 frames (3.4 s at 24 fps);
10–15 s scenes are 3–6 shots chained by first/last frames, as in the v8 long-form pipeline.
v9 trains the pieces — full shots, beat-ordered captions, scene sheets — not one 15 s clip.

---

## 4. Recipe

| | v9a motion | v9b scenes | v9c props |
|---|---|---|---|
| base | v8-merged | **v8-merged** (§0.1 D1; v9a has not run) | previous stage merged |
| low-noise LoRA | rank 32, fresh | rank 32, fresh | rank 16 per prop stage |
| **high-noise LoRA** | rank 16, fresh, on travel + transition clips only | rank 16, fresh, a **separate run** after the low-noise one, on travel / entrance spans (§0.1 D7) | frozen in v9c-phone; in v9c2 only if E1a shows the turn is set at the switch (RESEARCH_v9c_fixes §6) |
| data | `v9_motion` + 30% v8 replay | `v9_scenes` (static spans) + 35% stratified v8 replay with label anchors (§0.1 D10; 20% failed to protect motion in v9c) | v9c2: an **asset embedding** first (network frozen); a LoRA stage with anchors + 35% replay only if the embedding isn't enough |
| buckets | f13…f57 + **f81** (fast actions) | 9:16 480×848, f21 · f33 · f49 · f81; replay at its v8 buckets | in-use clips f21 · f41 · f61 (no stills) |
| loss | **motion-weighted**: per-token weight 1 + λ·change-mask, λ = 2 (the Tier-1 mask; **not yet implemented in training**) | same, only if a 50-step pilot beats plain MSE (§0.1 D8) | data MSE (+ preservation if a LoRA stage is used) |
| optimiser | adamw8bit, 5e-5, warmup 20, cosine → 10%, grad-accum 8 (unchanged — v8 showed it works) | | |
| budget | ≤ 600 steps, gate breaks at 100 / 300 | **400** (was ≤ 800): the curated set is ~565 clips, so 400 × 8 ≈ 5.7 epochs; Tier 1 at 100 / 200 / 300 / 400 | ≤ 150 per stage (v9c-phone plateaued by 125) |

**Why the high-noise expert now:** at high noise, in-place actions have ≈ 0 between-label
variance (ANALYSIS_v7 §4), so v7/v8 were right to leave it frozen. Travel and scene layout are
the opposite — the character's position changes from frame to frame, and that is decided at
high noise. Trained only on the clips where it matters, and bisectable (drop the file to
recover v8 behaviour).

**Incremental props (v9c):** one stage per prop group (≈ 5 props). From v9c2 on, each prop
is first an **asset embedding** (network frozen, so v8 prompts are unchanged by
construction). A LoRA stage with replay of everything before it is added only where the
embedding isn't enough. Each stage passes a **regression gate**, which since v9c-phone is
RESEARCH_v9c_fixes §5:
- paired per-kind margins (R1);
- per-t loss (R2);
- held-out prop loss (P1);
- text-attributable gain (P2);
- a visual identity bar per prop (ID1 for the phone).

The old averaged ≤ 1.05× loss bar passed a run that regressed motion. Earlier props'
stability may not drop either. A stage that fails is re-run, not kept.

**Compute** (A100 80 GB). v8 ran 39–43 s per clip at ≤ f57. v9c-phone measured **61 s per
clip** at 1024² (27 h for 200 steps, against the 15 h estimated), with block swapping. A
480×848 f81 clip is ≈ 33k tokens, ≈ ⅔ of a v9c sample. The high-noise LoRA is a separate run.

Estimates:
- v9a ≈ 70 h;
- v9b ≈ 60–90 h, plus ≈ 25 h for the high-noise run and ≈ 20 h of eval (§0.1);
- v9c ≈ 20 h per stage at 150 steps.

Verify each with a 20-step smoke test before committing.

---

## 5. Evaluation

v8's three tiers stay (Tier 0 weights, Tier 1 diffusion classifier, Tier 2 suites). New:

### 5.1 Phase 0 — before any training

1. **VAE ceiling.** Round-trip real running / jumping clips through the Wan VAE and score
   them with the §5.2 metrics. If the *reconstruction* already loses the stride, no LoRA can
   recover it, and fast actions need a higher frame rate or larger motion in the source.
   This decides whether 24 fps + f81 is enough.
2. **Dynamics metrics that ignore the camera** (§5.2) on v8@550 and the real clips — the v9
   bars are set from these numbers, before v9 data exists.
3. **Scene baseline**: `pudgy-v8-merged` (≡ v8@550) on the first frames of the 6 held-out
   skits' spans, with their v9b captions, 3 seeds, scored as in §5.2. This is the "before"
   for every v9b bar.
4. **Scene-scale VAE check** (decides §0.1 D2). Round-trip 20 kept spans through the Wan VAE
   at 480×848 and at 576×1024. Compare the leads' eyes, beak and outlines with the source,
   by eye and by PSNR inside a character crop. If 480×848 loses them, train at 576×1024.
   The phone (a few tokens at this scale, Assets_v9 §10) is checked the same way.
5. **Negative-prompt zoom test** (inference only, on v8). The zoom-out came from "static …
   shot" in the captions against "static" in Wan's default negative prompt (§1.3 #5). Re-run
   v8's zoom cases with "static" removed from the negative prompt, and separately with the
   new camera clause, and measure zoom (§5.2). If the negative prompt alone fixes it, the
   caption change (§0.1 D9) can wait.

### 5.2 New metrics

| metric | what | why |
|---|---|---|
| **stride frequency** | dominant frequency of the character's vertical bob / limb motion (FFT of the mask centroid) vs the real clip of the same action | "shuffles instead of strides" becomes a number |
| **travel** | character centroid displacement across the clip, camera-compensated on the background | tests enter / cross / exit |
| **pose change** | mean per-frame change inside the character mask, normalised by the real clip | fast = large per-frame change |
| **zoom** | ground-shadow width / head-line scale over the clip (REPORT_v8 §8) | the zoom-out regression |
| **prop stability** | largest single-frame jump + DINO similarity of each prop crop to frame 0 (from `qa_take.py`) | pops and morphs |
| **identity count** | `longform/dupcheck.py` | never two Pax / two Polly |
| **text-attributable gain** | Tier-1 loss under the true caption minus the mean over distractor captions (beat order swapped, the leads' actions exchanged, another skit's caption); RESEARCH_v9c_fixes P2 | whether training used the caption. In v9c-phone it did not (0.4% of the gain); scene captions carry what the first frame cannot |
| **motion ratio** | mean per-frame change of the generated span ÷ the real held-out span, same first frame | catches freezing (v9c F3) and over-motion |

### 5.3 Data-scaling experiment (tests F1 directly)

Train v9a three times on **25% / 50% / 100%** of the Round 5 running + jumping takes (same
steps, same replay), score stride frequency, pose change and Tier-1 loss, and fit error vs
data. This says whether 90 s per action is enough or where it flattens — and sizes the next
data request with a number instead of a guess. ≈ 3 × 20 h on a reduced step budget.

### 5.4 Pass bars

| run | proceed if |
|---|---|
| v9a | running + jumping Tier-1 loss ↓ ≥ 25% vs v8@550; stride frequency within 20% of real; periodicity ≥ 0.5 at f49/f81; `compose` motion ≥ 0.6 with expression ≥ 0.85; zoom ≤ 3%; no expression regression > 5% |
| v9b | **regression:** R1 and R2 on the v8 labels (RESEARCH_v9c_fixes §5). **learned:** held-out span loss < 1.00× base, and a text-attributable gain whose 95% CI excludes 0. **looks right** (held-out skits' spans, 3 seeds, vs the Phase 0 baseline): zero duplicate-character frames, zoom ≤ 3%, prop stability ≥ v8 + FLF, motion ratio 0.7–1.3, leads on-model for the whole span. **long-form:** a blind A/B of 3-shot chains on held-out skits vs the v8 long-form pipeline, preferred ≥ 70%. The high-noise run is gated on the same bars, separately |
| v9c | the v9c2 gate (RESEARCH_v9c_fixes §5: R1, R2, P1, P2, ID1) |

Ship bars from v8 (§1.2) carry forward unchanged; motion rows are the ones v9a must turn green.

**Lesson from v9c-phone:** a loss-only gate passed a run that changed nothing visible and
regressed motion. Every stage now needs a paired regression bar, a visual bar with ≥ 3
seeds, and a sample size set by power analysis (RESEARCH_v9c_fixes §5), all fixed before the
data.

---

## 6. Order and dependencies

| step | needs | can start |
|---|---|---|
| **v9c-phone** (§0) | iteration_6, v8 weights | **done** 2026-10-26 ([report](../../docs/training_reports/v9/REPORT_v9c_phone.md)); all suites done (`leak`: no phone in v8 prompts) |
| Phase 0 (VAE ceiling, metrics, scene baseline, scene-scale VAE, negative-prompt zoom test) | v8 weights, real clips; the held-out skits for the scene baseline | **now** on the GPU; the scene baseline needs §0.1 item 6 first |
| v9c2 experiments | v9c-phone checkpoints; `asset_embed_v9.py`, `phone_id_v9.py` | **running** ([RESEARCH_v9c_fixes §4](RESEARCH_v9c_fixes.md)) |
| skit curation → `v9_scenes` | iteration_1/2 (pulled and audited 2026-10-26) | **now**: §0.1 items 2–6 |
| augmentation → part of `v9_motion` | iteration_3–5 alpha clips | **now** |
| **v9b-1** (skit spans) | `v9_scenes`, `pudgy-v8-merged`, §0.1 items 2–9 | when those are done (≈ 1–1.5 weeks) |
| **v9b-2** (plate composites) | the client's background answers, `composite_plates.py` | after v9b-1 |
| **v9a** | Round 5 Tier A + re-exports (augmentation-only pilot possible first) | on delivery |
| data-scaling experiment | Round 5 Tier A | with v9a |
| **v9c2** and the next props | the experiments' results (RESEARCH_v9c_fixes §6 decision tree); client data A1–A3; prop in-use clips | on the latest merged base that passes its gate |

## 7. Risks and open questions

- **Keyframes for new scenes.** Everything above is image-to-video; a *new* scene needs its
  first frame. The long-form remakes used the client's frames. A keyframe generator (image
  LoRA on the characters + props) is the next workstream after v9c and needs the design
  sheets still outstanding from Rounds 3–5.
- ~~**Skit text overlays.**~~ Resolved 2026-10-26: the `_Base` skits carry no burned-in
  captions. Overlay graphics (bubbles, meters, arrows) are dropped by curation.
- **Curation errors.** The first per-shot pass wrongly kept 10 of 321 shots (a lookalike, a
  time-lapse, a fly-in, an overlay arrow, a missed cut), and the cut detector misses cuts,
  fades and camera moves. Every span is checked whole, not by its middle frame, and a random
  10% of the curator's spans get a second look before prep.
- **Two characters in one clip.** v8 never trained it, and v8's long-form lost a
  two-character table layout ~0.8 s in. Watch for identity swaps (Pax's colours on Polly) and
  duplicates: colour anchors in every caption, dupcheck on every window and every generated
  span.
- **Small leads at scene scale.** At 480×848 the leads in wide shots are ≈ 210–280 px tall,
  against ≈ 650 px in training so far. The Phase 0 VAE check (§5.1 #4) decides the
  resolution.
- **The holdout tests performances, not rooms.** Sets repeat across skits; unseen rooms are
  tested on the held-out plates (Assets_v9 §5.4).
- **Synthetic travel looking synthetic.** Foot-slide guard + ≤ 30% cap; drop it if the
  scaling experiment shows real takes dominate.
- **Forgetting.** Each run replays the previous data and is gated on v8's Tier-1 set.
- **Tier-2 small n.** 2–4 clips per cell; decisions stay on Tier-1 losses and paired
  comparisons, as in v8.
