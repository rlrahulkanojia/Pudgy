# Training Approach v9 — dynamic motion, long scenes, props

**One sentence:** keep what v8 proved (one joint LoRA on an exactly-merged base, selected by
likelihood), and spend v9 on the three things v8 cannot do — **fast motion**, **long scenes
with backgrounds and props**, and **named props that stay stable** — each fed by new data and
each with its own measurement before any training starts.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B + **`pudgy-v8-merged`** = v7fixed + v8@550, one exact LoRA (`lora_tools_v8.py combine`, as v8 did for v7) |
| **Trains** | low-noise LoRA (as v8) **+ a high-noise LoRA for the first time** — layout and travel are decided at high noise, where the v2 golden has been frozen since v2 |
| **Data** | three new sets: `v9_motion` (fast motion, Round 5 + augmentation), `v9_scenes` (mined from the 101 client skits), `v9_props` (prop images + the shots they appear in) |
| **Order** | Phase 0 (measure, no training) → **v9a** fast motion → **v9b** scenes → **v9c** props, incremental |
| **Supersedes** | [v8](../v8/Training_Approach_v8.md) — results in [`REPORT_v8.md`](../../docs/training_reports/v8/REPORT_v8.md) |
| **Status** | plan (2026-09-30; client data requested 2026-10-01, §2.1). v9a is blocked on the Round 5 data ([`Client_Data_Request_Round5.md`](../../docs/documents/Client_Data_Request_Round5.md)); Phase 0 and the skit mining can start now |

---

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

**Source:** the client's finished skits — **iteration_1: 29 skits (5.8 min)** and
**iteration_2: 72 skits + ~100 storyboard PDFs** (`pudgy/raw/iteration_2`, 1.38 GB), 1080×1920.
These are the only footage with real sets, props, camera moves and characters moving
through a room.

Pipeline (`finetune/wan/scenes/`, to write):
1. **Shot split** — scene-cut detection; drop shots < 1 s.
2. **Clean** — reject shots with burned-in text, captions, end cards or product shots
   (OCR + the curation rubric in `.claude/skills/video-understanding`); crop nothing.
3. **Identity check** — `dupcheck.py` (never two Pax / two Polly), character present.
4. **Caption** (§3.4) — from the storyboard PDF beat + a per-shot visual pass.
5. **Buckets** — 9:16 (e.g. 480×848) at f33 · f49 · f81; 24 fps as delivered.
6. **Holdout by skit, not by shot** — 6 whole skits (3 per iteration) never trained on, so
   the long-form test is a genuinely unseen scene.

**Backgrounds (next step 1.ii):** (a) the real sets inside the skits; (b) the single-character
alpha clips **composited onto empty set plates** (client ask Round 5 §5; until then, plates
cleaned from skit frames) with a soft contact shadow — without the shadow the model learns
floating characters. Composites stay ≤ 25% of `v9_scenes`.

### 3.3 `v9_props` — prop library (v9c)

- **Prop list** — every recurring prop named in the storyboards (bus, plate, pillow, bowl,
  phone, blanket, lamp, …), ranked by how many skits use it.
- **Images** — per prop: the client's asset art if they can send it, else clean crops from
  skit frames; 10–30 images each, several angles/scales, on the flat backgrounds. Trained as
  single-frame samples (musubi image datasets) captioned with the prop's name and a fixed
  description ("a round white plate with a blue rim").
- **Videos** — the `v9_scenes` shots the prop appears in, with the same prop wording in the
  caption. Image + video share the wording so the name binds to one look.

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
- Ordered actions ("then") so a 3 s shot with two beats is described as two beats.
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
| base | v8-merged | v8-merged + v9a | previous stage merged |
| low-noise LoRA | rank 32, fresh | rank 32, fresh | rank 16 per prop stage |
| **high-noise LoRA** | rank 16, fresh, on travel + transition clips only | rank 16, continues | frozen |
| data | `v9_motion` + 30% v8 replay | `v9_scenes` + 20% v9a replay | prop stage + 20% replay |
| buckets | f13…f57 + **f81** (fast actions) | 9:16 f33 · f49 · f81 | images + f33 · f49 |
| loss | **motion-weighted**: per-token weight 1 + λ·change-mask, λ = 2 (the Tier-1 mask, now in training) | same | same |
| optimiser | adamw8bit, 5e-5, warmup 20, cosine → 10%, grad-accum 8 (unchanged — v8 showed it works) | | |
| budget | ≤ 600 steps, gate breaks at 100 / 300 | ≤ 800 | ≤ 200 per stage |

**Why the high-noise expert now:** at high noise, in-place actions have ≈ 0 between-label
variance (ANALYSIS_v7 §4), so v7/v8 were right to leave it frozen. Travel and scene layout are
the opposite — the character's position changes from frame to frame, and that is decided at
high noise. Trained only on the clips where it matters, and bisectable (drop the file to
recover v8 behaviour).

**Incremental props (v9c):** one stage per prop group (≈ 5 props), each with 20% replay of
everything before it and a **regression gate**: v8 Tier-1 held-out loss may not rise > 5%,
and earlier props' stability may not drop. A stage that fails is re-run with more replay, not
kept.

**Compute** (A100 80 GB, from v8's 39–43 s/clip at ≤ f57): f81 costs ≈ 1.5× per clip; the
high-noise LoRA adds one more pass. Estimate v9a ≈ 70 h, v9b ≈ 90 h, v9c ≈ 15 h per stage.
Verify with a 20-step smoke test before committing.

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
3. **Scene baseline** — v8@550 on the 6 held-out skits' first frames, scored as in §5.2.

### 5.2 New metrics

| metric | what | why |
|---|---|---|
| **stride frequency** | dominant frequency of the character's vertical bob / limb motion (FFT of the mask centroid) vs the real clip of the same action | "shuffles instead of strides" becomes a number |
| **travel** | character centroid displacement across the clip, camera-compensated on the background | tests enter / cross / exit |
| **pose change** | mean per-frame change inside the character mask, normalised by the real clip | fast = large per-frame change |
| **zoom** | ground-shadow width / head-line scale over the clip (REPORT_v8 §8) | the zoom-out regression |
| **prop stability** | largest single-frame jump + DINO similarity of each prop crop to frame 0 (from `qa_take.py`) | pops and morphs |
| **identity count** | `dupcheck.py` | never two Pax / two Polly |

### 5.3 Data-scaling experiment (tests F1 directly)

Train v9a three times on **25% / 50% / 100%** of the Round 5 running + jumping takes (same
steps, same replay), score stride frequency, pose change and Tier-1 loss, and fit error vs
data. This says whether 90 s per action is enough or where it flattens — and sizes the next
data request with a number instead of a guess. ≈ 3 × 20 h on a reduced step budget.

### 5.4 Pass bars

| run | proceed if |
|---|---|
| v9a | running + jumping Tier-1 loss ↓ ≥ 25% vs v8@550; stride frequency within 20% of real; periodicity ≥ 0.5 at f49/f81; `compose` motion ≥ 0.6 with expression ≥ 0.85; zoom ≤ 3%; no expression regression > 5% |
| v9b | on held-out skits: characters stay on-model for the full shot, zero duplicate frames, prop stability ≥ v8+FLF, travel matches the storyboard beat; blind A/B vs v8 long-form preferred ≥ 70% |
| v9c | each prop recognisable by name (CLIP zero-shot ≥ 0.8 on prop crops); regression gate §4 |

Ship bars from v8 (§1.2) carry forward unchanged; motion rows are the ones v9a must turn green.

---

## 6. Order and dependencies

| step | needs | can start |
|---|---|---|
| Phase 0 (VAE ceiling, metrics, scene baseline) | v8 weights, real clips | **now** |
| skit mining → `v9_scenes` | iteration_1/2 from Azure | **now** |
| augmentation → part of `v9_motion` | iteration_3–5 alpha clips | **now** |
| prop list + crops | storyboards, `v9_scenes` | after mining |
| **v9a** | Round 5 Tier A + re-exports (augmentation-only pilot possible first) | on delivery |
| data-scaling experiment | Round 5 Tier A | with v9a |
| **v9b** | `v9_scenes`, v9a merged | after v9a |
| **v9c** | prop images (client asset art preferred) | after v9b |

## 7. Risks and open questions

- **Keyframes for new scenes.** Everything above is image-to-video; a *new* scene needs its
  first frame. The long-form remakes used the client's frames. A keyframe generator (image
  LoRA on the characters + props) is the next workstream after v9c and needs the design
  sheets still outstanding from Rounds 3–5.
- **Skit rights / text overlays.** Mined shots must be text-free; if most skits carry
  captions the clean versions (Round 5 §5) become a blocker for v9b.
- **Synthetic travel looking synthetic.** Foot-slide guard + ≤ 30% cap; drop it if the
  scaling experiment shows real takes dominate.
- **Forgetting.** Each run replays the previous data and is gated on v8's Tier-1 set.
- **Tier-2 small n.** 2–4 clips per cell; decisions stay on Tier-1 losses and paired
  comparisons, as in v8.
