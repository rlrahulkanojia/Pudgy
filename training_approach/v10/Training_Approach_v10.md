# Training Approach v10: production reels, end to end (final stage)

**In one sentence:** fine-tune the production model (`pudgy-v8-merged` + E5) on the client's
production skits (the 186 new clean `_Base` exports in `iteration_7`, many of them the clean
versions of its Instagram reels), together with the 99 skits v9b used, so that each *shot*
behaves like a production shot. Do it through the **high-noise expert first**, where layout, staging and motion
are decided (the expert v9b-1 left frozen). Then assemble end-to-end reels with the shot pipeline the
long-form box already runs.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B. Low-noise: `pudgy-v8-merged` (v7fixed + v8@550). High-noise: v2 golden. Phone: E5 embedding. The production setup in [`README` v9](../v9/README.md) |
| **Trains** | **Stage 1:** a fresh **high-noise** LoRA (rank 16), with the low-noise expert frozen. **Stage 2 (only if needed):** a low-noise LoRA at a lower lr, with a preservation loss. Two files, each gated on its own and each bisectable |
| **Data** | `v10_scenes_748`: **745 trainable windows** (18.2 min, 157 skits) from the clean `iteration_7` bases, **plus `v9b_535`'s 348 skit windows** (U7), plus 35% v8 replay: ≈ 1,570 clips after the holdout (§3). The 217 Instagram reels were the first plan; the clean bases replaced them on 2026-11-04 (§2) |
| **Output** | an end-to-end reel = a premise → a shot list → a keyframe per shot → one generation per shot (I2V or FLF) → QA → cuts, caption text, watermark and music added **in post** |
| **Budget** | ≈ 2.5–3 weeks of wall clock on one A100 80 GB: Phase 0 ≈ 1.5 days, prep ≈ 1 week (mostly CPU and curation), Stage 1 ≈ 1 day of training + ≈ 1.5 days of eval, Stage 2 the same if it runs |
| **If v10 fails its gate** | production stays v8merged + E5, and the v10 work ships as pipeline gains (keyframes, FLF, QA). There is always a final deliverable |
| **Status** | **draft, 2026-11-01; data curated 2026-11-04.** U1 is answered: the client sent clean `_Base` masters as `Data/raw/iteration_7` (293 videos; 190 new skits). 186 were curated and every trainable span independently verified (`finetune/wan/scenes/CURATION_v10.md`, `VERIFY_v10.md`). Captions: `scenes/captions_v10_windows.jsonl`, from `compile_captions_v10.py`: **637 kept windows (15.4 min, 144 skits)**, plus 108 U3 camera-move windows (2.8 min): **745 train**. 3 U5 lookalike windows (Fixer) wait on the client. **U3 = yes** (2026-11-04). **The client answered U5 and U9 on 2026-11-05** (§7). `v9b_535`'s windows checked for the yellow-foot design on 2026-11-09: none (U9). U2, U4 and U6–U8 are still open. Captions moved to `Data/processed/v10_scenes_748/`. **Next:** Phase 0, then the holdout and clip cutting on the GPU box |

---

## 1. What the evidence forces

Every row is a measurement from an earlier run, not a preference.

| # | Measured | → v10 decision |
|---|---|---|
| 1 | **v9b-1** trained a rank-32 **low-noise** LoRA on 348 skit windows + 35% replay. By step 100 every v8 motion label had lost margin, crying by −0.10. Held-out motion fell 9 → 6/12 (R1 ✗) and R2 was 1.066 at t = 0.85 ([REPORT_v9b](../../docs/training_reports/v9/REPORT_v9b.md)) | Don't put the scene behaviour into the low-noise expert first. That is where v8's labels live, and they are what regressed |
| 2 | v9b-1's scene gain **didn't use the caption**: text-attributable gain −0.0008, CI spanning 0, scene caption choice 6/15 → 6/15. v9c-phone showed the same thing (99.6% caption-independent) | Captions must carry what frame 0 doesn't (ordered beats, entrances, expression changes). The stage must also be measured on whether it uses them (G3) |
| 3 | v9c2's x₀ probe: what a shot ends up showing is **settled before the high-noise → low-noise switch** (12/12 at t = 0.9, 5/6 at 0.95) | Staging, travel and beat order are high-noise decisions, so **Stage 1 trains the high-noise expert**. That also leaves the low-noise expert, and v8's Tier-1 labels, **unchanged by construction** |
| 4 | v5: expression data on the high-noise expert **collapsed prompt control** (SSIM 0.9692 between opposite prompts). ANALYSIS_v7 §4: in-place actions have ≈ 0 between-label variance at high noise | That was in-place data, which carries no layout signal. Production skits do: positions, entrances and staging differ per caption. The v5 failure is still the main risk of Stage 1, so G3 (does it use text?) and G4 (do v8 prompts still separate?) are gates, not reports |
| 5 | v9b-1's damage was complete by step 100, and step 50 was never scored. v8 plateaued by step 300; v9c by step 125 | Lower lr and rank than v9b; **score from step 50**, every 50 |
| 6 | 35% replay didn't protect motion in v9b-1. The preservation loss exists (v9c2 pivotal, with the block-swap fix), and in expectation it equals replay, with a zero-variance target (RESEARCH_v9c_fixes C9) | Keep 35% replay **and** add preservation on replay prompts. Neither alone is trusted |
| 7 | E5 was trained through v8merged (low) + **golden (high)** | Any change to the high-noise expert means re-checking the phone with ID1 (G5) |
| 8 | Kandinsky 5.0 (expv1) "not better than v8@550". MiniMax H3 (expv2) failed its VAE gate on thin outlines | The base stays Wan2.2 + v8merged. No migration in the final stage |
| 9 | The long-form box made 7 skit remakes and 16 single-action reels with **v8merged + FLF + QA + reseeding**; about 20 takes were rejected on the way ([REPORT_longform_client](../../docs/training_reports/longform_client/REPORT_longform_client.md)) | End-to-end output is a **pipeline** product. v10 trains better shots; the pipeline makes the reel |
| 10 | Every generation unit is ≤ f81 (3.4 s) today. Reel shots are typically 1.5–2.6 s, but the long tail runs past 5 s (§2) | Add an **f121** bucket (5.0 s) **if** Phase 0 shows A14B holds up at 31 latent frames. At 480×848 it is 49k tokens, below the 61k replay f57 bucket that already fits at swap 24 |

---

## 2. The reels, measured (2026-11-01)

> **Superseded as the training source on 2026-11-04.** The client sent the clean `_Base`
> exports (`Data/raw/iteration_7`, route A, U1). 89 of the 217 reels now have a clean base and
> 128 older ones have none (`reels_without_base.txt`). v10 trains on the bases (§3). This
> section is kept as the record of why the masters were asked for. The reels stay the few-shot
> library for §6.

Source: `~/Documents/PudgyReels/`. 217 originals plus `playable/` (H.264 re-encodes of the same
217); `links.txt` holds 217 `instagram.com/reel/<code>` URLs. **The originals are the source of
truth**: `playable/` is a second-generation encode.

| | |
|---|---|
| **Total** | 217 reels, **51.9 min**. Median 11.8 s (p10 6.7, p90 19.0); 74 under 10 s, 123 at 10–20 s, 20 at ≥ 20 s (max 173 s) |
| **Resolution** | **133 at 720×1280**, 77 at 1080×1920, 7 others (2 landscape, 1 square, 4 odd sizes) |
| **Frame rate** | **151 at 24 fps**, **65 at 30 fps**, 1 at 60 fps. The 30 fps cadence (24→30 pulldown or native 30) is **unresolved**: a duplicate-frame test can't tell, because 24 fps reels hold frames just as often (40%, held drawings) |
| **Codec / quality** | 140 H.264, 77 VP9; **median 663 kbps** (p10 443). Instagram-grade compression on flat pastel fills |
| **Audio** | AAC on 216 of 217 (music, SFX). Not trainable (Wan is video-only); kept for timing |
| **Overlap with the skits** | **1 of 217**: `DInIYt4Nb1C` (posted 2025-04-19) is `PP-Burrito-Base.mp4`. Checked on a 256-bit dHash of the caption- and watermark-free band (32–74% of height; reels at 2 fps, skits at 4 fps), then by eye for the top 5. The other 4 candidates, and the 8 partial matches of the earlier full-frame hash, are false positives (similar plain backgrounds). The two sets cover different periods: the reels were **posted 2022-04-27 → 2025-05-29**, while the skits were **exported 2025-04 → 2026-06** (mp4 `creation_time`; Burrito: exported 2025-04-11, posted 8 days later). No reel duplicates another |
| **Shots** | scene detection gives **691 shots** (threshold 12) to **1,331** (threshold 5): about 3–4 shots per reel, median 1.5–2.6 s. The detector misses cuts between similar pale sets and fires on whip pans, so the curator pass decides, as in v9b |
| **Time by shot length** (threshold 12 / 5) | < f21: 0.9 / 2.4 min · f21–f49: 4.7 / 10.6 · f49–f81: 4.8 / 10.9 · f81–f121: 5.2 / 7.8 · ≥ f121: 36.3 / 20.2 |

**What is in them** (24 random reels × 4 frames, by eye; a sample, not a census):

| finding | in the sample | consequence |
|---|---|---|
| **Burned-in meme captions** ("When she takes a midday nap", "POV: …"), white with a dark outline, across the upper third; they change between shots | **21 / 24** | must not be trained as content: the model would learn to draw text. §3.2 |
| **`@pudgypenguins` watermark**, faint, bottom centre | nearly all | same |
| **Characters beyond Pax and Polly**: a **yellow penguin with earmuffs and a scarf** (frequent), a classic **black-and-white penguin**, a **bearded old penguin in a cap**, a white penguin in a bunny hood, crowds | yellow in ~6 / 24 | A **blue penguin in a red scarf and overalls** also appears in several reels next to Pax. **Client, 2026-11-05 (U5):** only the black-and-white penguin is acceptable; the yellow earmuffs penguin, the bearded old penguin and the red-scarf penguin stay out of training |
| **Live action** (a person with a giant plush) | 1 / 24 | drop |
| UI inserts (phone screens, DM lists, calendars), title cards ("JANUARY 2024", a to-do list), whip-pan blur | several | drop, as in v9b's rules |
| Sets: bedrooms, igloo exteriors, kitchen, shop aisles, living room; plain pale-blue grounds with furniture | most | the v9b setting rules apply as written |

**Usable footage, estimated.** v9b kept 48% of the skits' time (20.3 → 9.7 min). The reels add
captions, UI and live action, so the estimate is **35–50% → 18–26 min**. That is about 2–2.7× v9b's
footage, or ≈ 27–35 min together with `v9b_535`. It is enough to shift staging, acting and
pacing priors. It is **not** enough to teach new physics or new characters from text alone.
**Measured on the clean bases instead (2026-11-05):** 47% of the frames kept, 18.2 min
trainable; ≈ 27 min together with `v9b_535`, before the holdout.

---

## 3. Data: `v10_scenes_748` + `v9b_535`

### 3.1 Ingest: done (2026-11-04)

`Data/raw/iteration_7/`: 293 clean `_Base` exports at 1080×1920 (13.2 Mbps median, about 20×
the Instagram copies), with `README.md`, `CHANGELOG.md` and `reel_matches.tsv`, mirrored to
Azure. 99 are byte-identical re-sends of iteration_1/2 and 190 are new skits. The 217 reel
files stay in `~/Documents/PudgyReels/`, not ingested: they are needed only as the §6 few-shot
library and for the next ask (the 128 reels without a base).

### 3.2 Clean frames: route A delivered

The bases carry no meme caption and no watermark, so no loss mask (route B) or inpainting
(route C) is needed. The few spans where a base still showed a caption, a handle or an end card
were dropped as `graphic_overlay` or `ad_merch`. Routes B and C stay the fallback only if the
128 reels without a base are ever used. G2's OCR check stays a gate either way.

### 3.3 Curation: done (2026-11-04, client answers applied 2026-11-05)

186 of the 190 new skits curated with the v9b machinery (`finetune/wan/scenes/`), the v9b
schema and [`CURATION_v10.md`](../../finetune/wan/scenes/CURATION_v10.md). Every trainable span
was checked by an independent verifier ([`VERIFY_v10.md`](../../finetune/wan/scenes/VERIFY_v10.md));
about 1 in 3 got a fix. Camera moves are tagged (U3), side characters follow the client's
ruling (U5) and the yellow-foot design is out (U9). The reel-only fields (caption boxes, the
OCR'd premise) weren't needed.

### 3.4 Normalisation (at prep, on the GPU box)

- **fps (U4):** 739 of the 745 trainable windows are at 24 fps. The other 6 (238 frames) are
  `PP_CuteFit-BASE.mp4` at 29.97 fps: cadence test at prep, inverse telecine if pulldown, drop
  if native 30. Never resampled (v8: resampling hand-drawn animation judders or ghosts outlines).
- **Size:** 1080×1920 → 480×854 (Lanczos) → centre crop 480×848, as v9b. 6 skits have 4 px
  black columns at both edges (`PILLARBOX` in their spans): crop them first.
- **Buckets:** f21 · f33 · f49 · f81, plus **f121** if Phase 0 #3 passes. Windows are
  non-overlapping, never cross a cut, and need the lead in at least half the window.

### 3.5 Captions: done

`Data/processed/v10_scenes_748/captions_v10_windows.jsonl` (748 rows, 745 trainable), built by
`finetune/wan/scenes/compile_captions_v10.py` with v9b's compiler (`scenes/captions_v9b.py`)
unchanged. That compiler still reproduces all 348 of v9b's captions exactly. Structure:
`STYLE. CHARACTERS (registry anchors, screen side, outfit). SETTING. BEATS in order.
FRAMING, ANGLE and CAMERA clause.` Every caption passes prep's lint: D10 (no v8
action clause verbatim), no lead colour words outside the anchors, no on-screen text.

### 3.6 Mix and holdout

| part | source | windows | share |
|---|---|---|---|
| iteration_7 scene windows | `v10_scenes_748`, trainable rows minus the holdout | ≈ 670 | ≈ 43% |
| v9b skit windows | `v9b_535`'s 348 scene windows (U7; v9b's 6 held-out skits stay out) | 348 | ≈ 22% |
| v8 replay | `prep_v9` builders, stratified per (character, label), D10 anchors | ≈ 550 | **35%** |
| **total** | | **≈ 1,570** | |

**Holdout:** ≈ 16 whole iteration_7 skits (≈ 10% of 157), chosen at prep and stratified by cast
(single and two-character; 101 skits have a two-shot), set type (flat vs set) and camera (static
vs a move; 64 skits have one), plus v9b's 6 held-out skits. Their rows are removed before
training. From these come the Tier-1 scene windows (with distractor captions) and the Tier-2
shot suite.

### 3.7 v10's new data against v9b's (2026-11-05)

Same pipeline (480×848, 24 fps, the same compiler) and no shared footage: no common files or
names, and the closest frame hashes are different skits.

| | v9b (trained in v9b-1) | v10 new (`v10_scenes_748`) |
|---|---|---|
| source | iteration_1 + 2, 99 skits curated | iteration_7, 186 skits curated |
| scene windows | 348 (after a 6-skit holdout) | 745 (before the holdout) |
| skits / minutes | 78 / 8.7 min | 157 / 18.2 min |
| buckets f21 / f33 / f49 / f81 | 46 / 21 / 24 / 10% | 47 / 21 / 23 / 9% |
| both leads / Pax only / Polly only | 45 / 34 / 20% | 44 / 30 / 26% |
| camera moves | 0% | 14.5% (push-in 79, pan 17, pull-back 10, other 2) |
| framing medium wide / wide / medium close-up or tighter | 43 / 18 / 14% | 60 / 9 / 9% |
| high angle | 1% | 5% |
| side character | 2% | 4% |
| yellow-foot design | none (checked 2026-11-09, U9) | excluded (3,154 frames) |

Label words that stay thin even combined: surprised (10 + 6 windows), confused (5 + 6),
laughing (10 + 5).

---

## 4. Recipe

| | Stage 1: high-noise | Stage 2: low-noise (conditional) |
|---|---|---|
| when | always | only if Stage 1 passes and leaves a **detail** gap: small leads blurred at scene scale, outfits, side-character faces (Tier-2 review) |
| expert / t | high-noise, t ∈ [900, 1000] | low-noise, t ∈ [0, 900] |
| base merge | `lora_highnoise_GOLDEN_ep40` → `--base_weights` (as v8 did with v7fixed) | `pudgy-v8-merged` |
| LoRA | fresh, rank 16 / α 16 | fresh, rank 16 / α 16 |
| lr | **3e-5**, warmup 20, cosine → 0.1 | **2e-5**, same schedule |
| batch | 1 × grad-accum 8 | same |
| steps | **≤ 300**, save every 25, Tier 1 from step 50 | **≤ 200**, the same cadence |
| preservation | λ = 1 on replay samples (frozen-base target; the v9c2 implementation with the block-swap fix) | same |
| loss | plain MSE. The motion-weighted loss only after a 50-step pilot (v9 D8) | plain MSE |
| memory | `BLOCKS_TO_SWAP=24`. Smoke test on the heaviest bucket (replay f57 1024², or f121 if added); gate on the `memory mapping failed with OOM` count, not the peak | same |

**Why rank 16 and lr 3e-5:** v9b's rank 32 / lr 5e-5 did all its damage by step 100. This is
0.6× the lr and half the rank, so a smaller per-step weight change (the exact size is unmeasured;
Tier 0's ‖τ‖ trajectory will show it), with checkpoints dense enough to stop early. With about
1,570 clips (≈ 1,020 scene windows + 35% replay, §3.6), 300 steps × 8 is ≈ 1.5 epochs. v9b was already
at 3.0 epochs by step 200.

**Compute** (A100 80 GB, v9b's measured 187 s per step at 480×848): Stage 1 ≈ 16–20 h for 300
steps (f121 adds ≈ 1.3× on its samples). Tier 1 ≈ 85 min per checkpoint per expert. Tier 2 at 480×848
f81 ≈ 8–12 min per clip. **A second GPU roughly halves the eval.** Verify every number with a
20-step smoke test.

---

## 5. Evaluation and gates (fixed before the data)

### 5.0 Phase 0: before any v10 training (≈ 1.5 GPU-days)

| # | test | decides |
|---|---|---|
| 1 | **Tier 2 on the existing v9b@100 and v9b@200** (Azure `v9b/weights/`) vs v8merged, on the 8-job scene suite plus 52 v8 `core` prompts. v9b-1 was never generated | whether low-noise scene training made scenes *visibly* better despite failing R1. If yes, U2's trade-off is real and Stage 2 moves up. If not, the low-noise route is closed |
| 2 | **Baseline**: v8merged + golden on the held-out skits' shots (first frame + v10 caption, 3 seeds; and FLF with the real last frame) | the "before" for G1 and G2 |
| 3 | **f121 check**: v8merged at 121 frames, 480×848, 6 prompts × 3 seeds vs the same at f81: drift, loop-back, end-of-clip degradation | whether the f121 bucket exists (U6) |
| 4 | ~~**Compression check**~~ (reel vs clean base through the VAE) | **not needed**: v10 trains on the clean masters (13.2 Mbps median), not the 663 kbps reels (U1) |
| 5 | **30 fps cadence**: per-frame difference periodicity (period 5 ⇒ pulldown), now on one file: `PP_CuteFit-BASE.mp4` (29.97 fps, 6 windows) | U4; cheap enough to run at prep |

### 5.1 Gates (per stage, per checkpoint; the winner is the passing checkpoint with the best G1)

| id | check | bar |
|---|---|---|
| **G1** | **Production preference**: blind A/B on held-out skit shots (same first frame, caption and seed), v10 vs v8merged, judged by the user/team plus a Claude judge pre-validated against them | **v10 preferred**, exact one-sided binomial at α = 0.05: **≥ 48 of 80** paired judgements (60%). Sized for 0.85 power at a true 65% preference; 67 judgements would give only 0.78 |
| **G2** | **Automatic quality** on the same clips: duplicate leads = 0 (`dupcheck`); zoom ≤ 3% on static-camera prompts; motion ratio 0.7–1.3 vs the real shot; identity (DINO crop similarity to frame 0) ≥ v8merged; **no rendered text or watermark** (OCR on generated frames finds nothing absent from frame 0) | all hold |
| **G3** | **Uses the caption**: high-noise Tier 1 (`dcls_asset_v9.py --expert high`, t 0.95 / 0.92) on held-out scene windows, true caption vs distractors (beat order swapped, leads' actions exchanged, another shot's action) | text-attributable gain < 0, CI excluding 0. **v9b-1 failed exactly this** |
| **G4** | **v8 prompts still work**: Tier-2 `core` (52 prompts) label 1-NN per kind within 0.1 of v8merged; `compose` motion ≥ v8merged. Stage 2 also gets low-noise R1 and R2 at the bar U2 sets. Stage 1 leaves the low-noise Tier 1 unchanged by construction; this is verified once | all hold |
| **G5** | **Phone**: ID1 on the 24-clip confirm tier with E5 on the v10 stack | ≤ 1 / 24 (E5 alone: 0 / 24) |

**Stopping rule:** if no checkpoint up to step 100 passes G3 and G4, the stage stops. That was
v9b's lesson: the damage is in by step 100.

---

## 6. End to end: from premise to reel

v10 trains shots. A reel is assembled by the pipeline in `finetune/wan/longform_client/`
(keyframes, FLF, QA, reseed, colour match, assembly), extended:

1. **Premise → shot list.** A Claude prompt layer (`docs/documents/Prompt_Compiler_Spec.md`)
   turns a meme premise into 3–6 shots, each a v10 caption compiled from the registry. **The
   reels with a clean base are its few-shot library** (89 of 217, `iteration_7/reel_matches.tsv`):
   premise (OCR'd from the reel) ↔ shot list (the base's curation) ↔ video, so pacing (≈ 3–4
   shots, 1.5–2.6 s each) is calibrated on the client's own reels.
2. **Keyframes.** I2V needs a first frame per shot. Three sources: the client's frames;
   composites (plates + character cut-outs, `keyframes.py`); or the previous shot's last frame
   within a set. **New sets and new staging need a keyframe generator** (an image LoRA on frames
   from the clean bases, which carry no text). This is the main gap for truly new reels: **workstream v10-K**,
   which can run alongside Stage 1.
3. **Generate** each shot with v10 (I2V, or FLF where the layout must hold: two characters + props,
   as v8 found).
4. **QA** (`qa_take`, dupcheck, zoom, the OCR text check) → reseed failures, as the long-form
   box did (about 20 takes rejected).
5. **Post**: hard cuts, the **meme caption as crisp overlay text**, the watermark and music. All
   deterministic, never generated.

---

## 7. Decisions for the user (U1–U8)

| # | decision | recommendation | why |
|---|---|---|---|
| U1 | **Ask the client for clean masters** of the 217 reels (no captions, no watermark, 1080×1920, 24 fps)? | ✅ **done: the client sent 293 clean `_Base` exports on 2026-11-04** (`iteration_7`, for training). 89 of the 217 reels now have a base; the 128 older ones without one are listed for a later ask | it removed the biggest data risk: no loss mask, no inpainting, no 663 kbps footage |
| U2 | **What "best" means** for the final gate: production preference (G1) first, with v8 label guardrails (G4), or v9's strict per-label R1 | **G1 first, G4 as the guardrail**; R1 reported, not gating Stage 1 | v9's R1 is measured on 1024² single-character studio clips. The final product is 9:16 reels. Phase 0 #1 shows what the strict bar costs |
| U3 | Camera moves: include tagged spans (≤ 25%, explicit camera clause) or static only | **include, tagged**: ✅ **decided yes, 2026-11-04** (108 windows, 14.5% of the 745 trainable; `processed/v10_scenes_748`) | push-ins and pans are part of the reel style; v9b set them aside for "a later camera stage", and this is the last stage. G2's zoom bar guards static prompts |
| U4 | 30 fps footage: was 65 of 217 reels; now **one base**, `PP_CuteFit-BASE.mp4` (29.97 fps, 6 windows) | inverse telecine if pulldown; **drop if native 30** | no resampling of hand-drawn timing (v8) |
| U5 | New characters: names and registry phrases (yellow earmuffs, black-and-white, grandpa, bunny hood, …); is the **blue penguin in a red scarf and overalls** Pax or someone else? | ✅ **client answered, 2026-11-05.** Out of training: the red-scarf penguin, yellow earmuffs and the old bearded penguin (no names), and the bunny costume (a one-off). Accepted: the black-and-white penguin. The pearl-necklace penguin is Polly dressed up, so Purse, CancelPlans and PoopBreak train. **Still open:** Fixer's stubble and goggles (3 windows held), and whether other unnamed side penguins stay (25 trainable windows, kept for now) | a Pax lookalike in training teaches "two Paxes", the Polly's-mum case |
| U6 | f121 (5 s) bucket | **only if Phase 0 #3 passes** | A14B has only seen 21 latent frames; 31 is extrapolation |
| U7 | Include `v9b_535`'s skit windows | **yes** | curated and verified already, the same production domain, no footage shared with iteration_7 (§3.7), and free of the yellow-foot design (U9) |
| U8 | Stage 2 (low-noise) | **conditional** on Stage 1 passing and a visible detail gap | the low-noise expert is where v9b-1 regressed |
| U9 | The **yellow-foot variant** in iteration_7: current flat drawing, but yellow feet and beak (hue ≥ ~42°). About 12 skits, ~3,000 frames: Affection, Bromance, LastBite, QUALITYTIME, SkinCare, Upset, SAMEROOM, Blanket, ALREADYSEENIT, Wash, … | ✅ **exclude; client answered, 2026-11-05:** an older design, no longer acceptable (`off_style`, `OLD_DESIGN:` notes). **`v9b_535`'s 348 windows checked 2026-11-09: none.** A beak-hue probe (`scenes/check_beak_hue.py`, 6 frames per window) read a median 26° (salmon) on 307 windows. Its 5 hits at ≥ 42° were props (sparkles, a car jack), and the 41 windows with no readable beak (night, close-ups, back views) are current design by eye. Calibrated on iteration_7: 23 of 24 yellow-foot spans ≥ 42°, 0 of 120 kept spans | captions never name foot colour, so including it would teach yellow feet beside the current salmon/amber. Recorded in `finetune/wan/scenes/CURATION_v10.md` |

---

## 8. Risks

- **The high-noise LoRA could collapse prompt control (v5).** G3 and G4 gate it from step 50, and
  the file is bisectable (drop it to get v8 back).
- **Text and watermark leak**: low now. The bases are clean, and the few spans that still showed
  text were dropped. The G2 OCR check stays a hard bar.
- **Compressed footage teaches artifacts**: removed by the masters (U1).
- **New characters at low data**: a side character seen in 10 reels won't be drawn reliably from
  text. In I2V the look comes from the keyframe ("one name, one look"), and captions keep names consistent.
- **Small n in G1** if fewer held-out shots survive curation. 80 judgements is the floor. Add seeds
  before relaxing the bar.
- **Keyframes stay the bottleneck** for genuinely new reels (§6.2). v10 improves shots; v10-K
  decides how new a reel can be.
- **Calendar**: curation, the long pole, is done. What's left of prep is the holdout and cutting
  clips on the GPU box.
- **Late client answers**: the unnamed side penguins (U5) sit in 25 trainable windows. If the
  client rules them out, those rows come out before training; nothing else changes.

## 9. Order

| step | needs | can start |
|---|---|---|
| ingest + profile + mirror (§3.1) | — | ✅ done 2026-11-04 |
| client ask: masters, character names (U1, U5, U9) | — | ✅ answered 2026-11-04/05; Fixer and the other unnamed side penguins still open, **not blocking** |
| curation + captions (§3.3, §3.5) | — | ✅ done 2026-11-05 |
| `v9b_535` yellow-foot check (U9) | — | ✅ done 2026-11-09: clean |
| user decisions U2 (gate) and U7 (reuse `v9b_535`) | — | **now** |
| Phase 0 #1–#3 | a GPU; v9b checkpoints on Azure | **now** |
| prep: holdout, clip cutting, #5 cadence on CuteFit, replay build (§3.4, §3.6) | a GPU box; Phase 0 #3 for f121 | ≈ 1–2 days |
| **Stage 1** (high-noise) → gates | prep, Phase 0 | after prep |
| Stage 2 (low-noise) → gates | Stage 1 passing + a detail gap | after Stage 1 |
| v10-K keyframe LoRA | frames from the clean bases | alongside Stage 1 |
| final: the best of {v10, v8merged} + pipeline, with sample reels | gates | end |
