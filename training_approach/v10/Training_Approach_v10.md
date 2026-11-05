# Training Approach v10: production reels, end to end (final stage)

**In one sentence:** fine-tune the production model (`pudgy-v8-merged` + E5) on the client's
published Instagram reels, together with the 99 skits, so that each *shot* behaves like a
production shot. Do it through the **high-noise expert first**, where layout, staging and motion
are decided (the expert v9b-1 left frozen). Then assemble end-to-end reels with the shot pipeline the
long-form box already runs.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B. Low-noise: `pudgy-v8-merged` (v7fixed + v8@550). High-noise: v2 golden. Phone: E5 embedding. The production setup in [`README` v9](../v9/README.md) |
| **Trains** | **Stage 1:** a fresh **high-noise** LoRA (rank 16), with the low-noise expert frozen. **Stage 2 (only if needed):** a low-noise LoRA at a lower lr, with a preservation loss. Two files, each gated on its own and each bisectable |
| **Data** | `v10_reels`: the 217 reels in `~/Documents/PudgyReels` (51.9 min, measured §2), curated to static or tagged-camera spans, **plus the already curated `v9b_535` skit windows**, plus 35% v8 replay |
| **Output** | an end-to-end reel = a premise → a shot list → a keyframe per shot → one generation per shot (I2V or FLF) → QA → cuts, caption text, watermark and music added **in post** |
| **Budget** | ≈ 2.5–3 weeks of wall clock on one A100 80 GB: Phase 0 ≈ 1.5 days, prep ≈ 1 week (mostly CPU and curation), Stage 1 ≈ 1 day of training + ≈ 1.5 days of eval, Stage 2 the same if it runs |
| **If v10 fails its gate** | production stays v8merged + E5, and the v10 work ships as pipeline gains (keyframes, FLF, QA). There is always a final deliverable |
| **Status** | **draft, 2026-11-01; data curated 2026-11-04.** U1 is answered: the client sent clean `_Base` masters as `Data/raw/iteration_7` (293 videos; 190 new skits). 186 were curated and every trainable span independently verified (`finetune/wan/scenes/CURATION_v10.md`, `VERIFY_v10.md`). Captions: `scenes/captions_v10_windows.jsonl`, from `compile_captions_v10.py`: **637 kept windows (15.4 min, 144 skits)**, plus 108 U3 camera-move windows (2.8 min): **745 train**. 3 U5 lookalike windows (Fixer) wait on the client. **U3 = yes** (2026-11-04). **The client answered U5 and U9 on 2026-11-05** (§7). U2, U4 and U6–U8 are still open. Captions moved to `Data/processed/v10_scenes_748/` |

---

## 1. What the evidence forces

Every row is a measurement from an earlier run, not a preference.

| # | Measured | → v10 decision |
|---|---|---|
| 1 | **v9b-1** trained a rank-32 **low-noise** LoRA on 348 skit windows + 35% replay. By step 100 every v8 motion label had lost margin, crying by −0.10. Held-out motion fell 9 → 6/12 (R1 ✗) and R2 was 1.066 at t = 0.85 ([REPORT_v9b](../../docs/training_reports/v9/REPORT_v9b.md)) | Don't put the scene behaviour into the low-noise expert first. That is where v8's labels live, and they are what regressed |
| 2 | v9b-1's scene gain **didn't use the caption**: text-attributable gain −0.0008, CI spanning 0, scene caption choice 6/15 → 6/15. v9c-phone showed the same thing (99.6% caption-independent) | Captions must carry what frame 0 doesn't (ordered beats, entrances, expression changes). The stage must also be measured on whether it uses them (G3) |
| 3 | v9c2's x₀ probe: what a shot ends up showing is **settled before the high-noise → low-noise switch** (12/12 at t = 0.9, 5/6 at 0.95) | Staging, travel and beat order are high-noise decisions, so **Stage 1 trains the high-noise expert**. That also leaves the low-noise expert, and v8's Tier-1 labels, **unchanged by construction** |
| 4 | v5: expression data on the high-noise expert **collapsed prompt control** (SSIM 0.9692 between opposite prompts). ANALYSIS_v7 §4: in-place actions have ≈ 0 between-label variance at high noise | That was in-place data, which carries no layout signal. Reels do: positions, entrances and staging differ per caption. The v5 failure is still the main risk of Stage 1, so G3 (does it use text?) and G4 (do v8 prompts still separate?) are gates, not reports |
| 5 | v9b-1's damage was complete by step 100, and step 50 was never scored. v8 plateaued by step 300; v9c by step 125 | Lower lr and rank than v9b; **score from step 50**, every 50 |
| 6 | 35% replay didn't protect motion in v9b-1. The preservation loss exists (v9c2 pivotal, with the block-swap fix), and in expectation it equals replay, with a zero-variance target (RESEARCH_v9c_fixes C9) | Keep 35% replay **and** add preservation on replay prompts. Neither alone is trusted |
| 7 | E5 was trained through v8merged (low) + **golden (high)** | Any change to the high-noise expert means re-checking the phone with ID1 (G5) |
| 8 | Kandinsky 5.0 (expv1) "not better than v8@550". MiniMax H3 (expv2) failed its VAE gate on thin outlines | The base stays Wan2.2 + v8merged. No migration in the final stage |
| 9 | The long-form box made 7 skit remakes and 16 single-action reels with **v8merged + FLF + QA + reseeding**; about 20 takes were rejected on the way ([REPORT_longform_client](../../docs/training_reports/longform_client/REPORT_longform_client.md)) | End-to-end output is a **pipeline** product. v10 trains better shots; the pipeline makes the reel |
| 10 | Every generation unit is ≤ f81 (3.4 s) today. Reel shots are typically 1.5–2.6 s, but the long tail runs past 5 s (§2) | Add an **f121** bucket (5.0 s) **if** Phase 0 shows A14B holds up at 31 latent frames. At 480×848 it is 49k tokens, below the 61k replay f57 bucket that already fits at swap 24 |

---

## 2. The reels, measured (2026-11-01)

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

---

## 3. Data: `v10_reels`

### 3.1 Ingest (first, before any analysis)

1. Copy the 217 originals and `links.txt` into `Data/raw/iteration_7_reels/` (raw, never
   edited), with a `README.md` and `CHANGELOG.md`. Mirror it to Azure with
   `scripts/mirror_data_to_azure.py` (`.venv-dashboard/bin/python`). `playable/` is not mirrored;
   it can be rebuilt.
2. MD5 manifest. Record each reel's shortcode, URL, resolution, fps, duration and bitrate (the
   §2 probe, as `finetune/wan/reels/profile_reels.py`).

### 3.2 Clean frames: captions and watermark

The order of preference:

| route | what | cost | risk |
|---|---|---|---|
| **A. Client masters** (recommended, U1) | the client made these reels. Ask for the **caption-free, watermark-free exports** at 1080×1920, 24 fps, the same `_Base` exports iteration_1/2 arrived as | a client ask; nothing on our side | timing |
| B. Loss mask | OCR the caption and watermark boxes per frame (dilated) → a pixel mask → a latent mask (÷ 8). Those tokens are excluded from the loss. Needs the per-token loss-weight patch, the same one the motion-weighted loss needs (not yet in training) | ≈ 2–3 days | frame 0 still shows the text. At inference our keyframes carry none, but text-bearing first frames must never reach eval |
| C. Inpaint | video inpainting of the text boxes; the text sits mostly over flat pale ground | ≈ 2 days + review | smears on outlines when text crosses a character. Those spans get dropped |

**Recommendation:** start route A on day 1, and build route B in parallel as the fallback. Route
C only for spans where text sits on flat ground. **A span whose text crosses a lead is dropped** under
B and C. Gate G2 checks for rendered text either way.

### 3.3 Curation (the v9b machinery, extended)

Reuse `finetune/wan/scenes/` (curate_tools, `span_motion_check.py`, `curate_check.py`, the
CURATION brief): curators by batch, then verification passes over every kept span. New fields
and rules for reels:

- **caption and watermark boxes** per span, frame-ranged (for route B/C and for G2);
- **camera**: `static` or a tagged move (`push_in`, `pull_back`, `pan_l/r`, `whip`); see U3;
- **cast**: leads, plus side characters described by look. The client accepted the black-and-white
  penguin and ruled out yellow earmuffs, grandpa and the red-scarf penguin (U5). Unresolved
  lookalikes are dropped, as with Polly's mum in v9b;
- **drop**: live action, UI inserts, title and text cards, ads and merch, lead depictions, effects
  (vignettes, time-lapses, colour washes), cross-dissolves;
- **premise**: the OCR'd meme line per reel ("When your package says 'arriving soon'…"). It is
  **metadata for the story layer (§6), never in a training caption**.

### 3.4 Normalisation

- **fps (U4):** 24 fps stays. The 30 fps reels get a cadence test in Phase 0. A 24→30
  pulldown gets inverse-telecined to exact 24. A native 30 fps reel is dropped, not resampled:
  v8 measured that resampling hand-drawn animation judders or ghosts outlines.
- **Size:** → 480×854 (Lanczos) → centre crop 480×848, as v9b. Landscape and square reels are
  dropped.
- **Buckets:** f21 · f33 · f49 · f81, plus **f121** if Phase 0 #3 passes. Windows are
  non-overlapping and never cross a cut. The lead is visible in at least half the window.

### 3.5 Captions

v9b's compiler (`scenes/captions_v9b.py`), unchanged in form:
`STYLE. SETTING. CHARACTERS (registry anchors, screen side, outfit). BEATS in order
("first …, then …"), with a speed word. EXPRESSION changes. CAMERA clause.`

- D10 still holds: **no v8 action clause verbatim**, with the lint in place.
- The camera clause is "fixed camera, locked-off shot, …" for static spans, and explicit wording for
  tagged moves (U3).
- No on-screen text is transcribed, and nothing names the meme caption.
- A scene sheet per reel, so consecutive shots share wording.

### 3.6 Mix and holdout

| part | source | share |
|---|---|---|
| reel windows | `v10_reels` (curated) | ≈ 45% |
| skit windows | `v9b_535`'s 348 scene windows (curated and verified; the 6 held-out skits stay out) | ≈ 20% |
| v8 replay | `prep_v9` builders, stratified per (character, label), D10 anchors | **35%** |

**Holdout:** 20 whole reels (≈ 10%), stratified by cast (single and two-character, with and
without the yellow penguin), set type and camera tag, plus v9b's 6 held-out skits. From these
come the Tier-1 scene windows (with distractor captions) and the Tier-2 shot suite.

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
1,600 clips (≈ 1,050 scene windows + 35% replay), 300 steps × 8 is ≈ 1.5 epochs. v9b was already
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
| 2 | **Baseline**: v8merged + golden on the held-out reels' shots (first frame + v10 caption, 3 seeds; and FLF with the real last frame) | the "before" for G1 and G2 |
| 3 | **f121 check**: v8merged at 121 frames, 480×848, 6 prompts × 3 seeds vs the same at f81: drift, loop-back, end-of-clip degradation | whether the f121 bucket exists (U6) |
| 4 | **Compression check**: the same skit shot from the clean `_Base` export and from its Instagram reel (the one exact pair: `DInIYt4Nb1C` ↔ `PP-Burrito-Base.mp4`), both through the VAE at 480×848: lead-box PSNR and banding by eye | whether 663 kbps footage is trainable as delivered, or needs masters (U1) |
| 5 | **30 fps cadence**: per-frame difference periodicity (period 5 ⇒ pulldown) on all 65 | U4 |

### 5.1 Gates (per stage, per checkpoint; the winner is the passing checkpoint with the best G1)

| id | check | bar |
|---|---|---|
| **G1** | **Production preference**: blind A/B on held-out reel shots (same first frame, caption and seed), v10 vs v8merged, judged by the user/team plus a Claude judge pre-validated against them | **v10 preferred**, exact one-sided binomial at α = 0.05: **≥ 48 of 80** paired judgements (60%). Sized for 0.85 power at a true 65% preference; 67 judgements would give only 0.78 |
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
   217 reels are its few-shot library**: premise (OCR) ↔ shot list (curation) ↔ video, so
   pacing (≈ 3–4 shots, 1.5–2.6 s each) is calibrated on the client's own reels.
2. **Keyframes.** I2V needs a first frame per shot. Three sources: the client's frames;
   composites (plates + character cut-outs, `keyframes.py`); or the previous shot's last frame
   within a set. **New sets and new staging need a keyframe generator** (an image LoRA on the reel
   frames, with text removed). This is the main gap for truly new reels: **workstream v10-K**,
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
| U1 | **Ask the client for clean masters** of the 217 reels (no captions, no watermark, 1080×1920, 24 fps)? | **yes, day 1**; build the loss mask (§3.2 B) in parallel as the fallback | they produced them, iteration_1/2 came as clean `_Base` exports, and it removes the biggest data risk at no cost to us. Also confirm the client is happy for its published reels to be training data |
| U2 | **What "best" means** for the final gate: production preference (G1) first, with v8 label guardrails (G4), or v9's strict per-label R1 | **G1 first, G4 as the guardrail**; R1 reported, not gating Stage 1 | v9's R1 is measured on 1024² single-character studio clips. The final product is 9:16 reels. Phase 0 #1 shows what the strict bar costs |
| U3 | Camera moves: include tagged spans (≤ 25%, explicit camera clause) or static only | **include, tagged**: ✅ **decided yes, 2026-11-04** (108 windows, 14.5% of the 745 trainable; `processed/v10_scenes_748`) | push-ins and pans are part of the reel style; v9b set them aside for "a later camera stage", and this is the last stage. G2's zoom bar guards static prompts |
| U4 | 30 fps reels (65 of 217) | inverse telecine if pulldown; **drop if native 30** | no resampling of hand-drawn timing (v8) |
| U5 | New characters: names and registry phrases (yellow earmuffs, black-and-white, grandpa, bunny hood, …); is the **blue penguin in a red scarf and overalls** Pax or someone else? | ✅ **client answered, 2026-11-05.** Out of training: the red-scarf penguin, yellow earmuffs and the old bearded penguin (no names), and the bunny costume (a one-off). Accepted: the black-and-white penguin. The pearl-necklace penguin is Polly dressed up, so Purse, CancelPlans and PoopBreak train. **Still open:** Fixer's stubble and goggles (3 windows held), and whether other unnamed side penguins stay (25 trainable windows, kept for now) | a Pax lookalike in training teaches "two Paxes", the Polly's-mum case |
| U6 | f121 (5 s) bucket | **only if Phase 0 #3 passes** | A14B has only seen 21 latent frames; 31 is extrapolation |
| U7 | Include `v9b_535`'s skit windows | **yes** | curated and verified already, and the same production domain |
| U8 | Stage 2 (low-noise) | **conditional** on Stage 1 passing and a visible detail gap | the low-noise expert is where v9b-1 regressed |
| U9 | The **yellow-foot variant** in iteration_7: current flat drawing, but yellow feet and beak (hue ≥ ~42°). About 12 skits, ~3,000 frames: Affection, Bromance, LastBite, QUALITYTIME, SkinCare, Upset, SAMEROOM, Blanket, ALREADYSEENIT, Wash, … | ✅ **exclude; client answered, 2026-11-05:** an older design, no longer acceptable (`off_style`, `OLD_DESIGN:` notes) | captions never name foot colour, so including it would teach yellow feet beside the current salmon/amber. Recorded in `finetune/wan/scenes/CURATION_v10.md` |

---

## 8. Risks

- **The high-noise LoRA could collapse prompt control (v5).** G3 and G4 gate it from step 50, and
  the file is bisectable (drop it to get v8 back).
- **Text and watermark leak**, if route A slips and masking is imperfect. The G2 OCR check is a hard bar.
- **Compressed footage teaches artifacts** (banding, blocking on flat fills). Phase 0 #4; masters
  (U1) remove the risk.
- **New characters at low data**: a side character seen in 10 reels won't be drawn reliably from
  text. In I2V the look comes from the keyframe ("one name, one look"), and captions keep names consistent.
- **Small n in G1** if fewer held-out shots survive curation. 80 judgements is the floor. Add seeds
  before relaxing the bar.
- **Keyframes stay the bottleneck** for genuinely new reels (§6.2). v10 improves shots; v10-K
  decides how new a reel can be.
- **Calendar**: prep is ≈ 1 week, mostly curation. The curator pass is the long pole, as in v9b
  (12 curators, 3 verification passes).

## 9. Order

| step | needs | can start |
|---|---|---|
| ingest + profile + mirror (§3.1) | — | **now** |
| client ask: masters, character names, permission (U1, U5) | user sign-off | **now** |
| Phase 0 #1–#3, #5 | a GPU; v9b checkpoints on Azure | **now** |
| curation + captions + prep (§3.3–3.6) | ingest; route A or B decided | ≈ 1 week |
| Phase 0 #4 | the 8 skit-matched reels | with prep |
| **Stage 1** (high-noise) → gates | prep, Phase 0 | after prep |
| Stage 2 (low-noise) → gates | Stage 1 passing + a detail gap | after Stage 1 |
| v10-K keyframe LoRA | text-free reel frames | alongside Stage 1 |
| final: the best of {v10, v8merged} + pipeline, with sample reels | gates | end |
