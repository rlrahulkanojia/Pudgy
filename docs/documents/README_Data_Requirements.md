# Pudgy Penguins — Data Specification (Standalone)

Everything needed to assemble the training dataset for a custom Pudgy Penguins animation model. Self-contained — no other doc required.

**Characters:** Pax (blue ♂) and Polly (pink ♀). **Output target:** brand-faithful 3–5s looping GIFs (then 30s shorts), 9:16 portrait.

**Core principle:** **clean base renders, not social exports.** Posted/GIPHY versions are compressed and carry burned-in text/watermark/timecode that the model would learn as part of the picture. We need the source masters. (For reference, the existing social reels are 720×1280 H.264 @ ~0.5 Mbps — below our minimum and artifact-laden.)

---

## Part 1 — What we request from the client

### 1.1 Finished skits (the core training material)
- **15–25 skits** chosen for *variety of action*, which we cut into **60–80 short training clips**.
- Format: **ProRes 422 HQ `.mov` or lossless PNG image sequence** (avoid re-exported H.264).
- **Base render only:** no watermark, no on-screen text, no timecode/panel UI.
- **24 fps** constant; native resolution, **short edge ≥ 1080 px**; RGB/sRGB.

### 1.2 Character-isolated passes + clean background plates *(highest-value)*
Per skit: the **character(s) on a transparent background** + the **clean background plate** (no characters), as separate files.
- Format: **ProRes 4444 (alpha) or PNG-with-alpha**, same fps/res as 1.1.
- Why: lets us composite the neutral-gray and show-background versions ourselves with clean edges.
- Fallback if alpha isn't practical: render each clip **twice** — once over flat **#808080** gray, once over the show background.

### 1.3 Easier-to-share project equivalent (any one of)
- **A. Layered exports** *(preferred)* — character / background / prop passes as ProRes 4444 / PNG-with-alpha.
- **B. Collected project** — AE `File → Dependencies → Collect Files`, zipped.
- **C. Layered stills** — PSD/AI of key frames.
(Full `.aep` + linked assets also fine, but heavy and version-dependent.)

### 1.4 Artifacts library (props, backgrounds, FX)
Recurring non-character assets — igloo, couch, glass table, penguin chess set, iFin phone, bus, Polly's backpack, orca, storefronts.
- **Each prop** as an **isolated transparent PNG** (multi-angle if seen from several sides).
- **Each environment** as a **clean full-frame plate**.
- Keep the client's naming (`Prp_ChessBoard`, `Prp_iFin`, `Prp_Orca`…). Individual layers/exports, **not** the flattened art sheet.

### 1.5 Pax & Polly model sheets
- **Turnarounds:** neutral/T-pose from ~8 angles (front, 3/4-front L/R, profile L/R, 3/4-back L/R, back).
- **Expressions:** a spread each (neutral, happy, eating, surprised, anxious, affectionate, exasperated…).
- Format: **PNG, transparent**, short edge ≥ 1080, one pose per file. **50–100 images total.**

### 1.6 Style guide + access
- Brand bible / "Master Control" rules: PDF + reference images, **exact color hex + accessory names**.
- **Figma** export of model-sheet pages + **Notion** storyboard access (both currently login-gated for us — an export is fine).

### 1.7 Shot list / tags
Per clip: source file, in–out timecode, character(s), action label, background. CSV/sheet; existing tags are fine.

### 1.8 Purpose-shot reference clips (shot *for the model*) *(high value)*
Short, clean references — **one action, static camera, neutral-gray background, 3–5s.**
- Single-action clips per motion (walk, idle/breathe, blink, wave, head-turn, jump, sit, eat) — ~5 each.
- Turnaround **videos** (slow 360°) of Pax, Polly, and key props.
- Two-character primitives (facing, lean-in, head-on-shoulder, flipper-hold).
- Loop-ready cycles (idle/walk/bounce, first pose = last pose).
- Expression / eye-direction transitions.

### 1.9 Two answers (confirmed by client)
- **Frame rate:** animated on **true 24 fps** (not 12 doubled). → our 24 → 16 fps resample is clean; no de-duplication needed.
- **Prop animation:** **any prop can animate depending on the skit/prompt** — props are *not* inherently static. → we **cannot** blanket-caption "all props stay still"; the static lock applies per-clip only when a prop is actually static, and clips where a prop animates must **describe that prop's motion**. Implication: include both static-prop and animating-prop clips so the model learns each prop can hold *or* move on cue.

**Not needed yet:** audio (first deliverables are silent GIFs).

---

## Part 2 — Training-ready clip spec (what we process everything into)

Hard requirements for `CogVideoX1.5-5B-I2V`:

| Property | Value |
|---|---|
| **Resolution** | **768 × 1360** portrait (primary); landscape 1360 × 768 only if a horizontal GIF needs it |
| Constraint | min(W,H)=768; 768 ≤ max(W,H) ≤ 1360; max(W,H) % 16 = 0 |
| **Frame rate** | **16 fps** (resampled from 24) |
| **Frame count** | exactly **49** (~3.06 s) or **81** (~5.06 s) — must satisfy **8N+1** |
| **Background** | **70 %** neutral gray #808080 / **30 %** show background |
| **Color** | RGB, sRGB, background baked in (no alpha in the final clip) |
| Container | per Passenger12138 trainer (confirm: MP4 CRF ≤ 16 or PNG-folder), each paired with a `.txt` caption |

### Processing pipeline (masters → clips)
1. Transcode masters → lossless/ProRes intermediate; assert short edge ≥ 768.
2. **Segment** into atomic micro-actions (one action per clip); 2–5 s windows.
3. **Resample** 24 → 16 fps.
4. **Trim** to exactly 49 or 81 frames (tail-pad with cloned frames if a window is 1–2 frames short).
5. **Resize** to 768×1360 (9:16 → bucket differ <1 %; minimal crop/pad, never stretch).
6. **Composite** 70 % over #808080, 30 % over the show plate; matte/defringe edges.
7. **Export** + write the paired caption; append a row to `manifest.csv` (`clip_id, character(s), action_label, bg_type, props, frames, w, h, fps, source_skit`).

Tooling: `code/prep/build_dataset.py` (build + `verify`), `code/prep/caption_prune.py`.

### Composition gates (must pass before training)
- **Bucket split:** ~60 % single-character (36–48) · 25 % Tier-1 two-char, >30 % gap (15–20) · 15 % Tier-2 proximate (9–12).
- **Action parity:** ≥ 4–5 clips per action type; rebalance starved actions.
- **Background mix:** 70/30 gray/show across the whole set.
- **Artifacts:** recurring props appear in ≥ 2 clips.
- **Reference images:** 50–100 char + prop stills, captioned appearance-only, kept as a **minority** of the set (too many stills → static, low-motion output).

---

## Part 3 — Captioning data (what each sample needs)
CogVideoX uses a frozen T5-XXL text encoder → dense natural-language sentences, not tags.
- **Identity anchor** (15–20 words, hand-authored per character from the style guide), prepended to every caption — e.g. *"A stylized 2D cartoon animation of Pax, a pudgy blue penguin with thick black outlines, a white belly, and an orange beak,"*
- **Motion suffix** (VLM-generated): dynamics only — never appearance. Show-bg clips lock the **background** as static; props are locked only when actually still (*"…the background remains completely stationary"*), and **animating props get their motion described** instead.
- **Multi-character:** spatial anchors read **per clip from the actual frame** (left/right is scene-dependent — don't hardcode).
- **Pruning:** strip appearance terms from the suffix (blocklist + semantic similarity) so identity and motion don't entangle; spot-check 10 %.
- **Artifact stills:** captioned appearance-only (anchor-style), so the LoRA learns each prop's one canonical look.

---

## Part 4 — Status (2026-07-01)
- ✅ **Delivered** (`Data/client/`): one full "Survival" skit as a complete flow (concept PDF → animatic → art-asset sheet → final), 6 design-guideline slides, links doc. Source videos are 1080×1920 / 24 fps / ~16 Mbps but with burned-in text/watermark/UI.
- ⏳ **Pending:** clean base renders, isolated/alpha passes, artifacts library, Pax/Polly model sheets, Figma/Notion access, the broader 15–25 skits, and purpose-shot reference clips.
