# Pudgy Penguins Data We Need for Custom Model Training

Hi team to train a custom model that generates brand-faithful Pudgy Penguins animation (Pax and Polly + your props/worlds), here's exactly what we need from you and the format that works best. Getting the format right at the source saves a re-do cycle and directly drives the quality of the result.

## Three things that matter most
1. **Send clean base renders** your *no-watermark, no-text, no-timecode* exports not the posted/social versions.
2. Where you can, give us the **characters on a transparent background** and the **backgrounds/props separately**, so we control how they're combined.
3. Keep everything at **native 24 fps** and your **highest resolution** (short edge ≥ 1080).

> **Why the clean base renders?** The posted/GIPHY versions are compressed and have on-screen text + the @pudgypenguins watermark baked in the model would learn those as part of the picture. Your "base video, without watermark" export is exactly what we need.

---

## What to send

### 1. Finished skits *(the core material)*
- **15–25 skits**, chosen for *variety of action* (not just favorites). We'll cut these into the short clips the model trains on.
- **ProRes 422 HQ `.mov`** or a **lossless PNG image sequence** (please avoid re-exported H.264/MP4).
- **Base render only** no watermark, no on-screen text, no timecode/panel UI.
- **24 fps**, native resolution (short edge ≥ 1080), portrait 9:16.

### 2. Characters and backgrounds separated *(highest value)*
For each skit: the **character(s) on a transparent background**, plus the **clean background** (no characters) as separate files.
- **ProRes 4444** (with alpha) or **PNG sequence with alpha**.
- If alpha isn't practical, an easy alternative: render each clip **twice** once over flat 50% gray, once over the real background.

### 3. Easiest-to-share project equivalent *(any one whatever's least friction)*
- **A. Layered exports** *(preferred)* character / background / prop passes as ProRes 4444 or PNG-with-alpha.
- **B. Collected project** After Effects `File → Dependencies → Collect Files`, zipped.
- **C. Layered stills** PSD/AI of a few key frames per scene.
*(Full `.aep` + linked assets is fine too, just heavier to move.)*

### 4. Artifacts library (props, backgrounds, worlds)
Your recurring assets igloo, couch, glass table, penguin chess set, iFin phone, bus, Polly's backpack, orca, storefronts, etc.
- **Each prop** as an **individual transparent PNG** (multi-angle if it's seen from several sides).
- **Each environment** as a **clean full-frame background**.
- Keep your existing names (`Prp_ChessBoard`, `Prp_iFin`, `Prp_Orca`…). We need the **individual layers/exports**, not the flattened art sheet.

### 5. Pax & Polly model sheets
- **Turnarounds:** neutral/T-pose from multiple angles (front, 3/4-front L/R, profile L/R, 3/4-back L/R, back).
- **Expression sheets:** a spread each (neutral, happy, eating, surprised, anxious, affectionate, exasperated…).
- **PNG, transparent background**, one pose per file, with your exact color codes. **50–100 images total.**

### 6. Style guide + access
- Your brand bible / "Master Control" rules (PDF + reference images), with **exact color hex + accessory names**.
- **Figma** access or an export of the model-sheet pages, and **Notion** storyboard access (both are login-gated for us right now an export works just as well).

### 7. Shot list / tags
For the skits you send, a simple sheet: source file, in–out timecode, who's in it, the action, the background. Your existing tags are perfect.

### 8. New clips shot *for the model* *(high value)*
Short, clean references **one action, static camera, neutral-gray background, 3–5 seconds.** Shot for training, not for social.
- **Single-action clips** per motion: walk, idle/breathe, blink, wave, head-turn, jump, sit, eat (≈5 each).
- **Turnaround videos** a slow 360° rotation of Pax, Polly, and key props.
- **Two-character clips** facing each other, lean-in, head-on-shoulder, flipper-hold.
- **Loop-ready cycles** idle / walk / bounce that start and end on the same pose.
- **Expression & eye-direction transitions** neutral→happy, neutral→surprised, look left↔right, blinks.

**Not needed yet:** audio the first outputs are silent looping GIFs. We'll ask for audio when we move to longer narrative shorts.

---

## Delivery format quick reference

| Asset | Format | Notes |
|---|---|---|
| Skits / clips | ProRes 422 HQ or PNG sequence | 24 fps, ≥1080 short edge, **no watermark/text/UI** |
| Character + background passes | ProRes 4444 or PNG-with-alpha | transparent character + clean background plate |
| Props & environments | Transparent PNG (props) / full-frame (backgrounds) | keep `Prp_*` names |
| Model sheets | Transparent PNG, one pose per file | exact color codes |
| Style guide | PDF + images / Figma + Notion | exact hex + accessory names |
| Shot list | CSV / Google Sheet | timecodes + tags |

---

## Checklist

- [ ] 15–25 **clean base renders** (no watermark/text/UI), ProRes/PNG, 24 fps, ≥1080
- [ ] **Character-on-transparent + clean background** passes (or the gray + real-background double-render)
- [ ] **Artifacts library** props (transparent PNG) + backgrounds (full-frame), keeping `Prp_*` names
- [ ] **Pax & Polly model sheets** turnarounds + expressions, transparent PNG, with color codes
- [ ] **Style guide** + **Figma/Notion** access or exports
- [ ] **Shot list / tag sheet** for the skits sent
- [ ] **Purpose-shot reference clips** (single-action, turnarounds, two-character, loops, expressions)
- [ ] Skits chosen to spread across the **full range of actions**

---

## Thanks and a couple of things you've already confirmed
- ✅ Animation is at **true 24 fps** perfect, that's what we need.
- ✅ **Any prop can animate** depending on the skit good to know; we'll handle static and animating props accordingly.

**Already received:** the full "Survival" sample flow (concept → animatic → art assets → final) and your design-guideline slides thank you, very useful.
**Still needed:** the items in the checklist above.

**Delivery:** any shared drive/link works (Google Drive, Dropbox, WeTransfer) a folder per item is ideal. **If anything here is hard to produce, just tell us** we have fallbacks for almost everything.
