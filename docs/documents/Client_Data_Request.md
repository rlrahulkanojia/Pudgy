# Pudgy Penguins — Data Request for Model Training

**Purpose:** This is the exact list of assets we need from you to fine-tune a custom Pudgy Penguins animation model (`CogVideoX1.5-5B-I2V`). Getting the **format right at the source** saves us a re-request cycle and directly determines output quality, so please read the format notes per asset.

**One-line summary of what matters most:**
1. Send us your **clean base renders** — the **no-watermark, no-text** export you already produce — not the social/posted versions.
2. Where possible, render the **characters on a transparent background** (with alpha) and the **backgrounds separately**.
3. Keep everything at **native 24 fps** and the **highest resolution** you have.

> **Why the clean base renders specifically?** The Survival sample flow you sent is great quality (1080×1920 / 24 fps), but the **Finals have burned-in on-screen text + the @pudgypenguins watermark**, and the **animatics have a burned-in timecode + "Scene/Panel" UI and frame borders**. The model would learn those overlays as part of the picture. Your workflow already exports a **"base video, without watermark"** — that clean variant is exactly what we need (plus the clean line-art animatics without the timecode UI).

---

## What we need (7 asset classes)

### 1. Master-quality animation renders — *the core training material*
The highest-quality export of each selected skit.

| Property | Requested format |
|---|---|
| Codec / file type | **Apple ProRes 422 HQ** `.mov`, **or** a **lossless PNG image sequence**. Please avoid re-exported H.264/MP4. |
| Variant | The **base render: no watermark, no on-screen text, no timecode/panel UI** |
| Frame rate | **24 fps**, constant (not variable) |
| Resolution | Native, **short edge ≥ 1080 px**. Portrait 9:16 preferred (your Reels/GIF format) |
| Color | RGB, sRGB |
| How many | **15–25 skits** chosen for *variety of actions* (see "Action coverage" below). We'll cut these into 60–80 short training clips. |

> **Also valuable — clean animatics:** if you can export the **line-art animatics without the timecode/panel overlay**, send those too. The animatic→final pairing (rough pose layout → finished colored frame) is exactly the structure the engine learns to fill in, and it directly supports the "rough the key poses, let the engine finish them" workflow Oliver described.

### 2. Character-isolated renders **with transparency (alpha)** — *highest value if you can do it*
For each selected skit, the **character(s) rendered alone over a transparent background**, plus the **background rendered alone** (no characters), as separate files.

| Property | Requested format |
|---|---|
| File type | **ProRes 4444** `.mov` (with alpha) **or** **PNG sequence with alpha** |
| Frame rate / resolution | Same as #1 (24 fps, native res) |
| Layers | (a) character pass over transparent BG, (b) clean background plate |

> **Why:** with the character on transparent BG, *we* can place it on the neutral gray and on your real backgrounds ourselves, with perfectly clean edges. This is the single biggest quality lever.
>
> **If alpha renders aren't practical:** instead render each selected clip **twice** — once over a **flat 50% gray (#808080)** background, once over the **real show background**.

### 3. After Effects project files (or an easier-to-share equivalent)
The `.aep` files **+ their linked assets** (footage, comps, fonts) for **10–15 representative skits**.

> **Why:** this lets us re-render at the exact resolution and frame count we need, and isolate layers ourselves — so we don't have to come back to you for re-exports.

**A full `.aep` is often awkward to share** — the files are large, they break if any linked footage is missing, and they're tied to a specific AE version + plugins. So any **one** of these works, easiest first:

| Option | What to send | Notes |
|---|---|---|
| **A. Layered exports** *(preferred)* | Per scene, the **separate passes** — character pass, clean background plate, each prop — as **ProRes 4444 / PNG-with-alpha** | Most portable, no AE needed on our end, and it's exactly what we'd pull out of the project anyway (overlaps with #2 and #7) |
| **B. Collected project** | AE **`File → Dependencies → Collect Files`** output, zipped | Bundles the project + all footage into one folder so nothing is missing |
| **C. Layered stills** | Layered **PSD/AI** of a few key frames per scene | Good if full motion projects are too heavy to move |

> Send whichever is least friction for your team — we don't need all three.

### 4. Character turnarounds & expression sheets — **Pax (blue ♂) + Polly (pink ♀)**
Reference stills so the model learns each character's 3D volume and face. The character-sheet idea you already use (a grid of poses + color codes) is exactly right — we want the **canonical Pax/Polly model sheets**, ideally the ones in your Figma design guide (see #5).

| What | Detail |
|---|---|
| **Turnaround** (per character) | Neutral standing/T-pose from multiple angles: front, 3/4-front (L & R), profile (L & R), 3/4-back (L & R), back — aim for **8 angles** (your "180/360 view" T-pose) |
| **Expression sheets** (per character) | A spread of canonical expressions each — e.g. neutral, happy, eating, surprised, embarrassed/anxious, affectionate, exasperated/-_- |
| **Format** | **PNG, transparent background**, short edge ≥ 1080, **one pose per file** (the exact color codes / hex from your guide) |
| **How many** | **50–100 images total** across both characters |

### 5. Style guide / "Master Control" rules + Figma / Notion access
Your brand bible — color specs (exact hex), proportion rules, accessory definitions, do's/don'ts. We have your 6 design-guideline slides already; we still need the underlying source.

| Property | Requested format |
|---|---|
| Format | PDF + reference images |
| **Figma** | View/export access to the **"Pudgy Penguins — PudgyWorld Everyday Design"** file (the canonical Pax/Polly model sheets live here — currently login-gated for us) |
| **Notion** | Access to the **Storyboard Guide** (currently login-gated for us) |

> **Why:** we use this to write the model's "identity description" of Pax and Polly and to make sure captions don't fight the brand spec. A PNG/PDF **export** of the Figma model-sheet pages works just as well as direct access.

### 6. Shot list / tags (you mentioned assets are already tagged)
For the skits you send, a simple sheet listing each distinct action.

| Column | Example |
|---|---|
| Skit / file | `PP_Survival_base.mov` |
| In–out timecode | `00:06 – 00:09` |
| Character(s) | Pax + Polly |
| Action | "playing chess at table" |
| Background | igloo interior |

> Format: CSV / Google Sheet. If you don't have this, we can build it from the footage — but your existing tags will speed things up a lot.

### 7. Artifacts — props, backgrounds & FX (your non-character asset library)
Every skit is Pax + Polly **plus artifacts** — the props, environments and effects created "in the spirit of Pudgy" (in the Survival sheet: the igloo, couch, glass table, the penguin chess set, the iFin phone, the bus, Polly's backpack, the orca, the storefronts). These **recur across skits**, and the engine has to render them **consistently** and **keep them stable** — a prop shouldn't warp, drift, or re-design itself while the characters move. So we need them as first-class assets, not only baked into video.

| What | Detail |
|---|---|
| **Prop / FX assets** | Each recurring prop as an **isolated PNG with transparency** — multi-angle if it's seen from several sides (igloo, couch, table, vehicles, phone, food, orca…) |
| **Background plates** | Each environment as a **clean full-frame plate**, no characters/props composited on top |
| **Naming** | Keep your existing convention (`Prp_ChessBoard`, `Prp_iFin`, `Prp_PollyBackpack`, `Prp_Orca`, …) so we can map them |
| **Format** | PNG (transparent for props; full-frame for plates), short edge ≥ 1080. The labeled sheet you sent (`PP_Survival_ArtAssets.png`) is the right idea — we just need the **individual layers/exports**, not the flattened sheet |

> **How these get used in training (so the request makes sense):**
> - Background/prop plates feed the **30% "show-background"** clips (the other 70% are neutral gray) — this teaches the engine that backgrounds and props **exist and stay still**, instead of hallucinating or melting them.
> - Key props are **named and spatially placed** in the training captions; props that are still in a clip are locked static, while props that animate (per the skit) get their motion described — so the model learns each prop can hold *or* move on cue.
> - The isolated prop images join the same reference pool as the character turnarounds, so the engine learns each prop's **one canonical look** (the same igloo every time, not a new one per generation).
> - We track which props appear where, so recurring ones (igloo, couch, table, phone) show up across **several** clips rather than once.

**Not needed yet:** audio (the first deliverables are silent looping GIFs). We'll request audio for the Phase 1 30-second narrative work.

---

## Action coverage (please spread the 15–25 skits across these)
We need **at least 4–5 examples of each action type** so the model doesn't over-learn one motion. Aim to cover:

- **Locomotion:** walk, run, bounce/jump
- **Idle:** standing idle/breathing, sitting, lying down, phone-scrolling
- **Gesture:** wave, head-turn, look-around
- **Eating:** holding food, biting, fork-eating, hand-eating, slurping/drinking
- **Affection (two-character):** eye contact, lean-in, head-on-shoulder, holding flippers, nuzzle
- **Domestic tasks:** pillow-fluffing, towel-folding, tidying
- **Expression shifts:** brows lifting, blush appearing, posture opening

Two-character clips are valuable too — both **side-by-side** (with a clear gap between them) and **close/interacting**.

---

## One-page checklist (what "done" looks like)

- [ ] **15–25 clean base renders** (no watermark/text/UI) — ProRes 422 HQ or lossless PNG sequence, 24 fps, short edge ≥ 1080
- [ ] **Clean line-art animatics** (no timecode/panel overlay), if exportable
- [ ] **Character-on-transparent + clean background plates** (ProRes 4444 / PNG-with-alpha) — *or* the gray + show-bg double-render fallback
- [ ] **Artifacts library** — recurring props (transparent PNG) + clean background plates, keeping your `Prp_*` naming
- [ ] **10–15 skits' layered exports / Collected project / `.aep`** (whichever is easiest — see #3)
- [ ] **50–100 turnaround / expression PNGs** (transparent, one pose per file) — **Pax + Polly**
- [ ] **Style guide / Master Control rules** (PDF + images) + **Figma export & Notion access**, with exact color hex + accessory names
- [ ] **Shot list / tag sheet** (CSV) for the skits sent
- [ ] Skits chosen to cover the **action list above (≥ 4–5 per action)**

**Delivery:** any shared drive / link works (Google Drive, Dropbox, WeTransfer). Folder per asset class is ideal. If anything here is hard to produce, tell us — we have fallbacks (e.g., we can segment and re-render from the `.aep` files ourselves).
