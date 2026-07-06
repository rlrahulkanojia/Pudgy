# Data Discrepancies & Notes — 30_videos / 30_prompts

Found while building the training set. None are corruption/quality-control failures — the footage matches the storyboards. The issues are **structural** (things that make clips unusable for training) and a few **naming/versioning** mismatches worth confirming with the client.

## 1. Off-character content (footage without Pax/Polly)
- **Universe** — the clip is seahorses, fish, coral and a sunset, with **no penguins**. This matches its own prompt (beats 3–6 are deliberately metaphorical: "Polly as a fish," "pink coral and kelp," "flower and butterfly," "sunset"). Correct per storyboard, but useless for a penguin character LoRA → **excluded**.

## 2. Product / text end-cards baked into skits
Intentional per the prompts, but they're graphics, not character animation, so those shots were culled:
- **MothersDay26** → "HAPPY MOTHER'S DAY / LOVE YOU MOM" CTA card.
- **IB_Picnic** → "Icebreakers" game box + "150 QUESTIONS THAT…" card insert.
- **Customize** → phone product showcase (Pudgy phone charms).
- **Manifested** → "The night before we met" black title card.

## 3. Prompt ↔ video naming / versioning mismatches (please confirm)
- `2D_PP_Unhelpful_Base.mp4` maps to **`PP_Unhelpful_v2.pdf`, whose internal title is "FOLLOW_v2"** and whose inspo is a Reddit link — while there is *also* a separate `2D_PP_Follow_Base.mp4` ↔ `PP_Follow.pdf`. Worth confirming "Follow" and "Unhelpful/Follow_v2" aren't two versions of the same concept or mislabeled.
- `2D_PP_Fixer2_Base.mp4` ↔ `PP_Fixer.pdf` (Fixer vs Fixer2 — likely a v2, confirm).
- `PP_LIfeWithHer.pdf` — capitalization typo in filename.
- The Unhelpful/Follow_v2 prompt has a **numbering error** (two beats labelled "2").

## 4. Wipe / match-cut transitions inside clips
Several prompts explicitly call for transitions that hide a scene change: **TheresYou** ("WIPE TRANSITION"), **IB_Picnic** ("WIPE TRANSITION," "MATCH CUT"), **Unhelpful** ("CUT TO"). Automated scene detection can't catch these (no hard frame discontinuity), so windows spanning them had to be culled manually. Not an error — just why several otherwise-good skits yielded fewer clips.

## 5. Costume / model variance (may affect character consistency)
The model will see Pax and Polly in varied outfits/accessories: Pax in scarf + tie (TirePressure), headband (Workouts); Polly in an eye mask (MorningPerson), and notably **Polly in a monk's robe** (Manifested beat 3). Accessories are fine, but the monk-robe look is off-model and could dilute a consistent-identity LoRA if included.

## 6. Skits that contributed zero training clips
All their shots were shorter than 3.06 s or straddled transitions: **Disaster** (6 rapid cuts, snowstorm), **Overstimulated** (short, ~4 s), **Lamps** (fast cuts), **MorningPerson** partial, **MovingIn** (characters too small — snowglobe diorama establishing shot). Universe excluded per §1.

## Bottom line
No scene contradicted its prompt. The usable-clip count (22 of a possible ~30 skits) is driven by: fast editing (short shots), embedded end-cards, wipe/match-cut transitions, and a few off-character/tiny-character shots — all inherent to short-form social skits, not data defects. The practical fix is producing purpose-shot single-action reference clips (per the gap analysis), not re-cutting these.
