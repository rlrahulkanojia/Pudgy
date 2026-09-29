# v8 training run: in plain words

## What we're training

One add-on to the video model (a "LoRA") that teaches it Pax and Polly doing **5 movements**
and **8 expressions**. You pick one by describing it in the prompt:

- **Movements:** walking, running, waving, sitting, jumping
- **Expressions:** happy, surprised, angry, confused, crying, neutral, laughing, blushing (new)

It **starts from what v7 already learned** (v7's expression and motion add-ons, combined
correctly into one) and fine-tunes on top, adding the new expression and smooth loops.

It learns from **2,096 short clips** (up to about 2.4 seconds each), made from the client's
footage. One camera angle is kept out of this round's training so we can check the new
training carries over to clips it wasn't fitted on. (v7 did see that angle, so only the new
*blushing* clips are completely unseen.)

## How to write a prompt

Every training clip was described with the **same sentence pattern**. Prompts work best
when they follow it exactly. Six parts, always in this order:

| # | Part | What to write | Options |
|---|---|---|---|
| 1 | **Style** | Always the same opening | "A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing" |
| 2 | **Character** | Who, *always with their colour* | "Pax, a short round **blue** penguin" · "Polly, a short round **pink** penguin" |
| 3 | **Action** | What they do, described in detail | e.g. "walking in place with a steady waddling gait…" (one fixed description per movement or expression) |
| 4 | **Label** | The short name | e.g. "walk cycle", "angry expression" |
| 5 | **Camera** | How close, and from which side | close-up / medium / wide shot · front view, three-quarter view, or side profile |
| 6 | **Background** | A plain colour | white, pastel blue, peach, or mint |

**Example:**

> A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and
> flat pastel colors, showing **Pax, a short round blue penguin**, **walking in place with a
> steady waddling gait, feet alternating, flippers swinging gently at its sides, body
> bobbing softly**; **walk cycle**; **static close-up shot**, eye level, **facing the camera
> directly, front view**; **plain white studio background**.

Change one part and only that thing changes. Swap "walk cycle" for "waving gesture" (and
its action text) and you get waving, with the same character, camera and background.

## Good to know

- **Always say the colour.** Without "blue" or "pink" the model can mix the two characters up.
- **Keep "pink" out of Pax prompts.** Pax's blush is described as "rosy" for this reason.
- **Loops:** walking, waving, jumping and sitting can repeat smoothly for about 2 seconds. Add
  "repeating in a continuous loop" after the label. Running doesn't loop.
- **One thing at a time works best.** The model learned movements and expressions separately.
  Combining them ("walking while angry") hasn't been taught, and we test it as an experiment.
- **Weaker areas:** *confused* and *happy* have less footage (a quarter of the others), so
  expect them to be less reliable.
- **Not included yet:** two-character scenes (hugging, waving together), sad, scared,
  and a relaxed "standing idle".

## Where things are

- **Run results:** [`REPORT_v8.md`](../../docs/training_reports/v8/REPORT_v8.md) (winner: step 550)
- Dataset summary, data issues, and the maths behind v8: [`NOTES.md`](NOTES.md)
- Full technical plan: [`Training_Approach_v8.md`](Training_Approach_v8.md)
- Training clips: Azure `pudgy/processed/v8_joint_2096/`
- Weights, logs and results: Azure `pudgy/v8/` (the v8 base is `pudgy/v8/base/`)
