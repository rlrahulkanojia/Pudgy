# v9 training run: in plain words

## What we're training

v9 adds to v8 in stages, one new thing at a time, and every stage builds on the one before.
The first stage, **v9c-phone**, teaches Pax and Polly to hold and use **the iFin phone**, the
client's phone prop, while keeping everything v8 already did (5 movements, 8 expressions).

It **starts from v8** (v7 + v8 combined exactly into one file) and fine-tunes a small add-on
on top. It learns from **475 short clips**:

- **380 phone clips:** texting while sitting and standing, on 4 plain backgrounds, at
  3 lengths;
- **95 v8 clips** mixed back in (20%), so v8's movements and expressions aren't forgotten.

One camera angle (`QF2_R`) is held back for testing.

Next stages: **v9b** (real set backgrounds), then **v9a** (fast motion, once the client footage
arrives). See [`Training_Approach_v9.md`](Training_Approach_v9.md) §0 and
[`Assets_v9.md`](Assets_v9.md) §8.

## How to write a prompt with the phone

Use the same six-part pattern as v8 (see the [v8 README](../v8/README.md)), with the phone
named in the action **exactly** like this:

> **the iFin phone, a small pale grey phone with a black screen, three camera lenses on the
> back and a little tuft on top**

**Example** (the training caption for Pax texting while sitting):

> A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat
> pastel colors, showing **Pax, a short round blue penguin**, sitting on the ground with its legs
> stretched out in front, holding **the iFin phone, a small pale grey phone with a black screen,
> three camera lenses on the back and a little tuft on top,** up in front of its chest in one
> flipper and tapping at it with the other, the screen turned toward itself, head still and eyes
> on the phone; **texting on a phone, seated**; static close-up shot, eye level, facing the
> camera directly, front view; **plain white studio background**.

**Put the phone in the first frame.** The model makes videos *from a starting picture*, and the
phone's look comes from that picture, not from its name. This is the "one name, one look" rule
([`Assets_v9.md`](Assets_v9.md) §2): the same words in the prompt, and the phone's own pixels in
the first frame (and in the last frame too, when it has to stay put).

## Backgrounds

The background comes from the **starting picture**, and the prompt's last part should describe
it. Training used 4 plain colours: **white, pastel blue, peach, mint** ("plain pastel blue
background", and so on). Any of those is safe. Other colours or a real scene image can be put
behind the character, but the model never trained on them, so check the results. Real set
backgrounds are the next stage (v9b).

## Good to know

- **Saying "the iFin phone" without a phone in the picture doesn't reliably make one appear.**
  In tests the name did not beat generic words like "a phone" or "a TV remote".
- **The phone can turn round mid-clip.** When the phone faces the camera straight on, it often
  turns to show its screen, sometimes with a made-up call screen or lettering. v8 does the same; this
  stage didn't fix it. Fixes being tested: a negative prompt for screens and text, lower guidance,
  a phrase without "black screen", and first+last-frame mode.
- **Only two held poses exist:** texting while sitting or standing. Those are the only ones the
  client animated.
- **Movements may be slightly weaker than v8** (recognised 58% vs 75% on the test angle).
  Expressions are as good or better.

## How v9c-phone did

- **Passed its checks.** All four checkpoints tested (steps 125–200) cleared the bars that were set
  in advance. The winner is **step 200**.
- **It learned the phone footage.** On the test angle, how well it predicts phone clips improved by
  39%.
- **But the phone glitch is still there.** Side by side with v8, on the same start picture and
  seed, v9c redraws the phone area slightly, but the phone turns round in the same 5 of 8 clips.

## Where things are

- **Run results:** [`REPORT_v9c_phone.md`](../../docs/training_reports/v9/REPORT_v9c_phone.md)
  (winner: step 200)
- Dataset, data issues and the numbers behind the choices: [`NOTES.md`](NOTES.md)
- Why the phone didn't improve, and the plan for the next run: [`RESEARCH_v9c_fixes.md`](RESEARCH_v9c_fixes.md); storing assets as embeddings or references: [`RESEARCH_asset_memory.md`](RESEARCH_asset_memory.md)
- The plan: [`Training_Approach_v9.md`](Training_Approach_v9.md) · props and backgrounds:
  [`Assets_v9.md`](Assets_v9.md) · how to run it: [`RUNBOOK_v9c_phone.md`](RUNBOOK_v9c_phone.md)
- Weights: Azure `pudgy/v9c/weights/pudgy-v9c-phone-lownoise/` (load on top of
  `pudgy-v8-merged.safetensors`)
