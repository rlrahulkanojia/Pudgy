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

**Update (2026-10-30), v9c2:** the phone problem is fixed, and the setup changed. The phone is now
carried by a **new wording** plus a small **learned phone embedding (E5)**, with no add-on that touches
v8. See "How the phone works now" below.

Next stages: **v9b** (real set backgrounds), then **v9a** (fast motion, once the client footage
arrives). See [`Training_Approach_v9.md`](Training_Approach_v9.md) §0 and
[`Assets_v9.md`](Assets_v9.md) §8.

## How to write a prompt with the phone

Use the same six-part pattern as v8 (see the [v8 README](../v8/README.md)), with the phone
named in the action **exactly** like this:

> **the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top**

(Until 2026-10-27 the phrase also said "with a black screen". Those words made the model turn the phone
round to show a screen, so they were removed. The screen's state, e.g. lit with an alarm, now goes in the
shot's own action words.)

**Example** (the training caption for Pax texting while sitting):

> A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat
> pastel colors, showing **Pax, a short round blue penguin**, sitting on the ground with its legs
> stretched out in front, holding **the iFin phone, a small pale grey phone with three camera lenses
> on the back and a little tuft on top,** up in front of its chest in one flipper and tapping at it
> with the other, **holding it with its back toward the viewer**, head still and eyes on the phone;
> **texting on a phone, seated**; static close-up shot, eye level, facing the camera directly, front
> view; **plain white studio background**.

Never type the wording by hand: captions are compiled from the asset registry
(`finetune/wan/assets/registry.toml`), so training and inference always use the same words.

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

- **Saying "the iFin phone" without a phone in the picture still won't make one appear.** With the
  embedding the name is now tied to the phone, but producing a phone from nothing needs footage where
  one is picked up or taken out (client ask A1).
- **The phone turning round is fixed.** With the old wording it showed its screen in 12 of 24 test
  clips. With the new wording it's 1 of 24, and with the embedding 0 of 24.
- **Only two held poses exist:** texting while sitting or standing. Those are the only ones the
  client animated.
- **v8's movements and expressions are untouched** with the adopted setup: the embedding only acts on
  prompts that name the phone. The first phone add-on (v9c) did weaken movements (58% vs 75%); it is
  no longer used.

## How the phone works now (v9c2, adopted 2026-10-30)

| Part | What |
|---|---|
| Model | v8 (`pudgy-v8-merged`, low-noise) + v2 golden (high-noise), unchanged |
| Words | the registry phrase above, compiled into the caption |
| **Phone embedding (E5)** | `pudgy-v9c2-asset-dual-step00000150.asset.safetensors`: 25 × 4096 numbers that replace the phrase's text vectors, trained through both halves of the model on the phone clips |
| Generate with | `finetune/wan/eval_v9/gen_asset_v9.py --asset <E5> --asset-offset -- <usual musubi arguments>` |

What it does, on 24 test clips (8 start pictures × 3 seeds):
- **The phone keeps its back to the viewer in all 24** (base with the new words: 23; old words: 12).
- **Texting is steadier:** the flipper keeps tapping while the body and head hold still, as in the
  client's animation. Without it, the base often froze or drifted.
- **The name now means this phone:** on held-out clips "the iFin phone" fits the video better than
  "a phone", "a book", "a cup" or "a remote" in 3 of 4 (v9c: 0 of 4).
- A further add-on on top (the "pivotal" LoRA) passed every check but made no visible difference, so
  the simpler embedding was adopted. The add-on is kept on Azure.

Client sample clips: `/workspace/deliver/client_phone_samples_e5.zip`, also on Azure `v9c2/deliver/`.

## How v9c-phone did (first phone stage, superseded)

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
- Weights: phone embedding (adopted) on Azure `pudgy/v9c2/weights/pudgy-v9c2-asset-dual/`; v9c LoRA
  (superseded) `pudgy/v9c/weights/pudgy-v9c-phone-lownoise/`; pivotal LoRA `pudgy/v9c2/weights/pudgy-v9c2-pivotal/`
- v9c2 jobs and their pre-set decision rules: `finetune/wan/jobs_v9c2/`; GPU queue service:
  `finetune/wan/services/` (`gpu-queue`)
