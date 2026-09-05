# `ltx25/` — LTX-2.5 experiment, standalone

**This folder is deliberately independent of the v1–v7 line.** It shares no code, no
processed data, no config and no versioning with `training_approach/` or `finetune/wan/`.
Nothing in here is a port of anything. It reads `Data/raw/` and writes one self-contained
output folder.

That independence is the point: v1–v7 are a Wan lineage with continue-training, expert
splits and inherited recipes. This is a clean-slate LTX-2.5 experiment, so a result here
means something on its own rather than relative to a chain of prior runs.

```
ltx25/
├── docs/
│   ├── PIPELINE.md            raw -> eval, the design and the reasoning
│   └── LTX25_CAPABILITIES.md  what LTX-2.5 actually offers, and corrections to the older plan
├── prep/prep_ltx25.py         raw ProRes  ->  training corpus
├── eval/eval_ltx25.py         baselines, start frames, scoring, memorisation probe
└── configs/i2v_lora_ltx25.yaml   trainer config, every deviation justified inline
```

---

## The experiment in one paragraph

Generate a single short (≤2.5 s), on-model Pudgy clip of a named action or expression, for
a named character, from **a supplied first frame plus a caption**, delivered as an
individual GIF. No looping, no composition, no multi-shot: those are out of scope, and with
them go most of LTX-2.5's differentiators. One rank-32 I2V LoRA carries style and identity.

---

## Run order

```bash
# 0. GATE ZERO, before anything else. Round-trip Pudgy art through both 2.5 VAEs.
#    If thin black outlines shimmer or soften in a pure encode->decode, STOP:
#    no amount of data recovers what the VAE discards. See docs/PIPELINE.md §1 and §10.2.

# 1. Measure the SOURCE before judging any generation against it.
python ltx25/eval/eval_ltx25.py baseline

# 2. Build the corpus from raw.
python ltx25/prep/prep_ltx25.py --dry-run     # plan only, writes nothing
python ltx25/prep/prep_ltx25.py --workers 8

# 3. Start frames: a production pose library, and an eval set the model never sees.
python ltx25/eval/eval_ltx25.py startframes

# 4. Preprocess + train on the GPU box (see configs/i2v_lora_ltx25.yaml header for the
#    exact process_dataset.py invocation; use the bucket string from manifest.json).

# 5. Score. Generate through ltx-pipelines, NOT the trainer's validation sampler (#283).
python ltx25/eval/eval_ltx25.py probe --trained <gens_from_training_frames> \
                                      --unseen  <gens_from_unseen_frames>
python ltx25/eval/eval_ltx25.py score --gen <gens_from_unseen_frames>
```

---

## Five things this pipeline knows that are not obvious

Each was **measured** while building this, not assumed. All five are enforced in code.

1. **Frame length must not predict the label.** LTX needs `frames % 8 == 1`, which is
   coarse. Every label is also emitted at a shared floor of **17**, so one bucket carries
   all twelve labels. Asserted, and the build exits if it ever stops holding.

2. **The source is 37.9% frozen at the median, and `NEUTRAL` is 96.6% frozen** — a still
   image held for 60 frames. Laddering against raw length would have emitted 162 clips of
   static video, which teaches the model to generate static video. The ladder is built
   against the **active** length instead, trimming the dead tail but never below the floor.
   Frozen holds *inside* a clip are kept: those are the client's animation intent.

3. **The character fills ~92% of every source frame**, so there is no room to crop inward.
   Cropping gave `medium` and `close` a 2-point difference: three captions for one image.
   The shot ladder therefore zooms **out** by padding the canvas with flat ground, which is
   free because the ground is synthetic. Measured result: 45% / 66% / 90% character height.

4. **The nine camera angles are one performance from nine cameras**, so a random split
   leaks. The holdout is a whole **angle** (`SIDE_R`). This corpus can support a viewpoint
   holdout and a start-frame holdout, and **cannot support a novel-action holdout at all**.

5. **The stated success bar rewards memorisation.** "Closest to the training video" plus a
   supplied start frame is satisfied perfectly by replaying the training clip. Scoring
   therefore happens only on **unseen** start frames, and `eval probe` measures the gap
   between seen and unseen as the headline number.

---

## Raw-data defects the prep handles

Reported every run, never silently repaired, and nothing under `raw/` is ever modified.
The folder is authoritative for character and label; the filename supplies only the angle.

| Defect | Handling |
|---|---|
| Non-deterministic ProRes tearing | Tear check on the frames about to be written, up to 24 decode attempts |
| Files whose name disagrees with their folder (3 walking clips named `WAVING`) | Folder wins, disagreement reported |
| A Pax file filed under `POLLY/` | Folder wins, reported, and it happens to be a byte-duplicate |
| Byte-identical duplicate renders | Deduplicated by MD5 |
| Truncated streams (`PAX_MOTION_WALKING_QF1_R`: 11f of a claimed 16) | Dropped, by comparison against the cell's modal length |
| **Three different angle naming conventions** (`QF1_L`, `QF_L2`, `QF_L`, `FR`) | All normalised |

## Open items

- **`sad`, `scared` and `affectionate` have never been delivered.** Twelve labels exist;
  the original taxonomy wanted more.
- **No rear views of either character**, and no turnaround footage at all since the faulty
  front-hemisphere `PAX_TURNAROUND` was removed. Away-facing prompts are unconstrained.
- **Four files in the 2026-09-05 `CONFUSED` delivery are defective.** Two are truncated
  (`POLLY_MOTION_CONFUSED_QF1_R`, `POLLY_MOTION_CONFUSED_SIDE_L`: 18f against the cell's
  21f) and two tore on all 24 decode attempts and could not be written at all
  (`PAX_MOTION_CONFUSED_QF1_R`, `PAX_MOTION_CONFUSED_QF3_L`). Same fault class as the 16
  already on the re-export list. Add these four.
- **Nine cameras of one performance is one sample seen nine ways.** More *performances* per
  action would help far more than more angles.
