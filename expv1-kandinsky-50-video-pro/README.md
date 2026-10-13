# `expv1-kandinsky-50-video-pro/` — isolated experiment, not a version

**Status (2026-10-13, 23:50Z):** arm A trained (600 steps); winner **expv1@550**. Full
findings, numbers and logic: [`docs/REPORT.md`](docs/REPORT.md).

- **Phase 0:** G0.1 VAE **PASS**, G0.2 conditioning **PASS**, G0.3 layout **A** (generator
  patched, MODEL_NOTES §2.12).
- **Gates:** step 100 PASS (loss below base on 26/26 held-out clips); step 300 expression
  margin above base, motion not (reported, not a stop rule).
- **Tier 1 (held-out `QF2_R`), expv1@550:** expression **0.75** (base 0.375, v8 0.81), motion
  0.30 (base 0.30, v8 0.80). Expression is learned; motion labels are not.
- **Motion probe:** no arm-B trigger; the LoRA fixes under-motion on run/jump but over-moves
  small actions (1.5-4.5x source).
- **vs v8@550 (PLAN §8b): not better.** Both primaries favour v8, neither significantly
  (Tier 1 0.58 vs 0.81; Tier-2 CLIP 1-NN 0.385 vs 0.423). N2-N5 hold; N1 (blinded rubric) awaits
  a human rater, but cannot make it BETTER. expv1 is level/ahead on expressions and camera
  steadiness, behind on motion labels. A zoom-measurement bug was found and fixed on the way
  (REPORT §11.2). Not a replacement candidate.

**This is an isolated test.** It is not v10, and it is not a continuation of v1–v9. It
starts from the public **Kandinsky 5.0 Video Pro** image-to-video checkpoint. No earlier
weights, merged bases, processed datasets, configs or recipes are carried over. It gets
its own corpus built from `Data/raw/`, its own code and its own Azure prefix. A result here
stands on its own, not relative to a chain of earlier runs.

**But it is not a clean slate either.** Ten runs on this footage (v1–v8 on CogVideoX, Wan
and LTX-2.3, plus the standalone LTX-2.5 run) measured a lot about the art, the client data,
how to evaluate, and what goes wrong on the GPU box. expv1 uses all of that as
**knowledge**: every carried-over lesson is listed in
[`docs/LEARNINGS_CARRIED.md`](docs/LEARNINGS_CARRIED.md) with its source and how it is
applied. Lessons that came from a specific *model* (Wan's two experts, LTX's rank split) are
re-tested here as hypotheses, not assumed.

> **Isolation, precisely:** expv1 may *read* earlier code and docs, and may *copy* a script
> into this folder with its provenance noted. It never *imports* from `finetune/`,
> `ltx25/` or `prep/`, never loads a Wan or LTX LoRA, and never trains from a v-line
> checkpoint. That way, later changes to the v-line cannot change an expv1 result.

---

## The experiment in one paragraph

Does **Kandinsky 5.0 Video Pro I2V** (19B, a single transformer, 24 fps native, MIT
licence), fine-tuned with **one LoRA** on the client's alpha clips, produce **on-model,
flat-2D Pudgy clips** in which the **action or expression named in the prompt** plays out,
starting from a supplied first frame the model never trained on? It covers single
characters only, at 512×512, in clips of 2.4 s or less. No long-form scenes, two-character
shots, props or end-frame anchoring: those are later questions, and only worth asking if
this one passes.

| | |
|---|---|
| **Base** | `kandinsky5pro_i2v_sft_5s.safetensors` (HF `kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s`), as published |
| **Trainer** | musubi-tuner **v0.3.4** (Kandinsky 5 support built in) + 2 patches in `patches/` |
| **Trains** | one LoRA, rank 32, from zero |
| **Data** | own corpus from `Data/raw/iteration_3–5`, the `QF2_R` camera angle held out, two background colours never trained |
| **Verdict** | pre-registered bars ([`docs/PLAN.md`](docs/PLAN.md) §9), measured only on start frames the model did not train on |
| **vs v8@550** | v8@550's existing Tier-1 rows and `core` clips are re-scored with expv1's scorer and compared pair by pair (PLAN §8b: exact McNemar, Holm, non-inferiority gates). v8 is **not** a parent, and the comparison never changes the §9 verdict |

---

## Read in this order

| Doc | What it answers |
|---|---|
| [`docs/MODEL_NOTES.md`](docs/MODEL_NOTES.md) | What Kandinsky 5.0 Video Pro is, and **seven code-level gotchas** that would silently corrupt a first run (anti-cartoon default negative prompt, prompt rewriting, an I2V conditioning mismatch in both LoRA trainers, …) |
| [`docs/LEARNINGS_CARRIED.md`](docs/LEARNINGS_CARRIED.md) | What earlier runs taught, sorted into *carries over as fact*, *re-tested as hypothesis* and *does not carry over* |
| [`docs/PLAN.md`](docs/PLAN.md) | The design: Phase 0 measurements, corpus, recipe, evaluation tiers, gates, and the pass/fail rule fixed before any data exists |

---

## Run order

```text
Phase 0 — measure, no training                          (PLAN §4)
  G0.1  VAE gate: Pudgy art through Kandinsky's own HunyuanVideo-VAE code path
  G0.2  conditioning check: dump the I2V input tensors, train vs inference
  G0.3  base-model likelihood under both conditioning layouts (no LoRA)
  G0.4  zero-shot probe: unseen start frames, Pudgy negative prompt, 3 seeds
  G0.5  source baseline: motion energy / frozen % of the real clips
  G0.6  memory + throughput probe on the actual box
        -> STOP here if G0.1 fails (no LoRA recovers what the VAE discards)

Phase 1 — corpus                                        (PLAN §5)
  prep/prep_expv1.py  raw -> Data/processed/expv1_k5pro/  (asserts length/shot balance)

Phase 2 — train                                         (PLAN §6)
  one LoRA, <= 600 optimizer steps, save every 50, gate breaks at 100 and 300

Phase 3 — evaluate and decide                           (PLAN §7–9)
  Tier 1 diffusion classifier -> winner; Tier 2 generated suites -> verdict
```

---

## Layout

```text
expv1-kandinsky-50-video-pro/
├── README.md                 this file
├── docs/
│   ├── PLAN.md               design, gates, pre-registered verdict
│   ├── LEARNINGS_CARRIED.md  what earlier runs taught, and how expv1 uses it
│   └── MODEL_NOTES.md        the base model, and the gotchas found in its code
├── expv1_common.py          paths, labels, the frozen caption grammar, raw-source discovery
├── patches/                  resume-step fix (copied) + I2V conditioning fix (trainer, cacher, generator)
├── prep/                     (to write) prep_expv1.py: raw -> corpus
├── eval/
│   ├── vae_gate.py           G0.1
│   ├── cond_check.py         G0.2 (+ gen --layout A after G0.3)
│   ├── build_refs.py         Tier-1 clip sets, the real-clip bank, start frames
│   ├── dcls.py               Tier 1 diffusion classifier; G0.3 (`score --true-only`, `g03`)
│   ├── gen_core.py           Tier-2 `core` suite (one model load, layout A)
│   ├── metrics.py            measurements (copied from expv2)
│   ├── stats.py              exact McNemar, posterior P(better), Wilcoxon, bootstrap, Holm, power
│   └── compare_v8.py         PLAN §8b: is expv1 better than v8@550?
└── configs/                  (to write) dataset .toml, launch script
```

## Naming and storage

| | |
|---|---|
| Convention | `expvN-<model>`: isolated experiments, numbered on their own, never merged into the v-line. `ltx25/` predates this convention and is the same kind of experiment |
| Processed data | `Data/processed/expv1_k5pro/` |
| Azure | container `pudgy`, prefix `expv1-kandinsky-50-video-pro/{weights,eval,logs,docs}` |
| W&B | project `pudgy`, run names `expv1-k5pro-*` |
| A follow-up | `expv2-…` (e.g. 1024², two characters), never `v10` |
