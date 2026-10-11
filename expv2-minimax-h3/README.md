# `expv2-minimax-h3/` — isolated experiment, not a version

**Status:** plan (2026-10-11). Nothing in this folder has run yet.

**This is an isolated test.** It is not v10, not a continuation of v1–v9, and not a
follow-up to `expv1-kandinsky-50-video-pro`. It starts from the public **MiniMax H3 FL2VA**
checkpoint (the community AdaLN-pruned bf16 build). No earlier weights, merged bases,
processed datasets, configs or recipes are carried over. It gets its own corpus built from
`Data/raw/`, its own code and its own Azure prefix.

**It does carry knowledge.** The lessons measured on this footage in v1–v9 and `ltx25/` are
indexed, with sources, in expv1's register
[`../expv1-kandinsky-50-video-pro/docs/LEARNINGS_CARRIED.md`](../expv1-kandinsky-50-video-pro/docs/LEARNINGS_CARRIED.md)
(tags A1–E). That file is read as a document only. [`docs/LEARNINGS_CARRIED.md`](docs/LEARNINGS_CARRIED.md)
says how each lesson applies to H3 and adds what is new.

> **Isolation, precisely:** expv2 may *read* earlier code and docs, and may *copy* a script
> into this folder with its provenance noted. It never *imports* from `finetune/`, `ltx25/`,
> `prep/` or `expv1-*`, never loads a Wan, LTX or Kandinsky LoRA, and never trains from a
> v-line checkpoint. The one exception is **measurement**: G0.1 also round-trips the same
> clips through Wan's VAE as a same-clip reference, and §8 re-scores v8@550's existing clips.
> Neither touches training.

**Licence:** set aside at the user's request (2026-10-11). It is not a gate in this plan.
The notes are in the evaluation's Appendix A.

---

## The experiment in one paragraph

Can **MiniMax H3 FL2VA** (33B single-stream DiT, CFG-distilled, 24 fps native), fine-tuned
with **one LoRA** on the client's alpha clips, produce **on-model, flat-2D Pudgy clips** in
which the **action or expression named in the prompt** plays out from an unseen first frame?
The test is at **768×768, 22 frames (0.92 s)**, with longer 39/56-frame rungs only where the
footage allows. Single characters only, 12 of 13 labels (`running` has no legal-length
footage until Round 5). If it passes, a head-to-head with v8@550 on the same held-out start
frames says whether H3 is worth a Round 5 re-run as the replacement candidate.

| | |
|---|---|
| **Base** | `Comfy-Org/MiniMax-H3: diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` + `MiniMax/MiniMax-H3: FL2VA/{text_encoder,video_vae,audio_vae}` |
| **Trainer** | DiffSynth-Studio (pinned commit), two-stage cache → train, first frame only (`--extra_inputs "input_audio,input_image"`, no `end_image`), silent audio |
| **Trains** | one LoRA, rank 32 on `attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2` (155.1M params), guidance loss at scale 4 |
| **Data** | own corpus from `Data/raw/iteration_3–5`: 170 training sources, 980 clips; `QF2_R` held out (23 clips), two grounds never trained |
| **Verdict** | pre-registered bars ([`docs/PLAN.md`](docs/PLAN.md) §9), measured only on start frames the model did not train on |
| **Replacement read** | head-to-head with v8@550 on shared labels ([`docs/PLAN.md`](docs/PLAN.md) §8). It labels H3 a *replacement candidate* or not. The replacement decision itself waits for the Round 5 re-run |

---

## Read in this order

| Doc | What it answers |
|---|---|
| [`../docs/documents/research/MiniMax_H3_Evaluation.md`](../docs/documents/research/MiniMax_H3_Evaluation.md) | Why H3 at all: architecture, leaderboards, data fit, compute, trainer survey, pros and cons vs v1–v9 |
| [`docs/MODEL_NOTES.md`](docs/MODEL_NOTES.md) | The facts this plan relies on, and **16 gotchas** that would silently corrupt a first run (distillation, frame 0 not pinned, the 17n+5 snap, pruned-layout LoRAs, train/inference shift mismatch, the first+last-frame default, mp4 bitrate, …) |
| [`docs/LEARNINGS_CARRIED.md`](docs/LEARNINGS_CARRIED.md) | How each v1–v9 lesson applies to H3, plus the new facts (the raw length re-count) |
| [`docs/PLAN.md`](docs/PLAN.md) | Phase 0 measurements, corpus, recipe, pilots, evaluation tiers, the v8@550 head-to-head, the pass/fail rule fixed before any data exists, cost |

---

## Run order

```text
Phase 0 — measure, no training                                (PLAN §4)
  G0.1  VAE gate: Pax and Polly through H3's VAE at 768², Wan's VAE as reference
        -> STOP here if it fails (no LoRA recovers what the VAE discards)
  G0.2  pipeline checks: pruned-vs-full parity, conditioning dump, mixed-length cache,
        two-pass eval equivalence, LR schedule, lossless writer
  G0.3  base Tier-1 baseline on the 23 held-out clips (the control's likelihood)
  G0.4  zero-shot probe = the base's `core` + `native` suites (the control's generations)
  G0.5  source baseline: motion, holds, cadence, colour, ground of the real clips
  G0.6  memory + throughput on the actual box, f56 with guidance loss on

Phase 1 — corpus                                              (PLAN §5)
  prep/prep_expv2.py  raw -> Data/processed/expv2_h3/   (asserts lengths, holdout, balance)
  stage 1 cache (Qwen3-VL-32B + VAE), once

Phase 2 — train                                               (PLAN §6)
  pilots: shift 12 vs shift 2.22, 100 steps each -> Tier 1 picks
  main:   winner, from zero, 490 optimizer steps (3,920 samples), save every 50
  fallback arm D (DeCFG adapter) only if the pre-declared wash-out trigger fires

Phase 3 — evaluate and decide                                 (PLAN §7–9)
  Tier 1 on every checkpoint -> winner; Tier 2 suites -> verdict; §8 head-to-head

Phase 4 — Round 5 re-run (separate plan amendment, written when the data lands)
```

---

## Layout

```text
expv2-minimax-h3/
├── README.md                 this file
├── docs/
│   ├── PLAN.md               design, gates, pre-registered verdict, cost, runbook
│   ├── MODEL_NOTES.md        H3 facts this plan depends on, and the gotchas
│   └── LEARNINGS_CARRIED.md  v1–v9 lessons as they apply to H3, plus new ones
├── patches/                  (to write) DiffSynth: train-shift flag, cosine LR if absent
├── prep/                     (to write) prep_expv2.py: raw -> corpus, metadata.csv per bucket
├── eval/                     (to write) vae gate, pipeline checks, Tier 0–2, head-to-head
└── configs/                  (to write) launch scripts for cache, pilots, main, arm D
```

## Naming and storage

| | |
|---|---|
| Convention | `expvN-<model>`, numbered globally across isolated experiments. `expv2` is this H3 test; a Kandinsky follow-up would take the next free number, not `expv2` |
| Processed data | `Data/processed/expv2_h3/` |
| Azure | container `pudgy`, prefix `expv2-minimax-h3/{weights,eval,logs,docs}` |
| Run names | `expv2-h3-*` (`expv2-h3-s12`, `expv2-h3-s222`, `expv2-h3-main`, `expv2-h3-d`) |
| Source research | [`../docs/documents/research/MiniMax_H3_Evaluation.md`](../docs/documents/research/MiniMax_H3_Evaluation.md) |
