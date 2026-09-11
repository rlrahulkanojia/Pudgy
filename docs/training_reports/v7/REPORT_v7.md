# v7 — Motion + multi-expression LoRAs: run report

**Run window** 2026-09-05 → 2026-09-11 · **Hardware** 1× A100-80GB PCIe (driver 595.71.05)
**Status** All training complete. Gate coverage partial — see [`GATES_v7.md`](GATES_v7.md).
**GPU released** 2026-09-11 15:55 UTC.

---

## 1. Headline

| | |
|---|---|
| **Expression golden** | `pudgy-v7-expr-lownoise-step00002500.safetensors` |
| **Motion golden** | `pudgy-v7-motion-lownoise.safetensors` (step 2144) |
| **Frozen partner** | `lora_highnoise_GOLDEN_ep40` — v2, **untouched** |
| **Expression control** | 125/126 pairs distinct at 3 seeds |
| **Motion control** | all pairs distinct at every checkpoint |
| **Gates passed** | G-X, G-C, G-P · G-D partial |
| **Gates outstanding** | G-F, G-L, G-B, G-M, G-N, G-H, G-Z · G-R blocked |

**The result in one line:** v7 roughly tripled the vocabulary — 4 expressions to 7, plus 6
movements that did not exist — **without controllability degrading**, and put motion on the
low-noise expert so v2's motion prior stays frozen and bisectable.

---

## 2. What was trained

| Run | Expert | Init | Steps | Wall | Checkpoints | Final loss |
|---|---|---|---|---|---|---|
| `pudgy-v7-expr-lownoise` | low (ts 0–900) | v2 low golden | 3,600 | 43 h | 17 | 0.00203 |
| `pudgy-v7-motion-highnoise` | high (ts 900–1000) | v2 high golden | 2,144 | 16 h | 10 | 0.00569 |
| `pudgy-v7-motion-lownoise` | low (ts 0–900) | v2 low golden | 2,144 | 16 h | 10 | 0.00167 |

Recipe per plan §5 and **unchanged**: rank 16 / α 32, lr 3e-5, adamw8bit, fp16,
`shift` sampling with `discrete_flow_shift 5.0`, `preserve_distribution_shape`, seed 42,
1024², batch size 1, `num_repeats` 1, `save_every_n_steps 250`.

**Data:** `v7_primitives_2272` — 2,272 clips from 208 sources, 13 labels, 2 characters,
9 angles, 4 grounds, 3 shot sizes, duration ladder. 140 clips were new in the
iteration_4 delivery; the other 68 sources were already counted under v5/v6.

### Total cost, and why it beat v6 per unit of work

v6 was 2,992 steps in **52 h** (62.57 s/it) for one LoRA over 272 clips.
v7 was **7,888 steps in ~75 h** across three LoRAs over 2,272 clips.

**8× the footage and 3 models for 1.4× the time** — each step is ~30% faster than v6
(43.3 s/it expression, 26.8 s/it motion). The extra wall-clock is more work, not slower
work. Useful when explaining the schedule to non-technical readers.

---

## 3. Findings

### 3.1 More training does not buy separability — measured twice
| Sweep | Span | Spread in mean face-SSIM |
|---|---|---|
| Expression, 5 checkpoints | steps 500–3500 | **0.0040** |
| Motion, 4 checkpoints | steps 500–2000 | **0.0035** |

Both curves are flat. **The ~10 h continuation to the plan's 3,500-step ceiling was
therefore not run**, and the plan's §5.2 instruction — treat step count as an *output* of
the gates, not an input — is now supported by direct evidence on this dataset.

> **Correction to an earlier reading.** Mid-run this was reported as "the golden is early".
> That was wrong: lower SSIM means *more* distinguishable, so step 2500 (0.8092) is better
> than step 500 (0.8118), not worse. The flatness conclusion stands; the direction does
> not. v5's "golden is probably early" is **not** reproduced here.

### 3.2 Expression: one weak pair, and it localises to data
`confused|neutral` for **Polly on seed 42 only** — the other two seeds pass. `confused`
has **64 clips against 216** for mature labels and **7 source angles against 9**, after
two of its sources were dropped as damaged. It is also the subtlest expression in the set
(one eyebrow arching, a slight beak shift) against a deliberately still `neutral`.

**This is a bounded data ask, not a retrain.**

### 3.3 Motion belongs on the low-noise expert
See [`GATES_v7.md` §2](GATES_v7.md). M-high blurs actions together (0.9618, near v5's
"prompt ignored" 0.9692); M-low separates (0.9444) and animates ~50% more. The
**high-noise golden stays frozen**, so regressions remain bisectable and G-S is unnecessary.

### 3.4 Motion is real but subtle, and 2 s works
Generated motion runs 3–4× a generated static reference. `running` sustains at **2.9×**
its trained length (17 → 49 frames). Duration is controllable — all probe clips returned
exactly the requested length. But stretching gives **the action playing out over longer,
not the action repeating**: for a sustained walk or wave, beat chaining (plan T4) is still
required.

### 3.5 `standing_idle` is derived, and it shows
The client has never delivered an idle. `standing_idle` reuses `neutral` footage under a
second true caption, but that footage has a **frozen body** (measured MAE 0.028–0.064).
It therefore teaches "stop" as "freeze", and **G-N is expected to fail**. Beat chaining
needs a clean rest pose to hand off between beats, so this is on the critical path for
longer video. Requested in Client_Data_Request_Round4 §3.

---

## 4. Defects found in the plan and tooling

Recorded because each cost real time and will recur otherwise.

| # | Defect | Impact | Fix |
|---|---|---|---|
| 1 | **`BLOCKS_TO_SWAP=0` OOMs on expression.** The plan's value comes from v5, which only trained 21-frame clips; v7's f57 bucket is 2.7× that sequence length. | Run died at step 11 after a 28.6 GB model load. | Bisected to **16** (12 OOMs, 16 holds). ⚠️ **`PREFLIGHT_v6.md` §7.1 already documented this exact failure and used 32** — the finding never propagated into the v7 plan. Motion, whose longest bucket is f33, ran fine at **0** — worth ~10 h across the two arms. |
| 2 | **The goldens land where the trainer does not look.** `az storage blob download-batch` preserves the blob path, so they arrive at `v2_golden/v2/weights/curated/`, but `train_pudgy_v7.sh` reads `v2_golden/` flat. The handover's own §3 verification prints the nested path. | `FileNotFoundError` ~2 min in, after the DiT load. | Flatten after download. Verify against the path the **trainer** reads, not the one the download produced. |
| 3 | **`az download-batch` needs the destination to pre-exist.** | Goldens silently failed while the dataset download succeeded. | `mkdir -p` first. |
| 4 | **VAE/T5 paths in the handover are placeholders.** | §5 caching fails as written. | Real: `comfy21/split_files/vae/`, `t5/`. |
| 5 | **G-X's first verdict was wrong** — reused G-M's x-range bar on in-place actions. | Reported both arms failing on clips that visibly animate. | Metric now frame-to-frame MAE vs a generated static reference. **A threshold is only valid for the motion it was calibrated on.** |
| 6 | **The gate runner reloaded the model per clip.** | 501 clips → 501 loads ≈ 33 h of pure reloading. | Two-pass collect/replay batching: 15 loads. |
| 7 | **`--dry` runs wrote result files.** | 16 `status: dry` stubs reached Azure in directories named like real gate results — a directory that looked like a completed gate suite. | Dry runs now persist nothing. |
| 8 | **`upload_assets.py` flattened subdirectories.** | 70 sweep clips across 5 step folders share 14 filenames; 56 overwrote each other. | Keep the path relative to the source prefix. |
| 9 | **Blob layout doubled** (`v7/eval/eval_v7/…`). | Did not match the layout documented in the uploader's own docstring. | Drop the redundant segment. |

---

## 5. Where everything is

```
pudgy/v7/
├── weights/   expr-lownoise (17) · motion-lownoise (10) · motion-highnoise (10, the G-X loser, kept for the record)
├── eval/      sweep/ · sweep_motion/ · refine/ · gx/ · duration/ · keyframes/ · index.html
├── logs/      per-run tensorboard + trainer stdout
└── docs/      plan, handover, prep/train/eval/gate scripts
```

**wandb** — https://wandb.ai/imsaksham-/pudgy · expression `oer5dnli`, M-high `25aelj3y`,
M-low `l0ifads3`. (`3m6309nf` is the OOM'd first attempt, kept deliberately.)

**Review pages** (SAS, 14 days): expression sweep `v7/eval/index.html`, duration probe
`v7/eval/duration/index.html`.

**Dashboard** — v7 is the latest experiment, with live training status, the G-X verdict,
the dataset entry, and both video sections.

---

## 6. If someone picks this up

1. **Nothing needs retraining.** All weights are on Azure. Outstanding gates are pure
   inference against frozen checkpoints.
2. **Run G-F first** — it is the v5 failure mode and the single most valuable untested
   claim (~2 h).
3. **Get v2's 10 showcase prompts** from the author, or G-R can never run.
4. **Ask for `standing_idle` footage** — Round 4 §3. It unblocks G-N and beat chaining.
5. **Do not re-derive thresholds from another gate.** Defect 5 is the cautionary tale.
6. `/workspace` on that box was **not persistent**; everything here survived only because
   it was mirrored to Azure on a 30-minute timer.
