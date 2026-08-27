# v6 handover — findings, defects, and what to do next

Written to survive the GPU box. Everything here is measured, not assumed.
Companions: [`FINDINGS_v6.md`](./FINDINGS_v6.md) (summary) · [`REPORT_v6.md`](./REPORT_v6.md)
(run as executed) · [`PREFLIGHT_v6.md`](./PREFLIGHT_v6.md) (input state).

---

## 1. The run

| | |
|---|---|
| What | Multi-expression LoRA, continue-trained from the **v2 low-noise golden** |
| Data | 272 clips — 2 characters × 4 emotions × 3 shot sizes × 4 grounds, from 68 sources |
| Recipe | low-noise expert only · rank 16 / α 32 · lr 3e-5 · fp16 · flow-shift 5.0 · seed 42 |
| Cost | 11 epochs · 2,992 steps · **52 h** · 1×A100-80GB · `blocks_to_swap 32` |
| Golden | **epoch 11** — `v6/weights/pudgy-expr-v6-lownoise.safetensors` |
| Loss | 0.00308 → 0.00156 (non-monotonic at ep9→ep10) |

**Verdict: the run works and the capability is real.** The control (v2 golden alone,
no v6 LoRA) *cannot* render these expressions — asked for "angry" it produces a
malformed face: artifact marks for eyes, a mangled blob for the beak.

---

## 2. Per-gate results

| Gate | Result | Seeds | Key number |
|---|---|---|---|
| **G-L** length de-confound | **PASS** | 42, 7, 123 | 72/72 pairs distinct; worst 0.9109 vs 0.92 |
| **G-F** frame trigger | **PASS** | 42 | closest emotion match 0.8351 vs 0.92; no clip frozen |
| **G-B** unseen background | pending | 42, 7, 123 | — |
| **G-Z** shot size | pending | 42, 7, 123 | — |
| **G-H** sustained hold | pending | 42, 7, 123 | — |
| **G-L Polly** | pending | 42, 7, 123 | — |
| **G-C / G-P / G-R** | not run | — | — |

### G-L — the one that could have sunk the run

Each emotion ships at exactly one frame count (21 happy / 29 surprised / 37 angry /
57 neutral), so **clip length predicted emotion with 100% accuracy in the training
data**. The mitigation (a truncated duplicate of each longer emotion) was ruled out —
original lengths only — so it went in unmitigated.

It did not transfer. **f21, the pure-happy training length, shows the *greatest*
emotion separation** (mean 0.8514) — the opposite of what a learned shortcut produces.
The caption beat length as a signal, which is the contrastive design working.

Per-pair, across 3 seeds × 4 lengths (higher = harder to tell apart):

| pair | mean | worst |
|---|---|---|
| surprised \| neutral | 0.8475 | 0.9064 |
| happy \| surprised | 0.8465 | 0.8774 |
| surprised \| angry | 0.8433 | **0.9109** |
| happy \| neutral | 0.8424 | 0.8886 |
| angry \| neutral | 0.8352 | 0.8782 |
| happy \| angry | **0.8266** | 0.8639 |

**All three `surprised` pairs are the three hardest.** `surprised` is the weak class.

Margins are strongly seed-dependent — 0.046 (s42) → 0.014 (s7) → 0.009 (s123). A
single seed would have overstated robustness by ~5×.

### G-F — v5's actual failure mode, not reproduced

v5 learned "this frame → happy": driven from the training frame, motion collapsed to
0.8 px and the happy arc fired regardless of prompt. v6 does not do this. Driven from
the training frame at all three shot sizes with a prompt asking for *no* expression,
the output never resembles any emotion (closest 0.8351 vs a 0.92 bar) and nothing is
frozen (f0-vs-last SSIM 0.88–0.96, frame-diff 0.99–2.21).

### Golden sweep — all checkpoints are equivalent

| epoch | steps | mean | worst |
|---|---|---|---|
| 1 | 272 | 0.8248 | 0.8668 |
| 4 | 1,088 | 0.8220 | 0.8658 |
| 8 | 2,176 | 0.8231 | 0.8696 |
| 11 | 2,992 | 0.8221 | 0.8656 |

Spread 0.0027 — noise. All fail on the same pair. Visually equivalent on two emotions.
**Epoch 11 was chosen because it is the only checkpoint with full gate coverage, not
because it is better.**

Distance from the ep0 baseline decomposes as: **ep0→ep1 ≈ 76% of the total change**,
ep1→ep4 ≈ 18%, ep4→ep11 ≈ 7%.

---

## 3. Defects in the model as it stands

1. **`surprised` is the weakest expression.** Hardest to distinguish from every other
   emotion, and visibly weaker on unseen backgrounds (no flippers-to-face gesture).
2. **Wide framing loses expression legibility.** At 0.55× the face is ~55% linear size
   before the 8× VAE shrinks it again; the showcase's wide clip has barely readable
   brows. The ladder makes *framing* promptable; the *expression* does not survive it.
3. **Shortest length is the tightest.** f21 has the least separation of the four
   lengths (0.8514 vs 0.8261 at f37) — fewer frames for the arc to develop.
4. **Image-conditioned only.** The box holds `i2v-A14B` weights; every generation needs
   a start image. Text-only would need a separate t2v checkpoint.
5. **1024² is off Wan's canonical aspect list** (720×1280 / 480×832 / …) — a known
   quality tax inherited from square source art. Held constant for comparability.
6. **Coverage is uneven.** G-L got 3 seeds; G-F got 1. G-C/G-P/G-R never ran. The
   emotion × length matrix is Pax-only.

---

## 4. Process and tooling defects found (all fixed, all worth knowing)

Each of these **succeeded loudly while doing less than claimed** — the common shape.

| # | Defect | Consequence if unnoticed |
|---|---|---|
| 1 | **musubi writes the final epoch with no epoch number** (`<name>.safetensors`, not `-000011`) | A regex on the numbered form silently drops the last checkpoint — the likely golden — and reports success |
| 2 | **`--save_path` is always a *directory*** in musubi; files inside are timestamp-named | Passing `x.mp4` creates a *directory* named `x.mp4`. `exists()` then matches it, so reruns "skip" work never done, and `*.mp4` globs break |
| 3 | **Whole-frame SSIM understates facial change** | ~85% of the frame is body/background the prompt cannot alter. A visually obvious difference measured **0.0015** from "identical". Fixed: score on a face crop (separation widened 0.04 → 0.10) |
| 4 | **The plan's `x-range > 100 px` is wrong for G-F** | Calibrated from v5 where the character *walked*; G-F asks it to *turn in place*, which barely moves the centroid. Produced a **false FAIL** on clips performing the action correctly. Fixed: gate on "not frozen" |
| 5 | **`blocks_to_swap: 0` does not generalise** | Inherited from v5, which only trained 21-frame clips. v6's 57-frame bucket **OOMs on step 0**. Probed: bs=24 → 76.1 GB, bs=32 → 70.7 GB |
| 6 | **Inference cost is model loading, not generation** | GPU sits at **0–20%** while two 14B experts (~57 GB) load and fp8-quantise. Per-clip ≈ 20 min; batched via `--from_file` ≈ 33 min for 12 clips |
| 7 | **The Azure mirror silently skipped media** | `classify()` had no `.mp4` branch — uploads reported "0 files" while no video went up |
| 8 | **Deploy used blocking ZipDeploy** | App Service cuts sync SCM requests at ~230 s; the job had crept to 3m01s and hit **HTTP 504**. The 504 means the *request* timed out, not the deploy — so it reports failure for deploys that landed. Fix in PR #3 |

**Process failure worth recording:** the plan mandates a gate at ~epoch 2 (~9 h) before
committing the remaining ~42 h. It was approved and **never executed** — training ran
unattended to epoch 10. No data was lost, but the option to stop early was forfeited.

---

## 5. Improvements, ranked

**Data (needs the client)**
1. **More `surprised` footage** — the measurably weakest class.
2. **Expression coverage at smaller framings**, if wide shots matter for delivery. The
   current ladder is synthesised from close-up sources, so it teaches framing without
   teaching a legible face at distance.
3. **More than one performance per cell.** Every cell is still one take from N angles;
   three shot sizes × four grounds × nine angles of a single take is one take.

**Training**
4. **Train 1–4 epochs, not 11.** Epoch 1 is indistinguishable from epoch 11 on every
   measure applied and does ~76% of the work. Saves ~45 GPU-hours per run.
5. **Spend the saved time on evaluation breadth** — seeds and characters, not steps.
6. Re-probe memory whenever clip length changes; `blocks_to_swap` is length-dependent.

**Evaluation**
7. **Run the ~epoch-2 gate.** It exists to avoid paying for a broken run.
8. **≥3 seeds on every gate**, not just the convenient ones. Margins moved 5× across
   seeds here.
9. **Batch everything through `--from_file`.** One model load per checkpoint.
10. **Match the metric to the motion being asked for** — translation vs rotation vs
    expression are three different measurements.

**Infrastructure**
11. Merge PR #3 (async deploy + polling).
12. Consider committing `Data/`'s manifests, or at least their checksums — the tree
    lives outside git and Azure is currently its only second copy.

---

## 6. If this machine dies

**Safe in Azure** (`pudgytraining` / container `pudgy`, prefix `v6/`):

```
v6/weights/     11 checkpoints (3.4 GB) — includes the golden
v6/inference/   86 renders — showcase, G-L matrix, G-F, epoch series, ep00 control
v6/eval/        metrics_v6.json, gate JSON reports
v6/logs/        trainer stdout + tensorboard
v6/docs/        these documents
```

**Safe in git** — all code: `setup_wan_env.sh`, `train_pudgy_expr_v6.sh`,
`prep_expressions_v6.py`, `prep_eval_keyframes_v6.py`, the gate harness
(`gates_v6.py`, `gate_gl_v6.py`, `gate_gf_v6.py`, `gates_remaining_v6.py`),
`batch_render_v6.py`, `showcase_v6.py`, `sample_epochs_v6.py`, `azure_mirror_v6.py`.

**Lost, and acceptable:** the 11 `--save_state` resume directories (~1.7 GB local
only). They allow exact-resume; checkpoints are what matter and those are mirrored.

**Lost, and worth knowing:** `/workspace/wan_cache/latents_expr/` (3.2 GB of cached
latents + T5 embeddings). Regenerating costs ~25 min of GPU after re-downloading the
65 GB of base weights.

**To rebuild from nothing:**
```bash
bash setup_wan_env.sh                      # venv + musubi 0.3.4 + 65 GB weights
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace --pattern "processed/v6_expressions_272/*"
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace/wan_output/v2_golden \
   --pattern "v2/weights/curated/*GOLDEN_ep40*"
# then pre-cache (PREFLIGHT §7) and run train_pudgy_expr_v6.sh
```

⚠️ **`/workspace` is not a persistent volume.** Azure is the durable copy. Secrets live
in `/workspace/.env` (untracked) and are **not** backed up anywhere — the Azure
connection string and W&B key must be re-supplied on a new box.
