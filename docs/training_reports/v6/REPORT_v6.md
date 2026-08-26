# v6 — multi-expression LoRA on the v2 Wan2.2 goldens: run report

**Status:** 🟡 **trained, evaluation in progress.** Training is complete and final; the
behavioural gates that decide whether it is *good* have not yet run. Nothing in §4 should
be read as a quality verdict.
**Plan:** [`Training_Approach_v6.md`](../../../training_approach/v6/Training_Approach_v6.md)
**Input state:** [`PREFLIGHT_v6.md`](./PREFLIGHT_v6.md) — dataset, preprocessing, environment
**Precedent:** [`v5/REPORT_happy_pilot.md`](../v5/REPORT_happy_pilot.md)

---

## 1. What ran

| | |
|---|---|
| Base | Wan2.2-I2V-A14B · Wan2.1 **8×** VAE · UMT5-XXL |
| Init | `lora_lownoise_GOLDEN_ep40` (v2) via `--network_weights` — continue-train, fresh optimiser |
| Frozen | `lora_highnoise_GOLDEN_ep40` (v2) — inference partner, never trained |
| Trained | one LoRA, **low-noise expert only**, timesteps 0–900 |
| Data | 272 clips · 2 characters × 4 emotions × 3 shot sizes × 4 grounds |
| Recipe | rank 16 / α 32 · lr 3e-5 · adamw8bit · fp16 · flow-shift 5.0 · seed 42 |
| Steps | **2,992** (11 epochs × 272) |
| Wall time | **52 h 00 m** @ 62.57 s/it |
| Hardware | 1× A100-SXM4-80GB, `blocks_to_swap 32` |
| Output | `pudgy-expr-v6-lownoise` → Azure `pudgy/v6/` |
| wandb | [`udv3u9hi`](https://wandb.ai/rlrahulkanojia/pudgy/runs/udv3u9hi) (project `pudgy`) |

Run `y5e13qk6` in the same project is the **failed first launch** (OOM on step 0, §3.1) and
should be ignored.

## 2. Provenance — this is a continue-train

| Check | Result |
|---|---|
| Trainer log | `load network weights from …lora_lownoise_GOLDEN_ep40.safetensors: <All keys matched successfully>` |
| Key set, epoch 1 vs init | 1200 / 1200 identical |
| **Zeroed `lora_up` blocks, epoch 1** | **0 / 400** |
| Relative L2 drift vs init, epoch 1 | 0.0065 |
| Mean cosine similarity, epoch 1 | 0.9931 |

**0/400 is the load-bearing number.** A from-scratch LoRA initialises every `lora_up` to
zeros, so 0/400 proves the v2 golden was genuinely the starting point rather than a file
that merely loaded without error.

The drift is ~8× smaller than v5's quoted epoch-1 figure (0.054), which is expected and not
a concern: v5's number came from **run 1, the high-noise expert**, which started at loss
0.0255 — an order of magnitude above ours. v6 continues the *low-noise* expert from a
low-noise golden and started near 0.003, matching v5's run-2 profile, which also moved
gently. Small drift is the signature of continuing an already-fluent expert.

> ⚠️ Drift is **not** a quality signal. v5 §4.3 measured a reassuring 13.7% drift and read
> it as "prior safely preserved" — then the behavioural test found the model ignoring
> prompts entirely. Its own conclusion: *"weight-space distance is not a proxy for
> behavioural preservation."* §4 below is where quality gets decided.

## 3. Loss

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| avr_loss | .00308 | .00243 | .00214 | .00199 | .00199 | .00195 | .00184 | .00171 | .00160 | .00172 | **.00156** |

Fast decline through epoch 4, then a long flattening. **Not monotonic** (ep9 → ep10 rises),
which matters only as a reminder that loss is not the selection criterion. v5 §4.4 found
motion responsiveness *decayed* with training — ep04 moved 259 px where ep18 moved 221 px —
so the golden is chosen behaviourally, and may well be early.

### 3.1 The run had to be restarted once — block-swap

The first launch used the plan's `blocks_to_swap: 0` and **died on step 0** with
`OutOfMemoryError` inside `WanRMSNorm._norm`'s fp32 upcast. The plan's setting came from
v5 §4.7, but v5 only ever trained 21-frame clips; v6's 57-frame bucket is 2.5× the sequence
length. Probed on f57: **bs=0 OOM · bs=24 → 76.1 GB / 72.7 s/it · bs=32 → 70.7 GB / 74.3
s/it**. Chose 32 for double the headroom at ~2% cost. Full detail in
[`PREFLIGHT_v6.md` §7.1](./PREFLIGHT_v6.md). Nothing else about the recipe changed.

Throughput note: single-bucket probes predicted 41.7 s/it; the real interleaved run measured
**62.6 s/it**. Probes bound *memory* reliably and *throughput* only loosely.

## 4. Evaluation — IN PROGRESS

### 4.1 Stage 1 — epoch-wise visual series (running)

Two expressions per checkpoint × 11 checkpoints, single seed, rendered from the training
start frame at the trained length. Purpose is a quality-vs-epoch curve for narrowing the
golden search, **not** a gate.

First result, **epoch 1, Pax/happy**: on-model — correct blue, thick black outlines, flat
pastel fills, correct proportions. A legible happy arc (open-eyed neutral → eyes closing →
open-mouth grin with squinted eyes → held). No drift, no melting, no mid-clip vanish (the
v1/CogVideoX failure). Flat 2D preserved; Wan's photorealistic bias did not trigger,
consistent with v2 and v5.

> **Read stage 1 narrowly.** Every axis is the trained one: trained emotion, trained
> background, trained start frame, trained length, one seed. It shows the pipeline emits
> valid on-model output. It cannot certify a checkpoint — v4 §5.1 found seed 42 alone hid
> failure rates of 40–66%.

### 4.2 Gates — NOT YET RUN

| Gate | Question | Status |
|---|---|---|
| **G-L** | Does emotion track the **prompt** or the **clip length**? | ⏸ **highest priority** |
| **G-F** | Does the training start frame trigger an expression regardless of prompt? | ⏸ |
| **G-C** | Do the 4 emotions separate, ≥3 seeds? (pairwise SSIM < 0.95) | ⏸ |
| **G-M** | Does motion survive from a novel frame? | ⏸ |
| **G-P** | Polly parity with Pax | ⏸ |
| **G-B** | Unseen-background invariance | ⏸ |
| **G-H** | Expression held past f30 on long clips | ⏸ |
| **G-Z** | Shot size promptable; expression legible at wide | ⏸ |

**G-L runs first.** The length↔emotion bijection went into training **unfixed** — every
emotion exists at exactly one frame count, so length predicts emotion with 100% accuracy in
the training data. If the model learned that shortcut, controllability numbers are moot.
Recovery, if it failed, is `prep_expressions_v6.py --common-bucket 21` + retrain (plan
§10.2); that flag is implemented and tested.

**G-F is second.** It is the exact failure that killed v5's high-noise run — all 28 v5 clips
began from ~one frame and always ended happy, making that frame a near-deterministic
trigger (motion collapsed to 0.8 px from it, versus 221–267 px from a novel frame).

## 5. Process failures worth recording

Three, all caught, none affecting the trained weights — but each would have produced a
confident-looking wrong result:

1. **The epoch-2 gate was approved and never executed.** The plan says gate at ~9 h before
   committing the remaining ~42 h; training ran unattended to epoch 10 instead. No data was
   lost (every checkpoint is retained and mirrored) but the option to stop early was
   forfeited. The gates now cost GPU time instead of saving it.
2. **musubi writes the final epoch without an epoch number** (`pudgy-expr-v6-lownoise.safetensors`,
   not `-000011`). The sampler's regex matched only the numbered form, so it would have
   rendered epochs 1–10, reported success, and silently omitted epoch 11 — the most likely
   golden. Fixed with `with_final()`.
3. **`--save_path` is always a directory in musubi.** Passing `<tag>.mp4` created a
   *directory* of that name with the video inside, which made `exists()` checks match
   directories (so reruns "skip" work never done) and broke every `*.mp4` glob. Fixed by
   generating into a scratch dir and renaming; also switched `--output_type both` → `video`
   to stop writing a 6 MB unused latent per clip.

Common thread: each failure mode **succeeds loudly while doing less than claimed**. Worth
assuming the next one has the same shape.

## 6. Artifacts

Azure account `pudgytraining`, container `pudgy`, prefix `v6/`:

```
v6/
├── weights/     11 checkpoints (epochs 1-10 numbered + final)
├── logs/        trainer stdout + tensorboard events
├── docs/        this report, PREFLIGHT_v6.md, the plan
└── inference/   stage-1 epoch sample series (in progress)
```

Local: checkpoints + 11 `--save_state` dirs (1.7 GB total) at
`/workspace/wan_output/pudgy-expr-v6-lownoise/`. `/workspace` is **not** a persistent
volume — Azure is the durable copy.

## 7. Next

1. Finish stage 1; read the epoch series for a shortlist.
2. **G-L on the shortlist** — the decisive test.
3. G-F, then G-C/G-M at ≥3 seeds on surviving candidates.
4. Full suite on the golden; then the 10-scene v2 regression (G-R).
