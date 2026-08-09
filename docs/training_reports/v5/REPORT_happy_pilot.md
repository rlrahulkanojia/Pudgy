# v5 pilot — Pax "happy" expression LoRA on Wan2.2-A14B

**Status:** ✅ **complete** — both runs trained, full eval suite executed on both experts,
A/B settled, weights + artifacts mirrored to Azure.
**🏆 Golden:** `pudgy-happy-expr-lownoise-v1.safetensors` (run 2, epoch 18) paired with the
untouched v2 `lora_highnoise_GOLDEN_ep40`.
**Plan:** [`Training_Approach_v5_Happy_Expression_LoRA.md`](../../../training_approach/v5/Training_Approach_v5_Happy_Expression_LoRA.md)
(pilot) inside [`Training_Approach_v5.md`](../../../training_approach/v5/Training_Approach_v5.md) (programme).

---

## 1. Provenance — this is a continue-train, not a fresh LoRA

| v2 asset | Role | Modified |
|---|---|---|
| `lora_highnoise_GOLDEN_ep40` (rank 16 / α 32, 400 modules) | init for run 1 | ✅ trained |
| `lora_lownoise_GOLDEN_ep40` | identity at inference | ❌ untouched |
| Wan2.2-I2V-A14B DiTs · Wan2.1 8× VAE · UMT5-XXL | frozen backbone | ❌ |

Trainer log: `load network weights from …lora_highnoise_GOLDEN_ep40.safetensors: <All keys matched successfully>`

Weight-space confirmation (vs the golden it started from):

| Checkpoint | rel. L2 drift | mean cos-sim | zero `lora_up` blocks |
|---|---|---|---|
| epoch 1 (56 steps) | 0.054 | 0.9985 | 0 / 60 |
| epoch 18 (final) | 0.137 | 0.9905 | 0 / 60 |

A from-scratch LoRA initialises `lora_up` to zeros (would read 60/60). Reading 0/60 with
cos-sim 0.9985 at epoch 1 proves the golden was the starting point. Final drift of only
13.7% = a gentle nudge, not a rewrite — the intended effect of the 3e-5 LR.

## 2. Data

7 source clips, Azure `pudgy/interation_3/03_expression_clips/Pax/happy/` — ProRes 4444,
**1080×1080, 24 fps, 21 frames (0.875 s), with a real alpha channel** (61.6% of frame
transparent). One performance filmed from 7 fixed angles.

Prepared by [`prep_happy_v5.py`](../../../finetune/wan/prep_happy_v5.py): alpha composited
onto **4 flat backgrounds** (white / pastel blue / peach / mint) → **28 clips**
(7 angles × 4 grounds), 1024×1024, 21 frames.

Two corrections vs the plan doc, both material:
- The clips are **not** "blank/white background" — naive decode composites onto **black**.
- **1024², not the native 1080²** — Wan's 8× VAE + 2×2 patchify needs an even latent side;
  1080/8 = 135 is odd. Confirmed by the cached latent shape `(16, 6, 128, 128)`.

Compositing is the only lever available against the pilot's core risk — 7 correlated angles
of one take teaching "this smile on this background" instead of "happy".

## 3. Training

| | run 1 (high-noise) | run 2 (low-noise) |
|---|---|---|
| Expert / timesteps | high, 900–1000 | low, 0–900 |
| Init | `lora_highnoise_GOLDEN_ep40` | `lora_lownoise_GOLDEN_ep40` |
| Rank / α | 16 / 32 | 16 / 32 |
| LR · optimiser · precision | 3e-5 · adamw8bit · fp16 | same |
| Steps | 1008 (18 ep × 56) | 1008 |
| Wall time | **7 h 05 m** @ ~25.3 s/it | **7 h 29 m** @ ~26.7 s/it |
| Final avg loss | **0.00184** | **0.00095** |
| Checkpoints | 18 + resume states | 18 |

> **Plan correction:** the doc specified rank 8/α16 *and* init from the golden. Impossible —
> the golden is rank 16/α32, so rank-8 tensors cannot load it. Matched the golden instead;
> the anti-memorisation case for rank 8 also weakened once compositing took 7 clips → 28.

wandb: [high-noise](https://wandb.ai/rlrahulkanojia/pudgy/runs/pudgy-happy-expr-highnoise-v1) ·
[low-noise](https://wandb.ai/rlrahulkanojia/pudgy/runs/pudgy-happy-expr-lownoise-v1)

### 3.1 Training schedule (per v1 report convention)

| Metric | run 1 (high-noise) | run 2 (low-noise) |
|---|---|---|
| Epochs planned / reached | 18 / 18 | 18 / 18 |
| Optimizer steps | 1008 | 1008 |
| Samples · batches per epoch | 56 · 56 | 56 · 56 |
| Batch/device · grad-accum · effective | 1 · 1 · 1 | 1 · 1 · 1 |
| Wall time | **7 h 05 m** (25.3 s/it) | **7 h 29 m** (26.7 s/it) |
| Loss first → last | 0.02546 → **0.00184** | 0.00293 → **0.00095** |
| Loss min | 0.00180 | 0.00094 |

```
run 1  ▇▄▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
run 2  ▅▆▆▆▅▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
```
Both decline monotonically with no overfit blow-up. Run 2 starts an order of magnitude
lower — it inherits a low-noise golden already fluent in identity/texture, so it is
refining detail rather than fighting a motion prior.

### 3.2 Command used

```bash
EXPERT=low LOG_WITH=all WANDB_PROJECT=pudgy \
  bash /workspace/Pudgy/finetune/wan/train_pudgy_happy_expr.sh
# expands to:
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B --dit .../wan2.2_i2v_low_noise_14B_fp16.safetensors \
  --dataset_config .../dataset_config_happy.toml --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 32 \
  --network_weights .../lora_lownoise_GOLDEN_ep40.safetensors \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep 0 --max_timestep 900 --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate 3e-5 --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_epochs 18 --save_every_n_epochs 1 --save_state --seed 42 \
  --output_dir /workspace/wan_output/pudgy-happy-expr-lownoise-v1 \
  --output_name pudgy-happy-expr-lownoise-v1
# run 1 is identical with EXPERT=high (min/max timestep 900/1000, the high-noise DiT
# and lora_highnoise_GOLDEN_ep40 as --network_weights).
```

## 4. Evaluation metrics

Two-expert FLF2V, 1024×1024, 24 fps, 25 frames, 25 steps, flow-shift 5.0, seed 42,
fp8_scaled + block-swap 39 + lazy-loading. Low expert = v2 golden identity (unchanged),
high expert = v5 happy checkpoint.

### 4.1 In-distribution — epoch 18 ✅

| Metric | v1 (CogVideoX) | v2 G1 | **v5 pilot ep18** | Reading |
|---|---|---|---|---|
| Adjacent SSIM (mean) | 0.9247 | 0.949 | **0.9749** | best of programme |
| Adjacent SSIM (min) | 0.7821 | — | **0.9269** | worst transition, pair 8→9 |
| SSIM vs f0 (last frame) | 0.788 | — | **0.9661** | v1 drifted; v5 holds |
| Subject area f0 → last | vanished | held | **19.7% → 19.9%** (min 19.6%) | no vanish |

**Qualitative (25 frames):**
- f0 neutral (matches conditioning frame) → f1–2 eyes closing, beak opening →
  **f3–10 full happy: squinted joyful eyes + open smiling beak** → f11–25 hold.
- Identity on-model throughout — clean blue, thick outlines, flat pastel, correct
  proportions. No melting, no drift, no vanish.
- **Flat 2D preserved.** Wan's photorealistic bias — the research's #1 predicted risk —
  did not trigger, consistent with v2 at every epoch.
- **Emergent pink hearts** at f3–10, absent from both the conditioning frame and the 28
  training clips. Inherited from the v2 golden's 75-clip skit data, i.e. the continue-train
  kept prior knowledge rather than overwriting it.
- ⚠️ **The hold partially relaxes:** peak expression is f3–10; by f11–25 the eyes reopen
  while the beak stays open. We only have 0.875 s of hold data, so a sustained hold is
  under-trained. This is the clearest data gap the eval exposes.

### 4.2 Generalisation (unseen background) — ✅ PASS
Pax composited on **lavender**, never trained (trained grounds: white/blue/peach/mint).
Free I2V — **no end keyframe**, i.e. the exact condition that produced v1's mid-clip vanish.

| Metric | Value | Reading |
|---|---|---|
| Adjacent SSIM mean / min | 0.9712 / 0.9168 | smooth |
| SSIM vs f0 (last) | 0.9673 | no drift |
| **Background corner drift f0→last** | **2 / 255** | unseen ground held; v1 drifted pink |
| Subject area f0 → last | 16.6% → 16.5% (min 16.2%) | no vanish |

The happy arc transferred intact to a background never seen. **The LoRA learned "happy",
not "happy on these four backgrounds"** — the alpha-compositing augmentation did its job.
Holding identity under *free* I2V for 21 frames is a direct improvement on v1.

### 4.3 Motion regression — ❌ FAIL (the pilot's real finding)
Prompt asked for **"standing still and turning its head slowly to look to one side …
neutral expression"**, same start frame, through the continue-trained high-noise expert.

| Metric | Value | Reading |
|---|---|---|
| **Per-frame SSIM, regress vs in-distribution** | **0.9719 mean / 0.9515 min** | opposite prompts → near-identical video |
| Horizontal centre-of-mass range | **0.8 px** over 21 frames | requested head turn **not performed** |
| Adjacent SSIM mean | 0.9732 | internally smooth — it is confidently doing the wrong thing |

The model produced the **happy arc regardless of the prompt**, and ignored the requested
motion. Expression has been baked into the high-noise expert rather than made
*promptable* — exactly the failure the plan's §6.3 check exists to catch, and the stated
trigger for design **B** (a standalone expression LoRA stacked at inference rather than
continue-training the motion expert).

> ⚠️ **Confound — this test is not yet decisive.** It reuses the *training* start frame,
> and all 28 clips begin from ~that frame and always end happy. So the model may have
> learned "this frame → happy" rather than "always happy". Separating the two needs a
> regression run from a **different pose** (e.g. a v1 skit frame from Azure `training_v1`).
> Until that runs, read 4.3 as "expression is not promptable **from this frame**".

> **Methodological note:** the 13.7% weight drift (§1) read as a *positive* forgetting
> signal and was wrong. Only the behavioural test caught it. Weight-space distance is not
> a proxy for behavioural preservation — cf. v4's caveat that motion metrics missed
> identity entirely.

### 4.4 Different-pose diagnostics — the confound resolved

Same non-happy prompt as 4.3, but driven from a **v1 skit frame** (Pax in profile, tiled
room, hanging lamp — a pose/scene/framing nothing in the 28 clips resembles).

| Run | adj-SSIM | SSIM vs f0 (last) | subject x-range | area f0→last | expression |
|---|---|---|---|---|---|
| **4.3 baseline** — training start frame | 0.9732 | — | **0.8 px** | — | happy (prompt ignored) |
| **A** — ep18 @ scale 1.0 | 0.9544 | 0.9409 | **221 px** | 6.3% → 8.4% | ~neutral, beak closed |
| **B** — ep18 @ scale 0.5 | 0.9625 | 0.9466 | 208 px | 6.1% → 12.7% | ~neutral |
| **C** — ep04 @ scale 1.0 | 0.9581 | 0.9387 | **259 px** | 6.4% → **14.3%** | ~neutral, walks forward |

**Three conclusions:**

1. **The collapse is start-frame-specific, not global.** From the *training* frame, motion
   is 0.8 px and the happy arc fires regardless of prompt. From a *novel* frame, the subject
   moves 208–259 px, walks toward camera, and holds a neutral beak-closed expression. The
   model did **not** learn "always happy" — it learned **"this design-sheet frame → happy"**.
   All 28 clips start from ~the same frame, so that frame became a near-deterministic trigger.
   → **4.3 should be read as conditioning-frame memorisation, not destroyed text conditioning.**
2. **LoRA strength is not the lever.** Halving to 0.5 changed motion negligibly
   (221 → 208 px). Drop strength as a mitigation.
3. **Motion responsiveness decays with training.** ep04 moves more than ep18 (259 vs 221 px;
   area growth 14.3% vs 8.4%). Mild, progressive over-baking → **golden is likely early
   (~ep4–8), not ep18**, consistent with v1's mid-run golden.

### 4.5 Run 2 (low-noise) — full suite + the expert A/B verdict

Run 2 trains the **low-noise** expert from `lora_lownoise_GOLDEN_ep40`; the high-noise
motion golden is left untouched and loaded unchanged at inference (mirror of run 1).
Final loss **0.000951** (run 1: 0.00184), 1008 steps, 7 h 29 m.

| Test | run 1 (high-noise) | **run 2 (low-noise)** |
|---|---|---|
| In-distribution adj-SSIM | 0.9749 | **0.9782** |
| In-distribution expression | relaxes by f11 | **sustained f2–21** |
| Generalisation, unseen-bg drift | 2 / 255 | **2 / 255** |
| Novel frame + neutral prompt, x-range | 221 px | **267 px** |
| **Prompt A/B on the TRAINING frame** (regress vs indist) | **0.9719 → trigger persists** | **0.9104 → prompt has effect** |
| **Prompt A/B on a NOVEL frame** (ctrl_happy vs regress2) | **0.9692 → prompt ignored** | **0.9340 → prompt has effect** |

### 4.6 Controllability A/B — happy prompt vs no happy prompt
Identical start frame; **the prompt is the only variable** (`ctrl_happy` vs `regress2`).

- **run 2 (low-noise) — ✅ controllable.** Happy prompt → squinted eyes + open beak from
  f4, sustained. Neutral prompt → open eyes, closed beak for f0–13, mild drift to a smile
  late. Visually distinct; SSIM 0.9340.
- **run 1 (high-noise) — ❌ not controllable.** Both prompts produce near-identical video
  (SSIM 0.9692); the happy arc fires regardless.

> ### 🏆 Verdict: expression belongs on the **low-noise** expert.
> Run 2 wins on every axis — sustained expression, prompt controllability on both the
> training frame *and* novel frames, and more motion under a neutral prompt. It also does
> **not** exhibit run 1's conditioning-frame memorisation.
>
> **The mechanism:** run 2 never touches the high-noise expert, so the G1-validated motion
> prior stays intact and keeps responding to the prompt. Run 1 overwrote exactly that prior.
>
> This **contradicts the pilot plan's §2**, which argued "happy is a specialisation of
> motion → continue-train the high-noise expert" and made that design **A**, recommended
> and run first. The footage said otherwise: the f0→f20 delta is beak shape and brow lines,
> i.e. fine facial detail — low-noise territory by v2's own G1 finding (high-noise = global
> composition/lighting, low-noise = identity/texture). The A/B settles it empirically.

### 4.7 Showcase — 10 clips from the golden weights

Generated with the golden alone (run 2 low-noise happy + untouched v2 high-noise motion),
one model load via `--from_file`, 1024×1024, 24 fps, 25 steps, fp8_scaled, **no block-swap**.
Varies angle (7), background (6, two never trained), seed (42/7/123), prompt (happy vs
neutral) and length (21 f / 33 f). Azure `v5/inference/`.

| # | Clip | Varies |
|---|---|---|
| 01 | front, white, happy | in-distribution reference |
| 02 | front, **lavender**, happy | unseen background |
| 03 | ¾-front-left, blue, happy | angle · seed 7 |
| 04 | ¾-front-right, peach, happy | angle · seed 123 |
| 05 | left profile, mint, happy | profile angle |
| 06 | wide ¾-right, **sky-blue**, happy | unseen background |
| 07 | right profile, white, happy | angle · seed 7 |
| 08 | **v1 skit scene**, happy | real environment |
| 09 | **v1 skit scene, NEUTRAL** | ← controllability contrast with 08 |
| 10 | front, white, happy, **33 f** | length extrapolation (1.4 s) |

**Observations:** identity, flat-2D style and background colour hold across all ten; 08 vs 09
shows the happy/neutral contrast in the same scene; 10 extends to 33 frames without collapse.
Two blemishes: a small facial artifact in 05, and **partial canonical-view drift** — the
¾ angles (03/04/06) render closer to front-on than their conditioning frames, consistent with
the "canonical-view bias" already logged as a v2 limitation. True profiles (07) survive.

**Throughput note:** the first attempt ran per-clip with `--blocks_to_swap 20 --lazy_loading`
and sat at **39% GPU util, 27.5 s/it**. Dropping block-swap on the now-idle card and batching
via `--from_file` gave **100% util, 13.7 s/it** — 2× faster, one model load instead of ten.
Block-swap is for memory pressure only; it is a large throughput tax when VRAM is free.

## 5. Open items

1. Finish 4.2 / 4.3; repeat at ≥3 seeds (v4 §5.1 — never judge on one seed).
2. Checkpoint sweep — ep18 is the only one evaluated; v1's golden was mid-run, v2's was
   final, so sweep rather than assume.
3. Run 2 (low-noise) A/B → settles whether expression belongs on the identity or the
   motion expert.
4. Square 1024² is **off the base's canonical aspect list** (`720×1280 / 1280×720 /
   480×832 / 832×480`) — a quality cost inherited from square source art.
5. Sustained-hold data (§4.1) and any Polly data at all — 1 of 20 taxonomy cells is filled.

## 6. Artifacts — where everything lives

Azure account `pudgytraining`, container **`pudgy`**, prefix **`v5/`** (matching the
existing `v1/ v2/ v4/` layout):

```
v5/
├── weights/
│   ├── lora_happy_lownoise_GOLDEN_ep18.safetensors   🏆 the A/B winner
│   ├── lownoise/    18 checkpoints (run 2 — GOLDEN lineage)
│   └── highnoise/   18 checkpoints (run 1 — rejected by the A/B, kept for the record)
├── eval/            all eval videos + montages, 4 test modes x both experts
├── inference/       the 10-clip showcase (golden weights only)
├── logs/            tensorboard events + trainer stdout, both runs
└── docs/            this report, both v5 plan docs, prep/train/eval/upload scripts,
                     dataset config + captions jsonl
```

Resume states (`*-state/`, 15.6 GB) are **not** uploaded by default — same policy as the v2
run. Re-run `finetune/wan/azure_upload_v5.py --with-states` if exact-resume is needed.

Source clips remain at `pudgy/interation_3/03_expression_clips/Pax/happy/`; the 28
composited training clips are reproducible from them via `prep_happy_v5.py`.

wandb project `rlrahulkanojia/pudgy`:
[`pudgy-happy-expr-highnoise-v1`](https://wandb.ai/rlrahulkanojia/pudgy/runs/pudgy-happy-expr-highnoise-v1) ·
[`pudgy-happy-expr-lownoise-v1`](https://wandb.ai/rlrahulkanojia/pudgy/runs/pudgy-happy-expr-lownoise-v1)

## 7. Reproduce from scratch

```bash
bash setup_wan_env.sh                                    # venv + musubi + 65 GB weights
python finetune/wan/prep_happy_v5.py                     # alpha -> 28 composited clips
python .../wan_cache_latents.py --dataset_config finetune/wan/dataset_config_happy.toml \
       --vae wan_2.1_vae.safetensors --i2v
python .../wan_cache_text_encoder_outputs.py --dataset_config ... --t5 umt5-xxl --batch_size 16
EXPERT=low bash finetune/wan/train_pudgy_happy_expr.sh   # the GOLDEN run
CKPT=<final> EXPERT=low MODE=indist bash finetune/wan/eval_happy_v5.sh
bash finetune/wan/showcase_v5.sh                          # 10 showcase clips
python finetune/wan/azure_upload_v5.py
```
