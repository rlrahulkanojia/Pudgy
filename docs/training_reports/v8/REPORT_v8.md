# v8 — joint motion + expression LoRA on the combined v7: run report

**Run** 2026-09-26 20:03 → 2026-09-29 05:32 UTC · **Hardware** 1× A100 80GB PCIe (Vast.ai, not
persistent) · **Status** training complete, Tier 1 complete, **Tier 2 complete** incl. the v7fixed /
golden comparisons (§3.4, finished 2026-10-01 04:55 UTC). Updated 2026-09-30 — findings and
next steps in §9, plan in [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md).
Plan: [`Training_Approach_v8.md`](../../../training_approach/v8/Training_Approach_v8.md) ·
plain-language: [`README.md`](../../../training_approach/v8/README.md),
[`NOTES.md`](../../../training_approach/v8/NOTES.md) · v7 analysis that motivated it:
[`ANALYSIS_v7.md`](../v7/ANALYSIS_v7.md).

---

## 1. Headline

| | |
|---|---|
| **Winner** | `pudgy-v8-joint-lownoise-step00000550.safetensors` (rank 32 LoRA, low-noise expert) |
| **Load it as** | `--lora_weight pudgy-v7-fixed-combined.safetensors <winner> --lora_multiplier 1 1`; high-noise: `lora_highnoise_GOLDEN_ep40` × 1 |
| **vs its start (v7fixed)** | loss ↓ **40% (expr) / 43% (motion) on 26/26 held-out clips**; margin ↑ (95% CI +0.002 … +0.026); accuracy 0.63→0.81 expr, 0.60→0.80 motion |
| **vs the v7 specialists** | one joint LoRA beats v7-expression on expressions (loss 0.00301 vs 0.00323) and v7-motion on motion (0.00268 vs 0.00283) |
| **New label** | `blushing` (never seen by base or earlier runs) ranked **1st for both characters** |
| **Where it plateaued** | ≈ step 300–350; steps 350–600 are indistinguishable (score 0.042–0.044) |

**In one line:** building on the correctly combined v7 and training one joint LoRA undid the
v7 interference, then surpassed both v7 specialists, with one new expression added — and
most of it was learned in the first 300 of 600 steps.

---

## 2. What was trained

| | |
|---|---|
| Base | Wan2.2-I2V-A14B low-noise fp16 + **`pudgy-v7-fixed-combined`** merged (`--base_weights`) = v7 expr step 2500 + v7 motion final − v2 low golden, one exact rank-48 LoRA (rel. error 2.5·10⁻⁸) |
| LoRA | fresh, rank 32 / α 32, 400 modules, low-noise expert (t ∈ [0, 0.9]) |
| Data | `processed/v8_joint_2096` — 2,096 clips, 193 sources, 13 labels, QF2_R held out (26 sources) |
| Optimiser | adamw8bit, LR 5e-5, warmup 20, cosine → 10%, grad-accum 8 (effective batch 8), clip 1.0 |
| Steps | 600 optimizer steps = 4,800 clips ≈ 2.3 epochs; saved every 50 (12 checkpoints + states) |
| Memory | blocks_to_swap 20 (16 thrashed at f57: 81.2 GB, 31 allocator retries) |
| Wall-clock | 57.5 h incl. two gate breaks (41 + 50 min); **313–343 s/step ≈ 39–43 s/clip** |
| Logging | W&B `imsaksham-/pudgy` run `244d5e1p` (resumed twice into the same run); tensorboard |

## 3. Results

### 3.1 Tier 1 — diffusion classifier on the 26 held-out QF2_R clips (k = 2, paired noise)

| | expr loss | expr margin | motion loss | motion margin | acc expr / motion |
|---|---|---|---|---|---|
| v2 golden | 0.00564 | −0.0137 | 0.00604 | −0.0128 | 0.25 / 0.10 |
| v7 expr alone | 0.00323 | +0.0250 | 0.00577 | −0.0062 | 0.62 / 0.30 |
| v7 motion alone | 0.00555 | −0.0107 | 0.00283 | −0.0039 | 0.25 / 0.20 |
| v7naive (both files, 2× golden) | 0.04746 | +0.0012 | 0.04306 | −0.0046 | 0.44 / 0.30 |
| **v7fixed = v8 step 0** | 0.00499 | +0.0227 | 0.00467 | −0.0047 | 0.62 / 0.60 |
| v8 step 100 | 0.00334 | +0.0332 | 0.00311 | −0.0039 | 0.81 / 0.30 |
| v8 step 300 | 0.00309 | +0.0361 | 0.00275 | +0.0038 | 0.81 / 0.60 |
| v8 step 350 | 0.00307 | +0.0365 | 0.00278 | +0.0055 | 0.75 / 0.70 |
| v8 step 400 | 0.00304 | +0.0371 | 0.00276 | +0.0059 | 0.81 / 0.80 |
| v8 step 450 | 0.00303 | +0.0384 | 0.00274 | +0.0040 | 0.88 / 0.70 |
| v8 step 500 | 0.00301 | +0.0382 | 0.00271 | +0.0046 | 0.81 / 0.70 |
| **v8 step 550 (winner)** | **0.00301** | **+0.0369** | **0.00268** | **+0.0068** | **0.81 / 0.80** |
| v8 step 600 | 0.00301 | +0.0362 | 0.00269 | +0.0069 | 0.88 / 0.80 |
| v8 SWA 400–600 | 0.00301 | +0.0372 | 0.00270 | +0.0066 | 0.81 / 0.80 |

Chance: 0.125 expression (8 labels), 0.20 motion (5). n = 16 / 10 → accuracy moves in steps
of 0.06 / 0.10; decisions used losses and margins, never accuracy.

**Paired comparisons** (per clip, same noise draws; 95% CI from 5,000 bootstrap resamples):

| | margin change | loss |
|---|---|---|
| step 550 vs v7fixed | **+0.0132 (+0.0019 … +0.0262)**, higher on 73% of clips | lower on **100%** |
| step 550 vs step 300 | +0.0016 (−0.0019 … +0.0054) — not significant | lower on 81% |
| SWA vs step 550 | +0.0001 (−0.0010 … +0.0012) — identical | lower on 54% |

**Winner rule** (fixed before the data, `select_winner_v8.py`): highest expr + motion margin
among checkpoints beating v7fixed on score and both losses; ties (≤ 0.001) → lower loss →
SWA. SWA 0.0438, step 550 0.0437, 600, 400, 500 tied; step 550 had the lowest loss.

**Remaining misses at step 550** are all near-misses (rank 2–4): Pax jumping→walking,
laughing→happy, neutral→angry, waving→jumping; Polly neutral→angry.

**Per action** (held-out loss, mean of 2 clips; change-masked, so faster actions have more to
predict). The classifier recognises the fast actions, but they keep the most residual error:

| action | unique training footage | golden | v7fixed | **v8@550** | vs golden | ranks |
|---|---|---|---|---|---|---|
| sitting | 11.8 s | 0.00257 | 0.00153 | 0.00083 | −68% | 1, 1 |
| waving | 16.7 s | 0.00290 | 0.00193 | 0.00113 | −61% | 4, 1 |
| walking | 7.6 s | 0.00437 | 0.00435 | 0.00258 | −41% | 1, 1 |
| jumping | 22.0 s | 0.00949 | 0.00750 | **0.00436** | −54% | 2, 1 |
| running | 11.3 s | 0.01085 | 0.00806 | **0.00452** | −58% | 1, 1 |

### 3.2 The v7-base decision (thresholds fixed before step 100, plan §1.2)

| Gate | threshold | measured | verdict |
|---|---|---|---|
| step 100 | expr loss < 0.0040, motion < 0.0035, margins ≥ v7fixed | 0.00334 / 0.00311, +0.0332 / −0.0039 | ✅ pass |
| step 300 | restart from golden if loss ≥ 0.0045 / 0.0042 | 0.00309 / 0.00275 | ✅ keep v7 base |

v7fixed alone kept only 27% (expr) / 43% (motion) of each v7 specialist's loss reduction —
the two LoRAs interfere (ANALYSIS_v7 §1). v8 recovered all of it by step 300 and then went
past the specialists.

### 3.3 Tier 0 — weight trajectory (12 checkpoints)

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ‖τ‖ | 1.73 | 2.42 | 2.89 | 3.34 | 3.81 | 4.38 | 4.95 | 5.42 | 5.76 | 6.01 | 6.18 | 6.31 |
| cos(consecutive 50-step updates) | | 0.30 | 0.27 | 0.29 | 0.40 | 0.53 | 0.55 | 0.55 | 0.63 | 0.62 | 0.59 | |

- **Updates became far more consistent than v7's:** consecutive-update cosine rose from
  0.30 to 0.55–0.63 in the low-LR half. v7 (batch 1, constant LR) fell from 0.24 to 0.10.
  Accumulation + cosine decay did what they were for.
- ‖τ‖ growth flattens after step 400 as the LR anneals: τ(600) is 6% of the v7 base's norm.
- ⚠️ **The drift-share fit ‖τ‖² = D²t² + σ²t assumes a constant LR** and is invalid under a
  cosine schedule. Mid-run reports (21% at step 300; "12% at 600, bar not met") came from
  that fit and should be ignored; the consecutive-update cosine is the schedule-robust
  measure. For v7 (constant LR) the fit is valid.

### 3.4 Tier 2 — generated suites on the winner (CLIP / DINOv2 scoring, `score_v8.py`)

| suite | n | result |
|---|---|---|
| `core` — controllability, FRONT + QF2_R start frames | 52 | CLIP 1-NN label accuracy: expr **0.44** Pax / **0.50** Polly (chance 0.125); motion **0.40** / **0.60** (chance 0.20). Zero-shot by view: expr 0.50 / 0.38 on both views; motion FRONT 0.6 / 1.0 → QF2_R 0.4 / 0.6 |
| `compose` — motion × expression in one prompt | 18 | expression reads **0.94**, motion **0.44**; motion energy 12.9 = **78%** of motion-only clips |
| `gb` — unseen grounds (lavender, sky blue) | 12 | label 1-NN 1.0 / 0.67 (chance 0.33); corner drift **1.1–1.7 / 255** |
| `gd` — duration: walking/waving/jumping/sitting f49, neutral/laughing f57 | 12 | periodicity **−0.27 … 0.23** — no clip repeats its cycle |
| `guide` — label-contrastive negative prompt | 16 | 1-NN 1.0, zero-shot 0.25 (chance 0.25) |

**Against the v8 ship bars** (plan §5; n per cell 2–4, read ±0.1):

| bar | | |
|---|---|---|
| controllability 1-NN ≥ 0.57 (v7's range) | ❌ except Polly motion 0.60 | the v7 range came from a different clip set; like-for-like, v8 beats v7fixed and golden (below) |
| holdout view ≤ 0.15 below FRONT | expr ✅, motion ❌ (−0.2 / −0.4) | |
| composition: motion energy ≥ 80% | ❌ 78% | borderline |
| unseen grounds: drift ≤ 5/255 | ✅ | |
| duration: periodicity ≥ 0.5 | ❌ | long clips play one slow arc |
| Pax/Polly parity within 10% | expr ✅, motion ❌ | |

**Comparison** (same 52 `core` / 18 `compose` prompts, seeds and start frames; CLIP 1-NN
label accuracy, the primary metric):

| | golden | v7fixed | **v8@550** |
|---|---|---|---|
| Pax expression | 0.44 | 0.19 | **0.44** |
| Polly expression | 0.44 | 0.38 | **0.50** |
| Pax motion | 0.20 | 0.30 | **0.40** |
| Polly motion | 0.50 | 0.50 | **0.60** |
| `compose` motion / expression (zero-shot) | — | 0.39 / 0.89 | **0.44 / 0.94** |
| `compose` motion energy | — | 12.6 | 12.9 |

v8 is **best or tied in every cell**, so the bar misses above are **inherited, not a v8
regression**: golden and v7fixed miss the same bars, by more. The steps are small (n = 10
per motion cell → ±0.1–0.2; zero-shot is mixed, e.g. Pax motion 0.7 v7fixed vs 0.5 v8), and
golden ties v8 on Pax expression — the generated-clip probe separates the configs far less
than Tier 1 does. Motion is the weakest axis for **every** model trained on this data.
Clips: `pudgy/v8/eval/suites/{core,compose}/{golden,v7fixed,v8@550}/`.

**Reading.** Tier 1 improved decisively on every label; the Tier-2 misses are concentrated on
**motion** — controllability, holdout view, duration and composition. Visually (`compose`):
Polly *jumping + laughing* — full hop with the laugh building; Pax *walking + angry* — scowl
held, gait subtle; Pax *waving + crying* — the cry takes over and he covers his face.
**The model can play motion and expression together, but the expression wins.**
Clips: `pudgy/v8/eval/suites/<suite>/v8@550/`.

## 4. Defects found and fixed during v8

| # | Defect | Impact if missed | Fix |
|---|---|---|---|
| 1 | **21 client files short of real frames; ffmpeg pads gaps with duplicates** (18 sitting, 4 confused, 1 walking, 1 blushing) | frozen stutters mid-motion — v7 trained on them | rungs end before the first missing frame (`real_frame_gap`, frame timestamps); 5 sources dropped |
| 2 | `HAPPY/Pax` skipped as "unknown character" (case-sensitive) | 7 re-uploaded files ignored | case-insensitive lookup; md5 de-dup decides |
| 3 | "pink blush" in Pax captions | colour-anchor bleed (v4 rendered Polly blue) | "rosy"; asserted clean outside the ground clause |
| 4 | Loop-seam test on absolute MAE rejected 12/14 walks | no walking loops | seam ÷ median frame step (walking 0.6–1.0, running 2.3–2.7) |
| 5 | `blocks_to_swap 16` at rank 32 | thrash at 81 GB, slower | 20 (0 retries, faster) |
| 6 | **musubi v0.3.4 `--resume` restarts the step counter at 0** | step-100 resume would overwrite `step00000050`, run 700 steps, and push the cosine past its end | `patches/musubi-v0.3.4-resume-step.patch`; `gate_break_v8.sh` verifies every resume (worked at 100 and 300) |
| 7 | `.env` connection string unquoted | bash `source` truncated it at `;` | quoted |
| 8 | Monitor regex vs supervisor log (tqdm glyphs stripped, ANSI colours) | `step ?/?`, no W&B URL | patterns fixed |
| 9 | Diffusion classifier: frame 0 is the conditioning image; batching OOMs (per-token fp32 modulation) | near-zero signal; OOM | frame 0 excluded, change-mask weighting, batch 1 |
| 10 | Stacking the two v7 files naively | 2× golden: **10× the held-out denoising loss** (v7naive) | `lora_tools_v8.py combine` → one exact file |

Operational mistakes, recorded so they don't recur: a `pkill -f` pattern killed its own
shell twice (use PIDs); a W&B cleanup filtered by name regex and **deleted a user run,
`v7-preflight-smoketest` (2026-09-05)** — delete by exact run id only; the pre-run speed
estimate (≤ 29 s/clip) was wrong — block-swap costs every clip, not just long ones (39–43).

## 5. Where everything is (Azure `pudgytraining`, container `pudgy`)

| Path | What |
|---|---|
| `v8/weights/pudgy-v8-joint-lownoise/` | 12 step checkpoints (50…600), final (= step 600, metadata only differs), `swa400-600` (rank 64) |
| **`v8/base/pudgy-v7-fixed-combined.safetensors`** | **the v8 base — required to use any v8 checkpoint** |
| `v8/state/pudgy-v8-joint-lownoise/<step>-state/` | resume states for all 12 steps + final |
| `v8/logs/pudgy-v8-joint-lownoise/` | tensorboard (+ `diag/`), trainer log segments (to 100, to 300, to 600), monitor logs, `status.json`, `trajectory.json` |
| `v8/logs/box/` | prep / cache / setup / smoke-test logs |
| `v8/eval/dcls/` | Tier-1 JSON for every config above (per-clip rows) |
| `v8/eval/holdout/`, `v8/eval/keyframes/` | classifier clip set + index; start frames (incl. holdout view, unseen grounds) |
| `v8/eval/suites/` | generated clips + `meta.json`; `winner.json`; gate / final / baseline logs |
| `v8/eval/interaction/`, `v8/eval/zoomfix/` | two-character tests; zoom-out experiments (§8) |
| `v8/eval/holdout/cache/` | the classifier set's latent + text cache (saves rebuilding it) |
| `v8/longform/final/` | the two long-form scenes, with and without soundtrack |
| `v8/longform/workdir/` | the long-form working folder: keyframes, every take, takes JSONs, QA sheets, logs, scripts (as run) |
| `v8/cache/latents_v8/` | the 20 GB training latent + text cache for `v8_joint_2096` (rebuild: 2.5 h) |
| `processed/v8_joint_2096/` | the dataset (MD5-verified) |
| `raw/iteration_1` … `_5` | every client delivery (iteration_2's 72 skits live only here) |
| `v2/weights/curated/`, `v7/weights/` | the inputs the v8 base was built from |

Code: `main` of this repo (PRs #8–#17). Weights verified byte-identical by downloading every
file back and comparing MD5 (`archive_v8.py --verify`, last run 2026-10-01: 570/570 files,
15/15 weights).

**Box decommission check (2026-10-01).** Every local file was compared with Azure by name
and size before deletion: raw (602 files), processed (2,119), v7/v2 inputs, all eval outputs,
the long-form folder and the latent cache are on Azure. Deliberately *not* uploaded: the Wan
base models and CLIP/DINOv2 (public, re-downloaded by `setup_wan_env.sh`), the venvs, and
secrets (`.env`).

## 6. Still running / not done

1. v7-harness gates still open: G-F (training-frame trigger), G-M (novel frame), G-H (hold),
   G-N (idle — expected to fail, no idle footage), G-R (blocked: v2 showcase prompts missing).
2. Data: **21 files** still need re-exporting — the 20 from the earlier list came back in
   iteration_5 byte-for-byte identical, plus `PAX_EXPRESSION_BLUSHING_QF2_L` (41 frames, 33
   real). Requested again in [`Client_Data_Request_Round5.md`](../../documents/Client_Data_Request_Round5.md).

## 7. Rebuild on a new box

```bash
git clone https://github.com/rlrahulkanojia/Pudgy /workspace/Pudgy && cd /workspace/Pudgy
cp <your .env> .env                         # Azure + W&B keys (not in git)
bash setup_wan_env.sh                       # musubi v0.3.4 + resume patch, venv, 65 GB weights
# weights + base + goldens
az storage blob download-batch --account-name pudgytraining -s pudgy -d /workspace/restore --pattern "v8/weights/*"
az storage blob download-batch --account-name pudgytraining -s pudgy -d /workspace/restore --pattern "v8/base/*"
az storage blob download-batch --account-name pudgytraining -s pudgy -d /workspace/restore --pattern "v2/weights/curated/*GOLDEN_ep40*"
# flatten into the paths eval_v8/common.py expects:
#   /workspace/wan_output/pudgy-v8-joint-lownoise/, /workspace/wan_output/v7_baseline/pudgy-v7-fixed-combined.safetensors,
#   /workspace/wan_output/v2_golden/lora_{low,high}noise_GOLDEN_ep40.safetensors
# to TRAIN further: also processed/v8_joint_2096 (-> /workspace/data_v8), and either v8/cache/latents_v8
#   (-> /workspace/wan_cache/latents_v8, skips the 2.5 h rebuild) or finetune/wan/services/cache_v8.sh,
#   and v8/state/... for --resume; supervisor units in finetune/wan/services/
```

---

## 8. Addendum (2026-09-29 → 30): interaction, zoom-out, long-form

**Interaction (untrained).** Four two-character tests with v8@550 (waving together; hug;
waving → hug; waving from two separately placed frames): no identity blending in any, and the
action change to a different interaction works. This is Wan's base two-character ability kept
on-model, not learned interaction. `pudgy/v8/eval/interaction/`.

**Zoom-out in movement clips.** Measured (character head line + ground-shadow width per
frame): expressions hold size (±2%); running / walking / jumping / sitting shrink 8–28% in
shadow width with feet planted — reads as a camera pull-back. Training clips don't do it, and
v7 did too. Cause: Wan's default negative prompt contains "static / still / motionless frame"
while every caption says "static … shot", so guidance pushes against a still camera. Three
inference-side rounds (`eval_v8/experiments/run_zoomfix*.py`):

| negative | zoom (shadow) | motion | issue |
|---|---|---|---|
| default | −11.4% | 5.96 | zoom-out |
| static/still removed | −1.1% | −30% | sitting freezes |
| default + anti-zoom-out | −0.5% | −19% | sitting freezes; running approaches camera |
| default + camera-only, both directions | −5.5% | −12% | inconsistent per seed |
| guidance 3.5 | −11.1% | same | no effect — `--guidance_scale` is low-noise only; high-noise already 3.5 |

No inference-only fix is clean. **v9 fix:** replace "static … shot" in captions (e.g. "fixed
camera, locked-off shot, the character stays the same size") and add standing→seated clips.

**Long-form scenes.** LifeWithHer (14.5 s, 5 beats, plain I2V) and EatingStages (13.9 s,
5 beats; 4 anchored on start + end reference frames, colour-graded) at 1080×1920 / 24 fps.
Pipeline and QA: `finetune/wan/longform/README.md`. Every take passed drift / pop /
duplicate-character checks; EatingStages backgrounds are flat (grain 0.00) with a 0.0 palette
step at every cut. Findings: single-character beats are clean from one frame; two characters
plus props on a shared table lose the layout ~0.8 s in (pops, morphs) unless the end frame is
anchored. `pudgy/v8/longform/`.

---

## 9. Findings and next steps (2026-09-30)

**Findings**

1. **The faster the motion, the more data it needs.** Fast actions (running, jumping) are the
   least accurate: highest residual held-out loss of any motion (§3.1, per action), and in
   generated clips motion is where every Tier-2 bar fails (§3.4). The training data explains
   it: 7.6–22 s of unique footage per action, each **one performance from nine angles**,
   performed on the spot, longest running clip 0.71 s (< 2 strides) — and the VAE's 4×
   temporal compression leaves a stride about 2 latent steps.
2. **The model can combine animations in one clip** — motion and expression together
   (`compose`), with the expression reading 94% and the motion 44%: co-existence works, the
   balance favours the face.

**Next steps** → [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md)

1. **Longer videos** — i. with props, ii. with different backgrounds, iii. with captions for
   long-form: mine the 101 client skits (iteration_1 + iteration_2) into shots with scene
   captions; composite the alpha clips onto set backgrounds (v9b).
2. **More samples of fast animation** — Round 5 data request (tiered by motion complexity,
   ≥ 5 takes and ≈ 90 s per fast action) plus travel/speed augmentation, a high-noise LoRA,
   and a data-scaling experiment (v9a).
3. **Prop images** (bus, plate, pillow, …) trained incrementally with the videos they appear
   in, with replay and a regression gate per stage (v9c).

