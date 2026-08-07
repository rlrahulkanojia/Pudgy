# Pudgy Penguins — Training Approach v5: Single-Expression ("Happy") LoRA on Wan2.2-A14B

**Extends** [Training_Approach_v2.md](../v2/Training_Approach_v2.md) (the Wan2.2-I2V-A14B decoupled identity/motion line — **Gate G1: PASS**). This plan is scoped narrowly to the *Wan2.2* line specifically, per the client's request — it does **not** use v3 (AniSora, plan-only, never executed) or v4 (LTX-2, a separate architecture track with its own dataset). It picks up from the **golden checkpoints** v2 already produced.

**Goal:** teach the model one controllable expression — **Pax, happy** — as a third, independent axis alongside the identity and motion decomposition v2 already validated. Output must be drivable two ways: (1) I2V, from a starting Pax keyframe/image, and (2) text-prompted, without a client-supplied starting image. Budget/scale: deliberately small — this is a pilot on the 7 clips we have today, not a final-quality run.

---

## 0. What we're building on (consolidated from the repo + Azure)

| | |
|---|---|
| Base | Wan2.2-**I2V-A14B**, MoE with two experts, trained via **musubi-tuner 0.3.4** |
| Existing identity LoRA | low-noise expert, timesteps 0–900, rank 16/α32, all-linear incl. MLP, lr 5e-5, 40 epochs/3000 steps — **golden = epoch 40 final** |
| Existing motion LoRA | high-noise expert, timesteps 900–1000, same rank/targeting, lr 1e-4, 40 epochs — **golden = epoch 40 final** |
| Result | **Gate G1: PASS.** Held-out showcase: temporal SSIM 0.949, structural stability 0.880 (no mid-clip vanish, the v1/CogVideoX failure), source fidelity SSIM 0.905. Known limits: mild background drift on some scenes; "canonical-view bias" (LoRA pulls off-model poses back toward its trained canonical angle) |
| Training data used | the original 75-clip set, 768×1360 portrait, 33 frames (4·8+1) @16fps, caption = fixed identity **anchor** + VLM motion **suffix**, anti-entanglement pruned (`prep/caption_prune.py`) |
| Where the golden weights live | Azure `pudgytraining` / container `v2-decoupled-identity-motion`: `output/lora_lownoise_GOLDEN_ep40.safetensors`, `output/lora_highnoise_GOLDEN_ep40.safetensors` (+ all 20 intermediate checkpoints per expert under `weights/` and `weights_highnoise/`, `--save_state` dirs for exact resume, full logs, and eval reports/montages) |
| Logging | **wandb is already live**, not aspirational — project `pudgy`, account `rlrahulkanojia`, real run IDs from the v2 run. Wired via `--log_with all --log_tracker_name pudgy --wandb_run_name <name> --log_config` in `finetune/wan/train_pudgy_wan_a14b.sh` |
| Precedent for single-expression LoRA | **none.** Every executed run so far trained on the full mixed 75-clip set. The low/high-noise split, anti-entanglement captioning, and "LoRA not full-FT, pick golden by eye" philosophy carry over; the recipe below is new |

**New data for this plan:** `pudgy/interation_3/03_expression_clips/Pax/happy/` (Azure `pudgy` container) — 7 clips, uploaded this session.

---

## 1. Phase 0 — Diagnostics on the actual happy footage (done, informs everything below)

Per v2's own principle ("verify before you train"), I inspected the 7 clips directly rather than assume they match the earlier dataset shape. They don't, in a way that changes the plan:

| Property | Existing 75-clip dataset | New `happy/` clips |
|---|---|---|
| Resolution | 768×1360 portrait | **1080×1080 square** |
| FPS / frames | 16fps / 33 frames (4·8+1) | **24fps / 21 frames** (4·5+1 — still musubi-legal, just a different bucket) |
| Duration | ~2.06s | **~0.875s** |
| Background | real environments (rooms, tables, lamps) | **blank/white, no scene** |
| Content | narrative skit fragments, mixed motion | **one performance, filmed 7× from fixed angles** (front, quarter-front-L/R, quarter-front-2-L/R, side-L, side-R) |

Frame-by-frame check of `FRONT.mov` confirms it's a real transition, not a static pose held for the camera: frame 0 is a small closed-mouth near-neutral smile, frame 20 is a full open-mouth joyful smile with raised brows — i.e. it does follow the brief's "neutral → expression → hold" shape, just compressed into <1s. **This is character-design-sheet material (§1 of the client's data request — "expression sheet, 8 angles, happy"), not the narrative "expression clips" material (§3 — 10 videos/emotion, simple background, more varied performances).** That's not a defect, but it changes what this data can teach a video model:

- **Good:** 7 clean, zero-background, single-subject views of the same clear happy transition — ideal signal-to-noise for isolating "what does Pax's face/body do when happy," and multi-angle coverage the narrative clips wouldn't give.
- **Limited:** it is **one performance shot from 7 correlated angles**, not 7 independent instances of "happy." A LoRA trained only on this risks learning *this specific smile arc from these 7 cameras* rather than a generalizable, promptable "happy" concept. There's also an aspect-ratio/background/duration mismatch against the identity+motion training distribution.

**Gate G0 (informal): proceed, but as an explicit pilot.** Don't promise production quality off 7 clips from one take. In parallel, request the client's Round 3 "expression clips" deliverable for `happy` (10 clips, varied context/background) to retrain/extend once available — see §6.

---

## 2. Architecture decision: where does "expression" live?

v2's thesis is decoupling identity (low-noise) from motion (high-noise). "Happy" is neither pure identity nor generic motion — it's a *specialization* of motion. Two candidate designs:

**A — Continue-train the existing golden high-noise (motion) LoRA on the 7 happy clips (recommended, run first).**
Initialize from `lora_highnoise_GOLDEN_ep40.safetensors` (musubi `--network_weights`, not `--resume` — we want a fresh optimizer/scheduler on new data, not to continue the old training schedule), train only on the happy clips with a low LR, and save frequently. This starts from a model that already knows Pudgy-style bounce/timing and specializes it toward "happy," which is far more sample-efficient than learning motion from scratch on 7 clips. The existing low-noise **identity LoRA is left untouched** — reused as-is at inference.

**B — Fresh, small standalone "expression-happy" LoRA, stacked as a third module (fallback if A causes forgetting).**
Train a low-rank LoRA from scratch on just the happy clips, applied *in addition to* the existing low-noise identity + high-noise motion LoRAs at inference (three LoRAs stacked, independently weighted). Safer against catastrophic forgetting of general motion, but no prior in this repo confirms three-way LoRA stacking is supported/well-behaved in the musubi inference path — needs a quick spike before relying on it (`wan_generate_video.py` today only exposes one `--lora_weight` slot per expert; multi-file-per-expert support needs checking).

Run **A** first — it's cheaper, reuses proven infrastructure (`eval_flf2v_2expert.sh` already supports independently-scaled low/high LoRA weights, so swapping in the new happy high-noise checkpoint is a one-line change), and directly tests whether continue-training forgets general motion. Fall back to **B** only if A's "does it still bounce/act normally on non-happy prompts" check (Phase 4) regresses.

---

## 3. Phase 1 — Dataset & caption prep

1. **Repackage the 7 clips** into a new musubi dataset config (`finetune/wan/dataset_config_happy.toml`), separate from the main `dataset_config.toml` so the existing identity/motion dataset is untouched:
   ```toml
   [general]
   resolution = [1080, 1080]
   batch_size = 1
   enable_bucket = true
   bucket_no_upscale = true

   [[datasets]]
   video_jsonl_file = "/workspace/Pudgy/finetune/wan/dataset_happy.jsonl"
   cache_directory = "/workspace/wan_cache/latents_happy"
   target_frames = [21]
   frame_extraction = "head"
   num_repeats = 6   # 7 clips x 6 = 42 "samples" per epoch — counters the tiny dataset size
   ```
2. **Captions** follow the existing anchor+suffix convention, but with the anti-entanglement blocklist **inverted for expression terms**: `caption_prune.py`'s current blocklist drops "blush"/"cheeks" as identity-appearance noise, which is correct for the identity/motion run but wrong here — those *are* the signal we want. Add an explicit expression-description carve-out (or just hand-write these 7 captions; it's 7 lines) of the shape:
   `"<existing Pax identity anchor>, breaking into a big happy smile — eyebrows lifting, eyes crinkling, mouth opening into a wide joyful grin, on a plain white background, static camera, front-facing."`
   Vary only the camera-angle clause per file (front / quarter-front-left / quarter-front-right / quarter-front-2-left / quarter-front-2-right / side-left / side-right) so the model can learn "happy" is angle-independent rather than baking in one viewpoint.
3. **Precache** with the same VAE/T5 already used (`wan_2.1_vae.safetensors`, UMT5-XXL) into a separate cache dir (`wan_cache_latents_happy/`) so nothing collides with the main run's cache.

---

## 4. Phase 2 — Training config (pilot scale, run A)

New script `finetune/wan/train_pudgy_happy_expr.sh`, adapted directly from `train_pudgy_wan_a14b.sh` — same trainer, same two-expert machinery, deliberately smaller/gentler given 7 source clips:

| Knob | v2 (identity+motion, 75 clips) | v5 pilot (happy, 7 clips) | Why |
|---|---|---|---|
| Expert trained | both | **high-noise only** — low-noise identity LoRA is frozen/reused from golden ep40 | Identity isn't the target; don't touch what already passed G1 |
| Init | from base DiT | **`--network_weights` = `lora_highnoise_GOLDEN_ep40.safetensors`** | Continue-train, don't relearn motion from scratch |
| Rank / α | 16 / 32 | **8 / 16** (half capacity) | Less capacity to memorize 7 clips verbatim |
| LR | 1e-4 | **2e-5–3e-5** (gentle) — A/B both, pick by eval | Large step on tiny data risks wrecking the general motion prior |
| `min_timestep`/`max_timestep` | 900–1000 | same (900–1000) | Same high-noise expert range |
| `num_repeats` (dataset) | 1 | 6 | Counteract 7-clip dataset size within musubi's epoch/bucket loop |
| Epochs | 40 (~3000 steps) | **15–20** (~630–840 steps at repeat=6) | Tiny dataset overfits fast; fewer passes, watch closely |
| `save_every_n_epochs` | 2 | **1** | Need fine-grained golden-pick given fast overfit risk |
| `save_state` | on | on | Exact-resume if we need to extend once more data lands |
| `sample_every_n_epochs` | 2 | **1**, with `sample_prompts.txt` updated to explicit happy prompts | Catch overfitting/collapse visually every epoch, not every other |
| Optimizer / precision / seed | adamw8bit, fp16, seed 42 | unchanged | No reason to change what already works |
| Logging | wandb project `pudgy` | **same project**, new run name `pudgy-happy-expr-highnoise-pilot-v1` | Keep everything comparable in one wandb project |

Command shape (mirrors the existing script's structure exactly):
```bash
EXPERT=high \
NETWORK_WEIGHTS=/workspace/wan_output/pudgy-wan22-a14b-highnoise/pudgy-wan22-a14b-highnoise.safetensors \
DATASET=/workspace/Pudgy/finetune/wan/dataset_config_happy.toml \
LR=3e-5 RANK=8 ALPHA=16 EPOCHS=18 SAVE_EVERY=1 SAMPLE_EVERY=1 \
WANDB_RUN_NAME=pudgy-happy-expr-highnoise-pilot-v1 \
bash train_pudgy_happy_expr.sh
```
(`train_pudgy_wan_a14b.sh` doesn't currently expose `--network_weights`/`RANK`/`ALPHA` as env knobs — the new script needs those three lines added; everything else — accelerate launch, sdpa, gradient checkpointing, `PYTORCH_CUDA_ALLOC_CONF`, the wandb wiring — copies straight over.)

Run **A** (LR 3e-5) and, GPU budget permitting, a second cell at **LR 1e-4 rank 16** (i.e. v2's original settings, just continue-trained instead of from-scratch) as a comparison point — with only 7 clips the cost of a second short run is trivial next to the risk of committing to the wrong LR blind.

---

## 5. Phase 3 — Checkpointing, logging, storage (reuse the existing pipeline, new prefix)

- **wandb:** same project `pudgy`, new run(s) `pudgy-happy-expr-highnoise-pilot-v1` (and `-lr1e4-r16` for the comparison cell). Log loss curve, sample grids (from `sample_every_n_epochs=1`), and the same tracked config as v2 runs so they're directly comparable in the wandb UI.
- **Checkpoints:** every epoch (1–18), `--save_state` on throughout, so any epoch can be resumed exactly or picked as golden.
- **Golden pick:** same v2 discipline — score every epoch's checkpoint by eye on the fixed eval set (§ below), pick by inspection, not by last-step or lowest-loss (v2's own finding: fidelity peaks mid-run before overfitting starts).
- **Azure storage:** reuse `azure_upload.py`'s pattern against the **same** `v2-decoupled-identity-motion` container (keeps this pilot next to the baseline it's built from) under a new prefix so nothing overwrites the existing golden run:
  - `weights_happy_expr/pudgy-happy-expr-highnoise-{epoch}.safetensors`
  - `logs/train_happy_expr.log`
  - `output/happy_expr/` — golden pick, montages, the comparison report (below)
  Concretely: copy `azure_upload.sh`/`azure_upload.py`, change `WEIGHTS_DIR`/`LOGS_DIR`/`OUTPUT_DIR` to the new run's paths and `CONTAINER` stays `v2-decoupled-identity-motion` (or override via `AZURE_CONTAINER` env var, already supported).

---

## 6. Phase 4 — Evaluation (does it actually generalize "happy"?)

Reuse `eval_flf2v_2expert.sh` almost unchanged — it already loads low- and high-noise LoRAs independently (`LOW=... HIGH=...`), so pointing `HIGH` at each new happy-expert checkpoint is a one-line swap. Two distinct tests per checkpoint, because the real risk here is angle-memorization, not just image quality:

1. **In-distribution check:** FLF2V from one of the 7 happy clips' own start/end frames (sanity — should look at least as good as training data).
2. **Generalization check (the one that matters):** drive inference from a **Pax reference frame the happy LoRA never saw** — e.g. a neutral-pose frame pulled from the original 75-clip dataset, or a character-sheet neutral image once `01_character_design/Pax/turnaround_8_angles/` has content — with the prompt asking for "happy," and see whether the expression transfers to a novel pose/angle/background, or whether it only reproduces the training camera setups. This is the direct test of the §1 diagnostic's core risk.
3. **Regression check (only for run A, continue-trained):** run the *unmodified* existing eval prompts (generic motion, non-happy) through the new high-noise checkpoint to confirm it hasn't forgotten the general Pudgy bounce/timing it started from. If it has, that's the trigger to fall back to design **B**.

Score against the same rubric v2 used (temporal SSIM, structural stability, source fidelity) plus a manual "is this recognizably *happy*, on a *new* pose" judgment call — there's no automated expression classifier in this repo, so this stays a by-eye gate for now.

**Text-only generation:** the currently stood-up environment only has the **i2v-A14B** DiT weights (image-conditioned). "Generate from text" should be delivered as I2V driven from a stock/generated neutral Pax keyframe (once the character-design turnaround sheet exists, that becomes the standard starting image), not a separate text-to-video checkpoint — unless we explicitly decide to also provision a t2v-A14B DiT. Flag this as an open dependency to confirm before promising pure text-to-video to the client.

---

## 7. Known limitations going in (be upfront about these)

- **7 clips, one take, 7 correlated angles** — not 7 independent samples. Expect this pilot to teach "what Pax's happy face/body does," reasonably angle-robust, but not to generalize to contexts wildly different from a plain-background head-on portrait shot.
- **Format mismatch** (square/blank-bg/24fps/21-frame vs. the identity+motion training distribution's portrait/environment/16fps/33-frame) — handled via a separate dataset config/bucket, not by forcing a resample, but worth remembering if results look "flatter" than the environment-trained baseline.
- **No precedent** in this repo for single-expression LoRA training — the rank/LR/epoch numbers above are informed starting points (halved rank, an order of magnitude lower LR, ~half the epochs of the full run, scaled for a dataset ~10x smaller), not validated defaults. Treat this as a pilot to be A/B'd, exactly like v2's Phase 0.
- **Data gap:** the client's Round 3 ask (`docs/documents/Client_Data_Request_Round3.md`) wants 10 varied "happy" performance clips with real backgrounds — that data would let a follow-up run generalize much better than 7 character-sheet angles can. Recommend flagging this gap back to the client in parallel with running this pilot, not instead of it.

---

## 8. Immediate next actions

1. Write the 7 hand-authored captions (§3.2) — small enough to not need the VLM captioning pipeline.
2. Adapt `train_pudgy_wan_a14b.sh` → `train_pudgy_happy_expr.sh` (add `--network_weights`, `RANK`/`ALPHA` env knobs).
3. Precache latents/text-encoder outputs for the 7-clip happy dataset.
4. Launch run A (LR 3e-5, rank 8) on the GPU box; watch wandb loss + per-epoch samples live.
5. Once A finishes, run the generalization eval (§6.2) before declaring any epoch "golden."
6. Regardless of pilot outcome, send the Round 3 data-gap note (more "happy" performance clips, varied background) so the next iteration has better raw material.
