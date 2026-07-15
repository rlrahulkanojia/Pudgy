# Wan2.2-I2V-A14B LoRA stand-up (Training Approach v2)

Parallel training env for the v2 base migration — **separate from** the CogVideoX
setup (`setup_gpu_env.sh`, `../train_cogvideox_*`). Built for an **A100 80 GB**,
which lifts the v1 blocker (A14B was tight/fp8-only on the old 40 GB card).

## What's installed / where

| Thing | Location |
|---|---|
| Trainer | `/workspace/musubi-tuner` (kohya musubi-tuner 0.3.4) |
| Venv | `/workspace/Pudgy/.venv-wan` (torch 2.11.0+cu128, py3.12) |
| DiT low-noise (identity) fp16 | `/workspace/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors` |
| DiT high-noise (motion) fp16 | `…/wan2.2_i2v_high_noise_14B_fp16.safetensors` |
| Text encoder (UMT5-XXL) | `/workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth` |
| VAE (Wan2.1 **8×** — NOT the 5B 16×) | `/workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors` |
| Dataset (JSONL + TOML) | `dataset.jsonl`, `dataset_config.toml` (this dir) |
| Pre-cached latents + text embeds | `/workspace/wan_cache/latents/` |
| Training script | `train_pudgy_wan_a14b.sh` (this dir) |
| Outputs | `/workspace/wan_output/pudgy-wan22-a14b-{low,high}noise/` |

> ⚠️ **Disk is NOT a persistent volume** (`workspace_is_volume: false`). Weights,
> venv, and caches are wiped on recycle/destroy. Sync anything you keep off-box.

## Recipe (why these settings)

Straight from Training_Approach_v2 §4 + musubi `docs/wan.md`:

- **Base:** Wan2.2-I2V-A14B, **8× VAE** (v1 Phase-0.1 proved 8× is near-lossless
  for the flat-outline style; the 5B's 16× VAE is the one to avoid).
- **Two LoRAs** on the MoE experts, trained **separately** with the I2V
  timestep split + `--preserve_distribution_shape`:
  | Expert | Role | Timesteps | LR |
  |---|---|---|---|
  | low-noise | identity / texture | 0–900 | 5e-5 |
  | high-noise | motion / composition | 900–1000 | 1e-4 |
- **all-linear incl. MLP:** `networks.lora_wan` replaces `WanAttentionBlock`,
  whose Linear layers include self-attn, cross-attn **and the FFN** — i.e. the
  v2 "all-linear + MLP" target is the musubi default.
- **rank 16 / alpha 32** (α = 2r). **flow shift 5.0** (I2V). **fp16** training
  (matches the fp16 DiT; musubi's dtype matrix disallows bf16-training on fp16
  weights). Portrait **768×1360 · 33 frames · 16 fps** (dataset native).

## Run it

```bash
# both experts (each ~separate run; only one 14B DiT resident at a time)
EXPERT=low  bash /workspace/Pudgy/finetune/wan/train_pudgy_wan_a14b.sh
EXPERT=high bash /workspace/Pudgy/finetune/wan/train_pudgy_wan_a14b.sh
# tunables: EPOCHS=40 SAVE_EVERY=2 SAMPLE_EVERY=2 LR=...  (use tmux; multi-hour)
```
Checkpoints land every `SAVE_EVERY` epochs — pick the **golden** one by eye
(v2 §4: fidelity peaks mid-run before overfitting).

### Azure Blob backup (`azure_upload.sh`)

Uploads to storage account **pudgytraining**, container **`v2-decoupled-identity-motion`**
(the "folder" — Azure containers can't have underscores; 3-word v2 thesis = *decouple
identity from motion*). Secrets live in untracked `/workspace/.env`
(`AZURE_STORAGE_CONNECTION_STRING`, `AZURE_CONTAINER`) — **never** in the repo.

```bash
bash /workspace/Pudgy/finetune/wan/azure_upload.sh weights logs output   # what we pushed
bash azure_upload.sh weights output     # re-push final weights + eval later (idempotent)
bash azure_upload.sh all                # + optimizer state/ dirs (post-recycle resume backup)
```
Blob layout: `weights/` (LoRA ckpts) · `logs/` · `output/` (docs/v2 videos+montages+report) ·
`state/` (only with `all`). Files written in the last 30 s are skipped (avoids partial
checkpoints); live logs are captured on the next run. Container URL:
`https://pudgytraining.blob.core.windows.net/v2-decoupled-identity-motion/`

### Experiment tracking (wandb — project `pudgy`)

The script logs to **wandb project `pudgy`** (+ tensorboard) by default, one run per
expert (`--wandb_run_name` = the LoRA name). **Authenticate once before running:**
```bash
! wandb login                       # writes ~/.netrc (persists), OR
export WANDB_API_KEY=<your-key>     # picked up automatically by the script
```
Knobs: `WANDB_PROJECT=pudgy` (override project), `LOG_WITH=all|wandb|tensorboard|none`.
Without auth, `LOG_WITH=none` falls back to no tracker. (The currently running
low-noise run pre-dates this — it has no wandb; future runs do.)

### Preview during training & resume (`--save_state` / sampling)

The script always passes **`--save_state`** (saves optimizer/scheduler state each
checkpoint) so a run can be **stopped and resumed exactly**:
```bash
RESUME=/workspace/wan_output/<name>/<name>-<NNNN>-state EXPERT=low bash train_pudgy_wan_a14b.sh
```
It also does **in-process sampling** every `SAMPLE_EVERY` epochs (0 = off) from
`sample_prompts.txt` (plain I2V off the start keyframe, 480×832) → previews land in
`<output_dir>/sample/` as `.mp4`. No second process, no separate model load from *your* side.

> ⚠️ **VRAM caveat.** In-process sampling still loads the T5 (~11 GB) + VAE during
> generation. At **full-res fp16 training the DiT already holds ~67 GB** of the 80 GB
> card, leaving too little for sampling → it can OOM-kill the run. So sampling is safe
> only when the run has headroom: pair it with **`--fp8_base`** and/or a **lower-res**
> bucket (which frees 20-40 GB). For a full-res fp16 run, set `SAMPLE_EVERY=0` and
> preview after training via `eval_flf2v.sh` instead. `--save_state` has no such cost —
> it's always on and cheap.

> The **currently running** low-noise run pre-dates these flags (no `--save_state`,
> no sampling) — it can't be exactly resumed; it just runs to completion.

## Re-caching (only if the dataset changes)

```bash
cd /workspace/musubi-tuner ; PY=/workspace/Pudgy/.venv-wan/bin/python
$PY src/musubi_tuner/wan_cache_latents.py --dataset_config <toml> \
    --vae <wan_2.1_vae.safetensors> --i2v          # CLIP not needed for 2.2
$PY src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config <toml> \
    --t5 <models_t5_umt5-xxl-enc-bf16.pth> --batch_size 16
```

## Next after training (v2 §2 — the actual point)

The LoRAs are the *motion* half. Identity-pinning is an **inference** step:
`wan_generate_video.py` with the `flf2v-14B` / first+last-frame workflow (feed
human-QC'd keyframes as endpoints) so identity can't drift mid-clip — the fix
for v1's "subject vanishes" failure. Attach the low-noise LoRA (identity) and
optionally the high-noise LoRA (motion) at inference.
