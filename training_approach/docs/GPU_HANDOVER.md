# GPU Handover Runbook — Pudgy Penguins Training Approach v2

**Audience:** the GPU team executing Phase 0 (and standing up Phase 1). Written to run **without the author present** — every step is copy-paste.

**Author is on a Mac (no CUDA)** and cannot run any of this. Please execute the tasks below in order, capture the artifacts named in each "Report back" box, and return them so we can pass the decision gates.

**Read first, in this order:**
1. `Training_Approach_v1.md` — what run #1 was and why it fell short (the baseline v2 must beat).
2. `Training_Approach_v2.md` — the full plan, phases, gates, rubric.
3. This runbook — the executable steps for Phase 0 + Phase 1 setup.

---

## 0. Prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA CUDA, **A100/H100 80 GB recommended** (5090/32 GB works with the OOM tweaks in `pudgy-lora-repo/SETUP_GPU.md`). Not runnable on macOS/MPS. |
| Data | `training_dataset/` (75 clips + `metadata.json`) — transferred separately (scp / bucket / drive). Same data used in v1. |
| Repo | `pudgy-lora-repo/` — existing CogVideoX infra (env setup, `check_dataset.py`, gotchas). |
| Scripts | `Training_approach/scripts/` — new diagnostics for v2. |
| HF access | Some model repos are **gated** — accept the license on Hugging Face for Wan / Hunyuan / CogVideoX before first download, and `huggingface-cli login`. |

Transfer the data if not already on the box:
```bash
scp -r training_dataset  user@gpu-box:/data/
export DATASET_DIR=/data/training_dataset
```

---

## TASK 0.1 — VAE round-trip test  ·  ⭐ highest priority, decides the base model

**Goal:** find which candidate VAE preserves Pudgy's thick outlines + flat fills. The VAE is frozen, so this is a hard image-quality ceiling — a training-independent measurement. This is the single most decision-relevant number in Phase 0.

### Setup
```bash
cd /path/to/pudgy-lora-repo
source .venv/bin/activate            # or create a fresh venv
pip install -U "diffusers>=0.32" transformers accelerate torch \
               imageio imageio-ffmpeg numpy pillow scikit-image tqdm
```

### Run — once per VAE (each downloads that model's VAE, a few GB)
```bash
cd /path/to/Training_approach/scripts
OUT=./vae_out

python vae_roundtrip.py --vae wan21     --clips $DATASET_DIR/train --out $OUT   # Wan2.1 / Wan2.2-A14B, 8x
python vae_roundtrip.py --vae wan22_5b  --clips $DATASET_DIR/train --out $OUT   # Wan2.2-5B, 16x (expected worst)
python vae_roundtrip.py --vae hunyuan   --clips $DATASET_DIR/train --out $OUT   # HunyuanVideo, 8x
python vae_roundtrip.py --vae cogvideox --clips $DATASET_DIR/train --out $OUT   # incumbent, 8x
```
Smoke-test first with `--n 3` to confirm each model loads on your diffusers build.

### If a model class fails to import
The class names track your diffusers version. `pip install -U diffusers`, or open the model card's "Diffusers usage" snippet and update the `cls`/`repo` in `REGISTRY` at the top of `vae_roundtrip.py`. Repo IDs are commented there.

### Report back
> - `vae_out/<vae>/summary.json` for all four VAEs (PSNR, SSIM, **edge-SSIM**, **flat-MAE**).
> - `vae_out/<vae>/montages/*.png` — the visual source|recon|diff crops.
> - One-line call: **which VAE has the best edge-SSIM (outline fidelity) and lowest flat-MAE (least banding)?**

**Interpretation → Gate G0:** the winning VAE constrains the base choice (prefer 8× over 16×). If *every* VAE round-trips the outlines cleanly, image quality was a *training* problem, not a VAE ceiling → the fix is in Tasks 0.2/0.3. If outlines degrade even in the round-trip, no LoRA can fix it — pick the base by this test.

---

## TASK 0.2 — Caption A/B on the current CogVideoX run

**Goal:** isolate how much of v1's weakness was caption strategy vs the model. v1 captions densely re-describe identity ("Pax, a short round blue penguin…") — the base already renders that, so the identity loss signal is near-zero.

### Steps
1. Duplicate the dataset metadata to a **rare-token variant**: replace the identity description with a rare trigger token and keep only the *variable* content (background, action, camera). E.g.
   - v1: `"…showing Pax, a short round blue penguin, standing in a tiled room gesturing up toward a lamp…"`
   - A/B: `"pxngn0 style. a blue penguin gesturing up toward a hanging lamp in a tiled room, wooden door opening"` (identity → trigger token; describe what varies).
   - Keep `file_path`/`type`; only change `text`. Save as `metadata_raretoken.json`.
2. Re-run the **existing** launcher twice with identical settings, swapping only the metadata file:
```bash
cd /path/to/pudgy-lora-repo
# baseline (v1 captions)
DATASET_DIR=$DATASET_DIR bash finetune/scripts/train_pudgy_lora.sh
# rare-token captions
DATASET_META_NAME=$DATASET_DIR/metadata_raretoken.json \
DATASET_DIR=$DATASET_DIR OUTPUT_PATH=finetune/output_dir/pudgy-raretoken \
  bash finetune/scripts/train_pudgy_lora.sh
```
3. Generate the same held-out prompt set from both, score with the **§5 rubric** in `Training_Approach_v2.md`.

### Report back
> Rubric scores (identity / line-quality / motion / prompt-adherence / temporal) for both runs + a few sample clips. Did rare-token captions improve identity binding?

---

## TASK 0.3 — LoRA layer-targeting fix

**Goal:** confirm the cheapest known LoRA win — target **all linear layers incl. MLP/FFN**, not attention-only.

### Steps
- In the trainer, set the LoRA target modules to all linear layers (attention **and** the FFN/MLP projections), `--rank 16`, `--lora_alpha 32` (α = 2r). Check `train_cogvideox_image_to_video_lora.py` for how target modules are specified (a `target_modules` list or a `--lora_target` style flag); if it's hard-coded to attention only, patch it to include the MLP linears.
- Re-run with the rare-token metadata from 0.2, same held-out eval.

### Report back
> Rubric scores vs the 0.2 rare-token run. Did all-linear+MLP targeting improve fidelity? Note the exact target-module list used.

---

## Gate G0 — decision (we make this together from your reports)

Assemble from Tasks 0.1–0.3:
- **Base model chosen** (from 0.1 VAE ranking).
- **Image-quality root cause** identified (VAE ceiling vs training).
- **Caption + targeting deltas** measured (0.2, 0.3).

Once G0 passes we move to Phase 1.

---

## Phase 1 setup — new base env (start in parallel with Phase 0)

Stand up the chosen flow-matching base so we can train the moment G0 lands. Primary = **Wan 2.2 A14B**; run **AniSora** in parallel as the anime-native candidate.

### Recommended trainers (pick one to start; both are fine)
- **musubi-tuner** (kohya) — community default for Wan/Hunyuan character LoRAs. Trains the A14B high/low-noise experts separately. `https://github.com/kohya-ss/musubi-tuner`
- **diffusion-pipe** (tdrussell) — broadest model support, multi-GPU/block-swap. `https://github.com/tdrussell/diffusion-pipe`

### Dataset repackage
The 75 clips stay the same pixels; only the manifest/caption format changes to the chosen trainer's convention. Keep **768×1360 portrait**; resample fps to the base's native rate (Wan 16 fps — already matches v1; AniSora 24 fps — resample from the 24 fps *source*, not the 16 fps clips, to avoid double-resample). Use the **rare-token captions** from Task 0.2.

### Starting LoRA config (confirm at G0; full table in v2 §4)
```
method     = LoRA (not full fine-tune — overfits on 75 clips)
target     = all linear incl. MLP/FFN
rank / α   = 16–32 / 2r
LR / warmup= ~8e-5–1e-4 / ~100 steps
A14B       = TWO LoRAs — low-noise = identity/texture, high-noise = motion
shift      = matched to training resolution
VAE        = the Task 0.1 winner (prefer 8x spatial)
```

### Report back
> Env stood up (trainer + base weights downloaded), dataset repackaged, a first corrected-baseline LoRA trained + scored on the rubric. **Gate G1 = this beats the v1 CogVideoX run.**

---

## What to send back — checklist

- [ ] `vae_out/*/summary.json` + `montages/` for all 4 VAEs, + your pick (0.1)
- [ ] Rubric scores + samples: v1-captions vs rare-token (0.2)
- [ ] Rubric scores + samples: all-linear+MLP targeting (0.3), with the target-module list
- [ ] G0 recommendation: base model + root cause
- [ ] Phase 1: env up, dataset repackaged, corrected-baseline scores (G1)

Questions on any step → flag the specific task number. The plan, gates, and rubric are all in `Training_Approach_v2.md`.
