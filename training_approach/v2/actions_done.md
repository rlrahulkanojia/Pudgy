# Actions done — Wan2.2-I2V-A14B env stand-up (v2 Phase 1.1)

Reproducible record of standing up the **Wan2.2-I2V-A14B + musubi-tuner** training
environment called for in [`Training_Approach_v2.md`](./Training_Approach_v2.md)
§1.1 / §4. This is the concrete execution of the base migration, now that the box
is an **A100 80 GB** — which lifts the v1 blocker (A14B was fp8+block-swap-only on
the old 40 GB card; see [`FINDINGS.md`](../FINDINGS.md) §5).

**Date:** 2026-07-10 · **Status:** env built, pre-cached, smoke-tested. Not yet
trained. G0 VAE decision was already settled in [`phase0_diagnostics.md`](../phase0_diagnostics.md)
(8× VAE is near-lossless; avoid the 5B 16× VAE).

---

## 0. Hardware / preconditions

| | |
|---|---|
| GPU | NVIDIA A100 **80 GB** PCIe, driver 595.58.03, system CUDA 13.0 |
| Base image venv | `/venv/main` (torch 2.12+cu130) — **not reused** (see §2) |
| Dataset | `/workspace/training_dataset/` — 75 clips + `metadata.json`, already 768×1360 · 33f · 16 fps |
| ⚠️ Disk | `/workspace` is **NOT a persistent volume** (`workspace_is_volume: false`) → everything below is wiped on recycle/destroy. Sync off-box to keep it. |
| HF token | none needed — all repos below are public/ungated |

All work lives **outside git** under `/workspace` (weights/venv/caches); only the
repo-tracked configs/scripts/README are under `Pudgy/finetune/wan/`.

---

## 1. Clone the trainer

```bash
cd /workspace
git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git   # v0.3.4
```
Authoritative Wan guide: `musubi-tuner/docs/wan.md`. Key facts it establishes:
- Task = **`i2v-A14B`**; the A14B is **two DiT experts** (high + low noise) — download both.
- **VAE = the Wan2.1 8× VAE** (`Wan2.1_VAE.pth` / `wan_2.1_vae.safetensors`). The
  `Wan2.2_VAE.pth` is the **5B 16×** VAE — incompatible with 14B *and* the outline-softener v2 warns against.
- **CLIP is NOT required for Wan2.2** (it was for 2.1).
- Comfy-Org repacks the 14B I2V DiT only in **fp16** (no bf16 single-file) — fp16
  is the full-quality path; the dtype matrix requires **fp16 training** on fp16 weights.

## 2. Dedicated venv (do not reuse `/venv/main`)

musubi pins `diffusers==0.32.1`, `transformers==4.57.6`, `accelerate==1.6.0` — hard
conflicts with the base image. A100 is sm_80 → **cu128 / torch 2.7** is the safe wheel.

```bash
python3.12 -m venv /workspace/Pudgy/.venv-wan
source /workspace/Pudgy/.venv-wan/bin/activate
pip install --upgrade pip wheel
pip install "torch>=2.7.1" "torchvision>=0.22.1" --index-url https://download.pytorch.org/whl/cu128
cd /workspace/musubi-tuner && pip install -e .
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```
Verified: `torch 2.11.0+cu128`, `musubi-tuner 0.3.4`, `diffusers 0.32.1` import OK.

## 3. Download weights (~65 GB) to `/workspace/wan_models/`

Use **positional filenames** with `hf download` — the newer CLI silently ignores
`--include`, which half-fails the download (a real gotcha we hit).

```bash
HF=/venv/main/bin/hf   # hf CLI 1.18 (from base image)

# DiT experts (fp16, 28.6 GB each) — Comfy-Org repackaged
$HF download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
  split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
  --local-dir /workspace/wan_models/comfy22

# VAE (Wan2.1 8×, 0.25 GB)
$HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/vae/wan_2.1_vae.safetensors --local-dir /workspace/wan_models/comfy21

# Text encoder UMT5-XXL (11.4 GB)
$HF download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth --local-dir /workspace/wan_models/t5
```

Resulting paths (used by all scripts below):
```
/workspace/wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_{high,low}_noise_14B_fp16.safetensors
/workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors
/workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth
```

## 4. Convert the 75-clip dataset to musubi format

Captions live in a separate `captions/` dir, so the clean path is a **JSONL** built
from the existing `metadata.json` (which already pairs `file_path` + `text`):

```bash
python - <<'PY'
import json, os
data=json.load(open("/workspace/training_dataset/metadata.json"))
with open("/workspace/Pudgy/finetune/wan/dataset.jsonl","w") as f:
    for e in data:
        f.write(json.dumps({"video_path": os.path.join("/workspace/training_dataset", e["file_path"]),
                            "caption": e["text"]})+"\n")
PY
```

`Pudgy/finetune/wan/dataset_config.toml` (tracked in-repo):
```toml
[general]
resolution = [768, 1360]     # [W,H] portrait — matches source exactly
batch_size = 1
enable_bucket = true
bucket_no_upscale = true
[[datasets]]
video_jsonl_file = "/workspace/Pudgy/finetune/wan/dataset.jsonl"
cache_directory = "/workspace/wan_cache/latents"
target_frames = [33]         # 33 = 4N+1
frame_extraction = "head"
num_repeats = 1
```
> `source_fps` omitted on purpose — clips are already 16 fps (Wan-native), so all 33
> frames are used. The resulting latent is `16×9×170×96`, matching phase0's round-trip.

## 5. Pre-cache latents + text-encoder outputs

```bash
cd /workspace/musubi-tuner ; PY=/workspace/Pudgy/.venv-wan/bin/python

# Latents — note --i2v; CLIP NOT needed for Wan2.2
$PY src/musubi_tuner/wan_cache_latents.py \
  --dataset_config /workspace/Pudgy/finetune/wan/dataset_config.toml \
  --vae /workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors --i2v

# Text encoder (UMT5)
$PY src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config /workspace/Pudgy/finetune/wan/dataset_config.toml \
  --t5 /workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16
```
Produces **75 `*_wan.safetensors` + 75 `*_wan_te.safetensors`** in `/workspace/wan_cache/latents/`.
(Latent caching ~6 s/clip; text-encoder caching ~6 s total.)

## 6. Two-expert training scripts

`Pudgy/finetune/wan/train_pudgy_wan_a14b.sh` (tracked) implements v2 §1.3 — two
**separate** LoRAs with the I2V timestep split + `--preserve_distribution_shape`:

| Expert | Role | Timesteps | LR |
|---|---|---|---|
| low-noise | identity / texture | 0–900 | 5e-5 |
| high-noise | motion / composition | 900–1000 | 1e-4 |

Core command (per expert):
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B --dit <expert>.safetensors \
  --dataset_config .../dataset_config.toml --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 32 \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep <lo> --max_timestep <hi> --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate <lr> --gradient_checkpointing \
  --max_train_epochs 40 --save_every_n_epochs 2 --seed 42 \
  --output_dir /workspace/wan_output/<name> --output_name <name>
```
Run it:
```bash
EXPERT=low  bash /workspace/Pudgy/finetune/wan/train_pudgy_wan_a14b.sh
EXPERT=high bash /workspace/Pudgy/finetune/wan/train_pudgy_wan_a14b.sh
```

**Why these settings:** `networks.lora_wan` replaces `WanAttentionBlock` (self-attn +
cross-attn + **FFN**) → v2's "all-linear incl. MLP" is the default. rank 16 / α 32
(α=2r). fp16 because the DiT weights are fp16 (musubi's dtype matrix forbids bf16
training on fp16 weights). flow-shift 5.0 = official I2V shift.

---

## 7. Verification (smoke test)

Launched the low-noise run and confirmed it reaches training, then killed it:
- `create LoRA for U-Net/DiT: 400 modules` ✅ (all-linear incl FFN)
- `use 8-bit AdamW optimizer`, `Gradient checkpointing enabled` ✅
- `DiT dtype: torch.float16, device: cuda:0` ✅
- `total optimization steps: 75` / `epoch 1/1`, began stepping ✅

GPU released to 0 MiB afterward. **The pipeline is proven end-to-end without a full run.**

---

## 8. Reproduce from scratch (one block)

```bash
# 1 trainer + venv
cd /workspace && git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git
python3.12 -m venv /workspace/Pudgy/.venv-wan && source /workspace/Pudgy/.venv-wan/bin/activate
pip install -U pip wheel && pip install "torch>=2.7.1" "torchvision>=0.22.1" --index-url https://download.pytorch.org/whl/cu128
cd /workspace/musubi-tuner && pip install -e . && pip install tensorboard
# 2 weights (see §3 — positional filenames, not --include)
# 3 dataset jsonl + toml (see §4)
# 4 pre-cache (see §5)
# 5 train (see §6)
```

## 9. Next (not yet done)

1. Actually train both experts; pick the **golden checkpoint by eye** (v2 §4).
2. Wire **FLF2V identity-pinning inference** (`wan_generate_video.py`, `flf2v-14B` /
   first+last-frame) — the real fix for v1's mid-clip subject-vanish. Attach the
   low-noise (identity) LoRA, optionally the high-noise (motion) LoRA.
3. Score against the v2 §5 rubric vs the v1 CogVideoX baseline at **Gate G1**.

*Related: [`Training_Approach_v2.md`](./Training_Approach_v2.md), [`base_model_exploration.md`](../base_model_exploration.md), [`phase0_diagnostics.md`](../phase0_diagnostics.md), and the in-env `Pudgy/finetune/wan/README.md`.*
