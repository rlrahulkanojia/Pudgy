#!/bin/bash
# =============================================================================
# Stand up the Wan2.2-I2V-A14B training environment from nothing.
#
# The v6 runbook (Training_Approach_v6 section 8, step 0) calls this script but it
# did not exist — the v2 stand-up was done by hand and only written up prose-style in
# training_approach/v2/actions_done.md. Since `/workspace` is NOT a persistent volume
# and the v5 box was already lost once with its whole environment on it, "the recipe
# lives in a paragraph someone has to re-read" is a real failure mode. This is that
# paragraph, executable.
#
#   bash setup_wan_env.sh              # venv + trainer + ~65 GB of base weights
#   SKIP_WEIGHTS=1 bash setup_wan_env.sh
#
# Idempotent: every step checks before doing. Safe to re-run after an interruption.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/musubi-tuner}"
VENV="${VENV:-/workspace/Pudgy/.venv-wan}"
MODELS="${MODELS:-/workspace/wan_models}"
SKIP_WEIGHTS="${SKIP_WEIGHTS:-0}"
HF_BIN="${HF_BIN:-/venv/main/bin/hf}"

echo "=============================================================="
echo " Wan2.2-A14B env stand-up"
echo "   trainer : $REPO_DIR"
echo "   venv    : $VENV"
echo "   weights : $MODELS"
echo "=============================================================="

# --- 0. Preflight ------------------------------------------------------------------
command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no GPU visible"; exit 1; }

# The A14B fp16 DiT is ~28.6 GB and training resident set is ~67 GB at full res, so
# this is an 80 GB-class job. v2/FINDINGS section 5: on 40 GB it is fp8+block-swap only.
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 70000 ]; then
  echo "!! ${VRAM} MiB VRAM — below the 80 GB tier this recipe assumes."
  echo "!! Expect to need --fp8_base and --blocks_to_swap; see v2/actions_done.md."
fi

# --- 1. Trainer --------------------------------------------------------------------
# Pinned to the version v2 and v5 were trained with. musubi's LoRA key layout and the
# i2v-A14B timestep-split flags are version-sensitive, and the v2 goldens must load
# into the same module set (400 modules) they were trained as.
if [ ! -d "$REPO_DIR" ]; then
  git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git "$REPO_DIR"
else
  echo "-- trainer already present"
fi
grep -q 'version = "0.3.4"' "$REPO_DIR/pyproject.toml" \
  || echo "!! musubi-tuner is not 0.3.4 — v2/v5/v6 were all trained on 0.3.4"

# --- 2. Venv -----------------------------------------------------------------------
# Deliberately NOT /venv/main: musubi pins diffusers 0.32.1 / transformers 4.57.6 /
# accelerate 1.6.0, which conflict with the base image.
if [ ! -d "$VENV" ]; then
  python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

# Match the CUDA wheel to the HOST DRIVER, do not assume v2's cu128. CUDA minor-version
# compatibility means any 12.x wheel runs on a 12.x driver, but picking the matching
# minor avoids the PTX-JIT edge case entirely. A100 is sm_80, covered by every option.
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
case "${DRV_CUDA:-126}" in
  13*) IDX=cu130 ;;
  128|129) IDX=cu128 ;;
  *)   IDX=cu126 ;;
esac
echo "-- host driver reports CUDA ${DRV_CUDA:-unknown} -> torch index $IDX"
python -c "import torch" 2>/dev/null || \
  pip install -q "torch>=2.7.1" "torchvision>=0.22.1" \
      --index-url "https://download.pytorch.org/whl/$IDX"

pip install -q -e "$REPO_DIR"
# ascii-magic/matplotlib are musubi's optional preview deps; bitsandbytes provides
# adamw8bit (the optimiser v2/v5/v6 all use); scikit-image/opencv are for the gates.
pip install -q ascii-magic matplotlib tensorboard prompt-toolkit bitsandbytes \
    scikit-image opencv-python-headless azure-storage-blob

python - <<'PY'
import torch, diffusers, transformers, accelerate
assert torch.cuda.is_available(), "torch cannot see the GPU"
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
_ = (x @ x).float().mean().item()          # prove real kernels run, not just init
print(f"   torch {torch.__version__} (cuda {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
print(f"   diffusers {diffusers.__version__} · transformers {transformers.__version__} · accelerate {accelerate.__version__}")
PY

# --- 3. Base weights (~65 GB) ------------------------------------------------------
# All public/ungated — no HF token needed. Use POSITIONAL filenames: the newer hf CLI
# silently ignores --include and half-completes the download (a real gotcha from the
# v2 stand-up, see actions_done.md section 3).
if [ "$SKIP_WEIGHTS" != "1" ]; then
  mkdir -p "$MODELS"
  DIT="$MODELS/comfy22/split_files/diffusion_models"
  if [ ! -s "$DIT/wan2.2_i2v_low_noise_14B_fp16.safetensors" ]; then
    "$HF_BIN" download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
      split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
      split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
      --local-dir "$MODELS/comfy22"
  fi
  # The Wan2.1 8x VAE — NOT Wan2.2_VAE.pth, which is the 5B 16x VAE: incompatible with
  # the 14B and the outline-softener FINDINGS section 4 warns against.
  if [ ! -s "$MODELS/comfy21/split_files/vae/wan_2.1_vae.safetensors" ]; then
    "$HF_BIN" download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
      split_files/vae/wan_2.1_vae.safetensors --local-dir "$MODELS/comfy21"
  fi
  if [ ! -s "$MODELS/t5/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    "$HF_BIN" download Wan-AI/Wan2.1-I2V-14B-720P \
      models_t5_umt5-xxl-enc-bf16.pth --local-dir "$MODELS/t5"
  fi
  # CLIP is NOT required for Wan2.2 (it was for 2.1) — deliberately not downloaded.
  du -sh "$MODELS"/* 2>/dev/null || true
fi

cat <<EOF

=============================================================='
 environment ready.
   activate : source $VENV/bin/activate
 next (v6)  : pull data + goldens from Azure, then pre-cache and train
              see training_approach/v6/Training_Approach_v6.md section 8
=============================================================='
EOF
