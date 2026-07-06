#!/bin/bash
# =============================================================================
# One-time environment setup for the CogVideoX1.5-5B-I2V LoRA trainer.
# Run on the GPU machine (Linux + NVIDIA CUDA GPU). NOT on macOS.
# Verified on an RTX 5090 (Blackwell / sm_120); also works on A100/H100.
# Usage:  cd <repo> && bash setup_gpu_env.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 0) System packages: ffmpeg (provides ffprobe, used by check_dataset.py) + git
if command -v apt-get >/dev/null 2>&1; then
  apt-get update && apt-get install -y ffmpeg git
else
  echo "NOTE: install 'ffmpeg' and 'git' via your package manager if missing."
fi

# 1) Python env (venv or conda — venv shown)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 2) PyTorch with CUDA — MATCH TO YOUR GPU.
#    Default = cu121 (A100/H100, sm_80/sm_90).
#    RTX 50-series / Blackwell (sm_120) instead need cu128 wheels (torch >= 2.7) — see below.
pip install "torch>=2.5.0" torchvision --index-url https://download.pytorch.org/whl/cu121
# RTX 50-series / Blackwell alternative:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) diffusers FROM SOURCE — required. The CogVideoX1.5 I2V training code
#    is NOT in any pip release yet, so a released diffusers will fail.
[ -d diffusers ] || git clone https://github.com/huggingface/diffusers.git
pip install -e ./diffusers

# 4) Trainer dependencies (requirements.txt already includes the trainer's imports)
pip install -r requirements.txt
pip install "huggingface_hub[cli]"

# 5) Base model (~20 GB). THUDM and zai-org host the same weights.
#    (huggingface-cli was renamed to `hf`.)
hf download THUDM/CogVideoX1.5-5B-I2V \
  --local-dir finetune/models/CogVideoX1.5-5B-I2V
#   (fallback id if the above 404s:  zai-org/CogVideoX1.5-5B-I2V)

# 6) Accelerate default config (single GPU, bf16)
accelerate config default --mixed_precision bf16 || true

echo ""
echo "✅ Setup complete."
echo "   Verify GPU:               python -c \"import torch; print(torch.cuda.get_device_name(0))\""
echo "   Sanity-check the dataset: python check_dataset.py \$DATASET_DIR"
echo "   Start training:           DATASET_DIR=/path/to/training_dataset bash finetune/scripts/train_pudgy_lora.sh"
