#!/bin/bash
# =============================================================================
# One-time environment setup for the CogVideoX1.5-5B-I2V LoRA trainer.
# Run on the GPU machine (Linux + CUDA, A100 80GB recommended). NOT on macOS.
# Usage:  cd trainer_code && bash setup_gpu_env.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1) Python env (venv or conda — venv shown)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 2) PyTorch with CUDA (pick the build matching your CUDA; cu121 shown)
pip install "torch>=2.5.0" torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) diffusers FROM SOURCE — required. The CogVideoX1.5 I2V training code
#    is NOT in any pip release yet, so a released diffusers will fail.
[ -d diffusers ] || git clone https://github.com/huggingface/diffusers.git
pip install -e ./diffusers

# 4) Trainer dependencies (+ a few the repo assumes)
pip install -r requirements.txt
pip install accelerate transformers sentencepiece decord imageio imageio-ffmpeg \
            bitsandbytes wandb "huggingface_hub[cli]"

# 5) Base model (~20 GB). THUDM and zai-org host the same weights.
huggingface-cli download THUDM/CogVideoX1.5-5B-I2V \
  --local-dir finetune/models/CogVideoX1.5-5B-I2V
#   (fallback id if the above 404s:  zai-org/CogVideoX1.5-5B-I2V)

# 6) Accelerate default config (single GPU, bf16)
accelerate config default --mixed_precision bf16 || true

echo ""
echo "✅ Setup complete."
echo "   Sanity-check the dataset:   python finetune/scripts/../../check_dataset.py  (optional)"
echo "   Start training:             bash finetune/scripts/train_pudgy_lora.sh"
