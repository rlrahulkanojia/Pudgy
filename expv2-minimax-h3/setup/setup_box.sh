#!/usr/bin/env bash
# =========================================================================================
# expv2 box setup: system packages, venv, PyTorch, DiffSynth-Studio at the pinned SHA with
# the expv2 patches, and the evaluation/Azure dependencies. Idempotent: re-run after any
# interruption. It does NOT download weights (setup/fetch_models.py) or data
# (setup/pull_data.sh).
#
#   cd /workspace/Pudgy && bash expv2-minimax-h3/setup/setup_box.sh
#
# Overrides: TORCH_INDEX=<pytorch wheel index url>   (default: picked from the driver)
#            SKIP_APT=1                              (no root / packages already present)
# =========================================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../configs/env.sh"
SHA="$(python3 -c "import sys; sys.path.insert(0, '$EXPV2_ROOT'); import expv2_common as C; print(C.DIFFSYNTH_SHA)")"
say() { printf '\n== %s\n' "$*"; }

say "0. system packages"
if [ "${SKIP_APT:-0}" != 1 ] && command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ffmpeg git tmux python3-venv python3-dev build-essential >/dev/null
fi
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "ffmpeg/ffprobe missing"; exit 1; }

say "1. hardware"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
[ "$GPU_MEM_MB" -ge 79000 ] || echo "WARNING: GPU has ${GPU_MEM_MB} MB; the plan assumes 80 GB (PLAN section 3)"
RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)"
[ "$RAM_GB" -ge 120 ] || echo "WARNING: ${RAM_GB} GB host RAM; stage 1 (Qwen3-VL-32B) wants >= 128 GB"
mkdir -p "$EXPV2_WORK" "$DIFFSYNTH_MODEL_BASE_PATH" "$HF_HOME"
DISK_GB="$(df -BG --output=avail "$EXPV2_WORK" | tail -1 | tr -dc 0-9)"
[ "$DISK_GB" -ge 300 ] || echo "WARNING: ${DISK_GB} GB free under $EXPV2_WORK; budget is >= 300 GB (weights ~119 GB, +66 GB for the G0.2a parity DiT)"

say "2. venv at $EXPV2_VENV"
[ -x "$EXPV2_VENV/bin/python" ] || python3 -m venv "$EXPV2_VENV"
# shellcheck disable=SC1091
source "$EXPV2_VENV/bin/activate"
python -c 'import sys; assert sys.version_info >= (3, 10, 1), sys.version' || { echo "need python >= 3.10.1 (DiffSynth)"; exit 1; }
pip install -q --upgrade pip wheel setuptools

say "3. PyTorch"
if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  if [ -z "${TORCH_INDEX:-}" ]; then
    # Pick the newest CUDA build the driver can run. nvidia-smi prints "CUDA Version: X.Y".
    CU="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    if   [ "${CU:-0}" -ge 128 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu128
    elif [ "${CU:-0}" -ge 126 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu126
    else                               TORCH_INDEX=https://download.pytorch.org/whl/cu124
    fi
  fi
  echo "torch index: $TORCH_INDEX"
  pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'

say "4. DiffSynth-Studio @ $SHA (+ expv2 patches)"
if [ ! -d "$DIFFSYNTH_DIR/.git" ]; then
  git clone -q https://github.com/modelscope/DiffSynth-Studio "$DIFFSYNTH_DIR"
fi
if [ "$(git -C "$DIFFSYNTH_DIR" rev-parse HEAD)" != "$SHA" ]; then
  git -C "$DIFFSYNTH_DIR" fetch -q origin
  if ! git -C "$DIFFSYNTH_DIR" diff --quiet; then
    echo "DiffSynth checkout has local changes and is not at $SHA - inspect $DIFFSYNTH_DIR"; exit 1
  fi
  git -C "$DIFFSYNTH_DIR" checkout -q "$SHA"
fi
bash "$EXPV2_ROOT/patches/apply.sh"
pip install -q -e "$DIFFSYNTH_DIR"

say "5. expv2 dependencies"
# transformers must provide Qwen3VLModel (diffsynth/models/minimax_h3_text_encoder.py).
# diffusers: only for G0.1's Wan reference VAEs (eval/vae_gate.py). scipy: scoring.
pip install -q --upgrade "transformers>=4.57" "huggingface_hub[hf_transfer]" modelscope diffusers \
  "imageio[ffmpeg]" av safetensors pandas scipy pillow azure-storage-blob python-dotenv tensorboard
python -c 'from transformers import Qwen3VLModel; import torchaudio, diffsynth; print("imports ok")'

say "6. accelerate (1 GPU, bf16)"
accelerate config default --mixed_precision bf16 >/dev/null 2>&1 || true

say "7. record the environment"
mkdir -p "$EXPV2_WORK/eval/phase0"
python - <<EOF
import json, subprocess, sys, platform
import torch, transformers, diffsynth
sys.path.insert(0, "$EXPV2_ROOT"); import expv2_common as C
ds = "$DIFFSYNTH_DIR"
env = {
  "python": sys.version.split()[0], "platform": platform.platform(),
  "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
  "transformers": transformers.__version__,
  "diffsynth_sha": subprocess.check_output(["git", "-C", ds, "rev-parse", "HEAD"], text=True).strip(),
  "diffsynth_dirty_files": subprocess.check_output(["git", "-C", ds, "diff", "--name-only"], text=True).split(),
  "repo_sha": subprocess.check_output(["git", "-C", "$REPO_ROOT", "rev-parse", "HEAD"], text=True).strip(),
  "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).split("\n"),
}
C.write_json(C.PHASE0 / "env.json", env)
print("wrote", C.PHASE0 / "env.json")
EOF

say "done"
echo "next: python expv2-minimax-h3/setup/fetch_models.py --set core   (then RUNBOOK.md step 3)"
