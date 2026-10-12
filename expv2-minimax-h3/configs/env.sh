# expv2 box environment. Source it in every shell:  source expv2-minimax-h3/configs/env.sh
# Every path can be overridden by exporting it before sourcing.

EXPV2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPV2_ROOT
export REPO_ROOT="$(dirname "$EXPV2_ROOT")"

export PUDGY_DATA="${PUDGY_DATA:-/workspace/Data}"                     # raw/ + processed/expv2_h3/
export EXPV2_WORK="${EXPV2_WORK:-/workspace/expv2}"                    # cache/, output/, eval/
export DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-/workspace/DiffSynth-Studio}"   # pinned + patched checkout
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/workspace/models}"
export EXPV2_VENV="${EXPV2_VENV:-/workspace/venv-expv2}"
export HF_HOME="${HF_HOME:-/workspace/hf_home}"

# Weights are fetched once by setup/fetch_models.py (HuggingFace, plus ModelScope for the
# arm-D adapter). After that DiffSynth must never download: SKIP_DOWNLOAD makes it glob
# $DIFFSYNTH_MODEL_BASE_PATH/<model id>/<pattern> directly, so a run cannot stall on the
# network. fetch_models.py also links MiniMax/MiniMax-H3 -> MiniMaxAI/MiniMax-H3, because
# DiffSynth's default processor config names the ModelScope id.
export DIFFSYNTH_DOWNLOAD_SOURCE=HuggingFace
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Azure: the connection string lives in /workspace/.env, copied to the box by hand.
# It is never committed and never echoed.
if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ] && [ -f /workspace/.env ]; then
  AZURE_STORAGE_CONNECTION_STRING="$(grep '^AZURE_STORAGE_CONNECTION_STRING' /workspace/.env | cut -d= -f2- | tr -d '"')"
  export AZURE_STORAGE_CONNECTION_STRING
fi

if [ -f "$EXPV2_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$EXPV2_VENV/bin/activate"
fi

# Recipe constants shared by cache.sh / train.sh (PLAN section 6.2). Change them here only.
export EXPV2_H3_REPO="MiniMaxAI/MiniMax-H3"
export EXPV2_DIT="Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
export EXPV2_CACHE_MODELS="MiniMaxAI/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMaxAI/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMaxAI/MiniMax-H3:FL2VA/audio_vae/model.safetensors"
export EXPV2_ADAPTER="$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_for_comfy_dit.safetensors"
export EXPV2_LORA_TARGETS="attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
export EXPV2_LORA_RANK=32
export EXPV2_GRAD_ACCUM=8
export EXPV2_SAVE_EVERY_OPT=50          # optimizer steps; DiffSynth's --save_steps counts samples
# Trained length buckets (PLAN 5.2): f22 only. The corpus also has neutral-only f39/f56
# rungs; "22 39 56" trains them too (and re-enables G0.2f in run_expv2.sh).
export EXPV2_BUCKETS="${EXPV2_BUCKETS:-22}"
