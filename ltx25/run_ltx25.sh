#!/bin/bash
# run_ltx25.sh - steps 5 and 6 (preprocess, then train) for the standalone LTX-2.5
# experiment, on THIS box. Nothing here is shared with the v1-v7 Wan line.
#
#   ./ltx25/run_ltx25.sh gate         # step 2: VAE round-trip. Run this FIRST.
#   ./ltx25/run_ltx25.sh preprocess   # step 5: dataset -> .precomputed
#   ./ltx25/run_ltx25.sh train        # step 6
#
# Every value that could drift is READ, not hardcoded:
#   - the resolution-bucket string comes from manifest.json, which prep_ltx25.py wrote
#     from what it ACTUALLY emitted (the docs' two hand-written bucket strings disagree
#     with each other, so neither is trusted here)
#   - the config is generated from the committed template with paths substituted
#
# This box: 1 x A100 80GB (Ampere, SM 8.0), driver CUDA 13.2.
#   --num_processes 1, NOT the docs' 2: there is one GPU.
#   quantization stays null: ltx-kernels' FP8 GEMM needs SM89+ (Ada) and NVFP4 needs
#   Blackwell, so neither builds for SM 8.0 - and LTX-2 #296/#253 report fp8-cast
#   crashing with any LoRA on pre-Hopper anyway.
set -euo pipefail

ROOT=/workspace/Pudgy
LTX=/workspace/LTX-2
MODELS=/workspace/models/ltx-2.5
DATA=$ROOT/Data/processed/ltx25_experiment
MANIFEST=$DATA/manifest.json

DEV=$MODELS/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
TEXT=$MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
VVAE=$MODELS/vae/ltx-2.5-video-vae-bf16.safetensors
CVAE=$MODELS/vae/ltx-2.5-video-vae-conv-bf16.safetensors
AVAE=$MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors

need() { [ -f "$1" ] || { echo "MISSING: $1" >&2; exit 1; }; }

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

require_gpu() {  # never fight another tenant for the card
  local want=${1:-60000} free; free=$(gpu_free_mb)
  if [ "$free" -lt "$want" ]; then
    echo "GPU has ${free} MiB free, want >= ${want}. Something else is running - waiting." >&2
    echo "(ctrl-c to abort; this never kills another process)" >&2
    while [ "$(gpu_free_mb)" -lt "$want" ]; do sleep 60; done
    echo "GPU free, proceeding." >&2
  fi
}

buckets() {  # authoritative bucket string, straight from what prep emitted
  need "$MANIFEST"
  python3 -c "import json;print(json.load(open('$MANIFEST'))['resolution_buckets'])"
}

case "${1:-}" in

gate)
  need "$VVAE"; need "$CVAE"
  require_gpu 20000
  # Both characters, deliberately: Polly has historically been the weaker identity.
  mapfile -t CLIPS < <(ls "$DATA"/clips/pax_*_FRONT__*.mp4 2>/dev/null | head -2
                       ls "$DATA"/clips/polly_*_FRONT__*.mp4 2>/dev/null | head -2)
  [ ${#CLIPS[@]} -gt 0 ] || { echo "no clips in $DATA/clips - run prep first" >&2; exit 1; }
  cd "$LTX"
  uv run python "$ROOT/ltx25/eval/vae_gate_ltx25.py" \
      --vae "diffusion:$VVAE" --vae "conv:$CVAE" \
      --clips "${CLIPS[@]}" \
      --out "$ROOT/ltx25/eval/vae_gate.json" \
      --dump-frames "$ROOT/ltx25/eval/vae_gate_frames"
  echo
  echo ">> Now LOOK at ltx25/eval/vae_gate_frames/*.png before going further."
  echo ">> PIPELINE.md S1: if both decoders soften the outlines, STOP."
  ;;

preprocess)
  need "$DEV"; need "$TEXT"; need "$VVAE"; need "$AVAE"; need "$DATA/dataset.json"
  require_gpu 60000
  B=$(buckets); echo "buckets from manifest: $B"
  # FRESH .precomputed: Gemma-4 embeddings are not interchangeable with Gemma-3 and
  # stale conditions/*.pt are SKIPPED SILENTLY rather than rejected.
  cd "$LTX"
  uv run accelerate launch --num_processes 1 --mixed_precision bf16 \
    packages/ltx-trainer/scripts/process_dataset.py "$DATA/dataset.json" \
      --resolution-buckets "$B" \
      --model-path        "$DEV" \
      --text-encoder-path "$TEXT" \
      --video-vae-path    "$VVAE" \
      --audio-vae-path    "$AVAE" \
      --skip-audio --decode
  echo
  echo ">> EYEBALL $DATA/.precomputed/decoded_videos BEFORE training."
  ;;

train)
  need "$DEV"; need "$TEXT"; need "$VVAE"
  [ -d "$DATA/.precomputed" ] || { echo "no .precomputed - run preprocess first" >&2; exit 1; }
  require_gpu 60000
  CFG=$ROOT/ltx25/configs/i2v_lora_ltx25.workspace.yaml
  need "$CFG"
  cd "$LTX"
  uv run python packages/ltx-trainer/scripts/train.py "$CFG"
  echo
  echo ">> Do NOT pick a checkpoint from this run's own validation samples (LTX-2 #283)."
  echo ">> Render them through ltx-pipelines and score those: see PIPELINE.md S10.3."
  ;;

*)
  sed -n '2,12p' "$0"; exit 1 ;;
esac
