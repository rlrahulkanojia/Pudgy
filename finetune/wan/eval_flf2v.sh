#!/bin/bash
# =============================================================================
# FLF2V identity-pinning eval — Pudgy Wan2.2-I2V-A14B low-noise LoRA
# Implements the v2 §2 test: pin identity at t=0 AND t=1 with human-verified
# keyframes, let the model only in-between. This is the direct probe of whether
# v2 fixes v1's mid-clip "subject vanishes" failure (FINDINGS.md §3).
#
# Uses task i2v-A14B with BOTH --image_path (start) and --end_image_path (end).
# For this interim eval only the low-noise expert is used (high-noise LoRA not
# trained yet); wan.md: omitting --dit_high_noise uses low-noise for all steps.
#
# Usage:
#   bash eval_flf2v.sh                       # latest ckpt, clip 00000001, scale 1.0
#   SRC=00000065 CKPT=/path/to.safetensors SCALE=0.8 bash eval_flf2v.sh
# =============================================================================
set -euo pipefail

SRC="${SRC:-00000001}"                       # source clip for start/end keyframes
SCALE="${SCALE:-1.0}"                         # LoRA multiplier
STEPS="${STEPS:-25}"
SHIFT="${SHIFT:-5.0}"                          # I2V flow shift
GUID="${GUID:-5.0}"

REPO=/workspace/musubi-tuner
PY=/workspace/Pudgy/.venv-wan/bin/python
MODELS=/workspace/wan_models
DIT_LOW="$MODELS/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
VAE="$MODELS/comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5="$MODELS/t5/models_t5_umt5-xxl-enc-bf16.pth"
OUTDIR=/workspace/wan_output/pudgy-wan22-a14b-lownoise
CLIP=/workspace/training_dataset/train/${SRC}.mp4

# 1) latest LoRA checkpoint (unless CKPT given)
CKPT="${CKPT:-$(ls -t "$OUTDIR"/*.safetensors 2>/dev/null | head -1 || true)}"
if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "No LoRA checkpoint found in $OUTDIR yet — train first (or pass CKPT=...)."; exit 1
fi
echo "== FLF2V eval =="
echo "   LoRA   : $CKPT  (scale $SCALE)"
echo "   source : $CLIP"

# 2) extract start (frame 0) + end (last frame) keyframes at native size
WORK=$(dirname "$CKPT")/eval ; mkdir -p "$WORK"
STAMP=$(basename "$CKPT" .safetensors)
START="$WORK/${SRC}_start.png" ; END="$WORK/${SRC}_end.png"
NFR=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$CLIP")
ffmpeg -y -loglevel error -i "$CLIP" -vf "select=eq(n\,0)" -vframes 1 "$START"
ffmpeg -y -loglevel error -i "$CLIP" -vf "select=eq(n\,$((NFR-1)))" -vframes 1 "$END"
echo "   keyframes: $START (f0) + $END (f$((NFR-1))) of $NFR"

# 3) grab the source caption for the prompt
PROMPT=$($PY -c "import json;print(next(e['text'] for e in json.load(open('/workspace/training_dataset/metadata.json')) if e['file_path'].endswith('${SRC}.mp4')))")
SAVE="$WORK/flf2v_${STAMP}_${SRC}_s${SCALE}.mp4"

# 4) generate — 33 frames @ 768x1360, start+end pinned, low-noise LoRA attached.
# Memory-optimized: fp8 (+scaled for quality) DiT, fp8 T5, VAE cache on CPU, block
# swap. Full-res fp16 inference OOMs at load (model.to(dtype) transiently doubles the
# 28GB DiT). fp8_scaled quality tier is bf16/fp16 > fp8_scaled > fp8, so this is the
# best that fits. Override with FP8=0 to attempt fp16 (needs BLKSWAP high).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FP8="${FP8:-1}" ; BLKSWAP="${BLKSWAP:-20}"
MEM_ARGS=(--vae_cache_cpu --blocks_to_swap "$BLKSWAP")
[ "$FP8" = "1" ] && MEM_ARGS+=(--fp8 --fp8_scaled --fp8_t5)
cd "$REPO"
$PY src/musubi_tuner/wan_generate_video.py \
  --task i2v-A14B --dit "$DIT_LOW" --vae "$VAE" --t5 "$T5" \
  --lora_weight "$CKPT" --lora_multiplier "$SCALE" \
  --video_size 768 1360 --video_length 33 --fps 16 \
  --infer_steps "$STEPS" --flow_shift "$SHIFT" --guidance_scale "$GUID" \
  --image_path "$START" --end_image_path "$END" --trim_tail_frames 3 \
  --prompt "$PROMPT" --seed 42 --attn_mode sdpa \
  "${MEM_ARGS[@]}" \
  --save_path "$SAVE" --output_type both

# 5) contact-sheet montage for eye scoring (v2 §5 rubric)
MONT="${SAVE%.mp4}_montage.png"
ffmpeg -y -loglevel error -i "$SAVE" -vf "scale=192:-1,tile=6x6" "$MONT" || true
echo "== done =="
echo "   video   : $SAVE"
echo "   montage : $MONT"
