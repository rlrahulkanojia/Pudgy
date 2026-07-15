#!/bin/bash
# =============================================================================
# TWO-EXPERT FLF2V eval — the full v2 pipeline: low-noise identity LoRA + DiT
# for timesteps 0-boundary, high-noise motion LoRA + DiT for boundary-1000.
# musubi maps: --lora_weight -> low DiT, --lora_weight_high_noise -> high DiT.
#
# Memory: two 14B DiTs don't fit fp16 on 80GB -> fp8_scaled + block-swap +
# lazy_loading (loads each DiT on demand; --offload_inactive_dit is incompatible
# with --blocks_to_swap, so we use --lazy_loading).
#
# Usage:
#   bash eval_flf2v_2expert.sh                          # golden low + final high
#   LOW=/path HIGH=/path SRC=00000001 bash eval_flf2v_2expert.sh
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC="${SRC:-00000001}"
STEPS="${STEPS:-25}"; SHIFT="${SHIFT:-5.0}"; GUID="${GUID:-5.0}"; BOUNDARY="${BOUNDARY:-0.9}"
LSCALE="${LSCALE:-1.0}"; HSCALE="${HSCALE:-1.0}"; BLKSWAP="${BLKSWAP:-25}"

REPO=/workspace/musubi-tuner
PY=/workspace/Pudgy/.venv-wan/bin/python
M=/workspace/wan_models
DIT_LOW="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
VAE="$M/comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5="$M/t5/models_t5_umt5-xxl-enc-bf16.pth"
# golden LoRAs (defaults): low = epoch40 final, high = epoch40 final
LOW="${LOW:-/workspace/wan_output/pudgy-wan22-a14b-lownoise/pudgy-wan22-a14b-lownoise.safetensors}"
HIGH="${HIGH:-/workspace/wan_output/pudgy-wan22-a14b-highnoise/pudgy-wan22-a14b-highnoise.safetensors}"
CLIP=/workspace/training_dataset/train/${SRC}.mp4
OUT=/workspace/wan_output/2expert_eval; mkdir -p "$OUT"

echo "== TWO-EXPERT FLF2V =="
echo "   low  LoRA: $(basename "$LOW")  (x$LSCALE)"
echo "   high LoRA: $(basename "$HIGH")  (x$HSCALE)  boundary=$BOUNDARY"

# start/end keyframes
START="$OUT/${SRC}_start.png"; END="$OUT/${SRC}_end.png"
NFR=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$CLIP")
ffmpeg -y -loglevel error -i "$CLIP" -vf "select=eq(n\,0)" -vframes 1 "$START"
ffmpeg -y -loglevel error -i "$CLIP" -vf "select=eq(n\,$((NFR-1)))" -vframes 1 "$END"
PROMPT=$($PY -c "import json;print(next(e['text'] for e in json.load(open('/workspace/training_dataset/metadata.json')) if e['file_path'].endswith('${SRC}.mp4')))")
TAG="2expert_low$(basename "$LOW" .safetensors | grep -oE '[0-9]+$' || echo F)_high$(basename "$HIGH" .safetensors | grep -oE '[0-9]+$' || echo F)"
SAVE="$OUT/${TAG}_${SRC}.mp4"

cd "$REPO"
$PY src/musubi_tuner/wan_generate_video.py \
  --task i2v-A14B --dit "$DIT_LOW" --dit_high_noise "$DIT_HIGH" --timestep_boundary "$BOUNDARY" \
  --vae "$VAE" --t5 "$T5" \
  --lora_weight "$LOW" --lora_multiplier "$LSCALE" \
  --lora_weight_high_noise "$HIGH" --lora_multiplier_high_noise "$HSCALE" \
  --video_size 768 1360 --video_length 33 --fps 16 \
  --infer_steps "$STEPS" --flow_shift "$SHIFT" --guidance_scale "$GUID" \
  --image_path "$START" --end_image_path "$END" --trim_tail_frames 3 \
  --prompt "$PROMPT" --seed 42 --attn_mode sdpa \
  --fp8 --fp8_scaled --fp8_t5 --vae_cache_cpu --blocks_to_swap "$BLKSWAP" --lazy_loading \
  --save_path "$SAVE" --output_type both

REAL=$(find "$SAVE" -type f -name "*.mp4" 2>/dev/null | head -1)
MONT="$OUT/${TAG}_${SRC}_montage.png"
[ -n "$REAL" ] && ffmpeg -y -loglevel error -i "$REAL" -vf "scale=192:-1,tile=6x6" "$MONT"
echo "== done =="; echo "   video : $REAL"; echo "   montage: $MONT"
