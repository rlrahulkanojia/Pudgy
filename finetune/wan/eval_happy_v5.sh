#!/bin/bash
# =============================================================================
# v5 pilot eval — two-expert FLF2V at the trained geometry (1024x1024x21).
#   low-noise  = v2 golden identity LoRA (UNCHANGED — never trained in run 1)
#   high-noise = the v5 happy checkpoint under test
#
#   CKPT=<path> MODE=indist|generalise|regress bash eval_happy_v5.sh
#
# Memory-frugal by default: run 2 may still be training and holding ~54GB, so
# fp8_scaled + fp8_t5 + max block-swap + lazy loading keeps the resident set small.
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${MODE:-indist}"
CKPT="${CKPT:?set CKPT=/path/to/checkpoint.safetensors}"
STEPS="${STEPS:-25}"; SHIFT="${SHIFT:-5.0}"; GUID="${GUID:-5.0}"; BOUNDARY="${BOUNDARY:-0.9}"
LSCALE="${LSCALE:-1.0}"; HSCALE="${HSCALE:-1.0}"
FRAMES="${FRAMES:-21}"; SIZE="${SIZE:-1024}"; FPS="${FPS:-24}"
BLKSWAP="${BLKSWAP:-39}"; SEED="${SEED:-42}"

M=/workspace/wan_models
PY=/workspace/Pudgy/.venv-wan/bin/python
DIT_LOW="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
VAE="$M/comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5="$M/t5/models_t5_umt5-xxl-enc-bf16.pth"
GOLD=/workspace/wan_output/v2_golden
# EXPERT selects which slot the checkpoint under test occupies; the OTHER slot
# gets the untouched v2 golden. run 1 trained high-noise, run 2 trained low-noise.
EXPERT="${EXPERT:-high}"
if [ "$EXPERT" = "low" ]; then
  LOW_LORA="$CKPT";                                   HIGH_LORA="$GOLD/lora_highnoise_GOLDEN_ep40.safetensors"
else
  LOW_LORA="$GOLD/lora_lownoise_GOLDEN_ep40.safetensors"; HIGH_LORA="$CKPT"
fi
KF=/workspace/eval_v5/keyframes
OUT=/workspace/eval_v5/out; mkdir -p "$OUT"

STYLE="A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing Pax, a short round blue penguin"
HAPPY="breaking into a big happy smile, beak opening into a wide joyful grin, eyebrows lifting, cheeks lifting; happy expression"
CAM="static close-up shot, eye level, facing the camera directly, front view"

case "$MODE" in
  indist)      # should look at least as good as training data
    START="$KF/indist_start.png"; END_ARG=(--end_image_path "$KF/indist_end.png")
    PROMPT="$STYLE, $HAPPY; $CAM; plain white studio background." ;;
  generalise)  # background the LoRA has never seen -> did it learn "happy" or "this background"?
    START="$KF/unseen_lavender.png"; END_ARG=()
    PROMPT="$STYLE, $HAPPY; $CAM; plain pastel lavender background." ;;
  regress)     # NON-happy prompt: did continue-training forget general behaviour?
    START="$KF/indist_start.png"; END_ARG=()
    PROMPT="$STYLE, standing still and turning its head slowly to look to one side, gentle bouncy idle motion; neutral expression; $CAM; plain white studio background." ;;
  regress2)    # DIFFERENT pose/scene (v1 skit frame) + non-happy prompt.
               # Separates "this frame -> happy" from "always happy".
    START="$KF/v1_skit_pose.png"; END_ARG=()
    PROMPT="$STYLE, standing in a tiled room and turning its head slowly to look to one side, gentle bouncy idle motion; neutral calm expression, beak closed; static medium shot, eye level; light blue tiled wall with a hanging lamp." ;;
  ctrl_happy)  # SAME novel frame as regress2, but WITH the happy prompt.
               # regress2 + ctrl_happy = clean A/B: only the prompt differs.
    START="$KF/v1_skit_pose.png"; END_ARG=()
    PROMPT="$STYLE, $HAPPY; standing in a tiled room; static medium shot, eye level; light blue tiled wall with a hanging lamp." ;;
  *) echo "MODE must be indist|generalise|regress|regress2|ctrl_happy"; exit 1 ;;
esac

TAG="$(basename "$CKPT" .safetensors)_${EXPERT}_${MODE}_s${SEED}_h${HSCALE}"
SAVE="$OUT/${TAG}.mp4"
echo "== eval $MODE =="; echo "   high LoRA: $(basename "$CKPT")"; echo "   start    : $(basename "$START")"

cd /workspace/musubi-tuner
$PY src/musubi_tuner/wan_generate_video.py \
  --task i2v-A14B --dit "$DIT_LOW" --dit_high_noise "$DIT_HIGH" --timestep_boundary "$BOUNDARY" \
  --vae "$VAE" --t5 "$T5" \
  --lora_weight "$LOW_LORA" --lora_multiplier "$LSCALE" \
  --lora_weight_high_noise "$HIGH_LORA" --lora_multiplier_high_noise "$HSCALE" \
  --video_size "$SIZE" "$SIZE" --video_length "$FRAMES" --fps "$FPS" \
  --infer_steps "$STEPS" --flow_shift "$SHIFT" --guidance_scale "$GUID" \
  --image_path "$START" "${END_ARG[@]}" \
  --prompt "$PROMPT" --seed "$SEED" --attn_mode sdpa \
  --fp8 --fp8_scaled --fp8_t5 --vae_cache_cpu --blocks_to_swap "$BLKSWAP" --lazy_loading \
  --save_path "$SAVE" --output_type both

REAL=$(find "$OUT" -name "*.mp4" -newer "$CKPT" 2>/dev/null | head -1); REAL="${REAL:-$SAVE}"
ffmpeg -y -loglevel error -i "$REAL" -vf "scale=200:-1,tile=7x3" "$OUT/${TAG}_montage.png" || true
echo "== done: $REAL"
