#!/bin/bash
# 10 showcase clips from the v5 GOLDEN (run 2, low-noise happy) + the untouched
# v2 high-noise motion golden. Varies angle, background, seed, prompt and length.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/workspace/Pudgy/.venv-wan/bin/python
M=/workspace/wan_models
LOW=/workspace/wan_output/pudgy-happy-expr-lownoise-v1/pudgy-happy-expr-lownoise-v1.safetensors
HIGH=/workspace/wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors
KF=/workspace/eval_v5/showcase_kf
OUT=/workspace/eval_v5/showcase; mkdir -p "$OUT"

STYLE="A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing Pax, a short round blue penguin"
HAPPY="breaking into a big happy smile, beak opening into a wide joyful grin, eyebrows lifting, cheeks lifting; happy expression"
NEUTRAL="standing calmly with a gentle bouncy idle motion, looking around; neutral calm expression, beak closed"

# name|startframe|prompt_kind|seed|frames|scene_clause
JOBS=(
"01_front_white_happy|$KF/FRONT_white.png|happy|42|21|static close-up shot, eye level, facing the camera directly, front view; plain white studio background."
"02_front_lavender_happy|$KF/FRONT_lavender.png|happy|42|21|static close-up shot, eye level, front view; plain pastel lavender background."
"03_qfl_blue_happy|$KF/QF_L_blue.png|happy|7|21|static close-up shot, eye level, turned slightly to its left, three-quarter front view; plain pastel blue background."
"04_qfr_peach_happy|$KF/QF_R_peach.png|happy|123|21|static close-up shot, eye level, turned slightly to its right, three-quarter front view; plain pastel peach background."
"05_side_mint_happy|$KF/SIDE_L_mint.png|happy|42|21|static close-up shot, eye level, seen from its left side, profile view; plain pastel mint background."
"06_qf2r_sky_happy|$KF/QF2_R_sky.png|happy|42|21|static close-up shot, eye level, turned further to its right, wide three-quarter view; plain pastel sky-blue background."
"07_right_white_happy|$KF/Right_white.png|happy|7|21|static close-up shot, eye level, seen from its right side, profile view; plain white studio background."
"08_skit_scene_happy|/workspace/eval_v5/keyframes/v1_skit_pose.png|happy|42|21|standing in a tiled room; static medium shot, eye level; light blue tiled wall with a hanging lamp."
"09_skit_scene_neutral|/workspace/eval_v5/keyframes/v1_skit_pose.png|neutral|42|21|standing in a tiled room; static medium shot, eye level; light blue tiled wall with a hanging lamp."
"10_front_white_happy_33f|$KF/FRONT_white.png|happy|42|33|static close-up shot, eye level, facing the camera directly, front view; plain white studio background."
)

for J in "${JOBS[@]}"; do
  IFS='|' read -r NAME START KIND SEED FRAMES SCENE <<< "$J"
  [ "$KIND" = "happy" ] && BODY="$HAPPY" || BODY="$NEUTRAL"
  PROMPT="$STYLE, $BODY; $SCENE"
  echo "### $NAME  (seed $SEED, ${FRAMES}f)"
  $PY /workspace/musubi-tuner/src/musubi_tuner/wan_generate_video.py \
    --task i2v-A14B \
    --dit  "$M/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    --dit_high_noise "$M/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    --timestep_boundary 0.9 \
    --vae "$M/comfy21/split_files/vae/wan_2.1_vae.safetensors" \
    --t5  "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" \
    --lora_weight "$LOW" --lora_multiplier 1.0 \
    --lora_weight_high_noise "$HIGH" --lora_multiplier_high_noise 1.0 \
    --video_size 1024 1024 --video_length "$FRAMES" --fps 24 \
    --infer_steps 25 --flow_shift 5.0 --guidance_scale 5.0 \
    --image_path "$START" --prompt "$PROMPT" --seed "$SEED" --attn_mode sdpa \
    --fp8 --fp8_scaled --fp8_t5 --vae_cache_cpu --blocks_to_swap 20 --lazy_loading \
    --save_path "$OUT/$NAME" --output_type video || echo "FAILED $NAME"
done
echo "### SHOWCASE DONE"
