#!/bin/bash
# =============================================================================
# v6 eval — one two-expert FLF2V generation at the trained geometry (1024x1024).
#
#   low-noise  = the v6 checkpoint under test   (v6 trains ONLY this expert)
#   high-noise = the v2 golden motion LoRA      (frozen — never trained in v6)
#
# This is deliberately a single-generation primitive, not a gate. The gates are
# matrices over (emotion x character x length x shot size x seed) and they live in
# gates_v6.py, which builds the prompts — sharing the caption constants with
# prep_expressions_v6.py so eval prompts sit in the training distribution by
# construction rather than by careful retyping.
#
#   CKPT=<lora.safetensors> PROMPT="..." START=<png> FRAMES=21 SEED=42 \
#     [END=<png>] [TAG=name] [OUTDIR=/workspace/eval_v6/out] bash eval_v6.sh
#
# Differences from eval_happy_v5.sh, both deliberate:
#   * EXPERT is not a knob. v5 had to switch it because it trained both experts in
#     an A/B; v6 settled that (v5 section 4.6) and trains low-noise only.
#   * block-swap defaults to 0. v5 measured it as a 2x throughput tax when VRAM is
#     free (27.5 -> 13.7 s/it); it is for memory pressure only. Raise it if a run
#     is training concurrently on the same card.
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:?set CKPT=/path/to/v6_checkpoint.safetensors}"
PROMPT="${PROMPT:?set PROMPT=...}"
START="${START:?set START=/path/to/start.png}"
FRAMES="${FRAMES:-21}"
SEED="${SEED:-42}"
END="${END:-}"
SIZE="${SIZE:-1024}"; FPS="${FPS:-24}"
STEPS="${STEPS:-25}"; SHIFT="${SHIFT:-5.0}"; GUID="${GUID:-5.0}"; BOUNDARY="${BOUNDARY:-0.9}"
LSCALE="${LSCALE:-1.0}"; HSCALE="${HSCALE:-1.0}"
BLKSWAP="${BLKSWAP:-0}"
OUTDIR="${OUTDIR:-/workspace/eval_v6/out}"

M=/workspace/wan_models
PY=/workspace/Pudgy/.venv-wan/bin/python
DIT_LOW="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
VAE="$M/comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5="$M/t5/models_t5_umt5-xxl-enc-bf16.pth"
GOLD=/workspace/wan_output/v2_golden

LOW_LORA="$CKPT"
HIGH_LORA="$GOLD/lora_highnoise_GOLDEN_ep40.safetensors"
[ -f "$HIGH_LORA" ] || { echo "missing frozen v2 high-noise golden: $HIGH_LORA"; exit 1; }

TAG="${TAG:-$(basename "$CKPT" .safetensors)_f${FRAMES}_s${SEED}}"
mkdir -p "$OUTDIR"
SAVE="$OUTDIR/${TAG}.mp4"

END_ARG=()
[ -n "$END" ] && END_ARG=(--end_image_path "$END")

EXTRA=()
[ "$BLKSWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLKSWAP" --lazy_loading)

echo "== v6 eval: $TAG =="
echo "   low (under test): $(basename "$CKPT")"
echo "   high (frozen v2): $(basename "$HIGH_LORA")"
echo "   start: $(basename "$START")  frames: $FRAMES  seed: $SEED"

cd /workspace/musubi-tuner
"$PY" src/musubi_tuner/wan_generate_video.py \
  --task i2v-A14B --dit "$DIT_LOW" --dit_high_noise "$DIT_HIGH" --timestep_boundary "$BOUNDARY" \
  --vae "$VAE" --t5 "$T5" \
  --lora_weight "$LOW_LORA" --lora_multiplier "$LSCALE" \
  --lora_weight_high_noise "$HIGH_LORA" --lora_multiplier_high_noise "$HSCALE" \
  --video_size "$SIZE" "$SIZE" --video_length "$FRAMES" --fps "$FPS" \
  --infer_steps "$STEPS" --flow_shift "$SHIFT" --guidance_scale "$GUID" \
  --image_path "$START" "${END_ARG[@]}" \
  --prompt "$PROMPT" --seed "$SEED" --attn_mode sdpa \
  --fp8 --fp8_scaled --fp8_t5 --vae_cache_cpu \
  "${EXTRA[@]}" \
  --save_path "$SAVE" --output_type both

# musubi appends its own timestamp/suffix; resolve what it actually wrote.
REAL=$(find "$OUTDIR" -name "*.mp4" -newer "$START" 2>/dev/null | sort | tail -1)
REAL="${REAL:-$SAVE}"
echo "== wrote: $REAL"
