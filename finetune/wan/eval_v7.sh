#!/bin/bash
# =============================================================================
# v7 eval — one two-expert FLF2V generation at the trained geometry (1024x1024).
#
# Generalises eval_v6.sh on the ONE axis v7 needs: which expert carries the
# checkpoint under test. v6 could hardcode `low = CKPT, high = frozen v2 golden`
# because v6 trained low-noise only. v7 cannot:
#
#   * the motion run is an A/B (plan section 4) and the M-high arm puts the
#     checkpoint on the HIGH-noise expert;
#   * G-S loads TWO trained LoRAs at once, so neither side is the frozen golden.
#
# So both sides are explicit. Defaults keep the v6 behaviour exactly:
#
#   SIDE=low   (default) CKPT -> low-noise,  high = frozen v2 golden
#   SIDE=high            CKPT -> high-noise, low  = frozen v2 golden
#   SIDE=both            LOW_LORA and HIGH_LORA both given explicitly (G-S)
#
#   CKPT=<lora.safetensors> PROMPT="..." START=<png> FRAMES=21 SEED=42 \
#     [SIDE=low|high|both] [LOW_LORA=..] [HIGH_LORA=..] [END=<png>] [TAG=name] \
#     [OUTDIR=/workspace/eval_v7/out] bash eval_v7.sh
#
# Every gate before G-S runs the expert under test against the UNTOUCHED v2 golden
# partner, so a regression stays bisectable (plan section 4). That is why the
# default partner is the golden and not "whatever was trained last".
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROMPT="${PROMPT:?set PROMPT=...}"
START="${START:?set START=/path/to/start.png}"
SIDE="${SIDE:-low}"
FRAMES="${FRAMES:-21}"
SEED="${SEED:-42}"
END="${END:-}"
SIZE="${SIZE:-1024}"; FPS="${FPS:-24}"
STEPS="${STEPS:-25}"; SHIFT="${SHIFT:-5.0}"; GUID="${GUID:-5.0}"; BOUNDARY="${BOUNDARY:-0.9}"
LSCALE="${LSCALE:-1.0}"; HSCALE="${HSCALE:-1.0}"
BLKSWAP="${BLKSWAP:-0}"
OUTDIR="${OUTDIR:-/workspace/eval_v7/out}"

M=/workspace/wan_models
PY=/workspace/Pudgy/.venv-wan/bin/python
DIT_LOW="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH="$M/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
VAE="$M/comfy21/split_files/vae/wan_2.1_vae.safetensors"
T5="$M/t5/models_t5_umt5-xxl-enc-bf16.pth"
GOLD=/workspace/wan_output/v2_golden
GOLD_LOW="$GOLD/lora_lownoise_GOLDEN_ep40.safetensors"
GOLD_HIGH="$GOLD/lora_highnoise_GOLDEN_ep40.safetensors"

case "$SIDE" in
  low)  LOW_LORA="${LOW_LORA:-${CKPT:?set CKPT= or SIDE=both}}"; HIGH_LORA="${HIGH_LORA:-$GOLD_HIGH}" ;;
  high) HIGH_LORA="${HIGH_LORA:-${CKPT:?set CKPT= or SIDE=both}}"; LOW_LORA="${LOW_LORA:-$GOLD_LOW}"  ;;
  both) LOW_LORA="${LOW_LORA:?SIDE=both needs LOW_LORA=}"; HIGH_LORA="${HIGH_LORA:?SIDE=both needs HIGH_LORA=}" ;;
  *) echo "SIDE must be low|high|both"; exit 1 ;;
esac
for f in "$LOW_LORA" "$HIGH_LORA"; do
  [ -f "$f" ] || { echo "missing LoRA: $f"; exit 1; }
done

TAG="${TAG:-$(basename "${CKPT:-$LOW_LORA}" .safetensors)_f${FRAMES}_s${SEED}}"
mkdir -p "$OUTDIR"
# musubi ALWAYS treats --save_path as a directory (see eval_v6.sh) — give it a scratch
# dir and rename the result ourselves.
SAVE="$OUTDIR/.raw_${TAG}"
rm -rf "$SAVE"; mkdir -p "$SAVE"
FINAL="$OUTDIR/${TAG}.mp4"

END_ARG=(); [ -n "$END" ] && END_ARG=(--end_image_path "$END")
EXTRA=(); [ "$BLKSWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLKSWAP" --lazy_loading)

echo "== v7 eval: $TAG  (side=$SIDE) =="
echo "   low : $(basename "$LOW_LORA")"
echo "   high: $(basename "$HIGH_LORA")"
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
  --save_path "$SAVE" --output_type video

PRODUCED=$(find "$SAVE" -name "*.mp4" -type f | sort | tail -1)
if [ -z "$PRODUCED" ]; then
  echo "!! generation produced no mp4 in $SAVE"; ls -la "$SAVE"; exit 1
fi
mv "$PRODUCED" "$FINAL"
rm -rf "$SAVE"
echo "== wrote: $FINAL"
