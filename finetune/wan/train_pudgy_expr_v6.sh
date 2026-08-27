#!/bin/bash
# =============================================================================
# v6 — multi-expression LoRA, continue-trained from the v2 low-noise golden.
#
# Supersedes running v6 through train_pudgy_happy_expr.sh with DATASET= overridden.
# That worked, but every artefact it produced was named `pudgy-happy-expr-*`, which
# is actively wrong here: v6 is a distinct experiment on all four emotions and both
# characters, not the 7-clip Pax/happy pilot. Mis-named weights are how a checkpoint
# gets misattributed six months later.
#
#   bash train_pudgy_expr_v6.sh
#   LR=5e-5 EPOCHS=14 bash train_pudgy_expr_v6.sh
#
# What is NOT a knob, and why (Training_Approach_v6 section 5.1):
#   * expert   — low-noise only. v5 section 4.6 A/B settled it: continue-training the
#                high-noise expert destroyed prompt control (opposite prompts gave
#                SSIM 0.9692, i.e. identical video). The high-noise golden stays frozen
#                and is loaded unchanged at inference.
#   * rank/alpha — 16/32, fixed. The init IS rank 16; rank-8 tensors cannot load it.
# =============================================================================
set -euo pipefail

EXPERT=low                              # NOT a knob — see header
RANK="${RANK:-16}"                      # must match the init checkpoint
ALPHA="${ALPHA:-32}"
LR="${LR:-3e-5}"                        # 1e-4 would wreck the prior; 5e-5 = documented fallback
EPOCHS="${EPOCHS:-11}"                  # ~272 steps/epoch -> ~3000 steps, the v2 golden band
SAVE_EVERY="${SAVE_EVERY:-1}"           # every epoch: the golden is often early (v5 4.4)
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"       # 0 = off. In-process sampling loads T5+VAE on top
                                        # of a ~60GB DiT and can OOM-kill the run; the
                                        # gates evaluate properly afterwards instead.
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-32}"  # MEASURED on this box, and it contradicts the plan.
                                        # The plan says 0, citing v5 4.7 (block-swap is a 2x
                                        # throughput tax when VRAM is free) — but v5 only ever
                                        # trained 21-frame clips. v6 adds a 57-frame bucket at
                                        # 61,440 tokens, 2.5x the sequence length, and it OOMs
                                        # at 0 (peak 76.4/79.2 GB, dying in RMSNorm's fp32
                                        # upcast). Probed on the f57 bucket:
                                        #   bs=0  -> OOM
                                        #   bs=24 -> 76.1 GB peak (93%), 72.7 s/it
                                        #   bs=32 -> 70.7 GB peak (86%), 74.3 s/it  <- chosen
                                        # 32 costs ~2% speed for 2x the headroom; an OOM 20 h
                                        # in costs a whole epoch. Short buckets are unaffected
                                        # in practice: f21 runs 18.5 s/it here, still faster
                                        # than v5's 26.7 s/it baseline.
FP8_BASE="${FP8_BASE:-0}"
SEED="${SEED:-42}"                      # comparability with v2/v5
LOG_WITH="${LOG_WITH:-all}"             # wandb + tensorboard
WANDB_PROJECT="${WANDB_PROJECT:-pudgy}" # same project as v2/v5 -> runs are comparable
NAME="${NAME:-pudgy-expr-v6-lownoise}"
MIRROR="${MIRROR:-1}"                   # mirror checkpoints to Azure DURING the run

REPO=/workspace/musubi-tuner
ACC=/workspace/Pudgy/.venv-wan/bin/accelerate
PY=/workspace/Pudgy/.venv-wan/bin/python
DIT_DIR=/workspace/wan_models/comfy22/split_files/diffusion_models
GOLDEN=/workspace/wan_output/v2_golden
DATASET="${DATASET:-/workspace/data_v6/dataset_config_expressions_v6.workspace.toml}"

DIT="$DIT_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"
MIN_TS=0; MAX_TS=900                    # I2V low-noise range (musubi docs/wan.md)
NETWORK_WEIGHTS="${NETWORK_WEIGHTS:-$GOLDEN/lora_lownoise_GOLDEN_ep40.safetensors}"

# Load untracked secrets (WANDB_API_KEY, AZURE_*). Never committed.
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }

OUT=/workspace/wan_output/$NAME
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- preflight: fail before burning 40 h, not after -----------------------------
for f in "$DIT" "$NETWORK_WEIGHTS" "$DATASET" "$GOLDEN/lora_highnoise_GOLDEN_ep40.safetensors"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
"$PY" - "$NETWORK_WEIGHTS" "$RANK" <<'PY'
import sys
from safetensors.torch import safe_open
p, want = sys.argv[1], int(sys.argv[2])
with safe_open(p, framework="pt") as f:
    downs = [k for k in f.keys() if k.endswith("lora_down.weight")]
    r = f.get_slice(downs[0]).get_shape()[0]
if r != want:
    sys.exit(f"init is rank {r} but RANK={want} — the checkpoint cannot load "
             f"(v5 section 3 hit exactly this)")
print(f"   init verified: {len(downs)} modules, rank {r}")
PY

EXTRA=()
[ "$FP8_BASE" = "1" ] && EXTRA+=(--fp8_base)
[ "$BLOCKS_TO_SWAP" -gt 0 ] && EXTRA+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
EXTRA+=(--network_weights "$NETWORK_WEIGHTS")

if [ "$SAMPLE_EVERY" -gt 0 ] && [ -f /workspace/Pudgy/finetune/wan/sample_prompts.txt ]; then
  EXTRA+=(--vae /workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors \
          --t5 /workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth \
          --sample_prompts /workspace/Pudgy/finetune/wan/sample_prompts.txt \
          --sample_every_n_epochs "$SAMPLE_EVERY")
fi

if [ "$LOG_WITH" != "none" ]; then
  EXTRA+=(--log_with "$LOG_WITH" --logging_dir "$OUT/logs" --log_config
          --log_tracker_name "$WANDB_PROJECT" --wandb_run_name "$NAME")
  [ -n "${WANDB_API_KEY:-}" ] && EXTRA+=(--wandb_api_key "$WANDB_API_KEY")
fi

# --- mirror to Azure while training, not after ----------------------------------
# Risk 9.8: the v5 box was destroyed WITH its weights on it, and `/workspace` here is
# not a persistent volume either. Uploading only at the end means a 40 h run is one
# preemption away from total loss.
MIRROR_PID=""
if [ "$MIRROR" = "1" ] && [ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
  "$PY" /workspace/Pudgy/finetune/wan/azure_mirror_v6.py --watch "$OUT" --prefix "v6" \
        > "$OUT/mirror.log" 2>&1 &
  MIRROR_PID=$!
  echo "   mirroring to Azure v6/ (pid $MIRROR_PID)"
  trap '[ -n "$MIRROR_PID" ] && kill $MIRROR_PID 2>/dev/null || true' EXIT
elif [ "$MIRROR" = "1" ]; then
  echo "   !! MIRROR=1 but AZURE_STORAGE_CONNECTION_STRING is unset — NOT mirroring"
fi

echo "== v6 multi-expression LoRA =="
echo "   dit      : $(basename "$DIT")  (low-noise expert)"
echo "   init     : $(basename "$NETWORK_WEIGHTS")"
echo "   frozen   : lora_highnoise_GOLDEN_ep40.safetensors (inference partner, untouched)"
echo "   dataset  : $DATASET"
echo "   rank/α   : $RANK/$ALPHA   lr: $LR   epochs: $EPOCHS   ts: [$MIN_TS,$MAX_TS]"
echo "   output   : $OUT"

cd "$REPO"
"$ACC" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-A14B \
  --dit "$DIT" \
  --dataset_config "$DATASET" \
  --sdpa --mixed_precision fp16 \
  --network_module networks.lora_wan \
  --network_dim "$RANK" --network_alpha "$ALPHA" \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep "$MIN_TS" --max_timestep "$MAX_TS" --preserve_distribution_shape \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --max_train_epochs "$EPOCHS" --save_every_n_epochs "$SAVE_EVERY" \
  --save_state \
  --seed "$SEED" \
  --output_dir "$OUT" --output_name "$NAME" \
  "${EXTRA[@]}"

echo "== done: $OUT =="
# Final sweep so the last checkpoint and logs are definitely up.
[ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ] && \
  "$PY" /workspace/Pudgy/finetune/wan/azure_mirror_v6.py --once "$OUT" --prefix "v6" || true
echo "next: gates at ~epoch 2 —"
echo "  python finetune/wan/gates_v6.py --ckpt $OUT/${NAME}-000002.safetensors --gates ep2"
