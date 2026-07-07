#!/bin/bash
cd /workspace/Pudgy; PY=.venv/bin/python; export HF_HOME=/workspace/.hf_home; DS=/workspace/training_dataset
RUNS=(
  "132 0.4 5.0 25 champ_ckpt132_s040_g50_f25"
  "135 0.4 5.0 33 champ_ckpt135_s040_g50_f33"
)
for r in "${RUNS[@]}"; do
  set -- $r; ck=$1; sc=$2; g=$3; fr=$4; lb=$5
  echo "[$(date -u +%H:%M:%S)] $lb"
  $PY inference/eval_pudgy_lora.py --checkpoint $ck --lora_scale $sc --guidance_scale $g \
      --num_frames $fr --dataset_dir "$DS" --cpu_offload --seed 42 \
      --output_path sweeps/${lb}.mp4 > sweeps/${lb}.log 2>&1 \
      && echo "  OK $lb" || echo "  FAIL $lb"
done
echo "CHAMPS DONE $(date -u +%H:%M:%S)"
