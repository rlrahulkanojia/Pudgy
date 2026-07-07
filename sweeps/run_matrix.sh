#!/bin/bash
set -u
cd /workspace/Pudgy
PY=.venv/bin/python
export HF_HOME=/workspace/.hf_home
DS=/workspace/training_dataset
# columns: ckpt scale guidance frames  label
RUNS=(
  "128 0.5 6.0 33 ckpt128_s050_g60_f33"
  "132 0.5 6.0 33 ckpt132_s050_g60_f33"
  "135 0.5 6.0 33 ckpt135_s050_g60_f33"
  "139 0.4 6.0 33 ckpt139_s040_g60_f33"
  "139 0.3 6.0 33 ckpt139_s030_g60_f33"
  "132 0.4 5.0 33 ckpt132_s040_g50_f33"
  "135 0.35 5.0 33 ckpt135_s035_g50_f33"
  "135 0.4 6.0 25 ckpt135_s040_g60_f25"
)
i=0; total=${#RUNS[@]}
for r in "${RUNS[@]}"; do
  set -- $r; ckpt=$1; scale=$2; guid=$3; frames=$4; label=$5
  i=$((i+1))
  echo "=============================================================="
  echo "[$i/$total] $label  (ckpt=$ckpt scale=$scale guidance=$guid frames=$frames)"
  echo "started: $(date -u +%H:%M:%S)"
  $PY inference/eval_pudgy_lora.py \
      --checkpoint $ckpt --lora_scale $scale \
      --guidance_scale $guid --num_frames $frames \
      --dataset_dir "$DS" --cpu_offload --seed 42 \
      --output_path sweeps/${label}.mp4 \
      > sweeps/${label}.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f sweeps/${label}.mp4 ]; then
    echo "[$i/$total] OK  $(du -h sweeps/${label}.mp4 | cut -f1)  $(date -u +%H:%M:%S)"
  else
    echo "[$i/$total] FAILED rc=$rc  (see sweeps/${label}.log)"; tail -3 sweeps/${label}.log
  fi
done
echo "=============================================================="
echo "ALL DONE $(date -u +%H:%M:%S)"
ls -la sweeps/*.mp4
