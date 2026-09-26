#!/bin/bash
# Tier-1 baselines on the held-out clips, then start the v8 run under supervisor.
# Training starts even if a baseline fails - baselines can be re-run at the first gate
# break, and an idle A100 is the expensive outcome. The failure is logged loudly.
cd /workspace/Pudgy/finetune/wan/eval_v8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "== baselines start $(date -u +%FT%TZ)"
if /workspace/Pudgy/.venv-wan/bin/python dcls_v8.py --configs golden v7expr v7motion v7fixed v7naive --k 2; then
  echo "== baselines OK $(date -u +%FT%TZ)"
else
  echo "!! BASELINES FAILED (exit $?) $(date -u +%FT%TZ) - starting training anyway; re-run at the step-100 gate"
fi
supervisorctl start pudgy-v8-monitor pudgy-v8-train
echo "== training started $(date -u +%FT%TZ)"
