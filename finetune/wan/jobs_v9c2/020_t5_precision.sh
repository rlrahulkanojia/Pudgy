#!/bin/bash
# T5 precision at inference (RESEARCH_v9c_fixes 6b): v9c@200 + OLD wording (control C0old flips 12/24), T5 in fp32 vs fp8.
# 24 clips (8 starts x 3 seeds, 41 frames), ~6.7 h.
set -uo pipefail
W=/workspace; export HF_HOME=$W/.hf_home RUN_NAME=pudgy-v9c-phone-lownoise PUDGY_DATA=$W/Data
cd $W/Pudgy/finetune/wan/eval_v9
$W/Pudgy/.venv-wan/bin/python arms_v9c2.py --tier confirm --arms C0oldT5
echo "result: $(tail -1 $W/eval_v9/runs/pudgy-v9c-phone-lownoise/arms_v9c2/arms.log) | control C0old 12/24"
