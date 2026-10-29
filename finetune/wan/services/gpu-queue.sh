#!/bin/bash
# GPU job queue runner (finetune/wan/gpu_queue.sh) under supervisor.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/Pudgy
pty bash finetune/wan/gpu_queue.sh 2>&1
