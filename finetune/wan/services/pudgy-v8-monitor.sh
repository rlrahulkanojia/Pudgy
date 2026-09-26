#!/bin/bash
# v8 monitor: status + CPU weight diagnostics + Azure mirror every 30 min.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/Pudgy
pty /workspace/Pudgy/.venv-wan/bin/python finetune/wan/monitor_v8.py 2>&1
