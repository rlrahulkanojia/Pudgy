#!/bin/bash
# Thin wrapper: source secrets from /workspace/.env (untracked) then upload.
# Usage: bash azure_upload.sh [weights] [logs] [output] [state|all]
#   default (no args) = weights logs output
set -euo pipefail
set -a; . /workspace/.env; set +a
exec /venv/main/bin/python /workspace/Pudgy/finetune/wan/azure_upload.py "$@"
