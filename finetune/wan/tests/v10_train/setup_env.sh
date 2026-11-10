#!/bin/bash
# Build the environment for finetune/wan/tests/v10_train (CPU only; macOS or Linux). Idempotent.
#   V10_TEST_WORK=<dir> bash finetune/wan/tests/v10_train/setup_env.sh     (default ~/.cache/pudgy_v10_tests)
# Creates in the work dir:
#   venv/   Python 3.11.10 (uv-managed), torch 2.7.1 + torchvision 0.22.1 CPU, requirements.txt
#   musubi/ kohya-ss/musubi-tuner at tag v0.3.4 (30c658c), read-only: tests import it, never write to it
#   hf/     HF cache with the umT5 tokenizer (Wan-AI/Wan2.1-T2V-14B, subfolder google/umt5-xxl), ~21 MB
#   tmp/    test temp files
# Needs: uv (https://docs.astral.sh/uv/), git, network access for the first run (~100 MB of wheels on
# macOS arm64; uv's cache makes rebuilds fast).
set -euo pipefail
TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WK="${V10_TEST_WORK:-$HOME/.cache/pudgy_v10_tests}"
PYV=3.11.10
MUSUBI_TAG=v0.3.4
MUSUBI_SHA=30c658c4f4b0bf05038b3346eff9670259b10fc7
die() { echo "setup_env: $*" >&2; exit 1; }
command -v uv >/dev/null || die "needs uv (https://docs.astral.sh/uv/)"
command -v git >/dev/null || die "needs git"
mkdir -p "$WK/tmp"
echo "== work dir $WK"

if [ ! -x "$WK/venv/bin/python" ]; then
  uv venv -q --python "$PYV" "$WK/venv"
fi
"$WK/venv/bin/python" -c "import sys; assert '%d.%d.%d' % sys.version_info[:3] == '$PYV', sys.version" \
  || die "$WK/venv is not Python $PYV - delete it and re-run"
TORCH_IDX=()
[ "$(uname -s)" = Darwin ] || TORCH_IDX=(--index-url https://download.pytorch.org/whl/cpu)   # Linux: CPU wheels
uv pip install -q --python "$WK/venv/bin/python" ${TORCH_IDX[@]+"${TORCH_IDX[@]}"} torch==2.7.1 torchvision==0.22.1
uv pip install -q --python "$WK/venv/bin/python" -r "$TESTS/requirements.txt"
echo "== venv: $("$WK/venv/bin/python" -V), torch $("$WK/venv/bin/python" -B -c 'import torch; print(torch.__version__)')"

if [ ! -d "$WK/musubi/.git" ]; then
  git clone -q --depth 1 --branch "$MUSUBI_TAG" https://github.com/kohya-ss/musubi-tuner "$WK/musubi" 2>/dev/null
fi
[ "$(git -C "$WK/musubi" rev-parse HEAD)" = "$MUSUBI_SHA" ] || die "$WK/musubi is not $MUSUBI_TAG ($MUSUBI_SHA)"
[ -z "$(git -C "$WK/musubi" status --porcelain --untracked-files=no)" ] || die "$WK/musubi has local changes; it must stay read-only"
echo "== musubi-tuner $MUSUBI_TAG at $(git -C "$WK/musubi" rev-parse --short HEAD)"

HF_HOME="$WK/hf" MUSUBI="$WK/musubi" "$WK/venv/bin/python" -I -B - <<'PYEOF'
import os, sys
import torch
torch.cuda.current_device = lambda: 0          # musubi's t5.py evaluates this at import
sys.path.insert(0, os.environ["MUSUBI"] + "/src")
from musubi_tuner.wan.modules.tokenizers import HuggingfaceTokenizer
tok = HuggingfaceTokenizer(name="Wan-AI/Wan2.1-T2V-14B", subfolder="google/umt5-xxl", seq_len=512, clean="whitespace")
ids, mask = tok(["the iFin phone"], return_mask=True, add_special_tokens=True)
print("== umT5 tokenizer cached in", os.environ["HF_HOME"], "- 'the iFin phone' ->", ids[0, :int(mask[0].sum())].tolist())
PYEOF
echo "== ready: bash $TESTS/run_all.sh"
