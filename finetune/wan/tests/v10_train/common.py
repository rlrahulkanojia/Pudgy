"""Shared setup for the v10 training-side tests (CPU only, no GPU).

Paths: REPO is derived from this file's location; the work dir (venv, musubi clone, HF cache, temp
files) is $V10_TEST_WORK, default ~/.cache/pudgy_v10_tests (built by setup_env.sh). Import this
module FIRST in every test: it patches torch.cuda.current_device, which musubi's wan/modules/t5.py
evaluates at import time, and sets the env defaults before musubi / transformers are imported.
"""
import os
import sys
from pathlib import Path

import torch

torch.cuda.current_device = lambda: 0

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("REPO") or HERE.parents[3])
WORK = Path(os.environ.get("V10_TEST_WORK") or Path.home() / ".cache/pudgy_v10_tests")
MUSUBI = Path(os.environ.setdefault("MUSUBI", str(WORK / "musubi")))
os.environ.setdefault("HF_HOME", str(WORK / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ASSET_DEVICE", "cpu")
TMP = Path(os.environ.get("TEST_TMP") or WORK / "tmp")
TMP.mkdir(parents=True, exist_ok=True)
FW = REPO / "finetune/wan"
sys.path.insert(0, str(MUSUBI / "src"))
sys.path.insert(0, str(FW))

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def finish():
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)
