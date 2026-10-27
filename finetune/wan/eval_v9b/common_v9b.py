"""Paths and LoRA configurations for the v9b evaluation (Training_Approach_v9.md section 5;
runbook RUNBOOK_v9b.md section 6).

Reuses the v8/v9 tooling instead of copying it: the diffusion classifier (eval_v8/dcls_v8),
the v9c regression clip set (/workspace/eval_v9/holdout: v8 labels on the QF2_R / QF2_L
angles), and two-expert generation (eval_v8/gen_v8 settings). New here: the held-out SKIT
windows (build_eval_v9b.py) and the configurations below.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAN = HERE.parent
for p in (WAN, WAN / "eval_v8", WAN / "eval_v9", WAN / "assets", WAN / "scenes"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as C8  # noqa: E402  (eval_v8/common.py)
import common_v9 as C9  # noqa: E402

P = C8.P
WS = C8.WS
V8_MERGED = C9.V8_MERGED
RUN_NAME = os.environ.get("RUN_NAME", "pudgy-v9b-lownoise")
RUN = WS / "wan_output" / RUN_NAME
EVAL = Path(os.environ.get("V9B_EVAL", str(WS / "eval_v9b")))
SCENES = EVAL / "scenes"                 # held-out skit windows: clips, cache, index, suite jobs
REGRESSION = C9.HOLDOUT                  # /workspace/eval_v9/holdout (v8 labels, cached)
RESULTS = EVAL / "runs" / RUN_NAME       # dcls/, suite/, winner_v9b.json
# The v8merged regression rows were already scored for v9c with the same clips, noise draws,
# code and weights; tier1_v9b.py reuses them after re-scoring a spot-check subset.
V9C_BASE_DCLS = WS / "eval_v9/runs/pudgy-v9c-phone-lownoise/dcls/v8merged_k2.json"
SW, SH = 480, 848


def ckpt(step):
    if step in ("final", "last"):
        return RUN / f"{RUN_NAME}.safetensors"
    return RUN / f"{RUN_NAME}-step{int(step):08d}.safetensors"


def lora_config(name):
    """(low-noise [(path, mult)], high-noise [(path, mult)]).

    v8merged   the v9b training base (D1), = v8@550.
    v9b@<S>    the base + the v9b task vector at step S (or final).
    Anything else falls through to common_v9 (v9c@..., file:..., golden, ...).
    """
    hi = [(C8.GOLD_HIGH, 1.0)]
    if name == "v8merged":
        return [(V8_MERGED, 1.0)], hi
    if name.startswith("v9b@"):
        return [(V8_MERGED, 1.0), (ckpt(name[4:]), 1.0)], hi
    return C9.lora_config(name)
