"""Paths and LoRA configurations for the v9c-phone evaluation.

Reuses the v8 evaluation tooling (eval_v8/: diffusion classifier, batched generation)
instead of copying it; only the model configurations and the clip set are new. Captions
come from prep_v8 / prep_v9 / the asset registry - never retyped here.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAN = HERE.parent
for p in (WAN, WAN / "eval_v8", WAN / "assets"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as C8  # noqa: E402  (eval_v8/common.py)
import prep_v9 as P9  # noqa: E402

P = C8.P
WS = C8.WS
V8_MERGED = WS / "wan_output/v8_merged/pudgy-v8-merged.safetensors"
RUN_NAME = os.environ.get("RUN_NAME", "pudgy-v9c-phone-lownoise")   # v9_box.sh exports it
RUN = WS / "wan_output" / RUN_NAME
EVAL = Path(os.environ.get("V9_EVAL", str(WS / "eval_v9")))   # override for a local dry build
HOLDOUT = EVAL / "holdout"             # shared by every run: clip set + latent cache
KF = EVAL / "keyframes"
RESULTS = EVAL / "runs" / RUN_NAME     # per run: dcls/, suites/, winner_v9c.json

# Distractor objects for the binding test: the same caption with the prop phrase swapped.
# After training, the registry phrase should explain a held-out phone clip better than
# a generic phone and than other hand-held objects.
BINDING = {
    "phone": None,                                   # the registry phrase (true)
    "generic_phone": "a phone",
    "book": "a small red book",
    "cup": "a white cup",
    "remote": "a black TV remote control",
}


def ckpt(step):
    if step in ("final", "last"):
        return RUN / f"{RUN_NAME}.safetensors"
    return RUN / f"{RUN_NAME}-step{int(step):08d}.safetensors"


def lora_config(name):
    """(low-noise [(path, mult)], high-noise [(path, mult)]) for a named configuration.

    v8merged   the v9c training base: v7fixed + v8@550 as one exact file. Must score the
               same as v8@550 (checked by `v9_box.sh baseline`).
    v8@550     the same sum as two files, as v8 shipped it.
    v9c@<S>    the base + the v9c task vector at step S (or final).
    file:<p>   the base + any v9-style task vector.
    Anything else falls through to eval_v8 (golden, v7fixed, ...).
    """
    hi = [(C8.GOLD_HIGH, 1.0)]
    if name == "v8merged":
        return [(V8_MERGED, 1.0)], hi
    if name.startswith("v9c@"):
        return [(V8_MERGED, 1.0), (ckpt(name[4:]), 1.0)], hi
    if name.startswith("file:"):
        return [(V8_MERGED, 1.0), (Path(name[5:]), 1.0)], hi
    return C8.lora_config(name)


def phone_caption(job, prop_phrase=None, zoom=1.00, bg="white"):
    """The training caption of a phone job, optionally with the prop phrase swapped."""
    if prop_phrase is None:                       # byte-identical to the training caption
        return P9.prop_caption(job, P.BACKGROUNDS[bg][0], zoom)
    act = job["act"]
    action = act["action"].format(phrase=prop_phrase, name=prop_phrase)
    return (f"{P.STYLE}{P9.anchor(job['char'])}, {action}; {act['label_text']}; "
            f"{P.ZOOMS[zoom]}, eye level, {job['clause']}; {P.BACKGROUNDS[bg][0]}.")
