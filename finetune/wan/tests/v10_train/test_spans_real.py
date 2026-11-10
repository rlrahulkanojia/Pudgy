"""The phone-phrase token span (what E5 is spliced into) on REAL captions, with the real umT5 tokenizer:
found in exactly the captions whose text contains the registry phrase, never twice, and every
caption fits the 512-token T5 window.
  - v9b scene windows: finetune/wan/scenes/v9b_535_scene_windows.json (in the repo, always run)
  - v10 scene windows: $PUDGY_DATA/processed/v10_scenes_748/captions_v10_windows.jsonl (run when present;
    PUDGY_DATA defaults to <repo>/../../Data, the Mac layout)"""
import json
import os
from pathlib import Path

import common
from common import check
import asset_embed_v9 as AE

tok = AE.tokenizer()
ph = AE.phrase()
ph_ids = AE.token_ids(tok, ph)[:-1]
print(f"registry phrase: {len(ph_ids)} tokens")


def scan(name, caps):
    found = text = twice = 0
    mism, longest = [], 0
    for c in caps:
        ids = AE.token_ids(tok, c)
        longest = max(longest, len(ids))
        try:
            s, e = AE.find_span(ids, ph_ids)
        except ValueError:
            twice += 1
            continue
        found += e > s
        text += ph in c
        if (e > s) != (ph in c):
            mism.append(c[:120])
    check(not mism and not twice and found == text,
          f"{name}: {len(caps)} captions, phrase in text {text}, spans found {found}, mismatches {len(mism)}, twice {twice}")
    check(longest <= 512, f"{name}: longest caption {longest} tokens <= 512")


v9b = json.loads((common.FW / "scenes/v9b_535_scene_windows.json").read_text())["windows"]
scan("v9b_535 windows (repo)", [w["caption"] for w in v9b])
data = Path(os.environ.get("PUDGY_DATA") or common.REPO.parent.parent / "Data")
v10 = data / "processed/v10_scenes_748/captions_v10_windows.jsonl"
if v10.exists():
    scan("v10_scenes_748 windows", [json.loads(x)["caption"] for x in v10.read_text().splitlines() if x.strip()])
else:
    print(f"  SKIP v10 captions: {v10} not found (set PUDGY_DATA)")
common.finish()
