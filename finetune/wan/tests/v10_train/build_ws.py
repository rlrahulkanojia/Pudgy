"""Build a fake /workspace for test_train_sh.sh: fixtures.make_ws plus stand-in DiT / golden / E5 files.
    python build_ws.py <root>   -> prints the dataset toml path"""
import os
import sys
from pathlib import Path

import common  # noqa: F401
from fixtures import make_ws

root = Path(sys.argv[1])
cfg, cache, e5 = make_ws(root)
for p in ("wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
          "wan_output/v2_golden/lora_highnoise_GOLDEN_ep40.safetensors"):
    (root / p).parent.mkdir(parents=True, exist_ok=True)
    (root / p).write_bytes(b"x")
e = root / "wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors"
e.parent.mkdir(parents=True, exist_ok=True)
os.replace(e5, e)
print(cfg)
