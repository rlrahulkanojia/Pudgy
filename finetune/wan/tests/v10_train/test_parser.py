"""train_pudgy_v10.sh's exact musubi argument list, parsed by musubi v0.3.4's own parser, and the
wrapper flags parsed by asset_embed_v9.py's argparse (DRY_RUN; the cache check is stubbed out)."""
import os
import shlex
import subprocess
import sys
import tempfile

import common
from common import check
import musubi_tuner.wan_train_network as W

work = tempfile.mkdtemp(prefix="parser_", dir=common.TMP)
true = "/usr/bin/true" if os.path.exists("/usr/bin/true") else "/bin/true"
env = dict(os.environ, W=work, DRY_RUN="1", PY=true, ACC="accelerate", FW=str(common.FW))
for k in ("DIT", "GOLDEN", "E5", "DATASET"):        # must exist and be non-empty; never read in a dry run
    env[k] = __file__
out = subprocess.run(["bash", str(common.FW / "train_pudgy_v10.sh")], env=env, capture_output=True, text=True)
check("== DRY_RUN" in out.stdout, "dry run prints the command" + ("" if "== DRY_RUN" in out.stdout else ": " + out.stderr[-200:]))
argv = shlex.split(out.stdout.split("== DRY_RUN: would run in")[1].splitlines()[1])
wrapper, musubi = argv[argv.index("train") + 1: argv.index("--")], argv[argv.index("--") + 1:]

p = W.wan_setup_parser(W.setup_parser_common())
a, unknown = p.parse_known_args(musubi)
check(not unknown, f"musubi parser knows every flag (unknown: {unknown})")
a = W.read_config_from_file(a, p)
want = dict(task="i2v-A14B", dit_high_noise=None, base_weights_multiplier=[1.0], network_dim=16, network_alpha=16.0,
            network_args=None, timestep_sampling="shift", discrete_flow_shift=5.0, min_timestep=900, max_timestep=1000,
            preserve_distribution_shape=True, optimizer_type="adamw8bit", learning_rate=3e-5,
            lr_scheduler="cosine_with_min_lr", lr_warmup_steps=20, lr_scheduler_min_lr_ratio=0.1,
            gradient_accumulation_steps=8, max_train_steps=300, save_every_n_steps=25, save_state=True, seed=42,
            blocks_to_swap=24, mixed_precision="fp16", gradient_checkpointing=True, sdpa=True,
            network_module="musubi_tuner.networks.lora_wan", output_name="pudgy-v10-highnoise", resume=None,
            sample_prompts=None)
bad = {k: (getattr(a, k), v) for k, v in want.items() if getattr(a, k) != v}
check(not bad, f"musubi parser: {len(want)} Stage 1 values as specified {bad or ''}")
check(a.base_weights == [__file__], "one --base_weights (golden)")

sys.path.insert(0, str(common.FW))
import asset_embed_v9 as AE  # noqa: E402
got = {}
AE.cmd_train = lambda a, rest: got.update(vars(a), rest=rest)
sys.argv = ["asset_embed_v9.py", "train", *wrapper, "--", "x"]
AE.main()
check(got["init"] == __file__ and got["offset"] and got["freeze_asset"] and got["preserve"] and got["preserve_weight"] == 1.0
      and got["preserve_marker"] == "key" and got["log_every"] == 5 and got["strict_base_weights"]
      and got["base_weights_high"] is None and not got["embed_only"], "wrapper flags: E5 frozen offset, preserve x1 key, log 5, strict")
common.finish()
