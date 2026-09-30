"""Generate every beat of both skits with v8@550 at 720x1280 portrait. Existing takes are skipped,
so re-running after adding seeds/overrides only renders what's new. SEEDS/ONLY via argv."""
import sys; sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8"); sys.path.insert(0, "/workspace/longform")
import gen_v8, shots
sys.path.insert(0, "/workspace/musubi-tuner/src")
from musubi_tuner.wan.configs.shared_config import wan_shared_cfg
# Rule: at most one Pax and one Polly in any frame. Wan's default negative is kept (it carries the
# motion push) and anti-duplicate terms are appended; dupcheck.py gates every take afterwards.
NEG = wan_shared_cfg.sample_neg_prompt + ("，duplicate character, two Pax, two Polly, extra penguin, "
                                          "clone, second copy of the same character")
seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "42").split(",")]
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for name, lst in (("life_with_her", shots.LIFE_WITH_HER), ("eating_stages", shots.EATING_STAGES)):
    for s in lst:                      # one batch per shot, so each take lands (and can be reviewed) early
        if only and s["id"] not in only:
            continue
        suffix = "_flf" if s.get("end") else ""   # first+last-frame takes are kept apart from plain I2V takes
        jobs = [{"tag": f"{s['id']}_s{sd}{suffix}", "prompt": s["prompt"], "start": s["key"], "frames": s["frames"],
                 "seed": sd, "neg": NEG, **({"end": s["end"]} if s.get("end") else {})} for sd in seeds]
        gen_v8.run("v8@550", jobs, f"/workspace/longform/takes/{name}")
        print("SHOT DONE", s["id"], flush=True)
print("DONE")
