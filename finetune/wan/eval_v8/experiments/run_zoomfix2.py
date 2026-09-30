"""Zoom A/B, round 2. Round 1: removing static/still/motionless from the negative killed the
zoom (shadow -11.4% -> -1.1%) but also motion (sitting froze, -30% animation). Here:
  B2 = default negative + anti-zoom terms only (keeps the motion push), guidance 5.0
  C  = default negative, guidance 3.5 (Wan2.2-I2V official; the eval inherited 5.0 from v7)
  D  = default negative + anti-zoom terms, guidance 3.5"""
import sys, json, os, subprocess
sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8"); sys.path.insert(0, "/workspace/musubi-tuner/src")
import gen_v8
from common import caption, keyframe, P
from musubi_tuner.wan.configs.shared_config import wan_shared_cfg
default = wan_shared_cfg.sample_neg_prompt
anti = ("，zoom out, zooming, camera pulling back, dolly out, camera movement, "
        "character moving away from the camera, character getting smaller, receding into the distance")
variants = {"B2": (default + anti, None), "C": (None, "3.5"), "D": (default + anti, "3.5")}
base_extra = os.environ.get("GEN_EXTRA", "")
for tag, (neg, guide) in variants.items():
    jobs = []
    for char in ("Pax", "Polly"):
        for lab in ("running", "walking", "jumping", "sitting"):
            n = P.COMMON["motion"]
            j = {"tag": f"{char.lower()}_{lab}_FRONT_f{n}_s42_{tag}", "prompt": caption(char, lab, "FRONT", nframes=n),
                 "start": str(keyframe(char)), "frames": n, "seed": 42}
            if neg:
                j["neg"] = neg
            jobs.append(j)
    os.environ["GEN_EXTRA"] = base_extra + (f" --guidance_scale {guide}" if guide else "")
    gen_v8.run("v8@550", jobs, f"/workspace/eval_v8/zoomfix/{tag}")
    print("variant done", tag, flush=True)
print("DONE")
