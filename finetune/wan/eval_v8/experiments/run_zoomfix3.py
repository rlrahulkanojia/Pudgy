"""Zoom A/B, round 3. Round 2's B2 (default negative + anti-zoom) removed the zoom (shadow
-11.4% -> -0.5%) but "character getting smaller" also forbids SITTING DOWN (sitting froze), and
blocking only zoom-OUT let running drift toward the camera. B3 = default negative + camera-only
terms in both directions, nothing about the character's own size. Guidance left at defaults
(low 5.0; high-noise already the official 3.5 - musubi's --guidance_scale is low-noise only)."""
import sys, json
sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8"); sys.path.insert(0, "/workspace/musubi-tuner/src")
import gen_v8
from common import caption, keyframe, P
from musubi_tuner.wan.configs.shared_config import wan_shared_cfg
NEG_B3 = wan_shared_cfg.sample_neg_prompt + ("，zoom in, zoom out, camera pushing in, camera pulling back, dolly, "
          "camera movement, moving toward or away from the camera, receding into the distance")
jobs = []
for char in ("Pax", "Polly"):
    for lab in ("running", "walking", "jumping", "sitting"):
        n = P.COMMON["motion"]
        jobs.append({"tag": f"{char.lower()}_{lab}_FRONT_f{n}_s42_B3", "prompt": caption(char, lab, "FRONT", nframes=n),
                     "start": str(keyframe(char)), "frames": n, "seed": 42, "neg": NEG_B3})
json.dump({"neg_B3": NEG_B3}, open("/workspace/eval_v8/zoomfix/negative_B3.json", "w"), ensure_ascii=False, indent=1)
gen_v8.run("v8@550", jobs, "/workspace/eval_v8/zoomfix/B3")
print("DONE")
