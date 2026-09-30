"""A/B: does Wan's default negative prompt ("static", "still", "motionless") cause the zoom-out?
A = existing core clips (default negative). B = same seed/start/prompt, negative with those three
terms removed and explicit anti-zoom terms added. Only the negative prompt differs."""
import sys, json
sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8")
sys.path.insert(0, "/workspace/musubi-tuner/src")
import gen_v8
from common import caption, keyframe, P
from musubi_tuner.wan.configs.shared_config import wan_shared_cfg
default = wan_shared_cfg.sample_neg_prompt
drop = ["静止不动的画面，", "静态，", "静止，"]          # "motionless frame", "static", "still"
neg_b = default
for d in drop:
    assert d in neg_b, d
    neg_b = neg_b.replace(d, "")
neg_b += ("，zoom out, zooming, camera pulling back, dolly out, camera movement, "
          "character moving away from the camera, character getting smaller, receding into the distance")
jobs = []
for char in ("Pax", "Polly"):
    for lab in ("running", "walking", "jumping", "sitting"):
        n = P.COMMON["motion"]
        jobs.append({"tag": f"{char.lower()}_{lab}_FRONT_f{n}_s42_negB", "prompt": caption(char, lab, "FRONT", nframes=n),
                     "start": str(keyframe(char)), "frames": n, "seed": 42, "neg": neg_b,
                     "meta": {"char": char, "label": lab, "neg": "B"}})
json.dump({"neg_default": default, "neg_B": neg_b}, open("/workspace/eval_v8/zoomfix/negatives.json", "w"), ensure_ascii=False, indent=1)
out = gen_v8.run("v8@550", jobs, "/workspace/eval_v8/zoomfix/v8@550")
print("DONE", len(out))
