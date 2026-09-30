import sys; sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8"); sys.path.insert(0, "/workspace/longform")
import gen_v8, shots
s = shots.LIFE_WITH_HER[0]
for cfg in ("golden", "v8@550"):
    gen_v8.run(cfg, [{"tag": f"{s['id']}_{cfg.replace('@','')}_s42", "prompt": s["prompt"], "start": s["key"],
                      "frames": s["frames"], "seed": 42}], "/workspace/longform/test")
print("DONE")
