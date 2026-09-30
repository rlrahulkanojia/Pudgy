import sys, json
sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8")
import gen_v8
from common import P
KF = "/workspace/eval_v8/interaction/keyframes"
PAIR = f"{P.ANCHORS['Polly']}, and {P.ANCHORS['Pax']}"   # colour anchors for both
WAVE = ("standing side by side and waving at the camera together, each raising one flipper "
        "and swinging it side to side, eyes closed in happy smiles; waving together")
HUG = ("stepping toward each other and wrapping their flippers around each other in a warm "
       "hug, holding the embrace; hugging")
def cap(action, shot, view):
    return f"{P.STYLE}{PAIR}, {action}; {shot}, eye level, {view}; {P.BACKGROUNDS['white'][0]}."
FRONT = "facing the camera directly, front view"
QF1L = "turned slightly to its left, three-quarter front view"
jobs = [
  {"tag": "1_wave_from_wavingFRONT_f25_s42", "prompt": cap(WAVE, "static wide shot", FRONT),
   "start": f"{KF}/waving_FRONT.png", "frames": 25, "seed": 42},
  {"tag": "2_hug_from_huggingQF1L_f21_s42", "prompt": cap(HUG, "static wide shot", QF1L),
   "start": f"{KF}/hugging_QF1_L.png", "frames": 21, "seed": 42},
  {"tag": "3_hug_from_wavingFRONT_f25_s42", "prompt": cap(HUG, "static wide shot", FRONT),
   "start": f"{KF}/waving_FRONT.png", "frames": 25, "seed": 42},
  {"tag": "4_wave_from_composite_f25_s42", "prompt": cap(WAVE, "static wide shot", FRONT),
   "start": f"{KF}/composite_side_by_side.png", "frames": 25, "seed": 42},
]
out = gen_v8.run("v8@550", jobs, "/workspace/eval_v8/interaction/v8@550")
json.dump({j["tag"]: j for j in jobs}, open("/workspace/eval_v8/interaction/v8@550/meta.json", "w"), indent=1)
print("DONE", out)
