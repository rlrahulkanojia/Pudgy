"""Client long-form videos (~10 s each) with v8@550 on the client's own living-room sets.

Model: v8merged (v7fixed + v8@550 as one exact file) on the low-noise expert, v2 high-noise
golden on the high-noise expert - the shipped v8 winner (common_v9: scores the same as v8@550).
720x1280 portrait, 24 fps, 25 steps, Wan's default negative + anti-duplicate terms.

A video is TAKES chained takes of FRAMES frames, one continuous shot:
  mode "chain"  take 1 starts on the composited keyframe; take k starts on take k-1's last frame
  mode "loop"   every take is first+last-frame on the keyframe (start = end), seeds differ, so
                the takes join exactly and the framing cannot creep (motion zoom-out, REPORT_v8 §8)
3 x 81 frames - 2 shared seam frames = 241 frames = 10.0 s.

One model load per take stage: every video's take k goes into one batch.

python produce.py <stage 1|2|3> [ids...]      generate that stage (skips existing takes)
python produce.py plan                         print the jobs
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/Pudgy/finetune/wan/eval_v8")
import gen_v8                                   # noqa: E402
from common import P, GOLD_HIGH, WS             # noqa: E402

sys.path.insert(0, "/workspace/musubi-tuner/src")
from musubi_tuner.wan.configs.shared_config import wan_shared_cfg  # noqa: E402

ROOT = Path("/workspace/longform_client")
KF = ROOT / "keyframes"
TAKES_DIR = ROOT / "takes"
V8_MERGED = WS / "wan_output/v8_merged/pudgy-v8-merged.safetensors"
FRAMES, TAKES = 81, 3

# gen_v8 only knows eval_v8's configs; the v8@550 base + tau pair is not on this box, the
# exact merged file is.
gen_v8.lora_config = lambda name: ([(V8_MERGED, 1.0)], [(GOLD_HIGH, 1.0)])

NEG = wan_shared_cfg.sample_neg_prompt + ("，duplicate character, two Pax, two Polly, extra penguin, "
                                          "clone, second copy of the same character")
SETS = {
    "sofa":   "inside a cozy igloo living room with pale blue walls, a cream sofa behind it",
    "window": ("inside a cozy igloo living room with curved pale blue ice-block walls, a round window, "
               "a small side table with flowers and an open wooden door behind it"),
    "door":   "inside a cozy igloo living room with pale blue walls, a cream sofa, an open wooden door and a glass coffee table",
    "lounge": "inside a cozy igloo living room with pale blue walls, a cream sofa, a round window and a glass coffee table",
}
CAM = ("eye level, facing the camera directly, front view; fixed camera, locked-off shot, "
       "the character stays the same size")
LOOPS = {"walking", "waving", "sitting", "jumping"}           # cyclic in the v8 data (f49 tiled rung)
# Videos re-rolled after a bad take 1 (seed 41 + take + shift); the rejected takes are in takes/rejected/,
# the story edits use their clean first 2.75 s. Seed 42 went wrong from ~2.75 s (frames 66-80): waving and
# blushing (door set) slid toward the door and shrank to ~75%; crying stopped, smiled and jumped forward;
# walk_together walked off into the room, shrinking to 71% by 3.4 s; running hopped off toward the door.
# Takes made since are checked as they land (stream.make_take) and re-rolled the same way. jumping seed 42 shrank
# Pax to ~83% within its first 0.6 s (the v8 motion zoom-out) - before the end window the check looks at.
# surprised seed 42 shot Pax out of frame to the right in its last 6 frames. angry take 3 (seed 44) faded Polly
# pale and see-through and lost the anger (takes 1-2 kept; only take 3 is re-made, as seed 1044).
SEED_SHIFT = {"waving": 1000, "blushing": 1000, "crying": 1000, "walk_together": 1000, "running": 1000, "jumping": 1000,
              "surprised": 1000, "angry": 1000}
PAIR = f"{P.ANCHORS['Pax']}, and {P.ANCHORS['Polly']}"
INTERACTIONS = {
    "wave_together": ("standing side by side and waving at the camera together, each raising one flipper "
                      "and swinging it side to side, eyes closed in happy smiles; waving together"),
    "walk_together": ("walking in place side by side with a steady waddling gait, feet alternating, "
                      "flippers swinging gently; walking together"),
    "hug":           ("stepping toward each other and wrapping their flippers around each other in a warm "
                      "hug, holding the embrace; hugging"),
}


def videos():
    """id -> {key, prompt, mode}. One video per item; the character alternates by item."""
    meta = json.loads((KF / "keyframes.json").read_text())
    out = {}
    labels = {**P.MOTION, **P.EXPRESSIONS}
    order = ["walking", "running", "waving", "sitting", "jumping", "happy", "surprised", "angry",
             "confused", "crying", "neutral", "laughing", "blushing"]
    for i, item in enumerate(order):
        char = ("Pax", "Polly")[i % 2]
        k = f"{item}_{char.lower()}"
        lab = labels[item]
        loop = ", repeating in a continuous loop" if item in LOOPS else ""
        shot = "medium shot"            # "medium close-up" made crying_t1 cut to a close-up mid-take
        prompt = (f"{P.STYLE}{P.ANCHORS[char]}, {lab['action']}; {lab['label']}{loop}; "
                  f"{shot}, {CAM}; {SETS[meta[k]['plate']]}.")
        out[item] = dict(key=str(KF / f"{k}.png"), prompt=prompt, mode="chain", char=char)
    for item, action in INTERACTIONS.items():
        cam = CAM.replace("the character stays", "the characters stay")
        prompt = f"{P.STYLE}{PAIR}, {action}; medium wide shot, {cam}; {SETS[meta[item]['plate']]}."
        out[item] = dict(key=str(KF / f"{item}.png"), prompt=prompt, mode="chain", char="both")
    return out


def last_frame(mp4, png):
    if not png.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-sseof", "-1", "-i", str(mp4),
                        "-update", "1", "-q:v", "1", str(png)], check=True)
    return png


def jobs_for(stage, sel):
    jobs = []
    for vid, v in sel:
        tag = f"{vid}_{v['mode']}_t{stage}"
        seed = 41 + stage + SEED_SHIFT.get(vid, 0)
        if v["mode"] == "loop":
            jobs.append(dict(tag=tag, prompt=v["prompt"], start=v["key"], end=v["key"],
                             frames=FRAMES, seed=seed, neg=NEG))
            continue
        if stage == 1:
            start = v["key"]
        else:
            prev = TAKES_DIR / f"{vid}_{v['mode']}_t{stage - 1}.mp4"
            if not prev.exists():
                print(f"   skip {tag}: {prev.name} not generated yet")
                continue
            start = str(last_frame(prev, TAKES_DIR / f"{prev.stem}_last.png"))
        jobs.append(dict(tag=tag, prompt=v["prompt"], start=start, frames=FRAMES, seed=seed, neg=NEG))
    return jobs


def loop_job(vid, v):
    """The looping version's last take: from take 2's last frame back to the keyframe (first+last
    frame), so the video ends on its own first frame. Takes 1-2 are the non-looping version's."""
    prev = TAKES_DIR / f"{vid}_{v['mode']}_t2.mp4"
    start = str(last_frame(prev, TAKES_DIR / f"{prev.stem}_last.png"))
    return dict(tag=f"{vid}_{v['mode']}_t3loop", prompt=v["prompt"], start=start, end=v["key"],
                frames=FRAMES, seed=41 + TAKES + SEED_SHIFT.get(vid, 0), neg=NEG)


def select(ids):
    """[(id, video)] for "<id>" / "<id>:loop" (the loop variant of a video); no ids = every video."""
    vids, sel = videos(), []
    for x in ids or vids:
        vid, _, mode = x.partition(":")
        sel.append((vid, dict(vids[vid], mode=mode or vids[vid]["mode"])))
    return sel


def main():
    if sys.argv[1] == "plan":
        for vid, v in videos().items():
            print(f"{vid:14s} {v['mode']:5s} {Path(v['key']).name}\n   {v['prompt']}")
        return
    stage = int(sys.argv[1])
    jobs = jobs_for(stage, select(sys.argv[2:]))
    if not jobs:
        print("nothing to do"); return
    gen_v8.run("v8merged", jobs, TAKES_DIR)
    print("STAGE DONE", stage, [j["tag"] for j in jobs], flush=True)


if __name__ == "__main__":
    main()
