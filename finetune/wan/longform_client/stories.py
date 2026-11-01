"""Two ~15 s story edits cut from the batch clips - no generation. Each shot is a piece of a batch
clip, decoded straight from its takes (so a story needs only the takes its pieces fall in - take 1
when every piece is in the first 3.4 s), hard cuts, one encode with the batch's settings
(assemble_client), then Azure pudgy/2026-10-28_longform/stories/.

python stories.py [--no-upload] [names...]   -> final/stories/<name>_1080x1920.mp4 + qa/story_<name>.png
python stories.py --watch                    make + upload each story as soon as its takes exist

Every clip faces the camera, so the story is told by the order of the shots only.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/longform_client")
sys.path.insert(0, str(ROOT))
import assemble_client as AC            # noqa: E402
import upload                           # noqa: E402

DATE = "2026-10-28"
FPS = 24
OUT = ROOT / "final/stories"
# name -> [(batch clip id, start s, length s)]. All pieces lie in take 1 (frames 0-80), the part
# closest to the clean keyframe.
STORIES = {
    "surprise_visit": [("sitting", 0.5, 2.5),          # Polly at home on the sofa
                       ("waving", 0.25, 2.5),          # Pax at the open door, waving hello (slides away from 2.8 s)
                       ("happy", 0.75, 2.5),           # Polly lights up (ends on the big smile)
                       ("blushing", 0.6, 2.0),         # Pax gets shy (slides away from 2.75 s)
                       ("hug", 0.25, 3.0),             # they hug
                       ("wave_together", 0.5, 2.5)],   # both wave at the camera
    "cheer_up_polly": [("crying", 0.75, 2.0),          # Polly is crying (stops and smiles from 2.8 s)
                       ("confused", 0.5, 2.5),         # Pax notices, puzzled
                       ("walking", 0.5, 2.5),          # Pax waddles over
                       ("hug", 0.25, 3.0),             # they hug
                       ("laughing", 0.5, 2.5),         # Polly laughs again
                       ("walk_together", 0.5, 2.5)],   # they walk off together
}


def takes_for(vid, upto):
    """The takes holding a clip's first `upto` frames as they appear in its final video: take 1's 81,
    then 80 more per later take (its first frame is the join)."""
    n = max(1, -(-(upto - 1) // (AC.FRAMES - 1)))
    return [ROOT / f"takes/{vid}_chain_t{k}.mp4" for k in range(1, n + 1)]


def clip_frames(vid, upto):
    takes = takes_for(vid, upto)
    return np.concatenate([AC.decode(t)[:AC.FRAMES][min(k, 1):] for k, t in enumerate(takes)])[:upto]


def cut(name, shots):
    """The story video, or None while a take it needs isn't generated yet."""
    spans = [(vid, round(start * FPS), round((start + length) * FPS)) for vid, start, length in shots]
    if not all(t.exists() for vid, _, b in spans for t in takes_for(vid, b)):
        return None
    pieces = [clip_frames(vid, b)[a:b] for vid, a, b in spans]
    v = np.concatenate(pieces)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}_1080x1920.mp4"
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{AC.W}x{AC.H}", "-r", str(FPS), "-i", "-", "-vf",
                            "scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.0,format=yuv420p",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart",
                            "-an", str(dst)], stdin=subprocess.PIPE)
    enc.stdin.write(v.tobytes()); enc.stdin.close()
    if enc.wait():
        sys.exit(f"{name}: encode failed")
    sheet = np.concatenate([np.asarray(Image.fromarray(p[0]).resize((180, 320))) for p in pieces], 1)
    (ROOT / "qa").mkdir(exist_ok=True)
    Image.fromarray(sheet).save(ROOT / f"qa/story_{name}.png")
    print(f"{dst.name}: {len(shots)} shots, {len(v) / FPS:.1f} s", flush=True)
    return dst


def publish(dsts):
    for n, u in upload.main(DATE, files=dsts, sub="stories").items():
        print(f"## STORY {n} uploaded: {u}", flush=True)


def main():
    a = sys.argv[1:]
    names = [x for x in a if not x.startswith("--")] or list(STORIES)
    if "--watch" in a:
        todo = [n for n in names if not (OUT / f"{n}_1080x1920.mp4").exists()]
        while todo:
            for n in list(todo):
                dst = cut(n, STORIES[n])
                if dst:
                    publish([dst]); todo.remove(n)
            if todo:
                time.sleep(120)
        return
    dsts = [cut(n, STORIES[n]) for n in names]
    if None in dsts:
        sys.exit(f"takes not generated yet for {[n for n, d in zip(names, dsts) if d is None]}")
    if "--no-upload" not in a:
        publish(dsts)


if __name__ == "__main__":
    main()
