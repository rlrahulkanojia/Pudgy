#!/usr/bin/env python3
"""Pre-flight check for the Pudgy training dataset. Run before training.
    python check_dataset.py [path/to/training_dataset]
Verifies: metadata schema, every clip resolves, and frame-count/res/fps are
consistent with what train_pudgy_lora.sh requests (33 frames @16fps, 768x1360).
"""
import sys, os, json, subprocess

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "training_dataset")
root = os.path.abspath(root)
meta = os.path.join(root, "metadata.json")
print(f"dataset: {root}")

d = json.load(open(meta))
assert d, "metadata.json is empty"
bad_keys = [i for i, x in enumerate(d) if set(x) < {"file_path", "text", "type"}]
assert not bad_keys, f"entries missing required keys: {bad_keys[:5]}"

EXP_FRAMES, EXP_W, EXP_H, EXP_FPS = 33, 768, 1360, "16/1"
problems, seen_frames = [], set()
for x in d:
    p = x["file_path"] if os.path.isabs(x["file_path"]) else os.path.join(root, x["file_path"])
    if not os.path.exists(p):
        problems.append(f"missing file: {x['file_path']}"); continue
    if not x["text"].strip():
        problems.append(f"empty caption: {x['file_path']}")
    info = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
        "-of", "csv=p=0", p]).decode().strip().split(",")
    nb, w, h, fr = info[0], info[1], info[2], info[3]
    seen_frames.add(nb)
    if (int(nb), int(w), int(h), fr) != (EXP_FRAMES, EXP_W, EXP_H, EXP_FPS):
        problems.append(f"{x['file_path']}: {w}x{h} {fr} {nb}f (expected {EXP_W}x{EXP_H} {EXP_FPS} {EXP_FRAMES}f)")

print(f"clips: {len(d)} | distinct frame-counts seen: {sorted(seen_frames)}")
if len(seen_frames) == 1:
    n = int(seen_frames.pop())
    ok8 = (n - 1) % 8 == 0
    print(f"frame count {n}: 8N+1 rule {'OK' if ok8 else 'FAIL'}  -> set --video_sample_n_frames={n} --video_sample_stride=1")
if problems:
    print(f"\n❌ {len(problems)} problem(s):")
    for pb in problems[:20]:
        print("  -", pb)
    sys.exit(1)
print("\n✅ Dataset looks good — ready to train.")
