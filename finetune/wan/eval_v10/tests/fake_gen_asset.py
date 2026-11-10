"""Stands in for eval_v9/gen_asset_v9.py + musubi --interactive: reads one job line per stdin line,
writes <f> frames of the start image (or a crossfade start->end for --ei) and prints 'Video saved to:'."""
import re, subprocess, sys, time, json
from pathlib import Path
argv = sys.argv[1:]
i = argv.index("--")
own, rest = argv[:i], argv[i + 1:]
assert "--asset" in own and "--asset-offset" in own and "--asset-phrase" in own, own
assert "--interactive" in rest and "--from_file" not in rest
save = Path(rest[rest.index("--save_path") + 1])
vs = [k for k, x in enumerate(rest) if x == "--video_size"][-1]
h, w = int(rest[vs + 1]), int(rest[vs + 2])
print("fake musubi: loaded", json.dumps({"size": [h, w], "lora_high": rest[rest.index("--lora_weight_high_noise") + 1:rest.index("--lora_multiplier_high_noise")]}), flush=True)
while True:
    try:
        line = input("Enter prompt: ")
    except EOFError:
        break
    m = re.match(r"(?P<prompt>.*?) --i (?P<i>\S+) --f (?P<f>\d+) --d (?P<d>\d+)(?: --n (?P<n>.*?))?(?: --ei (?P<ei>\S+))?$", line)
    assert m, line
    f = int(m["f"]); out = save / f"{time.strftime('%Y%m%d-%H%M%S')}_{m['d']}_{time.time_ns() % 10**6}.mp4"
    if m["ei"]:
        vf = f"[0:v]scale={w}:{h},format=yuv420p[a];[1:v]scale={w}:{h},format=yuv420p[b];[a][b]blend=all_expr='A*(1-N/{f})+B*(N/{f})'"
        cmd = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", m["i"], "-loop", "1", "-i", m["ei"], "-filter_complex", vf,
               "-frames:v", str(f), "-r", "24", "-pix_fmt", "yuv420p", str(out)]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", m["i"], "-vf", f"scale={w}:{h}", "-frames:v", str(f),
               "-r", "24", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print(f"[asset] spliced E into rows 40:66" if "iFin" in m["prompt"] else "no phrase", flush=True)
    print(f"Video saved to: {out}", flush=True)
