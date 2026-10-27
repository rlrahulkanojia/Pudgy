"""Contact sheet: for each shot idx (shot_review.json), frames at the given fractions.
Usage: sheet.py review.json RAW OUT.jpg idx,idx,... [q,q,...]"""
import json, subprocess, io, sys
from PIL import Image, ImageDraw
r = json.load(open(sys.argv[1])); raw = sys.argv[2]
idx = [int(i) for i in sys.argv[4].split(",")]
qs = [float(q) for q in (sys.argv[5] if len(sys.argv) > 5 else "0.1,0.5,0.9").split(",")]
W, H = 216, 384
sheet = Image.new("RGB", (W * len(qs), (H + 18) * len(idx)), "white"); d = ImageDraw.Draw(sheet)
for row, i in enumerate(idx):
    s = r[i]; n = s["end"] - s["start"]
    d.text((4, row * (H + 18) + 3), f'{i}: {s["file"].split("/")[-1]} #{s["shot"]} {n}f {s["verdict"]} {s["reason"]}', fill="black")
    for k, q in enumerate(qs):
        fr = s["start"] + min(n - 1, int(n * q))
        b = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{raw}/{s['file']}", "-vf", f"select=eq(n\\,{fr}),scale={W}:{H}",
                            "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        sheet.paste(Image.open(io.BytesIO(b)), (k * W, row * (H + 18) + 18))
sheet.save(sys.argv[3], quality=85)
