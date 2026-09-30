#!/usr/bin/env python3
"""Assemble chosen takes into a client-ready long-form video.

  python assemble.py life_with_her   (reads /workspace/longform/takes/<name>.json)

takes JSON: {"name": ..., "reference": <ref mp4 for audio>, "shots": [{"clip": path, "trim_start": 0, "trim_end": 0}, ...]}
Output (in /workspace/longform/final/):
  <name>_1080x1920.mp4          silent, 24 fps, H.264 CRF 16, yuv420p, faststart
  <name>_1080x1920_audio.mp4    same picture + the reference skit's soundtrack, trimmed/faded to length
Every shot is upscaled 720x1280 -> 1080x1920 with lanczos and joined with hard cuts (as the
reference edits are). A QA line reports duration, resolution, frame count and black frames.
"""
import json
import subprocess
import sys
from pathlib import Path

LF = Path("/workspace/longform")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-2000:])
    return r.stdout


def main(name):
    spec = json.loads((LF / "takes" / f"{name}.json").read_text())
    out = LF / "final"; out.mkdir(exist_ok=True)
    tmp = out / f".{name}"; tmp.mkdir(exist_ok=True)
    parts = []
    for i, s in enumerate(spec["shots"]):
        n = int(run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", s["clip"]]).strip())
        a, b = s.get("trim_start", 0), n - s.get("trim_end", 0)
        p = tmp / f"{i:02d}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-i", s["clip"],
             "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,"
                    "scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.0,fps=24,format=yuv420p",
             "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-an", str(p)])
        parts.append(p)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    silent = out / f"{name}_1080x1920.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(silent)])
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(silent)]))
    if spec.get("reference"):
        run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", spec["reference"],
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-af", f"afade=t=out:st={max(dur - 0.8, 0):.2f}:d=0.8", "-t", f"{dur:.3f}",
             "-movflags", "+faststart", str(out / f"{name}_1080x1920_audio.mp4")])
    info = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames", "-of", "csv=p=0", str(silent)]).strip()
    bd = subprocess.run(["ffmpeg", "-i", str(silent), "-vf", "blackdetect=d=0.1:pix_th=0.05", "-an", "-f", "null", "-"],
                        capture_output=True, text=True).stderr.count("black_start")
    print(f"QA {name}: {dur:.2f} s | w,h,fps,frames = {info} | black segments: {bd} | shots: {len(parts)}")


if __name__ == "__main__":
    main(sys.argv[1])
