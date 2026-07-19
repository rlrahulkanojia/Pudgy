#!/usr/bin/env python3
"""
extract_frames.py — turn a local video into a small set of still frames that
Claude can Read (see) for dataset curation.

Pure Python 3 + ffmpeg/ffprobe. No pip dependencies.

Usage:
    python3 extract_frames.py CLIP.mp4
    python3 extract_frames.py CLIP.mp4 --max-frames 24 --width 1024
    python3 extract_frames.py CLIP.mp4 --mode scene --scene-threshold 0.3
    python3 extract_frames.py CLIP.mp4 --json            # machine-readable only

Output:
    - JPEG frames written to an output dir (printed on stdout / in JSON).
    - A JSON blob (always) describing frames + timestamps + token estimate.
    - A human-readable frame index (unless --json) that the agent reads before
      opening the frames.

Modes:
    uniform  (default) — evenly spaced across the whole clip. Best for short
                         training clips: guarantees temporal coverage so motion
                         and consistency can be judged.
    scene              — one frame per visual cut (ffmpeg scene detection).
                         Best for longer / edited footage.
    keyframe           — I-frames only. Fastest, sparse.

Dedup: mpdecimate drops near-duplicate frames (disable with --no-dedup).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile


def run(cmd, capture_stderr=False):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def ffprobe_meta(path):
    proc = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ])
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed for {path}:\n{proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    dur = stream.get("duration") or fmt.get("duration")
    duration = float(dur) if dur not in (None, "N/A") else 0.0

    fps = 0.0
    rate = stream.get("r_frame_rate")
    if rate and "/" in rate:
        num, den = rate.split("/")
        try:
            den = float(den)
            fps = float(num) / den if den else 0.0
        except ValueError:
            fps = 0.0
    return {"width": width, "height": height, "duration": duration, "fps": fps}


def build_filter(mode, scene_threshold, width, dedup, uniform_rate=None):
    parts = []
    if mode == "uniform":
        # Downsample inside the filtergraph so showinfo reports the pts of the
        # frames actually written (an output -r would resample after showinfo).
        parts.append(f"fps={uniform_rate:.6f}")
    elif mode == "scene":
        parts.append(f"select='gt(scene,{scene_threshold})'")
    elif mode == "keyframe":
        parts.append("select='eq(pict_type,I)'")
    else:
        raise SystemExit(f"unknown mode: {mode}")

    if width:
        parts.append(f"scale={width}:-2:flags=lanczos")
    if dedup:
        parts.append("mpdecimate")
    parts.append("showinfo")
    return ",".join(parts)


def parse_pts(stderr):
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr)]


def even_subsample(items, k):
    """Keep k items evenly spaced, always including first and last."""
    n = len(items)
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    # de-dup while preserving order
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def extract(path, out_dir, mode, scene_threshold, width, max_frames, dedup):
    meta = ffprobe_meta(path)
    os.makedirs(out_dir, exist_ok=True)
    # clear any stale frames
    for f in os.listdir(out_dir):
        if f.startswith("f_") and f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))

    uniform_rate = None
    if mode == "uniform":
        dur = meta["duration"] or 1.0
        target = max(max_frames, 1)
        uniform_rate = min(max(target / dur, 0.05), 30.0)

    vf = build_filter(mode, scene_threshold, width, dedup, uniform_rate=uniform_rate)
    out_pattern = os.path.join(out_dir, "f_%05d.jpg")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", path,
           "-vf", vf, "-fps_mode", "vfr", "-q:v", "3", out_pattern]

    proc = run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed for {path}:\n{proc.stderr[-2000:]}")

    pts = parse_pts(proc.stderr)
    files = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("f_") and f.endswith(".jpg")
    )
    # align timestamps to files (showinfo emits one line per output frame)
    times = pts[: len(files)] + [None] * max(0, len(files) - len(pts))

    keep = even_subsample(files, max_frames)
    kept_files = [files[i] for i in keep]
    kept_times = [times[i] for i in keep]

    # remove dropped frames from disk to keep the dir clean
    keep_set = set(kept_files)
    for f in files:
        if f not in keep_set:
            try:
                os.remove(f)
            except OSError:
                pass

    # token estimate (Claude ~= ceil(w*h/750) per frame)
    out_w = width or meta["width"]
    if meta["width"]:
        out_h = round(meta["height"] * out_w / meta["width"])
    else:
        out_h = meta["height"]
    per_frame = math.ceil((out_w * out_h) / 750) if out_w and out_h else 0
    est_tokens = per_frame * len(kept_files)

    frames = []
    for f, t in zip(kept_files, kept_times):
        frames.append({"path": f, "t": round(t, 2) if t is not None else None})

    return {
        "video": os.path.abspath(path),
        "duration_sec": round(meta["duration"], 2),
        "source_fps": round(meta["fps"], 3),
        "source_resolution": f'{meta["width"]}x{meta["height"]}',
        "frame_resolution": f"{out_w}x{out_h}",
        "mode": mode,
        "frame_count": len(frames),
        "est_tokens_claude": est_tokens,
        "out_dir": os.path.abspath(out_dir),
        "frames": frames,
    }


def fmt_ts(t):
    if t is None:
        return "  ?  "
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description="Extract frames from a local video for Claude to view.")
    ap.add_argument("video", help="path to a local video file")
    ap.add_argument("--mode", choices=["uniform", "scene", "keyframe"], default="uniform")
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=1024, help="output frame width (px); height auto")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--no-dedup", action="store_true", help="keep near-duplicate frames")
    ap.add_argument("--out-dir", default=None, help="where to write frames (default: temp)")
    ap.add_argument("--json", action="store_true", help="print JSON only (no human index)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"not a file: {args.video}")

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir or os.path.join(tempfile.gettempdir(), "claude_video_frames", stem)

    result = extract(
        args.video, out_dir,
        mode=args.mode,
        scene_threshold=args.scene_threshold,
        width=args.width,
        max_frames=args.max_frames,
        dedup=not args.no_dedup,
    )

    print(json.dumps(result, indent=2))
    if not args.json:
        print("\n=== FRAME INDEX (Read these images in order) ===", file=sys.stderr)
        for i, fr in enumerate(result["frames"]):
            print(f"[{i:02d}] t={fmt_ts(fr['t'])}  {fr['path']}", file=sys.stderr)
        print(
            f"\n{result['frame_count']} frames | ~{result['est_tokens_claude']} img tokens | "
            f"{result['frame_resolution']} | {result['duration_sec']}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
