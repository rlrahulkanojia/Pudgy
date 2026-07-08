#!/usr/bin/env python3
"""
build_dataset.py — Pudgy Penguins training-clip builder (Part B of the Training Runbook).

Turns client master renders into CogVideoX1.5-5B-I2V-conformant training clips:
segment -> resample 24->16fps -> trim to 8N+1 frames -> resize to 768x1360
-> composite 70/30 gray/show background -> export + paired caption + manifest.

Usage:
  # Build clips from a segment sheet
  python build_dataset.py build \
      --segments segments.csv \
      --src-dir  /vol/masters \
      --out-dir  /vol/dataset_v1

  # Verify an already-built dataset against the hard format spec
  python build_dataset.py verify --out-dir /vol/dataset_v1

segments.csv columns (one row per atomic micro-action):
  source_file,in,out,character,action_label,bg_type,alpha_file,bg_plate,props
    in/out      : HH:MM:SS.mmm or seconds (float)
    character   : "pax" | "polly" | "pax+polly"
    bg_type     : "gray" | "show"        (target 70% gray / 30% show overall)
    alpha_file  : optional ProRes4444/PNG-seq with alpha (for compositing)
    bg_plate    : optional clean background render (used when bg_type == show)
    props       : optional "|"-separated Prp_* ids present (e.g. "Prp_ChessBoard|Prp_iFin")

Requires: ffmpeg + ffprobe on PATH. No third-party Python deps.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# ---- Hard format spec (CogVideoX1.5-5B-I2V) --------------------------------
TARGET_W, TARGET_H = 768, 1360          # portrait primary bucket
TARGET_FPS = 16
VALID_FRAMES = (49, 81)                  # 8N+1; 49 -> ~3.06s, 81 -> ~5.06s
GRAY_HEX = "0x808080"                    # #808080 neutral background
LANDSCAPE = (1360, 768)                  # only if a horizontal deliverable needs it


def run(cmd: list[str]) -> str:
    """Run a command, return stdout, raise with stderr on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def to_seconds(ts: str) -> float:
    ts = ts.strip()
    if ":" not in ts:
        return float(ts)
    h, m, s = (["0", "0"] + ts.split(":"))[-3:]
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe(path: Path) -> dict:
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-count_frames", "-of", "json", str(path),
    ])
    return json.loads(out)["streams"][0]


def choose_frames(duration_s: float) -> int:
    """Pick the largest valid 8N+1 frame count that fits the window at 16fps."""
    target = duration_s * TARGET_FPS
    # prefer 81 if the window is long enough, else 49
    return 81 if target >= 81 else 49


@dataclass
class ClipRow:
    clip_id: str
    character: str
    action_label: str
    bg_type: str
    props: str
    frames: int
    width: int
    height: int
    fps: int
    source_skit: str


def build_clip(seg: dict, src_dir: Path, out_dir: Path, idx: int) -> ClipRow:
    src = src_dir / seg["source_file"]
    if not src.exists():
        raise FileNotFoundError(src)
    start = to_seconds(seg["in"])
    dur = to_seconds(seg["out"]) - start
    if not (2.0 <= dur <= 5.2):
        print(f"  ! warning: clip {idx} duration {dur:.2f}s outside 2-5s window")

    frames = choose_frames(dur)
    clip_id = f"{Path(seg['source_file']).stem}_{idx:03d}_{seg['action_label'].replace(' ', '-')}"
    out_path = out_dir / "clips" / f"{clip_id}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg_type = seg.get("bg_type", "gray").strip() or "gray"
    alpha = seg.get("alpha_file", "").strip()
    bg_plate = seg.get("bg_plate", "").strip()

    # --- ffmpeg graph ---------------------------------------------------------
    # 1) resample to 16fps, 2) clone-pad the tail so the window always has >= `frames`
    #    (a 3.0s window is only 48f at 16fps -> 1 short of 49; tpad covers the gap),
    # 3) scale-to-fill then crop to 768x1360 (preserve aspect, never stretch),
    # 4) trim to exactly `frames`.
    vf_geometry = (
        f"fps={TARGET_FPS},"
        f"tpad=stop_mode=clone:stop_duration=1,"
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        f"trim=start_frame=0:end_frame={frames},setpts=PTS-STARTPTS"
    )

    if alpha and bg_type == "gray":
        # Composite character-with-alpha over a flat #808080 source.
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={GRAY_HEX}:s={TARGET_W}x{TARGET_H}:r={TARGET_FPS}",
            "-ss", str(start), "-t", str(dur), "-i", str(src_dir / alpha),
            "-filter_complex",
            f"[1:v]{vf_geometry}[fg];[0:v][fg]overlay=shortest=1,trim=end_frame={frames}[v]",
            "-map", "[v]", "-frames:v", str(frames),
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(out_path),
        ]
    elif alpha and bg_type == "show" and bg_plate:
        # Composite character-with-alpha over the clean background plate.
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(dur), "-i", str(src_dir / bg_plate),
            "-ss", str(start), "-t", str(dur), "-i", str(src_dir / alpha),
            "-filter_complex",
            f"[0:v]{vf_geometry}[bg];[1:v]{vf_geometry}[fg];"
            f"[bg][fg]overlay=shortest=1,trim=end_frame={frames}[v]",
            "-map", "[v]", "-frames:v", str(frames),
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(out_path),
        ]
    else:
        # No alpha: background already baked in (gray/show double-render fallback).
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(dur), "-i", str(src),
            "-vf", vf_geometry, "-frames:v", str(frames),
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(out_path),
        ]
    run(cmd)

    # --- paired caption stub (Tier-1 anchor; Tier-2 suffix filled by captioner)
    cap_path = out_dir / "captions" / f"{clip_id}.txt"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    if not cap_path.exists():
        cap_path.write_text(_anchor(seg["character"]) + "  <<MOTION_SUFFIX>>\n")

    return ClipRow(
        clip_id=clip_id, character=seg["character"], action_label=seg["action_label"],
        bg_type=bg_type, props=seg.get("props", "").strip(),
        frames=frames, width=TARGET_W, height=TARGET_H,
        fps=TARGET_FPS, source_skit=seg["source_file"],
    )


def _anchor(character: str) -> str:
    # Characters are Pax (blue) and Polly (pink). Final wording (exact hex/accessory
    # names) comes from the client Figma/style guide (Part C1) — placeholders here.
    pax = ("A stylized 2D cartoon animation of Pax, a pudgy blue penguin with "
           "thick black outlines, a white belly, and an orange beak,")
    polly = ("A stylized 2D cartoon animation of Polly, a pudgy pink penguin with "
             "thick black outlines, blush cheeks, a swept head-tuft, and an orange beak,")
    if character == "pax":
        return pax
    if character == "polly":
        return polly
    # multi-character: spatial side is scene-dependent, so it is NOT hardcoded here.
    # The captioner fills <<LEFT>>/<<RIGHT>> per clip from the actual frame (Part C1).
    return ("A stylized 2D cartoon animation featuring Pax, a pudgy blue penguin with "
            "thick black outlines, positioned on the <<LEFT_OR_RIGHT>> third of the frame, "
            "and Polly, a pudgy pink penguin with blush cheeks, positioned on the "
            "<<OTHER_SIDE>> third of the frame,")


def cmd_build(args: argparse.Namespace) -> int:
    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    rows: list[ClipRow] = []
    with open(args.segments, newline="") as f:
        for i, seg in enumerate(csv.DictReader(f), 1):
            print(f"[{i}] {seg['source_file']} {seg['in']}-{seg['out']} "
                  f"({seg['action_label']}, {seg.get('bg_type','gray')})")
            try:
                rows.append(build_clip(seg, src_dir, out_dir, i))
            except Exception as e:  # noqa: BLE001 - keep going, report at end
                print(f"  ! FAILED: {e}")

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else
                           ["clip_id"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nBuilt {len(rows)} clips -> {manifest}")
    _report_gates(rows)
    return 0


def _report_gates(rows: list[ClipRow]) -> None:
    """Part B3: bucket split, action parity, bg mix — report, don't block."""
    if not rows:
        return
    n = len(rows)
    multi = sum(1 for r in rows if "+" in r.character)
    show = sum(1 for r in rows if r.bg_type == "show")
    from collections import Counter
    actions = Counter(r.action_label for r in rows)
    props = Counter(p for r in rows for p in r.props.split("|") if p)
    print("\n--- composition gates ---")
    print(f"  multi-character: {multi}/{n} ({multi/n:.0%})  (target ~40% Tier1+Tier2)")
    print(f"  show-bg mix:     {show}/{n} ({show/n:.0%})  (target ~30%)")
    starved = [a for a, c in actions.items() if c < 4]
    print(f"  action types:    {len(actions)}; starved (<4 clips): {starved or 'none'}")
    lonely = [p for p, c in props.items() if c < 2]
    print(f"  artifacts (Prp_*): {len(props)}; appear in only 1 clip: {lonely or 'none'}")


def cmd_verify(args: argparse.Namespace) -> int:
    """Part B verification gate 1: assert every clip meets the hard spec."""
    out_dir = Path(args.out_dir)
    clips = sorted((out_dir / "clips").glob("*.mp4"))
    if not clips:
        print("no clips found", file=sys.stderr)
        return 1
    bad = 0
    for c in clips:
        s = probe(c)
        w, h = int(s["width"]), int(s["height"])
        num, den = (s["avg_frame_rate"].split("/") + ["1"])[:2]
        fps = round(int(num) / max(int(den), 1))
        frames = int(s.get("nb_read_frames", 0))
        ok = ((w, h) in [(TARGET_W, TARGET_H), LANDSCAPE]
              and fps == TARGET_FPS and frames in VALID_FRAMES)
        if not ok:
            bad += 1
            print(f"  FAIL {c.name}: {w}x{h} {fps}fps {frames}f")
    print(f"\n{len(clips) - bad}/{len(clips)} clips conform "
          f"(need {TARGET_W}x{TARGET_H}, {TARGET_FPS}fps, frames in {VALID_FRAMES})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build clips from a segment sheet")
    b.add_argument("--segments", required=True)
    b.add_argument("--src-dir", required=True)
    b.add_argument("--out-dir", required=True)
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify", help="verify a built dataset against the hard spec")
    v.add_argument("--out-dir", required=True)
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
