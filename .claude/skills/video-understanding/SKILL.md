---
name: video-understanding
description: See and understand local video clips by extracting frames with ffmpeg and viewing them. Use when asked to watch, view, describe, analyze, curate, caption, filter, or quality-check a video file or a folder of clips — especially for building or cleaning a training dataset. Frames-only (no audio). Triggers on "watch this video", "look at this clip", "curate/filter these clips", "caption this video", "is this clip good for training".
---

# Video Understanding (frames-only, local files)

Claude cannot natively watch video. This skill makes a clip viewable by extracting
a small set of still frames with `ffmpeg`, then **you `Read` those frames** (the Read
tool renders images visually) and reason over them. No audio, no downloads, no API.

Primary purpose: **training-dataset curation** — decide keep/reject, caption, and
flag quality issues for candidate clips.

## Requirements
- `ffmpeg` and `ffprobe` on PATH (already verified present in this repo).
- Nothing else — the script is pure Python 3, no pip installs.

## Workflow

### 1. Extract frames
```bash
python3 .claude/skills/video-understanding/scripts/extract_frames.py <CLIP> [options]
```
Common options:
- `--max-frames N` (default 24) — cap; keeps first + last, evenly spaced.
- `--width W` (default 1024) — 1024 general; 1568 for fine detail / on-screen text; 768 to save tokens.
- `--mode uniform|scene|keyframe` (default `uniform`) — `uniform` for short clips (even coverage, best for judging motion/consistency); `scene` for long/edited footage; `keyframe` fastest/sparsest.
- `--scene-threshold 0.3` — only for `--mode scene` (0.2 slide/subtle, 0.4 fast-cut).
- `--no-dedup` — keep near-duplicate frames (default drops them via mpdecimate).

The script prints a JSON summary (stdout) and a `FRAME INDEX` with `t=MM:SS`
markers and file paths (stderr).

### 2. View the frames
`Read` each frame path in order. Use the `t=MM:SS` markers to reason temporally
(what changes across the clip → motion, flicker, consistency).

### 3. Emit a structured record (dataset curation)
Produce **one JSON object per clip** using this schema:

```json
{
  "clip": "<path>",
  "duration_sec": 0.0,
  "caption": "<clean, training-ready description of subject + action>",
  "style_match": "on_style | off_style | uncertain",
  "motion": "static | low | good | erratic",
  "temporal_consistency": "clean | minor_flicker | morphing | popping",
  "cut_count": 1,
  "quality_flags": ["blur","compression_artifacts","watermark","logo","letterbox","pillarbox","text_overlay","subtitles","interlacing"],
  "verdict": "keep | reject | needs_trim",
  "reason": "<one line: why this verdict>"
}
```

Rubric guidance (this is a 2D-animation / Pudgy Penguins style dataset):
- **style_match** — `on_style` = clean 2D animation consistent with the Pudgy IP look; `off_style` = photoreal, 3D render, live action, or a clashing art style.
- **motion** — the dataset wants real motion. A clip whose frames barely change is `static`/`low` and usually a weak training sample.
- **temporal_consistency** — watch for the character morphing, limbs/features popping, or flicker between frames (common in generated clips).
- **cut_count** — count distinct shots. Multi-cut clips (`cut_count > 1`) are usually `needs_trim` (split into single continuous shots).
- **quality_flags** — anything that would teach the model bad habits: watermarks, logos, burned-in text/subtitles, black bars (letterbox/pillarbox), heavy compression, blur, interlacing.
- **verdict** — `keep` clean single-shot on-style clips with good motion; `reject` off-style / heavily flagged / static; `needs_trim` when it's salvageable after cropping bars or splitting cuts.

### 4. Batch a folder → manifest
When given a directory, loop over the video files and append one JSON object per
clip as a line to a JSONL manifest (default `data/manifest.jsonl` unless told
otherwise). This manifest is the deliverable: filter `verdict == "keep"` to build
the cleaned set, and use `caption` for training text.

```bash
for f in <DIR>/*.mp4; do
  python3 .claude/skills/video-understanding/scripts/extract_frames.py "$f" --json
done
```
(Read the frames per clip between runs; write the record to the manifest.)

## Notes
- Token budget: Claude image cost ≈ `ceil(w*h/750)` per frame; ~24 frames @1024px is a few thousand tokens — you can curate several short clips per turn. Lower `--max-frames` or `--width` for long clips.
- Frames are written to a temp dir by default (`--out-dir` to override). They're
  disposable scratch — do not commit them.
