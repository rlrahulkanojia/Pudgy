#!/usr/bin/env python3
"""
Verify the corpus mp4s against their PNGs and against DiffSynth's video loader (G0.2e).

For every clip in manifest.json (train, holdout, calib):

  1. container   the mp4 holds exactly N + C.LOADER_TAIL frames (ffprobe packet count);
  2. bit-exact   ffmpeg's rgb24 decode of frames 0..N-1 equals the PNGs byte for byte, and
                 the tail frame equals frame N-1;
  3. loader      DiffSynth's training loader yields exactly N frames, identical to the PNGs.
                 On the box this imports the real `diffsynth.core.data.operators.LoadVideo`
                 with train.py's settings (17n+5, 24 fps, fix_frame_rate=True,
                 ImageCropAndResize 768x768 /32). Elsewhere it falls back to a faithful copy
                 of FrameSamplerByRateMixin + LoadVideo (DiffSynth-Studio acf2ad2,
                 diffsynth/core/data/operators.py) over imageio's v2 reader, which is what
                 DiffSynth calls.

It also checks that every training clip appears in exactly one metadata_f<N>.csv, with
`video` == `input_audio` and the bucket matching the clip length.

`--demo-tail` re-encodes one clip per bucket WITHOUT the tail frame into a temp dir and
reports what the loader makes of it: the reason the tail exists (MODEL_NOTES 2.17).

Usage:
    python expv2-minimax-h3/prep/check_loader.py                     # every clip
    python expv2-minimax-h3/prep/check_loader.py --limit 50 --demo-tail
    python expv2-minimax-h3/prep/check_loader.py --report $EXPV2_WORK/eval/phase0/check_loader.json
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv2_common as C  # noqa: E402

TIME_DIV, TIME_REM = 17, 5      # train.py MINIMAX_H3_TIME_DIVISION_FACTOR / _REMAINDER


# ---------------------------------------------------------------------------------------
# The loader under test
# ---------------------------------------------------------------------------------------
def real_loader(n):
    """DiffSynth's own operator, configured exactly as examples/minimax_h3/model_training/
    train.py builds it (default_video_operator -> LoadVideo)."""
    from diffsynth.core.data.operators import ImageCropAndResize, LoadVideo
    return LoadVideo(n, TIME_DIV, TIME_REM,
                     frame_processor=ImageCropAndResize(C.SIZE, C.SIZE, 1024 * 1024, 32, 32),
                     frame_rate=C.FPS, fix_frame_rate=True)


class FallbackLoadVideo:
    """Copy of DiffSynth-Studio acf2ad2 FrameSamplerByRateMixin + LoadVideo (operators.py
    l.137-195), with ImageCropAndResize reduced to the identity it is for 768² input (the
    real one resizes with torchvision BILINEAR, which PIL short-circuits at equal size)."""

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.time_division_factor = TIME_DIV
        self.time_division_remainder = TIME_REM
        self.frame_rate = C.FPS
        self.fix_frame_rate = True

    def get_reader(self, data):
        import imageio
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        duration = meta_data["duration"] if "duration" in meta_data else \
            total_original_frames / meta_data["fps"]
        return int(math.floor(duration * self.frame_rate))

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id, raw_frame_rate, total_raw_frames):
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        frame_id = int(round(target_time_in_seconds * raw_frame_rate))
        return min(frame_id, total_raw_frames - 1)

    def __call__(self, data):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()["fps"]
        num_frames = self.get_num_frames(reader)
        total_raw_frames = reader.count_frames()
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = Image.fromarray(reader.get_data(frame_id))
            if frame.size != (C.SIZE, C.SIZE):
                raise ValueError(f"fallback crop-resize only covers {C.SIZE}², got {frame.size}")
            frames.append(frame)
        reader.close()
        return frames


def make_loader(n, mode):
    return real_loader(n) if mode == "diffsynth" else FallbackLoadVideo(n)


def loader_mode():
    try:
        real_loader(C.FLOOR)
        return "diffsynth"
    except Exception:  # noqa: BLE001
        try:
            import imageio  # noqa: F401
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            sys.exit("neither diffsynth nor imageio + imageio-ffmpeg is importable")
        return "fallback"


# ---------------------------------------------------------------------------------------
def ffmpeg_rgb(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, C.SIZE, C.SIZE, 3)


def packets(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def check_clip(corpus, clip, mode):
    n = clip["frames"]
    mp4, png_dir = corpus / clip["mp4"], corpus / clip["png_dir"]
    res = {"clip_id": clip["clip_id"], "frames": n, "errors": []}
    pngs = sorted(png_dir.glob("*.png"))
    if len(pngs) != n:
        res["errors"].append(f"{len(pngs)} PNGs, want {n}")
        return res
    ref = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs])
    k = packets(mp4)
    if k != n + C.LOADER_TAIL:
        res["errors"].append(f"container has {k} frames, want {n + C.LOADER_TAIL}")
    dec = ffmpeg_rgb(mp4)
    if len(dec) < n or not np.array_equal(dec[:n], ref):
        bad = [i for i in range(min(n, len(dec))) if not np.array_equal(dec[i], ref[i])]
        res["errors"].append(f"ffmpeg decode differs from PNGs at frames {bad[:8]} "
                             f"(decoded {len(dec)})")
    elif len(dec) > n and not all(np.array_equal(dec[i], ref[-1]) for i in range(n, len(dec))):
        res["errors"].append("tail frame is not a copy of the last frame")
    try:
        got = make_loader(n, mode)(str(mp4))
        res["loader_frames"] = len(got)
        if len(got) != n:
            res["errors"].append(f"loader yields {len(got)} frames, want {n}")
        else:
            bad = [i for i, f in enumerate(got) if not np.array_equal(np.asarray(f.convert("RGB")), ref[i])]
            if bad:
                res["errors"].append(f"loader frames differ from PNGs at {bad[:8]}")
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"loader raised {type(e).__name__}: {e}")
    return res


def check_csvs(corpus, clips):
    errs = []
    train = {c["clip_id"]: c for c in clips if c["split"] == "train"}
    seen = {}
    for n in C.BUCKETS:
        p = corpus / f"metadata_f{n}.csv"
        if not p.is_file():
            errs.append(f"missing {p.name}")
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            cid = Path(r["video"]).stem
            if r["video"] != r["input_audio"]:
                errs.append(f"{p.name}: {cid} video != input_audio")
            if cid in seen:
                errs.append(f"{cid} listed in {seen[cid]} and {p.name}")
            seen[cid] = p.name
            c = train.get(cid)
            if c is None:
                errs.append(f"{p.name}: {cid} is not a training clip")
            elif c["frames"] != n:
                errs.append(f"{p.name}: {cid} has {c['frames']} frames")
            elif r["prompt"] != c["caption"]:
                errs.append(f"{p.name}: {cid} prompt differs from manifest caption")
    missing = sorted(set(train) - set(seen))
    if missing:
        errs.append(f"{len(missing)} training clips in no CSV, e.g. {missing[:3]}")
    try:
        import pandas
        for n in C.BUCKETS:
            df = pandas.read_csv(corpus / f"metadata_f{n}.csv")
            if list(df.columns) != ["video", "prompt", "input_audio"]:
                errs.append(f"metadata_f{n}.csv columns {list(df.columns)}")
    except ImportError:
        pass
    return errs


def demo_tail(corpus, clips, mode):
    """Show what the loader does with an exact-length file (no tail)."""
    out = []
    with tempfile.TemporaryDirectory() as td:
        for n in C.BUCKETS:
            c = next((c for c in clips if c["frames"] == n), None)
            if c is None:
                continue
            frames = ffmpeg_rgb(corpus / c["mp4"])[:n]
            p = Path(td) / f"exact_f{n}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{C.SIZE}x{C.SIZE}", "-r", str(C.FPS), "-i", "-",
                            "-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24", "-an", str(p)],
                           input=frames.tobytes(), check=True)
            got = len(make_loader(n, mode)(str(p)))
            out.append({"frames": n, "loader_frames_without_tail": got,
                        "loader_frames_with_tail": n})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--corpus", type=Path, default=C.CORPUS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N clips")
    ap.add_argument("--report", type=Path, default=None,
                    help="JSON report path (default <corpus>/check_loader.json)")
    ap.add_argument("--demo-tail", action="store_true")
    args = ap.parse_args()

    man = json.loads((args.corpus / "manifest.json").read_text())
    clips = man["clips"]
    todo = clips[: args.limit] if args.limit else clips
    mode = loader_mode()
    import imageio
    print(f"loader: {mode}; imageio {imageio.__version__}; {len(todo)} clips", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda c: check_clip(args.corpus, c, mode), todo))
    failed = [r for r in res if r["errors"]]
    csv_errs = check_csvs(args.corpus, clips)
    by_len = {}
    for r in res:
        b = by_len.setdefault(r["frames"], {"clips": 0, "pass": 0})
        b["clips"] += 1
        b["pass"] += not r["errors"]
    report = {"loader": mode, "imageio": imageio.__version__, "loader_tail": C.LOADER_TAIL,
              "checked": len(res), "passed": len(res) - len(failed), "failed": len(failed),
              "by_length": by_len, "failures": failed[:100], "csv_errors": csv_errs}
    if args.demo_tail:
        report["demo_tail"] = demo_tail(args.corpus, clips, mode)
    C.write_json(args.report or args.corpus / "check_loader.json", report)

    for n, b in sorted(by_len.items()):
        print(f"  f{n}: {b['pass']}/{b['clips']} pass")
    for r in failed[:20]:
        print(f"  ✗ {r['clip_id']}: {'; '.join(r['errors'])}")
    for e in csv_errs[:20]:
        print(f"  ✗ csv: {e}")
    for d in report.get("demo_tail", []):
        print(f"  demo: exact-length f{d['frames']} file -> loader reads "
              f"{d['loader_frames_without_tail']} frames (with the tail: {d['frames']})")
    ok = not failed and not csv_errs
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
