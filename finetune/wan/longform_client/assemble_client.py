"""Join a video's chained takes into one continuous ~10 s client video, and QA it.

python assemble_client.py <id> [mode=chain]   -> final/<id>_1080x1920.mp4 + qa/<id>.png + qa/<id>.json

Takes k >= 2 start on the frame take k-1 ended on (its last frame, or the keyframe in loop mode),
so their frame 0 is a duplicate and is dropped. Loop takes decode 85 frames: the 4 after frame 80
are the end-image latent itself (they move away from the keyframe again) and are cut.
Upscale 720x1280 -> 1080x1920 lanczos + luma unsharp, H.264 CRF 16, yuv420p, faststart
(the September long-form settings, finetune/wan/longform/assemble.py).

QA (the contact sheet is the verdict):
  size     character bbox height per frame / frame 0's (mask = pixels away from the clean set
           plate) - catches the motion zoom-out (REPORT_v8 §8)
  drift    mean |frame - plate| on the frame border, start vs end (camera creep, colour shift)
  seam     frame-to-frame change at each take join / median change (hitch at the join)
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/workspace/longform_client")
sys.path.insert(0, str(ROOT))
import keyframes as K                   # noqa: E402
W, H = 720, 1280
FRAMES = 81                             # produce.FRAMES
# frames cut from the end of a video: blushing's take 3 (door set) has a figure peeking in at the open door
# from its frame 73 (video frame 233) - every blushing take that reached the end of a part brought a visitor
TRIM_END = {"blushing": 8}


def border_mask(h, w):
    """The frame border (set only, in every shot): 60/40 px at 720x1280, scaled for other sizes."""
    m = np.zeros((h, w), bool)
    bh, bw = round(60 * h / H), round(40 * w / W)
    m[:bh] = m[-bh:] = True; m[:, :bw] = m[:, -bw:] = True
    return m


BORDER = border_mask(H, W)


def decode(p):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-i", str(p), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def match_colour(f, before, first):
    """The loop take (first+last frame) re-grades the palette within ~8 frames and keeps it (walking:
    wall -13,-5,+6 RGB, blue body -20,-16,0, whites and black ~unchanged), so the loop version jumped
    colour at its last join and again when it repeats. Per-channel tone curves, fitted from the
    take's last frame to the video's first (first+last frame lands on the same pose), undo the grade
    (a quadratic RGB map was tried: it fits the pastels but turned the black outlines navy). Each
    frame gets them in proportion to how far its set border has shifted from the frame the take
    continues from (`before`), so both joins match."""
    src, dst = f[-1].reshape(-1, 3).astype(np.int32), first.reshape(-1, 3).astype(np.int32)
    same = (np.abs(src - dst) < 40).all(1)      # drops the pixels whose content differs (pose)
    v = np.arange(256)
    lut = np.empty((3, 256), np.float64)
    for c in range(3):
        s_c, d_c = src[same, c], dst[same, c]
        n = np.bincount(s_c, minlength=256)
        med = np.array([np.median(d_c[s_c == i]) - i if n[i] >= 20 else np.nan for i in v])
        ok = ~np.isnan(med)
        off = np.interp(v, v[ok], med[ok])         # constant offset beyond the measured range
        off = np.convolve(np.pad(off, 4, mode="edge"), np.ones(9) / 9, "valid")
        lut[c] = np.maximum.accumulate(np.clip(v + off, 0, 255))
    b = border_mask(*f.shape[1:3])
    s0, e = before[b].mean(0), f[-1][b].mean(0)
    out = []
    for fr in f:
        k = np.clip(np.dot(fr[b].mean(0) - s0, e - s0) / max(np.dot(e - s0, e - s0), 1e-6), 0, 1)
        x = fr.astype(np.float64)
        g = np.stack([lut[c][fr[..., c]] for c in range(3)], -1)
        out.append(np.clip(x + k * (g - x), 0, 255).round().astype(np.uint8))
    return np.stack(out)


def plate_for(vid):
    """The clean set plate a video's keyframe was composited on, at the generation size."""
    meta = json.loads((ROOT / "keyframes/keyframes.json").read_text())
    kk = next(k for k in meta if k == vid or k.startswith(f"{vid}_"))
    return np.asarray(Image.open(K.BG / K.PLATES[meta[kk]["plate"]]).convert("RGB").resize((W, H), Image.LANCZOS),
                      np.float32)


def key_frame(vid):
    """The composited keyframe a video starts on, at the generation size."""
    meta = json.loads((ROOT / "keyframes/keyframes.json").read_text())
    kk = next(k for k in meta if k == vid or k.startswith(f"{vid}_"))
    return np.asarray(Image.open(ROOT / f"keyframes/{kk}.png").convert("RGB").resize((W, H), Image.LANCZOS))


def end_drift(take, vid):
    """How far the character moved, as shares of the frame height: (height ratio, feet line at 2.1-2.5 s vs the last
    0.3 s (medians), ground line - the lowest point, shadow included - at the take's start vs its end (maxima, so a
    jump in the air doesn't count)). Seed 42 walked off, shrank or lurched forward at the end of 5 of 11 takes
    (feet -0.08..+0.05); jumping seed 42 shrank Pax within its first 0.6 s (ground -0.12)."""
    plate, s = plate_for(vid), []
    for fr in decode(take)[:FRAMES].astype(np.float32):
        ys = np.where((np.abs(fr - plate).max(-1) > 40).sum(1) > 3)[0]
        s.append((ys.max() - ys.min(), ys.max()) if len(ys) else (np.nan, np.nan))
    s = np.array(s, np.float32)
    mid, end = np.nanmedian(s[50:61], 0), np.nanmedian(s[74:81], 0)
    ground = (np.nanmax(s[70:81, 1]) - np.nanmax(s[0:9, 1])) / H
    # sideways: the centre of Pax-blue / Polly-pink pixels, start vs end (surprised seed 42 shot Pax from x .48 to
    # .94 in its last 6 frames, 0.21 by these windows; clean takes so far <= 0.09)
    cx = []
    for fr in decode(take)[:FRAMES]:
        hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
        m = (cv2.inRange(hsv, *BODY_PAX) | cv2.inRange(hsv, *BODY_POLLY)).sum(0)
        cx.append(np.average(np.arange(W), weights=m) / W if m.any() else np.nan)
    cx = np.array(cx)
    side = float(np.nanmedian(cx[74:81]) - np.nanmedian(cx[0:9]))
    return float(end[0] / max(mid[0], 1)), float((end[1] - mid[1]) / H), float(ground), side


BODY_PAX, BODY_POLLY = ((100, 90, 120), (125, 255, 255)), ((150, 40, 150), (178, 255, 255))   # dupcheck's HSV


def drift_ok(h, feet, ground, vid="", side=0.0):
    """Ground line start->end within 0.03 (clean takes +-0.02; jumping/walk_together/crying/blushing seed 42:
    0.04-0.12), and feet line mid->end within 0.02 (catches the late slides: running, waving seed 42) except for
    jumping, which is in the air mid-take. Height is reported only (running t3 read x0.81 from the background)."""
    return abs(ground) <= 0.03 and (vid == "jumping" or abs(feet) <= 0.02) and abs(side) <= 0.15


def main(vid, mode="chain", loop=False):
    """loop: the looping version - takes 1-2 plus t3loop (aimed back at the keyframe)."""
    takes = sorted((ROOT / "takes").glob(f"{vid}_{mode}_t[0-9].mp4"))
    if loop:
        takes = [t for t in takes if not t.stem.endswith("_t3")] + [ROOT / f"takes/{vid}_{mode}_t3loop.mp4"]
    if not takes:
        sys.exit(f"no takes for {vid} {mode}")
    parts, seams = [], []
    for i, t in enumerate(takes):
        f = decode(t)
        if mode == "loop" or t.stem.endswith("loop"):   # an end-image take decodes 4 more: the end latent
            f = f[:FRAMES]
        if mode == "loop":                     # every take pinned to the keyframe at both ends: undo the palette shift
            key = key_frame(vid)
            f = match_colour(f, key, key)
        if i:                                  # chain: = last frame of take k-1; loop: = the keyframe both end on
            f = f[1:]
        if parts:
            seams.append(sum(len(p) for p in parts))
        parts.append(f)
    if loop:
        parts[-1] = match_colour(parts[-1], parts[-2][-1], parts[0][0])
    v = np.concatenate(parts)
    if not loop and TRIM_END.get(vid if mode == "chain" else ""):
        v = v[:len(v) - TRIM_END[vid]]
    out = ROOT / "final"; out.mkdir(exist_ok=True)
    name = (vid if mode == "chain" else f"{vid}_{mode}") + ("_loop" if loop else "")
    dst = out / f"{name}_1080x1920.mp4"
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-vf",
                            "scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.0,format=yuv420p",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart",
                            "-an", str(dst)], stdin=subprocess.PIPE)
    enc.stdin.write(v.tobytes()); enc.stdin.close()
    if enc.wait():
        sys.exit("encode failed")

    # QA against the clean set plate
    plate = plate_for(vid)
    vf = v.astype(np.float32)
    hts = []
    for fr in vf:
        m = np.abs(fr - plate).max(-1) > 40
        ys = np.where(m.sum(1) > 3)[0]
        hts.append(ys.max() - ys.min() if len(ys) else 0)
    hts = np.array(hts, np.float32) / max(hts[0], 1)
    drift = [float(np.abs(vf[i] - plate)[BORDER].mean()) for i in (0, len(vf) - 1)]
    d = np.abs(np.diff(vf[:, ::4, ::4], axis=0)).mean((1, 2, 3))
    med = float(np.median(d))
    seam = [round(float(d[s - 1] / max(med, 1e-6)), 1) for s in seams]
    qa = dict(video=dst.name, frames=len(v), seconds=round(len(v) / 24, 2), takes=len(takes),
              size_min=round(float(hts.min()), 3), size_end=round(float(hts[-5:].mean()), 3),
              border_drift_start_end=[round(x, 2) for x in drift], seam_jump_x_median=seam,
              max_jump_x_median=round(float(d.max() / max(med, 1e-6)), 1),
              frozen=round(float((d < 0.15).mean()), 3), loop_colour_matched=loop,
              # last frame -> first frame, as when the video repeats (loop versions: should be ~1)
              wrap_jump_x_median=round(float(np.abs(vf[-1, ::4, ::4] - vf[0, ::4, ::4]).mean() / max(med, 1e-6)), 1))
    q = ROOT / "qa"; q.mkdir(exist_ok=True)
    (q / f"{name}.json").write_text(json.dumps(qa, indent=1))
    idx = np.linspace(0, len(v) - 1, 10).round().astype(int)
    sheet = np.concatenate([np.asarray(Image.fromarray(v[i]).resize((180, 320))) for i in idx], 1)
    Image.fromarray(sheet).save(q / f"{name}.png")
    print(json.dumps(qa))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "loop"]
    main(*a, loop="loop" in sys.argv[1:])
