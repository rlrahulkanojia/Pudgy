"""expv1 measurement functions — one definition of every number G0.5, Tier 2, §8 and §9 use.

Copied verbatim from expv2-minimax-h3/eval/metrics.py @ 4832f98 (expv1 never imports another
experiment's code, PLAN §5). Only this header changed; the definitions are expv2's, so a number
means the same thing in both experiments. Frame size here is 512 (expv2: 768).

Numpy + scipy.ndimage + PIL on CPU. The two embedding backends (CLIP, DINOv2) are imported
lazily and run on CUDA when present, else CPU.

Provenance (copied, never imported):
  - CLIP / DINOv2 model ids, the 6-frame back-half embedding, corner drift and the
    1-NN idea: finetune/wan/eval_v8/score_v8.py (b902bde);
  - centroid travel scaled to a 1024 px frame: ltx25/eval/eval_ltx25.py;
  - ground shadow width / head line as the zoom reference: docs/training_reports/v8/
    REPORT_v8.md §8 (the code there was ad hoc; the definitions below are expv2's).

Conventions:
  frames      uint8 array (T, H, W, 3), RGB, any square size (768 in expv2)
  pair diff   mean |grey(t+1) - grey(t)| over the union of the two character masks,
              0-255 grey levels. Motion energy is its mean; holds are pairs under h.
  masks       from a FLAT ground of known colour. The art's ground shadow is the ground
              colour x 0.8 (measured: (204,204,204) on white = black at alpha 0.2), so it
              is separated from the character by colour, then cleaned by an opening.
              The character's white belly equals the white ground; it is recovered by
              filling holes inside the black outline.
"""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

GROUND_TOL = 14          # max-channel distance that still counts as ground
SHADOW_FACTOR = 0.8      # shadow = ground x 0.8 (measured on the source art)
SHADOW_TOL = 14
OUTLINE_LUM = 80         # black outlines: luminance below this inside the character
HOLD_EPS = 0.1           # a source pair under this (grey levels) is an exact hold
HOLD_FRACTION = 0.25     # h = 25% of the label's median moving-pair difference (PLAN G0.5)

CLIP_ID = "openai/clip-vit-large-patch14"     # copied: score_v8.py
DINO_ID = "facebook/dinov2-base"              # copied: score_v8.py


# --------------------------------------------------------------------------- io

def read_png_dir(d, n=None):
    files = sorted(Path(d).glob("*.png"))
    if n is not None:
        files = files[:n]
    if not files:
        raise FileNotFoundError(f"no PNG frames in {d}")
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def read_png_dir_one(path):
    """One PNG (e.g. a start frame) -> (H, W, 3) uint8."""
    return np.asarray(Image.open(path).convert("RGB"))


def read_video(p, size=None):
    """Decode a video to (T, H, W, 3) uint8 with ffmpeg (copied from score_v8.read_video).
    `size` rescales with Lanczos (used for v8's 1024^2 clips -> 768^2)."""
    vf = ["-vf", f"scale={size}:{size}:flags=lanczos"] if size else []
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = (size, size) if size else (int(probe[0]), int(probe[1]))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), *vf, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def resize_frames(frames, size):
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.LANCZOS))
                     for f in frames])


# --------------------------------------------------------------------------- masks

def _nd():
    from scipy import ndimage   # imported lazily; `pip install scipy` on the box
    return ndimage


def grey(frames):
    return frames.astype(np.float32).mean(-1)


def luminance(frames):
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def corner_size(h):
    return max(8, int(round(48 * h / 1024)))      # v8's 48 px corners at 1024


def ground_estimate(frame):
    """Median colour of the four corner patches: the ground as rendered."""
    k = corner_size(frame.shape[0])
    c = np.concatenate([frame[:k, :k], frame[:k, -k:], frame[-k:, :k], frame[-k:, -k:]], 0)
    return np.median(c.reshape(-1, 3), axis=0)


def _keep_large(mask, min_px, rel=None):
    nd = _nd()
    lab, n = nd.label(mask)
    if n == 0:
        return mask
    sizes = nd.sum(mask, lab, index=np.arange(1, n + 1))
    floor = min_px if rel is None else max(min_px, rel * sizes.max())
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= floor
    return keep[lab]


def frame_masks(frame, ground_rgb=None):
    """-> dict(char, shadow, outline, fill, ground) boolean (H, W) masks for one frame."""
    nd = _nd()
    h = frame.shape[0]
    scale = (h / 768.0) ** 2
    g = ground_estimate(frame) if ground_rgb is None else np.asarray(ground_rgb, np.float32)
    f = frame.astype(np.float32)
    non_ground = np.abs(f - g).max(-1) > GROUND_TOL
    shadow_col = np.abs(f - g * SHADOW_FACTOR).max(-1) <= SHADOW_TOL
    shadow = nd.binary_opening(shadow_col & non_ground, iterations=2)
    shadow = _keep_large(shadow, int(400 * scale))
    char = nd.binary_fill_holes(non_ground & ~shadow)
    char = _keep_large(char, int(200 * scale), rel=0.01)
    lum = luminance(frame[None])[0]
    outline = nd.binary_dilation(char & (lum < OUTLINE_LUM), iterations=1) & char
    fill = char & ~outline
    away = nd.binary_dilation(char | shadow, iterations=8)
    return {"char": char, "shadow": shadow, "outline": outline, "fill": fill,
            "ground": ~away}


def clip_masks(frames, ground_rgb=None):
    """Masks per frame. ground_rgb=None estimates the ground from frame 0's corners and
    uses that for every frame (a drifting ground then shows up as drift, not as mask)."""
    g = ground_estimate(frames[0]) if ground_rgb is None else ground_rgb
    return [frame_masks(f, g) for f in frames]


def bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# --------------------------------------------------------------------------- motion

def pair_diffs(frames, masks):
    """Per consecutive pair: mean |grey diff| inside the union of the character masks."""
    g = grey(frames)
    out = []
    for t in range(len(frames) - 1):
        m = masks[t]["char"] | masks[t + 1]["char"]
        out.append(float(np.abs(g[t + 1] - g[t])[m].mean(dtype=np.float64)) if m.any() else 0.0)
    return np.array(out, np.float64)


def hold_threshold(all_pair_diffs):
    """PLAN G0.5: h = 25% of the label's median MOVING-pair difference. Sources hold
    exactly, so 'moving' = above HOLD_EPS. Pass the pair diffs of every source clip of the
    label, concatenated."""
    d = np.asarray(all_pair_diffs, np.float64)
    moving = d[d > HOLD_EPS]
    return float(HOLD_FRACTION * np.median(moving)) if len(moving) else float("nan")


def frozen_fraction(pd, h):
    if h is None or not np.isfinite(h) or len(pd) == 0:
        return None
    return float(np.mean(pd < h))


def twos_index(pd, h):
    """PLAN G0.5: share of pairs (2j, 2j+1) that are holds while (2j+1, 2j+2) are not,
    maximised over the phase (0 or 1). On twos -> ~1; on ones or frozen -> 0."""
    if h is None or not np.isfinite(h) or len(pd) < 2:
        return None
    hold = pd < h
    best = 0.0
    for phase in (0, 1):
        idx = np.arange(phase, len(hold) - 1, 2)
        if len(idx):
            best = max(best, float(np.mean(hold[idx] & ~hold[idx + 1])))
    return best


def centroid_travel(masks, width):
    """Summed displacement of the character-mask centroid, in px of a 1024 px frame."""
    cs = []
    for m in masks:
        ys, xs = np.where(m["char"])
        cs.append((ys.mean(), xs.mean()) if len(ys) else (np.nan, np.nan))
    c = np.array(cs)
    step = np.sqrt((np.diff(c, axis=0) ** 2).sum(1))
    return float(np.nansum(step) * 1024.0 / width)


def shadow_widths(masks):
    out = []
    for m in masks:
        b = bbox(m["shadow"])
        out.append(np.nan if b is None else float(b[3] - b[2] + 1))
    return np.array(out)


def zoom_estimate(masks):
    """Signed apparent scale change, median over the last third of frames vs frame 0.
    Primary reference: ground-shadow width (v8 §8: feet planted, so a shrinking shadow
    reads as a camera pull-back). Fallback when frame 0 or most frames have no shadow:
    character bbox height. Returns (zoom, source, head_line_drift)."""
    T = len(masks)
    tail = slice(max(1, (2 * T) // 3), T)
    w = shadow_widths(masks)
    if np.isfinite(w[0]) and np.isfinite(w).mean() >= 0.5 and np.isfinite(w[tail]).any():
        z, src = float(np.nanmedian(w[tail] / w[0]) - 1.0), "shadow"
    else:
        hs = []
        for m in masks:
            b = bbox(m["char"])
            hs.append(np.nan if b is None else float(b[1] - b[0] + 1))
        hs = np.array(hs)
        z = float(np.nanmedian(hs[tail] / hs[0]) - 1.0) if np.isfinite(hs[0]) else float("nan")
        src = "bbox"
    tops, h0 = [], None
    for m in masks:
        b = bbox(m["char"])
        tops.append(np.nan if b is None else float(b[0]))
        if h0 is None and b is not None:
            h0 = float(b[1] - b[0] + 1)
    tops = np.array(tops)
    head = (float(np.nanmedian((tops[tail] - tops[0]) / h0))
            if h0 and np.isfinite(tops[0]) else float("nan"))
    return z, src, head


# --------------------------------------------------------------------------- colour / ground

def corner_drift(frames, rgb):
    """Mean |corner pixels - expected ground RGB| over the clip, grey levels (the '/255'
    of the bars). Corners are v8's 48 px at 1024, scaled. Copied from score_v8.corner_drift."""
    k = corner_size(frames.shape[1])
    c = np.concatenate([frames[:, :k, :k], frames[:, :k, -k:], frames[:, -k:, :k],
                        frames[:, -k:, -k:]], 1)
    return float(np.abs(c.astype(np.float64) - np.array(rgb, np.float64)).mean())


def ground_spread(frames, masks):
    """Pixel spread of the ground: per-channel std of ground pixels (8 px away from the
    character and its shadow), averaged over channels, median over frames."""
    vals = []
    for f, m in zip(frames, masks):
        px = f[m["ground"]].astype(np.float64)
        if len(px) > 100:
            vals.append(float(px.std(0).mean()))
    return float(np.median(vals)) if vals else float("nan")


def saturation(frames, masks):
    """Mean HSV saturation over the character's fill (character minus outline; the
    black outline has no hue), median over frames. 0-1."""
    vals = []
    for f, m in zip(frames, masks):
        px = f[m["fill"]].astype(np.float64)
        if len(px) == 0:
            continue
        mx, mn = px.max(1), px.min(1)
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        vals.append(float(s.mean()))
    return float(np.median(vals)) if vals else float("nan")


def fill_flatness(frames, masks, k=5):
    """Two readings of PLAN G0.5's 'luminance std inside the character mask minus the
    outline mask':
      local   mean over fill pixels of the k x k local luminance std (shading / texture;
              flat art -> small, 3D drift -> large). PRIMARY - used by V11.
      global  the literal std over the whole fill region (mixes the character's several
              flat colours, so it measures palette, not flatness). Reported only.
    Median over frames of each."""
    nd = _nd()
    loc, glob = [], []
    lum = luminance(frames).astype(np.float64)
    for L, m in zip(lum, masks):
        fill = nd.binary_erosion(m["fill"], iterations=1)
        if fill.sum() < 50:
            continue
        mu = nd.uniform_filter(L, k)
        var = np.maximum(nd.uniform_filter(L * L, k) - mu * mu, 0.0)
        loc.append(float(np.sqrt(var)[fill].mean()))
        glob.append(float(L[fill].std()))
    return ((float(np.median(loc)) if loc else float("nan")),
            (float(np.median(glob)) if glob else float("nan")))


PSNR_CAP = 99.0          # identical images: report 99 dB, not inf (inf is not JSON)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, float(20 * np.log10(255.0 / np.sqrt(mse))))


# --------------------------------------------------------------------------- embeddings

def square_crop(frame, mask, pad=16, fill=None):
    """Square crop around the mask's bbox (padded), filled with the ground colour."""
    b = bbox(mask)
    H, W = mask.shape
    if b is None:
        return Image.fromarray(frame)
    y0, y1, x0, x1 = b
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0) + 2 * pad
    t, l = int(round(cy - side / 2)), int(round(cx - side / 2))
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = ground_estimate(frame).astype(np.uint8) if fill is None else fill
    sy0, sx0 = max(t, 0), max(l, 0)
    sy1, sx1 = min(t + side, H), min(l + side, W)
    canvas[sy0 - t:sy1 - t, sx0 - l:sx1 - l] = frame[sy0:sy1, sx0:sx1]
    return Image.fromarray(canvas)


class Embedder:
    """CLIP ViT-L/14 image features and DINOv2-base CLS, L2-normalised (score_v8's models)."""

    def __init__(self, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(CLIP_ID).to(self.dev).eval()
        self.cp = CLIPProcessor.from_pretrained(CLIP_ID)
        self.dino = AutoModel.from_pretrained(DINO_ID).to(self.dev).eval()
        self.dp = AutoImageProcessor.from_pretrained(DINO_ID)

    def revisions(self):
        return {CLIP_ID: getattr(self.clip.config, "_commit_hash", None),
                DINO_ID: getattr(self.dino.config, "_commit_hash", None)}

    def _norm(self, x):
        return self.torch.nn.functional.normalize(x, dim=-1)

    def clip_images(self, imgs):
        t = self.torch
        with t.no_grad():
            o = self.clip.get_image_features(**self.cp(images=list(imgs), return_tensors="pt").to(self.dev))
            o = o if t.is_tensor(o) else o.pooler_output
        return self._norm(o.float()).cpu().numpy()

    def dino_images(self, imgs):
        t = self.torch
        with t.no_grad():
            o = self.dino(**self.dp(images=list(imgs), return_tensors="pt").to(self.dev)).last_hidden_state[:, 0]
        return self._norm(o.float()).cpu().numpy()

    def clip_embed(self, frames):
        """score_v8.embed: 6 frames spread over the back half (the label's peak), mean of
        normalised embeddings, renormalised. Returns (clip, dino)."""
        idx = np.linspace(len(frames) // 2, len(frames) - 1, 6).astype(int)
        imgs = [Image.fromarray(frames[i]) for i in idx]
        c = self.clip_images(imgs).mean(0)
        d = self.dino_images(imgs).mean(0)
        return c / (np.linalg.norm(c) + 1e-9), d / (np.linalg.norm(d) + 1e-9)

    def dino_sim(self, img_a, img_b):
        e = self.dino_images([img_a, img_b])
        return float(e[0] @ e[1])


# --------------------------------------------------------------------------- one clip

def clip_measures(frames, ground_rgb, h=None, start_frame=None, embedder=None, masks=None):
    """Every per-clip scalar. ground_rgb: the EXPECTED ground (corner drift reference). h: the label's hold threshold (from the source baseline) - frozen
    fraction and twos index are None without it. start_frame: the supplied start frame
    (first-frame fidelity). embedder: an Embedder for the DINOv2 measures, else None."""
    # Masks from each frame's OWN corner estimate of the ground, so a drifting ground in a
    # generation shows up in corner_drift / ground_spread instead of corrupting the masks.
    # ground_rgb (the expected colour) is the drift reference only.
    masks = masks or [frame_masks(f, None) for f in frames]
    pd = pair_diffs(frames, masks)
    z, zsrc, head = zoom_estimate(masks)
    flat_local, flat_global = fill_flatness(frames, masks)
    sw = shadow_widths(masks)
    out = {
        "frames": int(len(frames)),
        "motion_energy": float(pd.mean()) if len(pd) else 0.0,
        "frozen_fraction": frozen_fraction(pd, h),
        "twos_index": twos_index(pd, h),
        "centroid_travel": centroid_travel(masks, frames.shape[2]),
        "shadow_width0": float(sw[0]) if np.isfinite(sw[0]) else None,
        "zoom": z, "zoom_ref": zsrc, "head_line_drift": head,
        "corner_drift": corner_drift(frames, ground_rgb),
        "ground_spread": ground_spread(frames, masks),
        "saturation": saturation(frames, masks),
        "fill_flatness": flat_local, "fill_std_global": flat_global,
        "char_area0": int(masks[0]["char"].sum()),
        "late_identity": None, "ff_psnr": None, "ff_dino": None,
    }
    if start_frame is not None:
        out["ff_psnr"] = psnr(frames[0], start_frame)
    if embedder is not None:
        out["late_identity"] = embedder.dino_sim(square_crop(frames[0], masks[0]["char"]),
                                                 square_crop(frames[-1], masks[-1]["char"]))
        if start_frame is not None:
            sm = frame_masks(start_frame, None)["char"]
            union = sm | masks[0]["char"]
            out["ff_dino"] = embedder.dino_sim(square_crop(start_frame, union),
                                               square_crop(frames[0], union))
    return out, pd


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summarise(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if len(v) == 0:
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


# --------------------------------------------------------------------------- the real-clip bank

def load_bank(corpus):
    """CORPUS/bank/index.json -> list of normalised entries (INTERFACES §1). Tolerates a
    bare list or {"clips": [...]}, and lower-case character names."""
    import json
    corpus = Path(corpus)
    raw = json.loads((corpus / "bank" / "index.json").read_text())
    rows = raw.get("clips", raw.get("bank", [])) if isinstance(raw, dict) else raw
    out = []
    for r in rows:
        bid = r.get("bank_id") or r.get("id")
        char = str(r["char"]).capitalize()
        label = r["label"]
        out.append({"bank_id": bid, "char": char, "label": label,
                    "kind": r.get("kind") or ("motion" if label in
                                              ("walking", "waving", "sitting", "jumping", "running")
                                              else "expression"),
                    "angle": r.get("angle"), "frames": int(r.get("frames", 0)),
                    "split": r.get("split"), "untrained": bool(r.get("untrained", label == "running")),
                    "source": r.get("source"),
                    "png_dir": corpus / "bank" / "png" / bid})
    return out


def bank_embeddings(embedder, bank, window, cache_dir=None):
    """CLIP and DINOv2 embeddings of every bank clip at frames [0:window] (clips shorter
    than the window use all their frames). Cached per window as an .npz."""
    import json
    ids = [b["bank_id"] for b in bank]
    cache = Path(cache_dir) / f"bank_emb_w{window}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids and json.loads(str(z["rev"])) == embedder.revisions():
            return z["clip"], z["dino"]
    C, D = [], []
    for b in bank:
        c, d = embedder.clip_embed(read_png_dir(b["png_dir"], window))
        C.append(c); D.append(d)
    C, D = np.stack(C), np.stack(D)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(ids), clip=C, dino=D, rev=json.dumps(embedder.revisions()))
    return C, D


def nn_label(q, bank_vecs, bank, char, labels, exclude_angle=None):
    """1-NN label of query embedding q among bank clips of the same character whose label
    is in `labels`. exclude_angle drops every bank clip at that angle, for EVERY label
    (leave-angle-out): the start frame came from one of them, and dropping only the true
    label's clip would bias the vote toward the other labels' same-angle clips."""
    idx = [i for i, b in enumerate(bank) if b["char"] == char and b["label"] in labels
           and (exclude_angle is None or b["angle"] != exclude_angle)]
    if not idx:
        return None, None
    s = bank_vecs[idx] @ q
    j = int(np.argmax(s))
    return bank[idx[j]]["label"], float(s[j])


# --------------------------------------------------------------------------- the manual rubric (V2, §8)
#
# A rubric file is filled by a person (or an agent viewing the clips), one row per clip:
#   {"schema": "expv2-rubric-1", "instructions": "...", "rater": "<name>", "created": "...",
#    "clips": [{"rid": "r001", "view": "<path to mp4 or PNG dir>",
#               "flat_fills": true|false, "outlines_intact": true|false,
#               "no_3d_shading": true|false, "on_model": true|false,
#               "colour_correct": true|false, "washed_out": true|false, "notes": "",
#               ... identifying fields (item_id, seed, char, label, angle) unless blinded}]}
# Blinded files (the §8 head-to-head) keep the identity in a separate *_key.json:
#   {"<rid>": {"model": "h3" | "v8@550", "char": ..., "label": ..., "angle": ...}}
# A clip PASSES the rubric when flat_fills, outlines_intact, no_3d_shading and on_model are
# all true. colour_correct is its own bar (V2: 100%). washed_out feeds the arm-D trigger.

RUBRIC_FIELDS = ("flat_fills", "outlines_intact", "no_3d_shading", "on_model")
RUBRIC_EXTRA = ("colour_correct", "washed_out")
RUBRIC_INSTRUCTIONS = (
    "For each clip answer true/false. flat_fills: colour regions are flat, no gradients or "
    "texture. outlines_intact: thick clean black outlines everywhere, nothing dissolved or "
    "doubled. no_3d_shading: no rendered shading, specular light or depth lighting. on_model: "
    "the character's shape and features match Pax/Polly. colour_correct: Pax blue, Polly pink "
    "(rosy cheeks only when blushing). washed_out: the clip looks foggy / desaturated / low "
    "contrast. Leave notes for anything else. Judge at 100% and at 4x on a few frames.")


def rubric_row(view, rid=None, **ident):
    row = {"rid": rid, "view": str(view), **ident}
    row.update({k: None for k in RUBRIC_FIELDS + RUBRIC_EXTRA})
    row["notes"] = ""
    return row


def rubric_template(rows):
    import datetime
    return {"schema": "expv2-rubric-1", "instructions": RUBRIC_INSTRUCTIONS, "rater": None,
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "clips": rows}


def rubric_pass(row):
    """True/False when every field is filled, else None (unrated)."""
    vals = [row.get(k) for k in RUBRIC_FIELDS]
    if any(v is None for v in vals):
        return None
    return all(bool(v) for v in vals)
