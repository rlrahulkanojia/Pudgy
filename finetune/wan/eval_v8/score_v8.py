#!/usr/bin/env python3
"""Score generated suites with a SEMANTIC probe instead of pixel SSIM.

ANALYSIS_v7 measured the v7 G-C clips: nearest-neighbour on face-SSIM recovers the SEED
67-81% of the time and the LABEL at chance (5-19%, chance 14%), while CLIP image
embeddings recover the label 57-71% and the seed at chance. So:

  label_nn    leave-one-out 1-NN label accuracy on CLIP ViT-L/14 image embeddings,
              within (config, character, kind)                      - PRIMARY
  label_zs    CLIP zero-shot accuracy against one text phrase per label
  dino_nn     same as label_nn on DINOv2 CLS (a second, non-text-trained view)
  view_nn     1-NN recovery of the START VIEW - high is fine (framing), but label_nn
              must not collapse on the holdout view
  motion      mean frame-to-frame MAE over the character (animation energy)
  repeat      periodicity of the CLIP-embedding sequence (max autocorrelation after
              the first dip) - G-D: does a long cycle REPEAT or play once slowly?
  corner      background drift in the four corners (G-B, v5 bar 5/255)

Every number carries its n; with ~2-4 clips per label treat accuracies as +-0.1.

    python score_v8.py --suite core --configs golden v8@300
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL  # noqa: E402

PHRASE = {"happy": "smiling happily", "surprised": "surprised with wide eyes",
          "angry": "angry and scowling", "confused": "confused and puzzled",
          "crying": "crying with tears", "neutral": "calm with a neutral face",
          "laughing": "laughing with eyes squeezed shut", "blushing": "blushing shyly",
          "walking": "walking", "running": "running", "waving": "waving a flipper",
          "sitting": "sitting on the ground", "jumping": "jumping in the air"}

_M = {}


def models():
    if not _M:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _M.update(dev=dev, torch=torch,
                  clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval(),
                  cp=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
                  dino=AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval(),
                  dp=AutoImageProcessor.from_pretrained("facebook/dinov2-base"))
    return _M


def read_video(p, size=None):
    vf = ["-vf", f"scale={size}:{size}"] if size else []
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = (size, size) if size else (int(probe[0]), int(probe[1]))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), *vf, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def _feat(o):
    t = _M["torch"]
    return o if t.is_tensor(o) else o.pooler_output


def embed(v):
    """CLIP + DINO embeddings of the back half (the expression's peak), plus the per-frame
    CLIP sequence for periodicity."""
    m = models(); t = m["torch"]
    idx = np.linspace(len(v) // 2, len(v) - 1, 6).astype(int)
    frames = [v[i] for i in idx]
    with t.no_grad():
        c = _feat(m["clip"].get_image_features(**m["cp"](images=frames, return_tensors="pt").to(m["dev"])))
        c = t.nn.functional.normalize(c, dim=-1).mean(0)
        d = m["dino"](**m["dp"](images=frames, return_tensors="pt").to(m["dev"])).last_hidden_state[:, 0]
        d = t.nn.functional.normalize(d, dim=-1).mean(0)
        seq = _feat(m["clip"].get_image_features(**m["cp"](images=list(v), return_tensors="pt").to(m["dev"])))
        seq = t.nn.functional.normalize(seq, dim=-1)
    return (t.nn.functional.normalize(c, dim=0).cpu().numpy(),
            t.nn.functional.normalize(d, dim=0).cpu().numpy(), seq.cpu().numpy())


def text_emb(texts):
    m = models(); t = m["torch"]
    with t.no_grad():
        e = _feat(m["clip"].get_text_features(**m["cp"](text=texts, return_tensors="pt",
                                                        padding=True).to(m["dev"])))
    return t.nn.functional.normalize(e, dim=-1).cpu().numpy()


def motion_energy(v):
    g = v.astype(np.float32).mean(-1)
    bg = np.median(g[0][[0, 0, -1, -1], [0, -1, 0, -1]])
    mask = (np.abs(g - bg) > 12).any(0)
    if not mask.any():
        return 0.0
    return float(np.abs(np.diff(g, axis=0))[:, mask].mean())


def periodicity(seq):
    """Max normalised autocorrelation of the embedding sequence after its first dip:
    ~1 for a cycle that repeats, low for a single slow arc."""
    x = seq - seq.mean(0)
    n = len(x)
    ac = np.array([(x[:n - k] * x[k:]).sum() / ((x ** 2).sum() + 1e-9) for k in range(n // 2)])
    dips = np.where(np.diff(np.sign(np.diff(ac))) > 0)[0]
    return float(ac[dips[0] + 1:].max()) if len(dips) and dips[0] + 1 < len(ac) else 0.0


def corner_drift(v, rgb):
    c = np.concatenate([v[:, :48, :48], v[:, :48, -48:], v[:, -48:, :48], v[:, -48:, -48:]], 1)
    return float(np.abs(c.astype(np.float32) - np.array(rgb, np.float32)).mean())


def loo_nn(E, labels):
    S = E @ E.T
    np.fill_diagonal(S, -9)
    nn = S.argmax(1)
    return float(np.mean(np.array(labels)[nn] == np.array(labels)))


def score_dir(suite, d):
    meta = json.loads((d / "meta.json").read_text())
    rows = []
    for tag, m in meta.items():
        p = d / f"{tag}.mp4"
        if not p.exists():
            continue
        v = read_video(p)
        c, dn, seq = embed(v)
        rows.append({"tag": tag, **m, "clip": c, "dino": dn, "seq": seq,
                     "motion": motion_energy(v[:, ::4, ::4]), "repeat": periodicity(seq),
                     "frames": len(v), "v": v if suite == "gb" else None})
    out = {"n": len(rows)}
    if suite in ("core", "guide", "gb"):
        for key in ("label",):
            groups = {}
            for r in rows:
                if key in r:
                    groups.setdefault((r["char"], r.get("kind", "expression")), []).append(r)
            res = {}
            for (char, kind), rs in sorted(groups.items()):
                labs = [r["label"] for r in rs]
                if len(set(labs)) < 2:
                    continue
                C = np.stack([r["clip"] for r in rs]); D = np.stack([r["dino"] for r in rs])
                uniq = sorted(set(labs))
                T = text_emb([f"a 2D cartoon of a penguin {PHRASE[l]}" for l in uniq])
                zs = np.array(uniq)[(C @ T.T).argmax(1)]
                entry = {"n": len(rs), "labels": len(uniq), "chance": round(1 / len(uniq), 3),
                         "label_nn": round(loo_nn(C, labs), 3),
                         "dino_nn": round(loo_nn(D, labs), 3),
                         "label_zs": round(float(np.mean(zs == np.array(labs))), 3),
                         "motion": round(float(np.mean([r["motion"] for r in rs])), 3)}
                if "view" in rs[0]:
                    entry["view_nn"] = round(loo_nn(C, [r["view"] for r in rs]), 3)
                    for view in ("FRONT", "QF2_R"):
                        sub = [i for i, r in enumerate(rs) if r["view"] == view]
                        if not sub:
                            continue
                        entry[f"zs_{view}"] = round(float(np.mean(zs[sub] == np.array(labs)[sub])), 3)
                res[f"{char}/{kind}"] = entry
            out["control"] = res
    if suite == "compose":
        acts = sorted({r["motion"] for r in rows}); emos = sorted({r["expression"] for r in rows})
        Ta = text_emb([f"a 2D cartoon of a penguin {PHRASE[a]}" for a in acts])
        Te = text_emb([f"a 2D cartoon of a penguin {PHRASE[e]}" for e in emos])
        C = np.stack([r["clip"] for r in rows])
        out["motion_zs"] = round(float(np.mean(np.array(acts)[(C @ Ta.T).argmax(1)]
                                               == np.array([r["motion"] for r in rows]))), 3)
        out["expr_zs"] = round(float(np.mean(np.array(emos)[(C @ Te.T).argmax(1)]
                                             == np.array([r["expression"] for r in rows]))), 3)
        out["motion_energy"] = round(float(np.mean([r["motion"] for r in rows])), 3)
    if suite == "gd":
        out["clips"] = {r["tag"]: {"frames": r["frames"], "repeat": round(r["repeat"], 3),
                                   "motion": round(r["motion"], 3)} for r in rows}
    if suite == "gb":
        from common import UNSEEN_BG
        out["corner_drift"] = {r["tag"]: round(corner_drift(r["v"], UNSEEN_BG[r["bg"]][1]), 2)
                               for r in rows}
    return out


def score_suite(suite, configs):
    res = {}
    for cfg in configs:
        d = EVAL / "suites" / suite / cfg.replace("/", "_").replace(":", "_")
        if (d / "meta.json").exists():
            res[cfg] = score_dir(suite, d)
    dest = EVAL / "suites" / suite / "scores.json"
    old = json.loads(dest.read_text()) if dest.exists() else {}
    old.update(res)
    dest.write_text(json.dumps(old, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    a = ap.parse_args()
    score_suite(a.suite, a.configs)
