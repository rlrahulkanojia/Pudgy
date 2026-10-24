#!/usr/bin/env python3
"""Tier-2 for v9c-phone: generate (eval_v8/gen_v8.run, one model load per config) and score.

Suites (clip counts per config; ~4-6 min per clip on an A100 at 1024^2):
  phone     start = frame 0 of a phone clip (QF2_R never trained, FRONT trained), both
            characters, sitting + standing; prompt = the training caption at f41.     8 clips
  textcall  start = neutral FRONT (NO phone in frame); prompt = the standing texting
            caption. Does "the iFin phone" appear from the text alone? Expected weak in
            image-to-video - reported, not gated (Assets_v9 section 7).                  2 clips
  leak      start = neutral FRONT; prompts = v8 captions (waving, happy, sitting). A
            phone must NOT appear.                                                       6 clips

Scores (per clip, written to <suite>/scores.json):
  p_phone   CLIP ViT-L/14 zero-shot on 4 frames: P("holding a small grey phone") vs
            P("empty flippers"), max over frames - phone present?
  jump      largest single-frame change / median change (qa_take's POP flag at > 8)
  frozen    share of frame pairs with ~no change (qa_take's FROZEN flag at > 0.35)
Contact sheets: <suite>/<config>/<tag>.sheet.png. The sheets are the verdict; the numbers
flag what to look at.

    python suite_v9.py --suite all --configs v8merged v9c@150
    python suite_v9.py --suite phone --configs v9c@150 --dry
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402
import gen_v8  # noqa: E402

gen_v8.lora_config = C.lora_config        # run() resolves configurations through this
P, P9, C8 = C.P, C.P9, C.C8
SEED = 42
PHONE_TEXT = ["a cartoon penguin holding a small grey phone",
              "a cartoon penguin with empty flippers, holding nothing"]


def phone_jobs():
    kf = json.loads((C.KF / "keyframes.json").read_text())
    props = {(j["char"], j["label"], j["angle"]): j for j in P9.discover_props({"props_skipped": []})}
    jobs = []
    for name, m in sorted(kf.items()):
        if not name.startswith("phone_"):
            continue
        j = props[(m["char"], m["label"], m["view"])]
        jobs.append({"tag": f"{m['char'].lower()}_{m['label']}_{m['view']}_f41_s{SEED}",
                     "prompt": C.phone_caption(j), "start": str(C.KF / name), "frames": 41,
                     "seed": SEED, "split": m["split"]})
    return jobs


def textcall_jobs():
    props = {(j["char"], j["label"], j["angle"]): j for j in P9.discover_props({"props_skipped": []})}
    jobs = []
    for char in C8.CHARS:
        j = props[(char, "phone_texting_standing", "FRONT")]
        jobs.append({"tag": f"{char.lower()}_textcall_FRONT_f41_s{SEED}",
                     "prompt": C.phone_caption(j), "frames": 41, "seed": SEED,
                     "start": str(C.KF / f"neutral_{char.lower()}_FRONT.png")})
    return jobs


def leak_jobs():
    jobs = []
    for char in C8.CHARS:
        for lab in ("waving", "happy", "sitting"):
            n = P.COMMON[C8.kind_of(lab)]
            jobs.append({"tag": f"{char.lower()}_{lab}_FRONT_f{n}_s{SEED}",
                         "prompt": C8.caption(char, lab, "FRONT", nframes=n), "frames": n,
                         "seed": SEED, "start": str(C.KF / f"neutral_{char.lower()}_FRONT.png")})
    return jobs


SUITES = {"phone": phone_jobs, "textcall": textcall_jobs, "leak": leak_jobs}

_M = {}


def clip_models():
    if not _M:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _M.update(torch=torch, dev=dev,
                  m=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval(),
                  p=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))
    return _M


def frames(p, size=336):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"scale={size}:{size}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def score(path):
    v = frames(path)
    d = np.abs(np.diff(v.astype(np.float32), axis=0)).mean((1, 2, 3))
    M = clip_models()
    idx = np.linspace(len(v) // 4, len(v) - 1, 4).astype(int)
    with M["torch"].no_grad():
        inp = M["p"](text=PHONE_TEXT, images=[v[i] for i in idx], return_tensors="pt",
                     padding=True).to(M["dev"])
        probs = M["m"](**inp).logits_per_image.softmax(-1)[:, 0].cpu().numpy()
    sheet = Path(path).with_suffix(".sheet.png")
    step = max(1, len(v) // 8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf",
                    f"select='not(mod(n\\,{step}))',scale=256:256,tile=8x1", "-frames:v", "1",
                    str(sheet)])
    return {"p_phone": round(float(probs.max()), 3), "jump": round(float(d.max() / max(np.median(d), 1e-6)), 2),
            "frozen": round(float((d < 0.15).mean()), 3), "sheet": str(sheet)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for s in (SUITES if a.suite == "all" else [a.suite]):
        jobs = SUITES[s]()
        sdir = C.RESULTS / "suites" / s
        scores_p = sdir / "scores.json"
        scores = json.loads(scores_p.read_text()) if scores_p.exists() else {}
        for cfg in a.configs:
            out = gen_v8.run(cfg, jobs, sdir / cfg.replace("/", "_").replace(":", "_"), dry=a.dry)
            if a.dry:
                for j in jobs[:2]:
                    print(f"   e.g. {j['tag']}: {j['prompt'][:160]}...")
                continue
            sc = {tag: score(p) for tag, p in out.items() if p.exists()}
            scores[cfg] = sc
            pp = [v["p_phone"] for v in sc.values()]
            print(f"   [{s}] {cfg}: mean p_phone {np.mean(pp):.2f}  "
                  f"POP {sum(v['jump'] > 8 for v in sc.values())}  "
                  f"FROZEN {sum(v['frozen'] > 0.35 for v in sc.values())}  n={len(sc)}")
        if not a.dry:
            sdir.mkdir(parents=True, exist_ok=True)
            scores_p.write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
