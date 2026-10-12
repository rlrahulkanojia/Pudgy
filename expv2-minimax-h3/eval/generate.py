#!/usr/bin/env python3
"""Tier 2 generation - the `core`, `native`, `gb` and `probe` suites (PLAN 7.1, 7.4),
two-pass (MODEL_NOTES 2.7, C4).

Settings (C.INFERENCE, PLAN 7.1): DiffSynth MiniMaxH3Pipeline on the pruned build, LoRA
hot-loaded at 1.0, cfg 1.0, flow_shift 12, audio_flow_shift 3, 50 steps, 768x768, first
frame only (keyframe_indices [0]), seeds from eval_items.json (42-44; probe 42).

  pass 1  --encode            Qwen3-VL + video VAE: the pipeline's own unit chain for every
                              (item, seed) -> CACHE/eval/gen/<item_id>__s<seed>.pth.
                              Checkpoint-independent (the LoRA only touches the DiT), so
                              it runs ONCE for every checkpoint ever evaluated.
  pass 2  --ckpt SPEC[,SPEC]  DiT + video VAE: __call__'s denoise loop + decode on each
                              cached input -> EVAL/gen/<ckpt>/<suite>/<item_id>__s<seed>/
                              0000.png..., meta.json, preview.mp4 (--preview). Resumable:
                              finished items (meta.json done) are skipped.
  --one-pass                  plain pipe(...) with every model loaded (CPU offload), for
                              G0.2g's equivalence check -> EVAL/gen/<ckpt>__onepass/...

Audio is generated as part of the joint sequence and discarded (PLAN 7.1).

    python eval/generate.py --dry --suites core,native,gb,probe
    python eval/generate.py --encode --suites core,native,gb,probe
    python eval/generate.py --ckpt base --suites core,native
    python eval/generate.py --ckpt expv2-h3-main@all --suites probe
    python eval/generate.py --ckpt expv2-h3-main@150 --suites core,native,gb --preview
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3lib as L  # noqa: E402
C = L.C

ENC = C.CACHE / "eval" / "gen"
SUITES = ("core", "native", "gb", "probe")


def jobs(items, seeds=None):
    out = []
    for it in items:
        for s in (seeds or it["seeds"]):
            out.append((it, int(s)))
    return out


def enc_path(it, seed):
    return ENC / f"{it['item_id']}__s{seed}.pth"


def out_dir(name, it, seed, onepass=False):
    return C.EVAL / "gen" / (name + ("__onepass" if onepass else "")) / it["suite"] / f"{it['item_id']}__s{seed}"


def is_done(d, n):
    m = C.load_json(d / "meta.json")
    return bool(m and m.get("done") and len(L.png_files(d)) == n)


def start_image(it):
    from PIL import Image
    p = C.CORPUS / it["start_frame"]
    im = Image.open(p).convert("RGB")
    if im.size != (C.SIZE, C.SIZE):
        raise ValueError(f"{p}: start frame is {im.size}, must be exactly {C.SIZE}x{C.SIZE} (PLAN 7.1)")
    return im


def check_output(frames, it):
    if len(frames) != it["frames"]:
        raise RuntimeError(f"{it['item_id']}: {len(frames)} frames returned, expected {it['frames']}")
    if frames[0].size != (C.SIZE, C.SIZE):
        raise RuntimeError(f"{it['item_id']}: output {frames[0].size}, expected {C.SIZE}^2")


def write_item(d, frames, it, seed, name, path, t, preview, extra=None):
    L.save_frames(frames, d)
    meta = {"done": True, "item": it, "seed": seed, "ckpt": name, "lora": str(path) if path else None,
            "settings": C.INFERENCE, "seconds": round(t, 1), "frames": len(frames),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_sha": L.git_sha(),
            "diffsynth_sha": L.diffsynth_sha(), **(extra or {})}
    if preview:
        L.write_preview_mp4(d, d / "preview.mp4")
    C.write_json(d / "meta.json", meta)


def encode(js, force=False):
    import torch
    ENC.mkdir(parents=True, exist_ok=True)
    todo = []
    for it, seed in js:
        p = enc_path(it, seed)
        if p.exists() and not force:
            old = torch.load(str(p), map_location="cpu", weights_only=False)
            if old.get("caption_hash") == L.text_hash(it["caption"]) and old.get("num_frames") == it["frames"]:
                continue
        todo.append((it, seed, p))
    print(f"pass 1: {len(todo)} (item, seed) encodings ({len(js) - len(todo)} cached)")
    if not todo:
        return
    pipe = L.load_pipe(("te", "vae"))
    memo = {}
    for i, (it, seed, p) in enumerate(todo):
        t = time.time()
        im = start_image(it)
        enc = L.encode_generation(pipe, it["caption"], im, it["frames"], seed,
                                  memo=memo, memo_key=(it["caption"], it["start_frame"]))
        enc.update({"item_id": it["item_id"], "start_frame": it["start_frame"]})
        torch.save(enc, str(p))
        print(f"  [{i + 1}/{len(todo)}] {it['item_id']} s{seed}  {time.time() - t:.1f}s")
    pipe = None
    memo = None
    L.free()


def generate(js, ckpts, preview=False):
    import torch
    missing = [enc_path(it, s) for it, s in js if not enc_path(it, s).exists()]
    if missing:
        sys.exit(f"{len(missing)} pass-1 encodings missing (e.g. {missing[0]}): run --encode first")
    pipe = None
    for name, path in ckpts:
        todo = [(it, s) for it, s in js if not is_done(out_dir(name, it, s), it["frames"])]
        print(f"{name}: {len(todo)} clips to generate ({len(js) - len(todo)} done)")
        if not todo:
            continue
        if pipe is None:
            pipe = L.load_pipe(("dit", "vae"))
        L.detach_lora(pipe)
        L.attach_lora(pipe, path)
        t0 = time.time()
        for i, (it, seed) in enumerate(todo):
            enc = torch.load(str(enc_path(it, seed)), map_location="cpu", weights_only=False)
            if enc.get("caption_hash") != L.text_hash(it["caption"]):
                sys.exit(f"{enc_path(it, seed)} was encoded from a different caption: run --encode --force")
            t = time.time()
            frames = L.denoise(pipe, enc)
            check_output(frames, it)
            write_item(out_dir(name, it, seed), frames, it, seed, name, path, time.time() - t, preview)
            print(f"  [{i + 1}/{len(todo)}] {it['suite']}/{it['item_id']} s{seed}  {time.time() - t:.1f}s")
        print(f"  {name}: {len(todo)} clips in {(time.time() - t0) / 60:.1f} min")
    if pipe is not None:
        L.detach_lora(pipe)
        pipe = None
        L.free()


def one_pass(js, ckpts, preview=False):
    """The reference path for G0.2g: one pipe(...) call per clip, every model loaded with
    DiffSynth's CPU-offload VRAM config (the examples' low-VRAM setup)."""
    import torch
    pipe = L.load_pipe(("te", "dit", "vae", "audio_vae"))
    I = C.INFERENCE
    for name, path in ckpts:
        L.detach_lora(pipe)
        L.attach_lora(pipe, path)
        for it, seed in js:
            d = out_dir(name, it, seed, onepass=True)
            if is_done(d, it["frames"]):
                continue
            t = time.time()
            with torch.no_grad():
                video, _audio = pipe(
                    prompt=it["caption"], negative_prompt=" ",
                    height=I["height"], width=I["width"], num_frames=it["frames"],
                    num_inference_steps=I["num_inference_steps"], seed=seed, rand_device=I["rand_device"],
                    cfg_scale=I["cfg_scale"], flow_shift=I["flow_shift"], audio_flow_shift=I["audio_flow_shift"],
                    tiled=I["tiled"], tile_size=I["tile_size"], tile_overlap=I["tile_overlap"],
                    keyframes=[start_image(it)], keyframe_indices=[0],
                )
            check_output(video, it)
            write_item(d, video, it, seed, name, path, time.time() - t, preview, {"one_pass": True})
            print(f"  one-pass {it['item_id']} s{seed}  {time.time() - t:.1f}s")
    L.detach_lora(pipe)
    pipe = None
    L.free()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", default="core,native,gb,probe")
    ap.add_argument("--items", default="", help="comma list of item_ids or substrings")
    ap.add_argument("--seeds", default="", help="override the items' seeds, e.g. 42")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--one-pass", action="store_true")
    ap.add_argument("--preview", action="store_true", help="also write preview.mp4 (viewing only)")
    ap.add_argument("--force", action="store_true", help="re-encode pass 1")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    suites = [s for s in args.suites.split(",") if s]
    bad = [s for s in suites if s not in SUITES]
    if bad:
        sys.exit(f"unknown suites {bad}; choose from {SUITES}")
    items = L.load_eval_items(suites, [k for k in args.items.split(",") if k] or None)
    seeds = [int(s) for s in args.seeds.split(",") if s] or None
    js = jobs(items, seeds)
    if args.limit:
        js = js[:args.limit]

    if args.dry:
        from collections import Counter
        from PIL import Image
        probs = []
        cnt = Counter(it["suite"] for it, _ in js)
        print(f"generation: {len(js)} clips  {dict(cnt)}  (PLAN 7.4: core 156, native 24, gb 36, probe 8)")
        for it in items:
            p = C.CORPUS / it["start_frame"]
            if not p.exists():
                probs.append(f"{it['item_id']}: start frame {p} missing")
                continue
            sz = Image.open(p).size
            if sz != (C.SIZE, C.SIZE):
                probs.append(f"{it['item_id']}: start frame {sz}")
            if it["frames"] % 17 != 5:
                probs.append(f"{it['item_id']}: {it['frames']} frames off the 17n+5 grid")
            if it["char"] == "Pax" and " pink " in it["caption"]:
                probs.append(f"{it['item_id']}: 'pink' in a Pax caption (A10)")
        have = sum(enc_path(it, s).exists() for it, s in js)
        print(f"  pass-1 encodings cached: {have}/{len(js)}")
        if args.ckpt:
            for name, path in L.resolve_ckpts(args.ckpt, must_exist=False):
                done = sum(is_done(out_dir(name, it, s), it["frames"]) for it, s in js)
                print(f"  ckpt {name:28s} {path or '(no LoRA)'}  done {done}/{len(js)}")
        print("DRY: problems:\n  - " + "\n  - ".join(probs) if probs else "DRY: inputs OK")
        return 1 if probs else 0

    if args.encode:
        encode(js, force=args.force)
    if args.ckpt:
        ckpts = L.resolve_ckpts(args.ckpt)
        if args.one_pass:
            one_pass(js, ckpts, args.preview)
        else:
            generate(js, ckpts, args.preview)
    if not (args.encode or args.ckpt):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
