#!/usr/bin/env python3
"""Download every expv2 input of PLAN section 3 at a pinned revision into the layout DiffSynth
reads ($DIFFSYNTH_MODEL_BASE_PATH/<repo id>/<file>), verify it, and record the pins in
EVAL/phase0/pins.json.

    python expv2-minimax-h3/setup/fetch_models.py --set core          # ~119 GB: TE, DiT, VAEs, processor
    python expv2-minimax-h3/setup/fetch_models.py --set parity        # +66 GB: full DiT, G0.2a only
    python expv2-minimax-h3/setup/fetch_models.py --set adapter       # 0.6 GB: arm D only (ModelScope)
    python expv2-minimax-h3/setup/fetch_models.py --delete parity     # after G0.2a passes
    python expv2-minimax-h3/setup/fetch_models.py --set core --verify # also sha256 every big file

Revisions were resolved on 2026-10-12 (HF API). `--revision-latest` re-resolves them, and the
new SHAs land in pins.json; do that only deliberately, because LoRAs are build-specific
(MODEL_NOTES 2.6).

The G0.1 reference VAEs (Wan2.1, Wan2.2-TI2V-5B) are fetched by eval/vae_gate.py itself.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expv2_common as C  # noqa: E402

HF_REVISIONS = {
    "MiniMaxAI/MiniMax-H3": "42ed227ee7df40d41602854ae760620d6eb651fe",   # lastModified 2026-08-13
    "Comfy-Org/MiniMax-H3": "e5eb578a89295337b8ff433a035929ce0279e0b6",   # lastModified 2026-09-29
}
# LFS sha256 (HF paths-info / ModelScope repo/files), checked with --verify.
SHA256 = {
    "Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors":
        "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4",
    "Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors":
        "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
    "MiniMaxAI/MiniMax-H3/FL2VA/video_vae/source/model.safetensors":
        "5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3",
    "MiniMaxAI/MiniMax-H3/FL2VA/audio_vae/model.safetensors":
        "37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2",
    "DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_for_comfy_dit.safetensors":
        "0bbb44691f3931efdf66c07ba8e2356c7b311f2815edf94f3fd5c7ab1c3ddc5b",
}
SETS = {
    "core": [
        ("hf", "MiniMaxAI/MiniMax-H3", ["FL2VA/text_encoder/*", "FL2VA/processor/*",
                                        "FL2VA/video_vae/source/*", "FL2VA/audio_vae/*",
                                        "FL2VA/model_index.json"]),
        ("hf", "Comfy-Org/MiniMax-H3", ["diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"]),
    ],
    "parity": [
        ("hf", "Comfy-Org/MiniMax-H3", ["diffusion_models/minimax_h3_fl2va_bf16.safetensors"]),
    ],
    "adapter": [
        ("ms", "DiffSynth-Studio/MiniMax-H3-TrainingAdapter", ["model_for_comfy_dit.safetensors"]),
    ],
}


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(repo, patterns, latest):
    from huggingface_hub import HfApi, snapshot_download
    rev = HfApi().model_info(repo).sha if latest else HF_REVISIONS[repo]
    dest = C.MODELS / repo
    snapshot_download(repo, revision=rev, allow_patterns=patterns, local_dir=str(dest),
                      max_workers=8)
    return rev, dest


def fetch_ms(repo, patterns):
    from modelscope import snapshot_download
    dest = C.MODELS / repo
    snapshot_download(repo, allow_file_pattern=patterns, local_dir=str(dest))
    return "master", dest


def link_modelscope_alias():
    """DiffSynth's default processor config is MiniMax/MiniMax-H3:FL2VA/processor/ and, with
    DIFFSYNTH_SKIP_DOWNLOAD=True, it is not redirected to the HF id - so alias it."""
    alias = C.MODELS / "MiniMax" / "MiniMax-H3"
    target = C.MODELS / "MiniMaxAI" / "MiniMax-H3"
    if target.is_dir() and not alias.exists():
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target, target_is_directory=True)
        print(f"linked {alias} -> {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", choices=sorted(SETS), default=[])
    ap.add_argument("--delete", action="append", choices=["parity"], default=[])
    ap.add_argument("--verify", action="store_true", help="sha256 every file with a known hash")
    ap.add_argument("--revision-latest", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    pins = C.load_json(C.PINS, {}) or {}
    pins.setdefault("files", {})
    for name in args.set:
        for src, repo, patterns in SETS[name]:
            print(f"[{name}] {src}:{repo} {patterns}", flush=True)
            rev, dest = fetch_hf(repo, patterns, args.revision_latest) if src == "hf" else fetch_ms(repo, patterns)
            pins.setdefault("repos", {})[repo] = {"source": src, "revision": rev}
            for key, want in SHA256.items():
                if not key.startswith(repo + "/"):
                    continue
                f = C.MODELS / key
                if not f.exists():
                    continue
                entry = {"size": f.stat().st_size, "sha256_expected": want}
                if args.verify:
                    got = sha256(f)
                    entry["sha256"] = got
                    if got != want:
                        sys.exit(f"SHA256 MISMATCH {f}: {got} != {want}")
                    print(f"  sha256 ok  {key}")
                pins["files"][key] = entry
    link_modelscope_alias()

    for name in args.delete:
        for _, repo, patterns in SETS[name]:
            for pat in patterns:
                f = C.MODELS / repo / pat
                if f.exists():
                    f.unlink()
                    print(f"deleted {f}")
                pins["files"].pop(f"{repo}/{pat}", None)
            pins.setdefault("deleted", []).append(name)
    # The DiffSynth SHA and patch state are recorded by setup_box.sh (env.json); repeat here.
    pins["diffsynth_sha"] = C.DIFFSYNTH_SHA
    C.write_json(C.PINS, pins)
    free = shutil.disk_usage(C.MODELS).free / 1e9
    print(f"pins -> {C.PINS}   ({free:.0f} GB free under {C.MODELS})")


if __name__ == "__main__":
    main()
