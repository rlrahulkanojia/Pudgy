#!/usr/bin/env python3
"""Paths, LoRA configurations and generation for the v10 evaluation
(training_approach/v10/Training_Approach_v10.md section 5; decisions Q1-Q20 in section 0).

Reuses the v8/v9 tooling instead of copying it: the diffusion classifier (eval_v8/dcls_v8), the
v9 regression clip set (/workspace/eval_v9/holdout), the E5 wrapper (eval_v9/gen_asset_v9.py),
gen_v8's sampler settings, the v8 suites (eval_v8/suite_v8 + score_v8) and the phone check
(eval_v9/phone_id_v9). New here: the configurations below, the Mac-built eval set
(build_eval_v10.py -> /workspace/eval_v10) and one generator used by every v10 suite.

Configurations (lora_config):
  v8merged   low [pudgy-v8-merged]              high [golden]                the production base
  v10@<S>    low [pudgy-v8-merged]              high [golden, v10 step S]    HIGH ONLY (Stage 1)
  v9b@<S>    low [pudgy-v8-merged, v9b step S]  high [golden]                Phase 0 #1
  anything else falls through to eval_v9/common_v9 (pivotal:..., file:..., golden, ...).
  NB: `pivotal:<p>` would merge p into BOTH experts, which is wrong for a high-noise-only LoRA.

E5 (the frozen phone embedding) is applied at inference on every generation: every prompt that
contains the registry phone phrase gets E5 added to the phrase rows (gen_asset_v9 --asset-offset).
Prompts without the phrase are bit-identical, so the switch is a no-op for v8 prompts.

GPU: eval runs on GPU 1. CUDA_VISIBLE_DEVICES is set from V10_EVAL_GPU (default 1) when the
caller did not set it; this module must be imported before torch. On a one-GPU box, or to run
an eval on GPU 0 once training is over: V10_EVAL_GPU=0 (or CUDA_VISIBLE_DEVICES=0).

Environment (all optional):
  RUN_NAME       pudgy-v10-highnoise          the v10 run (checkpoints, results dir)
  V10_EVAL       /workspace/eval_v10          the eval set + every v10 eval output (runs/<RUN_NAME>/)
  V9_HOLDOUT     /workspace/eval_v9/holdout   the v8-label regression set with its latent cache
  E5_ASSET       /workspace/wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors
  V9B_RUN_DIR    /workspace/wan_output/pudgy-v9b-lownoise   (v9b@S checkpoints, Phase 0 #1)
  V10_EVAL_GPU   1
  GEN_EXTRA      extra musubi flags, e.g. "--blocks_to_swap 30 --lazy_loading"

Box helper (fetches the tokenizer and scorer models into the HF cache once, so later steps can
run offline):
    cd /workspace/Pudgy/finetune/wan/eval_v10 && /workspace/Pudgy/.venv-wan/bin/python common_v10.py prefetch
    /workspace/Pudgy/.venv-wan/bin/python common_v10.py configs v8merged v10@50    # resolve + check files
"""
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("V10_EVAL_GPU", "1"))

HERE = Path(__file__).resolve().parent
WAN = HERE.parent
for p in (WAN, WAN / "eval_v8", WAN / "eval_v9", WAN / "assets", WAN / "scenes", WAN / "longform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as C8  # noqa: E402  (eval_v8/common.py: model paths, MUSUBI, PY, GOLD_HIGH)
import common_v9 as C9  # noqa: E402

WS = Path("/workspace")
EVAL = Path(os.environ.get("V10_EVAL", str(WS / "eval_v10")))
SCENES = EVAL / "scenes"                 # Tier-1 windows: clips/, index.json, scenes_f33.{jsonl,toml}, cache/f33
G1DIR = EVAL / "g1"                      # G1/G2 jobs: jobs.json, first/, last/, real/
G4KF = EVAL / "g4" / "keyframes"         # v8 start frames for the core / compose suites
G5DIR = EVAL / "g5"                      # phone confirm tier: keyframes/, jobs.json
RUN_NAME = os.environ.get("RUN_NAME", "pudgy-v10-highnoise")
RUN = WS / "wan_output" / RUN_NAME
RESULTS = EVAL / "runs" / RUN_NAME       # tier1/, suite/, g4/, g5/, g1/
REGRESSION = Path(os.environ.get("V9_HOLDOUT", str(WS / "eval_v9" / "holdout")))
E5 = Path(os.environ.get("E5_ASSET", str(WS / "wan_output/e5/pudgy-v9c2-asset-dual-step00000150.asset.safetensors")))
V9B_RUN = Path(os.environ.get("V9B_RUN_DIR", str(WS / "wan_output" / "pudgy-v9b-lownoise")))
V8_MERGED = C9.V8_MERGED
GOLD_HIGH = C8.GOLD_HIGH
GEN_ASSET = WAN / "eval_v9" / "gen_asset_v9.py"

SW, SH = 480, 848
SCENE_SIZE = (SH, SW)                    # musubi --video_size is HEIGHT WIDTH
SEEDS = (42, 43, 44)

# E5 was trained on this wording (arms_v9c2.REG_PHRASE_NEW, the 2026-10-27 registry). The
# generator pins it; the live registry must still say the same, or E5 would silently not apply.
E5_PHRASE = "the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top"


def registry_phrase():
    reg = tomllib.loads((WAN / "assets" / "registry.toml").read_text())
    return reg["assets"]["phone"]["phrase"]


assert registry_phrase() == E5_PHRASE, "registry phone phrase changed: E5 was trained on the 2026-10-27 wording"


def ckpt(step):
    if str(step) in ("final", "last"):
        return RUN / f"{RUN_NAME}.safetensors"
    return RUN / f"{RUN_NAME}-step{int(step):08d}.safetensors"


def v9b_ckpt(step):
    if str(step) in ("final", "last"):
        return V9B_RUN / "pudgy-v9b-lownoise.safetensors"
    return V9B_RUN / f"pudgy-v9b-lownoise-step{int(step):08d}.safetensors"


def lora_config(name):
    """(low-noise [(path, mult)], high-noise [(path, mult)])."""
    hi = [(GOLD_HIGH, 1.0)]
    if name == "v8merged":
        return [(V8_MERGED, 1.0)], hi
    if name.startswith("v10@"):
        return [(V8_MERGED, 1.0)], hi + [(ckpt(name[4:]), 1.0)]
    if name.startswith("v9b@"):
        return [(V8_MERGED, 1.0), (v9b_ckpt(name[4:]), 1.0)], hi
    return C9.lora_config(name)


def cfg_dir(name):
    return name.replace("/", "_").replace(":", "_").replace("@", "_")


def missing_files(name):
    low, high = lora_config(name)
    return [str(p) for p, _ in low + high if not Path(p).exists()]


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


class locked:
    """Exclusive lock on <path>.lock for a read-modify-write of a shared results file (two scoring
    processes at once would otherwise drop each other's rows)."""

    def __init__(self, path):
        self.path = Path(str(path) + ".lock")

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def ci95(x, seed=0, n=5000):
    """(mean, lo, hi): the WIDER of the percentile bootstrap and the Student-t 95% interval
    (gate_pivotal_v9.boot_ci; RESEARCH_v9c_fixes 6c: with small n the bootstrap alone is too narrow)."""
    import numpy as np
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    blo, bhi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.inf
    return float(x.mean()), float(min(blo, x.mean() - half)), float(max(bhi, x.mean() + half))


# ---------------------------------------------------------------------------------------------
# Generation: gen_v8's two-expert call (same steps, shift, guidance, boundary, fp8 flags), routed
# through gen_asset_v9.py for E5, in musubi's --interactive mode so every clip is decoded and saved
# as soon as it is done (longform_client/stream.py's pattern). A crash or a stop loses at most the
# clip in flight, and --max-new N (stop after N new clips, print "REMAINING <k>", exit 0) lets the
# box's GPU-1 scheduler interleave each checkpoint's Tier 1 between generation slices.
# ---------------------------------------------------------------------------------------------
STEPS, SHIFT, GUIDE, BOUNDARY = 25, 5.0, 5.0, 0.9          # = eval_v8/gen_v8.py
# prompt_toolkit wants a terminal and stdin here is a pipe; hidden, musubi falls back to input()
BOOT = ("import os, runpy, sys; sys.modules['prompt_toolkit'] = None; sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
        "runpy.run_path(sys.argv[0], run_name='__main__')")


def job_line(j):
    line = f"{j['prompt']} --i {j['start']} --f {j['frames']} --d {j['seed']}"     # gen_v8's line
    if j.get("neg"):
        line += f" --n {j['neg']}"
    if j.get("end"):                       # first + last frame (FLF): pins the final state of the shot
        line += f" --ei {j['end']}"
    return line


def musubi_cmd(config, raw, size=None, e5=True):
    low, high = lora_config(config)
    h, w = size or (1024, 1024)
    own = ["--asset", str(E5), "--asset-offset", "--asset-phrase", E5_PHRASE] if e5 else []
    cmd = [str(C8.PY), "-c", BOOT, str(GEN_ASSET), *own, "--",
           "--task", "i2v-A14B", "--dit", str(C8.DIT_LOW), "--dit_high_noise", str(C8.DIT_HIGH),
           "--timestep_boundary", str(BOUNDARY), "--vae", str(C8.VAE), "--t5", str(C8.T5),
           "--lora_weight", *[str(p) for p, _ in low], "--lora_multiplier", *[str(m) for _, m in low],
           "--lora_weight_high_noise", *[str(p) for p, _ in high],
           "--lora_multiplier_high_noise", *[str(m) for _, m in high],
           "--video_size", str(h), str(w), "--fps", "24",
           "--infer_steps", str(STEPS), "--flow_shift", str(SHIFT),
           "--guidance_scale", str(GUIDE), "--attn_mode", "sdpa",
           "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
           "--interactive", "--save_path", str(raw), "--output_type", "video"]
    return cmd + os.environ.get("GEN_EXTRA", "").split()


def frame_count(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out.split(",")[0])
    except ValueError:
        return -1


class Generator:
    """One model load per configuration; one prompt line per clip on stdin."""

    def __init__(self, config, raw, size=None, e5=True):
        if not C8.PY.exists():
            sys.exit(f"no trainer venv python at {C8.PY}")
        miss = missing_files(config)
        if e5 and not E5.exists():
            miss.append(str(E5))
        if miss:
            sys.exit(f"missing weights for {config}: {miss}")
        self.raw = Path(raw); self.raw.mkdir(parents=True, exist_ok=True)
        cmd = musubi_cmd(config, self.raw, size, e5)
        (self.raw / "cmd.json").write_text(json.dumps(cmd, indent=1))
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
                   MUSUBI=str(C8.MUSUBI))
        self.log = open(self.raw / "gen.log", "a")
        self.log.write(f"\n==== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {config} "
                       f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n")
        self.p = subprocess.Popen(cmd, cwd=str(C8.MUSUBI), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)

    def take(self, j, dst):
        line = job_line(j)
        self.log.write(f"\n#### {j['tag']}\n{line}\n"); self.log.flush()
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        for out in self.p.stdout:
            self.log.write(out); self.log.flush()
            if "Video saved to:" in out:
                src = Path(out.split("Video saved to:", 1)[1].strip())
                if not src.is_absolute():
                    src = C8.MUSUBI / src
                src.rename(dst)
                return dst
        sys.exit(f"generator exited (code {self.p.wait()}) during {j['tag']} - see {self.raw / 'gen.log'}")

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait()
        self.log.close()


def generate(config, jobs, outdir, size=None, e5=True, max_new=None, check_frames=True):
    """jobs: [{tag, prompt, start, frames, seed, end?, neg?}] -> (want {tag: Path}, remaining: int).
    Existing <tag>.mp4 files are skipped (resumable); at most `max_new` new clips are made (None =
    all), in job order. A clip whose frame count differs from the job's is kept but logged to
    <outdir>/frame_mismatch.json (the scorers compare by length)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    want = {j["tag"]: outdir / f"{j['tag']}.mp4" for j in jobs}
    pending = [j for j in jobs if not want[j["tag"]].exists()]
    print(f"   [{config}] {len(jobs)} clips, {len(pending)} to generate -> {outdir}", flush=True)
    rec = outdir / f"jobs_{cfg_dir(config)}.json"
    old = load_json(rec, {})
    old.update({j["tag"]: j for j in jobs})
    write_json(rec, old)
    todo = pending if max_new is None else pending[:max(0, int(max_new))]
    if not todo:
        return want, len(pending)
    gen = None
    try:
        for i, j in enumerate(todo):
            if gen is None:
                gen = Generator(config, outdir / f".raw_{cfg_dir(config)}", size, e5)
            t1 = time.time()
            dst = gen.take(j, want[j["tag"]])
            if check_frames:
                n = frame_count(dst)
                if n != j["frames"]:
                    print(f"   WARNING {j['tag']}: {n} frames, job asked for {j['frames']}", flush=True)
                    mm = load_json(outdir / "frame_mismatch.json", {})
                    mm[j["tag"]] = {"got": n, "want": j["frames"]}
                    write_json(outdir / "frame_mismatch.json", mm)
            print(f"   [{config}] {i + 1}/{len(todo)} {j['tag']} {(time.time() - t1) / 60:.1f} min", flush=True)
    finally:
        if gen is not None:
            gen.close()
    return want, len(pending) - len(todo)


# ---------------------------------------------------------------------------------------------
def prefetch():
    """Tokenizer (E5 splice), DINOv2 (G2 identity), CLIP + DINO (score_v8), RapidOCR (G2 text)."""
    import asset_embed_v9 as A
    tok = A.tokenizer()
    print("umt5 tokenizer ok:", len(A.token_ids(tok, E5_PHRASE)) - 1, "phrase tokens")
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    AutoModel.from_pretrained("facebook/dinov2-base"); AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14"); CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("dinov2-base + clip-vit-large-patch14 ok")
    try:
        from rapidocr_onnxruntime import RapidOCR
        RapidOCR()
        print("rapidocr_onnxruntime ok")
    except ImportError:
        sys.exit("rapidocr_onnxruntime missing: /workspace/Pudgy/.venv-wan/bin/pip install rapidocr_onnxruntime")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prefetch":
        prefetch()
    elif len(sys.argv) > 1 and sys.argv[1] == "configs":
        bad = 0
        for n in sys.argv[2:]:
            low, high = lora_config(n)
            miss = missing_files(n)
            bad += bool(miss)
            print(f"{n}: low {[(str(p), m) for p, m in low]}\n{'':{len(n)}}  high {[(str(p), m) for p, m in high]}"
                  + (f"\n  MISSING {miss}" if miss else ""))
        sys.exit(1 if bad else 0)
    else:
        print(__doc__)
