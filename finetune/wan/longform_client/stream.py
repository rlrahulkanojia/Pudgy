"""The full batch through ONE model load, one video at a time: each video's three takes run back
to back, then it is assembled, QA'd and uploaded straight away, so finished videos reach Azure
every ~1 h 40 min instead of all together at the end of stage 3.

musubi's --interactive mode keeps both experts loaded and reads one prompt line per take from
stdin; the take is decoded and saved before the next line is sent, which is what chaining needs
(take k starts on take k-1's last frame). Same model, LoRAs, sampler settings, prompts and seeds
as produce.py's staged batches: the command is gen_v8's with --interactive in place of
--from_file, and the take lines come from produce.jobs_for.

Resumable: existing takes are skipped and upload skips blobs that are already there.
Progress lines for the operator start with "##" (logs/stream.log has the generator's output).

python stream.py <YYYY-MM-DD> [--sub samples] [--profile full|light4] [--loops] [ids or id:mode ...]
   --loops: also a looping version of the motion cycles (produce.LOOPS), right after each one   (default: every video, in produce order)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/longform_client")
sys.path.insert(0, str(ROOT))
import produce as PR                    # noqa: E402  (also points gen_v8 at the v8merged pair)
import assemble_client                  # noqa: E402
import upload                           # noqa: E402
from gen_v8 import STEPS, SHIFT, GUIDE, BOUNDARY            # noqa: E402
from common import DIT_LOW, DIT_HIGH, VAE, T5, MUSUBI, PY   # noqa: E402

LX = Path("/workspace/wan_models/lightx2v")
PROFILES = {      # name -> base DiTs (low, high), steps, guidance (low, high), extra musubi args
    "full": dict(dit=(DIT_LOW, DIT_HIGH), steps=STEPS, guide=(GUIDE, None), extra=[]),   # gen_v8's
    # lightx2v Wan2.2 I2V 4-step distill baked into the bases (merge_lightning.py), v8 LoRAs on top as
    # before. Guidance off - every step runs the cond pass only - and plain Euler, as it was distilled.
    "light4": dict(dit=(LX / "wan2.2_i2v_low_noise_14B_fp16_lightx2v4step1022.safetensors",
                        LX / "wan2.2_i2v_high_noise_14B_fp16_lightx2v4step1022.safetensors"),
                   steps=4, guide=(1.0, 1.0),
                   extra=["--sample_solver", "vanilla", "--cfg_skip_mode", "alternate", "--cfg_apply_ratio", "0"]),
}
RAW = PR.TAKES_DIR / ".raw_stream"
LOG = ROOT / "logs/stream.log"
# prompt_toolkit wants a terminal and stdin here is a pipe; hidden, musubi falls back to input()
BOOT = ("import os, runpy, sys; sys.modules['prompt_toolkit'] = None; sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
        "runpy.run_path(sys.argv[0], run_name='__main__')")


RETRIES = 3


def say(*a):
    print("##", time.strftime("%H:%M UTC", time.gmtime()), *a, flush=True)


class Generator:
    def __init__(self, profile="full"):
        low, high = PR.gen_v8.lora_config("v8merged")
        prof = PROFILES[profile]
        RAW.mkdir(parents=True, exist_ok=True)
        cmd = [str(PY), "-c", BOOT, "src/musubi_tuner/wan_generate_video.py",
               "--task", "i2v-A14B", "--dit", str(prof["dit"][0]), "--dit_high_noise", str(prof["dit"][1]),
               "--timestep_boundary", str(BOUNDARY), "--vae", str(VAE), "--t5", str(T5),
               "--lora_weight", *[str(p) for p, _ in low],
               "--lora_multiplier", *[str(m) for _, m in low],
               "--lora_weight_high_noise", *[str(p) for p, _ in high],
               "--lora_multiplier_high_noise", *[str(m) for _, m in high],
               "--video_size", "1024", "1024", "--fps", "24",
               "--infer_steps", str(prof["steps"]), "--flow_shift", str(SHIFT),
               "--guidance_scale", str(prof["guide"][0]), "--attn_mode", "sdpa",
               "--fp8", "--fp8_scaled", "--fp8_t5", "--vae_cache_cpu",
               "--interactive", "--save_path", str(RAW), "--output_type", "video"]
        if prof["guide"][1] is not None:
            cmd += ["--guidance_scale_high_noise", str(prof["guide"][1])]
        cmd += prof["extra"] + os.environ.get("GEN_EXTRA", "").split()
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        self.log = open(LOG, "a")
        self.p = subprocess.Popen(cmd, cwd=str(MUSUBI), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)

    def take(self, j):
        line = f"{j['prompt']} --i {j['start']} --f {j['frames']} --d {j['seed']}"   # gen_v8's line
        if j.get("neg"):
            line += f" --n {j['neg']}"
        if j.get("end"):
            line += f" --ei {j['end']}"
        self.log.write(f"\n#### {j['tag']}\n{line}\n"); self.log.flush()
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        for out in self.p.stdout:
            self.log.write(out); self.log.flush()
            if "Video saved to:" in out:
                src = Path(out.split("Video saved to:", 1)[1].strip())
                dst = PR.TAKES_DIR / f"{j['tag']}.mp4"
                src.rename(dst)
                return dst
        sys.exit(f"generator exited (code {self.p.wait()}) during {j['tag']} - see {LOG}")

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    a = sys.argv[2:]
    sub = a.pop(a.index("--sub") + 1) if "--sub" in a else ""
    profile = a.pop(a.index("--profile") + 1) if "--profile" in a else "full"
    first = a.pop(a.index("--first") + 1).split(",") if "--first" in a else []
    loops = "--loops" in a
    a = [x for x in a if x not in ("--sub", "--profile", "--loops", "--first")]
    date, sel = sys.argv[1], PR.select(a)
    if profile != "full":                 # its own takes and finals: <id>_<profile>_t<k>.mp4
        sel = [(vid, dict(v, mode=profile)) for vid, v in sel]
    gen = None

    def make_take(tag, job, vid):
        """Generate a take; one that drifts at its end (assemble_client.end_drift) is set aside in
        takes/rejected/ and re-made with the seed + 1000, up to RETRIES tries (else the least drifted)."""
        nonlocal gen
        if (PR.TAKES_DIR / f"{tag}.mp4").exists():
            return
        if gen is None:
            say("loading the model")
            gen = Generator(profile)
        tries = []
        for attempt in range(RETRIES):
            j = job()
            j["seed"] += 1000 * attempt
            dst = PR.TAKES_DIR / "rejected" / f"{tag}_s{j['seed']}.mp4"
            if dst.exists():                    # rejected before a restart: re-checked, else counts as a try
                h, feet, ground, side = assemble_client.end_drift(dst, vid)
                if dst.with_suffix(".manual").exists():     # rejected by eye (e.g. an extra character): stays out
                    tries.append((9.0, dst))
                    continue
                if assemble_client.drift_ok(h, feet, ground, vid, side):
                    dst.rename(PR.TAKES_DIR / f"{tag}.mp4")
                    say(f"take {tag} seed {j['seed']} restored: passes the updated drift check "
                        f"(feet {feet:+.3f}, ground {ground:+.3f})")
                    return
                tries.append((abs(ground) + abs(feet) + abs(side), dst))
                continue
            t = time.time()
            take = gen.take(j)
            if tag.endswith("loop"):            # ends pinned on the keyframe
                say(f"take {tag} done in {(time.time() - t) / 60:.0f} min")
                return
            h, feet, ground, side = assemble_client.end_drift(take, vid)
            ok = assemble_client.drift_ok(h, feet, ground, vid, side)
            say(f"take {tag} seed {j['seed']} done in {(time.time() - t) / 60:.0f} min, drift: height "
                f"x{h:.2f} feet {feet:+.3f} ground {ground:+.3f} side {side:+.3f}" + ("" if ok else " - REJECTED"))
            if ok:
                return
            dst.parent.mkdir(exist_ok=True)
            take.rename(dst)
            tries.append((abs(ground) + abs(feet) + abs(side), dst))
        auto = [t for t in tries if not t[1].with_suffix(".manual").exists()]
        best = min(auto or tries)[1]
        best.rename(PR.TAKES_DIR / f"{tag}.mp4")
        say(f"take {tag}: no clean seed in {RETRIES} tries, kept the least drifted ({best.name})")

    say("videos:", " ".join(f"{vid}:{v['mode']}" for vid, v in sel))
    by_id = dict(sel)
    for vid in first:                     # --first a,b,..: take 1 of these before anything else (story edits)
        v = by_id[vid]
        make_take(f"{vid}_{v['mode']}_t1", lambda: PR.jobs_for(1, [(vid, v)])[0], vid)
    for n, (vid, v) in enumerate(sel, 1):
        versions = [False] + ([True] if loops and vid in PR.LOOPS else [])
        for loop in versions:
            jobs = ([PR.loop_job] if loop else
                    [lambda vid, v, k=k: PR.jobs_for(k, [(vid, v)])[0] for k in range(1, PR.TAKES + 1)])
            tags = [f"{vid}_{v['mode']}_t3loop"] if loop else [f"{vid}_{v['mode']}_t{k}" for k in range(1, PR.TAKES + 1)]
            for tag, make in zip(tags, jobs):
                make_take(tag, lambda: make(vid, v), vid)
            assemble_client.main(vid, v["mode"], loop=loop)
            name = (vid if v["mode"] == "chain" else f"{vid}_{v['mode']}") + ("_loop" if loop else "")
            links = upload.main(date, files=[ROOT / f"final/{name}_1080x1920.mp4"], sub=sub)
            say(f"VIDEO {n}/{len(sel)} {name} uploaded: {links[f'{name}_1080x1920.mp4']}")
    if gen:
        gen.close()
    say("ALL DONE")


if __name__ == "__main__":
    main()
