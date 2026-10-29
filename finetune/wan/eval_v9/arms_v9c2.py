#!/usr/bin/env python3
"""v9c2 experiment arms: inference fixes for the phone turning round, and the asset embedding (V2).

Each arm generates the same jobs with one change against a control, and every clip is scored by
ID1 (phone_id_v9.py: the phone keeps its back to the viewer). RESEARCH_v9c_fixes section 4, rev. 2.

  screen tier : the two standing-FRONT starts (flip onsets at frames 2 and 5) x 3 seeds, 21 frames.
                The control must reproduce the flip at 21 frames; if it fails fewer than 4 of 6,
                the screen is invalid and the remaining arms switch to 41 frames automatically.
  confirm tier: all 8 phone starts x 3 seeds at 41 frames (--tier confirm --arms ...).

    python arms_v9c2.py --tier screen
    python arms_v9c2.py --tier confirm --arms C0 E2a
    python arms_v9c2.py --tier screen --arms V2 --asset <E.asset.safetensors>
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v9 as C  # noqa: E402
import suite_v9 as S  # noqa: E402  (wires gen_v8 to the v9 configs)
import gen_v8  # noqa: E402
import phone_id_v9 as ID  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = C.RESULTS / "arms_v9c2"
SEEDS = (42, 7, 1234)
NEG_DEFAULT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
               "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
               "杂乱的背景，三条腿，背景人很多，倒着走")
NEG_SCREEN = NEG_DEFAULT + ", phone screen, smartphone display, glowing screen, call screen, app interface, text, letters, buttons"
PHRASE_OLD = "with a black screen, three camera lenses on the back and a little tuft on top"
PHRASE_NEW = "seen from the back, with three camera lenses and a little tuft on top"
CLAUSE_OLD = "the screen turned toward itself"
CLAUSE_NEW = "holding it with its back toward the viewer"


# Registry wording before / after the 2026-10-27 change (user-approved after E2c): the identity
# phrase loses "with a black screen,"; the texting clause loses the screen.
REG_PHRASE_OLD = ("the iFin phone, a small pale grey phone with a black screen, three camera lenses on the back "
                  "and a little tuft on top")
REG_PHRASE_NEW = "the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top"
REG_CLAUSE_NEW = CLAUSE_NEW


def old_wording(p):
    """Rebuild the pre-2026-10-27 caption from a caption compiled with either registry version."""
    if REG_PHRASE_OLD not in p:
        assert REG_PHRASE_NEW in p and REG_CLAUSE_NEW in p, "unknown phone wording"
        p = p.replace(REG_PHRASE_NEW, REG_PHRASE_OLD).replace(REG_CLAUSE_NEW, CLAUSE_OLD)
    return p


def new_wording(p):
    assert REG_PHRASE_NEW in p and REG_CLAUSE_NEW in p and "screen" not in p.split(";")[0], \
        "registry not yet updated to the 2026-10-27 wording"
    return p


def no_screen_words(p):
    assert PHRASE_OLD in p and CLAUSE_OLD in p, "caption wording changed; update arms_v9c2.py"
    return p.replace(PHRASE_OLD, PHRASE_NEW).replace(CLAUSE_OLD, CLAUSE_NEW)


# name: (config, description, job overrides fn, extra musubi args, wrapper args)
def arms(asset):
    return {
        "C0":  ("v9c@200", "control: v9c@200, default settings (+ x0 probe at the switch)", None, [], ["--x0-probe"]),
        "E2a": ("v9c@200", "negative prompt adds screen / display / call screen / text / buttons",
                lambda j: {**j, "neg": NEG_SCREEN}, [], []),
        "E2c": ("v9c@200", "no screen words: phrase 'seen from the back…', clause 'back toward the viewer'",
                lambda j: {**j, "prompt": no_screen_words(j["prompt"])}, [], []),
        "E2b": ("v9c@200", "guidance 3.5 / 3.5 (official Wan2.2 I2V default; suites used 5.0 low-noise)",
                None, ["--guidance_scale", "3.5", "--guidance_scale_high_noise", "3.5"], []),
        "V2":  ("v8merged", "asset embedding E (V1b, 40 steps, offset, old-phrase rows) on v8merged, OLD wording",
                lambda j: {**j, "prompt": old_wording(j["prompt"])}, [],
                ["--asset", asset or "", "--asset-offset", "--asset-phrase", REG_PHRASE_OLD]),
        "C0b": ("v8merged", "control for V2: v8merged, OLD wording, no embedding",
                lambda j: {**j, "prompt": old_wording(j["prompt"])}, [], []),
        "E2d": ("v9c@200", "first+last frame: the start frame also pinned as the end frame",
                lambda j: {**j, "end": j["start"]}, [], []),
        "E1b": ("v9c@200", "timestep boundary 0.95 (6 high-noise steps instead of 9)",
                None, ["--timestep_boundary", "0.95"], ["--x0-probe"]),
        "E3":  ("v9c@200x2", "v9c LoRA at multiplier 2", None, [], []),
        # confirm tier for the registry change (all 8 starts x 3 seeds, 41 frames)
        "R":     ("v9c@200", "registry wording 2026-10-27 (no screen words) on v9c@200",
                  lambda j: {**j, "prompt": new_wording(j["prompt"])}, [], []),
        "Rb":    ("v8merged", "registry wording 2026-10-27 on v8merged (is the v9c LoRA needed?)",
                  lambda j: {**j, "prompt": new_wording(j["prompt"])}, [], []),
        "C0old": ("v9c@200", "control: old registry wording on v9c@200",
                  lambda j: {**j, "prompt": old_wording(j["prompt"])}, [], []),
        # T5 precision at inference (RESEARCH_v9c_fixes 6b): the old wording, where the control flips 12/24,
        # with the text encoder in fp32 instead of fp8 (fp32 = the accurate reference and the precision of the
        # v9c2 training cache; GPU bf16 has per-token errors up to 3.8x); control C0old (same everything else)
        "C0oldT5": ("v9c@200", "old wording, T5 fp32 at inference (no --fp8_t5); control C0old",
                    lambda j: {**j, "prompt": old_wording(j["prompt"])}, ["!drop:--fp8_t5"], ["--t5-fp32"]),
        # pivotal stage 2: cross-attn k/v LoRA (both experts) on top of the frozen E5 embedding; control Rb / E5
        "PV":    ("pivotal:" + os.environ.get("PIVOTAL_LORA", ""), "pivotal LoRA (both experts) + frozen E + new wording",
                  lambda j: {**j, "prompt": new_wording(j["prompt"])}, [],
                  ["--asset", asset or "", "--asset-offset", "--asset-phrase", REG_PHRASE_NEW]),
        # E5: the asset embedding trained through both experts on the new wording; its control is
        # Rb (base v8merged) or R (base v9c@200), whichever base E5 was trained on (E5_CONFIG)
        "E5":    (os.environ.get("E5_CONFIG", "v8merged"), "asset embedding E5 (both experts, offset) + new wording",
                  lambda j: {**j, "prompt": new_wording(j["prompt"])}, [],
                  ["--asset", asset or "", "--asset-offset", "--asset-phrase", REG_PHRASE_NEW]),
    }


REUSE = {"C0old": [C.RESULTS / "suites/phone/v9c@200", OUT / "screen/C0"],
         "C0b": [C.RESULTS / "suites/phone/v8merged"]}


def reuse_identical(adir, jobs, srcs):
    """Copy already-generated clips with byte-identical settings and prompt (same config, seed,
    frames, start, wording) into the arm dir, so gen_v8.run skips them."""
    adir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in srcs:
        jf = next(iter(src.glob("jobs_*.json")), None)
        if not jf:
            continue
        old = {j["tag"]: j for j in json.loads(jf.read_text())}
        for j in jobs:
            o, dst = old.get(j["tag"]), adir / f"{j['tag']}.mp4"
            if o and not dst.exists() and (src / dst.name).exists() and all(
                    o.get(k) == j.get(k) for k in ("prompt", "start", "frames", "seed")) and not o.get("neg"):
                os.link(src / dst.name, dst)
                n += 1
    return n


def lora_config(name):
    if name == "v9c@200x2":
        low, high = C.lora_config("v9c@200")
        return [(low[0][0], 1.0), (low[1][0], 2.0)], high
    return C.lora_config(name)


gen_v8.lora_config = lora_config


def jobs_for(tier, frames):
    base = S.phone_jobs()
    if tier == "screen":
        base = [j for j in base if "standing_FRONT" in j["tag"]]
    out = []
    for j in base:
        for sd in SEEDS:
            tag = j["tag"].rsplit("_f41_", 1)[0] + f"_f{frames}_s{sd}"
            out.append({**j, "tag": tag, "frames": frames, "seed": sd})
    return out


@contextlib.contextmanager
def wrapped(extra, wrapper_args):
    """Route gen_v8's musubi call through gen_asset_v9.py and append extra musubi args."""
    orig = gen_v8.subprocess.run

    drop = {x[len("!drop:"):] for x in extra if x.startswith("!drop:")}     # e.g. "!drop:--fp8_t5"
    add = [x for x in extra if not x.startswith("!drop:")]

    def run(cmd, *a, **kw):
        assert cmd[1] == "src/musubi_tuner/wan_generate_video.py"
        assert all(d in cmd for d in drop), f"cannot drop {drop - set(cmd)}: not in the musubi command"
        cmd = [cmd[0], str(HERE / "gen_asset_v9.py"), *wrapper_args, "--", *[c for c in cmd[2:] if c not in drop], *add]
        return orig(cmd, *a, **kw)

    gen_v8.subprocess.run = run
    try:
        yield
    finally:
        gen_v8.subprocess.run = orig


def run_arm(name, spec, tier, frames, log):
    config, desc, fn, extra, wargs = spec
    jobs = [fn(j) if fn else j for j in jobs_for(tier, frames)]
    adir = OUT / tier / name
    if name in REUSE:
        print(f"   reused {reuse_identical(adir, jobs, REUSE[name])} identical control clips", flush=True)
    want = {j["tag"]: adir / f"{j['tag']}.mp4" for j in jobs}
    pending = [j for j in jobs if not want[j["tag"]].exists()]
    wargs = [x for w in wargs for x in (["--x0-probe", str(adir / "probe")] if w == "--x0-probe" else [w])]
    t0 = time.time()
    with wrapped(extra, wargs):
        gen_v8.run(config, jobs, adir)
    if "--x0-probe" in spec[4]:                    # probe_k follows the pending order
        for k, j in enumerate(pending):
            src = adir / "probe" / f"probe_{k:03d}.mp4"
            if src.exists():
                src.rename(adir / "probe" / f"{j['tag']}.x0.mp4")
    rows = [ID.score(str(want[j["tag"]])) for j in jobs]
    probe = [ID.score(str(p)) for p in sorted((adir / "probe").glob("*.x0.mp4"))] if (adir / "probe").exists() else []
    res = {"arm": name, "config": config, "desc": desc, "tier": tier, "frames": frames, "extra": extra,
           "wrapper": wargs, "defects": sum(r["defect"] for r in rows), "n": len(rows), "rows": rows,
           "probe_defects": sum(r["defect"] for r in probe), "probe": probe,
           "minutes": round((time.time() - t0) / 60, 1)}
    (adir / "id1.json").write_text(json.dumps(res, indent=1))
    line = (f"{name:4s} {res['defects']}/{res['n']} defective"
            + (f" | x0 at switch: {res['probe_defects']}/{len(probe)}" if probe else "")
            + f" | {desc} | {res['minutes']} min")
    print("== " + line, flush=True)
    with open(log, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {tier} f{frames} {line}\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("screen", "confirm"), default="screen")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--asset", help="E for arm V2 (*.asset.safetensors)")
    ap.add_argument("--frames", type=int)
    a = ap.parse_args()
    A = arms(a.asset)
    names = a.arms or list(A)
    if {"V2", "E5", "PV"} & set(names) and not a.asset:
        sys.exit("arms V2, E5 and PV need --asset")
    if "PV" in names and not Path(os.environ.get("PIVOTAL_LORA", "")).is_file():
        sys.exit("arm PV needs PIVOTAL_LORA=<lora .safetensors>")
    frames = a.frames or (21 if a.tier == "screen" else 41)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "arms.log"
    for name in names:
        res = run_arm(name, A[name], a.tier, frames, log)
        if a.tier == "screen" and name == "C0" and frames == 21 and res["defects"] < 4:
            with open(log, "a") as f:
                f.write(f"   control reproduced only {res['defects']}/6 flips at 21 frames -> screen invalid, "
                        f"switching to 41 frames\n")
            frames = 41
            run_arm(name, A[name], a.tier, frames, log)


if __name__ == "__main__":
    main()
