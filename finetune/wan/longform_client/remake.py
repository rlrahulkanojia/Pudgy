"""Remakes of client skits (raw/iteration_2) with the v8 model, generated and delivered at 480x848, not
upscaled (the user's call, 2026-10-29: ~3.5x faster than 720x1280; run with
GEN_EXTRA="--video_size 848 480"). Every scene is re-generated between the
skit's own frames (first+last frame, a part per <=80 frames, split evenly), colour-matched to them
(assemble_client.match_colour), and cut at the skit's own cut points, so the skit's soundtrack still
fits. Different skits from the September remakes (finetune/wan/longform: LifeWithHer, EatingStages).

python remake.py plan [names]   print every part: frames, start/end frame, prompt
python remake.py run [names]    generate missing parts (one model load), QA each part and re-roll a failed
                                one with a new seed (up to 3 tries), assemble, upload each skit when done
-> final/skits/<name>_480x848.mp4 (silent), _480x848_audio.mp4 (skit soundtrack), _compare.mp4
   (original | remake), qa/skit_<name>.png; Azure pudgy/2026-10-28_longform/skits/
Scene notes come from the v9b skit curation (finetune/wan/scenes/curation); scenes it left without
notes (camera moves, very short shots) are described in NOTES below from the skit's frames.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/longform_client")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/workspace/Pudgy/finetune/wan/longform")
import assemble_client as AC            # noqa: E402
import dupcheck                         # noqa: E402  (September: at most one Pax and one Polly per frame)
import produce as PR                    # noqa: E402
import upload                           # noqa: E402

DATE = "2026-10-28"
CUR = Path("/workspace/Pudgy/finetune/wan/scenes/curation")
SRC = ROOT / "skits/src"                # the client's skit videos (Azure raw/iteration_2/72_videos)
KEYS = ROOT / "skits/keys480"
GW, GH = 480, 848                       # generation = delivery size (~9:16, multiples of 16)
OUT = ROOT / "final/skits"
MAX_PART = 80                           # frame intervals per part: 81 frames, what the model was trained on
SEEDS = (42, 1042, 2042)
GEN = "480p"                            # in the take names: .../takes/skit_<name>_<a>_<b>_480p_s<seed>.mp4
SKITS = {"coming_home": "iteration_2__PP-Cominghome-Base.json",
         "reunion": "iteration_2__2D_PP_Reunion_Base.json",
         "not_mad": "iteration_2__PP-NotMad-Base.json",
         # added 2026-10-29: the fewest props of the 99 (bed / pillow / blanket; 4AM Show: a microphone)
         "little_spoon": "iteration_2__PP-LittleSpoon-Base.json",
         "four_am_show": "iteration_2__PP-4amshow-Base.json",
         # added 2026-10-29: two more bedroom skits like those (no hand props; WakeUpEarly: an alarm clock)
         "morning_fart": "iteration_2__PP-MorningFart-Base.json",
         "wake_up_early": "iteration_2__PP-WakeUpEarly-Base.json"}
CAMERA = {"static": "fixed camera, locked-off shot", "pan_right": "the camera pans slowly to the right",
          "pan_left": "the camera pans slowly to the left", "push_in": "the camera pushes in slowly"}
# (skit, scene start) -> what the curation left out, from the skit's frames
REUNION_PARK = "a sunny park lawn with round cafe tables and striped parasols below a rose-coloured hill"
NOT_MAD_STREET = "a snowy village street lined with igloo-domed shops under striped awnings"
MF_BED = "a bedroom with a wide bed under a pale duvet and a rounded yellow headboard"
WE_BED = "a bedroom with a big yellow bed between two nightstands with lamps"
NOTES = {
    ("reunion", 0): dict(chars=[("pax", "o1")], framing="medium", setting=REUNION_PARK, beats=[
        "Pax stands leaning on a crutch, gazing ahead with a hopeful, beaming face"]),
    ("reunion", 100): dict(chars=[("polly", "o2")], framing="medium",
                           setting="a sunny lawn in front of an ice cream shop with a green door", beats=[
        "Polly turns and waddles off to the left, holding two ice cream cones"]),
    ("not_mad", 0): dict(chars=[("pax", None), ("polly", None)], framing="wide", setting=NOT_MAD_STREET, beats=[
        "Pax stands with his flippers spread wide toward Polly, who holds a red handbag"]),
    ("not_mad", 6): dict(chars=[("pax", None), ("polly", None)], framing="medium", setting=NOT_MAD_STREET, beats=[
        "Pax waddles up to Polly with his flippers open for a hug while she stands with her back to the "
        "camera, a red handbag in her fin"]),
    ("not_mad", 27): dict(chars=[("pax", None), ("polly", None)], framing="medium", setting=NOT_MAD_STREET, beats=[
        "Pax wraps his flippers around Polly in a hug with his eyes closed"]),
    ("morning_fart", 91): dict(chars=[("pax", None), ("polly", None)], framing="medium close-up", setting=MF_BED, beats=[
        "Polly turns her head toward Pax lying beside her", "the duvet ripples over Pax and Polly's eyes go wide, "
        "her flippers raised under her beak"]),
    ("wake_up_early", 0): dict(chars=[("pax", None), ("polly", None)], framing="wide", setting=WE_BED, beats=[
        "Pax and Polly lie fast asleep side by side under the blanket with their beaks open, an alarm clock "
        "on the nightstand"]),
    ("wake_up_early", 106): dict(chars=[("pax", None), ("polly", None)], framing="wide", setting=WE_BED, beats=[
        "the blanket is pulled all the way up over Pax and Polly, a lump shifting underneath it"]),
    ("not_mad", 33): dict(chars=[("pax", None), ("polly", None)], framing="medium close-up", setting=NOT_MAD_STREET,
                          beats=["Seen from behind Pax, he hugs Polly tightly while she stands stiffly with her "
                                 "flippers hanging down, a red handbag in one fin, staring blankly"]),
}


# (skit, part start) -> beats replacing the skit's, generated from the skit's first frame only (no end
# frame: it would pull toward the replaced pose). Coming Home lifts Polly off the floor in both hugs, which
# the model never learned (its hug clips are standing hugs); the user asked for standing hugs, 2026-10-29.
FREE = {
    ("coming_home", 80): ["Pax leans in and presses his face against Polly's cheek",
                          "Pax and Polly wrap their flippers around each other in a tight hug, both standing "
                          "with their feet on the floor, eyes squeezed shut and blushing cheeks",
                          "they keep hugging and sway gently from side to side"],
    ("coming_home", 218): ["Polly clasps her flippers together with blushing cheeks as Pax turns toward her",
                           "Pax leans down and Polly wraps her flippers around him in a hug, both standing with "
                           "their feet on the floor",
                           "they hug tightly with their eyes shut, swaying gently together"],
}


def frames_4k1(n):
    return max(9, -(-(n - 1) // 4) * 4 + 1)


def source(name):
    """The skit at the generation size, all frames."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-i", str(SRC / f"{name}.mp4"), "-vf",
                          f"scale={GW}:{GH}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, GH, GW, 3)


def key(name, src, i):
    p = KEYS / name / f"f{i:04d}.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(src[i]).save(p)
    return p


def scenes(name):
    """[(start, end, prompt-parts)] per shot of the skit, end exclusive."""
    d = json.loads((CUR / SKITS[name]).read_text())
    out = []
    for s in d["spans"]:
        n = NOTES.get((name, s["start"]))
        if n:
            chars, framing, setting, beats = n["chars"], n["framing"], n["setting"], n["beats"]
        else:
            chars = [(c["id"], (c.get("outfit") or [None])[0]) for c in s["characters"]]
            framing, beats = s["framing"], [b["text"] for b in s["beats"]]
            setting = d["settings"][s["setting"]]["desc"] + (f", {s['view']}" if s.get("view") else "")
        who = []
        for cid, o in chars:
            a = PR.P.ANCHORS[cid.capitalize()]
            who.append(a + (f" with {d['outfits'][o]}" if o else ""))
        out.append(dict(start=s["start"], end=s["end"], who=", and ".join(who), ids=[cid for cid, _ in chars], beats=beats,
                        beat_spans=[(b["start"], b["end"]) for b in s.get("beats") or []], framing=framing,
                        camera=CAMERA.get(s["camera"], "fixed camera, locked-off shot"), setting=setting))
    return out


def parts(name, src=None):
    """Every part to generate: first+last frame on the skit's own frames, <= MAX_PART intervals."""
    out = []
    for sc in scenes(name):
        last = sc["end"] - 1
        n = max(1, -(-(last - sc["start"]) // MAX_PART))
        bounds = np.linspace(sc["start"], last, n + 1).round().astype(int)
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            # the beats that happen in this part (all of them when the curation gave no timing)
            beats = [t for t, (s0, s1) in zip(sc["beats"], sc["beat_spans"]) if s0 <= b and s1 > a] or sc["beats"]
            free = FREE.get((name, int(a)))
            prompt = (f"{PR.P.STYLE}{sc['who']}. " + "; then ".join(free or beats) +
                      f"; {sc['framing']} shot, eye level, {sc['camera']}; {sc['setting']}.")
            out.append(dict(tag=f"skit_{name}_{a:04d}_{b:04d}" + ("_free" if free else ""), a=int(a), b=int(b),
                            n=int(b - a + 1), ids=sc["ids"], frames=frames_4k1(int(b - a + 1)), prompt=prompt,
                            start=str(key(name, src, a)) if src is not None else None,
                            end=str(key(name, src, b)) if src is not None and not free else None))
    return out


def part_frames(p, take, src):
    """A generated part resampled to its length in the skit and colour-matched to the skit's frames."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-i", str(take), "-vf",
                          f"scale={GW}:{GH}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, GH, GW, 3)[:p["frames"]]
    f = f[np.linspace(0, len(f) - 1, p["n"]).round().astype(int)]
    # first+last frame re-grades the palette; a part with no end frame (FREE) keeps the skit's
    return AC.match_colour(f, src[p["a"]], src[p["b"]]) if p.get("end") else f


def dups(f):
    """Frames with 2+ Pax-blue / 2+ Polly-pink bodies (September's dupcheck)."""
    small = [np.asarray(Image.fromarray(x).resize((dupcheck.W, dupcheck.H))) for x in f]
    return (sum(dupcheck.bodies(x, *dupcheck.PAX) >= 2 for x in small),
            sum(dupcheck.bodies(x, *dupcheck.POLLY) >= 2 for x in small))


def qa(p, f, src):
    seg = src[p["a"]:p["b"] + 1]
    (pax, pol), (pax0, pol0) = dups(f), dups(seg)
    jump, ref = jumpiness(f), jumpiness(seg)
    land = float(np.abs(f[-1].astype(np.float32) - src[p["b"]]).mean()) if p.get("end") else 0.0
    # Both bars are the original scene's: its own fast moments (a door flung open) reach 6-9x, and pink
    # scenery (Reunion's hills and parasols) reads as extra Pollys in 7-31 of its own frames - so a colour
    # only counts when that character is in the scene.
    # ...and first+last frame must land on the skit's frame (good parts: 1.7-6.9; Reunion's 0.5 s whip pan
    # at 23 had Polly still smiling where she should have turned and run off)
    ok = ((pax <= pax0 + 2 or "pax" not in p["ids"]) and (pol <= pol0 + 2 or "polly" not in p["ids"])
          and jump < max(5.0, 1.5 * ref) and land < 12.0)
    return dict(ok=bool(ok), dup_frames=[int(pax), int(pol)], skit_dup_frames=[int(pax0), int(pol0)],
                max_jump_x_median=round(jump, 1), skit_max_jump=round(ref, 1), end_landing=round(land, 1))


def jumpiness(f):
    """Largest frame-to-frame change / the median one (0 for parts too short to tell)."""
    d = np.abs(np.diff(f[:, ::4, ::4].astype(np.float32), axis=0)).mean((1, 2, 3))
    return float(d.max() / max(np.median(d), 0.3)) if len(d) >= 12 else 0.0


def assemble(name, src, chosen):
    """chosen: [(part, frames)]. Parts of one scene share their boundary frame; scenes meet at a cut."""
    v = np.concatenate([f[1:] if i and p["a"] == chosen[i - 1][0]["b"] else f for i, (p, f) in enumerate(chosen)])
    assert len(v) == len(src), (len(v), len(src))
    OUT.mkdir(parents=True, exist_ok=True)
    silent, audio, cmp_ = (OUT / f"{name}_{GW}x{GH}.mp4", OUT / f"{name}_{GW}x{GH}_audio.mp4", OUT / f"{name}_compare.mp4")
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{GW}x{GH}", "-r", "24", "-i", "-", "-vf", "format=yuv420p",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart",
                            "-an", str(silent)], stdin=subprocess.PIPE)
    enc.stdin.write(v.tobytes()); enc.stdin.close()
    if enc.wait():
        sys.exit(f"{name}: encode failed")
    sk = str(SRC / f"{name}.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(silent), "-i", sk, "-map", "0:v", "-map", "1:a",
                    "-c", "copy", "-shortest", "-movflags", "+faststart", str(audio)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "4", "-i", sk, "-i", str(silent),
                    "-filter_complex", f"[0:v]scale={GW}:{GH}[l];[l][1:v]hstack,format=yuv420p[v]",
                    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
                    "-movflags", "+faststart", str(cmp_)], check=True)
    idx = np.linspace(0, len(v) - 1, 10).round().astype(int)
    rows = [np.concatenate([np.asarray(Image.fromarray(x[i]).resize((180, 318))) for i in idx], 1) for x in (src, v)]
    Image.fromarray(np.concatenate(rows, 0)).save(ROOT / f"qa/skit_{name}.png")
    return [silent, audio, cmp_]


def main():
    cmd, names = sys.argv[1], sys.argv[2:] or list(SKITS)
    if cmd == "plan":
        for name in names:
            ps = parts(name)
            print(f"== {name}: {len(ps)} parts, {sum(p['frames'] for p in ps)} frames to generate")
            for p in ps:
                print(f"  {p['tag']}  {p['n']} -> gen {p['frames']}\n     {p['prompt'][len(PR.P.STYLE):]}")
        return
    import stream as ST                 # only for run: it builds the generator
    gen = None
    for name in names:
        src = source(name)
        chosen, report = [], []
        for p in parts(name, src):
            best = None
            for seed in SEEDS:
                take = PR.TAKES_DIR / (f"{p['tag']}_{GEN}_s{seed}.mp4")
                if not take.exists():
                    if gen is None:
                        ST.say("loading the model")
                        gen = ST.Generator("full")
                    gen.take(dict(p, tag=take.stem, seed=seed, neg=PR.NEG))
                f = part_frames(p, take, src)
                q = qa(p, f, src)
                ST.say(f"part {take.stem}: {q}")
                if best is None or (q["ok"], -q["max_jump_x_median"]) > (best[1]["ok"], -best[1]["max_jump_x_median"]):
                    best = (f, q, seed)
                if q["ok"]:
                    break
            chosen.append((p, best[0]))
            report.append(dict(tag=p["tag"], seed=best[2], **best[1]))
        (ROOT / "qa").mkdir(exist_ok=True)
        (ROOT / f"qa/skit_{name}.json").write_text(json.dumps(report, indent=1))
        links = upload.main(DATE, files=assemble(name, src, chosen), sub="skits")
        for n_, u in links.items():
            ST.say(f"SKIT {n_} uploaded: {u}")
    if gen:
        gen.close()
    ST.say("SKITS DONE")


if __name__ == "__main__":
    main()
