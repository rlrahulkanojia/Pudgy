#!/usr/bin/env python3
"""Build the eval keyframes the v7 gates drive I2V from.

Two families, and the distinction is the whole point of G-F vs G-M (plan section 7):

  TRAINING frames — frame 0 of an actual v7 training clip, composited exactly as the
                    trainer saw it. Driving from these reproduces the condition that
                    caused v5's memorisation, so G-F can test for it.
  NOVEL frames    — a v1 skit frame: different pose, scene, framing and palette.
                    Nothing in the 2,272 clips resembles it, so G-M can show motion
                    survives away from the training distribution.

Unlike prep_eval_keyframes_v6.py this does NOT need the ProRes-4444 raws: the v7 clips
are already on the box at /workspace/data_v7/clips, and frame 0 of one IS the training
start frame by construction. Only the novel frames are fetched (from Azure
processed/v1v2_75clip/), which is two small mp4s rather than tens of GB of raws.

Filenames are exactly what gates_v7.py expects:
    <char>_neutral_start.png        expression gates
    <char>_idle_start.png           motion gates (same standing pose, motion naming)
    <char>_neutral_<zoom>_start.png G-F, per shot size
    novel/<char>_novel_start.png    G-M
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

CLIPS = Path("/workspace/data_v7/clips")
OUT = Path("/workspace/eval_v7/keyframes")
# Solo-character v1 skit clips (verified from the v1v2_75clip captions): the character
# is alone and standing, so the novel frame is a clean single-subject conditioning frame.
NOVEL = {"Pax": "00000030", "Polly": "00000002"}
BLOB = "processed/v1v2_75clip/train"


def frame0(src, dest, frame=0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", str(dest)],
                   check=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bg", default="white", help="ground for the training frames")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    made, meta = [], {}
    man = json.loads(Path("/workspace/data_v7/manifest.json").read_text())

    # --- training frames: frame 0 of a real v7 neutral clip -------------------------
    for char in ("Pax", "Polly"):
        c = char.lower()
        # f21 is the COMMON bucket — every label appears there, so a frame from it is
        # not associated with any one label's length.
        src = CLIPS / f"{c}_neutral_FRONT__{args.bg}_f21.mp4"
        if not src.exists():
            sys.exit(f"missing training clip {src}")
        frame0(src, out / f"{c}_neutral_start.png")
        made.append(f"{c}_neutral_start.png")
        meta[f"{c}_neutral_start.png"] = {"background": args.bg, "source": src.name}
        meta[f"{c}_idle_start.png"] = {"background": args.bg, "source": src.name}
        # Motion gates use the same standing pose under the motion naming.
        frame0(src, out / f"{c}_idle_start.png")
        made.append(f"{c}_idle_start.png")
        # G-F drives from a training frame at EACH SHOT SIZE. Zoom is a property of the
        # clip (the manifest ties it to the character/angle/background combination), so
        # the frames must be selected from the manifest rather than re-cropped here —
        # re-cropping would produce a frame the trainer never saw, which is precisely
        # what G-F must not do.
        for zoom, tag in ((1.00, "close"), (0.75, "medium"), (0.55, "wide")):
            cand = [o for cl in man["clips"] if cl["label"] == "neutral"
                    and cl["character"] == char and cl["angle"].startswith("FRONT")
                    for o in cl["outputs"] if o["zoom"] == zoom and o["frames"] == 21]
            if not cand:
                print(f"   !! no {char} neutral FRONT clip at zoom {zoom} — skipping {tag}")
                continue
            pick = cand[0]
            frame0(CLIPS / pick["name"], out / f"{c}_neutral_{tag}_start.png")
            made.append(f"{c}_neutral_{tag}_start.png")
            meta[f"{c}_neutral_{tag}_start.png"] = {"zoom": zoom,
                                                    "background": pick["background"],
                                                    "source": pick["name"]}

    # --- novel frames: a v1 skit frame, fetched from Azure ---------------------------
    from azure.storage.blob import BlobServiceClient
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set (expected in /workspace/.env)")
    svc = BlobServiceClient.from_connection_string(cs)
    cc = svc.get_container_client("pudgy")
    tmp = Path("/tmp/v1novel"); tmp.mkdir(exist_ok=True)
    (out / "novel").mkdir(parents=True, exist_ok=True)
    for char, clip in NOVEL.items():
        c = char.lower()
        local = tmp / f"{clip}.mp4"
        if not local.exists():
            local.write_bytes(cc.get_blob_client(f"{BLOB}/{clip}.mp4").download_blob().readall())
        # Frame 8, not 0: v1 clips often open mid-fade, and a fade frame is a poor
        # conditioning image (low contrast biases the first denoising steps).
        #
        # v1 is 768x1360 portrait and the gates run at 1024x1024, so it must be squared.
        # CENTRE CROP, matching prep_eval_keyframes_v6: letterboxing would add black bars
        # that the model reads as scene content, and a plain rescale would squash the
        # character — which would corrupt G-M's subject x-range measurement.
        dest = out / "novel" / f"{c}_novel_start.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(local), "-vf",
                        "select=eq(n\\,8),crop='min(iw,ih)':'min(iw,ih)',scale=1024:1024",
                        "-frames:v", "1", str(dest)], check=True)
        made.append(f"novel/{c}_novel_start.png")

    # Sidecar: which ground/zoom each frame actually has. The gates need the ground to
    # build a subject mask (subject_stats_flat keys off the flat background colour), and
    # guessing "white" would silently corrupt every x-range measurement.
    (out / "keyframes.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(made)} keyframes to {out}")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
