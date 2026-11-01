"""Start frames for the client long-form videos: frame 0 of the client's own alpha clip
(iteration_6, ProRes 4444 with contact shadow) composited onto the client's living-room set
plates (iteration_6 BGS, 1080x1920), then downscaled to the 720x1280 generation size.

python keyframes.py            -> keyframes/<item>_<char>.png (+ keyframes_sheet.png)
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path("/workspace/longform_client")
CUT = ROOT / "cutouts"
BG = ROOT / "raw/BGS/BG_LIVINGROOM"
OUT = ROOT / "keyframes"
W, H = 720, 1280                     # generation size (portrait)

PLATES = {                           # unique plates picked from plates_sheet.png
    "sofa":   "Frame_001 (3).png",                          # cream sofa centred, open floor
    "door":   "AFrame_002.png",                        # sofa left, open door, glass table
    "window": "Frame_03.png",                          # igloo wall, round window, door
    "lounge": "PP_BG_Ep089Sc04_TVRoom_002.png",                          # sofa right, round window, glass table
}

# item -> (cutout file pattern with {C} for PAX/POLLY, plate, body height as a share of frame
# height, feet line as a share of frame height, horizontal centre as a share of width)
SINGLE = {
    "walking":   ("MOTION_CLIPS__MOTION_WALKING__{C}__{C}_MOTION_WALKING_FRONT.png", "sofa", .34, .84, .5),
    "running":   ("MOTION_CLIPS__MOTION_RUNNING__{C}__{C}_MOTION_RUNNING_FRONT.png", "window", .34, .84, .5),
    "waving":    ("MOTION_CLIPS__MOTION_WAVING__{C}__{C}_MOTION_WAVING_FRONT.png", "door", .32, .88, .35),  # .27 slid to ~.35
    "sitting":   ("MOTION_CLIPS__MOTION_SITTING__V2__{C}__{C}_MOTION_SITTING_FRONT.png", "sofa", .30, .84, .5),
    "jumping":   ("MOTION_CLIPS__MOTION_JUMPING__{C}__{C}_MOTION_JUMPING_FRONT.png", "window", .32, .86, .5),
    "happy":     ("EXPRESSIONS__HAPPY__{C}__{C}_HAPPY_FRONT.png", "window", .40, .86, .5),
    "surprised": ("EXPRESSIONS__SURPRISE__{C}__{C}_EXPRESSION_SURPRISE_FRONT.png", "sofa", .40, .86, .5),
    "angry":     ("EXPRESSIONS__ANGRY__{C}__{C}_ANGRY_FRONT.png", "window", .40, .86, .5),
    "confused":  ("EXPRESSIONS__CONFUSED__V2__{C}__{C}_MOTION_CONFUSED_FRONT.png", "lounge", .32, .90, .68),
    "crying":    ("EXPRESSIONS__CRYING__{C}__{C}_MOTION_CRYING_FRONT.png", "window", .40, .86, .5),
    "neutral":   ("EXPRESSIONS__NEUTRAL__{C}__{C}_EXPRESSION_NEUTRAL_FR.png", "sofa", .40, .86, .5),
    "laughing":  ("EXPRESSIONS__LAUGHING__{C}__{C}_EXPRESSION_LAUGHING_FRONT.png", "window", .40, .86, .5),
    "blushing":  ("EXPRESSIONS__BLUSHING__{C}__{C}_EXPRESSION_BLUSHING_FRONT.png", "door", .34, .88, .35),  # (2026-10-29)
}
PAIR = {                             # two-character interaction clips (both characters in one cutout)
    "wave_together": ("INTERACTION_CLIPS__WAVING__INTERACTION_WAVING_FRONT.png", "sofa", .30, .86, .5),
    "walk_together": ("INTERACTION_CLIPS__WALKING__INTERACTION_WALKING_FRONT.png", "window", .30, .86, .5),
    "hug":           ("INTERACTION_CLIPS__HUGGING__INTERACTION_HUGGING_QF1_L.png", "sofa", .30, .86, .5),
}


def cutout(name):
    """The client file names differ in case/prefix per folder: HAPPY/Pax/HAPPY_FRONT.mov."""
    p = CUT / name
    if not p.exists() and "HAPPY" in name:
        p = CUT / name.replace("__PAX__PAX_HAPPY", "__Pax__HAPPY").replace("__POLLY__POLLY_HAPPY", "__POLLY__POLLY_HAPPY")
    return Image.open(p).convert("RGBA")


def compose(cut, plate, body_h, feet_y, cx):
    bg = Image.open(BG / PLATES[plate]).convert("RGB").resize((W, H), Image.LANCZOS)
    a = cut.getchannel("A")
    body = a.point(lambda v: 255 if v > 128 else 0).getbbox()      # character, without the shadow
    full = a.getbbox()                                              # incl. the contact shadow
    s = body_h * H / (body[3] - body[1])
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    x = round(cx * W - (body[0] + body[2]) / 2 * s)
    y = round(feet_y * H - body[3] * s)                             # feet on the floor line
    assert full[3] * s + y <= H, "shadow runs off the frame"
    bg.paste(cut, (x, y), cut)
    return bg, dict(scale=round(s, 4), x=x, y=y)


def main():
    OUT.mkdir(exist_ok=True)
    meta = {}
    for item, (pat, plate, bh, fy, cx) in SINGLE.items():
        for C in ("PAX", "POLLY"):
            im, m = compose(cutout(pat.format(C=C)), plate, bh, fy, cx)
            k = f"{item}_{C.lower()}"
            im.save(OUT / f"{k}.png"); meta[k] = dict(plate=plate, **m)
    for item, (f, plate, bh, fy, cx) in PAIR.items():
        im, m = compose(cutout(f), plate, bh, fy, cx)
        im.save(OUT / f"{item}.png"); meta[item] = dict(plate=plate, **m)
    (OUT / "keyframes.json").write_text(json.dumps(meta, indent=1))
    ks = sorted(meta)
    tw, th, cols = 144, 256, 8
    sheet = Image.new("RGB", (cols * tw, -(-len(ks) // cols) * th), "white")
    for i, k in enumerate(ks):
        sheet.paste(Image.open(OUT / f"{k}.png").resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    sheet.save(ROOT / "keyframes_sheet.png")
    print(len(ks), "keyframes:", " ".join(ks))


if __name__ == "__main__":
    main()
