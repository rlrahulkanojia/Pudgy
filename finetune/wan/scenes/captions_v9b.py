#!/usr/bin/env python3
"""Compile v9b training captions from a curation JSON (scenes/CURATION_v9b.md) and the
asset registry. Captions are never typed by hand: the curator writes what happens (beats,
with frame ranges); this module adds the identity anchors, the setting, the framing and
the camera clause, so every caption has the same structure and wording
(Training_Approach_v9.md section 3.4, decisions D4, D5, D9, D10).

    caption = STYLE + characters (anchor, outfit, screen side) + setting (+ view)
              + the beats inside the window, in order
              + framing, angle, camera clause

    PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY captions_v9b.py curation/iteration_2__PP-Chair-Base.json     # preview every kept span

Window captions: prep_v9b.py cuts a kept span into windows (f21/f33/f49/f81). A window's
caption carries only the beats it shows: a beat is included if the window holds at least
3/4 of a short beat (all of one under 5 frames), at most 12 frames of a long one. A window
that falls inside one long beat gets that beat.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "assets"))
import prep_v8 as P  # noqa: E402
import registry as R  # noqa: E402

REG = R.load()
LEAD_NAMES = {"pax": "Pax", "polly": "Polly"}
# D9: the camera clause for scene clips. "the characters stay the same size" is only true
# when nobody walks toward or away from the camera, so it is dropped when size_change.
CAMERA_CLAUSE = "fixed camera, locked-off shot"
SAME_SIZE = "the characters stay the same size"
SCREEN_WORDS = {"left": "on the left", "right": "on the right", "center": "in the middle",
                "left-center": "left of center", "right-center": "right of center"}


def anchor(cid):
    return REG[cid]["phrase"]


def resolve_refs(text, cast, outfits):
    """{x1} -> the extra's description; a lead's name is written as the name."""
    def sub(m):
        cid = m.group(1)
        if cid == "ifin":
            return "{ifin}"                 # resolved per caption: phrase first, then name
        if cid in LEAD_NAMES:
            return LEAD_NAMES[cid]
        return cast[cid]["desc"]
    return re.sub(r"\{(\w+)\}", sub, text)


# A character who is on screen for only a few frames of a window (comes in at its very
# end, leaves at its very start) is not named in that window's caption, unless a beat the
# window shows names them (curator-09: MessUp f21 window with Polly visible for 2 frames).
MIN_PRESENT = 8


def present(c, a, b, sp=None):
    """Frames of [a, b) in which cast member c (a span characters entry) is on screen."""
    s = c["enters_at"] if c.get("presence") in ("enters", "enters_exits") else (sp["start"] if sp else a)
    e = c["exits_at"] if c.get("presence") in ("exits", "enters_exits") else (sp["end"] if sp else b)
    return max(0, min(b, e) - max(a, s))


# Time-of-day words as they read after the setting (the registry's living_room.times style).
TIME_WORDS = {"morning": "in the morning", "afternoon": "in the afternoon", "evening": "in the evening",
              "night": "at night", "sunset": "at sunset", "sunrise": "at sunrise", "dusk": "at dusk",
              "dawn": "at dawn", "storm": "on a stormy day"}


def screen_at(c, a):
    """Screen side at frame a: the span-start side, updated by `moves` up to a."""
    side = c["screen"]
    for m in sorted(c.get("moves", []), key=lambda m: m["frame"]):
        if m["frame"] <= a:
            side = m["screen"]
    return side


def character_phrase(c, d, a):
    cid = c["id"]
    outfit = [d["outfits"][o] for o in c.get("outfit", [])]
    wear = f", wearing {' and '.join(outfit)}" if outfit else ""
    if cid in LEAD_NAMES:
        head = f"{anchor(cid)}{wear}"
    else:
        head = f"{d['cast'][cid]['desc']}{wear}"
    if c.get("presence") in ("enters", "enters_exits") and c["enters_at"] > a:
        where = f"coming in from the {c['screen']}" if c["screen"] in ("left", "right") else "coming in"
    else:
        where = SCREEN_WORDS[screen_at(c, a)]
    return f"{head}, {where}"


def beat_needs(n):
    """Frames of an n-frame beat a window must show to claim it: 3/4 of a short beat (all
    of one under 5 frames), at most 12 for a long one (an ongoing action is fine to show in
    part). curator-03: with 'half the beat', a 4-frame beat leaked in on 2 frames."""
    return max(min(4, n), min(12, -(-3 * n // 4)))


def pick_beats(beats, a, b):
    out = []
    for bt in beats:
        ov = min(b, bt["end"]) - max(a, bt["start"])
        if ov >= beat_needs(bt["end"] - bt["start"]):
            out.append(bt)
    if not out:
        out = [max(beats, key=lambda bt: min(b, bt["end"]) - max(a, bt["start"]))]
    return out


def sentence(s):
    s = s.strip().rstrip(".")
    return s[0].upper() + s[1:]


def then(s):
    """'Then ' + the beat, keeping a proper name capitalised ("Then Pax ...")."""
    s = s.strip().rstrip(".")
    if s.lower().startswith(("then ", "meanwhile", "at the same time", "at last", "finally")):
        return sentence(s)
    first = s.split()[0]
    keep = first.rstrip(",'s") in LEAD_NAMES.values() or first.isupper()
    return "Then " + (s if keep else s[0].lower() + s[1:])


def compile_window(d, sp, a=None, b=None):
    a = sp["start"] if a is None else a
    b = sp["end"] if b is None else b
    beats = pick_beats(sp["beats"], a, b)
    said = " ".join(bt["text"] for bt in beats)
    def named(c):
        if c["id"] in LEAD_NAMES:
            return re.search(rf"\b{LEAD_NAMES[c['id']]}\b", said) is not None
        return "{" + c["id"] + "}" in said
    chars = [c for c in sp["characters"]
             if present(c, a, b, sp) >= min(MIN_PRESENT, (b - a) // 4) or named(c)]
    # Leads first, left to right; extras after.
    order = {"left": 0, "left-center": 1, "center": 2, "right-center": 3, "right": 4}
    chars.sort(key=lambda c: (c["id"] not in LEAD_NAMES, order[screen_at(c, a)]))
    names = [character_phrase(c, d, a) for c in chars]
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]
    st = d["settings"][sp["setting"]]
    where = f"in {st['desc']}" + (f", {sp['view']}" if sp.get("view") else "")
    time = sp.get("time", "day")
    if time and time != "day":
        where += ", " + (time if time.startswith(("in ", "at ", "on ")) else TIME_WORDS.get(time, f"at {time}"))
    acts = [resolve_refs(bt["text"], d["cast"], d["outfits"]) for bt in beats]
    action = ". ".join(sentence(x) if i == 0 else then(x) for i, x in enumerate(acts)) + "."
    same = SAME_SIZE if len(chars) > 1 else SAME_SIZE.replace("characters stay", "character stays")
    cam = f"{sp['framing']} shot, {sp['angle']}, {CAMERA_CLAUSE}" + ("" if sp["size_change"] else f", {same}")
    cap = f"{P.STYLE}{who}, {where}. {action} {sentence(cam)}."
    # The iFin phone: the registry phrase on first mention, its short name after that
    # (Assets_v9 section 2, "one name, one look").
    # The phrase has commas inside, so when it sits mid-sentence it is closed with a comma
    # ("holds the iFin phone, a small ... tuft on top, and looks at it").
    def first_ifin(m):
        nxt = cap[m.end():m.end() + 1]
        return REG["phone"]["phrase"] + ("," if nxt == " " else "")
    cap = re.sub(r"\{ifin\}", first_ifin, cap, count=1).replace("{ifin}", REG["phone"]["name"])
    return cap


def kept_spans(d):
    return [sp for sp in d["spans"] if sp.get("keep")]


def main():
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # quiet when piped into head
    for f in sys.argv[1:]:
        d = json.loads(Path(f).read_text())
        print(f"== {d['skit']}")
        for sp in kept_spans(d):
            print(f"\n[{sp['start']},{sp['end']}) {sp['end'] - sp['start']}f")
            print(compile_window(d, sp))


if __name__ == "__main__":
    main()
