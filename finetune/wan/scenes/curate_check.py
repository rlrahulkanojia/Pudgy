#!/usr/bin/env python3
"""Validate v9b curation JSONs (schema: scenes/CURATION_v9b.md). Every rule here is a
decision from Training_Approach_v9.md section 0.1 (D3-D6, D10), enforced mechanically so
that 99 hand-curated files cannot drift apart.

    PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY curate_check.py curation/iteration_2__PP-Chair-Base.json     # one file
    $PY curate_check.py curation/                                    # all, plus a summary

Exit status 1 if any file has an error. Warnings do not fail.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prep_v8 as P  # noqa: E402

RAW = P.DATA / "raw"
MIN_FRAMES = 21                       # D6: the shortest bucket
DROP_REASONS = {"transition", "camera_move", "graphic_overlay", "ad_merch", "no_lead",
                "lead_hidden", "lookalike", "lead_copy", "off_style", "effect", "too_short",
                "other"}
BOUNDARIES = {"start", "cut", "transition", "after_transition", "camera", "same_shot"}
CAMERAS = {"static", "pan_left", "pan_right", "tilt_up", "tilt_down", "push_in", "pull_back",
           "zoom_in", "zoom_out", "shake", "rotate", "whip", "other"}
FRAMINGS = {"extreme close-up", "close-up", "medium close-up", "medium", "medium wide", "wide"}
ANGLES = {"eye level", "high angle", "low angle", "overhead"}
SCREEN = {"left", "center", "right", "left-center", "right-center"}
PRESENCE = {"full", "partial", "enters", "exits", "enters_exits"}
PROP_EVENTS = {"takes_out", "puts_away", "picks_up", "puts_down", "enters", "exits", "uses",
               "hands_over", "drops"}
LEADS = {"pax", "polly"}
# Words that bind to a lead's identity (registry RESERVED) or describe the camera (the
# compiler adds the camera clause itself; a camera word in an action would contradict it).
COLOUR = {"blue", "pink"}
CAMERA_WORDS = {"camera", "zoom", "zooms", "zooming", "zoomed", "panning", "close-up", "closeup"}
# Also camera words, but with everyday meanings in these skits (a frying pan, a picture
# frame): a warning, for the curator to confirm.
CAMERA_MAYBE = {"pan", "pans", "shot", "frame", "framed"}


def ngrams(text, n=6):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


V8_CLAUSES = [d["action"] for d in list(P.MOTION.values()) + list(P.EXPRESSIONS.values())]
V8_GRAMS = set().union(*(ngrams(c) for c in V8_CLAUSES))


_COUNTS = {}


def frame_count(path):
    """Frames in the stream (= passthrough decode count for these H.264 files)."""
    if path not in _COUNTS:
        import subprocess
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                              "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True).stdout.strip()
        _COUNTS[path] = int(out.split(",")[0])
    return _COUNTS[path]


def words(s):
    """Words, with hyphenated compounds split too: "grey-blue" must count as "blue" (it
    reached two captions before this split), while "close-up" is still caught whole."""
    low = s.lower()
    return set(re.findall(r"[a-z\-]+", low)) | set(re.findall(r"[a-z]+", low))


def check(path):
    errs, warns = [], []
    E = errs.append
    W = warns.append
    try:
        d = json.loads(Path(path).read_text())
    except Exception as e:                           # noqa: BLE001
        return [f"not JSON: {e}"], []
    for k in ("skit", "frames", "cast", "outfits", "settings", "spans", "curated_by"):
        if k not in d:
            E(f"missing top-level key {k!r}")
    if errs:
        return errs, warns
    if not (RAW / d["skit"]).exists():
        E(f"skit not found: {RAW / d['skit']}")
    else:
        nf = frame_count(RAW / d["skit"])
        if d["frames"] != nf:
            E(f"frames is {d['frames']} but the skit has {nf} frames")
    cast, outfits, settings = d["cast"], d["outfits"], d["settings"]
    for lead in LEADS & set(cast):
        if cast[lead].get("kind") != "lead":
            E(f"cast {lead}: kind must be 'lead'")
    for cid, c in cast.items():
        if cid in LEADS:
            continue
        if c.get("kind") != "extra":
            E(f"cast {cid}: non-lead cast must be kind 'extra'")
        if not c.get("desc"):
            E(f"cast {cid}: needs desc")
        elif words(c["desc"]) & COLOUR:
            E(f"cast {cid}: desc uses a lead colour word {sorted(words(c['desc']) & COLOUR)}")
        if c.get("lookalike"):
            W(f"cast {cid}: marked lookalike - every span with it must be dropped")
    for oid, o in outfits.items():
        if not isinstance(o, str) or not o:
            E(f"outfit {oid}: must be a non-empty string")
        elif words(o) & COLOUR:
            E(f"outfit {oid}: uses a lead colour word")
        elif not o.startswith(("a ", "an ", "the ")) and not re.match(r"^[a-z]+ ", o):
            W(f"outfit {oid}: write it as a noun phrase, e.g. 'a yellow hard hat'")
    for sid, s in settings.items():
        if s.get("kind") not in ("flat", "set"):
            E(f"setting {sid}: kind must be 'flat' or 'set'")
        if not s.get("desc"):
            E(f"setting {sid}: needs desc")
        elif words(s["desc"]) & COLOUR:
            E(f"setting {sid}: desc uses a lead colour word {sorted(words(s['desc']) & COLOUR)}")

    spans = d["spans"]
    n = d["frames"]
    prev = 0
    for i, sp in enumerate(spans):
        tag = f"span {i} [{sp.get('start')},{sp.get('end')})"
        s0, s1 = sp.get("start"), sp.get("end")
        if not isinstance(s0, int) or not isinstance(s1, int) or s1 <= s0:
            E(f"{tag}: start/end must be ints with end > start"); continue
        if s0 != prev:
            E(f"{tag}: starts at {s0} but the previous span ended at {prev} (spans must tile every frame)")
        prev = s1
        b = sp.get("boundary")
        if b not in BOUNDARIES:
            E(f"{tag}: boundary {b!r} not in {sorted(BOUNDARIES)}")
        if i == 0 and b != "start":
            E(f"{tag}: the first span's boundary must be 'start'")
        cam = sp.get("camera")
        if cam not in CAMERAS:
            E(f"{tag}: camera {cam!r} not in {sorted(CAMERAS)}")
        keep = sp.get("keep")
        if not isinstance(keep, bool):
            E(f"{tag}: keep must be true/false"); continue
        if not keep:
            if sp.get("drop_reason") not in DROP_REASONS:
                E(f"{tag}: drop_reason {sp.get('drop_reason')!r} not in {sorted(DROP_REASONS)}")
            if sp.get("drop_reason") == "other" and not sp.get("notes"):
                E(f"{tag}: drop_reason 'other' needs notes")
            continue
        # ---- kept spans ----
        if s1 - s0 < MIN_FRAMES:
            E(f"{tag}: kept span is {s1 - s0} frames, below f{MIN_FRAMES} (drop it as too_short)")
        if cam != "static":
            E(f"{tag}: kept span must be camera 'static' (D3), got {cam!r}")
        if b in ("transition",):
            E(f"{tag}: a transition span cannot be kept")
        if sp.get("framing") not in FRAMINGS:
            E(f"{tag}: framing {sp.get('framing')!r} not in {sorted(FRAMINGS)}")
        if sp.get("angle") not in ANGLES:
            E(f"{tag}: angle {sp.get('angle')!r} not in {sorted(ANGLES)}")
        if sp.get("setting") not in settings:
            E(f"{tag}: setting {sp.get('setting')!r} not defined in settings")
        view = sp.get("view", "")
        if view and not view.startswith("with "):
            E(f"{tag}: view must start with 'with ' (it is appended to the setting phrase)")
        if words(view) & COLOUR:
            E(f"{tag}: view uses a lead colour word {sorted(words(view) & COLOUR)}")
        if words(view) & CAMERA_WORDS:
            E(f"{tag}: view uses a camera word {sorted(words(view) & CAMERA_WORDS)}")
        if not isinstance(sp.get("size_change"), bool):
            E(f"{tag}: size_change must be true/false (does a character grow/shrink by moving toward/away from the camera?)")
        if sp.get("overlays"):
            E(f"{tag}: kept span has overlays {sp['overlays']} (drop or split around them)")
        chars = sp.get("characters", [])
        ids = [c.get("id") for c in chars]
        if not LEADS & set(ids):
            E(f"{tag}: kept span has no lead in characters")
        if len(ids) != len(set(ids)):
            E(f"{tag}: a character is listed twice")
        for c in chars:
            cid = c.get("id")
            if cid not in cast:
                E(f"{tag}: character {cid!r} not in cast"); continue
            if cast[cid].get("lookalike"):
                E(f"{tag}: kept span contains lookalike {cid} (D4: drop)")
            if c.get("screen") not in SCREEN:
                E(f"{tag}: {cid} screen {c.get('screen')!r} not in {sorted(SCREEN)}")
            if c.get("presence", "full") not in PRESENCE:
                E(f"{tag}: {cid} presence {c.get('presence')!r} not in {sorted(PRESENCE)}")
            if c.get("presence") in ("enters", "exits", "enters_exits"):
                for k in ([] if c["presence"] == "exits" else ["enters_at"]) + ([] if c["presence"] == "enters" else ["exits_at"]):
                    if not isinstance(c.get(k), int) or not s0 <= c[k] < s1:
                        E(f"{tag}: {cid} {c['presence']} needs {k} inside the span")
            for m in c.get("moves", []):
                if not isinstance(m.get("frame"), int) or not s0 <= m["frame"] < s1 or m.get("screen") not in SCREEN:
                    E(f"{tag}: {cid} move {m} needs a frame inside the span and a screen in {sorted(SCREEN)}")
            for o in c.get("outfit", []):
                if o not in outfits:
                    E(f"{tag}: {cid} outfit {o!r} not defined in outfits")
        beats = sp.get("beats", [])
        if not beats:
            E(f"{tag}: kept span needs beats")
        else:
            if beats[0].get("start") != s0:
                E(f"{tag}: first beat must start at the span start {s0}")
            if beats[-1].get("end") != s1:
                E(f"{tag}: last beat must end at the span end {s1}")
            last_end = s0
            for j, bt in enumerate(beats):
                bs, be, tx = bt.get("start"), bt.get("end"), bt.get("text", "")
                if not isinstance(bs, int) or not isinstance(be, int) or not s0 <= bs < be <= s1:
                    E(f"{tag} beat {j}: start/end must lie inside the span with end > start"); continue
                if bs > last_end + 6:
                    E(f"{tag} beat {j}: gap of {bs - last_end} frames after the previous beat (describe every moment)")
                if bs < last_end - 24:
                    W(f"{tag} beat {j}: overlaps the previous beat by {last_end - bs} frames")
                last_end = max(last_end, be)
                if not tx or len(tx.split()) < 3:
                    E(f"{tag} beat {j}: text too short")
                wv = words(tx)
                if wv & COLOUR:
                    E(f"{tag} beat {j}: uses a lead colour word {sorted(wv & COLOUR)} (the anchors carry colour)")
                if wv & CAMERA_WORDS:
                    E(f"{tag} beat {j}: camera/framing word {sorted(wv & CAMERA_WORDS)} (the compiler adds the camera clause)")
                if wv & CAMERA_MAYBE:
                    W(f"{tag} beat {j}: {sorted(wv & CAMERA_MAYBE)} - fine if it is an object, not the camera")
                if '"' in tx or "“" in tx:
                    E(f"{tag} beat {j}: quotes - never transcribe on-screen text or dialogue")
                hit = ngrams(tx) & V8_GRAMS
                if hit:
                    E(f"{tag} beat {j}: copies a v8 action clause verbatim ({sorted(hit)[0]!r}...) - D10")
                for ref in re.findall(r"\{(\w+)\}", tx):
                    if ref != "ifin" and ref not in cast:
                        E(f"{tag} beat {j}: {{{ref}}} is not in cast")
                for nm, cid in (("Pax", "pax"), ("Polly", "polly")):
                    if re.search(rf"\b{nm}\b", tx) and cid not in ids:
                        E(f"{tag} beat {j}: mentions {nm} but {cid} is not in the span's characters")
        for ev in sp.get("prop_events", []):
            if ev.get("event") not in PROP_EVENTS:
                E(f"{tag}: prop event {ev.get('event')!r} not in {sorted(PROP_EVENTS)}")
            if not isinstance(ev.get("frame"), int) or not s0 <= ev["frame"] < s1:
                E(f"{tag}: prop event frame must lie inside the span")
            if ev.get("prop") == "phone" and ev.get("is_ifin") not in (True, False, "unsure"):
                E(f"{tag}: a phone event needs is_ifin true/false/'unsure'")
    if prev != n:
        E(f"spans end at {prev} but the skit has {n} frames")
    return errs, warns


def summary(files):
    kept = drop = 0
    kept_f = 0
    reasons = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for sp in d["spans"]:
            if sp.get("keep"):
                kept += 1; kept_f += sp["end"] - sp["start"]
            else:
                drop += 1
                reasons[sp.get("drop_reason")] = reasons.get(sp.get("drop_reason"), 0) + sp["end"] - sp["start"]
    print(f"\n{len(files)} files: {kept} kept spans ({kept_f / 24 / 60:.1f} min), {drop} dropped spans")
    print("dropped frames by reason:", {k: v for k, v in sorted(reasons.items(), key=lambda x: -x[1])})


def main():
    args = sys.argv[1:] or [str(HERE / "curation")]
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    bad = 0
    for f in files:
        errs, warns = check(f)
        if errs or warns:
            print(f"{f.name}: {len(errs)} error(s), {len(warns)} warning(s)")
            for e in errs:
                print(f"  ✗ {e}")
            for w in warns:
                print(f"  ! {w}")
        bad += bool(errs)
    if not bad:
        print(f"OK: {len(files)} file(s) valid")
    if len(files) > 1:
        summary(files)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
