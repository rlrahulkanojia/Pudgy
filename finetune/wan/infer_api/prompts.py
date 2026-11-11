"""Prompt vocabulary, the guided prompt compiler and the prompt checks for the inference API.

The words come from where the model learned them, read without importing anything heavy (the
API runs in a small venv without numpy/torch):
  prep_v8.py               STYLE, ANCHORS, MOTION, EXPRESSIONS   (the v8 caption convention)
  longform_client/produce  SETS, CAM, INTERACTIONS, LOOPS        (the long-form client videos)
  assets/registry.toml     the phone phrase + its in-use actions (E5 is applied on this phrase)
They are parsed as Python literals (ast), so an edit there flows here and nothing can drift.
"""
import ast
import re
import tomllib
from functools import lru_cache
from pathlib import Path

WAN = Path(__file__).resolve().parents[1]

# produce.NEG: Wan's default negative prompt + these anti-duplicate terms (two-character scenes
# grew a second Pax/Polly without them).
NEG_EXTRA = ("，duplicate character, two Pax, two Polly, extra penguin, clone, "
             "second copy of the same character")
LOOP_SUFFIX = ", repeating in a continuous loop"
SHOTS = ("medium shot", "medium wide shot")        # the long-form vocabulary; "close-up" cut mid-take
MAX_PROMPT = 1500
MAX_DETAILS = 300


def _literals(path, names):
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{path}: no literal {sorted(missing)}")
    return out


@lru_cache(maxsize=1)
def vocab():
    p8 = _literals(WAN / "prep_v8.py", ("STYLE", "ANCHORS", "MOTION", "EXPRESSIONS"))
    lf = _literals(WAN / "longform_client" / "produce.py", ("SETS", "CAM", "INTERACTIONS", "LOOPS"))
    reg = tomllib.loads((WAN / "assets" / "registry.toml").read_text())["assets"]
    phone = reg["phone"]
    actions = {}
    for k, v in {**p8["MOTION"], **p8["EXPRESSIONS"]}.items():
        actions[k] = {"kind": "motion" if k in p8["MOTION"] else "expression", "who": "one",
                      "label": v["label"], "clause": v["action"], "loops": k in lf["LOOPS"]}
    for k, clause in lf["INTERACTIONS"].items():
        actions[k] = {"kind": "interaction", "who": "both", "label": k.replace("_", " "), "clause": clause,
                      "loops": False}
    for a in phone.get("actions", []):
        actions[a["label"]] = {"kind": "phone", "who": "one", "label": a["label_text"],
                               "clause": a["action"].format(phrase=phone["phrase"], name=phone.get("name", "")),
                               "loops": False}
    return {"style": p8["STYLE"], "anchors": p8["ANCHORS"], "actions": actions, "sets": lf["SETS"],
            "cam": lf["CAM"], "phone_phrase": phone["phrase"]}


def presets():
    """What the dashboard's guided builder offers (served by GET /v1/info)."""
    v = vocab()
    return {
        "characters": {"pax": "Pax", "polly": "Polly", "both": "Pax and Polly"},
        "actions": {k: {"kind": a["kind"], "who": a["who"], "label": a["label"]} for k, a in v["actions"].items()},
        "sets": {k: s for k, s in v["sets"].items()},
        "shots": list(SHOTS),
        "phone_phrase": v["phone_phrase"],
    }


class PromptError(ValueError):
    pass


def compile_guided(g):
    """{characters, action, setting | setting_text, shot?, details?} -> the prompt, in produce.videos()'s form:
    STYLE + anchor, action clause[, details]; label[ loop]; shot, camera; set."""
    v = vocab()
    who = g.get("characters")
    if who not in ("pax", "polly", "both"):
        raise PromptError("characters must be pax, polly or both")
    act = v["actions"].get(g.get("action") or "")
    if not act:
        raise PromptError(f"unknown action {g.get('action')!r}; see GET /v1/info for the list")
    if (who == "both") != (act["who"] == "both"):
        raise PromptError("two-character actions (wave_together, walk_together, hug) need characters=both, "
                          "and the other actions need a single character")
    if g.get("setting_text"):
        place = clean(g["setting_text"], "setting_text", MAX_DETAILS)
    elif g.get("setting") in v["sets"]:
        place = v["sets"][g["setting"]]
    else:
        raise PromptError(f"setting must be one of {sorted(v['sets'])} or a setting_text")
    details = clean(g.get("details") or "", "details", MAX_DETAILS)
    # The clause is "<what happens>; <label>" (interactions carry their label inline); details go
    # right after what happens, before the label.
    if who == "both":
        subject = f"{v['anchors']['Pax']}, and {v['anchors']['Polly']}"
        shot = g.get("shot") or "medium wide shot"
        cam = v["cam"].replace("the character stays", "the characters stay")
        head, _, label = act["clause"].partition("; ")
    else:
        subject = v["anchors"]["Pax" if who == "pax" else "Polly"]
        shot = g.get("shot") or "medium shot"
        cam = v["cam"]
        head, label = act["clause"], act["label"] + (LOOP_SUFFIX if act["loops"] else "")
    if shot not in SHOTS:
        raise PromptError(f"shot must be one of {list(SHOTS)}")
    body = head + (f", {details}" if details else "") + (f"; {label}" if label else "")
    return f"{v['style']}{subject}, {body}; {shot}, {cam}; {place}."


def clean(text, field, limit):
    """Plain one-line text. musubi's interactive mode reads ONE line per take and splits options on
    ' --', so a newline or '--' in user text would start another generation or inject an option
    (--i /some/path, --cn ...): both are refused, not silently rewritten."""
    text = (text or "").strip()
    if "--" in text:
        raise PromptError(f"{field} must not contain '--'")
    if re.search(r"[\x00-\x1f\x7f]", text):
        raise PromptError(f"{field} must be a single line of text (no line breaks or control characters)")
    if len(text) > limit:
        raise PromptError(f"{field} is {len(text)} characters; the limit is {limit}")
    return text


def check(prompt):
    """Advice for a free-form prompt: the model is most reliable in the words it was trained on.
    Returns [warning strings]; nothing here blocks a job."""
    v = vocab()
    out = []
    low = prompt.lower()
    if not prompt.startswith(v["style"].strip()[:40]):
        out.append("No style prefix: start with \"" + v["style"].strip() + "\" for the trained look.")
    for name, anchor in v["anchors"].items():
        if re.search(rf"\b{name}\b", prompt) and anchor not in prompt:
            out.append(f"{name} is named without the trained wording \"{anchor}\"; the character may drift.")
    if not any(re.search(rf"\b{n}\b", prompt) for n in v["anchors"]):
        out.append("Neither Pax nor Polly is named; the model only knows these two characters.")
    if re.search(r"\b(i?phone|smartphone|mobile)\b", low) and v["phone_phrase"] not in prompt:
        out.append("A phone is mentioned without the registry wording \"" + v["phone_phrase"]
                   + "\", so the learned phone look (E5) will not apply.")
    if "close-up" in low or "close up" in low or "closeup" in low:
        out.append("Close-ups are not in the training vocabulary (a take cut to one mid-shot); "
                   "\"medium shot\" is the reliable framing.")
    return out
