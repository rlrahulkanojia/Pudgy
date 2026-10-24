#!/usr/bin/env python3
"""Load and lint the asset registry (registry.toml), and build caption fragments from it.

Every consumer - prep_v9.py, eval_v9, the long-form prompt compiler and keyframe
composer - imports this module instead of typing asset wording, so a name and its look
cannot drift apart. Design: training_approach/v9/Assets_v9.md section 3.

    python finetune/wan/assets/registry.py lint [--data /workspace/Data]
    python finetune/wan/assets/registry.py show phone
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # the box venv may be 3.10: musubi ships `toml`
    import toml as _toml

    class tomllib:                      # noqa: N801 - mimic the stdlib module
        @staticmethod
        def loads(s):
            return _toml.loads(s)

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.toml"
DATA = Path(os.environ.get("PUDGY_DATA", "/workspace/Data"))

# Colour words that identify the characters. v4 rendered Polly blue when her captions
# lacked "pink" (FINDINGS_v6 section 3), so no prop or set wording may use them.
RESERVED = {"blue", "pink"}
KINDS = {"character", "prop", "set"}


def load(path=REGISTRY):
    return tomllib.loads(Path(path).read_text())["assets"]


def phrase(reg, aid):
    return reg[aid]["phrase"]


def name(reg, aid):
    return reg[aid].get("name") or reg[aid]["phrase"]


def actions(reg, aid):
    return reg[aid].get("actions", [])


def action_clause(reg, aid, act):
    """An in-use action clause with the asset's phrase substituted (first mention)."""
    return act["action"].format(phrase=phrase(reg, aid), name=name(reg, aid))


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref_views(reg, aid, data=DATA):
    """(train, holdout) reference views of a prop, de-duplicated by MD5.

    The client re-sends views under new numbers (PRP_PHONE_25-27 = 01-03); a duplicate
    would let a held-out view leak in under another name, so dedup is by content.
    """
    a = reg[aid]
    files = sorted((Path(data) / a["refs_dir"]).glob(a.get("refs_glob", "*.png")))
    seen, uniq, dupes = {}, [], []
    for f in files:
        h = _md5(f)
        if h in seen:
            dupes.append((f.name, seen[h])); continue
        seen[h] = f.name
        uniq.append(f)
    hold = set(a.get("holdout_views", []))
    held_md5 = {_md5(f) for f in uniq if f.stem in hold}
    train = [f for f in uniq if f.stem not in hold and _md5(f) not in held_md5]
    return train, [f for f in uniq if f.stem in hold], dupes


def _words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def lint(reg, data=None):
    """Problems as strings; empty means the registry is safe to build from."""
    probs = []
    names = {}
    for aid, a in reg.items():
        kind = a.get("kind")
        if kind not in KINDS:
            probs.append(f"{aid}: kind {kind!r} not in {sorted(KINDS)}")
        if not a.get("phrase"):
            probs.append(f"{aid}: no phrase")
            continue
        if kind != "character":
            texts = [a["phrase"], a.get("name", "")] + [x["action"] for x in a.get("actions", [])] \
                + [x.get("label_text", "") for x in a.get("actions", [])] \
                + list(a.get("states", {}).values()) + list(a.get("times", {}).values())
            bad = set().union(*(_words(t) for t in texts)) & RESERVED
            if bad:
                probs.append(f"{aid}: uses character colour word(s) {sorted(bad)}")
            if not a.get("name"):
                probs.append(f"{aid}: a {kind} needs a short `name`")
            elif a["name"] not in a["phrase"]:
                probs.append(f"{aid}: phrase must contain the name {a['name']!r}")
        for x in a.get("actions", []):
            if "{phrase}" not in x["action"]:
                probs.append(f"{aid}/{x.get('label')}: action never names the asset ({{phrase}})")
            for k in ("folder", "label", "action", "label_text"):
                if not x.get(k):
                    probs.append(f"{aid}: an action is missing `{k}`")
        if a.get("name"):
            names[aid] = a["name"].lower()
    # A name inside another name (or phrase) would let one prompt match two assets.
    for a1, n1 in names.items():
        for a2, n2 in names.items():
            if a1 != a2 and (n1 in n2 or re.search(rf"\b{re.escape(n1)}\b", reg[a2]["phrase"].lower())):
                probs.append(f"names clash: {a1} {n1!r} appears in {a2}")
    labels = [x["label"] for a in reg.values() for x in a.get("actions", [])]
    if len(labels) != len(set(labels)):
        probs.append(f"duplicate action labels: {sorted({l for l in labels if labels.count(l) > 1})}")
    if data is not None:
        for aid, a in reg.items():
            if a.get("refs_dir"):
                d = Path(data) / a["refs_dir"]
                if not d.is_dir():
                    probs.append(f"{aid}: refs_dir missing: {d}"); continue
                train, held, dupes = ref_views(reg, aid, data)
                missing = set(a.get("holdout_views", [])) - {f.stem for f in held}
                if missing:
                    probs.append(f"{aid}: holdout views not found: {sorted(missing)}")
                if not train:
                    probs.append(f"{aid}: no training views")
            if a.get("clips"):
                d = Path(data) / a["clips"]["clips_dir"]
                for x in a.get("actions", []):
                    if not list((d / x["folder"]).rglob("*.mov")):
                        probs.append(f"{aid}: no clips under {d / x['folder']}")
            if a.get("plates_dir"):
                d = Path(data) / a["plates_dir"]
                listed = [p["file"] for p in a.get("plates", [])]
                for f in listed + a.get("holdout_plates", []):
                    if not (d / f).exists():
                        probs.append(f"{aid}: plate missing: {f}")
                hs = {}
                for f in listed:
                    if (d / f).exists():
                        h = _md5(d / f)
                        if h in hs:
                            probs.append(f"{aid}: plates {hs[h]} and {f} are the same bytes")
                        hs[h] = f
                for t in {p["time"] for p in a.get("plates", [])} - set(a.get("times", {})):
                    probs.append(f"{aid}: plate time {t!r} has no entry in times")
    return probs


def snapshot(reg):
    """What a training manifest records, so a run can be traced to the wording it saw."""
    return {aid: {k: v for k, v in a.items() if k in ("kind", "name", "phrase", "client_ids",
                                                      "holdout_views", "actions", "states")}
            for aid, a in reg.items()}


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    li = sp.add_parser("lint"); li.add_argument("--data", type=Path, default=None)
    sh = sp.add_parser("show"); sh.add_argument("asset")
    a = ap.parse_args()
    reg = load()
    if a.cmd == "lint":
        probs = lint(reg, a.data)
        for p in probs:
            print(f"  ✗ {p}")
        if probs:
            sys.exit(f"registry: {len(probs)} problem(s)")
        kinds = {}
        for x in reg.values():
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        print(f"registry OK: {len(reg)} assets {kinds}" + ("" if a.data else " (files not checked: pass --data)"))
    else:
        print(json.dumps(reg[a.asset], indent=1))
        for x in actions(reg, a.asset):
            print(f"\n{x['label']}: {action_clause(reg, a.asset, x)}")


if __name__ == "__main__":
    main()
