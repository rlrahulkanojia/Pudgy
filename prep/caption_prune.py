#!/usr/bin/env python3
"""
caption_prune.py — caption anti-entanglement pruning (Part C3 of the Training Runbook).

Each final caption = fixed identity ANCHOR + VLM-generated MOTION SUFFIX. The suffix
must describe *dynamics only*; any appearance description duplicates the anchor and
entangles identity with motion. This script removes appearance sentences from the
suffix via two stages, then samples 10% for manual spot-check calibration.

Stage 1  keyword blocklist   — drop sentences containing anchor appearance terms.
Stage 2  semantic similarity — drop sentences whose embedding cosine-similarity to an
                               anchor clause exceeds THRESHOLD (default 0.7).
                               Requires `sentence-transformers`; skipped (with warning)
                               if unavailable so Stage 1 still runs.

Input  : a JSONL file, one object per clip: {"clip_id","anchor","suffix"}
Output : <out>/captions/<clip_id>.txt   (anchor + pruned suffix)
         <out>/spotcheck.csv            (10% sample: before/after for manual review)

Usage:
  python caption_prune.py --in captions_raw.jsonl --out /vol/dataset_v1 \
      --threshold 0.7 --sample 0.10
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Appearance terms already covered by the identity anchor (+ common VLM rephrasings).
BLOCKLIST = [
    r"penguin", r"pudgy", r"plump", r"rotund", r"chubby", r"bird", r"avian",
    r"white belly", r"heart[- ]shaped", r"thick (black )?outline", r"black outline",
    r"orange beak", r"beak", r"flipper(?=s? (is|are) (orange|black))",  # color of flipper only
    r"scarf", r"neckwear", r"hat", r"blush", r"cheeks", r"tuft",
    r"crimson", r"azure", r"cornflower", r"rose[- ]pink", r"cartoon style", r"art style",
]
BLOCK_RE = re.compile("|".join(BLOCKLIST), re.IGNORECASE)

SENT_SPLIT = re.compile(r"(?<=[.;,])\s+|\s+(?:while|and)\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip(" ,.;") for s in SENT_SPLIT.split(text) if s.strip(" ,.;")]
    return parts


def stage1_blocklist(sentences: list[str]) -> tuple[list[str], list[str]]:
    keep, dropped = [], []
    for s in sentences:
        (dropped if BLOCK_RE.search(s) else keep).append(s)
    return keep, dropped


def stage2_semantic(sentences: list[str], anchor: str, threshold: float):
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except ImportError:
        print("  ! sentence-transformers not installed; skipping Stage 2 "
              "(install: pip install sentence-transformers)")
        return sentences, []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    anchor_clauses = split_sentences(anchor)
    a_emb = model.encode(anchor_clauses, convert_to_tensor=True)
    keep, dropped = [], []
    for s in sentences:
        s_emb = model.encode(s, convert_to_tensor=True)
        sim = float(util.cos_sim(s_emb, a_emb).max())
        (dropped if sim > threshold else keep).append(s)
    return keep, dropped


def prune_one(rec: dict, threshold: float) -> dict:
    anchor, suffix = rec["anchor"].strip(), rec["suffix"].strip()
    sents = split_sentences(suffix)
    kept, d1 = stage1_blocklist(sents)
    kept, d2 = stage2_semantic(kept, anchor, threshold)
    pruned_suffix = ", ".join(kept)
    final = f"{anchor.rstrip(',')}, {pruned_suffix}.".replace(" ,", ",")
    return {
        "clip_id": rec["clip_id"],
        "anchor": anchor,
        "suffix_in": suffix,
        "suffix_out": pruned_suffix,
        "dropped": "; ".join(d1 + d2),
        "final": final,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="captions_raw.jsonl")
    ap.add_argument("--out", required=True, help="dataset dir (writes captions/ + spotcheck.csv)")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--sample", type=float, default=0.10)
    args = ap.parse_args()

    out = Path(args.out)
    cap_dir = out / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(prune_one(json.loads(line), args.threshold))

    for r in results:
        (cap_dir / f"{r['clip_id']}.txt").write_text(r["final"] + "\n")

    # Deterministic 10% sample (every Nth) for manual spot-check calibration.
    step = max(1, int(round(1 / args.sample))) if args.sample > 0 else len(results) + 1
    sample = results[::step]
    sc = out / "spotcheck.csv"
    with open(sc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "suffix_in", "suffix_out", "dropped", "final"])
        w.writeheader()
        for r in sample:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"Pruned {len(results)} captions -> {cap_dir}")
    print(f"Spot-check sample ({len(sample)} rows, ~{args.sample:.0%}) -> {sc}")
    print("Review spotcheck.csv: confirm no appearance terms survive and no motion is "
          "over-pruned, then re-run with an adjusted --threshold if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
