#!/usr/bin/env python3
"""Apply verifier reports (verify2_report_*.json) to the curation files.

Handles the fix forms the second-pass brief asked for:
  verdict "drop"                       keep -> false, drop_reason (+ notes)
  fix.beats                            full replacement beats for the span
  fix.new_start / fix.new_end          move a span boundary; fix.new_dropped_span (or a list,
                                       fix.new_dropped_spans) fills the frames cut off, so spans
                                       still tile the skit
  fix.size_change / travel / props / characters / framing / angle / time / view   field changes
Anything else in `fix` is printed and NOT applied, for a by-hand decision. Every applied
change appends a note to the span. Run curate_check.py afterwards.

    /workspace/Pudgy/.venv-wan/bin/python apply_verify.py /workspace/Data/processed/v9b_work/verify2_report_*.json [--dry]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("size_change", "travel", "props", "characters", "framing", "angle", "time", "view", "camera", "effects", "prop_events")
KNOWN = {"beats", "new_start", "new_end", "new_dropped_span", "new_dropped_spans", "drop_reason", "notes", "boundary",
         *FIELDS}


def main():
    dry = "--dry" in sys.argv
    reports = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    manual = []
    n_applied = 0
    for rp in reports:
        for r in json.loads(rp.read_text()):
            if r["verdict"] == "ok":
                continue
            f = HERE / r["out"]
            d = json.loads(f.read_text())
            hit = [s for s in d["spans"] if s["start"] == r["start"] and s["end"] == r["end"]]
            if len(hit) != 1:
                manual.append((rp.name, r, "span not found (already changed?)")); continue
            sp = hit[0]
            fx = r.get("fix") or {}
            extra = set(fx) - KNOWN
            if extra:
                manual.append((rp.name, r, f"unknown fix keys {sorted(extra)}")); continue
            tag = f"verification pass 2 ({rp.stem}): "
            if r["verdict"] == "drop":
                sp["keep"] = False
                sp["drop_reason"] = fx.get("drop_reason") or "other"
                if fx.get("camera"):
                    sp["camera"] = fx["camera"]
                sp["notes"] = (sp.get("notes", "") + " | " + tag + (fx.get("notes") or "; ".join(r.get("issues", [])))[:400]).strip(" |")
            else:
                for k in FIELDS:
                    if k in fx:
                        sp[k] = fx[k]
                if "beats" in fx:
                    sp["beats"] = fx["beats"]
                if "boundary" in fx:
                    sp["boundary"] = fx["boundary"]
                news = fx.get("new_dropped_spans") or ([fx["new_dropped_span"]] if fx.get("new_dropped_span") else [])
                i = d["spans"].index(sp)
                if "new_start" in fx:
                    sp["start"] = fx["new_start"]
                if "new_end" in fx:
                    sp["end"] = fx["new_end"]
                for nd in news:
                    span = {"start": nd["start"], "end": nd["end"], "boundary": nd.get("boundary", "same_shot"), "keep": False,
                            "drop_reason": nd["drop_reason"], "camera": nd.get("camera", "static"),
                            "notes": tag + nd.get("notes", "")}
                    d["spans"].append(span)
                d["spans"].sort(key=lambda s: s["start"])
                sp["notes"] = (sp.get("notes", "") + " | " + tag + "; ".join(r.get("issues", []))[:400]).strip(" |")
            n_applied += 1
            print(f"{'(dry) ' if dry else ''}{r['verdict']:<4} {r['out']} [{r['start']},{r['end']})")
            if not dry:
                f.write_text(json.dumps(d, indent=1))
    print(f"\napplied {n_applied}; by hand: {len(manual)}")
    for name, r, why in manual:
        print(f"  MANUAL {name}: {r['out']} [{r['start']},{r['end']}) {r['verdict']} - {why}\n     fix: {json.dumps(r.get('fix'))[:600]}")


if __name__ == "__main__":
    main()
