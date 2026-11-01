"""Collect what the skit-remake write-up needs, per skit: every part's prompt, frames, seeds tried, the kept
seed and its QA, part timings from logs/skits.log, and fresh read-only Azure links.

python skit_report.py little_spoon four_am_show ...   -> qa/skit_report.json
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/longform_client")
sys.path.insert(0, str(ROOT))
import remake as R                      # noqa: E402
import upload                           # noqa: E402


def main(names):
    from azure.storage.blob import BlobServiceClient
    c = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
    log = (ROOT / "logs/skits.log").read_text(errors="ignore")
    out = {}
    for name in names:
        d = json.loads((R.CUR / R.SKITS[name]).read_text())
        kept = {r["tag"]: r for r in json.loads((ROOT / f"qa/skit_{name}.json").read_text())}
        parts = []
        for p in R.parts(name):
            tries = re.findall(rf"## (\d\d:\d\d) UTC part {p['tag']}_480p_s(\d+): (\{{.*\}})", log)
            seeds = sorted({int(s) for _, s, _ in tries})
            parts.append(dict(tag=p["tag"], frames=f"{p['a']}-{p['b']}", length_s=round(p["n"] / 24, 2),
                              generated_frames=p["frames"], end_anchored=not p["tag"].endswith("_free"),
                              prompt=p["prompt"][len(R.PR.P.STYLE):], seeds_tried=seeds,
                              kept_seed=kept[p["tag"]]["seed"], qa={k: v for k, v in kept[p["tag"]].items()
                                                                     if k not in ("tag", "seed")}))
        files = {f"{name}_{R.GW}x{R.GH}_audio.mp4": "with original soundtrack",
                 f"{name}_{R.GW}x{R.GH}.mp4": "silent", f"{name}_compare.mp4": "side by side (original | remake)"}
        out[name] = dict(title=d["storyboard_title"], source=d["skit"], frames=d["frames"],
                         seconds=round(d["frames"] / d["fps"], 2), scenes=len(d["spans"]), parts=parts,
                         links={v: upload.link(c, f"2026-10-28_longform/skits/{k}", days=7) for k, v in files.items()})
    (ROOT / "qa/skit_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: dict(v, parts=len(v["parts"])) for k, v in out.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1:])
