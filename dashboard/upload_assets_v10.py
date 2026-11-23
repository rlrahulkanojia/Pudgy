#!/usr/bin/env python3
"""Publish the v10 showcase to the dashboard container (`AZURE_DASHBOARD_CONTAINER`).

Source of truth is the training container (`pudgy`, v10/remakes/<DATE>/, written by the GPU box's
remake run), so this runs from any machine with the .env. Each clip is remuxed with +faststart
(moov atom first) so the browser can start playback before the whole file arrives, and renamed so
the dashboard caption reads as the skit's title.

Groups (manifest.py `video_groups` for v10, displayed with "_" -> " "):
  Full-length_remakes  held-out skits remade end to end with v10 step 300, with the skit's own sound
  Side_by_side         original (held out) | v8merged (production) | v10 step 300
gentle_one is left out on purpose (user, 2026-11-13): its street-walk shot is weak in both models.
A side-by-side is only published once it has all three panels (1440 px wide); before the v8merged
remake of a skit exists it is a two-panel clip and is skipped. Re-running is safe: a blob whose
recorded source size matches is skipped.
Prompts: v10/prompts.json (manifest.py `prompts_blob`), every shot's prompt and seed from the run's
qa/v10_300_<skit>.json, shown under the videos. Rewritten on every run (small).

    python dashboard/upload_assets_v10.py [--dry-run]
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
SRC, DST = "pudgy", os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
BASE = "v10/remakes/2026-11-13"
SKITS = {"sleepless": "Sleepless", "leftovers": "Leftovers", "highlighter": "Highlighter",
         "beach_activities": "Beach_Activities", "working_in_bed": "Working_In_Bed", "blossom": "Blossom",
         "charging": "Charging", "security": "Security", "dyli": "DYLI"}
THREE_PANELS = 1440
# remake_v10.negative(): Wan's shared sample_neg_prompt plus these terms.
NEG_EXTRA = "duplicate character, two Pax, two Polly, extra penguin, clone, second copy of the same character"
NOTE = ("Each remake is made shot by shot: every shot is generated between the original skit's own first and "
        "last frames from the prompt shown, then joined at the original cuts. v8 and v10 got the same prompts. "
        "Most prompts use the caption format v10 was trained on; a few short camera-move shots have hand-written "
        "prompts. The seed is the one v10 used (the best of up to three, picked by the automatic checks). "
        f"Negative prompt: Wan 2.2's standard one plus \"{NEG_EXTRA}\".")

ITEMS = []
for key, title in SKITS.items():
    ITEMS.append((f"{BASE}/final/v10_300/{key}_audio.mp4", f"Full-length_remakes/{title}.mp4", None))
    ITEMS.append((f"{BASE}/compare/{key}_compare.mp4", f"Side_by_side/{title}.mp4", THREE_PANELS))


def width(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]["width"]


def prompts_doc(src):
    """Per-shot prompts from the remake run's QA records, keyed like the video captions (title, "_" -> " ")."""
    skits = {}
    for key, title in SKITS.items():
        qa = json.loads(src.get_blob_client(f"{BASE}/qa/v10_300_{key}.json").download_blob().readall())
        fps, shots, prev = qa["frames"] / qa["seconds"], [], -1
        for p in qa["parts"]:
            a, b = (int(x) for x in p["tag"].rsplit("_", 2)[1:])   # tag: <skit>_<first frame>_<last frame>
            assert a <= prev + 1 < b + 1, (key, p["tag"])            # in order, no gap
            shots.append(dict(start=round(a / fps, 2), end=round((b + 1) / fps, 2), frames=p["frames"],
                              seed=p["seed"], manual=p["manual"], prompt=p["prompt"]))
            prev = b
        assert prev == qa["frames"] - 1, (key, prev, qa["frames"])   # the shots reach the skit's last frame
        skits[title.replace("_", " ")] = dict(seconds=qa["seconds"], shots=shots)
    return dict(note=NOTE, skits=skits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    svc = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    src, dst = svc.get_container_client(SRC), svc.get_container_client(DST)
    sent = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for s, d, need_width in ITEMS:
            d = f"v10/{d}"
            sc = src.get_blob_client(s)
            if not sc.exists():
                print(f"  - {d}: source missing ({s})"); skipped += 1; continue
            size = str(sc.get_blob_properties().size)
            dc = dst.get_blob_client(d)
            if dc.exists() and dc.get_blob_properties().metadata.get("src_size") == size:
                print(f"  = {d}: already published"); continue
            if a.dry_run:
                print(f"  {s}  ->  {DST}/{d}"); sent += 1; continue
            i, o = Path(tmp, "in.mp4"), Path(tmp, "out.mp4")
            i.write_bytes(sc.download_blob().readall())
            if need_width and width(i) != need_width:
                print(f"  - {d}: not three panels yet ({width(i)} px), skipped"); skipped += 1; continue
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(i), "-c", "copy", "-movflags", "+faststart", str(o)],
                           check=True)
            with open(o, "rb") as fh:
                dc.upload_blob(fh, overwrite=True, metadata={"src_size": size},
                               content_settings=ContentSettings(content_type="video/mp4"))
            print(f"  ✓ {d}"); sent += 1
    doc = prompts_doc(src)
    what = f"v10/prompts.json: {len(doc['skits'])} skits, {sum(len(s['shots']) for s in doc['skits'].values())} shots"
    if a.dry_run:
        print(f"  {what} (dry run)")
    else:
        dst.get_blob_client("v10/prompts.json").upload_blob(
            json.dumps(doc, indent=1, ensure_ascii=False), overwrite=True,
            content_settings=ContentSettings(content_type="application/json"))
        print(f"  ✓ {what}")
    print(f"{len(ITEMS)} videos: {sent} {'to publish (dry run)' if a.dry_run else 'published'}, {skipped} skipped "
          f"-> {DST}/v10/")


if __name__ == "__main__":
    main()
