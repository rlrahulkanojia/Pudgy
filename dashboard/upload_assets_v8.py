#!/usr/bin/env python3
"""Publish the v8 showcase to the dashboard container (`AZURE_DASHBOARD_CONTAINER`).

Source of truth is the training container (`pudgy`), so this runs from any machine with the
.env — no GPU box needed. Each clip is remuxed with +faststart (moov atom first) so the
browser can start playback before the whole file arrives, and renamed so the dashboard
caption reads as a label ("Polly blushing"), not a file path.

Groups (manifest.py `video_groups` for v8, displayed with "_" -> " "):
  Long-form_scenes       the two storyboard remakes, with the original soundtrack
  Expressions            8 expressions x Pax/Polly, front view (core suite, step 550)
  Movements              2-second loops of walking/waving/jumping/sitting (duration suite)
  Motion_and_expression  6 compositions where both the action and the expression read
  Interaction            the 4 two-character tests (untrained - base-model ability)
Every clip here passed the QA used for the long-form work, including the one-Pax/one-Polly
duplicate check.

    python dashboard/upload_assets_v8.py [--dry-run]
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
SRC, DST = "pudgy", os.environ.get("AZURE_DASHBOARD_CONTAINER", "pudgy-dashboard")
CORE, GD = "v8/eval/suites/core/v8@550", "v8/eval/suites/gd/v8@550"
COMP, INTER = "v8/eval/suites/compose/v8@550", "v8/eval/interaction/v8@550"

ITEMS = [("v8/longform/final/life_with_her_1080x1920_audio.mp4", "Long-form_scenes/Life_With_Her.mp4"),
         ("v8/longform/final/eating_stages_1080x1920_audio.mp4", "Long-form_scenes/Eating_Stages.mp4")]
for c in ("pax", "polly"):
    for lab in ("happy", "surprised", "angry", "confused", "crying", "neutral", "laughing", "blushing"):
        ITEMS.append((f"{CORE}/{c}_{lab}_FRONT_f21_s42.mp4", f"Expressions/{c.capitalize()}_{lab}.mp4"))
    for lab in ("walking", "waving", "jumping", "sitting"):
        ITEMS.append((f"{GD}/{c}_{lab}_f49_s42.mp4", f"Movements/{c.capitalize()}_{lab}_loop.mp4"))
for src, name in [("pax_jumping+laughing", "Pax_jumping_while_laughing"), ("pax_waving+laughing", "Pax_waving_while_laughing"),
                  ("pax_walking+laughing", "Pax_walking_while_laughing"), ("polly_jumping+laughing", "Polly_jumping_while_laughing"),
                  ("polly_waving+laughing", "Polly_waving_while_laughing"), ("polly_waving+crying", "Polly_waving_then_crying")]:
    ITEMS.append((f"{COMP}/{src}_f25_s42.mp4", f"Motion_and_expression/{name}.mp4"))
for src, name in [("1_wave_from_wavingFRONT_f25_s42", "1_Waving_together"), ("2_hug_from_huggingQF1L_f21_s42", "2_Hug"),
                  ("3_hug_from_wavingFRONT_f25_s42", "3_From_waving_to_a_hug"),
                  ("4_wave_from_composite_f25_s42", "4_Waving_from_two_separate_frames")]:
    ITEMS.append((f"{INTER}/{src}.mp4", f"Interaction/{name}.mp4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    svc = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    src, dst = svc.get_container_client(SRC), svc.get_container_client(DST)
    with tempfile.TemporaryDirectory() as tmp:
        for s, d in ITEMS:
            d = f"v8/{d}"
            if a.dry_run:
                print(f"  {s}  ->  {DST}/{d}"); continue
            i, o = Path(tmp, "in.mp4"), Path(tmp, "out.mp4")
            i.write_bytes(src.download_blob(s).readall())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(i), "-c", "copy", "-movflags", "+faststart", str(o)],
                           check=True)
            with open(o, "rb") as fh:
                dst.upload_blob(d, fh, overwrite=True, content_settings=ContentSettings(content_type="video/mp4"))
            print(f"  ✓ {d}")
    print(f"{len(ITEMS)} videos {'listed' if a.dry_run else 'published'} to {DST}/v8/")


if __name__ == "__main__":
    main()
