"""PDF of the "Pudgy skit remakes - bedroom set" doc (Claude Docs, rev 7), rebuilt from the same sources the
doc was written from: qa/skit_report.json (prompts, seeds, links) + the details and observations below.
Rendered with fpdf2 (isolated venv /workspace/tools/pdfvenv), DejaVu fonts for the Unicode.

/workspace/tools/pdfvenv/bin/python make_doc_pdf.py   -> docs/pudgy_skit_remakes_bedroom.pdf (+ .md)
"""
import json
from pathlib import Path

import markdown
from fpdf import FPDF

ROOT = Path("/workspace/longform_client")
OUT = ROOT / "docs"
STYLE = "A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing "

LEAD = ("Four client skits from iteration 2 (Little Spoon, 4AM Show, Morning Fart, Wake Up Early) were remade scene by "
        "scene with the v8 model at 480×848. All four keep the original's cuts, timing and soundtrack. 14 of 17 parts "
        "passed every check on the first seed, 2 passed on a later seed, and 1 is kept as the best of three with a "
        "small known flaw.")

HOW = """## How they were made

Every scene is regenerated between the skit's own frames, so the remakes follow the original's action, cuts and soundtrack. These settings apply to all four skits.

| Setting | Value |
| --- | --- |
| Source | Client skits, Azure raw/iteration_2/72_videos; scene notes from the October skit curation |
| Model | Wan2.2-I2V-A14B with the v8 LoRAs (v8 merged on the low-noise expert, v2 golden on the high-noise expert) |
| Sampling | 25 steps, guidance 5.0 low-noise / 3.5 high-noise, flow shift 5, fp8 weights |
| Size | Generated and delivered at 480×848, 24 fps, not upscaled |
| Anchoring | Each scene split into parts of up to 3.4 s; every part starts and ends on the original's frames (first + last frame) |
| Colour | First+last-frame generation shifts the palette; each part is colour-matched back to the original's frames |
| Assembly | Cut at the original's cut points, same frame count, original soundtrack added |
| Checks per part | No extra Pax or Polly; no sudden jump above 5× the median frame change (or 1.5× the original scene's own biggest); ends within 12 levels of the original's frame |
| Retries | A failed part is remade with seed 1042, then 2042; if all three fail, the smoothest is kept |

Every prompt follows one pattern: "A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing " + the characters + the scene's actions joined by "then" + the framing, camera and set. The prompts below start after that shared opening.

Each skit comes as three files: with the original soundtrack, silent, and side by side (original left, remake right). Links are read-only and valid until 5 Nov 2026.
"""

SKITS = {
    "little_spoon": dict(
        title="Little Spoon",
        lead="The closest of the four: every beat in order, all 5 parts passed on the first seed.",
        details=[("Original", "PP-LittleSpoon-Base.mp4, 8.2 s (196 frames)"),
                 ("Scenes / parts", "4 scenes, 5 parts, all still camera, close-ups"),
                 ("Props", "None (bed, pillow, blanket)"),
                 ("Seeds", "All parts seed 42, no retries"),
                 ("GPU time", "About 32 min, including model load")],
        obs=["Same close-ups, bedroom, blanket and lamp; beats in order: Polly asks to cuddle, Pax pops up excited, "
             "Polly snuggles and rolls over, looks back puzzled, Pax turns away wiggling his feet.",
             "Polly's roll-over and look-back land a fraction of a second earlier than in the original.",
             "Untrained moments (lying in bed, rolling over) came out naturally, because every part is pinned to the "
             "original's frames.",
             "Softer than the client's 1080×1920 original, as expected at 480×848."]),
    "four_am_show": dict(
        title="4AM Show",
        lead="Close to the original, push-in camera moves included; Polly's biggest singing moment is a little smaller.",
        details=[("Original", "PP-4amshow-Base.mp4, 9.0 s (215 frames)"),
                 ("Scenes / parts", "3 scenes, 4 parts, every scene a slow camera push-in"),
                 ("Props", "1: a microphone (plus a laptop on the floor as set dressing)"),
                 ("Seeds", "Parts 1–3 seed 42; part 4 seed 1042 (seed 42 had a sudden jump of 18.7× against the "
                           "original's 4.6×)"),
                 ("GPU time", "About 45 min, including one retry")],
        obs=["Beats match: drowsy Pax nods along in bed, Polly sings into the microphone with the laptop on the floor, "
             "Pax dozes off and jolts awake; the slow push-ins are kept.",
             "Polly's most dramatic moment (flipper flung wide, beak stretched open) is smaller than in the original.",
             "In part 1 the model darkened the whole frame for a moment (a palette shift); the colour match corrected "
             "it, so brightness stays steady like the original.",
             "Untrained moments: singing into a microphone, lying back in bed; both read clearly."]),
    "morning_fart": dict(
        title="Morning Fart",
        lead="The story and the soundtrack gag land on cue; the yawn runs later and quicker than the original's.",
        details=[("Original", "PP-MorningFart-Base.mp4, 7.8 s (186 frames)"),
                 ("Scenes / parts", "3 scenes, 4 parts; still camera except a 0.9 s pan to Pax"),
                 ("Props", "None (bed, duvet, pillows)"),
                 ("Seeds", "Part 1 seed 2042 (seeds 42 and 1042 had jumps of 5.4× and 7.7× against a bar of 5.0×; "
                           "2042 passed at 4.9×); parts 2–4 seed 42"),
                 ("GPU time", "About 45 min, including model load and two retries"),
                 ("Scene note", "The pan (part 3) had no curation notes; its prompt was written from the skit's frames")],
        obs=["Same bedroom and framing; Polly wakes, yawns and stretches, rubs her eye, turns toward Pax, stares "
             "startled, rolls away, peeks and settles.",
             "The yawn starts about 0.5 s later than in the original and is quicker; this is the part that needed "
             "three seeds.",
             "In the short pan, Polly faces the camera instead of turning toward Pax.",
             "At the startled moment she lifts up a little more than in the original.",
             "Untrained moments: lying in bed, rolling over, the rippling duvet; all read clearly."]),
    "wake_up_early": dict(
        title="Wake Up Early",
        lead="Faithful to the original; the opening push-in is kept as the best of three seeds, with a faint "
             "background patch in its first 0.6 s.",
        details=[("Original", "PP-WakeUpEarly-Base.mp4, 6.6 s (159 frames)"),
                 ("Scenes / parts", "4 scenes, 4 parts; opens with a slow push-in, then still camera"),
                 ("Props", "1: an alarm clock on the nightstand"),
                 ("Seeds", "Part 1 seed 1042, best of three (42: 6.5×, 1042: 5.0×, 2042: 14.3× jump, against the "
                           "original's 1.3× and a bar of 5.0×); parts 2–4 seed 42"),
                 ("GPU time", "About 41 min, including two retries"),
                 ("Scene notes", "Parts 1 and 3 had no curation notes; their prompts were written from the skit's "
                                 "frames")],
        obs=["Beats match: both asleep, the alarm jiggles, they pop up to sit, Pax leans toward the alarm, they sink "
             "back under the blanket, then sleep sprawled out, snoring.",
             "Opening push-in: no seed matched the original's very smooth motion. In the kept seed a faint lighter "
             "patch shows on the back wall for the first 0.6 s, then fades; the characters themselves are fine.",
             "The alarm clock's digits are not stable; they change from frame to frame.",
             "Untrained moments: sleeping, sitting up in bed, reaching for the alarm, pulling up the blanket; all read "
             "clearly."]),
}


def skit_md(name, rep):
    s, r = SKITS[name], rep[name]
    links = r["links"]
    out = [f"## {s['title']}", "", s["lead"], "",
           "**Azure links:** " + " · ".join(f"[{label}]({links[key]})" for label, key in (
               ("with original soundtrack", "with original soundtrack"), ("silent", "silent"),
               ("side by side", "side by side (original | remake)"))), "",
           "**Details**", "", "| Field | Value |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in s["details"]]
    out += ["", "**Prompt**", "", "| Part | Time | Prompt |", "| --- | --- | --- |"]
    for i, p in enumerate(r["parts"], 1):
        a, b = (int(x) for x in p["frames"].split("-"))
        out.append(f"| {i} | {a / 24:.2f}–{(b + 1) / 24:.2f} s | {p['prompt']} |")
    out += ["", "**Observation**", ""] + [f"- {o}" for o in s["obs"]] + [""]
    return "\n".join(out)


def main():
    rep = json.loads((ROOT / "qa/skit_report.json").read_text())
    md = "\n".join([f"# Pudgy skit remakes — bedroom set", "", "30 Oct 2026 · Asynq", "", LEAD, "", HOW] +
                   [skit_md(n, rep) for n in SKITS])
    OUT.mkdir(exist_ok=True)
    (OUT / "pudgy_skit_remakes_bedroom.md").write_text(md)
    html = markdown.markdown(md, extensions=["tables"])
    # fpdf2's HTML tables: give the prompt tables a narrow Part/Time column and a wide Prompt column
    # (every column needs a width once one has)
    html = html.replace("<th>Part</th>", '<th width="7%">Part</th>').replace("<th>Time</th>", '<th width="15%">Time</th>')
    html = html.replace("<th>Prompt</th>", '<th width="78%">Prompt</th>')
    html = html.replace("<th>Setting</th>", '<th width="22%">Setting</th>').replace("<th>Field</th>", '<th width="22%">Field</th>')
    html = html.replace("<th>Value</th>", '<th width="78%">Value</th>')

    pdf = FPDF(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, 15)
    d = "/usr/share/fonts/truetype/dejavu/"
    pdf.add_font("DejaVu", "", d + "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", d + "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", d + "DejaVuSans-Oblique.ttf")
    pdf.add_font("DejaVu", "BI", d + "DejaVuSans-BoldOblique.ttf")
    pdf.add_page()
    pdf.set_font("DejaVu", size=9)
    pdf.write_html(html, font_family="DejaVu", table_line_separators=True)
    dst = OUT / "pudgy_skit_remakes_bedroom.pdf"
    pdf.output(str(dst))
    print(dst, dst.stat().st_size, "bytes,", pdf.pages_count, "pages")


if __name__ == "__main__":
    main()
