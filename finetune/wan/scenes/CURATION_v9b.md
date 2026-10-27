# v9b curation: brief and schema

The curator pass turns the 99 client skits into **spans**: frame-accurate pieces of footage,
each kept or dropped with a reason. Every kept span gets a description of what happens in it,
with frame ranges. `prep_v9b.py` cuts kept spans into training windows (f21 / f33 / f49 /
f81), and `captions_v9b.py` compiles each window's caption from this file and the asset
registry. Plan: [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md)
§0.1 (decisions D1–D10) and §3.2–§3.4. The audit behind it:
[`SKIT_AUDIT_v9b.md`](../../../docs/training_reports/v9/SKIT_AUDIT_v9b.md).

A wrong kept span teaches the model something false, for every window cut from it. When in
doubt, drop, and say why in `notes`.

## Inputs

| what | where |
|---|---|
| skits | `/workspace/Data/raw/<skit>` (paths in `scenes/skits_v9b.json`) |
| storyboard text per skit | `scenes/storyboards.json` (key = skit path). Numbered beats; story intent, not what is drawn |
| v4 curation (iteration_2 only) | `/workspace/Data/processed/v9b_work/v4_curator/<name>.json`: cut times at 0.1 s, keep/drop, `action_beats`, `environment`, `camera`, `other_characters`. **A seed, not the truth**: v4 kept zooms, pans and overlays, and merged short scenes across cuts. Tokens `pxngn0` = Pax, `plngn0` = Polly |
| first-pass shot verdicts | `scenes/audit/shot_review.json` (shots ≥ 1 s; `verdict`, `reason`, `inside_shot`, `notes`). Also a seed, with known misses |
| tools | `scenes/curate_tools.py` (below), run with `/workspace/Pudgy/.venv-wan/bin/python` |
| validator | `scenes/curate_check.py` |
| a finished example | `scenes/curation/iteration_2__PP-Chair-Base.json` |

## Tools

Run from `/workspace/Pudgy/finetune/wan/scenes` with `PY=/workspace/Pudgy/.venv-wan/bin/python`.
Images land in `/workspace/Data/processed/v9b_work/<skit stem>/`; open them with the Read tool.

- `$PY curate_tools.py profile SKIT`: hard-cut candidates (the frame printed is the **first
  frame of the new shot**), gradual-change candidates (dissolves, fades, camera moves, big
  motion), very dark/bright frames, and per candidate shot a camera hint from **keep12**:
  the share of edges still in place 12 frames later. Locked shots measure median ≥ 0.8. A
  low value means *look*: it is either a camera move or a character filling the frame
  while moving.
- `$PY curate_tools.py strip SKIT A B --step S [--width 150 --cols 10]`: frames A..B-1 every
  S frames, each labelled with its frame number. Use step 4–6 for an overview and **step 1
  around every boundary** to place it exactly.
- `$PY curate_tools.py compare SKIT F1 F2`: two frames side by side, plus an edge overlay
  (black = edge in both, red = only F1, green = only F2) and a difference map. With a
  locked camera the background's edges are black: they overlap exactly. Doubled red/green
  furniture edges mean the camera moved. Use it on every kept span (its first vs last
  frame), and wherever keep12 says *look*.

Frame numbers are 0-based, in decode order (every real frame, nothing inserted). `frames`
in your file must equal the skit's frame count (`profile` prints it).

## Procedure, per skit

1. Read the storyboard text, the v4 JSON (iteration_2) and the skit's rows in
   `shot_review.json`.
2. `profile` the skit. Strip the whole skit at step 4–6 to see it.
3. Place every boundary at frame accuracy with step-1 strips: hard cuts, the first and last
   frame of every dissolve, fade or wipe, and where a camera move starts and stops.
4. Split into spans that **tile every frame** (0 to `frames`, no gaps, no overlaps):
   - a transition (dissolve, fade, wipe) is its own dropped span (`transition`);
   - a shot whose camera moves for part of it becomes static span(s) + moving span(s)
     (`boundary: "camera"`), and the moving parts are dropped (`camera_move`);
   - anything shorter than 21 frames is dropped (`too_short`).
5. For each span, decide keep or drop (rules below). Check every kept span with `compare`
   (its first vs last frame), and look at its frames: a locked camera, no overlay anywhere
   in it, no lookalike, the lead visible.
6. For kept spans, write the fields below, especially the **beats**.
7. Run `$PY curate_check.py curation/<your file>.json`. Fix every ✗ error. Read every !
   warning and fix it or confirm it is fine.

## Keep / drop rules

**Keep** only spans where all of these hold:
- the **camera is locked** (no pan, tilt, zoom, push, pull, shake). Characters may move freely;
- at least one lead (Pax or Polly) is visible for most of the span;
- nothing is drawn **over** the picture: no captions, title cards, chat or speech bubbles
  with text, arrows, meters, UI overlays, logos, end cards;
- no **depiction** of a lead anywhere in the picture: photos, posters, prints or faces on mugs and
  headbands, plushes, charms or figurines in a lead's look, a screen showing a lead (rule settled on
  2026-10-26 across all curators: recognisable as Pax or Polly ⇒ drop as `lead_copy`, whatever the size;
  a tiny generic penguin charm that is neither lead is a prop);
- no **lookalike** of a lead (a character drawn like Pax or Polly with a small difference,
  e.g. Polly's mum = Polly + a pearl necklace in MothersDay2), and no **copy** of a lead
  (plush toy, poster, photo, drawing, dream or thought bubble showing them);
- on style (the normal Pudgy 2D look), and no special effect that changes the whole picture
  (time-lapse, day/night flashing, fly-ins through light bursts, flashes, vignettes, colour
  washes, blur transitions);
- at least 21 frames long.

**Drop reasons** (`drop_reason`): `transition`, `camera_move`, `graphic_overlay`, `ad_merch`,
`no_lead` (inserts: phone or map screens, empty rooms, objects), `lead_hidden` (lead under
a blanket, behind an object, only a sliver visible), `lookalike`, `lead_copy`, `off_style`
(isometric cut-aways, underwater fantasy, a different art style), `effect` (as listed
above), `too_short`, `other` (explain in `notes`).

**Allowed, and described in the beats:** small in-style cartoon effects that belong to the
characters: hearts, sparkles, sweat drops, tears, anger marks, "zzz", motion lines, a
question or exclamation mark, musical notes, blush marks. In-world text (shop signs,
packaging, phone or clock screens in the scene) is fine but **never transcribed** in a beat.

**Side characters** (anyone who is not Pax or Polly, and not a lookalike): keep, add to
`cast` as an `extra` with a short visual description ("a grey penguin in sunglasses",
"a crowd of dark penguin silhouettes"), and name them in beats as `{x1}`, `{x2}`. If a
side character could be mistaken for a lead, it is a lookalike: drop.

**Outfits on a lead** (hat, scarf, headband, tie, backpack, shower cap, hood, beard, mask, …):
add to `outfits` as a noun phrase ("a yellow hard hat") and list its id on the character in
every span where it is worn. A lead in an outfit that hides who they are is
`lead_hidden`.

**The iFin phone** (the client's phone prop: pale grey back, three camera lenses, a little
tuft on top): write `{ifin}` in a beat where it is clearly the iFin; the compiler inserts
the registry phrase. Any other phone, or one you can't identify, is written "a phone".
Record phone moments in `prop_events` with `is_ifin` true / false / "unsure".

## Beats: how to write them

Beats say **what is visible**, in order, with frame ranges. They become the middle of every
caption cut from the span, so they must be true for any window that shows them.

- **One action per beat.** "Pax waddles in from the right" and "Pax drops the bags" are two
  beats. Actions of different characters at the same time are one beat joined by "while".
- Beats run from the span's first frame to its last, in order, without gaps: the first
  beat starts at `start`, the last ends at `end`, and a gap between beats is at most 6
  frames. Short holds count ("Polly stands still, smiling").
- Start each beat with who acts, by name: "Pax …", "Polly …", "{x1} …". Pronouns ("he",
  "she", "his", "her") are fine inside a beat once the name is given.
- Describe motion and visible expression, not story or intent: "Pax flops back on the bench
  with his eyes closed and a blissful smile", not "Pax is exhausted after five hours of
  shopping".
- **Never** use the words "blue" or "pink" (they are the leads' identity anchors; use
  "teal", "navy", "rose", "coral", or no colour). No camera words ("camera", "zoom",
  "close-up", "panning"): the compiler adds framing and camera. No quotes, and no
  transcribed text or dialogue.
- Don't copy v8's caption clauses (e.g. "sitting on the ground with its legs stretched
  out in front"): describe the skit's own action in your own words. The validator rejects
  any 6-word run from a v8 clause (decision D10).
- Present tense, plain words, 6–25 words per beat.

## Schema

One file per skit: `scenes/curation/<iteration>__<file stem, spaces → _>.json` (the `out`
field in `skits_v9b.json`).

```json
{
 "skit": "iteration_2/72_videos/PP-Chair-Base.mp4",
 "frames": 223,                         // the skit's frame count
 "fps": 24,
 "curated_by": "curator-NN",
 "curated_at": "2026-10-26",
 "storyboard_title": "CHAIR",
 "cast": {                              // everyone who appears in a kept span
  "pax": {"kind": "lead"}, "polly": {"kind": "lead"},
  "x1": {"kind": "extra", "desc": "a grey penguin in sunglasses"}
  // a lookalike, if listed at all: {"kind": "extra", "desc": "...", "lookalike": true}
 },
 "outfits": {"o1": "a yellow hard hat"},
 "settings": {                          // noun phrases, used as "in <desc>"
  "s1": {"kind": "flat", "desc": "a clothes shop drawn on a plain pale grey background"},
  "s2": {"kind": "set",  "desc": "a snowy village street with round igloo houses"}
 },
 "spans": [ {
   "start": 0, "end": 86,              // [start, end), frames
   "boundary": "start",                 // start | cut | transition | after_transition | camera | same_shot
   "keep": true,
   "drop_reason": "",                   // required when keep is false
   "camera": "static",                  // static | pan_left | pan_right | tilt_up | tilt_down | push_in | pull_back | zoom_in | zoom_out | shake | rotate | whip | other
   "framing": "wide",                   // extreme close-up | close-up | medium close-up | medium | medium wide | wide
   "angle": "eye level",                // eye level | high angle | low angle | overhead
   "setting": "s1",
   "view": "with racks of hanging clothes behind them",   // optional; starts with "with "
   "time": "day",                       // day | morning | evening | night | sunset | ... (non-day is added to the caption)
   "size_change": false,                // true if a character walks toward/away from the viewer and grows/shrinks
   "characters": [
    {"id": "pax", "screen": "left", "presence": "full", "outfit": []},
    {"id": "polly", "screen": "right", "presence": "enters", "enters_at": 30, "outfit": ["o1"],
     "moves": [{"frame": 60, "screen": "center"}]}
   ],
   "beats": [ {"start": 0, "end": 62, "text": "Polly flips through the clothes on the rack while Pax holds the shopping bags"}, ... ],
   "props": ["shopping bags"],          // notable objects in view or handled
   "prop_events": [{"prop": "phone", "event": "takes_out", "frame": 40, "by": "polly", "is_ifin": true}],
   "effects": ["hearts"],               // in-style effects present (also in the beats)
   "overlays": [],                      // must be empty for a kept span
   "in_world_text": ["shop sign"],      // what kind, never the words
   "travel": "none",                    // none | enters | exits | crosses | moves_within
   "notes": ""
 } ]
}
```

Field rules (the validator enforces them):
- `screen`: left | left-center | center | right-center | right, where the character is at
  the span's first frame (or where they come in, for `enters`). Add a `moves` entry when a
  character ends up on a different side for a second or more.
- `presence`: full | partial (part of the body is outside the picture the whole time) |
  enters (`enters_at`) | exits (`exits_at`) | enters_exits (both).
- `framing` by the lead's visible height in the picture: extreme close-up (face only),
  close-up (head and chest, more than ~60% of the height), medium close-up (to the belly),
  medium (whole body, ~35–60%), medium wide (~20–35%), wide (under ~20%).
- `angle`: eye level unless the view clearly looks down (high angle, overhead) or up (low
  angle) at the characters.
- `view`: what is in the picture beyond the setting, as "with …". Leave it empty for plain
  close-ups.

## What prep does with your file (for orientation)

- Kept spans are cut into non-overlapping windows of 81, 49, 33 and 21 frames from the
  span's start. Each window's caption takes the beats it shows (a beat counts if the window
  holds 3/4 of a short beat, all of one under 5 frames, at most 12 frames of a long one).
- A window in which no lead is visible for most of its frames is skipped.
- Whole skits are held out for evaluation; which ones is decided after curation.
