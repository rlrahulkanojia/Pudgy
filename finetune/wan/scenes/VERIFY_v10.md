# v10 verification: an independent second look at every span that can train

In v9b, about **1 kept span in 4 needed a change** after curation:
- camera moves inside a span;
- lead depictions;
- beats that said eyes were closed when they were open;
- missed jumps;
- a missed `size_change`.

So for v10, every span that can reach training gets checked by someone who did not curate it:
- every **kept** span;
- every **filled** dropped span (`camera_move` or `lookalike` with beats), since a U3/U5 decision
  can flip those.

You did not curate these skits. Assume nothing in the JSON is right until you see it in the
frames. The rules are the curator's: [`CURATION_v9b.md`](CURATION_v9b.md) and
[`CURATION_v10.md`](CURATION_v10.md). Read both first.

## Per span

1. **Boundaries.** Look at step-1 strips over the span's first and last 6 frames (and the frames just
   outside it). A cut, dissolve or camera move must not leak in.
2. **Camera.** Run `motion` on a background band over the whole span, against its first frame,
   `--step 2` (trust cc ≥ 0.95). Then `compare` its first vs last frame. Shakes, eased push-ins
   that snap back, and drifts near a cut have all been missed before.
   - a kept span must be truly locked;
   - a filled `camera_move` span must name the right move.
3. **Picture.** Check every 4th frame of the span for:
   - a meme caption or watermark (top third, bottom fifth);
   - any graphic over the picture;
   - a lookalike or depiction of a lead (photos, prints, plushes, screens, charms);
   - a lead visible for most of it.
4. **Beats.** Each beat must be true for its own frame range: who acts, the action, eyes
   open or closed, mouth, which fin, props, direction. A beat must not claim something that
   happens outside its range, and nothing important may be missing. Also check the v9b word
   rules: no "blue" or "pink", no camera words, no transcribed text.
5. **Fields.**
   - `framing` (by the lead's height in the picture) and `angle`;
   - `screen` side at the first frame, `moves`, `presence` and `enters_at`/`exits_at`;
   - `size_change` (does anyone walk toward or away from the viewer?) and `travel`;
   - `time`, `props`, `effects`, `in_world_text`;
   - for the iFin, `is_ifin`.
6. **Caption.** Compile it with `$PY captions_v9b.py --include-dropped <file>` and read it
   against the frames once more. It must be true for the whole span.

## Verdicts

- **`ok`**: everything holds.
- **`fix`**: the span stays, with changes. Give replacement `beats` (the full list), and/or
  any of these field changes: `size_change`, `travel`, `props`, `characters`, `framing`,
  `angle`, `time`, `view`, `camera`, `effects`, `prop_events`, `boundary`. To trim the span,
  use `new_start`/`new_end` plus `new_dropped_spans` for the frames cut off. Spans must still
  tile the skit.
- **`drop`**: the span must not train. Give `drop_reason` (and `camera` if it moves) and
  `notes`.
- **`keep`**: a **dropped** span that a current ruling makes keepable, such as a full-picture
  dream, a loose figurine, or a "depiction" that turns out to be no lead. It becomes a kept
  span. The `fix` must make it complete: every kept-span field it lacks (`framing`, `angle`,
  `setting` (an id already in the file's `settings`), `characters`, `beats`, `size_change`,
  `travel`, `time`, `view`, `props`, `effects`, `in_world_text`, `overlays: []`), and
  `camera: "static"`. Only use it on a span you have checked like a kept one.
  If it needs a setting, cast member or outfit the file lacks, add `add_settings` /
  `add_cast` / `add_outfits` (id → entry, new ids only) to the `fix`.

## Output

Write one JSON list to `$V9B_WORK/verify_v10_report_<your id>.json`. This is the format
`apply_verify.py` reads:

```json
[{"out": "curation/iteration_7__PP-Gasp-Base.json", "start": 40, "end": 112, "verdict": "fix",
  "issues": ["beat 2: Polly's eyes are open from f88", "size_change: Pax walks toward the viewer f95-112"],
  "fix": {"size_change": true, "beats": [ ...the full corrected list... ]}}]
```

- Write one entry per span you were given, `ok` ones included.
- Do **not** edit the curation files. The fixes are applied centrally with
  `apply_verify.py`, then validated.
- Write nothing else, anywhere.

## Environment

The same as the curators' (`CURATOR_TASK_v10.md` in the session scratchpad):
`PUDGY_DATA`, `V9B_WORK`, `PY`, run from `finetune/wan/scenes`, quote skit paths.
