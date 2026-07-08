# Skit → Shot Prompt-Compiler — Internal Spec

**Purpose:** let the client's skit artists author in their native screenplay style, and automatically emit the machine-facing prompts the CogVideoX1.5-I2V engine consumes. Bridges the granularity + register gap between how they write and what the model reads.

**Where it sits:** authoring layer *above* the inference pipeline in `Training_Runbook.md` Part D4. Compiler output feeds the per-shot first-frame + per-clip generation prompt.

---

## The gap it closes
| | Artist's input | Model's input |
|---|---|---|
| Unit | Skit (multi-shot, 7–14 s) | One atomic clip (3–5 s = 49/81 frames) |
| Register | Screenplay beats, intent, staging, comedy | Dense literal caption, one action |
| Identity | "Pax", "Polly" by name | Full appearance description (anchor) |
| Timing | "then", "beat", "REALLY HARD" | Frame count only |

The compiler translates left → right **without** the artist writing model captions.

---

## Inputs
1. **Skit script** — screenplay-style prose (like `Data/client/SampleFlow/PP_Survival.pdf`): scene headers (INT./EXT. + location), character beats, action, comedy.
2. **Asset registry** — Pax/Polly identity anchors + the `Prp_*` artifact library (names → canonical descriptions), from the style guide.
3. **Style corpus** — 5–10 of the client's real skit scripts + matching finals + terminology glossary (used to tune the compiler to their voice; see §5).

## Output (structured, per shot)
For each shot the compiler emits a record:
```
shot_id
scene: {location, background_plate_id}
characters: [{name, side: left|right|center}]     # spatial, per shot
first_frame_spec:                                  # what the artist composes / we assemble
  layout_notes, prop_ids, staging (from guidelines: penguins small, 3/4 angle…)
clip(s): [                                          # one atomic action each
  {
    action_label,
    motion_prompt,        # dense NL, TRAINING-CAPTION STYLE, no appearance words
    prop_motion: [{prop_id, moving: bool, motion_desc?}],
    frames: 49 | 81,      # → duration; NOT a text token
  }
]
artist_flags: [ camera_move?, precise_timing_beat?, occlusion? ]   # things text can't do → route to artist
```
The **identity anchor is injected at generation time**, not written by the artist. `motion_prompt` must match the exact caption template the model was trained on (`Training_Runbook.md` Part C).

---

## Translation stages
1. **Segment** the skit into **shots** (scene header / camera change) → then into **atomic clips** (split "waddle → stop → throw flipper" into separate clips, one action each).
2. **Resolve entities** — map names → identity anchors; map mentioned objects → `Prp_*` ids; flag any prop not in the registry.
3. **Assign spatial sides per shot** from the beat text (Pax/Polly left/right/center) — read per shot, never hardcoded.
4. **Write the motion prompt** for each clip: dense natural language, dynamics-only, in the client's vocabulary. Lock the background static; describe prop motion only where the prop actually moves.
5. **Map timing → frames**: each atomic clip → 49 (~3 s) or 81 (~5 s). Multi-beat sequences become **multiple clips to stitch**, not one long prompt. Sub-clip beat timing ("then… REALLY HARD") → `artist_flags.precise_timing_beat` (handled by start/end poses or edit, not text).
6. **Flag the uncontrollables** — camera moves, exact comedic timing, heavy occlusion → `artist_flags` for the artist to handle via first-frame composition / key poses / the cut.

---

## Adapting to the client's style (the core requirement)
Two levers, both seeded from the §1.3 style corpus:
1. **Vocabulary alignment** — the *training captions* are written using the client's own terms (waddle, 2-shot, reverse angle, squash-and-stretch, beat, POV). So prompts phrased in their voice land on what the model learned. The compiler draws motion wording from the same glossary.
2. **Few-shot compiler tuning** — the compiler (LLM-based) is prompted/few-shot-conditioned on **paired examples**: client skit excerpt → the shot/clip breakdown we authored. This teaches it to read their register (screenplay beats, their shorthand) and produce faithful clip prompts. Refresh the examples as more scripts arrive.

**Hard constraint:** compiler output vocabulary/structure must equal the training-caption distribution. If we caption training data one way and the compiler emits another, adherence drops. Caption template and compiler output template are the **same artifact** — define once.

---

## Implementation
- **LLM prompt-compiler** (Claude) with: system prompt = the caption template + client glossary + staging rules; few-shot = paired skit→breakdown examples; tools/schema = the structured output above (validate against a JSON schema).
- **Deterministic post-checks:** every clip resolves to a known character/prop id; `frames ∈ {49,81}`; background lock present; unknown props raised, not silently dropped.
- **Human-in-the-loop:** artist reviews/edits the breakdown before generation (it mirrors their animatic/shot-planning step, so it's natural).

## Dependencies / open items
- Style guide (identity anchors, `Prp_*` descriptions) — pending from client.
- 5–10 skit scripts + finals + glossary — pending (we have "Survival").
- Lock the shared caption/prompt template in `Training_Runbook.md` Part C **before** captioning, so training data and compiler output match.
- Confirm the artifact registry naming from the delivered asset sheets.
