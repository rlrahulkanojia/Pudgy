# v10 curation: the iteration_7 skit bases

The schema, keep/drop rules and beat rules are **v9b's, unchanged**:
[`CURATION_v9b.md`](CURATION_v9b.md). Read it in full first. This file lists only what is
different for the 186 iteration_7 files (`skits_v10.json`, batches in
`curation_batches_v10.json`). Captions compile with `captions_v9b.py` as before
([`Training_Approach_v10.md`](../../../training_approach/v10/Training_Approach_v10.md) §3.5).

## The client's answers (2026-11-05)

These settle the open questions and override the rulings below where they differ.

| Question | Answer | What it did |
|---|---|---|
| Yellow feet and beak (U9) | Older design, no longer acceptable | Stays `off_style` with an `OLD_DESIGN:` note |
| Pearl necklace (Purse, PoopBreak, DrivingPartner, CancelPlans) | Polly dressed up | Purse and CancelPlans flipped to kept; PoopBreak `[0,34)` filled as a pull-back; DrivingPartner stays out as a vehicle-mounted shot |
| Bunny costume (Easter) | A one-off, remove it | Back to an unfilled `lookalike` |
| Stubble and goggles (Fixer) | **Not answered** | Still a filled `lookalike`, held and not trainable |
| Blue penguin in a red scarf and navy overalls | Remove | No trainable span had it. Bright, Cooking, Cute and Bromance never flip |
| Yellow penguin in earmuffs, old penguin with a white beard | Not acceptable, they have no names | No trainable span had them |
| Black-and-white penguin | Acceptable | Stays an `extra` with that description |
| Plush in Unboxing | A Pudgy product | Stays `ad_merch` |
| Clean copies | Client is looking into them | Only `PP_HowTo_Animation_v5` is still worth having; the others carry a removed character or design |

The other unnamed side penguins (grey, brown, tan, mint and so on) were not asked about and stay
as `extra`s for now.

## Running locally (not on the GPU box)

```bash
cd finetune/wan/scenes
export PUDGY_DATA=~/Documents/Projects/Saksham/Pudgy/Data   # raw/iteration_7/293_videos/...
export V9B_WORK=<a scratch dir>                              # strips, profiles, compares
PY=<a python with opencv-python-headless, numpy, pillow>
$PY curate_tools.py profile "iteration_7/293_videos/PP-AngerSigns-Base.mp4"
$PY curate_check.py curation/iteration_7__PP-AngerSigns-Base.json
$PY captions_v9b.py curation/iteration_7__PP-AngerSigns-Base.json      # preview captions
```

Skit paths are relative to `$PUDGY_DATA/raw`. Quote them: several names have spaces.

## What is different from v9b

- **No storyboards, no v4 seed, no shot_review rows.** Work from the footage alone. The
  `storyboard_title` field is `""`.
- **`frames` and `fps` come from `skits_v10.json`.** Both are as measured, including the one
  29.97 fps file (`PP_CuteFit-BASE.mp4`). Use the file's own frame numbers.
- **Overlays do appear.** These are the client's "clean" bases, but some still carry a meme
  caption or the `@pudgypenguins` watermark. That is known for `PP_Cooking_9x16_base_v2`,
  `PP_Cute_9x16_base_v2`, `PP_Bright_9x16_base`, `PP_Hot_and_Cold_9x16_base` and
  `PP_HowTo_Animation_v5`, and there are likely others. Look at the top third and the bottom fifth of
  every kept span. A caption or watermark on any frame of a span → `graphic_overlay`. Put the
  kind in `overlays` (`"meme caption"`, `"watermark"`) and, in `notes`, the frame range where
  it shows.
- **Camera moves (U3: decided yes on 2026-11-04; filled camera-move spans train, with their move clause).** Drop them as in v9b (`keep: false`,
  `drop_reason: "camera_move"`). But when a moving span would otherwise be kept (at least 21
  frames, a lead visible, no overlay, on style), **also fill every kept-span field**: framing,
  angle, setting, view, time, size_change, characters, beats, props, effects, in_world_text,
  travel. Use the real move in `camera` (`push_in`, `pan_left`, …). If U3 says yes, these
  spans flip to kept with a camera tag and need no second pass.
- **New characters (decision U5 is open).** Side characters (a yellow penguin in earmuffs, a
  black-and-white penguin, a grandpa penguin with a beard, a penguin in a bunny hood, …) are
  `extra`s with a short visual description, as in v9b. **Use the same description for the
  same character in every span of a skit**, and prefer these wordings where they fit, so
  skits agree:
  - "a yellow penguin wearing earmuffs"
  - "a black-and-white penguin"
  - "an old penguin with a white beard"

  Never use the words "blue" or "pink" for an extra.
- **Pax or not?** A penguin drawn like Pax or Polly that you cannot place for sure
  (for example a blue penguin in a red scarf and overalls) is a `lookalike` drop, as v9b's
  rule says. Start its `notes` with `U5:` and a description, so these spans can be flipped
  once the client names the characters. A lead you *can* place, wearing clothes, is the lead
  with `outfits`.
- **Live action, logo animations, title cards, UI or screen-only inserts** are dropped with
  the v9b reasons (`off_style`, `graphic_overlay`, `no_lead`).
- **Rulings from the pilot (curator-v10-01, 2026-11-04):**
  - **A graphic drawn over the picture is `graphic_overlay`.** That covers a heart-rate (ECG)
    line with a heart icon, a meter, a progress bar or a chat UI. v9b's "meters, UI overlays"
    rule applies. Small effects attached to a character (hearts popping off them, sweat drops)
    stay allowed.
  - **Older Pudgy design** (yellow feet, paler rose and periwinkle bodies, glossy shaded
    furniture, often seen from high above): `off_style`, with `notes` starting `OLD_DESIGN:`.
    No fields needed. Several ProRes `*_9x16_base*.mov` files are in this style.
  - **Several reasons for one span:** use the most permanent one, in this order: `transition`,
    `too_short`, `off_style`, `no_lead`, `lead_hidden`, `lead_copy`, `ad_merch`, `effect`,
    `graphic_overlay`, `lookalike`, `camera_move`. So `camera_move` (U3) and `lookalike` (U5)
    are used only when they are the **only** reason. Those are the two cases that get every
    field filled.
  - **A run of dropped material with one reason** may be one span. Split it only at a hard cut
    where what follows could be kept.
  - **A lead on a screen or avatar** (a chat header, a photo, a video call) is `lead_copy`, as
    in v9b.
  - **Write nothing except your curation JSONs.** No agent memory file, no manifest, no other
    repo file, nothing under `Data/`.
- **Tools.** `curate_check.py` now also checks dropped spans that have beats, so filled
  camera-move and lookalike spans must be as valid as kept ones. `profile` can miss a cut
  between two shots on the same flat background, so confirm cuts on a strip. To tell a slow
  push-in from a character moving, run `$PY curate_tools.py motion SKIT A B --step 4 --band Y0 Y1`.
  It fits the camera on a background band against frame A.
  - **A locked camera** reads scale 1.0000 and 0 px at cc 1.000. Calibrated on Chair
    [168,223) with `--band 0.75 0.95`, the textured foreground shelves.
  - **A push-in** climbs steadily: FastText [0,28) goes from 1.019 to 1.068.
  - **Trust rows only at cc ≥ 0.95, and only when the band has texture.** The tool prints the
    band's edge share across and down, and flags **TOO LITTLE TEXTURE** below 1% in either
    direction. A plain band can converge to 1.0000 / cc 1.000 without matching anything, so
    that reading proves nothing (curator-v10-08).
  - Choose a band with no characters in it. "no fit" means a plain ground, so judge by eye.
- **Preview filled spans:** `$PY captions_v9b.py --include-dropped curation/<file>.json`
  compiles filled dropped spans too. Camera-move spans get a move clause ("the camera
  pushes in") in place of "fixed camera". Every filled span must compile.
- **Rulings from curators (applied to everyone; verifiers enforce them):**
  - **A lookalike who is most likely a lead in disguise** (curator-v10-19, Fixer: Polly with a
    stubble beard and goggles) is filled **as that lead**, with the differences as `outfits`.
    Its `notes` start with `U5:`. If the client confirms it is the lead, it flips to kept; if
    not, it stays dropped. A lookalike that is clearly someone else (Polly's mum) stays an
    unfilled `lookalike` drop.
  - **A camera move shorter than 21 frames is `too_short`** (the reason order).
  - **Real brands are `ad_merch`.** A legible logo, store name or sponsor sign of a real brand
    (Walmart, Walgreens) is an ad. **Pudgy's own merchandise shown as a product** (trading-card
    packs, boxed plushes) is `ad_merch` too. Invented Pudgy-world names (a "Huddlebucks" café)
    are in-world text: allowed, never transcribed.
  - **Ad end cards and title cards:** `off_style` when rendered in another style (3D plushes),
    otherwise `ad_merch` or `graphic_overlay`, following the reason order.
  - **The standard plain ground** (pale grey-cyan, about RGB 210, 229, 233) is worded
    `"a plain pale grey background"` in setting descriptions, as in v9b.
  - **Steady night or dim lighting** is kept, with `time` set. Only whole-picture effects
    (flashes, washes, vignettes) are `effect`.
  - **`motion` bands** must avoid anything that moves by itself: drifting clouds, water, steam,
    blinking lights. They fake a pan (PlushHug: −20 px on the sky, 0 px on the igloo).
  - **An outfit or prop handed over inside a span** goes on the character who has it for most
    of the span. The beats describe the hand-over.
  - **Dreams and imagined scenes** (curator-v10-07):
    - A scene that **fills the whole picture**, is drawn on style, and has no bubble border or
      whole-picture effect is an ordinary shot. Keep it if every other rule holds, and put
      `DREAM:` in its `notes`.
    - A dream inside a thought bubble or cloud, or a lead that morphs from a plush or picture,
      is `lead_copy`.
    - The cloud wipes into and out of a dream are `transition`.
  - **Local light:** a light cone or glow on part of the picture (a phone flash, a lamp) is
    in-world and is described in a beat. A flash that whitens most of the picture is `effect`.
  - **OLD_DESIGN cues:** any of these marks the older design, in `.mp4` files too (10MM, WFH):
    - a yellow beak or yellow feet;
    - a periwinkle body with gradient shading on the face;
    - glossy shaded furniture.
  - **Tracking shots** (the background slides while the lead stays in place) are `pan_left` or
    `pan_right` with `travel: "none"`.
  - **A phone point of view** with the other lead on its screen is `lead_copy`.
  - **`motion` on near-plain grounds can mislead even at cc ≥ 0.95** (curator-v10-04,
    Unboxing). Where the only texture is a few lines or specks, confirm with `compare`: every
    background mark must overlap exactly. A pan over a fully plain ground can't be seen and
    can't teach anything, so `compare` decides there.
  - **An outfit put on near the end of a span** is not listed in `outfit` (the compiler would
    claim it for every window). The beats say when it goes on.
  - **3D or game footage on a screen** is `off_style`, per the reason order.
  - **A dim room in daytime** (curtains drawn) keeps `time: "day"`, and the dimness goes in
    `view` ("with the curtains drawn and the room dim"). Light that changes inside a span
    because of an in-world action (Pax closes the curtains) is in-world: describe it in a
    beat. It is not an `effect`.
  - **Framing for a lead lying down or partly hidden:** judge by how large the lead is drawn,
    not by the visible height alone. A lead in bed with only the head showing is not
    automatically "wide".
  - **A Polly-shaped penguin with a pearl necklace** (often with a bow) is never kept outright.
    The pearl necklace is v9b's marker for Polly's mum.
    - If the skit shows her as someone other than Polly (Polly appears beside her, or a
      Mother's Day story), she is an unfilled `lookalike` drop.
    - If she plays Polly's part (Pax's partner, no other Polly in the skit), she is a
      `lookalike` filled as `polly`. The necklace and bow go in `outfits`, and `notes` start
      `U5: pearl necklace`. The client decides. (Decided 2026-11-05: Polly dressed up, in the four
      skits asked about.)
  - **Captions and watermarks are `graphic_overlay`.** That includes the brand's own
    `@pudgypenguins` handle: these are the overlays a loss mask could later remove. A brand
    sign-off end card is `ad_merch`.
  - **Small or half-cut lead depictions still count** (ornaments, snow globes, card prints):
    `lead_copy`, whatever the size, as in v9b.
  - **OLD_DESIGN** (settled 2026-11-04 from a contact sheet of every skit tagged with it).
    There are two older looks, and both are dropped as `off_style` with an `OLD_DESIGN:` note.
    1. **The yellow-foot variant.** This is the current flat drawing, but the feet and beak are
       **yellow, hue at or above about 42°** (e.g. 240, 208, 48 ≈ 50°; Wash 255, 214, 66 ≈ 47°).
       It covers Affection, Bromance, LastBite, QUALITYTIME, SkinCare, Upset, SAMEROOM,
       Blanket, ALREADYSEENIT and Wash. Captions never name foot colour, so training on it
       would teach yellow feet. Excluded: the client confirmed it is no longer acceptable (U9).
    2. **The older palette:** periwinkle Pax, a paler rose Polly with gradient shading on the
       face, glossy shaded rendering (10MM, WFH).

    **Current design** has salmon to amber feet and beak, **hue below about 40°**: salmon
    ≈ 27° (240, 160, 96), amber ≈ 35° (240, 160, 48; WhichOne, PostWorkout, 5Mins). Measure
    the hue on the feet, away from outlines. Current-design leads on glossy furniture are not
    OLD_DESIGN (GFMoods).
  - **A mirror reflection of a lead** is `lead_copy`.
  - **A lead seen only as a flipper at the frame edge** is not listed in `characters`. The beat
    says "a flipper reaches in from the right", without naming anyone. An insert showing only a
    flipper is `no_lead`.
  - **Leads under a blanket with their heads and faces visible** are kept, not `lead_hidden`.
  - **Exits and re-entries:** a lead who climbs into a box or behind something `exits` at that
    frame. A lead who leaves and comes back within one take needs a `same_shot` split, because
    `enters_exits` can't express a return.
  - **`motion` bands of long horizontal lines** (sofa bases, table edges, shelves) can fit
    falsely even at cc 0.996: a −47 px jump was reported on a locked shot. Use bands with
    texture in both directions, and confirm with `compare`.
  - **Compound moves** (two clearly visible directions at once, such as a pan with a
    pull-back, or a pan drifting up as fast as it moves sideways) get `camera: "other"` and
    are **not filled**: any single move clause would be false.
    "Clearly visible" means the second direction is **more than about half the main move**
    (verifier-v10-22, FoodPic: a tilt at 2/3 of the pan, a 16% push under a pan). A smaller drift
    stays under the main move. Fit a full affine and compare it with a translation-only fit:
    `motion`'s translation reading can hide a zoom. A move whose speed changes
    (a fast burst easing into a slow push) is split at the change.
  - **Focus blur on a lead** (depth of field, a focus pull) is `effect`. The flat Pudgy look
    has no lens blur. A soft-focus background behind a sharp lead is allowed.
  - **Black edge bars from export** (a few px of pure black at the left and right, as in
    Sniff and Foot) are kept, with `notes` starting `PILLARBOX:`. Prep crops them.
  - **Tracking shots from a moving vehicle** (the car stays put, the scenery recedes) are
    `camera_move` with `camera: "other"`, unfilled. No move clause would be true for them.
  - **Easing tails and split moves:** a move of 2–4 frames at the start or end of a shot is its
    own `too_short` span, so the static part starts clean. Two moves separated by a short rest
    are separate spans, and each follows the rules on its own.
  - **A screen turning to show a lead inside a live shot:** end the kept span at the first
    frame where the screen faces out.
  - **An outfit taken off or put on mid-shot:**
    - With 21 or more frames on each side, split there (`same_shot`), so "wearing …" is true
      for every window.
    - Otherwise leave it off `outfit` and let the beats say when it changes.
  - **Jump cuts on a locked background:** a character who pops to a new held position with no
    in-between frames, on an identical background, is a cut inside the shot. Split there with
    `boundary: "same_shot"` (Dramatic, LoveIsland, Finally, Dishwasher). A 2-frame pose
    snap that is animation (a fast turn) is not split.
  - **A reason that changes inside one continuous shot** gets a `same_shot` split at the
    change, so each part carries its own reason. A depiction can leave mid-pan, for example,
    and the pan after it can then flip under U3.
  - **A span that would need both U3 and U5 to flip** (a disguised-lead lookalike in a camera
    move) stays an unfilled `lookalike`. Its `notes` explain how to fill it.
  - **An inset-panel margin** (the background drawn smaller than the picture, leaving a
    lighter border) is an export artifact: `other`, and it stays dropped. A faint seam along
    **one** edge (a strip of a few % of the width, a few levels lighter; Relocate) is noted in
    `notes` and kept.
    A border on **two or more** edges, even over only part of an edge (Traffic [156,194): the
    left, right and bottom), counts as an inset panel: `other`, dropped (verifier-v10-26).
  - **A short stretch with no lead inside a kept span** (under 21 frames, e.g. an empty
    doorway before a lead walks in) can stay. Its beat names what is visible ("The doorway
    stands empty while …"). Prep skips windows where no lead shows for most frames. A
    lead-free stretch of 21 frames or more is split off as `no_lead`.
  - **Thought or speech bubbles showing an icon or text** (a dish, a crossed-out dish) are
    `graphic_overlay`, as in v9b (Knowing, Handful). Small empty thought puffs are an allowed
    effect. A bubble showing a lead is `lead_copy`.
  - **Following a character over a flat 2D background** (the street slides sideways behind
    walking leads, clouds rise past a falling lead) is `pan_left`/`pan_right`/`tilt_up`/
    `tilt_down`, filled. A vehicle-mounted tracking shot stays `other`, unfilled.
  - **Steady set lighting is kept:** a light beam in a dark room, a candle-lit counter. It is
    `effect` only where a vignette or whole-picture wash is actually visible.
  - **Lights-off rooms** (curator-v10-11, Unhelpful): a **steady** dim or cool cast over the
    whole room is night lighting, even where it greys a lead's white belly. Keep it, with
    `time: "night"`. Use `effect` when the cast **changes** within the span (fading in or
    out, flashing) or is an unnatural colour wash (a red alarm tint, a sepia flashback).
  - **An insert showing only a beak** (or a flipper) is `no_lead`.
  - **Loose figurines and toys that are neither lead are props**, on a shelf or in a basket
    (Easter's crowned and igloo figurines). Only merch **shown as a product** (boxed,
    displayed, unboxed with sparkle) is `ad_merch`.
  - **Dissolves inside a dropped run:** where nothing on either side of a dissolve could be
    kept, the run can be one span with the most permanent reason. A `transition` needs its
    own span only next to keepable footage.
  - **Generic penguin pictures that are neither lead are props:** a penguin-face mug in
    another colour, a cream penguin on a book cover. Anything recognisable as Pax or Polly is
    `lead_copy`.
  - **Copies of a lead on screen at once** (identical extra Pollys) are `lead_copy`, not
    `lookalike`. A lead figurine sitting on a prop (a Polly figurine on a phone) is
    `lead_copy` wherever it shows.
  - **Drawings on a lead's body** (highlighter doodles) go in `outfits`, so every caption
    names them. **Clothing printed with a lead** (a jersey with a small Pax on the chest) is
    `lead_copy`, as with v9b's Champ and Tone crests.
  - **A camera settle bounce** after a move (a few px of wobble) belongs to the move span.
    The kept span starts once the background is still.
  - **`motion` on flat-coloured objects** (lamp shades, fridge doors, blurred cupboards) fits
    falsely even at cc ≥ 0.95: drift of 8–20 px, or a scale of 0.88 at cc 1.000. Use a
    hard-edged object (a drawer front, the bed foot) or a raw pixel difference.
  - **Weather drawn over the set** (rain, snow or blizzard streaks with a steady overcast
    palette) is in-world. Keep it and put the weather in `view` ("with snow falling"). It is
    not an `effect`.
  - **Fog and steam:** a whiteout that hides a change of place is `transition`. A
    whole-picture wash fading out is `effect`. Steam that stays local is in-world.
  - **In-world animation is neither a camera move nor an `effect`:** clock hands, a sheen on
    glass, drifting clouds.
  - **`time` comes from what the picture shows.**
    - `night`: a dark window or sky, lamps lit, a night palette.
    - `sunset`: the sun on the horizon.
    - A character asleep in a daylight-coloured room is `day`. A sky colour alone sets
      nothing.
- **Version pairs were removed before batching** (see `skipped` in `skits_v10.json`).
  If you still meet a file whose picture repeats another in your batch, curate it and say so
  in `notes` on its first span.

## Output

One file per skit at the `out` path in `skits_v10.json`
(`curation/iteration_7__<stem, spaces → _>.json`). `curated_by` is your curator id;
`curated_at` is today. `curate_check.py` must report no ✗ errors, and every ! warning must be read.
