# Pudgy skit remakes — bedroom set

30 Oct 2026 · Asynq

Four client skits from iteration 2 (Little Spoon, 4AM Show, Morning Fart, Wake Up Early) were remade scene by scene with the v8 model at 480×848. All four keep the original's cuts, timing and soundtrack. 14 of 17 parts passed every check on the first seed, 2 passed on a later seed, and 1 is kept as the best of three with a small known flaw.

## How they were made

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

## Little Spoon

The closest of the four: every beat in order, all 5 parts passed on the first seed.

**Azure links:** [with original soundtrack](azure:pudgy/2026-10-28_longform/skits/little_spoon_480x848_audio.mp4) · [silent](azure:pudgy/2026-10-28_longform/skits/little_spoon_480x848.mp4) · [side by side](azure:pudgy/2026-10-28_longform/skits/little_spoon_compare.mp4)

**Details**

| Field | Value |
| --- | --- |
| Original | PP-LittleSpoon-Base.mp4, 8.2 s (196 frames) |
| Scenes / parts | 4 scenes, 5 parts, all still camera, close-ups |
| Props | None (bed, pillow, blanket) |
| Seeds | All parts seed 42, no retries |
| GPU time | About 32 min, including model load |

**Prompt**

| Part | Time | Prompt |
| --- | --- | --- |
| 1 | 0.00–1.17 s | Polly, a short round pink penguin. Polly lies on her side under the blanket and opens her eyes; then Polly lifts her head and looks up to the left with rosy cheeks, her beak moving as she speaks; close-up shot, eye level, fixed camera, locked-off shot; a dim bedroom with a mustard headboard, a duck-shaped bedpost and a peach lamp, with her head on a pillow under a pale blanket. |
| 2 | 1.17–2.42 s | Pax, a short round blue penguin. Pax rocks upright on the pillow with one fin raised, squeezing his eyes shut for a moment; then Pax looks up to the right with wide shining eyes, holding his fin raised, his beak moving; close-up shot, eye level, fixed camera, locked-off shot; a dim bedroom with a mustard headboard, a duck-shaped bedpost and a peach lamp, with his head on a pillow under a pale blanket. |
| 3 | 2.42–4.29 s | Polly, a short round pink penguin. Polly gazes up from the pillow with wide eyes and rosy cheeks; then Polly shuts her eyes, smiling, and nestles her head down into the pillow; then Polly rolls over to turn her back, tucking herself under the blanket; then Polly lies curled with her back turned, wriggling a little as she settles; medium close-up shot, eye level, fixed camera, locked-off shot; a dim bedroom with a mustard headboard, a duck-shaped bedpost and a peach lamp, with a crumpled pale blanket pulled over the bed. |
| 4 | 4.25–6.17 s | Polly, a short round pink penguin. Polly lies curled with her back turned, wriggling a little as she settles; then Polly rolls her head back over her shoulder and stares with wide puzzled eyes; medium close-up shot, eye level, fixed camera, locked-off shot; a dim bedroom with a mustard headboard, a duck-shaped bedpost and a peach lamp, with a crumpled pale blanket pulled over the bed. |
| 5 | 6.17–8.17 s | Pax, a short round blue penguin. Pax lies curled on his side with his round back turned, shifting under the blanket; then Pax pokes his feet out from under the blanket and wiggles them, still facing away; medium close-up shot, eye level, fixed camera, locked-off shot; a dim bedroom with a mustard headboard, a duck-shaped bedpost and a peach lamp, with a crumpled pale blanket pulled over the bed. |

**Observation**

- Same close-ups, bedroom, blanket and lamp; beats in order: Polly asks to cuddle, Pax pops up excited, Polly snuggles and rolls over, looks back puzzled, Pax turns away wiggling his feet.
- Polly's roll-over and look-back land a fraction of a second earlier than in the original.
- Untrained moments (lying in bed, rolling over) came out naturally, because every part is pinned to the original's frames.
- Softer than the client's 1080×1920 original, as expected at 480×848.

## 4AM Show

Close to the original, push-in camera moves included; Polly's biggest singing moment is a little smaller.

**Azure links:** [with original soundtrack](azure:pudgy/2026-10-28_longform/skits/four_am_show_480x848_audio.mp4) · [silent](azure:pudgy/2026-10-28_longform/skits/four_am_show_480x848.mp4) · [side by side](azure:pudgy/2026-10-28_longform/skits/four_am_show_compare.mp4)

**Details**

| Field | Value |
| --- | --- |
| Original | PP-4amshow-Base.mp4, 9.0 s (215 frames) |
| Scenes / parts | 3 scenes, 4 parts, every scene a slow camera push-in |
| Props | 1: a microphone (plus a laptop on the floor as set dressing) |
| Seeds | Parts 1–3 seed 42; part 4 seed 1042 (seed 42 had a sudden jump of 18.7× against the original's 4.6×) |
| GPU time | About 45 min, including one retry |

**Prompt**

| Part | Time | Prompt |
| --- | --- | --- |
| 1 | 0.00–2.29 s | Pax, a short round blue penguin. Pax reclines against the pillow in bed with heavy-lidded eyes and a weary little smile; then Pax blinks slowly and nods along, his eyes drooping; medium shot, eye level, the camera pushes in slowly; a dim bedroom with a golden wooden bed, a duck-topped bedpost and pale bedding, drawn on a plain grey background. |
| 2 | 2.29–4.04 s | Polly, a short round pink penguin. Polly hunches over a microphone with her eyes squeezed shut, belting out a song; then Polly turns sideways and rocks back, still singing into the microphone; then Polly faces the front and sings with her beak wide open; medium shot, eye level, the camera pushes in slowly; a dim room with a pale icy floor, a small round yellow lamp and an open laptop on the floor, drawn on a plain grey background. |
| 3 | 4.00–5.79 s | Polly, a short round pink penguin. Polly faces the front and sings with her beak wide open; then Polly flings one flipper out wide and belts with her beak stretched open and her eyes shut; medium shot, eye level, the camera pushes in slowly; a dim room with a pale icy floor, a small round yellow lamp and an open laptop on the floor, drawn on a plain grey background. |
| 4 | 5.79–8.96 s | Pax, a short round blue penguin. Pax lies back against the pillow propping his cheek on a flipper, his eyes drooping; then Pax lets his eyes fall shut and dozes off; then Pax snaps his eyes open with a startled little jolt; medium shot, eye level, the camera pushes in slowly; a dim bedroom with a golden wooden bed, a duck-topped bedpost and pale bedding, drawn on a plain grey background. |

**Observation**

- Beats match: drowsy Pax nods along in bed, Polly sings into the microphone with the laptop on the floor, Pax dozes off and jolts awake; the slow push-ins are kept.
- Polly's most dramatic moment (flipper flung wide, beak stretched open) is smaller than in the original.
- In part 1 the model darkened the whole frame for a moment (a palette shift); the colour match corrected it, so brightness stays steady like the original.
- Untrained moments: singing into a microphone, lying back in bed; both read clearly.

## Morning Fart

The story and the soundtrack gag land on cue; the yawn runs later and quicker than the original's.

**Azure links:** [with original soundtrack](azure:pudgy/2026-10-28_longform/skits/morning_fart_480x848_audio.mp4) · [silent](azure:pudgy/2026-10-28_longform/skits/morning_fart_480x848.mp4) · [side by side](azure:pudgy/2026-10-28_longform/skits/morning_fart_compare.mp4)

**Details**

| Field | Value |
| --- | --- |
| Original | PP-MorningFart-Base.mp4, 7.8 s (186 frames) |
| Scenes / parts | 3 scenes, 4 parts; still camera except a 0.9 s pan to Pax |
| Props | None (bed, duvet, pillows) |
| Seeds | Part 1 seed 2042 (seeds 42 and 1042 had jumps of 5.4× and 7.7× against a bar of 5.0×; 2042 passed at 4.9×); parts 2–4 seed 42 |
| GPU time | About 45 min, including model load and two retries |
| Scene note | The pan (part 3) had no curation notes; its prompt was written from the skit's frames |

**Prompt**

| Part | Time | Prompt |
| --- | --- | --- |
| 1 | 0.00–1.92 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Polly sleeps on her side under the duvet with her eyes closed while Pax lies turned away at the left, only the back of his head showing; then Polly rolls onto her back and stretches both flippers up in a wide yawn, her eyes still closed; medium wide shot, eye level, fixed camera, locked-off shot; a bedroom with a wide bed under a pale duvet and a rounded yellow headboard, with white pillows and a yellow bedside table at the right. |
| 2 | 1.88–3.79 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Polly rolls onto her back and stretches both flippers up in a wide yawn, her eyes still closed; then Polly drops her flippers back down and blinks her eyes open; then Polly rubs one eye with the tip of her flipper as she lies back on the pillow; then Polly lowers her flipper and starts to turn her head toward Pax; medium wide shot, eye level, fixed camera, locked-off shot; a bedroom with a wide bed under a pale duvet and a rounded yellow headboard, with white pillows and a yellow bedside table at the right. |
| 3 | 3.79–4.67 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Polly turns her head toward Pax lying beside her; then the duvet ripples over Pax and Polly's eyes go wide, her flippers raised under her beak; medium close-up shot, eye level, the camera pans slowly to the left; a bedroom with a wide bed under a pale duvet and a rounded yellow headboard. |
| 4 | 4.67–7.75 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Polly stares with wide startled eyes and her flippers raised under her beak while the duvet ripples beside Pax; then Polly rolls back onto her side away from Pax, still wide-eyed; then Polly lies still on her side and peeks sideways with wide, startled eyes; then Polly closes her eyes and lies still with her beak slightly open and a flipper tucked under her chin; medium wide shot, eye level, fixed camera, locked-off shot; a bedroom with a wide bed under a pale duvet and a rounded yellow headboard, with white pillows and a yellow bedside table at the right. |

**Observation**

- Same bedroom and framing; Polly wakes, yawns and stretches, rubs her eye, turns toward Pax, stares startled, rolls away, peeks and settles.
- The yawn starts about 0.5 s later than in the original and is quicker; this is the part that needed three seeds.
- In the short pan, Polly faces the camera instead of turning toward Pax.
- At the startled moment she lifts up a little more than in the original.
- Untrained moments: lying in bed, rolling over, the rippling duvet; all read clearly.

## Wake Up Early

Faithful to the original; the opening push-in is kept as the best of three seeds, with a faint background patch in its first 0.6 s.

**Azure links:** [with original soundtrack](azure:pudgy/2026-10-28_longform/skits/wake_up_early_480x848_audio.mp4) · [silent](azure:pudgy/2026-10-28_longform/skits/wake_up_early_480x848.mp4) · [side by side](azure:pudgy/2026-10-28_longform/skits/wake_up_early_compare.mp4)

**Details**

| Field | Value |
| --- | --- |
| Original | PP-WakeUpEarly-Base.mp4, 6.6 s (159 frames) |
| Scenes / parts | 4 scenes, 4 parts; opens with a slow push-in, then still camera |
| Props | 1: an alarm clock on the nightstand |
| Seeds | Part 1 seed 1042, best of three (42: 6.5×, 1042: 5.0×, 2042: 14.3× jump, against the original's 1.3× and a bar of 5.0×); parts 2–4 seed 42 |
| GPU time | About 41 min, including two retries |
| Scene notes | Parts 1 and 3 had no curation notes; their prompts were written from the skit's frames |

**Prompt**

| Part | Time | Prompt |
| --- | --- | --- |
| 1 | 0.00–2.25 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Pax and Polly lie fast asleep side by side under the blanket with their beaks open, an alarm clock on the nightstand; wide shot, eye level, the camera pushes in slowly; a bedroom with a big yellow bed between two nightstands with lamps. |
| 2 | 2.25–4.42 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Pax and Polly lie asleep side by side under the blanket with their beaks open; then Pax pops up to sit straight in bed as the alarm clock jiggles, and Polly sits up beside him; then Pax and Polly sit up in bed blinking sleepily while Polly rubs one eye with her flipper; then Pax leans over to the left toward the alarm clock while Polly holds the edge of the blanket; then Pax and Polly sink back down and pull the blanket up over themselves; wide shot, eye level, fixed camera, locked-off shot; a bedroom with a big yellow bed between two nightstands with lamps, with a digital alarm clock on the left nightstand. |
| 3 | 4.42–4.92 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. the blanket is pulled all the way up over Pax and Polly, a lump shifting underneath it; wide shot, eye level, fixed camera, locked-off shot; a bedroom with a big yellow bed between two nightstands with lamps. |
| 4 | 4.92–6.62 s | Pax, a short round blue penguin, and Polly, a short round pink penguin. Pax and Polly lie asleep side by side under the blanket, flippers resting on the edge, eyes closed; then Pax opens his beak in a snore and flops a flipper out while Polly rolls toward him; then Polly sleeps with one flipper flung across Pax, both of them with their beaks hanging open; wide shot, eye level, fixed camera, locked-off shot; a bedroom with a big yellow bed between two nightstands with lamps, with a digital alarm clock on the left nightstand. |

**Observation**

- Beats match: both asleep, the alarm jiggles, they pop up to sit, Pax leans toward the alarm, they sink back under the blanket, then sleep sprawled out, snoring.
- Opening push-in: no seed matched the original's very smooth motion. In the kept seed a faint lighter patch shows on the back wall for the first 0.6 s, then fades; the characters themselves are fine.
- The alarm clock's digits are not stable; they change from frame to frame.
- Untrained moments: sleeping, sitting up in bed, reaching for the alarm, pulling up the blanket; all read clearly.
