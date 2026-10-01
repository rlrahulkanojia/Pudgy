"""
Static manifest for the Pudgy Penguins training/inference dashboard.

This data is curated from the docs in `training_approach/` and `docs/training_reports/`
(dataset sizes, base models, gate status, etc.) — it is historical record, not something
derivable at runtime, so it's kept here as plain data rather than re-parsed from markdown
on every page load.

`video_prefix` is the blob-name prefix (folder) under the dashboard's Azure container
where that version's final-output videos live. Sub-groups (e.g. LTX's BEST/FINAL/FINAL2)
are represented as nested prefixes and are discovered dynamically at render time via
`azure_utils.list_videos`.

`id` stays the internal version key (it's the blob prefix and the link back to the docs);
`name` is what the client sees. TRAINING_APPROACHES is ordered newest-first, so the first
entry is the latest experiment — the one the main page highlights.

`status_color` is a Streamlit badge colour name, not a hex value.
"""

DATASETS = [
    {
        "name": "Joint motion + expression set (v8)",
        "used_by": ["v8"],
        # 14 NEW clips: the iteration_5 blushing delivery. Everything else was already
        # counted under v5-v7; iteration_5 re-sent 216 files, which md5 de-dup drops.
        "clip_count": 14,
        "training_clips": 2096,
        "resolution": "1080×1080 source → 1024×1024 trained",
        "fps": "24",
        "frames": "13 – 57, plus 49-frame loops (4N+1)",
        "duration": "0.54s – 2.4s",
        "notes": "One combined set for one model: 5 movements (walking, running, waving, sitting, "
                 "jumping) and 8 expressions — blushing is new — for both characters, 2,096 "
                 "training clips from 193 client performances. One camera angle (26 "
                 "performances) is held back to test the model on clips it was not trained on. "
                 "New this round: smooth 2-second looped versions of walking, waving, jumping and "
                 "sitting, made only where the client's cycle joins seamlessly. While building it "
                 "we found 21 client files that are missing frames — the decoder silently repeats "
                 "the previous frame, which puts a stutter in the motion — so those clips are cut "
                 "before the first missing frame (a re-export has been requested). Known gaps: "
                 "confused and happy were delivered at one short length only (48 clips each vs 192), "
                 "and there is still no standing-idle, sad, scared or two-character interaction footage "
                 "in training.",
    },
    {
        "name": "Motion + expression primitives (v7)",
        "used_by": ["v7"],
        # 140 NEW clips from the iteration_4 delivery. The other 68 sources are the
        # iteration_3 expression clips already counted under v5 (7) and v6 (61) — counting
        # them again would inflate the client-delivered total.
        "clip_count": 140,
        "training_clips": 2272,    # 208 sources x 4 grounds x shot ladder x duration ladder
        "resolution": "1080×1080 source → 1024×1024 trained",
        "fps": "24",
        "frames": "13 / 17 / 21 / 25 / 29 / 33 / 37 / 57 (4N+1)",
        "duration": "0.54s – 2.4s",
        "notes": "208 source clips across 13 labels — 7 expressions (happy, surprised, angry, "
                 "confused, crying, neutral, laughing) and 6 movements (walking, running, "
                 "waving, sitting, jumping, standing idle) — for 2 characters at 9 camera "
                 "angles. Each is alpha-composited onto 4 flat grounds and rendered across a "
                 "close-up/medium/wide shot ladder AND a duration ladder, giving 2,272 "
                 "training clips: 8× the v6 set. Character balance is near-even (111 Pax / "
                 "115 Polly). "
                 "The duration ladder is the substantive change: every label is also emitted "
                 "at a common short length (13 frames for movement, 21 for expression), so "
                 "clip length cannot act as a shortcut for the label — in v6 each emotion "
                 "shipped at exactly one length, which made length a perfect predictor. "
                 "1,368 of the 2,272 clips (60%) are head-truncations of longer footage and "
                 "are captioned 'opening frames only', so the caption matches what is on "
                 "screen rather than the full arc. Nothing is loop-tiled or time-stretched: "
                 "the longest clip for every label is exactly what the client shot. "
                 "Known gaps: 'standing idle' was never delivered and is DERIVED from the "
                 "neutral footage, where the body is frozen — so it teaches 'stop' as "
                 "'freeze'. Three clips were dropped as unrecoverably damaged, leaving "
                 "'happy' at 14 sources and 'confused' and 'walking' at 16, against 18 for "
                 "every other label; 'confused' is also the one label that failed the "
                 "expression distinctness gate, for Polly.",
    },
    {
        "name": "Original 75-clip set",
        "used_by": ["v1", "v2"],
        "clip_count": 75,          # as delivered by the client
        "training_clips": 75,      # what actually reached the trainer
        "resolution": "768×1360 (portrait)",
        "fps": "16",
        "frames": "33 (4×8+1)",
        "duration": "~2.06s",
        "notes": "Narrative skit fragments, mixed motion, real room backgrounds. "
                 "The founding dataset for the CogVideo and Wan2.2 runs.",
    },
    {
        "name": "iteration_2_v4 (LTX-2.3)",
        "used_by": ["v4"],
        "clip_count": 249,             # as delivered by the client
        "training_clips": "~303 windows",  # after LTX re-encode + 49-frame windowing
        "resolution": "1080×1920 native → 544×960 re-encoded",
        "fps": "24 native → 25 (LTX-native)",
        "frames": "49-frame windows (÷32 crop)",
        "duration": "variable (scene-split)",
        "notes": "249 human-curated scene-split clips from 70 source skits, rebuilt to LTX-native "
                 "format. Re-encode step yields ~303 windows from 196 clips (~280 excluding "
                 "internal-cut clips).",
    },
    {
        "name": "Expression set (all 4 emotions)",
        "used_by": ["v6"],
        # 61 NEW clips; the 7 Pax/happy clips the v5 pilot used came in the same
        # iteration_3 delivery and are already counted there. 68 sources in total.
        "clip_count": 61,
        "training_clips": 272,     # 68 sources x 4 backgrounds, one zoom per pair
        "resolution": "1080×1080 source → 1024×1024 trained",
        "fps": "24",
        "frames": "21 / 29 / 37 / 57 (per emotion, 4N+1)",
        "duration": "0.875s – 2.4s",
        "notes": "68 ProRes-4444 alpha clips: 2 characters × 4 emotions (happy, surprised, "
                 "angry, neutral) × 7–9 camera angles. Alpha-composited onto 4 flat grounds "
                 "and rendered across a synthesised close-up/medium/wide shot ladder → 272 "
                 "training clips. First set in the programme with even character balance "
                 "(136 Pax / 136 Polly) and 100% colour-grounded captions. Each emotion "
                 "ships at exactly one frame count, which made clip length a perfect "
                 "predictor of emotion in training — tested and cleared at eval (G-L).",
    },
    {
        "name": "Happy-expression pilot set",
        "used_by": ["v5"],
        "clip_count": 7,           # as delivered by the client
        "training_clips": 28,      # 7 angles x 4 composited backgrounds
        "resolution": "1080×1080 source → 1024×1024 trained",
        "fps": "24",
        "frames": "21 (4×5+1)",
        "duration": "~0.875s",
        "notes": "7 source clips: ProRes 4444 with a real ALPHA channel (61.6% of frame "
                 "transparent) — not white backgrounds; a naive decode composites onto black. "
                 "One performance filmed from 7 fixed angles (front, quarter-front L/R, "
                 "quarter-front-2 L/R, side L/R). Alpha lets the same performance be composited "
                 "onto 4 flat grounds → 28 training clips, which is what buys background-"
                 "invariance. Trained at 1024² not 1080²: Wan's 8× VAE + 2×2 patchify needs an "
                 "even latent side (1080/8 = 135 is odd). Pilot scale, not a final-quality set.",
    },
]

TRAINING_APPROACHES = [
    {
        "id": "v8",
        "name": "Joint Motion + Expression Wan 2.2",
        "base_model": "Wan2.2-I2V-A14B, fine-tuned on top of v7 (both v7 models, combined correctly)",
        "status": "Trained · fully evaluated · long-form scenes produced · v9 planned",
        "status_color": "green",
        "thesis": "Build on everything v7 learned, but as one model for movement and expression "
                  "instead of two that interfere; add blushing and looped movement; and choose the "
                  "best checkpoint with a test that measures whether the model understands each "
                  "label, on a camera angle held out of this round's training.",
        "summary": [
            "**Better at both jobs than v7's specialists.** On held-out test clips, the single v8 "
            "model reconstructs expressions better than v7's expression model and movements better "
            "than v7's movement model (denoising error 0.0030 vs 0.0032, and 0.0027 vs 0.0028).",
            "**Recognisable labels.** On clips from a camera angle held out of v8's training, v8 picks "
            "the right expression 81% of the time (chance: 12.5%) and the right movement 80% (chance: "
            "20%), up from 63% and 60% for the v7 starting point. The new blushing expression is "
            "recognised for both characters.",
            "**A problem in v7 found and fixed.** Loading v7's two models together counted a shared "
            "component twice — a 76% error that made the combined model about 10× worse at "
            "reconstructing real clips. v8 starts from the corrected combination.",
            "**Best model so far on generated clips too.** On the same test prompts, seeds and start "
            "frames, v8 produces the asked-for expression or movement as often as or more often than "
            "both earlier models — e.g. movement: Pax 40% (v7: 30%, v2: 20%), Polly 60% (v7 and v2: "
            "50%). Small test sets (10 clips per cell), so read these as ±10–20 points.",
            "**Finding 1 — the faster the movement, the more footage it needs.** Running and jumping "
            "are the least accurate movements, and they are weaker still in the earlier models — so "
            "the limit is the footage, not the training: each movement is one short performance "
            "(running: 0.7 s, under two strides) filmed from nine angles, always on the spot.",
            "**Finding 2 — two animations in one clip works, unevenly.** Asked for a movement and an "
            "expression together, the expression comes through in 94% of clips and the movement in "
            "44%; the expression tends to take over later in the clip.",
            "**Next steps (v9).** (1) Longer scenes — with props, different backgrounds, and captions "
            "written for whole scenes, cut from the client's 101 finished skits. (2) Much more footage "
            "of fast movement — several different takes per action, 3–5 s clips, characters crossing "
            "the frame (Round 5 data request). (3) Images of the props that appear in scenes (bus, "
            "plate, pillow, …), added a few at a time together with the clips they appear in.",
            "**Most of the learning happens early.** The biggest gains came in the first 300 of 600 "
            "training steps; the chosen checkpoint is step 550.",
            "**Long-form scenes.** Two of the client's skits, LifeWithHer (14.5 s) and EatingStages "
            "(13.9 s), were re-made shot by shot from the storyboards at 1080×1920, 24 fps. Each "
            "shot starts from the matching frame of the original skit; shots with several props on "
            "a shared table are also anchored on the original's end frame to keep the props steady.",
            "Known limits: in some movement clips the camera slowly pulls back (a zoom-out) — "
            "traced to the base model and the standard negative prompt, with a caption fix planned "
            "for v9; longer movement clips play the action once slowly rather than repeating it; "
            "two-character interaction works from the base model's own ability but has not been "
            "trained yet.",
        ],
        "video_prefix": "v8",
        "video_groups": ["Long-form_scenes", "Expressions", "Movements", "Motion_and_expression", "Interaction"],
    },
    {
        "id": "v7",
        "name": "Motion + Expressions Wan 2.2",
        "base_model": "Wan2.2-I2V-A14B (continue-trained from the v2 goldens)",
        "status": "Trained · expressions 41/42 · movement model chosen (G-X)",
        "status_color": "orange",
        "thesis": "Widen v6 from four expressions to seven, add six motion primitives, and "
                  "train each on its own expert: expression continues the low-noise golden, "
                  "motion is an A/B across both experts decided by gate G-X. Every label is "
                  "also emitted at a common short length, so clip length cannot become a "
                  "shortcut for the label.",
        "summary": [
            "**Expression is promptable across all 7 labels.** 41 of 42 label pairs separate "
            "at every checkpoint, from one start frame with only the caption changing — the "
            "contrastive fix from v6 holds at nearly double the label count.",
            "**One measured failure: `confused` vs `neutral` for Polly** (face-SSIM ~0.95 "
            "against a 0.92 bar), consistent across the whole run. `confused` is the thinnest "
            "class in the set (64 clips against 216 for the mature labels, and 7 source angles "
            "where others have 9) and is also the subtlest expression, so this reads as a data "
            "gap rather than a training failure.",
            "**More training does not help.** Mean separability is flat across 3,000 steps "
            "(0.8070 at step 500 → 0.8030 at step 3,500), matching v5's finding that "
            "responsiveness plateaus early. The golden is early, not late.",
            "**Movement works, and we picked where it lives.** The same 1,072-clip movement "
            "set was trained into each half of the model separately and gate G-X generated "
            "the same five actions from both. The fine-detail half won: its actions stay "
            "distinguishable (0.944 against a 0.95 bar) and it animates ~50% more than the "
            "alternative, which blurred actions together (0.962, close to ignoring the "
            "prompt entirely).",
            "**That is the safer outcome.** Because the broad-motion half was left "
            "untouched, it still acts as a fixed reference: if something regresses later we "
            "can swap one piece back and prove which half caused it. Had the other arm won, "
            "both halves would have changed and that ability would be gone.",
            "Known limits on movement: the actions animate but subtly — these are 0.5-second "
            "test clips at the shortest trained length, and `sitting` is a seated idle by "
            "design so it barely moves. Longer, stronger samples come after the winning "
            "model finishes training.",
            "Known limits: the sweep below is 1 seed — the plan requires ≥3 for a verdict, so "
            "these rank checkpoints rather than select the golden. Motion gates have not run.",
        ],
        # The LATEST checkpoint only. The full 5-checkpoint sweep is diagnostic evidence
        # and lives in the training container under v7/eval/sweep/; showing every step
        # here buries the current result under 70 near-identical clips.
        # NOTE: latest is not the same as golden. Separability is flat across the run
        # (0.8070 at step 500 -> 0.8030 at step 3,500), so the golden is expected to be
        # EARLY; whichever checkpoint the 3-seed refinement selects should replace this.
        "video_prefix": "v7",
        "video_groups": ["sweep", "motion"],
    },
    {
        "id": "v6",
        "name": "Expressions Wan 2.2",
        "base_model": "Wan2.2-I2V-A14B (continue-trained from Wan2.2's golden checkpoints)",
        "status": "Trained · G-L ✓ · G-F ✓",
        "status_color": "green",
        "thesis": "Teach all four expressions to both characters at once, so expression "
                  "becomes a promptable axis. Trained contrastively — the same start frame "
                  "maps to four different labelled outcomes — on the low-noise expert only, "
                  "leaving Wan2.2's validated motion prior untouched.",
        "summary": [
            "**Expressions are promptable and distinct.** All 4 emotions × both characters "
            "render correctly from one start frame with only the caption changing. Measured "
            "on the face region, every emotion pair separates (0.83–0.85 typical) far beyond "
            "what changing the checkpoint does (0.95).",
            "**Verified across 3 seeds.** 72/72 emotion pairs distinct at every clip length. "
            "`surprised` is the weakest class and the shortest length the tightest — both "
            "still pass, and both are where more data would help most.",
            "**No hidden dependence on clip length.** Each emotion was only ever trained at "
            "one duration, so length could have become a shortcut for emotion. It did not: "
            "asking for any emotion at any length gives that emotion.",
            "**No start-frame lock-in.** Driving from the exact training frame with a prompt "
            "asking for *no* expression yields no expression — the failure that broke the "
            "Happy Expression pilot does not recur.",
            "**The step it adds is essential.** Wan2.2's golden alone cannot render these "
            "expressions — asked for 'angry' it produces a malformed face. ",
            "Known limits: 1 seed on the start-frame test; the golden checkpoint is still "
            "being selected (epoch 1 separates as well as epoch 11, and epoch 1 accounts for "
            "~76% of the total change from baseline).",
        ],
        "video_prefix": "v6",
        "video_groups": ["showcase"],
    },
    {
        "id": "v5",
        "name": "Happy Expression Wan 2.2",
        "base_model": "Wan2.2-I2V-A14B (continue-trained from Wan2.2's golden checkpoints)",
        "status": "Pilot complete — golden: low-noise expert",
        "status_color": "green",
        "thesis": "Teach one controllable expression (Pax, happy) as a third axis alongside "
                  "Wan2.2's identity/motion decomposition. Both experts were trained and run "
                  "head-to-head to find where expression belongs.",
        "summary": [
            "**Expression belongs on the LOW-noise expert.** Two runs of 1008 steps (~7h each) "
            "continue-trained from Wan2.2's goldens: high-noise (final loss 0.00184) vs low-noise "
            "(0.00095). The low-noise run wins on every axis and is the only one that stays "
            "**promptable** — it leaves Wan2.2's G1-validated motion prior untouched.",
            "**Controllability verified.** Same start frame, prompt as the only variable: "
            "the happy prompt gives squinted eyes + open beak; the neutral prompt gives open "
            "eyes + closed beak (SSIM 0.934). The high-noise run ignored the prompt (SSIM 0.969).",
            "**Best temporal stability in the programme:** adjacent-frame SSIM 0.978 "
            "(CogVideo 0.925, Wan2.2 0.949); subject never vanishes (area 19.7% → 19.9%).",
            "**Generalises to unseen backgrounds** — background drift of 2/255 on a ground never "
            "trained, holding identity under free I2V (no end keyframe), the exact condition "
            "that broke CogVideo.",
            "Known limits: expression *hold* relaxes late (only 0.875s of hold data exists); "
            "partial canonical-view drift on ¾ angles; conditioning-frame memorisation when "
            "driven from the training frame — drive from real scene frames instead.",
            "Image-conditioned only: the box holds i2v-A14B weights, so text-only generation "
            "would need a separate t2v checkpoint. Every generation takes a start image.",
        ],
        "video_prefix": "v5",
        "video_groups": None,
    },
    {
        "id": "v4",
        "name": "Ltx",
        "base_model": "Lightricks LTX-2.3-22B",
        "status": "G1 ✓ · G2 ✓ · Phase 3 trained (eval pending)",
        "status_color": "green",
        "thesis": "Stylized-2D-native base + IC-LoRA edge/Canny structure conditioning + "
                  "Claude-driven prompt pipeline, on a rebuilt/grown LTX-native dataset.",
        "summary": [
            "Beats CogVideo: motion survives the LoRA, characters stay stable (no mid-clip "
            "dissolve).",
            "IC-LoRA edge conditioning fixes the catastrophic identity/colour failures that no "
            "inference knob could solve alone.",
            "Long clips (5s/10s) via edge control — removes the ~97-frame ceiling.",
            "Seamless loops (endpoint keyframes + ffmpeg crossfade); high quality delivery at "
            "1536×2688 (4.1MP) via ×2 spatial upscaler.",
            "Four runs trained: pudgy_lora_A_768, pudgy_lora_B_768 (G1 golden), pudgy_ic_768 "
            "(production IC-LoRA), pudgy_p3_768 (Phase 3, eval pending).",
        ],
        "video_prefix": "v4",
        "video_groups": ["BEST", "FINAL", "FINAL2", "phase2_variations"],
    },
    {
        "id": "v2",
        "name": "Wan2.2",
        "base_model": "Wan2.2-I2V-A14B (MoE, two experts)",
        "status": "Gate G1: PASS — golden",
        "status_color": "green",
        "thesis": "Two-expert LoRA (identity low-noise + motion high-noise) + FLF2V keyframe "
                  "interpolation to decouple identity from motion.",
        "summary": [
            "Held-out showcase: temporal SSIM 0.949, structural stability 0.880 (no mid-clip "
            "vanish — the CogVideo failure), source fidelity SSIM 0.905.",
            "Golden checkpoints: low-noise (identity) epoch 40, high-noise (motion) epoch 40.",
            "Trained via musubi-tuner 0.3.4 on the same 75-clip dataset as CogVideo.",
            "Known limits: mild background drift on some scenes; canonical-view bias.",
            "Foundation that the Happy Expression run extends.",
        ],
        "video_prefix": "v2",
        "video_groups": None,
    },
    {
        "id": "v1",
        "name": "CogVideo",
        "base_model": "THUDM/CogVideoX1.5-5B-I2V",
        "status": "Superseded",
        "status_color": "gray",
        "thesis": "Single character/style LoRA, attention-only, on the original 75-clip set.",
        "summary": [
            "Learned the Pudgy style and Pax/Polly identity in the first ~5 frames.",
            "Every checkpoint loses the character mid-clip (drift → vanish) — a temporal/scene "
            "failure, not a style one.",
            "VAE round-trip proved the VAE was not the quality ceiling (PSNR ~38dB, SSIM ~0.996); "
            "the problem was the generation path (432×768 portrait cap + attention-only + free-I2V drift).",
            "Result: fixed 4 real trainer bugs, completed cleanly (9h39m, final loss 0.0294), "
            "but architecturally superseded by Wan2.2's decoupled identity/motion approach.",
        ],
        "video_prefix": "v1",
        "video_groups": None,
    },
]


def get_approach(approach_id):
    return next((a for a in TRAINING_APPROACHES if a["id"] == approach_id), None)


def latest_approach():
    """The newest experiment — TRAINING_APPROACHES is ordered newest-first."""
    return TRAINING_APPROACHES[0]


def approach_name(approach_id):
    """Client-facing name for an internal version id (e.g. 'v2' -> 'Wan2.2')."""
    approach = get_approach(approach_id)
    return approach["name"] if approach else approach_id


def total_clip_count():
    """Clips as delivered by the client — not the derived training-set size."""
    return sum(d["clip_count"] for d in DATASETS)
