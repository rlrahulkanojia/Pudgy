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
        "video_groups": ["showcase", "gl", "epoch_00"],
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
