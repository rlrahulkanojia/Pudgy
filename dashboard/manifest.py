"""
Static manifest for the Pudgy Penguins training/inference dashboard.

This data is curated from the docs in `training_approach/` and `docs/training_reports/`
(dataset sizes, base models, gate status, etc.) — it is historical record, not something
derivable at runtime, so it's kept here as plain data rather than re-parsed from markdown
on every page load.

`video_prefix` is the blob-name prefix (folder) under the dashboard's Azure container
where that version's final-output videos live. Sub-groups (e.g. v4's BEST/FINAL/FINAL2)
are represented as nested prefixes and are discovered dynamically at render time via
`azure_utils.list_videos`.
"""

DATASETS = [
    {
        "name": "Original 75-clip set",
        "used_by": ["v1", "v2"],
        "clip_count": 75,
        "resolution": "768×1360 (portrait)",
        "fps": 16,
        "frames": "33 (4×8+1)",
        "duration": "~2.06s",
        "notes": "Narrative skit fragments, mixed motion, real room backgrounds. "
                 "The founding dataset for the CogVideoX (v1) and Wan2.2-A14B (v2) runs.",
    },
    {
        "name": "iteration_2_v4 (LTX-2.3)",
        "used_by": ["v4"],
        "clip_count": 249,
        "resolution": "1080×1920 native → 544×960 re-encoded",
        "fps": "24 native → 25 (LTX-native)",
        "frames": "49-frame windows (÷32 crop)",
        "duration": "variable (scene-split)",
        "notes": "249 human-curated scene-split clips from 70 source skits, rebuilt to LTX-native "
                 "format. Re-encode step yields ~303 windows from 196 clips (~280 excluding "
                 "internal-cut clips).",
    },
    {
        "name": "Happy-expression pilot set",
        "used_by": ["v5"],
        "clip_count": 28,
        "resolution": "1080×1080 source → 1024×1024 trained",
        "fps": 24,
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
        "id": "v1",
        "title": "v1 — CogVideoX1.5-5B-I2V",
        "base_model": "THUDM/CogVideoX1.5-5B-I2V",
        "status": "Superseded",
        "status_color": "#8a8f98",
        "thesis": "Single character/style LoRA, attention-only, on the original 75-clip set.",
        "summary": [
            "Learned the Pudgy style and Pax/Polly identity in the first ~5 frames.",
            "Every checkpoint loses the character mid-clip (drift → vanish) — a temporal/scene "
            "failure, not a style one.",
            "VAE round-trip proved the VAE was not the quality ceiling (PSNR ~38dB, SSIM ~0.996); "
            "the problem was the generation path (432×768 portrait cap + attention-only + free-I2V drift).",
            "Result: fixed 4 real trainer bugs, completed cleanly (9h39m, final loss 0.0294), "
            "but architecturally superseded by v2's decoupled identity/motion approach.",
        ],
        "video_prefix": "v1",
        "video_groups": None,
    },
    {
        "id": "v2",
        "title": "v2 — Wan2.2-A14B (decoupled identity/motion)",
        "base_model": "Wan2.2-I2V-A14B (MoE, two experts)",
        "status": "Gate G1: PASS — golden",
        "status_color": "#2f9e44",
        "thesis": "Two-expert LoRA (identity low-noise + motion high-noise) + FLF2V keyframe "
                  "interpolation to decouple identity from motion.",
        "summary": [
            "Held-out showcase: temporal SSIM 0.949, structural stability 0.880 (no mid-clip "
            "vanish — the v1/CogVideoX failure), source fidelity SSIM 0.905.",
            "Golden checkpoints: low-noise (identity) epoch 40, high-noise (motion) epoch 40.",
            "Trained via musubi-tuner 0.3.4 on the same 75-clip dataset as v1.",
            "Known limits: mild background drift on some scenes; canonical-view bias.",
            "Foundation that v5 (happy expression) extends.",
        ],
        "video_prefix": "v2",
        "video_groups": None,
    },
    {
        "id": "v4",
        "title": "v4 — LTX-2.3-22B (current lead)",
        "base_model": "Lightricks LTX-2.3-22B",
        "status": "G1 ✓ · G2 ✓ · Phase 3 trained (eval pending)",
        "status_color": "#2f9e44",
        "thesis": "Stylized-2D-native base + IC-LoRA edge/Canny structure conditioning + "
                  "Claude-driven prompt pipeline, on a rebuilt/grown LTX-native dataset.",
        "summary": [
            "Beats v1: motion survives the LoRA, characters stay stable (no mid-clip dissolve).",
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
        "id": "v5",
        "title": "v5 — Happy-Expression LoRA (Wan2.2-A14B extension)",
        "base_model": "Wan2.2-I2V-A14B (continue-trained from v2's golden checkpoints)",
        "status": "Pilot complete — golden: low-noise expert",
        "status_color": "#2f9e44",
        "thesis": "Teach one controllable expression (Pax, happy) as a third axis alongside "
                  "v2's identity/motion decomposition. Both experts were trained and run "
                  "head-to-head to find where expression belongs.",
        "summary": [
            "**Expression belongs on the LOW-noise expert.** Two runs of 1008 steps (~7h each) "
            "continue-trained from v2's goldens: high-noise (final loss 0.00184) vs low-noise "
            "(0.00095). The low-noise run wins on every axis and is the only one that stays "
            "**promptable** — it leaves v2's G1-validated motion prior untouched.",
            "**Controllability verified.** Same start frame, prompt as the only variable: "
            "the happy prompt gives squinted eyes + open beak; the neutral prompt gives open "
            "eyes + closed beak (SSIM 0.934). The high-noise run ignored the prompt (SSIM 0.969).",
            "**Best temporal stability in the programme:** adjacent-frame SSIM 0.978 "
            "(v1 0.925, v2 0.949); subject never vanishes (area 19.7% → 19.9%).",
            "**Generalises to unseen backgrounds** — background drift of 2/255 on a ground never "
            "trained, holding identity under free I2V (no end keyframe), the exact condition "
            "that broke v1.",
            "Known limits: expression *hold* relaxes late (only 0.875s of hold data exists); "
            "partial canonical-view drift on ¾ angles; conditioning-frame memorisation when "
            "driven from the training frame — drive from real scene frames instead.",
            "Image-conditioned only: the box holds i2v-A14B weights, so text-only generation "
            "would need a separate t2v checkpoint. Every generation takes a start image.",
        ],
        "video_prefix": "v5",
        "video_groups": None,
    },
]


def get_approach(approach_id):
    return next((a for a in TRAINING_APPROACHES if a["id"] == approach_id), None)


def total_clip_count():
    return sum(d["clip_count"] for d in DATASETS)
