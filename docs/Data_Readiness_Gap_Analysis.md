# Pudgy Penguins — Training-Data Readiness & Gap Analysis

**Scope:** Deep review of the CogVideoX1.5-5B-I2V LoRA training pipeline against the assets currently in the `Data/` folder. Answers two questions: (1) can what we have be used for training, and (2) what is missing.

**Prepared:** July 2026 · Sprint lead review · Base model: `THUDM/CogVideoX1.5-5B-I2V` · Trainer: `Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train`

---

## 1. Verdict (read this first)

**The 30 clips we have are usable — but only as *raw material that must be processed*, not as a training set. On their own they are not trainable, and they do not cover the full dataset the plan calls for.**

Three headline facts:

1. **Technically clean, format-ready.** All 30 base skits are 1080×1920, 24 fps, H.264 — which maps almost losslessly onto the model's 768×1360 portrait training resolution (5 px crop) and resamples cleanly to the model's native 16 fps. No re-encoding pain.
2. **They are finished multi-shot skits, not training clips.** They must be cut on scene changes into single-action shots. After doing that, **108 single shots** exist — but most are very short. Only **38 shots reach the 49-frame minimum** and only **19 reach the plan's 81-frame default**. That is below the plan's 60–80-clip dataset target and near the tested 25-clip floor.
3. **The high-value assets are entirely absent.** There are **zero** character turnarounds, T-poses, transparent/alpha character passes, isolated props, or neutral-background reference clips in the folder. Every image is flat RGB. These are exactly the assets the data request asks the client to produce, and the pipeline leans on them.

Bottom line: **we can start shot-splitting, captioning, and baseline testing immediately, but this delivery confirms — rather than satisfies — the data request. The character-reference and single-action material is still owed.**

---

## 2. What the pipeline actually requires

Verified against the model card, diffusers docs, the CogVideoX finetune README, and the Passenger12138 trainer source (citations in §7). A few points **correct or tighten the assumptions in the Final Sprint Plan** — flagged with ⚠️.

### 2.1 Model / clip constraints (CogVideoX1.5-5B-I2V)

| Constraint | Requirement | Our source | Status |
|---|---|---|---|
| Resolution | Short side = **768**; long side **768–1360**, both ÷16. Max **1360×768** | 1080×1920 → 768×1360 (crop 5 px) | ✅ near-lossless |
| Frame count | **8N+1** (25, 33, 41, 49, 65, 81, 161). **81 safest** for 1.5-I2V (VAE `patch_size_t=2`) | must be cut to length | ⚠️ see §4 |
| Native fps | **16 fps** (not 8; 8 fps is a v1.0/export artifact) | 24 fps | ⚠️ must resample |
| Max duration | ~161 frames ≈ 10 s | n/a | — |
| Precision | **bf16** | n/a | — |
| I2V first frame | **Auto-extracted from frame 0** of each clip — not supplied separately | n/a | ✅ simplifies work |
| Min clip length | <25 frames trains poorly; **≥25 frames** for new concepts | many shots <25 frames | ⚠️ see §4 |

### 2.2 Trainer data format (Passenger12138 / CogVideoX-Fun convention)

This is **not** the `videos.txt` + `prompt.txt` layout. It expects:

```
dataset/
├── train/
│   ├── 00000001.mp4     ← raw H.264 mp4, decoded on the fly (decord)
│   ├── 00000002.jpg     ← static image (T-pose etc.), type:"image"
│   └── ...
└── train.json           ← single metadata array
```

```json
[
  {"file_path":"train/00000001.mp4","text":"<dense prose caption>","type":"video"},
  {"file_path":"train/00000002.jpg","text":"<dense prose caption>","type":"image"}
]
```

- **One prose caption per clip, inline in `text`.** No separate caption files.
- **Bucketing supported** (`--enable_bucket`, aspect-ratio dicts), so mild resolution variation is tolerated — but pinning to 768×1360 maximizes consistency.
- **Frames sampled** via `--video_sample_n_frames` (default 49) × `--video_sample_stride`. A clip must physically contain enough frames to yield the request.
- ⚠️ **Joint image+video training caveat:** the `type:"image"` path exists, but in the current repo snapshot the **image branch in `get_batch` is partially commented out**. Do not assume T-pose stills "just work" — this needs a code check/patch in Week 1 before relying on it.

### 2.3 Research-driven corrections to the plan

Three plan assumptions are weaker than stated and should be revisited:

- ⚠️ **81-frame default is aggressive.** At 16 fps that needs ≥5.06 s of *continuous single-shot* action, which only 19 of our shots have. A **49-frame default (3.06 s)** roughly doubles the usable set to 38 and still exceeds the ≥25-frame floor. Recommend 49 as the working default, reserving 81 for the long single-shot clips.
- ⚠️ **The 70/30 neutral-gray background rule is not supported by the literature.** A single flat backdrop tends to get "baked into" the LoRA. Best practice is to **vary backgrounds** (which our show footage already does) or, for true isolation, use **alpha-matte background removal** — not a solid gray fill. The plan's specific 70/30 ratio appears to be folklore. This *reduces* the burden on the client (they may not need to mass-render neutral-gray plates) but *raises* the value of transparent/alpha passes.
- ⚠️ **T-poses "teaching character volume" via joint training is unproven** for this stack, and (see §2.2) partially disabled in the trainer. Keep turnarounds in scope as an **experiment**, not a load-bearing assumption. Character identity will mostly come from the I2V first frame + video clips.

---

## 3. What we have — validated inventory

| Asset | Count | Specs | Training value |
|---|---|---|---|
| `30_videos/` base skits | 30 | 1080×1920, 24 fps, H.264, 4–21 s | **Primary raw material** — must be shot-split |
| `30_prompts/` storyboard PDFs | 30 | 1-per-skit, ~140 beats total | **Caption seed material** — narrative briefs, not T5 captions (see §5-D) |
| `client/SampleFlow/` Survival versions | 3 | 1080×1920, 24 fps | Reference for concept→animatic→final flow |
| `client/SampleFlow/PP_Survival_ArtAssets.png` | 1 | 8910×5432, **flat RGB** | Reference only — shows BG plates + `Prp_*` props, but flattened, not isolated layers |
| `client/guidelines/` pages | 6 | ~2048×1150 | **High value for captioning** — posing/staging/camera rules, Pax=screen-left / Polly=screen-right |
| `Reels/` | 5 | **720×1280**, compressed | ❌ Too low-res — skip for training |
| `frames/` | ~55 PNGs | flat RGB, from Reels | ❌ Low-res, no alpha — not useful |
| `client/Documentation.docx` | 1 | — | Links to full Google Drive library, Figma design guide, Notion storyboard, experiments sheet (all gated/external) |

**Characters:** Pax (blue) and Polly (pink). **Total base footage:** 363 s (6.1 min) across 30 skits.

**Note on scale:** the client states they have **150+ animations** and T-pose/character-sheet artifacts; this folder holds a **30-clip sample plus links**. The full library and reference sheets exist on their side (Google Drive / Figma / Notion) but are **not in hand**.

---

## 4. Conformance analysis — the shot-length reality

Every base skit was run through scene-cut detection (PySceneDetect, `ContentDetector`). Result: **108 single-shot segments**, mean length **3.36 s**. Because the model needs a whole-number of continuous frames at 16 fps, short shots fall out fast:

| Target clip | Frames (8N+1) | Min length @16 fps | Usable single shots | Notes |
|---|---|---|---|---|
| Minimum viable | 25 | 1.56 s | **80** | ≥25-frame floor; below this trains poorly |
| Short | 33 | 2.06 s | 67 | |
| — | 41 | 2.56 s | 51 | |
| **Recommended default** | **49** | **3.06 s** | **38** | Best size/quality trade for our footage |
| — | 65 | 4.06 s | 25 | |
| **Plan's default** | **81** | **5.06 s** | **19** | Only 19 shots qualify |
| Long-form | 161 | 10.06 s | 6 | |

**Reading this:** if we standardize on the plan's 81-frame clips, we get **19 training clips** — thin. Dropping to **49-frame clips yields 38**, and a **mixed 25–49-frame set yields up to ~80** (though the very short 25-frame clips are lower quality). Practically, expect a **usable dataset of ~40–60 clips** after quality culling — enough to train (research floor is 25, "best" target is 100), but at the low end. This is the single most important number in the review, and it is well under the plan's assumed 60–80.

**Levers to grow the set:** (a) request the fuller library from the linked Drive (client has 150+); (b) accept 33–41-frame clips for simple actions; (c) commission the single-action reference clips (§5), which are purpose-built to be the right length.

**Frame-rate handling:** 24 → 16 fps is **not an integer stride**, so do **not** rely on the trainer's frame-skip. Pre-resample each clip with `ffmpeg -r 16` before packing. (Confirmed: the `--fps` flag is metadata/selection, not a true resampler.)

**Excluded/needs-handling within the 30:** a few clips carry on-screen text or non-character cutaways (e.g. `Manifested` title card, `Universe` seahorse shot) — exclude those shots or crop the text.

---

## 5. What's missing (the gaps)

Ordered by impact on training quality.

**A. Character reference assets — MISSING entirely.**
No turnarounds, no T-poses, no multi-angle character sheets, no expression sheets in the folder. The client confirms these exist (T-pose "180/360 view," character grids with color codes) but they live in Figma/Drive, not here. These anchor character identity and color. **Needed:** transparent-PNG turnarounds + expression sheets for Pax and Polly with exact color codes.

**B. Isolated / alpha character & prop passes — MISSING entirely.**
Every asset is a flattened composite; zero alpha channels, no `.mov`/`.psd`/`.aep`/PNG-sequence layers. Given the research (alpha-matte removal beats solid-gray fill for isolation), **transparent character passes are now the highest-value production ask** — more so than neutral-gray plates. **Needed:** Pax/Polly on transparent background per action, plus `Prp_*` props as individual transparent PNGs and backgrounds as clean plates.

**C. Single-action reference clips — MISSING.**
The existing footage is narrative and fast-cut; it does not cleanly cover one-action-per-clip motion primitives (walk, idle/breathe, blink, wave, head-turn, jump, sit, eat) at a controlled length. These are the cheapest way to both fix the shot-count shortfall (§4) and guarantee action parity. **Needed:** short single-action clips, static camera, ~3–5 s each, ~5 per action.

**D. Captions — storyboard seeds now exist, but no training captions.**
The `30_prompts/` PDFs (added later) are **storyboard scripts, not T5 captions** — one per skit, ~140 numbered beats total, containing POV hook lines, Instagram "inspo" links, INT/EXT headings, per-beat action descriptions, camera directions (ZOOM/cut), song cues, and art notes. They **cannot be used as captions as-is** (multi-shot, narrative, contain non-visual dialogue/framing). But they are a real upgrade: each beat already describes its action in human-written natural language, which **seeds and validates the Tier-2 action/motion suffix** and improves accuracy over blind VLM captioning. Net effect: captioning is still **our** job (build the two-tier identity-anchor + VLM-suffix pipeline over ~40–80 per-clip captions), but now with a human-authored action reference per beat to align against. The beats also give a semantic label for each split shot, aiding the §4 shot-splitting/culling.

**E. Dataset packaging — not done.**
Nothing is in the trainer's expected form. Required conversion: cut → resample to 16 fps → crop/resize to 768×1360 → assemble `train/*.mp4` + `train.json` metadata with `type` labels. Straightforward but not yet started.

**F. Fuller library access — pending.**
The 150+ library, Figma model sheets, and Notion storyboards are linked but login-gated. Pulling even 20–40 more skits materially improves the dataset size problem in §4.

### What is NOT a gap
- Resolution / codec / fps — all fine, cleanly convertible.
- Style/brand coverage — the 30 skits are on-brand and varied (environments, both characters, range of actions).
- Neutral-gray plates — **de-prioritized** per §2.3; alpha passes are the better investment.

---

## 6. Recommended next actions

1. **Build the processing pipeline on what we have now** (unblocked): PySceneDetect split → cull non-single-action & text shots → `ffmpeg -r 16` → crop/resize 768×1360 → produce a first-pass `train/` + `train.json` at a **49-frame default**. This yields a ~38-clip starter set for a diagnostic training run.
2. **Send the client a tightened asset request** — now evidence-backed. Prioritize, in order: (i) transparent turnarounds + expression sheets for Pax/Polly w/ color codes; (ii) 20–40 more base skits from the Drive; (iii) single-action reference clips (~5 per motion); (iv) transparent character/prop passes. Explicitly **drop or downgrade the neutral-gray plate ask**.
3. **Patch/verify the trainer's image-branch** before committing to T-pose joint training (§2.2).
4. **Revisit the plan's 81-frame + 70/30-background decisions** (§2.3) and lock a 49-frame default.
5. **Stand up the captioning pipeline** in parallel — it's the long pole and doesn't depend on client deliverables.

---

## 7. Sources

Model & constraints: [CogVideoX1.5-5B-I2V model card](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V) · [diffusers CogVideoX pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) · [diffusers CogVideoX training](https://huggingface.co/docs/diffusers/en/training/cogvideox) · [official finetune README](https://github.com/zai-org/CogVideo/blob/main/finetune/README.md) · [I2V patch_size_t issue #532](https://github.com/zai-org/CogVideo/issues/532)

Trainer format: [Passenger12138 repo](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train) · [its finetune README](https://raw.githubusercontent.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train/main/finetune/README.md) · [dataset class](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train/blob/main/finetune/data/dataset_image_video.py) · [CogVideoX-Fun Training-Lora wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora) · [finetrainers dataset docs](https://github.com/huggingface/finetrainers/blob/main/docs/dataset/README.md)

Dataset best practices: [CogVideoX paper (single-shot ~6 s clips)](https://arxiv.org/abs/2408.06072) · [HunyuanVideo report (PySceneDetect split)](https://arxiv.org/html/2412.03603v2) · [LVD-2M (cuts→scene transitions)](https://arxiv.org/html/2410.10816) · [SVD](https://arxiv.org/html/2311.15127) · [PySceneDetect](https://www.scenedetect.com/) · [RunPod video-LoRA guide (rank/background/overfitting)](https://runpod.ghost.io/complete-guide-to-training-video-loras/)

*Local measurements (resolution, fps, scene-cut shot counts, alpha-channel checks) computed directly on the `Data/` folder with ffprobe + PySceneDetect.*
