# Pudgy Penguins — Training Approach v4 (LTX-2)

**A distinct base-model track**, to be **baked off against [Training_Approach_v3.md](./Training_Approach_v3.md) (AniSora V3.2)**. Where v2→v3 kept the same 75-clip dataset and swapped the base, **v4 changes three things at once on purpose**: (1) base → **LTX-2** (Lightricks LTX-2.3-22B), (2) the dataset is treated as **unconstrained** — we rebuild it to LTX's native format and grow it, and (3) a new **AI-driven prompt system** replaces the "feed the caption string directly" approach.

**Goal (sharpened for v4):** the **smoothest, highest-quality, temporally-consistent** 2D animation of **Pax** (blue) and **Polly** (pink) — flat pastel, thick black outlines — with **exact character consistency and strong control over the generation**. Budget uncapped; dataset flexible.

**Core thesis:** LTX-2 is the open base most explicitly built for *stylized 2D*, it trains a LoRA in **under an hour** (vs 10–20 h), and it ships the exact control primitives our goal needs (IC-LoRA structure control, keyframe interpolation, an official looping sampler). Its two real risks — a *character-LoRA-kills-motion* bug and a *$10M-revenue-gated license* — are things we design around, not tune away. And because LTX uses a **Gemma-3 text encoder and is extremely prompt-sensitive**, the **prompt pipeline is part of the model**, not an afterthought — so v4 makes prompt generation/expansion a Claude-driven subsystem.

> **Version note.** "LTX-2" the GitHub repo (`Lightricks/LTX-2`) currently hosts **LTX-2.3 (22B)** — the March-2026 refresh of the Oct-2025 19B LTX-2. This doc means **LTX-2.3-22B** throughout. Do **not** use the older `Lightricks/LTX-Video-Trainer` (that trains 0.9.x 2B/13B only); the LTX-2 trainer is `packages/ltx-trainer` inside the LTX-2 repo.

---

## 1. Approach in one line

**LTX-2.3-22B** as the base, driven for 2D by **(a)** a light **video-native style LoRA** for the Pudgy look, **(b)** **IC-LoRA edge/structure conditioning** (thick outlines → clean Canny edges) to pin character construction, **(c)** **image-conditioning + keyframe interpolation** to lock identity at shot endpoints, and **(d)** the **official Looping Sampler** (temporal tiling + AdaIn) for seamless, drift-free continuous output — all fed by a **Claude-authored, schema-consistent prompt pipeline** that matches training-caption distribution at inference time.

---

## 2. LTX-2 strengths (independent assessment)

| Strength | Why it matters for Pax/Polly | Confidence |
|---|---|---|
| **Explicitly built for stylized 2D** | Lightricks' own docs: *"2D animation styles produce fewer temporal artifacts than photorealistic output — simpler geometry and flat color reduce the model's burden."* Flat fills + hard edges *suppress* the model's weak spots (no skin/cloth physics to get wrong). | High (primary) |
| **IC-LoRA structure control (Canny/edge, Union, Pose, Motion-Track)** | Thick black outlines *are* a clean Canny signal — edge conditioning preserves character construction architecturally, not via text. This is the most principled outline-preservation mechanism of any candidate. | High |
| **Keyframe interpolation + first/last-frame** | `KeyframeInterpolationPipeline` pins identity at endpoints (the v2/v3 decoupling thesis) natively. | High |
| **Official Looping Sampler** | Purpose-built seamless loops via temporal tiling + **AdaIn color normalization** (prevents drift/oversaturation across tiles). Neither Wan nor CogVideoX ships this. Directly serves "smooth, streamed, consistent." | High |
| **10–20× faster LoRA training (<1 h)** | 3–5 training iterations *per day* vs per-week. Radically tightens the golden-checkpoint search. | High |
| **Video-native LoRA training** | LoRA trains on real motion clips, not stills — temporal consistency is *learned*, not faked. | High |
| **Fastest inference of the group** | Report claims ~18× Wan-2.2 per-step on H100; distilled model = 8 steps, sub-minute 5 s clips. Enables near-real-time / "streamed" iteration. | Med (vendor benchmark) |
| **24 fps native** | No RIFE post-interpolation (CogVideoX was 16 fps). | Med (see fps caveat §3) |
| **Multi-LoRA stacking + ReStyle IC-LoRA** | Character + style + structure LoRAs simultaneously; a community ReStyle IC-LoRA already covers "Disney 2D / flat 2D / cel-shaded / line-art." | Med |
| **Native audio (can be disabled)** | Not needed now (silent output), but free optional SFX/dialogue later; disabling costs nothing. | High |
| **Rebuilt VAE (2.3) with better edge/text preservation** | Better odds of surviving thick-outline/flat-fill round-trip — but still must be measured (§6.1). | Med |

---

## 3. LTX-2 weaknesses & risks (honest)

| Risk | Detail | Severity | Mitigation in v4 |
|---|---|---|---|
| **Character-LoRA-kills-motion bug** | At weight 1.0 a trained character LoRA produces near-static output; at 0.5 it ignores action prompts ([HF #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36)). | **CRITICAL** | **Design around it:** prefer **IC-LoRA structure + image-conditioning + a *light* style LoRA (weight 0.55–0.9)** over a heavy identity LoRA; use the community sigma-schedule workaround; verify on cartoon content early (§6.2). |
| **End-of-clip bright-flash artifact** | Flash in final frames, open bug ([#148](https://github.com/Lightricks/LTX-2/issues/148)); worst exactly at a loop seam. | HIGH (for loops) | Generate longer, trim tail before looping; the Looping Sampler's overlap blending also helps. |
| ~~License is NOT Apache — $10M revenue gate~~ ✅ **RESOLVED** | "LTX-2 Community License": free commercial use under $10M entity revenue; acceptable-use restrictions still apply. **License confirmed a non-issue for this project (2026-07-12)** — LTX is eligible; no paid license needed. | ~~MEDIUM~~ CLEARED | Still observe the acceptable-use terms (no undisclosed-AI/deepfake/impersonation use); derivatives (our LoRAs) remain bound to the same community license. |
| **Single-identity IC-LoRA** | IC-LoRA references one identity at a time — a real limit for **two-character (Pax+Polly) shots**. | MEDIUM | Per-character conditioning + compositing, or regional/keyframe separation; revisit Phantom/VACE-style multi-subject (as in v2 §3.4). |
| **fps** ✅ resolved | Trainer default is **`frame_rate: 25.0`** (marketing's "50 fps" is the high-end/interpolated mode). | LOW | **Train/generate at 25 fps.** Note the 2D-timing implication: LTX generates on a *smooth* grid, not held "2s-on-twos" cartoon frames — snappy timing may need prompting or post. |
| **Priors/enhancer skew cinematic** | Base + `enhance_prompt` inject photoreal/lighting vocabulary that fights flat pastel. | MEDIUM | **Enhancer OFF** for final; our Claude prompt pipeline (§7) supplies flat-style vocabulary + LoRA trigger token instead. |
| **VAE softening of thin outlines / flat-fill banding** | Classic failure of high-compression video VAEs; LTX VAE ≈8× spatial. | MEDIUM | Phase-0 VAE round-trip on real Pudgy art at target resolution (§6.1) — we already have the tooling. |
| **Long-clip degradation past ~20 s** | Quality drifts on very long single generations. | LOW (short shots) | Short shots + `extend`/looping. |
| **Newer, smaller LoRA ecosystem than Wan** | Fewer community references for LTX-2.3 specifically; flat-mascot use is empirically untested for *all* candidates. | LOW | Our own Phase-0/1 evidence settles it. |

---

## 4. Correlating LTX-2 with our actual goal

Decompose the goal — *smoothest + highest-quality + streamed-consistent + character-consistent + controllable* — and map each to an LTX-2 mechanism and its gap:

| Goal dimension | LTX-2 mechanism | Gap / how v4 closes it |
|---|---|---|
| **Smoothest motion** | 24 fps native; video-native LoRA learns real motion; distilled 8-step for fast iteration | Watch 2D-timing (smooth-grid vs held frames); protect motion from the character-LoRA bug by keeping identity in IC-LoRA/conditioning, not a heavy LoRA. |
| **Highest image quality** | Rebuilt 2.3 VAE (better edges); Dev checkpoint 30–50 steps for finals; 2x spatial upscaler | Confirm VAE preserves thick outlines (Phase 0.1); use Dev (not Distilled) for delivery renders. |
| **Streamed / temporally consistent** | Official **Looping Sampler** (temporal tiling + AdaIn), keyframe interpolation, fast gen for near-real-time | End-of-clip flash → trim/overlap; for *true* long-form streaming later, evaluate an autoregressive layer (see `docs/documents/research/LongLive_Research.md`). |
| **Character consistency** | IC-LoRA (edge/structure) + image-conditioning locks construction; keyframe endpoints bound drift; light style LoRA for the look | Two-character blending → per-character conditioning; identity precision without a heavy LoRA is the key experiment. |
| **Control over generation** | IC-LoRA Canny/Union/Pose/Motion-Track/camera LoRAs; first/last/mid keyframes; multi-LoRA stack | This is LTX's strongest axis — more native control surface than v1/v3. Requires building edge-paired training data (§6.3). |

**Net:** LTX-2's strengths line up unusually well with *control* and *stylized-2D quality*; the honest risk concentration is **identity precision under motion** (the character-LoRA bug) and **two-character shots**. v4's architecture (§5) is shaped specifically to route identity through conditioning/IC-LoRA rather than a motion-killing character LoRA.

---

## 5. v4 pipeline architecture

```
   IMAGE / STRUCTURE DOMAIN  (identity + construction — controlled)
   ┌──────────────────────────────────────────────────────────────┐
   │  Pax/Polly keyframes (human-QC'd)  +  edge/Canny maps from     │
   │  thick outlines  ──►  IC-LoRA structure conditioning           │
   │  (light style LoRA, weight 0.55–0.9, for the flat-pastel look) │
   └──────────────────────────────────────────────────────────────┘
                              │ locked construction + endpoints
                              ▼
   MOTION DOMAIN  (LTX-2 generation — fast, 2D-native)
   ┌──────────────────────────────────────────────────────────────┐
   │  KeyframeInterpolationPipeline (endpoints)  OR  ICLoraPipeline │
   │  ──►  Looping Sampler (temporal tiling + AdaIn)  ──► shot clip │
   └──────────────────────────────────────────────────────────────┘
                              ▲
                              │ prompt (LTX-native, distribution-matched)
   PROMPT DOMAIN  (Claude-driven — §7)
   ┌──────────────────────────────────────────────────────────────┐
   │  short brief  ──►  Claude expand  ──►  structured prompt JSON  │
   │  ──►  render <200-word LTX paragraph (+ identity token, style, │
   │  camera, lighting; enhancer OFF)  ──►  validate  ──►  generate │
   └──────────────────────────────────────────────────────────────┘
```

**The identity bet (central to v4):** because a heavy character LoRA kills motion on LTX-2, we keep **identity in the conditioning path** (image + IC-LoRA edge maps + keyframes) and use the LoRA only to carry the **flat-pastel/thick-outline *style*** at low weight. If Phase-1 proves a light identity LoRA *can* hold Pax/Polly without killing motion on cartoon content, we add it; if not, IC-LoRA + conditioning is the fallback that bypasses the bug entirely.

**Trainer:** `packages/ltx-trainer` (LoRA / IC-LoRA / full-FT; a flexible conditioning framework covering t2v/i2v/extend/inpaint). 80 GB recommended; a 32 GB RTX-5090 works via the **low-VRAM INT8 config**. Our A100-40 GB sits between — fp8/INT8 + gradient checkpointing + offload.

**Inference/eval:** new LTX pipelines replace the CogVideoX-bound `inference/eval_pudgy_lora.py`; the model-agnostic `training_report.py` and the multi-VAE `training_approach/scripts/vae_roundtrip.py` (add the LTX VAE to its `REGISTRY`) carry over.

---

## 6. Dataset strategy (now unconstrained)

v1–v3 were locked to the existing 75 clips (768×1360, 16 fps, 33 frames). **v4 rebuilds the dataset to LTX-native format and grows it** — this is explicitly allowed now.

### 6.1 LTX-native format
- **fps:** re-derive from the 24 fps *source* to **25 fps** (trainer default `frame_rate: 25.0`; no double-resample) — matches, not fights, the base.
- **Frame count:** `frames % 8 == 1` (e.g. 25, 49, 81, 121). Pick per-shot length; longer than v1's 33 is fine (up to ~20 s, but keep shots short for snappy 2D + to dodge long-clip drift).
- **Resolution:** width/height **divisible by 32** — ⚠️ v1's 768×**1360** is invalid (1360 not ÷32). On the **80 GB** box train up to ~**1 MP** (**768×1344×49**, bf16) or trade res for length (**544×960×81**); base generates ~0.5–1 MP then **delivers 1080p–4K via the ×2 spatial upscaler at inference**. **Multiple GPUs raise throughput, not single-sample max resolution** (data-parallel; each GPU holds the full 22B model). Keep the preprocess bucket == the generation bucket (#155).
- **Audio:** silent (audio stream disabled).
- ⚠️ **Re-run the build pipeline** (`prep/build_dataset.py` currently hard-codes the stale 49/81-frame CogVideoX spec, and the `split6.py`/`assemble3.py` tiling scripts are missing from the repo — reconstruct for the 24 fps / (F−1)%8 target).

### 6.2 New data types to add (drives the golden result)
| Data | Purpose |
|---|---|
| **Character turnarounds + expression sheets w/ color codes** | Strongest identity anchors for keyframes + IC-LoRA (client ask #1, per v2 §6). |
| **High-quality stills** | Joint image+video training; sharpen per-frame identity. |
| **Clean single-action clips** (static cam, one action) | Motion-primitive coverage; breaks the tiled-window redundancy of v1. |
| **Edge/Canny-paired clips** | *New for v4* — training pairs for **IC-LoRA structure conditioning** (extract edges from the thick outlines; the outline *is* the signal). |
| **Per-character single-subject clips** | Isolate Pax vs Polly to fight two-character blending. |
| **20–40 more skits** from the 150+ library | Diversity of environments/actions. |

### 6.3 VAE gate (do first)
Run the multi-VAE round-trip incl. the **LTX-2.3 VAE** on real Pax/Polly art at target resolution. Expected pass (8× VAE round-tripped CogVideoX cleanly, PSNR ~38 dB), but LTX's VAE is different and must be measured before scaling — thin-outline softening is the specific thing to look for.

---

## 7. Prompt system redesign (Claude-driven)

**Today:** captions are a fixed identity **ANCHOR + VLM motion SUFFIX**, anti-entanglement-pruned (`prep/caption_prune.py`), stored as flat `captions/*.txt` + `metadata.json`/`prompts.txt`, and **fed to the model verbatim** at both train and inference time. That was tuned for CogVideoX's frozen-T5 encoder.

**Why change it for LTX-2:** LTX uses a **Gemma-3 text encoder**, is **extremely prompt-sensitive**, wants a **single flowing <200-word paragraph** ordered *action → precise motion → character & environment → camera → lighting*, and ships an `enhance_prompt` LLM step that **injects photoreal/cinematic vocabulary** (bad for flat 2D). Two consequences:
1. Our **training captions** must be authored in LTX's native structure/vocabulary (not the old dense T5 prose).
2. **Inference prompts must match the training-caption distribution** or the LoRA underperforms — so we cannot let users type free-form briefs straight into the model, nor rely on the generic enhancer.

**v4 answer — one shared prompt schema, two Claude-driven entry points, structured storage:**

### 7.1 Structured prompt storage (replaces flat strings)
Store each prompt as a **structured JSON record**, not a raw string. The flat paragraph the model sees is a *rendering* of the record.

```jsonc
// prompts/00000042.json
{
  "clip_id": "00000042",
  "schema_version": "v4.1",
  "identity": ["pxngn0"],              // rare-token trigger(s); appearance lives in the bible, not here
  "style": "2d cartoon animation, cel-shaded, bold black outlines, flat pastel fills, no gradients, no shadows",
  "action_beats": [                    // chronological, one main action per ~2–3 s
    "the blue penguin waddles in from the left and stops by a hanging lamp",
    "it raises one flipper and waves twice"
  ],
  "environment": "cozy tiled room, warm interior",
  "camera": "static medium shot, eye level",
  "lighting": "soft even ambient light, flat",
  "negatives": ["photorealistic", "3d render", "gradient shading", "motion blur"],
  "audio": null,
  "rendered": "pxngn0. 2d cartoon animation, cel-shaded, bold black outlines, flat pastel fills… the blue penguin waddles in from the left and stops by a hanging lamp, then raises one flipper and waves twice, in a cozy tiled room; static medium eye-level shot, soft flat ambient light.",
  "provenance": { "author": "claude-opus-4-8", "source": "vlm_frames+beats", "created": "2026-07-12" }
}
```

- **`character_bible.json`** holds the canonical Pax/Polly data once (shape, proportions, **exact color codes**, rare token) — referenced by id, never re-described per clip (preserves the anti-entanglement discipline `caption_prune.py` enforces).
- **Benefits over flat strings:** programmatic control, A/B by field, re-render when the schema changes, per-field validation, reproducibility (schema+model+timestamp), and one source of truth for both training and inference.

### 7.2 Entry point A — Claude captioner (training time)
Pipeline per clip: sample frames → VLM frame description + storyboard beat → **Claude** composes the **structured record** in the fixed schema (identity token only, chronological beats, flat-style vocabulary), enforcing the LTX <200-word single-paragraph render. Replaces the ad-hoc anchor+suffix; keeps the *anti-entanglement rule* (identity = token, description = variables only) but in LTX-native form.

### 7.3 Entry point B — Claude prompt expander (inference time)
Short human brief → **Claude** expands into a **full structured record in the same schema**, pulling the canonical identity block from the bible, adding flat-style tokens + camera + lighting, and rendering the LTX paragraph. Crucially it is prompted to **match the training-caption distribution** (same vocabulary, order, length) — a character-aware, distribution-locked replacement for the generic `enhance_prompt` (which stays **OFF**).

- Input: `"Pax waving next to a lamp"` → Output: the full record above.
- Implemented as a small `inference/prompt_expand.py` calling the Claude API (Opus 4.8), with the schema + bible + a captioning system prompt as context. (We already have Claude access.)

### 7.4 Validation gates (cheap, automatic)
Every rendered prompt is checked before it reaches the model:
- **length** ≤ 200 words, single paragraph (no lists/line breaks);
- **required fields** present (identity token, style, ≥1 action beat, camera, lighting);
- **forbidden-appearance-terms** absent from action beats (anti-entanglement — reuse `caption_prune.py`'s blocklist);
- **style-token present** + **negatives present** (flat-look guard);
- **schema_version** stamped.

### 7.5 Why this raises quality (not just tidiness)
- **Train/inference distribution match** is the single biggest lever for LoRA fidelity on a prompt-sensitive base — the expander guarantees it.
- **Flat-style vocabulary + enhancer-off** stops the base from drifting toward photoreal.
- **Structured records** let us A/B prompt strategy (Phase 0.2's rare-token idea) mechanically across the whole set, and reproduce any generation exactly.

---

## 8. Phased plan & gates

| Phase | Action | Gate |
|---|---|---|
| **0 — de-risk** | (a) **VAE round-trip** incl. LTX-2.3 VAE on real Pudgy art (§6.3). (b) Stand up `ltx-trainer` + LTX-2.3 env on the A100-40 GB (INT8/offload); reproduce a vanilla t2v + one keyframe-interpolation + one Looping-Sampler sample; record VRAM + wall-clock. (c) **Bootstrap the prompt system** (schema + bible + Claude captioner/expander + validators). (d) **Zero-shot cartoon probe**: does base LTX hold flat-color style, and does a *light* character LoRA kill motion on *our* content? | **G0:** VAE ✅; env works; prompt pipeline emits valid records; character-LoRA-motion risk quantified on cartoon data. |
| **1 — corrected baseline** | Rebuild dataset to LTX-native (24 fps, (F−1)%8), Claude-authored captions; train a **light style LoRA**; generate a fixed held-out set via the expander. Score vs v1 (and vs v3 if available) on the §-rubric. | **G1:** beats the v1 CogVideoX run; motion survives the style LoRA. |
| **2 — control pipeline** | Build **edge-paired data** → train/apply **IC-LoRA structure conditioning**; wire **image-conditioning + keyframe endpoints + Looping Sampler**. First end-to-end decoupled shot. | **G2:** end-to-end shot, locked identity + construction, no mid-clip drift, seamless loop (flash trimmed). |
| **3 — data + two-character** | Add turnarounds/stills/single-action clips + per-character clips; solve Pax+Polly two-shot (per-character conditioning / compositing / multi-subject ref). | Two-character shot with no blending. |
| **4 — eval / select** | Rubric (v2 §5) + automated scoring per checkpoint; Dev-checkpoint finals; pick golden by eye + metrics. | Golden checkpoint chosen. |

Reuse the **v2 §5 rubric** (character identity, line/color quality, motion robustness, prompt adherence, temporal stability) so v1/v3/v4 are directly comparable at the bake-off.

---

## 9. LTX-2 (v4) vs AniSora (v3) — the bake-off

| Axis | **v4 LTX-2.3** | **v3 AniSora V3.2** |
|---|---|---|
| Style prior | Stylized-2D-optimized (general 2D) | Anime-native (trained on 10M+ anime clips) |
| Identity pinning | IC-LoRA structure + keyframes + image-cond | Native arbitrary-keyframe interp + spatial masks |
| Control surface | **Richest** (IC-LoRA Canny/Union/Pose/Motion/camera, looping sampler) | Keyframes + motion masks |
| Character LoRA | ⚠️ motion-killing bug → route identity via conditioning | Per-expert LoRA (low-noise expert misbehaves) |
| Training speed | **<1 h/LoRA** (10–20× faster) | Hours (Wan-class, two experts) |
| fps | 24–25 (ambiguous) | 16 (F=8x+1) |
| License | ⚠️ $10M revenue gate | Apache-2.0 |
| Two-character | Single-identity IC-LoRA limit | Spatial masks / anymask |
| Maturity for our style | Fast-moving, official trainer; flat-mascot untested | Anime-native but thinner tooling; flat-mascot untested |

**Decision rule:** run both through **G1** on the identical rubric + held-out prompt set. Lead with whichever holds **identity under motion** best at G1; LTX's iteration speed means it can run more experiments in the same wall-clock. **License is confirmed a non-issue (2026-07-12)**, so LTX is not gated — with its ~10–20× faster iteration and best-fit-for-2D profile, **v4/LTX is the lead exploration track**, with v3/AniSora run in parallel as the anime-native hedge.

---

## 10. Open questions to verify (before scaling)

1. ~~License eligibility~~ ✅ **RESOLVED (2026-07-12) — not an issue; LTX is eligible.**
2. ~~Native fps~~ ✅ **RESOLVED — 25 fps** (trainer default `frame_rate: 25.0`).
3. **Exact VAE compression** + outline fidelity on our art (Phase 0.2 settles fidelity regardless of the published ratio) — **incl. the flat-region grid bug [#202](https://github.com/Lightricks/LTX-2/issues/202)**, the single biggest base-level risk for our flat fills.
4. **Trainer recipe** — rank / LR / steps / sigma-schedule from `packages/ltx-trainer/docs` (config-reference, training-guide) — extract before the first run.
5. **Character-LoRA-motion bug on cartoon content** — does it manifest for flat mascots, or only photoreal? (G0 probe.)
6. **Two-character** conditioning approach — pick one at Phase 3.

---

## 11. Immediate next actions

1. ✅ **License confirmed a non-issue (2026-07-12)** — track is green; proceed to the technical de-risks below.
2. **Add the LTX-2.3 VAE to `training_approach/scripts/vae_roundtrip.py`** and run it on real Pudgy art (Phase 0.1).
3. **Stand up `ltx-trainer` + LTX-2.3** on the A100-40 GB (INT8/offload); reproduce base t2v + keyframe interp + Looping Sampler; log VRAM/time.
4. **Bootstrap the prompt subsystem** (§7): `prompt_schema.json`, `character_bible.json` (from `docs/documents/CHARACTER_SHEETS.md`), a Claude captioner, `inference/prompt_expand.py`, and the validators (reuse `caption_prune.py`'s blocklist).
5. **Zero-shot cartoon probe** for the character-LoRA-motion bug (§10.5) — the single highest-risk unknown.
6. **Rebuild a small LTX-native dataset slice** (24 fps, (F−1)%8) with Claude captions; train the first light style LoRA (G1).

---

## 12. Execution runbook — GPU handoff

**Audience:** the GPU team executing v4. Written to run **without the author present** (author is on a Mac, no CUDA). Read v4 §1–11 first for the *why*; this section is the *how*. Every step has a **Report back** box — capture those artifacts so we can pass the gates. **Commands below are grounded in the real LTX-2 repo + `packages/ltx-trainer` docs** (verbatim where quoted); anything still to confirm on the box is tagged **⚠️VERIFY**.

### 12.0 Prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA CUDA, **CUDA 13+**, Linux. **80 GB (H100/A100-80G) available ✅** — the recommended tier. Train in **bf16** for best quality. Note: the 22B model (~44 GB) + Gemma-12B encoder (~24 GB) together nearly fill 80 GB — so **precompute latents/text embeddings first, then unload the text encoder at train time** (`load_text_encoder_in_8bit: false`; it's only needed during `process_dataset.py`). That frees ~24 GB for activations and is why 80 GB is comfortable but not infinite (some users still OOM — #180). INT8 stays available as headroom for larger buckets. Not runnable on macOS. |
| Package mgr | **`uv`** (the LTX-2 repo uses `uv sync` / `uv run`). Install uv first. |
| HF access | `hf auth login`; accept licenses for `Lightricks/LTX-2.3` and `google/gemma-3-12b-it-qat-q4_0-unquantized`. |
| Anthropic API | `ANTHROPIC_API_KEY` for the prompt subsystem (Task 0.5). |
| Data | **Raw sources only: `30_videos/` (30 source skits, ~24 fps) + `30_prompts/` (storyboard beats, one per skit).** The LTX training set is **built from these on the box** (Task 1.1). The old 75-clip set is *not* reused — wrong fps (16) and resolution (768×1360, not ÷32). |

### 12.1 Environment + model download

```bash
git clone https://github.com/Lightricks/LTX-2
cd LTX-2
uv sync                                   # installs the repo + packages/ltx-trainer

hf auth login
# checkpoints (dev = finals, distilled = fast iteration, upscaler = two-stage)
hf download Lightricks/LTX-2.3 \
    ltx-2.3-22b-dev.safetensors \
    ltx-2.3-22b-distilled-1.1.safetensors \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir models/ltx-2.3
# Gemma-3 text encoder (REQUIRED for LTX-2)
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir models/gemma-3-12b
```

> **Report back:** `uv sync` clean? All four files + Gemma downloaded? Disk used.

---

### Phase 0 — de-risk

#### TASK 0.1 — Env smoke test (zero-shot 2D) · ⭐ do first
Confirms the stack runs on 40 GB and that base LTX holds a flat-2D look before any training.

```bash
cd LTX-2
uv run python -m ltx_pipelines.distilled \
    --distilled-checkpoint-path models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --spatial-upsampler-path   models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root models/gemma-3-12b \
    --quantization fp8-cast --offload cpu \
    --seed 42 --output-path smoke.mp4 \
    --prompt "2d cartoon animation, cel-shaded, bold black outlines, flat pastel fills, no gradients, a round blue penguin waddles in and waves one flipper twice in a cozy tiled room, static medium eye-level shot, soft flat ambient lighting"
```

> **Report back:** runs on 40 GB with `fp8-cast` + `--offload cpu`? Wall-clock + peak VRAM. **Does the zero-shot output preserve flat color / thick outlines**, or drift photoreal? (Try `--offload disk` if OOM.)
>
> ⚠️ **Do NOT use `--quantization fp8-scaled-mm`** — it crashes from the CLI (`TypeError: QuantizationPolicy.fp8_scaled_mm() takes 1 positional argument`, open bug [#146](https://github.com/Lightricks/LTX-2/issues/146)). Use `fp8-cast`. Also avoid the `ltx-2.3-22b-distilled-fp8` checkpoint — it produces bad/noisy results ([#193](https://github.com/Lightricks/LTX-2/issues/193), [#244](https://github.com/Lightricks/LTX-2/issues/244)); use the plain `distilled-1.1` or `dev`.

#### TASK 0.2 — VAE round-trip (outline **+ flat-fill** fidelity)
The frozen VAE is a hard ceiling; measure it on our art before scaling.
- Extend `training_approach/scripts/vae_roundtrip.py` `REGISTRY` with the **LTX-2.3 VAE** class (⚠️VERIFY the autoencoder class name/path — it ships in the LTX-2 repo, e.g. `src.ltx_vae.load_ltx_vae`, not diffusers). Encode→decode 2–3 real Pax/Polly clips at a **÷32 portrait bucket** (see 1.1), no diffusion.
- ⚠️ **Also run the constant-fill diagnostic from [issue #202](https://github.com/Lightricks/LTX-2/issues/202):** encode→decode a constant `(0,128,0)` clip and inspect the recon + its FFT. A known **open bug**: the LTX-2.3 VAE stamps a **regular grid/tile pattern on flat regions** — which is *exactly* our flat pastel fills. This is the highest-priority thing to confirm for our aesthetic; banding/grid on solid fills would be a base-level ceiling no LoRA can fix.

> **Report back:** PSNR / SSIM / **edge-SSIM** / flat-MAE + source|recon|diff montages, **plus the constant-green recon + FFT**. Two lines: **(a) do thick outlines survive?** **(b) is there visible grid/banding on flat fills (#202)?** (CogVideoX 8× VAE hit ~38 dB and was clean — LTX's VAE is different and #202 is a real risk here.)

#### TASK 0.3 — Character-LoRA-motion probe · ⭐ CRITICAL go/no-go
The single highest-risk unknown ([HF #36](https://huggingface.co/Lightricks/LTX-2.3/discussions/36)): does a trained LoRA freeze motion on *cartoon* content?
- Use the quick style LoRA from Task 1.3 (or a community 2D LoRA) and generate the **same action prompt** ("…waves one flipper twice…") at **LoRA weight 0.5 / 0.7 / 0.9 / 1.0**.

> **Report back:** at which weights does the wave motion survive vs freeze? If motion dies even at 0.5 → **route identity through IC-LoRA + image-conditioning (§5), not a heavy character LoRA.** Note the community sigma-schedule workaround result if tried.

#### TASK 0.4 — Keyframe-interp + looping probe
- `KeyframeInterpolationPipeline` with the **same image as first and last** keyframe → seamless loop. Then try the ComfyUI **Looping Sampler** (`temporal_tile_size≈80`, `temporal_overlap≈24`, `adain_factor 0.0–0.1`).

> **Report back:** loop smoothness; **is the end-of-clip bright-flash ([#148](https://github.com/Lightricks/LTX-2/issues/148)) present?** (If so, generate longer + trim tail.)

#### TASK 0.5 — Prompt subsystem bootstrap (can run off-GPU / on the Mac)
Build §7: `prompt/prompt_schema.json`, `prompt/character_bible.json` (from `docs/documents/CHARACTER_SHEETS.md` — Pax/Polly shape, proportions, **exact color codes**, rare tokens `pxngn0`/`plngn0`), `inference/prompt_expand.py` (Anthropic API, Opus 4.8; brief → structured record → rendered <200-word LTX paragraph, `enhance_prompt` OFF), and validators (reuse `prep/caption_prune.py`'s blocklist).

> **Report back:** a short brief expands to a schema-valid record that passes all §7.4 gates; 3 sample rendered prompts.

**→ Gate G0:** stack runs on our GPU; VAE ✅; **character-LoRA-motion risk quantified**; prompt pipeline emits valid records.

---

### Phase 1 — corrected baseline

#### TASK 1.1 — Build the LTX-native dataset from raw (`30_videos/` + `30_prompts/`)
The box starts with **30 source skits + 30 storyboard prompts** — build the training set here. Hard targets (trainer docs): **width/height ÷ 32**, **`frames % 8 == 1`**, **fps 25**, silent. Canonical bucket **544×960×49** (≈2.0 s; use 81 for ≈3.2 s).
> ⚠️ The old 75-clip set is wrong fps/res — do **not** reuse. `prep/build_dataset.py` hard-codes a stale 49/81-frame spec and the original `split6.py`/`assemble3.py` tiling scripts are **absent** — this task **reconstructs** the build.

**a) Shot detection** — split each skit into shots (PySceneDetect, threshold 22, as in the original build):
```bash
mkdir -p shots
for v in 30_videos/*.mp4; do
  scenedetect -i "$v" detect-content --threshold 22 split-video -o shots/
done
```

**b) Re-encode + window to LTX spec** — 25 fps, ÷32 crop to 544×960, non-overlapping 49-frame clips, silent:
```bash
mkdir -p clips
ffmpeg -i shots/shot_001.mp4 \
  -vf "fps=25,scale=544:960:flags=lanczos:force_original_aspect_ratio=increase,crop=544:960" \
  -an -c:v libx264 -crf 16 -pix_fmt yuv420p -reset_timestamps 1 \
  -f segment -segment_frames 49,98,147,196,245 clips/shot001_%03d.mp4
```
⚠️VERIFY each clip is **exactly 49 frames** (`ffprobe -count_frames -show_entries stream=nb_read_frames`); drop short tails so every clip satisfies `frames % 8 == 1`. (Robust alternative: extract fixed 49-frame windows with `-ss <start> -frames:v 49` in a loop.)

**c) Content cull** — drop the non-training windows the original build removed: product/text end-cards, off-character shots, off-model costumes, wipe/match-cut straddlers. (Manual pass or reuse the original cull list.)

**d) Caption each clip** (the Task 0.5 prompt subsystem) — for every clip: pull its source skit's **beat from `30_prompts/`**, add a **per-shot VLM frame description**, and have the **Claude captioner** emit the LTX-native structured record (rare-token identity + variable-only action, flat-style vocab, single <200-word paragraph). Anti-entanglement per `prep/caption_prune.py`.

**e) Assemble `dataset.json`** (LTX `caption`/`video` schema):
```json
[
  { "caption": "pxngn0. 2d cartoon animation, cel-shaded, bold black outlines, flat pastel fills, the blue penguin waddles in and waves twice by a hanging lamp, cozy tiled room, static medium shot, flat lighting", "video": "clips/shot001_000.mp4" }
]
```

> **Report back:** clip count after cull (expect a few dozen); sample frames confirming the ÷32 crop didn't clip characters; every clip = **exactly 49 frames @ 25 fps @ 544×960**; `dataset.json` passes the §7.4 validators.

#### TASK 1.2 — Preprocess (precompute latents + text embeddings)

```bash
cd packages/ltx-trainer
uv run python scripts/process_dataset.py /data/pudgy_ltx/dataset.json \
    --resolution-buckets "544x960x49" \
    --model-path        ../../models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path ../../models/gemma-3-12b \
    --lora-trigger "pxngn0" --decode          # --decode verifies videos decode
# writes to .precomputed/{latents,conditions,...}
```

> **Report back:** `.precomputed/latents` + `conditions` populated; any decode failures.

#### TASK 1.3 — Train the light style LoRA
Start from the shipped low-VRAM config and edit for our run:

```bash
cp configs/t2v_lora_low_vram.yaml configs/pudgy_style_lora.yaml
```

Edit these keys (grounded in the real config schema):

```yaml
model:
  model_path: "../../models/ltx-2.3/ltx-2.3-22b-dev.safetensors"
  text_encoder_path: "../../models/gemma-3-12b"
  training_mode: "lora"
lora:
  rank: 32            # bump from 16; α = rank
  alpha: 32
  target_modules: ["to_k","to_q","to_v","to_out.0"]   # default = attention-only;
                      # ⚠️ our v1 lesson: try adding MLP linears (verify module names) as an A/B
optimization:
  learning_rate: 5e-5 # style LoRA (1e-4 default is a touch hot for style)
  steps: 2000
  optimizer_type: "adamw8bit"
  enable_gradient_checkpointing: true
acceleration:
  mixed_precision_mode: "bf16"
  quantization: null                 # 80 GB → train in bf16 for best quality; set "int8-quanto" only to fit the largest buckets
  load_text_encoder_in_8bit: false   # encoder is only needed at precompute (Task 1.2); unloaded at train time
data:
  preprocessed_data_root: "/data/pudgy_ltx/.precomputed"
validation:                          # periodic eval — fixed held-out prompt+seed set (single-char / 2-char / action)
  video_dims: [544, 960, 49]         # == the preprocess bucket (#155); kept modest so every-100-step eval stays fast
  frame_rate: 25.0
  generate_audio: false              # we output silent
  guidance_scale: 4.0
  interval: 100                      # eval every 100 steps → live quality-vs-step curve
  seed: 42
  negative_prompt: "photorealistic, 3d render, gradient shading, soft shadows, motion blur, worst quality, inconsistent motion, blurry, jittery, distorted"
checkpoints:
  interval: 250
  keep_last_n: -1                    # keep ALL checkpoints (LoRA files are tens of MB); pick golden post-hoc
  precision: "bfloat16"
wandb:
  enabled: true                      # live loss + validation videos across both GPUs
  project: "pudgy-ltx-v4"
output_dir: "outputs/pudgy_lora_A"
```

Launch — **2×GPU = two concurrent configs** (never data-parallel one run on 75 clips):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py configs/pudgy_lora_A.yaml &   # e.g. attention-only
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py configs/pudgy_lora_B.yaml &   # e.g. all-linear+MLP (v1 lesson)
wait
```

> **Report back:** trains on 80 GB (bf16)? time/1000 steps; loss curve (**video component** — audio is off); the per-`interval` validation videos; the saved `checkpoint-<step>/` set. **Does motion survive with the LoRA applied** (ties to Task 0.3)? Which of A/B scored better?

#### TASK 1.4 — Evaluate vs v1
Generate the fixed held-out prompt set via the Task 0.5 **expander → dev pipeline** (30–50 steps, CFG ~3–4), score on the **v2 §5 rubric** (identity / line-quality / motion / prompt-adherence / temporal).

> **Report back:** rubric scores + sample clips. **→ Gate G1:** beats the v1 CogVideoX run *and* motion survives the style LoRA.

---

### Phase 2 — control pipeline

| # | Task | Command / note |
|---|---|---|
| 2.1 | **Edge-paired data.** Extract Canny/edge maps from clips; add a `reference_video` (or `ref_media_path`) column pointing at the edge clip. | `process_dataset.py` writes these to `.precomputed/reference_latents/`. |
| 2.2 | **Train IC-LoRA** (structure conditioning; higher rank ~128). | ⚠️VERIFY the IC-LoRA `training_mode`/config variant in the trainer docs. |
| 2.3 | **Wire end-to-end:** `ICLoraPipeline` (edge/ref) + `KeyframeInterpolationPipeline` (endpoints) + Looping Sampler → full shot from human-QC'd Pax/Polly keyframes. | Dev checkpoint for finals. |

**→ Gate G2:** end-to-end shot, locked identity + construction, no mid-clip drift, seamless loop (flash trimmed).

---

### Hardware notes — 80 GB, multi-GPU, and the Gemma text encoder

**80 GB single-GPU** is the comfortable tier: **train in bf16** (better than INT8). Max useful training resolution ≈ **1 MP** (`768×1344×49`) or trade res for length (`544×960×81`); deliver 1080p–4K via the **×2 spatial upscaler at inference** — don't train higher than ~1 MP (diminishing returns + VRAM). See the token formula in v4 §6/§12; keep the preprocess bucket == the generation bucket ([#155](https://github.com/Lightricks/LTX-2/issues/155)).

**2×GPU operating plan (our setup).** Two GPUs buy **experiment throughput, not higher resolution** (data-parallel: each GPU holds the full 22B model — and on 75 clips a large DP batch *hurts* LoRA fidelity, so **never split one run across both**). Use them as **two independent trainers**:
- **Precompute once, up front:** `uv run accelerate launch --num_processes 2 scripts/process_dataset.py dataset.json --resolution-buckets "768x1344x49" --model-path … --text-encoder-path …` (both GPUs cache latents/embeddings, then the encoder is unloaded for training).
- **Sweep mode (default):** GPU0 and GPU1 each run one config concurrently → **2 experiments/hour**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py configs/pudgy_lora_A.yaml &  # e.g. rank 32, attention-only
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py configs/pudgy_lora_B.yaml &  # e.g. rank 32, all-linear+MLP
  wait
  ```
  Each run self-evaluates (native validation) + saves checkpoints (below). After each pair finishes, run the post-hoc rubric eval on a freed GPU, pick the golden checkpoint, launch the next 2 configs.
- **Decisive-run mode (alt):** for the final run, **GPU0 trains** while **GPU1 continuously evaluates each new checkpoint** on the full rubric — zero contention, cleanest golden-checkpoint curve.
- ⚠️ **Don't use FSDP model-sharding** — reported to NaN the loss for LoRA/full-FT ([#114](https://github.com/Lightricks/LTX-2/issues/114)).

**Periodic evaluation — two layers:**
- **In-training (native, cheap, live):** `validation.interval: 100` on a **fixed** held-out prompt+seed set (mix of single-char, 2-char, action-primitive — the same set as the rubric) → quality-vs-step curve, catches divergence early. `wandb.enabled: true` logs validation videos + loss across both GPUs in one dashboard.
- **Post-hoc (thorough, picks golden):** run the held-out set through the prompt expander → **dev** pipeline (30–50 steps) and score the **v2 §5 rubric** per saved checkpoint (a `compare_checkpoints.py`-style one-load sweep, ported to the LTX pipeline).

**Saving weights:** `checkpoints.interval: 250`, `keep_last_n: -1` (**keep ALL** — LoRA adapters are tens of MB), `precision: bfloat16`. **Pick the golden checkpoint post-hoc — fidelity usually peaks mid-run, not at the last step (the v1 lesson).** Mirror checkpoints + reports into a gitignored `report/` dir as in v1.

**Smaller Gemma? No — and you don't need one.** The text encoder is a **fixed dependency: Gemma-3-12B** (the connector/cross-attention was trained against it; a 4B/1B Gemma → out-of-distribution embeddings → degraded/broken output). Instead:
- **At train time its cost is ~0** — `process_dataset.py` caches text embeddings to `.precomputed/conditions/`, then the encoder is **unloaded** (`load_text_encoder_in_8bit: false`, never built). This is the real fix, not downsizing.
- **At inference/precompute** (encoder ~23 GB bf16): keep Gemma-3-12B but shrink its footprint — the default `-qat-q4_0-unquantized` is Google's **QAT-int4** checkpoint (load 4-bit ≈ 6–7 GB), or `load_text_encoder_in_8bit: true`, or **cache embeddings so repeated prompts skip the encoder entirely** ([#232](https://github.com/Lightricks/LTX-2/issues/232)), or pin it to a **separate GPU**. On 80 GB it's a non-issue.

### Report-back checklist
- [ ] 12.1 env + downloads OK
- [ ] 0.1 smoke: runs on 40 GB? VRAM/time? flat-style held? (`smoke.mp4`)
- [ ] 0.2 VAE round-trip numbers + montages; outline verdict
- [ ] 0.3 **character-LoRA-motion** weights table (the go/no-go)
- [ ] 0.4 keyframe loop + end-of-clip-flash verdict
- [ ] 0.5 prompt pipeline: valid records + 3 samples
- [ ] G0 recommendation
- [ ] 1.1 dataset **built from raw 30 videos + 30 prompts** (÷32 / 25 fps / 49-frame clips + captions) → 1.2 preprocessed → 1.3 style LoRA trained (config + checkpoints)
- [ ] 1.4 rubric scores vs v1 → **G1**

### Training heads-up from the LTX-2 issue tracker (read before Phase 1)
Harvested from `Lightricks/LTX-2` issues — the ones that will bite our exact setup:

| # | Heads-up | What to do |
|---|---|---|
| [#202](https://github.com/Lightricks/LTX-2/issues/202) | **VAE stamps a grid/tile pattern on FLAT regions** (constant-input diagnostic). Directly threatens flat pastel fills. | Run the constant-green test in Task 0.2 **before** training. If present, it's a base ceiling — weigh against v3/AniSora. |
| [#180](https://github.com/Lightricks/LTX-2/issues/180) | **Training VRAM is heavy.** An **80 GB A100 OOM'd at 2000 steps**; a 32 GB 5090 needs ~31 GB even *images-only*. | We have **80 GB ✅ + multi-GPU**. Still: **cache latents/embeddings then unload the text encoder** (Gemma ~23 GB + 22B model ~44 GB ≈ 68 GB resident — [#232](https://github.com/Lightricks/LTX-2/issues/232)), keep grad-checkpointing on, keep validation light — one user still OOM'd at 80 GB at 2000 steps. bf16 fine at ~0.5–1 MP; add INT8 only for the largest buckets. |
| [#175](https://github.com/Lightricks/LTX-2/issues/175) | **Official: train the LoRA on the FULL `dev` model, not the distilled one** — "makes a big difference." LoRAs from dev work in the distilled pipeline with no special handling. | Our config uses `dev` for training (✓). Iterate inference with distilled, ship finals with dev. |
| [#94](https://github.com/Lightricks/LTX-2/issues/94) | **Loss is dominated by audio** (audio latents ~4× video). Total loss ~1.3 at 200 steps is normal; **video component ~0.13**. | We disable audio, but if any audio path is active, **read the `video:` loss component**, not the total. |
| [#146](https://github.com/Lightricks/LTX-2/issues/146) | `--quantization fp8-scaled-mm` **crashes from the CLI** (open). | Use `fp8-cast`. (Runbook already does.) |
| [#193](https://github.com/Lightricks/LTX-2/issues/193) / [#244](https://github.com/Lightricks/LTX-2/issues/244) / [#197](https://github.com/Lightricks/LTX-2/issues/197) | **`distilled-fp8` gives bad/noisy results**; distilled-1.1 introduced an audio bug. | Use `distilled-1.1` (video) or `dev`; avoid `distilled-fp8`. |
| [#249](https://github.com/Lightricks/LTX-2/issues/249) | **Character LoRA can be trained on images only** (community doing 60/120-image per-character sets). | Viable low-VRAM path + fits our per-character plan (§6.2); confirm image-only settings in trainer docs. |
| [#123](https://github.com/Lightricks/LTX-2/issues/123) | **Shape mismatch fusing a distilled LoRA under `fp8-cast`.** | If it hits, apply the LoRA on the non-fp8 path or bf16. |
| [#116](https://github.com/Lightricks/LTX-2/issues/116) | `Gemma3TextConfig … rope_local_base_freq` AttributeError — a **transformers-version** mismatch running the pipeline. | **`pip install transformers==4.57.6`** (confirmed fix by multiple users). |
| [#155](https://github.com/Lightricks/LTX-2/issues/155) | **Latent-shape mismatch** when the generation bucket ≠ the `--resolution-buckets` used at preprocess. | Keep `video_dims` / inference size identical to the preprocessed bucket. |
| [#182](https://github.com/Lightricks/LTX-2/issues/182) (closed) | **No first-/last-frame *training* code** yet (only the inference `KeyframeInterpolationPipeline`). | Endpoint identity-pinning is an **inference-time** technique for us, not a training mode. |
| [#165](https://github.com/Lightricks/LTX-2/issues/165) | Reports of **noise in trained-LoRA output** — usually preprocessing/bucket or LR issues. | If outputs are noisy, re-check bucket match (#155), LR, and step count before blaming the model. |

### Known gotchas (LTX-specific, verified)
- **`frames % 8 == 1` and dims ÷ 32** — the old 768×1360 bucket is invalid (1360 not ÷32). Use 544×960 / 768×1344.
- **fps = 25** is the trainer default (`frame_rate: 25.0`) — settles the earlier 24-vs-50 ambiguity.
- **`generate_audio: false`** for our silent output.
- **`enhance_prompt` OFF** — it injects photoreal vocabulary; use our expander + trigger token instead.
- **Distilled = 8 steps / CFG 1** (iteration); **Dev = 30–50 steps / CFG ~3–4** (finals).
- **Character LoRA:** keep weight **0.55–0.9** + sigma-schedule workaround to preserve motion.
- **End-of-clip flash:** generate longer, trim the tail before looping.
- **40 GB:** `quantization: int8-quanto` (train) / `--quantization fp8-*` + `--offload cpu|disk` (infer).
- **LTX-2.0 LoRAs are incompatible with 2.3** (rebuilt VAE) — train from scratch.
- **`uv`** is mandatory — use `uv sync` / `uv run`, not bare `pip`/`python`.
- **Pin `transformers==4.57.6`** — other versions throw `Gemma3TextConfig … rope_local_base_freq` ([#116](https://github.com/Lightricks/LTX-2/issues/116)).
- **Gemma-3-12B is a fixed dependency** — don't substitute a smaller Gemma; precompute + unload it instead (see Hardware notes).

---

*References: LTX-2 repo https://github.com/Lightricks/LTX-2 (trainer `packages/ltx-trainer`; checkpoints `ltx-2.3-22b-dev` / `-distilled-1.1`; `enhance_prompt` param) · HF https://huggingface.co/Lightricks/LTX-2.3 · technical report arXiv 2601.03233 · prompting guide https://ltx.io/blog/prompting-guide-for-ltx-2 · 2D-animation guide https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models · IC-LoRA https://ltx.io/blog/how-to-use-ic-lora-in-ltx-2 · looping sampler https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md · license https://github.com/Lightricks/LTX-2/blob/main/LICENSE · char-LoRA bug https://huggingface.co/Lightricks/LTX-2.3/discussions/36 · end-of-clip flash https://github.com/Lightricks/LTX-2/issues/148 · ecosystem https://github.com/wildminder/awesome-ltx2. Internal: [`docs/documents/research/LTX2.3_HDR_Research.md`](../docs/documents/research/LTX2.3_HDR_Research.md), [`Training_Approach_v3.md`](./Training_Approach_v3.md), [`FINDINGS.md`](./FINDINGS.md).*
</content>
