# Training Pipeline — Deep Dive: Inference Conditioning & Open Details

**Scope:** A technical look at what `THUDM/CogVideoX1.5-5B-I2V` (our locked base) actually accepts at **inference**, focused on the question *"can we feed frame references / additional artifacts (character sheets, props) at generation time?"* — plus other details the sprint plan should pin down. Pairs with `Training_Runbook.md`.

---

## TL;DR

1. **The base model conditions on exactly ONE image — the first frame — plus the text prompt.** It has **no native support** for a separate character-sheet reference, multiple reference images, an end frame, or prop reference images at inference.
2. **Character/prop consistency comes from the trained LoRA + the composited first frame — NOT from an inference-time reference image.** This is the key difference from Seedance/Kling/closed tools (where you pass a character-sheet image with the prompt). In our pipeline the LoRA *is* the character sheet, learned once.
3. **Additional artifacts at inference = bake them into the first frame.** The artist composites Pax + Polly + props + background into the single 768×1360 first frame; the LoRA holds them consistent while they animate. Fully supported, zero extra infra. **This is the v1 path.**
4. **Frame-reference / structural control (end frame, ControlNet/pose from the animatic, trajectory) exists — but only on *different model variants* (CogVideoX-Fun, or the kijai ComfyUI wrapper), which would need their own LoRA training track.** Powerful for the "rough the poses, let the engine finish" workflow Oliver described, but it's a **separate, post-v1 track**, not the locked base.

---

## 1. What the base model conditions on

`CogVideoX1.5-5B-I2V` = T5-v1.1-XXL (text, frozen) + CLIP ViT-H (vision, for the single I2V image) + single-pass DiT. At inference (`CogVideoXImageToVideoPipeline`) the inputs are:

| Input | Supported? |
|---|---|
| **First-frame image** (1 image) | ✅ the I2V conditioning |
| **Text prompt** | ✅ dense natural language (must match training caption style) |
| Second reference image / character sheet | ❌ not accepted |
| End / last frame | ❌ not in base I2V |
| Multiple keyframes | ❌ |
| Prop reference image(s) | ❌ |
| ControlNet / pose / lineart map | ❌ not in base I2V |

**So "pass a character sheet + prompt" (the client's mental model) does not map to CogVideoX.** Consistency is delivered by the **LoRA** (learned Pax/Polly/props/style) and the **first frame** (per-shot composition). That's the whole point of training.

---

## 2. The four ways to get "frame references / additional artifacts" in

| Path | What it is | Native to our base? | Needs | Use |
|---|---|---|---|---|
| **1. Bake into the first frame** *(recommended v1)* | Artist composites Pax + Polly + props + bg into the 768×1360 first frame; model animates it | ✅ **Yes** | Nothing extra — it's standard I2V | All v1 GIFs |
| **2. Start+end keyframe** | Condition on first **and** last frame (interpolate between) | ⚠️ **No** in base; available via **kijai ComfyUI-CogVideoXWrapper** (start+end interpolation) and **CogVideoX-Fun "InP"** models | A different model variant + its own LoRA | Loop closure / "get from pose A to pose B" |
| **3. ControlNet / structural control** | Drive structure from the **line-art animatic** (canny/lineart) or **pose** skeletons | ⚠️ **No** in base I2V; available on **CogVideoX-Fun-Control / -Pose** (Canny, Depth, Pose, MLSD) and `TheDenk/cogvideox-controlnet` | A control-capable variant + its own LoRA | The "rough key poses → engine finishes" workflow Oliver wants |
| **4. Reference / IP-Adapter** | Pass a reference image that injects identity (no training) | ❌ **Does not exist** for CogVideoX as of mid-2026 (no IP-Adapter, no face embedding, no identity-token injection) | — | N/A — the LoRA replaces this |

**Critical caveat for Paths 2 & 3:** a LoRA trained on `THUDM/CogVideoX1.5-5B-I2V` (via the Passenger12138 trainer) does **not** transfer to CogVideoX-Fun — Fun is a separate fork with different weights, frame count (85 @ 8 fps vs our 81 @ 16 fps) and resolution buckets. Choosing the control path means **training the LoRA on the Fun variant instead**, with all the downstream fps/loop changes that implies. It is a deliberate model-track decision, not a toggle.

### 2.1 Frame-control engine options (scope ceiling)

Decided scope: conditioning goes **up to frame control** (first / last / sparse keyframes), **not** multi-reference image injection. Within that ceiling, these are the engines worth knowing — ranked for our use. (A 2026 ecosystem scan also surfaced multi-reference engines — VACE, Phantom, SkyReels-R2V, etc. — but those are **out of scope** and parked.)

| Engine | Frame control | Why relevant here | Catch |
|---|---|---|---|
| **CogVideoX-Fun / Wan2.2-Fun (InP)** | first+last frame; +Canny/Depth/Pose | **Least-disruption** extension of our locked base — same Alibaba PAI repo/framework, first-party LoRA + control | 8 fps / ~85 frames; single first frame only (fine — we don't want multi-ref) |
| **AniSora V3.2** (Wan2.2-based) | first/last + arbitrary keyframes; +line-art/pose/depth | **Only open model trained on anime/2D** → best flat-color fidelity; line-art guidance maps onto the client's animatics; training+RL code released | Less standardized stack (ComfyUI node + GGUF, no diffusers) |
| **LTX-2.3** | first/last + multi-keyframe; **~30–60 s single pass** | Strongest official trainer; long-form in one pass; canny/pose/depth control | Photoreal default (needs style LoRA); **portrait is OOD** and 1360 isn't ÷32 → use 768×1344/1280 |
| **ToonComposer** (ToonCrafter successor, Aug 2025) | keyframe → tween, sketch/line-art guided | Cartoon-specialized motion from drawn keyframes — fits "rough poses → finish" | In-betweener/post-keyframing, not a base generator |
| **AnimateDiff + lineart ControlNet** | SparseCtrl keyframes | Best flat-cel *look*; cheapest LoRA; mature anime lineart control | Legacy SD1.5; weak motion; short windows |

**Cross-cutting flags (in-scope):**
- **Flat-2D fidelity is the #1 risk** — every photoreal-biased base (incl. CogVideoX) is unverified on flat cartoon; **AniSora V3.2** is the only 2D-trained open model. A small **CogVideoX1.5-I2V vs AniSora V3.2 bake-off on real Pudgy art** is worth running before committing the full training budget.
- **768×1360 portrait is non-standard** (most engines want ÷16/÷32) — we may land on **768×1344**.
- **30 s is not native with consistency anywhere** — confirms the "generate ~5 s, stitch via first/last-frame" plan.

---

## 3. Recommendation by phase

- **v1 (looping GIFs, the sprint):** Path 1 + our `THUDM 1.5-I2V` LoRA + latent loop closure (the locked plan). The artist's composited first frame carries every artifact; the LoRA keeps them on-model. No reference-image or ControlNet infra needed. The client's "character sheet" is folded into the LoRA at training time.
- **Phase 1 / experimental control track:** evaluate **CogVideoX-Fun-V1.5-5b (InP + Control)** as a *parallel* model to unlock (a) **animatic-line-art → animation** via ControlNet and (b) start+end keyframes. This directly serves Oliver's "substitute character models onto the roughs" idea, and the client already produces the line-art animatics (see `Data/client/SampleFlow`). Budget a separate LoRA training run on Fun for this. **Decision needed** (see §5).
- **Production engine (ComfyUI):** use **kijai's ComfyUI-CogVideoXWrapper** — it has the richest feature set (I2V, start+end interpolation, ControlNet, Tora trajectory, FreeNoise context windows for longer video, Fun-model support, LoRA loading). Confirm our Passenger12138-trained LoRA loads in it (key-naming/format) before committing.

---

## 4. How this changes the first-frame production step

Because **everything rides on the composited first frame**, that frame becomes a first-class deliverable, not an afterthought:
- It must be exactly **768×1360**, on-model, with Pax/Polly + props + bg already placed (respecting the design-guideline staging: penguins small, 3/4 angle, props placed per the perspective guide).
- Worth building a **first-frame template** (PSD/AE comp at 768×1360 with safe-area guides) so the artist composites valid frames fast. This is the human touchpoint of the "animator-assist" goal.
- **Prop animation:** client confirmed **any prop can animate depending on the skit**. Independent prop motion (orca chomp, iFin screen) is a *learned* behavior — it must appear in training clips (captioned with the prop's motion) for the model to reproduce it from the first frame. A prop the LoRA has only seen static will tend to just hold/move-with-the-scene; budget animating-prop clips for any prop that needs to move on cue.

---

## 5. Other details to pin down

| # | Detail | Why it matters | Action |
|---|---|---|---|
| 1 | **Base-variant decision: THUDM 1.5-I2V vs CogVideoX-Fun** | Determines whether ControlNet/end-frame are ever available; changes fps (16 vs 8), frame count (81 vs 85), loop method | Decide before training. Recommend THUDM 1.5-I2V for v1; spin Fun as a parallel experiment if control is wanted |
| 2 | **LoRA ↔ inference-engine format compatibility** | A Passenger12138-trained LoRA must load in the production engine (diffusers pipeline *and/or* kijai wrapper). Key-naming differs between them | Verify on Week 2 diagnostic run; convert keys if needed |
| 3 | **Identity drift under motion** (CVPR 2026 "IPRO"; T2V sometimes beats I2V on Cog-1.5) | Faster/longer motion drifts the character off-model | Keep GIFs short, motion bounded; strong LoRA; latent loop closure already mitigates |
| 4 | **Inference VRAM / engine** | 5B I2V at 768×1360×81 ≈ 19–40 GB | Enable VAE tiling + CPU offload for 24–40 GB cards; A100 40 GB comfortable |
| 5 | **fps / frame-count / resolution lock** | 1.5-I2V is native **16 fps, 81-frame (8N+1), ≤1360 edge**; Fun is 8 fps / 85-frame / different buckets | Our dataset spec (768×1360, 16 fps, 49/81 frames) is correct **for THUDM 1.5-I2V only** — re-derive if we ever switch to Fun |
| 6 | **Multi-character placement at inference** | Spatial-token diffusion (two penguins blur into one) is set by the **first frame** layout + per-clip spatial captions | Keep a clear gap in the first frame; mirror training caption spatial anchors in the inference prompt |
| 7 | **Loop closure depends on single-pass** | Latent loop-closure injection works because 1.5 is single-pass; Fun/AR variants need a different loop method | If we move off 1.5, revisit loop strategy (see `research/Helios_Experimentation_Plan.md`) |
| 8 | **Inference prompt = training caption shape** | The model only responds well to prompts shaped like its training captions (identity anchor + motion) | Ship a prompt template that mirrors `Training_Runbook.md` Part C; don't free-form prompt |

---

## Sources
- [CogVideoX (zai-org/CogVideo) — official repo](https://github.com/zai-org/CogVideo)
- [kijai/ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) — start+end, ControlNet, Tora, context windows, Fun + LoRA support
- [aigc-apps/VideoX-Fun (CogVideoX-Fun)](https://github.com/aigc-apps/VideoX-Fun) and [CogVideoX-Fun-V1.5-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP), [CogVideoX-Fun-V1.1-5b-Pose](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose)
- [CogVideoX diffusers API + training docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
- [Passenger12138 CogVideoX1.5-5B-I2V LoRA trainer](https://github.com/Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train)
- Repo research: `research/wan_vs_cog.md` (I2V drift, IPRO CVPR 2026), `research/ViMax_Research.md` (no IP-Adapter/identity injection in the ecosystem), `research/LTX2.3_HDR_Research.md` (first/last-frame keyframing comparison)
