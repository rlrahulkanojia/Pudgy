# Pudgy Penguins — Training Approach v3 (AniSora)

**Supersedes** the "Wan 2.2 A14B primary / AniSora in parallel" framing of
[`Training_Approach_v2.md`](../v2/Training_Approach_v2.md). v3 **commits to AniSora V3.2 as the
primary base** and makes concrete the identity-pinning mechanism v2 only sketched.

**Goal (unchanged from v1/v2):** 2D cartoon animation of **Pax** (blue) and **Polly** (pink) —
flat pastel, thick black outlines — with **exacting character consistency, robustness, and high
image quality**. Budget uncapped.

**What changed vs v2:** v2 correctly diagnosed the problem (pipeline *shape*, not tuning — see
[`FINDINGS.md`](../FINDINGS.md)) and named the fix (decouple identity from motion via
keyframe/FLF2V interpolation on a flow-matching, 8× VAE base). It left the base as an open A/B
between **Wan 2.2 A14B** and **AniSora**. The Phase-0 diagnostics + base-model exploration now
resolve that A/B: **AniSora V3.2 is Wan 2.2 A14B, but anime-native, RLHF-quality-tuned, and it
ships the keyframe-interpolation primitive v2 wants — natively, in the base.** v3 is therefore
"v2, with the base decided and the identity-pinning made native."

---

## 1. Approach in one line

**AniSora V3.2** (Bilibili Index-anisora — a Wan 2.2-A14B MoE fine-tune on 10M+ curated anime
clips, 8× VAE, Apache-2.0) as the base, driven by its **native arbitrary-keyframe interpolation
+ spatiotemporal motion masks** to pin Pax/Polly identity at fixed temporal positions, with a
**character/style LoRA** (trained via musubi-tuner on the Wan 2.2 base) layered on for the exact
flat-pastel / thick-outline look.

AniSora pins **pose and position**; the LoRA pins the **Pudgy style**. Neither alone is enough —
that split is the whole thesis.

---

## 2. Why AniSora over the v2 "Wan 2.2 A14B" default

AniSora V3.2 **is** Wan 2.2-A14B under the hood (same MoE architecture: high-noise + low-noise
experts, same 8× VAE, same 16 fps / F=8x+1 frame grid, Apache-2.0). So everything v2 liked about
A14B is inherited *for free*, and AniSora adds three things on top:

| Axis | Plain Wan 2.2 A14B (v2 primary) | **AniSora V3.2 (v3 primary)** |
|---|---|---|
| Style prior | General video; anime is out-of-distribution | **Anime/2D-native** — trained on 10M+ curated anime clips; closest prior to flat-pastel + thick-outline |
| Identity pinning | FLF2V only via community ComfyUI workflows | **Native** arbitrary-frame interpolation: `image_path@@prompt&&position`, position ∈ [0,1] (0=first, 0.5=mid, 1=last) — multiple keyframes at any temporal fraction |
| Spatial control | None native | **Native spatiotemporal motion masks** (`anisora_anymask`): lock a character region, or re-inject frames to repair flicker/shake |
| Quality tuning | Base only | **RLHF via AnimeReward + GAPO** (arXiv 2504.10044) — reward-tuned specifically to cut *motion distortion and flickering* and improve appearance/consistency on anime |
| Extra primitives | — | Character **360° 3D rotation**, video **style transfer**, 90p→720p/1080p **super-res** |

**Bottom line:** AniSora is a strict superset of the v2 A14B plan for this style. Choosing it
costs nothing v2 valued and adds the anime prior + the exact identity-pinning primitive v2 was
going to have to bolt on via community FLF2V workflows.

---

## 3. Pros over v2 (concrete)

1. **The identity-pinning primitive is native, not bolted-on.** v2's decoupled pipeline
   (§2 of v2) depended on community ComfyUI FLF2V workflows on raw Wan. AniSora ships
   **arbitrary-frame interpolation in the base** via a documented prompt format
   (`img@@prompt&&position`). We feed human-QC'd Pax/Polly keyframes at t=0/0.5/1 directly — the
   v2 architecture, with the risky integration step removed.

2. **Anime-native base → less work for the LoRA.** On CogVideoX (v1) and general Wan (v2), the
   LoRA had to teach the model *what 2D flat-pastel even is*. AniSora already renders that
   distribution, so the LoRA only has to bind **Pax/Polly identity**, not the whole style — a
   much easier, less overfit-prone target on 75 clips.

3. **The existing 75-clip dataset already matches AniSora's native format.** ⚠️ **Corrects a v2
   assumption:** [`GPU_HANDOVER.md`](../docs/GPU_HANDOVER.md) §Phase-1 says "AniSora 24 fps —
   resample from the 24 fps source." The actual V3/V3.2 inference scripts run at **16 fps with
   F=8x+1 frames**. Our dataset is **16 fps, 33 frames (33 = 8×4+1)** — it *already* satisfies
   AniSora's grid. **No fps re-derivation, no re-tiling from source needed** for a first LoRA;
   only the manifest format changes. This deletes an entire painful step (and dodges the
   discrepancy that `prep/build_dataset.py` still hard-codes the stale 49/81-frame spec, and that
   the `split6.py`/`assemble3.py` tiling scripts are missing from the repo).

4. **The 432×768 rotary cap disappears.** v1's hard ceiling was CogVideoX1.5's rotary grid
   (`Expected size 63 but got 48` — see [`FINDINGS.md`](../FINDINGS.md) §1). AniSora/Wan has no
   such cap; portrait is a `--size 720*1280` swap. We recover real portrait resolution, which was
   half of why v2 wanted to migrate at all.

5. **RLHF quality tuning targets exactly our failure modes.** v1's outputs flickered and the
   subject vanished mid-clip. AnimeReward/GAPO is reward-tuned to reduce *flickering and motion
   distortion* — a quality gain v2's plain-A14B path didn't include.

6. **Two-expert LoRA maps to the identity/motion split we already wanted.** v2 §1.3 proposed
   low-noise=identity, high-noise=motion on A14B. AniSora V3.2's MoE gives us exactly those two
   experts — the split is architectural, not something we impose.

7. **A built-in eval reward model.** The `reward/` dir ships AnimeReward + `reward_infer.py`.
   We can score checkpoints against a learned anime-quality reward *in addition to* the v2 §5
   rubric — a cheaper, automatable golden-checkpoint selector than eyeballing montages.

---

## 4. Expected improvements on top of the v2 outcome

Measured against v1's documented failure (LoRA learned style + both characters in the first ~5
frames, then **drift → subject vanishes by ~f24**; [`FINDINGS.md`](../FINDINGS.md) §3) and the v2
rubric ([`Training_Approach_v2.md`](../v2/Training_Approach_v2.md) §5):

| Rubric dimension | v1 result | v3 expected | Mechanism |
|---|---|---|---|
| **Character identity** | on-model early, **vanishes mid-clip** | identity held across the whole clip; no vanish | keyframes pin t=0/0.5/1; drift bounded to interpolation, can't accumulate |
| **Line & color quality** | soft under motion | crisper, on-model flat fills | anime-native base + 8× VAE (VAE already proven non-limiting, ~38 dB PSNR) + RLHF anti-flicker |
| **Motion robustness** | door melts to a slab, bodies smear | coherent bouncy motion | Wan 2.2 MoE motion prior >> CogVideoX-5B; RLHF cuts distortion |
| **Temporal stability** | flicker + identity wobble | reduced flicker | `anymask` frame re-injection + AnimeReward tuning |
| **Two-character (Pax+Polly)** | Polly faint/absent | both held, no blending | per-character keyframes + spatial masks (later: Phantom/VACE ref-conditioning per v2 §3.4) |
| **Clip length** | 2.06 s (33f) | up to ~5 s (81f @ 16fps) | AniSora native F=8x+1 up to 81 frames |

**Net expectation:** v3 should clear **Gate G1** (beat v1 on the rubric) and, because the
identity-pinning is native rather than integration-dependent, reach **Gate G2** (end-to-end
decoupled shot, locked identity, no mid-clip drift) with materially less bring-up risk than the
v2 Wan-FLF2V path.

---

## 5. Engineering plan (grounded in the current repo)

The current `finetune/` path is a HuggingFace-diffusers CogVideoX trainer and **cannot be
reused** — AniSora/Wan are not first-class in diffusers. Concrete deltas:

### 5.1 Trainer swap
- **Drop:** `finetune/train_cogvideox_image_to_video_lora.py`, `finetune/data/`, and the
  `accelerate launch` flag surface in `finetune/scripts/train_pudgy_lora.sh`.
- **Adopt:** **[kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)** (primary — the
  standard Wan 2.1/2.2 LoRA trainer; TOML-config driven, supports `fp8_base`, `fp8_scaled`,
  `blocks_to_swap` for memory) or **[diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)**
  (alternative). ⚠️ **AniSora's own repo does NOT ship V3/V3.2 training code** — only V1 (SAT-based
  full fine-tune on CogVideoX-5B) and V2 have trainers; V3/V3.2/anymask/rl dirs are
  **inference-only**. So the LoRA route is community-tooling on the underlying Wan 2.2 base, which
  is architecture-identical to V3.2.
- ⚠️ **MoE LoRA gotcha:** on the V3.2 MoE, LoRA training/extraction on the **low-noise expert
  misbehaves** ([nomadoor HF discussion](https://huggingface.co/nomadoor/diff_lora-r8_anisora_wan2.1_i2v/discussions/1)).
  Plan **per-expert LoRAs**; if the low-noise expert won't train cleanly, fall back to a
  Lightning-LoRA workaround on it (community pattern) and train only the high-noise expert
  ourselves initially.

### 5.2 Dataset repackage (pixels stay — big simplification vs v2)
- Keep the **existing 75 clips as-is: 768×1360, 16 fps, 33 frames** — already AniSora-native (see
  §3 pro #3). No re-derivation from 24 fps source.
- Convert `metadata.json` (`{file_path,text,type}`) → musubi's TOML dataset config + `.txt`
  caption sidecars. The repo already emits `captions/*.txt` and `prompts.txt`, so this maps
  cleanly.
- Adopt **rare-token identity captions** (Phase 0.2 strategy — trigger token + describe only the
  variable content), not the dense identity descriptions v1 used.

### 5.3 LoRA config (from v2 §4, confirm by A/B)
| Knob | Value | Note |
|---|---|---|
| Method | LoRA (not full FT) | full FT overfits on 75 clips |
| Target | all linear incl. MLP/FFN | v1 was attention-only — the classic miss |
| Rank / α | r = 16–32, α = 2r | regularization > capacity on small data |
| Experts | two LoRAs: low-noise = identity/texture, high-noise = motion | mind the low-noise gotcha (§5.1) |
| LR / warmup | ~8e-5–1e-4 / ~100 steps | Wan community standard |
| Timestep shift | matched to resolution (`--sample_shift 5` is the inference default) | governs global temporal structure |
| VAE | AniSora 8× (Wan 2.1-class) | proven non-limiting on our style |

### 5.4 Inference / keyframe pipeline (new)
- The AniSora entry point is `anisoraV3.2/generate-pi-i2v-any.py` with prompt lines
  `image_path@@prompt&&image_position`. Wire a small driver that: (a) takes human-QC'd Pax/Polly
  keyframes, (b) places them at temporal fractions, (c) calls AniSora with `--task i2v-A14B
  --ckpt_dir_highname high_noise_model --ckpt_dir_lowname low_noise_model --size 720*1280
  --sample_steps 8 --sample_guide_scale 1 --sample_shift 5 --frame_num 81`.
- For localized/2-character control, use `anisora_anymask/generate-pi-i2v-any-mask1_spa.py`
  (spatial) and `...mask1.py` (temporal re-injection).

### 5.5 Eval tooling (mostly reusable)
- **Reuse:** `inference/training_report.py` (log/frame metrics — model-agnostic) and the
  multi-VAE `training_approach/scripts/vae_roundtrip.py` (already has a Wan-VAE `REGISTRY`).
- **Rewrite:** `inference/eval_pudgy_lora.py` + `compare_checkpoints.py` are hard-wired to
  `CogVideoXImageToVideoPipeline` and the 432×768 rotary clamp — replace pipeline construction
  with AniSora's; the resolution cap logic is deleted.
- **Add:** AniSora `reward/reward_infer.py` as an automated checkpoint scorer alongside the v2 §5
  rubric.

---

## 6. Risks & mitigations (v3-specific, on top of v2 §8)

| Risk | Evidence | Mitigation |
|---|---|---|
| **No official V3.2 training code** | AniSora repo V3/V3.2 dirs are inference-only | Train the LoRA on the Wan 2.2-A14B base via musubi-tuner (architecture-identical); AniSora weights load as the base. |
| **Low-noise expert LoRA misbehaves** | nomadoor HF discussion; kijai WanVideoWrapper #1316 | Per-expert LoRAs; high-noise first; Lightning-LoRA fallback on low-noise. |
| **VRAM: 720p A14B is heavy** | issue #61 — OOMs even on 8×4090 24 GB w/ FSDP+offload; issue #51 — ~minutes for 5 s 832×480 on A100 | Our box is **A100-40 GB** → fp8 + `blocks_to_swap`; prefer an **80 GB H100**. For inference under 24 GB use [QuantStack GGUF](https://huggingface.co/QuantStack/Index-Anisora-V3.2-GGUF) (Q6 12.5 GB) via ComfyUI-WanVideoWrapper. Prototype at 480p, final at 720p. |
| **Fragile deps** | issues #52 (`fastvideo`), #31 (`req_fastvideo.txt` conflict), #29 (pyav), #21 (flash-attn build) | Pin a known-good env early; budget setup time; keep the CogVideoX env as fallback. |
| **"360p in 8 s" not reproducible** | issue #51 (A100 = minutes, min size 832×480) | Don't plan around the marketing throughput number; benchmark real wall-clock on our hardware first. |
| **Portrait / min-resolution limits** | issue #17 (V1 horizontal-only; V2+ multi-res), min ~832×480 | Verify 768×1360 renders cleanly before committing; fall back to a supported portrait size and letterbox if needed. |
| **License clarity for V3.2** | repo top-level Apache-2.0; no per-dir V3.2 LICENSE file found | Confirm Apache-2.0 covers V3.2 weights before commercial ship. |
| **RLHF on V3.2 unconfirmed** | issue #57 unanswered; only `5B_RL` CogVideoX is confirmed RLHF | Treat RLHF quality gains as "likely, not guaranteed" for V3.2; the AnimeReward *reward model* is usable regardless as an eval. |

---

## 7. Phased plan (deltas on top of v2's phases)

| Phase | v3 action | Gate |
|---|---|---|
| **0 — de-risk** | (a) Run the multi-VAE round-trip incl. the AniSora/Wan 8× VAE (confirm outline fidelity — expected ✅). (b) Stand up a **known-good AniSora V3.2 inference env** and reproduce a vanilla I2V + one keyframe-interpolation sample at 720×1280 on our hardware; record wall-clock + VRAM. | **G0:** env works; VAE ✅; real perf/VRAM known. |
| **1 — corrected baseline** | Repackage the 75 clips to musubi TOML (no re-tiling); train a high-noise identity/style LoRA on the Wan 2.2-A14B base, rare-token captions, all-linear+MLP, r=16–32. Score vs v1 on the rubric + AnimeReward. | **G1:** beats the v1 CogVideoX run. |
| **2 — decoupled pipeline** | Wire the keyframe driver (`img@@prompt&&position`): human-QC'd Pax/Polly keyframes at t=0/0.5/1 → interpolate → full shot. Add spatial masks for 2-character shots. | **G2:** end-to-end shot, locked identity, no mid-clip drift. |
| **3 — data + scale** | As v2 §3: turnarounds/color codes → stronger keyframes; stills → joint image+video; single-action clips + more skits; evaluate anymask vs Phantom/VACE for 2-shots. Only here does re-deriving clips (e.g. longer, 81-frame) become worthwhile. | — |
| **4 — eval/select** | v2 §5 rubric on a fixed held-out prompt set + AnimeReward per checkpoint; pick golden by both. | — |

---

## 8. Immediate next actions

1. **Stand up an AniSora V3.2 inference env** (pin fastvideo/flash-attn/pyav) and reproduce one
   I2V + one keyframe-interpolation sample at 720×1280 on the A100-40 GB (fp8 + block-swap) —
   record VRAM + wall-clock. *This is the single highest-ROI de-risk; it validates the whole base
   choice before any training.*
2. **Add the AniSora/Wan 8× VAE to the multi-VAE round-trip** (`training_approach/scripts/vae_roundtrip.py`)
   — cheap confirmation of outline fidelity on the actual target VAE.
3. **Repackage the 75 clips to musubi TOML** (no re-derivation — they're already 16fps/33f) with
   rare-token captions.
4. **Train the first high-noise identity LoRA**; score vs v1 (Gate G1).
5. **Confirm the low-noise-expert LoRA path** early (it's the known MoE gotcha).

---

## 9. What v3 inherits and what it corrects

- **Inherits from v2:** the decouple-identity-from-motion thesis, the "data + objective beat
  parameters" principle, the §5 rubric, the client-asset ranking (§6), and the Phase-0 VAE
  finding (8× VAE is not the ceiling).
- **Corrects in v2:** the base is now decided (**AniSora V3.2**, not an open Wan/AniSora A/B); the
  identity-pinning is **native** (not community FLF2V); and the dataset is **already
  format-compatible** (v2/handover's "AniSora = 24 fps, re-derive from source" was wrong — it's
  16 fps / F=8x+1, which our 33-frame clips already satisfy).

---

*References: AniSora repo https://github.com/bilibili/Index-anisora · weights
https://huggingface.co/IndexTeam/Index-anisora · papers arXiv 2412.10255 (AniSora) & 2504.10044
(AnimeReward/GAPO) · GGUF https://huggingface.co/QuantStack/Index-Anisora-V3.2-GGUF · ComfyUI
https://github.com/kijai/ComfyUI-WanVideoWrapper · trainers
https://github.com/kohya-ss/musubi-tuner · https://github.com/tdrussell/diffusion-pipe · MoE-LoRA
note https://huggingface.co/nomadoor/diff_lora-r8_anisora_wan2.1_i2v/discussions/1. See also
[`FINDINGS.md`](../FINDINGS.md), [`base_model_exploration.md`](../base_model_exploration.md),
[`Training_Approach_v2.md`](../v2/Training_Approach_v2.md).*
</content>
</invoke>
