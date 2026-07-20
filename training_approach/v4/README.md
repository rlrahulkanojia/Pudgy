# Training Approach v4 (LTX-2.3) — Execution README

**Status: G1 ✅ passed · G2 ✅ passed · Phase 3 trained (evaluation pending)**

This document records what was actually built, measured, and learned executing
[`Training_Approach_v4.md`](./Training_Approach_v4.md) on a 2×A100-80GB box. It is the
practical companion to the plan: the plan says *what we intended*, this says *what happened*,
including the mistakes and the corrections.

---

## 1. TL;DR

We trained a working, controllable 2D-Pudgy video generation system on **LTX-2.3-22B**:

| Capability | Status | How |
|---|---|---|
| Flat-2D Pudgy style + identity | ✅ | style LoRA (config B) |
| Motion survives the LoRA | ✅ | verified — the HF#36 bug did **not** manifest |
| Beats v1 (CogVideoX) | ✅ | v1 characters dissolve mid-clip; v4 is stable |
| **Precise control** (draw edges → get on-model characters) | ✅ | **IC-LoRA** edge/Canny conditioning |
| Identity locked under motion | ✅ | IC-LoRA fixes the catastrophic "raccoon" failure |
| **Long clips (5 s / 10 s)** | ✅ | edge control removes the ~97-frame ceiling |
| Seamless loop | ✅ | endpoint keyframes + ffmpeg crossfade |
| High quality | ✅ | **1536×2688 (4.1 MP)** via ×2 spatial upscaler |
| Two characters, correct colours | ✅ | IC-LoRA + **colour-grounded prompts** |

**The single most valuable discovery:** the identity/colour problems were **not** solvable by any
inference knob (strength, CFG, negative prompt, LoRA stacking) — but they *are* solved by
(a) **IC-LoRA edge conditioning** for species/construction and (b) **two words of colour in the
prompt** for colour.

---

## 2. Models trained

All checkpoints, configs and training stats are in Azure (container `pudgy`).

| Run | What | Steps | Notes | Azure |
|---|---|---|---|---|
| `pudgy_lora_A_768` | style LoRA, attention-only | 2000 | loss 0.117 | `training_v4/pudgy_lora_A_768/` |
| `pudgy_lora_B_768` | style LoRA, attention + FFN | 2000 | loss 0.115 — **G1 golden** | `training_v4/pudgy_lora_B_768/` |
| `pudgy_ic_768` | **IC-LoRA** (edge/structure), rank 128 | 2000 | **the production model** | `training_v4/pudgy_ic_768/` |
| `pudgy_p3_768` | Phase 3 style LoRA (colour-grounded + Polly-rebalanced) | 2000 | eval pending | `training_v4/pudgy_p3_768/` |

W&B project `rlrahulkanojia/pudgy` — key runs: `l68akllr` (A_768), `7oh0da5v` (B_768),
`xm3lyj8g` (IC-LoRA), `f0bbtnbv` (Phase 3).

> All 8 checkpoints per run are retained (`keep_last_n: -1`) — fidelity often peaks mid-run.

---

## 3. Pipeline as executed (reproducible)

### 3.0 Environment
```bash
git clone https://github.com/Lightricks/LTX-2 && cd LTX-2 && uv sync
hf download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors \
    ltx-2.3-22b-distilled-1.1.safetensors ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir models/ltx-2.3
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir models/gemma-3-12b
uv add azure-storage-blob            # for the Azure upload we added
```
`transformers==4.57.6` resolved automatically (issue #116 needs this pin). ~117 GB of models.

### 3.1 Dataset (Step 2)
```bash
python prep_ltx.py --bucket 768x1344x49        # from iteration_2_v4/
```
298 windows @ **768×1344, 25 fps, 49 frames, silent** (÷32 dims, frames%8==1).
*Note: the plan's canonical bucket was 544×960; we upgraded to 768×1344 (~1 MP) because the
80 GB tier supports it.*

### 3.2 Preprocess
```bash
uv run accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  scripts/process_dataset.py <dataset.json> --resolution-buckets "768x1344x49" \
  --model-path .../ltx-2.3-22b-dev.safetensors --text-encoder-path .../gemma-3-12b \
  --skip-audio --decode
```
- `--skip-audio` — our clips are silent (matches `audio: null` in the config).
- **We deliberately omit `--lora-trigger`** — the captions already embed `pxngn0`/`plngn0`
  inline per-clip; prepending would double-tag and wrongly stamp Polly-only clips.

### 3.3 Train style LoRA
`configs/pudgy_v4_768.yaml` (A) / `pudgy_v4_B_768.yaml` (B) — dev checkpoint (#175),
rank 32, lr 5e-5, bf16, `audio: null`, validation dims == preprocess bucket (#155).

### 3.4 Edge-paired data + IC-LoRA (Phase 2)
```bash
uv run python scripts/compute_reference.py <clips_dir> -o <dataset.json> --override   # Canny
uv run accelerate launch ... scripts/process_dataset.py <dataset.json> ...            # + reference_latents
uv run python scripts/train.py configs/pudgy_ic_v4.yaml                               # IC-LoRA
```
`compute_reference.py` uses `cv2.Canny` and writes a `reference_video` column.
**Make the reference paths absolute** — the script writes basenames relative to the clips dir,
which the preprocessor may misresolve.

---

## 4. Results

### Gate G1 — style + identity + motion ✅
- **Golden: config B, step 2000** — the only checkpoint scoring identity 5/5 on all 6 held-out prompts.
- Scored by 6 parallel Claude vision judges on the v2 §5 rubric; config B avg **4.5** vs A **4.2**.
- **Motion survives** — walk/jump/hug robustly animate; the character-LoRA-kills-motion bug
  (HF #36) did **not** manifest on cartoon content.
- **Beats v1** decisively — v1 (CogVideoX) is soft/painterly and the character *dissolves and
  vanishes* mid-clip; v4 is stable flat-2D.

### Gate G2 — controlled, identity-locked, seamless loop ✅
Feeding a Canny edge map through the IC-LoRA produces a shot with:
locked construction (follows the edge frame-by-frame), locked identity, no drift, flat-2D style,
and a seamless loop (endpoint keyframes + crossfade). **Edge conditioning suppressed BOTH
failure basins at once.**

### Phase 3 — colour-grounded + Polly-rebalanced (trained, eval pending)
- All 298 captions colour-grounded (`plngn0, a pink penguin.`); solo-Polly oversampled 3× (33→99);
  364 total samples. Text embeddings recomputed with `process_captions.py` (video latents reused).
- ⚠️ **Prioritisation caveat:** Phase 3 retrained the **style LoRA**, but the production path uses
  the **IC-LoRA** — so its benefits do **not** automatically transfer. To reach production it would
  need an IC-LoRA retrain on the Phase 3 data. Its main intended benefit (colour) was also largely
  superseded by the free prompt fix (§5.3).

---

## 5. Key findings (the valuable part)

### 5.1 Two seed-dependent failure basins
The model has **two distinct failure modes**, both determined by the seed, both invisible if you
only test one seed:

| Basin | Symptom | Who it hits |
|---|---|---|
| **A — "raccoon"** | wrong species (ears + striped tail) | solo-Polly (~66% of seeds), two-char (~50%); **never** solo-Pax |
| **B — "3D claymation"** | right species, 3D render not flat-2D | ~40% of Pax seeds even at 49 frames |

On bad seeds the **whole scene** collapses together — including Pax, who is bulletproof solo.
G1 used seed 42, which dodged both, hiding the true rates.

> **Rule: never select a checkpoint or judge quality on a single seed.**

### 5.2 What does NOT fix them (all tested, all negative)
- **LoRA strength** 0.8→1.0→1.2: rescued 1 of 2 failing seeds. Not reliable.
- **Guided dev inference** (full dev model, CFG 4.0 + STG + negative "3d render"): **still 3D**
  on both Pax seeds, **still raccoon** on Polly. Guidance does not escape either basin.
- **LoRA stacking** (IC + style-B @0.7): still blue.
- **First-frame keyframe**: frame 0 is correct, reverts within ~16 frames.

The seed determines the basin **identically across distilled and guided pipelines** → the fix is
structural, not an inference knob.

### 5.3 What DOES fix them
1. **IC-LoRA edge conditioning → fixes species/construction.** Progression on the same Polly edge:
   step 250 = raccoon → step 1000 = penguin → step 2000 = penguin. It also suppresses the 3D basin.
2. **⭐ Colour-ground the prompt → fixes colour.** Clean A/B on the *same* IC-LoRA:
   - `"plngn0. 2d cartoon…"` → Polly renders **BLUE** ❌
   - `"plngn0, a pink penguin. 2d cartoon…"` → Polly renders **PINK** ✅

   Root cause: **zero of the 33 solo-Polly captions mention "pink"** (anti-entanglement design put
   appearance in the character bible, not the captions). The rare token alone carries almost no
   colour signal. Two words of prompt colour supply the missing grounding — **free, no retraining.**

### 5.4 Character size in frame drives identity quality
Canny edge detail scales with how much of the frame the character occupies. **Small/distant
characters → thin edges → weak structure lock → generic off-model penguins** (this is what ruined
the `reunion_walk` clip). Medium/close framing → strong lock.

> **Rule: pick source shots where characters are large in frame.**

### 5.5 Length: the ceiling is a *style-LoRA* limitation, not a model limitation
- **Unconditioned style LoRA:** flat-2D holds to ~97 frames (3.9 s); by 121 f it drifts to 3D, by
  249 f to the wrong species. (Isolation-tested: length alone causes it, not prompt complexity.)
- **With IC-LoRA edge control:** identity and style hold to **10 s (249 frames)** — verified.

### 5.6 Looping ≠ long-form
Building a "10 s clip" by looping a 49-frame edge produces the same 2-second action 5× —
repetitive and inauthentic, and it introduces **hard cuts every 49 frames** (characters teleport;
seam motion up to **528×** the median). Ping-pong (forward+reverse) edges reduce the seam
**105×** in absolute terms, but only smooth the repetition — they don't add motion.

> **For genuine long-form: concatenate *different* consecutive source clips.** PP-BF-Base has 20
> consecutive clips = 39 s of real evolving action. This produces real story beats — at the cost of
> scene cuts, since the source is 2-second scene-split data.

### 5.7 VRAM / training constraints (measured)
- **Gradient checkpointing is mandatory at 768×1344.** Disabling it OOMs the backward pass
  (~78 GB needed). With it, training sits at ~45 GB; IC-LoRA (reference tokens ~double the
  sequence) sits at ~62 GB.
- The idle VRAM **cannot** be usefully reclaimed: the frozen 22 B model dominates, and the only
  lever (checkpointing off) overflows.
- 2 GPUs buy **experiment throughput, not resolution** — run two configs concurrently, never
  data-parallel one run.

---

## 6. Production recipe

**Use the IC-LoRA path.** Text-only generation is seed-fragile; edge-conditioned generation is not.

### ⭐ Optimal settings (measured by parameter sweep — do not use defaults)

| Setting | Value | Evidence |
|---|---|---|
| **IC-LoRA checkpoint** | **step 1000**, *not* 2000 | jerk **0.20** vs 0.49 at step 2000 — **2.4× smoother**, identical identity. Step 750 is worse (0.55), so 1000 is a genuine sweet spot, not "earlier is better". |
| `conditioning_attention_strength` | **1.0** (smoothest) or **0.8** (most continuous) | cond 0.8 cuts held/frozen frames 2.5× (20.8% → 8.3%) at a small jerk cost |
| IC-LoRA strength | 1.0 | |
| Seeds | **1 is enough with edge control** | same edge at seeds 42/7/123 → all correct on-model; only backgrounds vary. The seed lottery affects **text-only** generation only. |

> Every clip generated before this sweep used step 2000 — i.e. sub-optimal motion smoothness.

### Source-material selection (the biggest quality lever)
Run `eval_v4/source_scan.py` — it scores every clip on text/bubbles, character size, and motion.
Measured on our 206 two-character clips: **only 17 are text-free with both characters.**
- **No speech bubbles** — text renders as gibberish. (Validated detector included.)
- **Characters large in frame** — `char_frac` ≥ ~0.14. Small characters → thin Canny edges →
  off-model output (this is what ruined `reunion_walk`).
- **Verify the source visually** — `report.csv`'s `has_internal_cuts` flag missed a clip that
  intercuts two shots; the model faithfully reproduced the cut. Trust your eyes over the flag.

```bash
python eval_v4/g2_generate.py \
  --ic-lora .../pudgy_ic_768/checkpoints/lora_weights_step_02000.safetensors \
  --edge-ref <canny_edge_video.mp4> \
  --prompt "pxngn0, a blue penguin, and plngn0, a pink penguin. 2d cartoon animation, cel-shaded, bold black outlines, flat pastel fills, no gradients, no shadows; <action>; static medium shot, eye level; soft even ambient light, flat." \
  --frames 49 --gpu 0 [--keyframe kf.png --loop] [--width 1536 --height 2688]
```

Checklist:
1. **Colour-ground every character** — `pxngn0, a blue penguin` / `plngn0, a pink penguin`.
2. **Drive with an edge track**; `num_frames` must equal the edge's frame count.
3. **Large characters in frame** in the source/edge.
4. **Avoid source scenes with speech bubbles** — text renders as gibberish.
5. **Long-form:** concatenate *different* consecutive clips' edges, don't loop one.
6. **HQ:** pass `--width 1536 --height 2688` — stage 1 runs at the trained 768×1344, then the ×2
   upscaler doubles it (4.1 MP).
7. **Loop:** `--keyframe <png> --loop` (frame 0 replaces, last frame guides) + ffmpeg crossfade.

Batch many clips with `eval_v4/g2_batch.py` — loads the 46 GB model once instead of per clip.

---

## 7. Evaluation tooling (in `eval_v4/`)

| Script | Purpose |
|---|---|
| `gen_eval.py` | text-to-video eval via the distilled pipeline + style LoRA |
| `guided_gen.py` | guided one-stage (dev + CFG + STG + negative prompt) |
| `g2_generate.py` | **IC-LoRA edge-conditioned generation** (+ keyframes/loop) |
| `g2_batch.py` | batch IC-LoRA generation, one model load |
| `metrics.py` | motion-survival (frame-diff + optical flow → ALIVE/WEAK/FROZEN) |
| `micro_anim_eval.py` | **micro-animation smoothness**: jerk, frozen %, flicker, loop-seam spikes |
| `make_montages.py` | contact sheets for visual/agent rubric scoring |

**Vision-judging without an API key:** spawn Claude sub-agents that `Read` the montage PNGs and
score the rubric. Six parallel judges scored the G1 sweep this way.

### ⚠️ Metric caveats (learned the hard way)
- **Motion metrics miss identity.** `reunion_walk` scored best on smoothness and was completely
  off-model. **Always visually verify before recommending.**
- **Whole-frame optical flow under-counts localized motion** — a flipper wave on a static camera
  reads "WEAK" even when clearly animating.
- **Flow-normalized metrics inflate on near-static clips.** A seam ratio of 35× corresponded to an
  absolute flow of 0.074 (i.e. fine), because the median was ~0.016. Read ratios alongside
  absolute values.

---

## 8. Known limitations

1. **Garbled text** — any source scene with speech bubbles produces gibberish ("xngno", "angea").
   Video models can't render legible text; composite text in post.
2. **Scene cuts in long clips** — inherent to 2-second scene-split source data. One unbroken >4 s
   shot needs longer source footage or an animatic.
3. **Polly is under-represented** — 33 solo-Polly vs 59 solo-Pax clips; 69% of data is two-char
   (diluted per-character signal). Mitigated by prompt grounding; the root fix is more solo-Polly data.
4. **Text-only generation is seed-fragile** (§5.1). Use edge conditioning, or generate several seeds
   and select.
5. **IC-LoRA is distilled-only** at inference (`ICLoraPipeline`), and has **no CFG/negative prompt**
   — the flat-2D look must come from the LoRA + positive prompt.
6. **No Looping Sampler in this repo** — the LTXV Looping Sampler is ComfyUI-only. Loops are done
   with endpoint keyframes + ffmpeg crossfade.

---

## 9. Where everything lives (Azure container `pudgy`)

```
training_v4/
├── pudgy_lora_A_768/  pudgy_lora_B_768/  pudgy_ic_768/  pudgy_p3_768/
│   ├── checkpoints/           # all 8 checkpoints + training-state (resume)
│   └── reports/               # training_config.yaml, training_stats.json
├── eval/
│   ├── G1_REPORT.md           # gate G1 verdict + rubric scores
│   ├── PROBE_INSIGHTS.md      # the failure-basin analysis (§5)
│   ├── metrics.csv            # motion-survival
│   ├── micro_anim.csv         # micro-animation smoothness
│   └── probe_montages/        # all visual evidence
└── inference/
    ├── BEST/                  # curated showcase clips
    ├── g1_sweep/  probes_identity/  phase2_ic_test/  phase2_guided_3d_test/
    ├── phase2_g2/             # G2 controlled + loop shot
    ├── phase2_long_hq/        # 5 s, 10 s, HQ 1536×2688
    ├── phase2_variations/     # 5 scenarios × 2 lengths, two-character
    └── prompt_sets/           # every prompt JSON, for reproducibility
```

Secrets live in `/workspace/pudgy_train.env` (never committed): `WANDB_API_KEY`,
`AZURE_STORAGE_CONNECTION_STRING`, `HF_TOKEN`.

**Trainer instrumentation we added** (`packages/ltx-trainer`): an `AzureConfig` block +
`src/ltx_trainer/azure_upload.py` — background-threaded, non-fatal uploads of checkpoints,
training-state and reports, hooked into the checkpoint-save funnel. wandb and checkpoint
save/resume were already native (`WandbConfig`, `CheckpointsConfig`).

---

## 10. Recommended next steps

1. **Retrain the IC-LoRA on the Phase 3 data** — this is the run that would actually reach the
   production path (colour-grounded captions + rebalanced Polly, inside the controlled pipeline).
2. **Dialogue-free source filtering** — scan source clips for speech bubbles and exclude them from
   showcase generation.
3. **Longer source tracks** — the single biggest quality unlock for long-form. A full-length
   animatic or longer shots would remove the scene-cut limitation entirely.
4. **More solo-Polly data** (turnarounds, expression sheets, solo clips) — the root fix for the
   remaining Polly fragility.
5. **Two-character work (Phase 3 of the plan)** — currently ~50% seed-reliable unconditioned, but
   reliable under edge control; more per-character data would firm this up.
6. **Bake-off vs v3 (AniSora)** on the same rubric, as the plan intends.
