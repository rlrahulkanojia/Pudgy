# GPU Handoff — `iteration_2_v4` dataset → LTX-2 LoRA training

**Audience:** the GPU box operator. Turns the curated **`iteration_2_v4`** dataset into a trained LTX-2 (v4) LoRA. Plugs into **[`Training_Approach_v4.md` §12](./Training_Approach_v4.md)** (LTX env, checkpoints, trainer config, gotchas) — read §12.0–12.3 alongside this; here we cover only **how to consume THIS dataset**.

> **Transfer note.** This doc reaches the box **via git** (it's in `training_approach/`). The **dataset itself is moved manually** — it lives outside the repo (`…/Pudgy/Data/iteration_2_v4/`, ~271 MB). Copy it to the box and set `DATASET=/abs/path/to/iteration_2_v4`. The transform script **`prep_ltx.py` ships inside that folder** (it's data-coupled).

> **What the dataset is:** 249 human-curated **scene-split clips** of Pax/Polly (from 70 source skits) at **native 1080×1920 / 24 fps**, each with a v4 §7.1 prompt, plus an object catalog. It is **not yet LTX-native** — the re-encode to 25 fps / ÷32 / silent / 49-frame windows is **Step 2** (deferred to the box on purpose: no GPU on the authoring machine, and to preserve max resolution).

---

## Dataset contents (`$DATASET/`)

| Path | What |
|---|---|
| `clips/` | 249 scene clips, native 1080×1920/24fps (the training source) |
| `prompts/` | 249 v4 JSON records (`rendered` = training caption; tokens `pxngn0`=Pax, `plngn0`=Polly) |
| `report.csv` / `report.md` | per-clip flags: `duration_s`, `too_short`, `has_internal_cuts`, `merged`, `unique_artifact_present`, `pdf_validation`, … |
| `artifacts/`, `catalog.json` | object/character crop catalog (1,194 labels) — reference only, **not** training input |
| `character_bible.json` | canonical Pax/Polly data + trigger tokens |
| **`prep_ltx.py`** | **the Step-2 transform** (re-encode + window → `ltx_dataset.json`) |
| `dataset.json` | caption↔native-clip map (superseded by `ltx_dataset.json` after Step 2) |

---

## Step 0 — Prerequisites (see v4 §12.0–12.1)
Stand up the LTX-2 env per **v4 §12.1**: clone `Lightricks/LTX-2`, `uv sync`, `hf auth login`, download `ltx-2.3-22b-dev` + `-distilled-1.1` + `spatial-upscaler` + `google/gemma-3-12b`. Pin **`transformers==4.57.6`** (v4 #116). CUDA 13+, 80 GB tier (train bf16). `ffmpeg`+`ffprobe` on PATH (Step 2 needs no CUDA).

## Step 1 — Place the dataset
Manually copy `iteration_2_v4/` to the box; `export DATASET=/abs/path/to/iteration_2_v4`.

## Step 2 — Build the LTX-native training set  ⭐ (the deferred re-encode)
```bash
cd "$DATASET"
python prep_ltx.py --bucket 544x960x49          # canonical (~2 s windows)
# 80 GB box, higher res:   python prep_ltx.py --bucket 768x1344x49
# max temporal smoothness: add  --exclude-internal-cuts
```
Per kept clip: re-encode to the bucket res (dims ÷32), **25 fps, silent**, then window into non-overlapping **49-frame** segments (`frames % 8 == 1`); each window inherits its scene caption. Writes:
- `ltx_clips/` — the 544×960×49 silent training windows
- `ltx_dataset.json` — `[{ "caption": …, "video": "<abs path>" }]`

**Expected yield (canonical bucket):** ~**303 windows** from 196 clips (~**280** with `--exclude-internal-cuts`). Sub-window clips auto-skip.

### Which clips to include — the flags (`report.csv`)
- **`too_short` (53, <2.04 s):** auto-dropped by windowing. Rescue ~1 s ones with `--include-short` + a `…x25` bucket if desired.
- **`has_internal_cuts` (26 merged):** a window may straddle a hard cut → can teach discontinuity. **Recommend** an A/B: train with vs `--exclude-internal-cuts`, compare motion at G1 (~23 windows delta).
- **`pdf_validation == conflict` (13):** finished video diverged from storyboard; captions are watch-first ground truth → **keep**. Only matters if you want captions to match *intended* story beats.
- **other-character clips (6, `unique_artifact_present`):** contain non-lead penguins (crowds) — fine for style, weaker pure-identity signal.

## Step 3 — Preprocess (latents + text embeddings) — v4 §12.2
```bash
cd packages/ltx-trainer
uv run python scripts/process_dataset.py "$DATASET/ltx_dataset.json" \
    --resolution-buckets "544x960x49" \
    --model-path        ../../models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path ../../models/gemma-3-12b \
    --lora-trigger "pxngn0" --decode
```
Keep the bucket **identical** to Step 2 (#155 latent-shape trap). Captions carry both `pxngn0`/`plngn0` inline; `--lora-trigger` is just the primary tag. Encoder unloads after this (v4 Hardware notes).

## Step 4 — Train — v4 §12.3
`cp configs/t2v_lora_low_vram.yaml configs/pudgy_v4.yaml`, set per **v4 §12.3**: `model_path`=**dev** (#175), `training_mode: lora`, rank 32/α 32, `learning_rate 5e-5`, `mixed_precision_mode: bf16`, `quantization: null`, `load_text_encoder_in_8bit: false`, `preprocessed_data_root`=Step-3 `.precomputed`, `validation.video_dims:[544,960,49]` + `frame_rate 25.0` + `generate_audio: false`, `checkpoints.keep_last_n: -1`. Launch:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py configs/pudgy_v4.yaml
# 2 GPUs → two concurrent configs (v4 §12.3 sweep mode); never data-parallel one run
```

## Step 5 — Evaluate / golden — v4 §12.4 + §8
Held-out set → **dev** pipeline (30–50 steps, CFG ~3–4) → score the **v2 §5 rubric** per checkpoint → pick golden by eye+metrics. **G1** = beats the v1 CogVideoX run *and* motion survives the style LoRA.

---

## Report-back checklist
- [ ] Step 2 → N windows in `ltx_clips/` + `ltx_dataset.json` (bucket + internal-cuts in/out)
- [ ] Step 3 → `.precomputed/{latents,conditions}` populated; decode failures
- [ ] Step 4 → trains bf16 @ 80 GB; time/1000 steps; **video** loss curve; per-interval validation videos; **does motion survive the LoRA?** (v4 §10.5)
- [ ] Step 5 → rubric vs v1 → **G1**

## Gotchas carried from v4 §12 (don't relearn)
- Pin `transformers==4.57.6`; use `uv run`. Train on **dev**, iterate on **distilled-1.1** (avoid `distilled-fp8`).
- fps=**25**, dims **÷32**, frames **8N+1** — all enforced by `prep_ltx.py`.
- `enhance_prompt` **OFF** at inference — captions are already distribution-matched.
- Open base-level risks to probe first (v4 Phase 0): VAE flat-fill grid (#202) and character-LoRA-kills-motion (HF #36).
