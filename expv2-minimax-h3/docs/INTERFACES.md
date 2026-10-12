# expv2 — file layouts and interfaces

Every expv2 script reads and writes the layouts below, and nothing else. Paths are written
relative to the roots in [`../expv2_common.py`](../expv2_common.py): `CORPUS` (the prep
output), `WORK` (box scratch), `CACHE`, `RUNS` and `EVAL`. `configs/env.sh` sets the box
defaults.

---

## 1. Corpus — `CORPUS = Data/processed/expv2_h3/` (written by `prep/prep_expv2.py`)

| Path | Content |
|---|---|
| `clips/<clip_id>.mp4` | one training clip: **N + 1 frames** (the last frame duplicated once, `LOADER_TAIL`; see MODEL_NOTES 2.17), 768², 24 fps, `libx264rgb -qp 0`, `rgb24`, no audio stream |
| `png/<clip_id>/0000.png …` | the same clip, exactly **N** RGB frames. All scoring reads these, never the mp4 |
| `metadata_f22.csv`, `metadata_f39.csv`, `metadata_f56.csv` | **training clips only**, one CSV per bucket (only f22 is cached and trained by default, PLAN §5.1). Columns `video,prompt,input_audio`. `video` and `input_audio` are both `clips/<clip_id>.mp4` (no audio stream → DiffSynth's `--silent_on_missing_audio` fallback) |
| `holdout/{clips,png}/…`, `holdout/metadata_f22.csv` | the 23 Tier-1 held-out clips: angle `QF2_R`, ground white, zoom 1.00, f22, true captions. CSV paths are relative to `holdout/` |
| `calib/{clips,png}/…`, `calib/metadata_f22.csv` | Tier-1 calibration clips: angle `FRONT`, white, zoom 1.00, f22, every trained label with a usable FRONT source (23 built). Trained sources, re-rendered at a fixed zoom, so a calib clip can be byte-identical to the training clip of the same id. CSV paths are relative to `calib/` |
| `vae_gate/png{768,512}/<id>/…` | G0.1 clips: Pax and Polly × {angry f39, waving f22, walking tiled f22} × {white, mint}, FRONT, zoom 1.00, plus `flatfill_mint_f22`. One resample from the 1080² source at each size |
| `bank/png/<bank_id>/…`, `bank/index.json` | the **real-clip bank**: every usable source (training and held-out) once, on white at zoom 1.00, at the longest legal rung (22/39/56) its usable length reaches. Running is included at its 17 frames, flagged `untrained`. `bank_id = <char>_<label>_<angle>`. Used by G0.5, Tier-2 1-NN and the §8 head-to-head |
| `start_frames/<char>_<label>_<angle>__<ground>.png` | frame 0 at 768², zoom 1.00, for every (character, label incl. running, angle ∈ {FRONT, QF2_R}, ground ∈ {white, lavender, skyblue}) whose source exists |
| `manifest.json` | see below |
| `eval_items.json` | every generation item of every suite (below) |
| `tier1_items.json` | every holdout and calibration clip with its candidate captions (below) |

### `manifest.json`

```json
{
  "created": "...", "git_sha": "...", "prep_args": {...},
  "counts": {"<label>": {"usable_f22": 18, "holdout": 2, "train_sources": 16,
                          "f22_clips": 64, "f39_sources": 16, "f56_sources": 0}, "...": {}},
  "clips": [{"clip_id": "...", "split": "train|holdout|calib", "char": "Pax", "label": "angry",
             "kind": "expression", "angle": "QF1_L", "ground": "mint", "zoom": 0.75,
             "frames": 22, "source": "iteration_4/...mov", "source_md5": "...",
             "usable_frames": 40, "held_frame": false, "tiled": false, "truncated": true,
             "caption": "..."}],
  "excluded": [{"source": "...", "md5": "...", "reason": "..."}],
  "sources": [{"char": "...", "label": "...", "angle": "...", "source": "...", "md5": "...",
               "usable_frames": 40, "first_gap": null, "split": "train|holdout|start_only"}]
}
```

### `eval_items.json`

```json
[{"item_id": "core__pax_angry_QF2_R__white_f22", "suite": "core|native|gb|probe",
  "char": "Pax", "label": "angry", "kind": "expression", "angle": "QF2_R", "ground": "white",
  "frames": 22, "seeds": [42, 43, 44], "start_frame": "start_frames/pax_neutral_QF2_R__white.png",
  "start_source_label": "neutral", "caption": "...", "truncated": true}]
```

Suites (PLAN §7.4):
- `core`: 13 labels × 2 characters × {FRONT, QF2_R}, white, f22, seeds 42–44 → 156 clips;
- `native`: angry, laughing, blushing at f39 and neutral at f56, QF2_R, white, seeds 42–44 → 24;
- `gb`: angry, waving, jumping × 2 characters × {lavender, skyblue}, QF2_R, f22, seeds 42–44 → 36;
- `probe`: walking, waving, sitting, jumping × 2 characters, QF2_R, white, f22, seed 42 → 8.

**Start frames:** every item's `start_frame` is the character's **neutral** source at that
angle and ground (`start_frames/<char>_neutral_<angle>__<ground>.png`, `"start_source_label":
"neutral"`), v8's rule, so the label signal is in the prompt only. The per-label files in
`start_frames/` are kept for analysis.

`truncated` follows v8's training-caption rule against the **label's** own source at that
angle: true when that source's usable length exceeds `frames`. Running is never truncated
(17 < 22).

### `tier1_items.json`

```json
[{"clip_id": "...", "split": "holdout|calib", "char": "Polly", "label": "waving",
  "kind": "motion", "angle": "QF2_R", "frames": 22, "png_dir": "holdout/png/<clip_id>",
  "truncated": true,
  "candidates": {"walking": "<caption>", "waving": "<caption>", "sitting": "...", "jumping": "..."}}]
```

Candidates are every **trained** label of the clip's kind (8 expressions, or 4 motions).
Only the label clause differs between them; the truncation suffix is held at the true
clip's value.

---

## 2. Box scratch — `WORK = /workspace/expv2/`

| Path | Written by | Content |
|---|---|---|
| `cache/train/f22/0/*.pth`, `f39/`, `f56/` | `configs/cache.sh` (DiffSynth stage 1) | training cache. Stage 2 points `--dataset_base_path` at `cache/train/`, and DiffSynth's recursive `.pth` search merges the three buckets |
| `cache/eval/<set>/…` | `eval/generate.py --encode`, `eval/dcls.py --encode` | pass-1 encodings (Qwen3-VL + VAE) for two-pass evaluation. Never under `cache/train/` |
| `output/<run>/step-<samples>.safetensors` | `configs/train.sh` | DiffSynth checkpoints. **The number counts samples, not optimizer steps**: optimizer step = samples / 8 (`C.opt_step`) |
| `output/<run>/loss.csv`, `training_args.json`, `train.log` | DiffSynth / `train.sh` | logs |

Run names: `expv2-h3-s12`, `expv2-h3-s222` (pilots), `expv2-h3-main`, `expv2-h3-d` (arm D).

**Checkpoint specs** (every eval script's `--ckpt`): `base` (no LoRA) · `<run>@<optstep>`
(e.g. `expv2-h3-main@250` → `output/expv2-h3-main/step-2000.safetensors`) · `<run>@all` ·
`<run>@last` · a path to a `.safetensors`. A ckpt's **name** in output paths is `base` or
`<run>@<optstep>`.

---

## 3. Evaluation output — `EVAL = /workspace/expv2/eval/` (mirrored to Azure `expv2-minimax-h3/eval/`)

| Path | Written by |
|---|---|
| `phase0/pins.json` | `setup/fetch_models.py` (+ DiffSynth SHA, patches applied) |
| `phase0/vae_gate/report.json`, `sheets/*.png`, `decoded/…` | `eval/vae_gate.py` (G0.1) |
| `phase0/pipeline_checks/report.json`, `dumps/…` | `eval/pipeline_checks.py` (G0.2 a–h) |
| `phase0/source_baseline.json` | `eval/source_baseline.py` (G0.5, CPU) |
| `phase0/mem_probe.json` | `eval/mem_probe.py` (G0.6) |
| `tier1/<ckpt>.json` | `eval/dcls.py` (its `summary` also carries a `per_clip` list, for the bootstrap) |
| `tier1/calibration.json` | `eval/dcls.py --calibrate` (§6.3: the two frozen sigmas + the shift choice) |
| `tier0/<run>.json` | `eval/tier0.py` (CPU) |
| `gen/<ckpt>/<suite>/<item_id>__s<seed>/0000.png …`, `meta.json`, `preview.mp4` | `eval/generate.py` |
| `scores/<ckpt>/<suite>.json` | `eval/score.py` (CPU + CLIP/DINOv2 on GPU if present) |
| `reports/winner.json` | `eval/select_winner.py` (§7.3) |
| `reports/h2h_v8.json` | `eval/h2h_v8.py` (§8) |
| `reports/verdict.json`, `reports/verdict.md` | `eval/verdict.py` (§9) |

### `tier1/<ckpt>.json`

```json
{"ckpt": "expv2-h3-s12@122", "sigmas": [0.5, 0.875], "draws": 2, "noise_seed": 1234,
 "clips": [{"clip_id": "...", "split": "holdout", "kind": "motion", "label": "waving",
            "losses": {"waving": [[l_s0d0, l_s0d1], [l_s1d0, l_s1d1]], "walking": [[...]]},
            "losses_unmasked": {"...": [[...]]}}],
 "summary": {"expression": {"n": 16, "acc": 0.81, "mean_margin": 0.0042, "below_base": 15},
             "motion": {"n": 7, "acc": 0.71, "mean_margin": 0.0031, "below_base": 6}}}
```

Per clip: `loss_true` = mean over sigmas × draws of the true candidate; `margin` = mean of the
wrong candidates' losses − `loss_true` (positive = the true caption denoises best);
predicted label = argmin of the mean loss. `below_base` counts clips whose `loss_true` is
below the base model's at the same sigmas and draws (needs `tier1/base.json`).

### `scores/<ckpt>/<suite>.json`

```json
{"ckpt": "...", "suite": "core",
 "clips": [{"item_id": "...", "seed": 42, "char": "...", "label": "...", "angle": "...",
            "nn_label_clip": "angry", "nn_label_dino": "angry",
            "motion_energy": 0.0, "frozen_fraction": 0.0, "twos_index": 0.0,
            "zoom": 0.0, "corner_drift": 0.0, "ground_spread": 0.0, "late_identity": 0.0,
            "saturation": 0.0, "fill_flatness": 0.0,
            "ff_psnr": 0.0, "ff_dino": 0.0}],
 "summary": {"...": {}}}
```
