# Pudgy Penguins — Training Approach v1 (executed baseline)

**Status:** ✅ built and run · ⚠️ results below bar on consistency + image quality → **superseded by [Training_Approach_v2.md](./Training_Approach_v2.md)**.

This documents the **first approach**, exactly as executed, so the GPU team has the baseline that v2 is measured against. Nothing here is aspirational — it's what actually ran.

---

## 1. Approach in one line

A single **character/style LoRA** on **CogVideoX1.5-5B-I2V**, trained on 75 short clips cut and tiled from the 30 base skits. Identity + motion learned jointly in one video LoRA; character identity at inference comes from the I2V first frame + the LoRA weights.

## 2. Data (as trained)

Built from `30_videos/` + `30_prompts/` only — no external assets. Full build in `Data_Readiness_Gap_Analysis.md` / `DATASET_BUILD.md`.

| Property | Value |
|---|---|
| Clips | **75** (`train/00000001.mp4 … 00000075.mp4`) |
| Resolution | 768 × 1360 (portrait, ÷16) |
| Frame rate | 16 fps (resampled from source 24 fps via `ffmpeg -r 16`) |
| Frame count | **33** (8N+1) → 2.06 s/clip |
| Codec | H.264, yuv420p, CRF 16, silent |
| Manifest | `metadata.json` (CogVideoX-Fun / Passenger12138 schema) |
| Captions | dense prose, identity anchor + per-shot action from storyboard beats |
| Source coverage | 25 of 30 skits, 46 unique shots, tiled into non-overlapping 33-frame windows |

## 3. Trainer & config (as executed)

- **Trainer:** Passenger12138 CogVideoX-5B-I2V-v1.5 LoRA fork (`finetune/train_cogvideox_image_to_video_lora.py`).
- **Base model:** `THUDM/CogVideoX1.5-5B-I2V`, bf16.
- **Launch script:** `pudgy-lora-repo/finetune/scripts/train_pudgy_lora.sh`.

Key flags actually used:

```
--video_sample_n_frames=33 --video_sample_stride=1 --fps=16   # mandatory for 33-frame clips
--image_sample_size=768 --video_sample_size=768 --token_sample_size=768 --enable_bucket
--rank 64 --lora_alpha 32 --lora_dropout 0.0
--learning_rate 3e-5 --lr_scheduler cosine --lr_warmup_steps 200
--optimizer AdamW --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0
--train_batch_size 1 --gradient_accumulation_steps 4 --gradient_checkpointing
--max_train_steps 2500 --checkpointing_steps 250 --mixed_precision bf16 --seed 42
```

Checkpoints every 250 steps → `finetune/output_dir/pudgy-lora-v1/checkpoint-*/`. Golden checkpoint picked by eye (fidelity expected to peak ~1000–2000 steps).

## 4. Result

Ran to completion; **output quality was not good enough** on the three target dimensions — character consistency wobbles under motion, image/line quality soft, robustness low. This triggered the reassessment.

## 5. Why it fell short — root causes (from the reassessment)

These are the specific reasons v1 underperformed, each fixed or re-scoped in v2:

1. **Wrong-generation base.** CogVideoX1.5 is the last of the old eps/v-pred diffusion class (not flow matching), on a custom license, and outside the two trainers the field now uses (diffusion-pipe / musubi-tuner).
2. **Identity + motion entangled in one video LoRA.** Free I2V lets identity drift mid-clip; nothing pins the character between endpoints. → v2 **decouples** identity (image domain) from motion (FLF2V).
3. **VAE ceiling never tested.** The frozen VAE caps image quality; flat color + thick outlines are exactly what it can damage. Never measured. → v2 Phase 0.1.
4. **Captions backwards.** Every caption densely re-describes "Pax, a short round blue penguin…"; the base already renders that, so identity barely binds. → v2 rare-token strategy.
5. **LoRA under-targeted / mis-sized.** rank 64 with default targeting on 75 clips risks overfit; targeting likely not all-linear+MLP. → v2 all-linear incl. MLP, r=16–32, α=2r.
6. **Thin, redundant data.** 75 motion-redundant tiled windows, no turnarounds/stills/single-action clips. → v2 data upgrade (Phase 3).

## 6. What v1 leaves behind (reusable)

- The **75-clip dataset** — still the Phase 0/1 baseline input for v2 (repackage for the new trainer; also the VAE round-trip test clips).
- The **pudgy-lora-repo** infra (env setup, `check_dataset.py`, GPU gotchas doc) — reference for standing up the new base.
- All **build provenance** (`metadata_provenance.json`) and the discrepancy notes.

---

*v1 = the baseline the v2 rubric (Training_Approach_v2.md §5) must beat at Gate G1.*
