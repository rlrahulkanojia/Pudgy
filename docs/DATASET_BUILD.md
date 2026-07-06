# Pudgy Penguins — CogVideoX1.5-5B-I2V Training Dataset (v3, tiled)

Built from `30_videos/` (footage) + `30_prompts/` (storyboard beats) only. No external assets used.

## Directory layout

```
training_dataset/
├── train/               75 clips, 00000001.mp4 … 00000075.mp4
├── captions/            matching 00000001.txt …
├── metadata.json        primary manifest (CogVideoX-Fun / Passenger12138 schema)
├── videos.txt           file list (diffusers-style trainers)
└── prompts.txt          caption list, line-aligned to videos.txt
```

`metadata.json` entries: `{"file_path","text","type":"video","source_skit","source_shot","window"}`

## Clip spec (every clip conforms exactly)

| Property | Value |
|---|---|
| Resolution | 768 × 1360 (portrait, ÷16 both axes) |
| Frame rate | 16 fps (resampled from source 24 fps via `ffmpeg -r 16`) |
| Frame count | 33 (8N+1) → 2.06 s per clip |
| Codec | H.264, yuv420p, CRF 16 · silent |

Trainer: `video_sample_n_frames=33`, `video_sample_stride=1`, bucket 768×1360. Frame 0 = I2V conditioning frame.

## How it was built (window-tiling for volume)

1. **Scene-cut split** — PySceneDetect `ContentDetector` (threshold 22) on each of the 30 skits → 64 shots ≥2.06 s.
2. **Tile every shot** into consecutive **non-overlapping** 33-frame windows (a 12 s shot → 6 windows). This is the volume lever: long continuous shots now yield multiple clips instead of one.
3. **Resample + crop** — `scale=768:1366:lanczos, crop=768:1360, fps=16`, 33 frames.
4. **Self-check** — every window re-scanned for internal hard cuts; any straddling a cut was dropped (this is why some shots contribute fewer windows than their length allows).
5. **Content cull** — dropped windows that are: product/text end-cards (Icebreakers, "150 Questions", "Available Now", "Perfect Pair", Mother's Day CTA), off-character (Universe seahorses), tiny-character diorama (MovingIn snowglobe), off-model costume (Manifested monk robe), and wipe/match-cut straddlers detection couldn't catch.
6. **Captioning** — dense prose (frozen-T5 friendly): consistent Pax/Polly identity anchor + per-shot action from the storyboard beats, verified against frames. All windows of one continuous shot share that shot's caption; distinct sub-actions (e.g. Workouts mat vs pull-ups) are captioned separately.

Non-overlapping windows keep near-duplication low while still multiplying count. Scripts `split6.py` + `assemble3.py` in the build folder.

## Composition (75 clips)

- **Total footage:** ~155 s (75 × 2.06 s). **Source skits:** 25 of 30. **Unique source shots:** 46.
- **Character presence:** Pax in 57, Polly in 51. Two-character: 33. Pax-solo: 24. Polly-solo: 18.
- **Actions:** eating/drinking (meal, cake, fries, ice cream, chugging), brushing teeth, making coffee, walking/waddling, shopping, cuddling/comforting, sitting-reactions, chess, escalator fall, driving/car, sled-driving, pull-ups & mat workout, waving, searching-in-crowd, storming off.
- **Clips per skit (top):** BestWeekend / EatingStages / Knowing 6 each; LifeWithHim 5; BackToBed / Drinker / Intentional / LifeWithHer / TheresYou 4 each.

## Notes & limitations

- **75 is the clean maximum from full non-overlapping tiling.** The raw tiling produced 109 self-checked windows; 34 were culled as product cards / tiny-character / off-model / straddlers (see step 5). Reaching 120–150 would require re-including those lower-value windows or adding 50%-overlap windows (near-duplicates) — both trade quality for count.
- **Some duplication is inherent:** the multi-window skits (EatingStages ×6, Knowing ×6, etc.) are sequential slices of one continuous action, so those clips are visually similar. Monitor for overfitting; consider capping to 3–4 windows/shot if it appears.
- **No neutral-BG / transparent passes / T-poses** — identity is taught from motion + the I2V first frame only.
- **1 off-model clip** kept intentionally: Fixer2 (Pax in a hard hat). Remove if strict identity consistency matters.

## Regeneration / expansion

- `split6.py` parameters: `FRAMES` (33), window overlap (currently non-overlapping — set a stride fraction for more/fewer). Re-run with client footage to grow.
- To push toward 120–150 without new footage: enable 50%-overlap windowing in `split6.py` (adds ~40 near-duplicate clips) — not recommended for a first LoRA.
- Highest-value expansion remains client-produced single-action reference clips (gap analysis §5).
