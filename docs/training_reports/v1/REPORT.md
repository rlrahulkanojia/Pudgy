# Training run report — Pudgy CogVideoX1.5-5B-I2V LoRA

**Status:** ✅ completed  ·  **Evaluated checkpoint:** `final (step 2500)`

## 1. Training schedule

| Metric | Value |
|---|---|
| Epochs (planned) | 132 |
| Optimizer steps (planned) | 2500 |
| Steps reached (from log) | 2500 |
| Examples / batches per epoch | 75 / 37 |
| Batch size/device · grad-accum · effective | 2 · 2 · 4 |
| Wall time (last log tick) | 9:39:36 |
| Loss first → last | 0.306 → 0.0294 |
| Loss min · mean(last 50) | 0.0185 · 0.050648 |
| Final LR | 0.0 |

Loss trend: `▄▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▂▁`

## 2. Command used

```bash
DATASET_DIR=/workspace/training_dataset \
/workspace/Pudgy/.venv/bin/python3 /workspace/Pudgy/.venv/bin/accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 train_cogvideox_image_to_video_lora.py \
  --pretrained_model_name_or_path /workspace/Pudgy/finetune/models/CogVideoX1.5-5B-I2V/ \
  --cache_dir /workspace/Pudgy/finetune/cache/ \
  --train_data_dir=/workspace/training_dataset/ \
  --train_data_meta=/workspace/training_dataset/metadata.json \
  --image_sample_size=592 \
  --video_sample_size=592 \
  --token_sample_size=592 \
  --enable_bucket \
  --video_sample_n_frames=33 \
  --video_sample_stride=1 \
  --fps=16 \
  --video_repeat 1 \
  --rank 64 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --learning_rate 3e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --max_train_steps 2500 \
  --checkpointing_steps 250 \
  --checkpoints_total_limit 12 \
  --dataloader_num_workers 4 \
  --mixed_precision bf16 \
  --seed 42 \
  --allow_tf32 \
  --output_dir /workspace/Pudgy/finetune/output_dir/pudgy-lora-v1 \
  --nccl_timeout 1800000
```

## 3. Dataset

- **Directory:** `/workspace/training_dataset`
- **Clips:** 75
- **Per-clip:** 768×1360 · 33 frames · 16.0 fps · h264 (e.g. `train/00000001.mp4`)
- **Sample captions:**
  - A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing Pax, a short round blue penguin, standing in a tiled room and gesturing up toward a 
  - A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines and flat pastel colors, showing Polly, a short round pink penguin with rosy cheeks, standing beside a small glowing 
  - A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines, soft cel shading and flat pastel colors, showing Polly, a short round pink penguin with a white face and belly, ros

## 4. Generated video — consistency analysis

- **Video (stored locally):** `final.mp4`
- **All-frames contact sheet:** `frames_montage.png`

![frames](frames_montage.png)

| Consistency metric | Value | Reading |
|---|---|---|
| Adjacent-frame SSIM (mean) | 0.9247 | 1.0 = identical; >~0.85 = smooth temporal coherence |
| Adjacent-frame SSIM (min) | 0.7821 | worst transition, at pair 7->8 |
| Adjacent-frame SSIM (std) | 0.0642 | low = uniform motion; spikes = flicker |
| Adjacent hist-corr (mean) | 0.9866 | palette stability frame-to-frame |
| Palette drift f0→last | 0.6725 | ~0 = colors hold; large = identity/color drift |
| Flicker-suspect pairs | 7->8, 8->9, 9->10 | adjacent SSIM >2σ below mean |

### Per-frame drift vs. conditioning frame (frame 0)

| frame | SSIM vs f0 | hist-corr vs f0 |
|---|---|---|
| 0 | 1.0 | 1.0 |
| 1 | 0.997 | 0.9151 |
| 2 | 0.9907 | 0.9517 |
| 3 | 0.977 | 0.9846 |
| 4 | 0.9606 | 0.9902 |
| 5 | 0.9273 | 0.9845 |
| 6 | 0.8903 | 0.9749 |
| 7 | 0.8561 | 0.9766 |
| 8 | 0.7863 | 0.9468 |
| 9 | 0.7759 | 0.6004 |
| 10 | 0.7624 | 0.4541 |
| 11 | 0.7586 | 0.4207 |
| 12 | 0.7713 | 0.4195 |
| 13 | 0.7796 | 0.4257 |
| 14 | 0.7725 | 0.4004 |
| 15 | 0.7701 | 0.3922 |
| 16 | 0.7602 | 0.3639 |
| 17 | 0.7662 | 0.3552 |
| 18 | 0.7832 | 0.3979 |
| 19 | 0.7817 | 0.3702 |
| 20 | 0.7846 | 0.3749 |
| 21 | 0.7806 | 0.3693 |
| 22 | 0.7809 | 0.3729 |
| 23 | 0.7649 | 0.3469 |
| 24 | 0.7858 | 0.3584 |
| 25 | 0.782 | 0.3432 |
| 26 | 0.7841 | 0.3407 |
| 27 | 0.783 | 0.3403 |
| 28 | 0.7881 | 0.3408 |
| 29 | 0.7882 | 0.3341 |
| 30 | 0.7881 | 0.3299 |
| 31 | 0.7876 | 0.3271 |
| 32 | 0.7876 | 0.3275 |

### Qualitative evaluation (frame-by-frame visual inspection)

Prompt/condition: frame 0 of `train/00000001.mp4` — Pax (blue penguin) in a tiled room with a
hanging lamp; caption expects a door to open and Polly (pink penguin) to peek in.

**Frame-by-frame walkthrough**
- **f0–f4 (strong):** Pax is clean and on-model — thick outlines, flat pastel blue/white, correct
  round body, orange feet, tuft of hair. Lamp + tiled wall reproduced faithfully. This is the
  best segment and shows the LoRA clearly learned the Pudgy style and Pax's identity.
- **f5–f8 (door opens, degradation begins):** a brown door swings in and a second penguin appears.
  Bodies start to smear/melt; the worst temporal jumps are here (adjacent SSIM min 0.78 at 7→8,
  flicker flagged 7→8, 8→9, 9→10). hist-corr vs f0 collapses 0.95→0.45 as the brown door floods
  the frame.
- **f9–f23 (single penguin + oversized door):** one blue penguin stands beside a large, flat,
  malformed brown door slab (the door is rendered too big and loses its outline/knob detail).
  Character is passable but not crisp; some warping.
- **f24–f28 (Polly appears, then collapses):** a pink penguin (Polly) shows up bottom-left, but
  instead of entering it **shrinks frame-over-frame and vanishes**.
- **f29–f32 (subject gone):** the penguins have disappeared entirely — the last four frames are just
  the empty room + brown door. This is a failure: the video ends with no character on screen.

**Assessment**
- **Character identity (Pax/Polly):** Good early (f0–f7), unstable after the door opens; Polly
  appears but disintegrates. Net: partial.
- **Style fidelity (thick outlines, flat pastel):** **Held throughout** — colors, outlines and the
  cel-shaded Pudgy look are correct even where geometry fails. The *style* transferred well.
- **Motion / prompt adherence:** The "door opens → Polly peeks" beat is *attempted* but breaks down;
  motion turns into morphing/dissolving rather than the gentle bouncy motion in the caption.
- **Temporal coherence:** **Poor in the back half.** Adjacent-SSIM mean 0.92 is inflated by the
  static empty-room tail; the real story is the sharp f8–f11 break (hist-corr 0.95→0.42) and the
  subject vanishing by f28.

**Verdict:** ⚠️ **Overfit / past the golden window at step 2500.** The style is learned, but the
final checkpoint collapses the subject (characters shrink and disappear). Consistent with the repo's
own note that fidelity usually peaks ~1000–2000 steps. **Recommend evaluating earlier checkpoints
(1000 / 1250 / 1500) and picking the golden one** — all 10 are saved in `../`. Re-run e.g.
`python inference/eval_pudgy_lora.py --checkpoint 1250 --output_path report/ckpt1250.mp4`, or lower
the strength with `--lora_scale 0.4`.

---

## 5. Checkpoint comparison (golden-checkpoint sweep)

Seven checkpoints generated with the **same** conditioning frame (frame 0 of `train/00000001.mp4`),
prompt, seed (42), scale (0.5) and resolution (432×768), then inspected frame-by-frame.

> ⚠️ **The numeric "subject-retention" flag below is unreliable** — the oversized brown door
> dominates each frame's color histogram, so hist-corr-vs-frame0 reads "LOST" even when the penguin
> is clearly on screen. The **visual assessment** column is the ground truth.

| Ckpt | adj-SSIM mean | adj-SSIM min | Pax identity (f0–f23) | Polly (pink) | Subject at end (f24–f32) | Background | Verdict |
|---|---|---|---|---|---|---|---|
| 750 | 0.935 | 0.800 | clean blue | faint pink blob | vanishes | **drifts PINK** (unstable) | reject — bg color instability |
| 1000 | 0.935 | 0.785 | **clean blue, crisp** | no | vanishes ~f24 | stable blue | strong first ⅔, empty end |
| 1250 | 0.917 | 0.781 | drifts **black-patched** | renders clearly | vanishes | stable blue | off-model color |
| 1500 | 0.916 | 0.826 | drifts **black-patched** | no | **persists to f32** | stable blue | best retention, off-model color |
| 1750 | 0.932 | 0.824 | **clean blue** | attempts (f24–28) | vanishes ~f29 | stable blue | ≈ 2000 |
| 2000 | 0.938 | 0.822 | **cleanest blue** | attempts (f24–28) | vanishes ~f29 | stable blue | **best all-around** |
| 2500 (final) | 0.925 | 0.782 | clean early | briefly (f24–26) | vanishes ~f28, oversized door | stable blue | overfit tail |

Per-checkpoint montages/videos in this folder: `ckpt{750,1000,1250,1500,1750,2000}_montage.png` + `.mp4`,
and `frames_montage.png` / `final.mp4` for 2500.

### Findings
- **Systematic across every checkpoint:** frames 0–5 are excellent (on-model Pax + style); the
  **"door opens" beat (f6–f11) starts the breakdown**, and the last ~4–8 frames **empty out** to just
  an oversized brown door. This is tied to this busy two-character conditioning clip + the 33-frame
  horizon, **not** something a different checkpoint fixes.
- **Style/identity was learned well** — clean blue Pax and the Pudgy look are reproduced; the failure
  is *temporal/scene* stability, not the LoRA failing to learn the character.
- **Color-drift band:** checkpoints **1250–1500** push Pax toward a black-patched look (over-baked);
  **1000 and 1750–2000** keep the clean blue.

### Golden checkpoint
🏆 **checkpoint-2000** — cleanest sustained blue Pax, stable background, and it even attempts pink
Polly. **checkpoint-1000** is the alternative if you only need the crispest Pax in the usable first
~23 frames. Avoid 1250/1500 (color drift) and 750 (background instability).

### Suggested next steps to fix the empty-tail failure
1. **Shorter horizon / simpler condition:** the tail collapses regardless of checkpoint — try a
   simpler single-character conditioning frame, or generate fewer frames.
2. **Lower LoRA strength:** `--lora_scale 0.4` reduces the door-domination (the door is the
   over-fired concept).
3. **Data:** the clip mixes two characters + a scene change in 2 s; single-action clips would likely
   train a more temporally stable character LoRA.

---

## 6. LoRA-strength test — checkpoint-2000 @ --lora_scale 0.4

Re-generated the golden checkpoint at 0.4 (vs the default 0.5) to test the "door over-fire"
hypothesis. Same conditioning frame/prompt/seed. Video: `ckpt2000_scale0.4.mp4`,
montage: `ckpt2000_scale0.4_montage.png`.

| | scale 0.5 | scale 0.4 |
|---|---|---|
| Pax f0–f5 | clean blue | clean blue (≈ same) |
| Door geometry | oversized malformed **slab**, no knob | **proper door with a visible knob**, correct proportions ✅ |
| Pax color f6–f23 | clean blue | slightly darker/navy |
| Subject at end (f24–f32) | vanishes ~f29 | **still vanishes ~f24** ❌ |
| adj-SSIM mean | 0.938 | 0.916 |

**Conclusion:** lowering the scale to 0.4 **clearly improves scene coherence** — the brown door
becomes a real, well-proportioned door (knob and all) instead of the over-fired slab — but it does
**not** fix the empty tail: the penguin still exits the frame by ~f24. This confirms the empty-tail
is a **temporal/scene-horizon** problem tied to this busy two-character conditioning clip, **not** a
LoRA-strength problem. Scale 0.4 is the better setting for the usable first ~23 frames.

**Net recommendation:** ship **checkpoint-2000 @ scale 0.4–0.5**; to actually fix the vanishing
subject, address the data/horizon (single-action conditioning clip and/or fewer frames), not the
LoRA scale.
