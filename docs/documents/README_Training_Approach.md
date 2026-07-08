# Pudgy Penguins — Training Approach (Standalone)

The full high-level training approach for the custom Pudgy Penguins animation model. Self-contained — no other doc required.

**Goal:** generate brand-faithful animation of Pax (blue ♂) + Polly (pink ♀) and recurring props, starting with 3–5s looping GIFs, scaling to 30s shorts.

**Strategy in one line:** fine-tune **CogVideoX1.5-5B-I2V** with a **LoRA** that learns the characters/props/style, drive it from an **artist-composited first frame**, and post-process into a clean 24 fps looping GIF.

---

## Locked decisions
- **Base model:** `THUDM/CogVideoX1.5-5B-I2V` — strong I2V, native 16 fps / up to 1360×768, single A100-trainable; chosen partly to avoid Wan's photoreal bias against flat 2D.
- **Trainer:** `Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train` (correct RoPE / OFS fixes for v1.5 I2V).
- **One LoRA:** characters + props + style learned together (no separate image LoRA).
- **Consistency mechanism:** the **LoRA (learned) + the composited first frame** — *not* an inference-time reference image. CogVideoX has no reference/IP-adapter pathway.
- **Conditioning ceiling: frame control.** v1 = single first-frame I2V; the only sanctioned extension is start+end / sparse keyframes via an InP-style fine-tune (e.g. CogVideoX-Fun InP). **No** multi-reference re-architecture.
- **Scope:** both leads + multi-character (Tier-1 side-by-side, Tier-2 proximate).
- **Runtime:** ComfyUI for inference; RIFE for frame interpolation; RunPod A100 (80 GB train / 40 GB infer).

---

## Pipeline
```
Data prep ─▶ Captioning ─▶ LoRA training ─▶ Inference (ComfyUI) ─▶ Evaluation
```

### 1. Data prep
Segment client masters into **atomic** micro-action clips at **768×1360 / 16 fps / 49 or 81 frames (8N+1)**, with **70 % neutral-gray (#808080) / 30 % show-background**, plus 50–100 character/prop reference stills (a minority of the set). Gates before training: ~60/25/15 single/Tier-1/Tier-2 split, ≥4–5 clips per action, 70/30 bg mix, recurring props in ≥2 clips.

### 2. Captioning (two-tier, T5-XXL needs dense sentences)
- **Identity anchor** (15–20 words/character, from the style guide), prepended to every caption.
- **VLM motion suffix:** dynamics only, never appearance; show-bg clips lock the **background** static, lock only props that are actually still, and **describe the motion** of any prop that animates (props can animate per skit).
- **Multi-character:** spatial anchors read per-clip from the frame (left/right is scene-dependent).
- **Pruning:** blocklist + semantic-similarity removal of appearance terms from the suffix; 10 % manual spot-check. Artifact stills captioned appearance-only.

### 3. LoRA training (single A100 80 GB)
| Param | Value |
|---|---|
| Rank / Alpha | **64 / 32 or 64** |
| Learning rate | **3e-5**, cosine → 1e-6 |
| Optimizer | **8-bit AdamW** |
| Precision | **bf16** |
| Batch / grad-accum | **1 / 4** |
| Steps / checkpoints | **4000**, every **500** (8 total) |
| Resolution | **81 × 768 × 1360** (~35 GB VRAM) |

**Execution:** 3–5 runs, 10–14 h each (overnight). Run 1 = baseline; Run 2 = LR sweep / alpha / caption fix; Run 3+ = dataset or rank refinement. **Golden checkpoint** = best character fidelity + motion before overfit (expected steps ~1.5k–2.5k; overfit signal after ~2.5k–3k). Output: `lora.safetensors`. Verify it loads in the inference engine (key-naming/format) on a Week-2 diagnostic 500-step run.

### 4. Inference (ComfyUI)
```
artist first frame (768×1360 PNG)
  → CogVideoX1.5 I2V + trained LoRA  [latent loop closure injected at denoise step 75–85% into final 6–10 frames]
  → VAE decode
  → loop SSIM check (> 0.92; else FILM/RIFE cross-fade fallback on last/first 12 frames)
  → RIFE interpolation 16 → 24 fps
  → LoRA img2img cleanup @ 0.15–0.20 denoise
  → GIF export (24 fps)
```
~10–15 min/GIF; batch = 4 seeds/input. The **first frame is a first-class deliverable** — build a 768×1360 template (safe-area guides) so the artist composites valid frames (this is the "animator-assist" touchpoint). Inference prompts must mirror the training caption shape (anchor + motion).

### 5. Evaluation
- **Quantitative:** FID, LPIPS, RAFT temporal consistency, Loop SSIM (> 0.92), **HeliosBench drift** (aesthetic / motion-smoothness / semantic / naturalness) — all calibrated against the client's hand-animated baselines.
- **Qualitative rubric** (animation director signs off *before* training): Character Identity · Motion Quality · Loop Closure · Background Stability · Overall Aesthetic, scored 1–4. **Pass: avg ≥ 3.0, no dimension = 1.**
- **Throughput:** success rate, time-per-GIF, cost-per-GIF, human-vs-AI.

---

## Why this approach
- **LoRA + first frame** keeps Pax/Polly/props on-model without reference-image conditioning the model doesn't support.
- **70 % neutral-gray** clips teach character physics; **30 % show-bg** clips teach that **backgrounds** stay put and props are consistent objects that hold or move on cue.
- **Latent loop closure + RIFE** solve the two hard production needs: seamless loops and 24 fps delivery from a 16 fps model.
- **Frame-control ceiling** keeps loops / pose-to-pose open (a fine-tune) without a multi-reference research detour mid-sprint.

## Key risks → mitigations
- **Identity drift under motion** → strong LoRA, short clips, bounded motion, loop closure.
- **Prop drift / background melt** → show-bg clips + "props stationary" caption lock + prop reference stills.
- **Multi-character blur (spatial token diffusion)** → clear gap in the first frame + per-clip spatial captions.
- **Flat-color fights (model adds shading)** → base model chosen to minimize this; watch in eval.
- **Prop must animate on its own** (orca chomp, iFin screen) → client confirmed any prop can animate per skit; independent prop motion is *learned* — include animating-prop clips (captioned with the prop's motion), don't rely on a blanket "props static" lock.

## Compute & budget
RunPod: A100 80 GB train (40–80 GPU-hr across runs) + A100 40 GB inference. ~**$750** total for the training phase incl. experimentation buffer + VLM captioning.
