# Pudgy Penguins — Training Runbook (Internal)

**Scope:** End-to-end execution for fine-tuning `THUDM/CogVideoX1.5-5B-I2V` with a LoRA adapter on Pudgy Penguins data. Covers data processing → captioning → training → inference → evaluation. Pairs with [`Client_Data_Request.md`](./Client_Data_Request.md) (what we ask the client for) and the locked specs in [`Final_Sprint_Plan.md`](./Final_Sprint_Plan.md) / [`Training_Requirements.md`](./Training_Requirements.md).

**Locked decisions:** Base model = CogVideoX1.5-5B-I2V. Scope = both leads (Blue PP-001, Pink PP-002) + Tier-1/Tier-2 multi-character. Dataset = 60–80 clips + 50–100 reference images.

---

## Part B — Data processing & formatting

Turns client masters into a training-ready dataset. Implemented in `code/prep/build_dataset.py` (+ ffmpeg), writing a versioned folder to the RunPod network volume.

### B1. Target per-clip spec (hard requirements)
| Property | Value |
|---|---|
| Resolution | **768 × 1360 portrait** (primary). Landscape 1360 × 768 only if a horizontal GIF deliverable needs it. |
| Constraints | min(W,H)=768; 768 ≤ max(W,H) ≤ 1360; max(W,H) % 16 = 0 (1360/16 = 85 ✓) |
| Frame rate | resample to **16 fps** |
| Frame count | exactly **49** (≈3.06 s) or **81** (≈5.06 s) — must satisfy **8N+1** |
| Source window | segment 2–5 s at native 24 fps before resampling |
| Color | RGB, sRGB; background **baked in** (no alpha in the final training clip) |
| Container | per Passenger12138 dataloader — **confirm before B7** (default: high-bitrate MP4 CRF ≤ 16, or PNG frame folders), each paired with a `.txt` caption |

### B2. Steps (per skit → clips)
1. **Transcode** masters → lossless/ProRes intermediate. Assert short edge ≥ 768.
2. **Segment** into **atomic micro-actions** using the shot list — one action per clip. Split sequences ("walk → stop → wave") into separate clips. Windows 2–5 s.
3. **Resample** 24 → 16 fps.
4. **Trim** to exactly **49 or 81 frames**. For loops, cut on matching start/end poses.
5. **Resize** to 768 × 1360. Source 9:16 (0.5625) vs bucket (0.5647) differ < 1 % → minimal center-crop/pad, no visible distortion. Never stretch.
6. **Composite backgrounds** from the alpha (A2) renders: **70 %** of clips over flat **#808080**, **30 %** over the show background plate. Apply matte/defringe so there's no edge halo. *(If client used the gray/show double-render fallback, this step is just selection.)*
7. **Export** to trainer format + write the paired caption (Part C).
8. **Manifest:** append to `manifest.csv`: `clip_id, character(s), action_label, bg_type, frames, width, height, fps, source_skit`.

### B3. Composition & parity gates (must pass before training)
- **Bucket split:** ~60 % single-character (36–48), ~25 % Tier-1 two-char side-by-side >30 % gap (15–20), ~15 % Tier-2 proximate (9–12).
- **Action parity:** ≥ 4–5 clips per action type (walk, idle/breathe, wave/gesture, eat, jump/bounce, head-turn, expression shift, domestic task, affection). Re-balance starved actions.
- **Background mix:** 70/30 gray/show across the whole set.
- **Reference images:** 50–100 turnaround/expression PNGs packed into the joint image-video dataset.

### B4. Artifacts (props, backgrounds & FX)
Skits = Pax + Polly **+ artifacts** (the client's term for the recurring `Prp_*` props/environments — igloo, couch, table, chess set, iFin, bus, backpack, orca…). Note (client-confirmed): **any prop can animate depending on the skit** — props are not inherently static. So the goal is props that are **on-model and controllable** (hold when meant to hold, move when meant to move), not "always frozen":
- **Show-bg clips (the 30% bucket):** composite the character pass over the **clean background plate + its props** (asset class A2/#7). These teach that **backgrounds** stay put and props exist as consistent objects — include **both static-prop and animating-prop** clips (e.g. orca chomp, iFin screen) so motion is learned as on-cue, not default.
- **Prop reference images:** add the isolated transparent-PNG props to the **joint image dataset** alongside character turnarounds, so each prop has one canonical look. Caption them appearance-only ("a Pudgy-style igloo, …") — anchor-style, motion pruned.
- **Per-clip captions:** lock the **background** static; lock only the props actually still in that clip, and **describe the motion** of any prop that moves (Part C2). Optionally name + spatially place key props ("a glass chess table in the lower-center").
- **Manifest:** extend rows with `props` (list of `Prp_*` ids present) so recurring props (igloo, couch, table, phone) are verified to appear across **multiple** clips, not once — same parity logic as actions.
- **Risk it mitigates:** prop drift / re-design and **background** melt — the failure mode when artifacts are only ever seen baked into motion.

---

## Part C — Captioning

CogVideoX uses a frozen **T5-XXL** text encoder → dense natural-language sentences, not tag lists.

### C1. Tier 1 — fixed identity anchor (15–20 words, hand-authored, prepended to every caption)
- **Pax (blue ♂):** `"A stylized 2D cartoon animation of Pax, a pudgy blue penguin with thick black outlines, a white belly, and an orange beak,"`
- **Polly (pink ♀):** mirror structure, include her accents (blush cheeks, swept head-tuft, pink body). Use the **exact hex + accessory names from the client Figma/style guide** (A5) — placeholder wording only until that lands.
- **Multi-character:** add spatial anchors **per clip, read from the actual frame** — `"…positioned on the left third of the frame…"` / `"…on the right third…"`. Do **not** hardcode a fixed side: the client's left/right convention is scene-dependent (e.g. Pax screen-left in bed scenes, but Polly screen-left in the chess scene). The captioner must label each clip from what's on screen.

### C2. Tier 2 — VLM motion suffix (GPT-4o API or local MiniCPM-V)
Type-specific prompts describing **dynamics only, never appearance**:
- **T-pose / turnaround:** angle + limb placement only.
- **Neutral-bg clip:** motion, speed, direction, timing only.
- **Show-bg clip:** motion + lock the **background** static (`"the background environment remains completely stationary"`). Props are **not** auto-locked — any prop can animate per the skit; lock only the props that are actually still in that clip, and **describe the motion** of any prop that moves.

### C3. Tier 3 — pruning (anti-entanglement) — `code/prep/caption_prune.py`
1. **Keyword blocklist:** strip appearance terms already in the anchor (penguin, pudgy, blue/pink, white belly, thick outlines, beak, blush, tuft + synonyms like "bird/avian/plump").
2. **Semantic similarity:** embed each suffix sentence with `all-MiniLM-L6-v2`; drop any with **cosine > 0.7** to an anchor clause.
3. **Manual spot-check 10 %** of pruned captions to calibrate the blocklist + threshold before the full run.

**Final caption = anchor + pruned motion suffix.** Examples:
- *T-pose:* "A stylized 2D cartoon animation of Pax … standing stationary in a perfect profile view, left flipper resting flat against its side, against a neutral gray background."
- *Neutral-bg clip:* "… waddling forward in a continuous loop with a bouncy squash-and-stretch motion, waving its right flipper."
- *Show-bg clip (static props):* "… reaching across a glass table during a chess game, while the background igloo interior remains completely stationary."
- *Show-bg clip (animating prop):* "… as the orca lunges upward and snaps its jaws shut, while the background iceberg remains completely stationary."

---

## Part D — Training

### D1. Environment (Week 1, days 1–2)
1. Build & push Docker image: base weights + `diffusers` (source branch) + **Passenger12138/CogVideoX-5B-I2V-v1.5-lora-train** + ComfyUI + RIFE + HeliosBench eval + **version-pinned** deps. Push to Docker Hub.
2. Provision **RunPod A100 80 GB**; create **100–200 GB network volume**.
3. HF token + access for `THUDM/CogVideoX1.5-5B-I2V`. Confirm commercial-license clearance (legal) before training.

### D2. Dataset load & smoke test (Week 2)
1. Upload processed dataset + captions + reference images to the network volume.
2. Run the **B3 gates** from `manifest.csv` (bucket / parity / bg-mix).
3. **Zero-shot baseline:** feed client layouts into the *untrained* base model; generate 10–15 clips; log failure modes (beak shrink, color drift, bg warp) + FID/LPIPS/RAFT/SSIM + HeliosBench vs the client's hand-animated clips. **This is the bar to beat.**
4. **Diagnostic 500-step run:** verify VRAM (~35 GB), gradient flow, checkpoint saving, no OOM/crash.

### D3. LoRA training (Week 3 — dedicated)
**Locked config:** Rank **64**, Alpha **32 or 64**, LR **3e-5** with **cosine → 1e-6**, **8-bit AdamW**, batch **1**, grad-accum **4**, **4000 steps**, checkpoint every **500** (8 total), **bf16**, resolution **81 × 768 × 1360**.

| Run | Purpose | Vary |
|---|---|---|
| 1 | Baseline | locked config |
| 2 | LR sweep / alpha / caption fix | per Run 1 findings (LR 1e-5 or 5e-5) |
| 3 (if needed) | dataset composition / rank | 70/30 ratio, or Rank 32/128 |

- Runs are 10–14 h → launch evening, evaluate next morning.
- Evaluate all 8 checkpoints visually.
- **Golden checkpoint:** best character fidelity + motion before overfit — expected **steps 1,500–2,500** (overfit signal after 2,500–3,000). Run FID/LPIPS vs baseline. Output `lora.safetensors`.

### D4. Inference & post-processing (Week 4 — ComfyUI)
```
artist first frame (768×1360 PNG)
  → CogVideoX1.5 I2V + trained LoRA  [latent loop closure injected at denoise step 75–85% into final 6–10 frames]
  → VAE decode
  → loop SSIM check (target > 0.92; if below → FILM/RIFE cross-fade fallback on last/first 12 frames)
  → RIFE interpolation 16 → 24 fps
  → LoRA img2img cleanup @ 0.15–0.20 denoise (interpolated frames only)
  → GIF export (1360-edge, 24 fps)
```
~10–15 min/GIF; batch = 4 seeds/input (~50–60 min). Watch-folder automation for batch generation.

### D5. Evaluation (Weeks 4–5)
- **Quantitative:** FID (`clean-fid`), LPIPS, RAFT temporal consistency, Loop SSIM (> 0.92), **HeliosBench drift** (aesthetic / motion-smoothness / semantic / naturalness) — all calibrated against the client's hand-animated baselines.
- **Qualitative rubric** (animation director signs off *before* training): Character Identity, Motion Quality, Loop Closure, Background Stability, Overall Aesthetic, scored 1–4. **Pass bar: avg ≥ 3.0, no dimension = 1.**
- **Throughput:** success rate, time-per-GIF, cost-per-GIF, human-vs-AI.
- **Production sprint:** 15–25 single-char GIFs (4 seeds) + 5–8 Tier-1 / 2–3 Tier-2 multi-char; re-generate anything < 3.0.

### D6. Handoff (Week 6)
Technical report; "glass-box" 1-click RunPod deployment (frozen image + LoRA, web UI, no CLI); recorded training sessions (happy path + recovery); Phase 1 proposal (ML-Ops retainer, multi-character roadmap, 30 s scaling).

---

## Verification
1. **Format conformance:** `ffprobe` every output clip → assert 768×1360 (or 1360×768), 16 fps, frame count ∈ {49, 81}; build fails otherwise.
2. **Composition gates:** script over `manifest.csv` asserts bucket split (60/25/15 ±tol), action parity (≥ 4–5/action), bg mix (70/30).
3. **Caption sanity:** 10 % manual spot-check; no blocklisted term and no > 0.7-cosine sentence survives pruning.
4. **Pipeline smoke test:** 500-step diagnostic completes without OOM; checkpoints written every 500 steps.
5. **End-to-end:** generate a GIF from a held-out client layout through the full ComfyUI chain; confirm 24 fps, loop SSIM > 0.92, rubric ≥ 3.0.
6. **Lift check:** golden-checkpoint metrics beat the D2 zero-shot baseline on FID/LPIPS/HeliosBench.

## Open items to confirm
- Exact Passenger12138 dataloader input format (MP4 vs PNG-folder; caption pairing convention) — confirm against the repo before B7.
- Client alpha-render feasibility (A2) vs the gray/show double-render fallback.
- GIPHY deliverable aspect (portrait 768×1360 vs any landscape) — decides whether a second resolution bucket is added.
