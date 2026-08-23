# Training Approach v6 — Multi-Expression LoRA on the v2 Wan2.2 goldens

**One sentence:** continue-train the **v2 low-noise golden** on all **272 clips** of
`Data/processed/v6_expressions_272`, teaching four *contrastively-labelled* expressions to
two characters, so that expression becomes **promptable** rather than baked in.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B (fp16 DiT, Wan2.1 **8×** VAE, UMT5-XXL) — unchanged since v2 |
| **Init** | `v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors` (Azure, 306.8 MB, rank 16 / α 32) |
| **Untouched** | `v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors` — loaded at inference, never trained |
| **Data** | `Data/processed/v6_expressions_272` — 272 clips, 68 sources, 2 characters × 4 emotions × 3 shot sizes |
| **Trains** | one LoRA, low-noise expert only |
| **Cost** | ~3.5–4.5 h/epoch · gate at ~9 h · full run ~40–50 h |
| **Supersedes** | the v5 pilot (7 Pax/happy clips). v6's data is a strict superset of v5's. |

---

## 1. What this inherits, and from where

Everything below already exists and is verified. Nothing in this plan requires new client data.

| Asset | Location | State |
|---|---|---|
| Low-noise golden (**the init**) | Azure `pudgy/v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors` | rank 16 / α 32, 400 modules, all-linear incl. FFN |
| High-noise golden (**inference only**) | Azure `pudgy/v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors` | G1-validated motion prior — do not train |
| Training clips | `Data/processed/v6_expressions_272/clips/` (36 MB) | 272 × 1024², 24 fps, silent |
| Dataset configs | same folder, `*.workspace.{toml,jsonl}` | 4 buckets, box paths pre-written |
| Raw sources | Azure `pudgy/raw/iteration_3/03_expression_clips/` | 68 ProRes-4444 alpha clips |
| Trainer | musubi-tuner 0.3.4, `networks.lora_wan` | v2 stack, documented in [`../v2/actions_done.md`](../v2/actions_done.md) |
| Train script | [`finetune/wan/train_pudgy_happy_expr.sh`](../../finetune/wan/train_pudgy_happy_expr.sh) | takes `DATASET=` — runs v6 with no code change |

> ⚠️ **There is no GPU box.** The v5 box was destroyed and its disk was not a persistent
> volume. Step 0 of the runbook re-provisions and re-downloads ~65 GB of base weights.
> This is also why the v5 training clips had to be rebuilt — see
> `Data/processed/v5_happy_28/README.md`.

---

## 2. What the evidence forces

Every design decision below is inherited from a measured result, not chosen fresh.

| # | Observation | Source | → Decision here |
|---|---|---|---|
| 1 | **Expression belongs on the low-noise expert.** Continue-training high-noise destroyed prompt control: opposite prompts gave SSIM **0.9692** (identical video). Low-noise gave **0.9340** (prompt has effect). | v5 §4.5–4.6 A/B | **Train low-noise only.** High-noise stays frozen. Not re-litigated. |
| 2 | **Rank cannot change.** The golden is rank 16 / α 32; rank-8 tensors cannot load it. | v5 §3 plan correction | **rank 16 / α 32**, fixed. |
| 3 | **1e-4 would wreck the prior** on a small set. v2 used 5e-5 on 75 clips; v5 used 3e-5 on 28. | v2 recipe; v5 §3 | **3e-5**, with 5e-5 as the documented fallback if under-fitting at the gate. |
| 4 | **All 28 v5 clips started from ~one frame → that frame became a deterministic "happy" trigger.** Motion from it collapsed to **0.8 px**; from a novel frame the same weights moved **221–267 px**. | v5 §4.4 | The core problem v6 is designed to fix — see §3. |
| 5 | **Motion responsiveness decays with steps.** ep04 moved 259 px, ep18 moved 221 px. | v5 §4.4 | **Sweep checkpoints; never assume the last.** Save every epoch. |
| 6 | **LoRA strength is not a mitigation.** Halving to 0.5 moved 221 → 208 px. | v5 §4.4 | Don't tune scale to fix behaviour. Fix it in data. |
| 7 | **Weight-space distance is not a proxy for behaviour.** 13.7 % drift read as "safely preserved"; only the behavioural test caught the regression. | v5 §4.3 note | **All gates are behavioural.** Loss and drift are logged, never gating. |
| 8 | **Alpha-compositing generalises.** Unseen background drift **2/255**; the arc transferred to a ground never trained. | v5 §4.2 | Keep the 4-ground scheme. Reserve ≥2 unseen grounds for eval. |
| 9 | **Never judge on one seed.** | v4 §5.1, v5 §5.1 | **≥3 seeds** on every gate. |
| 10 | **Block-swap is a throughput tax when VRAM is free**: 27.5 → 13.7 s/it, 2× faster. | v5 §4.7 | `BLOCKS_TO_SWAP=0` unless a measured OOM forces it. |
| 11 | **Polly renders blue when the caption omits her colour** — 0 of 33 solo-Polly v4 captions said "pink". | v4 §5.3 | ✅ Already fixed: **136/136 Polly captions say "pink"**, 136/136 Pax say "blue" (verified). |
| 12 | **Canonical-view bias**: ¾ angles drift toward front-on. | v2 showcase §07; v5 §4.7 | Known carry-over. v6 has 9 camera clauses vs v5's 7 — measure whether more angle data helps. |
| 13 | **Wan's photorealistic bias never triggered** in v2 (10/10 scenes) or v5. | v2 showcase; v5 §4.1 | Downgraded from headline risk to a monitored one. |
| 14 | **The 8× VAE is not the ceiling** — PSNR 38.9 dB / SSIM 0.996 on real Pudgy art. | FINDINGS §4 | Base and VAE choice are settled. Do not revisit. |

---

## 3. Thesis — why v6 should fix what v5 could not

v5's defining failure was **conditioning-frame memorisation**. All 28 clips began from
roughly the same design-sheet pose and *always ended happy*, so the model learned
`this frame → happy` and fired the arc regardless of the prompt.

v6 does not fix this by adding a regulariser or lowering the LR. It fixes it **structurally,
for free**, because the same start frame now maps to **four different outcomes**:

```
                    ┌──▶  "happy expression"      →  squint + open grin      (21 f)
   same neutral     ├──▶  "surprised expression"  →  wide eyes + round gasp  (29 f)
   start frame  ────┤
   (per angle)      ├──▶  "angry expression"      →  brow drop + hard frown  (37 f)
                    └──▶  "neutral expression"    →  steady, beak closed     (57 f)
```

The caption is the **only** thing that disambiguates them. A model cannot memorise
`frame → happy` when that same frame is also labelled angry, surprised and neutral. The
contrastive signal v5 lacked is now present by construction, and **`neutral` is the
counter-example class** — the thing that makes "don't emote" a trainable target rather than
an absence.

**A second mechanism reinforces the first.** v5's trigger was sharp because the *pixels* of
the start frame were near-identical across all 28 clips. The shot-size ladder (§4.3) breaks
that too: every source performance is now rendered at 1.00×, 0.75× **and** 0.55×, taking the
set from **68 to 204 distinct character renderings — exactly 3×**. There is no longer a
single canonical start frame to memorise, only a pose seen at three scales. Contrastive
labelling attacks the *label* side of the trigger; the ladder attacks the *pixel* side.

`neutral` earns its 57 frames twice over: it is also the only **sustained-hold** data in the
programme. v5's clearest quality gap — "the hold relaxes by f11, we only have 0.875 s of
hold data" — is addressed by clips 2.7× longer.

**This thesis is falsifiable and Gate G-C (§7) is the test.** If the 4-way prompt separation
does not appear, the contrastive hypothesis is wrong and §10.1 is the fallback.

---

## 4. The data, and two defects worth knowing before you train

`Data/processed/v6_expressions_272` — 272 clips from 68 sources, 1024×1024, 24 fps, silent,
alpha-composited onto white / pastel blue / peach / mint, across **three shot sizes** (§4.3).

| Emotion | Frames | Pax | Polly | Source clips | Training clips | Latent frames | Tokens/clip |
|---|---|---|---|---|---|---|---|
| happy | 21 | 7 | 7 | 14 | 56 | 6 | 24,576 |
| surprised | 29 | 9 | 9 | 18 | 72 | 8 | 32,768 |
| angry | 37 | 9 | 9 | 18 | 72 | 10 | 40,960 |
| neutral | 57 | 9 | 9 | 18 | 72 | 15 | 61,440 |
| | | **34** | **34** | **68** | **272** | | |

**Character balance is exactly even** — 136 Pax / 136 Polly. That is the first time in the
programme; v4's corpus was 59 solo-Pax to 33 solo-Polly. Captions are colour-grounded on
both sides (verified: 100 % of Polly captions contain "pink").

### 4.1 ⚠️ Defect A — emotion and clip length are perfectly confounded

Each emotion appears at **exactly one** frame count, and no other emotion shares it:

```
21 frames ⟺ happy      29 frames ⟺ surprised
37 frames ⟺ angry      57 frames ⟺ neutral
```

musubi buckets by frame count, so the model sees "6 latent frames" and "happy" co-occur in
**100 %** of examples. Sequence length is a perfectly predictive shortcut for emotion, and
shortcuts are exactly what a small-data LoRA latches onto. The failure mode at inference:
**you request 21 frames and get happy no matter what you prompt.**

This is the same class of bug as v5's start-frame trigger, one level up. It is not
hypothetical — v5 proved this model memorises the most predictive available cue.

**Mitigation (recommended, ~10 min of prep):** emit a **second, common-length copy** of every
emotion so the bijection breaks. Truncating all four to 21 frames adds 216 clips at the
cheapest bucket:

| Bucket | Contents after the fix | Clips |
|---|---|---|
| f21 | **all four emotions**, both characters | 272 |
| f29 / f37 / f57 | surprised / angry / neutral at full length | 72 / 72 / 72 |

Now every emotion appears at ≥2 lengths and f21 alone carries all four, so length predicts
nothing. Cost: +56 % clips in the cheapest bucket, ≈ +25 % epoch time. `prep_expressions_v6.py`
already scans and truncates per emotion — this is a `--common-bucket 21` flag, not a rewrite.

**If you skip it**, Gate G-L (§7) exists to detect the confound, and §10.2 is the recovery.
Skipping is defensible for a first run; shipping without *testing* for it is not.

### 4.2 Defect B — happy is angle-poor and hold-poor

`happy` has 7 angles; the other three have 9 (they add `QF3_L`/`QF3_R` near-profiles and two
`QF1_R` takes). So happy is 56 clips against 72, and it is also the shortest hold. Expect
happy to be the weakest cell — the opposite of v5, where it was the only cell.

Not worth correcting by oversampling: `num_repeats` on the f21 block would re-introduce a
length↔emotion imbalance from the other direction. Note it, and read happy's eval scores in
that light.

### 4.3 Augmentation — two, both deliberate

**1. Alpha compositing onto 4 flat grounds.** 68 sources → 272 clips. Validated, not
decorative: v5 measured **2/255** background drift on lavender, a ground never trained,
with the expression arc intact (v5 report §4.2).

**2. A shot-size ladder — 1.00× close-up / 0.75× medium / 0.55× wide.** The client spec
asked for a "mix of close-up / medium / wide" and none arrived; every delivered clip is the
same fixed framing. It is synthesised from the alpha channel and **named in the caption**, so
framing becomes promptable rather than baked in.

| | |
|---|---|
| Distribution | **91 wide / 91 medium / 90 close-up** |
| Dataset size | **272 clips — unchanged** |
| Epoch cost | **unchanged** (~3.4 h) |

Zoom is **distributed, not multiplied**: one zoom per (clip, background) pair rather than
every clip at every scale. Tripling to 816 clips would have tripled the GPU bill *and* the
memorisation pressure from near-duplicate frames — the precise v5 failure mode.

> ⚠️ **Zoom must not become Defect A again.** §4.1's whole lesson is that this model latches
> onto whatever predicts the label. Zoom is therefore assigned round-robin over a stable
> enumeration and then **asserted balanced** against emotion, character, background and angle
> — 24/24/24 within every emotion, 45/46 per character, 22/23 per background, identical
> within every frame bucket. `prep_expressions_v6.py` **exits non-zero** if any factor level
> deviates more than 34 % from an even split. Do not disable that check.

**Zoom-out only, and this is measured, not a preference.** Across all 68 sources the
character's alpha bbox is a median **654×928 of 1024** — 91 % of frame height — and **10
clips already touch the top edge**. The largest centre-anchored zoom-*in* that keeps the
character whole is **~1.15×**, which is visually indistinguishable from 1.0×. A real tight
shot would have to cut the body, which the brief excluded. Verified after the build: every
zoomed clip has margin on all four sides.

Quality note: each zoom re-decodes from the **1080 source** straight to the target size — one
resample, not 1080→1024→target — so outlines are not softened twice. The flat ground is
regenerated rather than resampled, so there are no edge halos.

**Horizontal flip — checked and rejected.** Flipping `SIDE_R` lands almost exactly on
`SIDE_L`, so the character is mirror-symmetric and flipping is *safe*. That same symmetry
makes it near-useless: the client shot **both** directions, so QF1/QF2/SIDE are L-and-R
complete in every cell and a flip fills exactly **one** gap in 68 sources (Pax/neutral has
`QF3_L`, no `QF3_R`). Doubling the set for ~1.5 % new information is a bad trade.

**Actively avoid:**

| Augmentation | Why not |
|---|---|
| Colour jitter | Exact brand colour **is** the deliverable; the guidelines ship colour codes and v4 already rendered Polly blue. This attacks what we are protecting. |
| Random crop / scale (per-sample) | The zoom ladder already covers scale, deterministically and balanced. Random crops would also cut the character. |
| Temporal jitter / sliding windows | Breaks the labels — captions describe an *arc*; a window starting mid-arc is mislabelled. `head` extraction is correct, not lazy. |
| Rotation | Off-model. |

**What augmentation cannot fix:** each cell is *one performance from N angles*, not N
performances. Three shot sizes × four grounds × nine angles of a single take is still a
single take. Within-cell variety is the thin axis, and only client data reaches it.

**New eval consequence:** shot size is now a promptable axis, so it needs a gate — see
**G-Z** in §7. If 0.55× wide costs expression legibility (the face is ~55 % the linear size,
and the 8× VAE shrinks it again), that will show there.

### 4.4 What is *not* in this set

The 240-frame Pax turnaround is deliberately excluded (identity reference, not a
performance). No `sad`, `scared`, `confused`, `laughing`, `crying`, `affectionate`. No
motion or interaction clips. No design-sheet stills — so **v5 §5.2's joint image+video
training remains impossible**, and cross-angle consistency stays untaught.

---

## 5. Recipe

Identical to the v5 golden run except the dataset and the schedule. Deliberately so: changing
the recipe and the data together would make the result unattributable.

| Knob | Value | Why |
|---|---|---|
| Expert | **low-noise only**, timesteps **0–900** | Evidence #1 — settled by A/B |
| Init | `lora_lownoise_GOLDEN_ep40` via `--network_weights` | continue-train, not fresh |
| Rank / α | **16 / 32** | Evidence #2 — must match the init |
| LR | **3e-5** | Evidence #3 |
| Optimiser / precision | `adamw8bit` · fp16 | v2/v5; Comfy-Org ships fp16-only 14B |
| Timestep sampling | `shift`, `--discrete_flow_shift 5.0`, `--preserve_distribution_shape` | official I2V |
| Resolution | 1024×1024 | inherited from square source art — see §9.4 |
| Buckets | 4 blocks (21/29/37/57 f), `batch_size = 1` | musubi buckets by frame count; one block cannot hold two |
| `num_repeats` | **1** | 272 clips is ample; repeats would only inflate epoch time |
| Steps/epoch | **272** | = clips, at `num_repeats = 1`, `batch_size = 1` |
| Save cadence | **every epoch** | Evidence #5 — the golden may be early |
| Block-swap | **0** | Evidence #10 — only if a measured OOM forces it |
| Seed | 42 | comparability with v2/v5 |
| Target steps | **~3,000** (≈ 11 epochs) | v2 golden = 3,000 steps; v5 golden = 1,008. Sweep inside that band. |

### 5.1 Explicitly rejected

- **Training the high-noise expert.** v5 §4.6 settled it; re-running it would spend ~40 h to
  reproduce a known negative.
- **Four separate per-emotion LoRAs.** Re-introduces the "always fires" failure, destroys the
  contrastive signal that §3 depends on, and cannot compose at inference.
- **Lowering rank to reduce memorisation.** Impossible (evidence #2) and unnecessary — 272
  clips is ~10× the v5 set.
- **Tuning LoRA scale to restore motion.** Evidence #6: measured, does not work.

---

## 6. Compute — derived, not guessed

v5 measured **26.7 s/it** on the low-noise expert at 1024² × 21 frames = 6 latent frames =
24,576 tokens. Scaling by token count (a **floor** — attention is super-linear in sequence
length):

| Bucket | Clips | Tokens | Est. s/it | Est. epoch time |
|---|---|---|---|---|
| f21 | 56 | 24,576 | 26.7 (measured) | 25 min |
| f29 | 72 | 32,768 | ~36 | 43 min |
| f37 | 72 | 40,960 | ~45 | 53 min |
| f57 | 72 | 61,440 | ~67 | 80 min |
| | **272** | | | **≈ 3.4 h/epoch (floor)** |

Plan for **3.5–4.5 h/epoch**; ~3,000 steps ≈ **11 epochs ≈ 40–50 h**. That is the same order
as v2's 44.5 h per expert, and this is one expert, not two.

`neutral` alone is **40 %** of the compute. It is worth it — it is both the contrastive class
and the only sustained-hold data — but if the budget forces a cut, truncating neutral to 37
frames saves ~25 % of the run at the cost of the hold signal. Cut §4.1's fix last, not first.

**Gate G-C fires at ~epoch 2 (≈ 9 h, 544 steps).** Do not commit the full 50 h before it passes.

---

## 7. Gates

All behavioural (evidence #7). All at **≥3 seeds** (evidence #9). Inference is two-expert
FLF2V — low = the v6 checkpoint, high = the untouched v2 golden — matching v5 §4.

| Gate | When | Test | Passes when |
|---|---|---|---|
| **G-C** — controllability | ep 2, then every sweep point | Same start frame, **prompt is the only variable**, 4 emotions × 2 characters | Pairwise SSIM between emotion outputs **< 0.95** for all 6 pairs, *and* the four are visually distinguishable. v5 calibration: 0.9692 = prompt ignored, 0.9340 = prompt works. |
| **G-L** — length de-confound | after G-C | Generate **every emotion at every length** (21/29/37/57) | Emotion tracks the **prompt**, not the frame count. Failure = §4.1 confirmed → apply §10.2. |
| **G-M** — motion preserved | ep 2 and final | Neutral prompt + "turn head slowly", from a **novel** frame (v1 skit frame, Azure `training_v1`) | Subject x-range **> 200 px** over the clip. v5 calibration: 221–267 px healthy, 0.8 px = collapse. |
| **G-F** — no frame trigger | ep 2 and final | Same as G-M but from a **training** start frame, at **each of the 3 shot sizes** | x-range **> 100 px**, and the neutral prompt does not fire an expression. This is the exact v5 failure, and the joint test of both §3 mitigations. |
| **G-B** — background invariance | final | Two **unseen** grounds (lavender, sky-blue) | Corner drift **≤ 5/255**. v5: 2/255. |
| **G-P** — Polly parity | final | Every G-C test, per character | Polly renders **pink**, on-model, and scores within **10 %** of Pax. First real test of Polly. |
| **G-H** — hold | final | 57-frame neutral and 37-frame angry | Expression sustained past f30 without relaxing. v5 relaxed at f11 on 21 f. |
| **G-Z** — shot size | final | Same emotion + character, **shot-size clause is the only variable** (close-up / medium / wide) | Framing tracks the prompt, *and* expression stays legible at wide. If wide loses the expression, restrict the ladder to 1.00×/0.75× and re-train. |
| **G-R** — no regression | final | v2's 10-scene showcase prompts, re-run | Temporal SSIM ≥ **0.94**, struct-stability ≥ **0.87**, no identity blending in 2-char shots. v2 baseline: 0.949 / 0.880. |

`finetune/wan/eval_happy_v5.sh` already implements `indist` / `generalise` / `regress` /
`regress2` / `ctrl_happy`. G-C and G-L need a new mode that loops emotion × length; everything
else is a prompt swap.

---

## 8. Runbook

```bash
# 0. Provision — A100-80GB or H100-80GB. The v5 box is gone; disk is NOT persistent.
bash setup_wan_env.sh                       # venv + musubi 0.3.4 + ~65 GB base weights

# 1. Pull the v2 goldens (the init, and the frozen inference partner)
az storage blob download-batch --source pudgy --destination /workspace/wan_output/v2_golden \
   --pattern "v2/weights/curated/*GOLDEN_ep40*"

# 2. Prep runs LOCALLY, then you stage the output. The shipped set already carries
#    the shot-size ladder (section 4.3) — rebuild only if you change something.
#      python finetune/wan/prep_expressions_v6.py              # ladder on (shipped state)
#      python finetune/wan/prep_expressions_v6.py --no-zoom    # reproduce the pre-ladder set
#
#    STILL TO IMPLEMENT — the length de-confound of section 4.1. One flag, ~10 min of prep;
#    it composes with the ladder (a clip's zoom is fixed per background, so both the
#    full-length and 21-frame copies inherit it and the balance assertion still holds):
#      python finetune/wan/prep_expressions_v6.py --common-bucket 21
#
#    Stage: Data/processed/v6_expressions_272/  ->  /workspace/data_v6/
#           clips/ -> /workspace/data_v6/expressions_train/ ; *.workspace.* alongside

# 3. Pre-cache (VAE + T5) — 4 buckets, one pass
cd /workspace/musubi-tuner ; PY=/workspace/Pudgy/.venv-wan/bin/python
$PY src/musubi_tuner/wan_cache_latents.py \
    --dataset_config /workspace/data_v6/dataset_config_expressions_v6.workspace.toml \
    --vae /workspace/wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors --i2v
$PY src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config /workspace/data_v6/dataset_config_expressions_v6.workspace.toml \
    --t5 /workspace/wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16
# sanity: f21 latents must be (16, 6, 128, 128); f57 must be (16, 15, 128, 128)

# 4. Train — low-noise only. Reuses the v5 script unchanged via DATASET=.
DATASET=/workspace/data_v6/dataset_config_expressions_v6.workspace.toml \
EXPERT=low EPOCHS=11 SAVE_EVERY=1 LR=3e-5 BLOCKS_TO_SWAP=0 \
LOG_WITH=all WANDB_PROJECT=pudgy \
  bash finetune/wan/train_pudgy_happy_expr.sh

# 5. GATE at ~epoch 2 (~9 h) — do not wait for the full run
CKPT=<ep02> bash finetune/wan/eval_v6.sh   # G-C, G-L, G-M, G-F  (new: emotion x length loop)

# 6. Sweep every saved epoch on G-C + G-M; pick the golden by behaviour, not by loss
# 7. Full gate suite on the golden; then the 10-scene showcase
# 8. Mirror to Azure pudgy/v6/{weights,eval,inference,logs,docs}
```

> `train_pudgy_happy_expr.sh` needs **no edit** — `DATASET`, `EPOCHS`, `SAVE_EVERY`, `LR` and
> `BLOCKS_TO_SWAP` are all env overrides. Its `NAME` will still read `pudgy-happy-expr-*`;
> either accept it or pass a new `NAME`. Consider copying it to `train_pudgy_expr_v6.sh` so
> the v5 provenance in the filename does not mislead later readers.

---

## 9. Risks

| # | Risk | Severity | Handling |
|---|---|---|---|
| 9.1 | **Length↔emotion confound** (§4.1) | **High** — structural, and this model has already proven it takes shortcuts | Fix in prep (step 2b); detect with G-L; recover with §10.2 |
| 9.2 | **Start-frame memorisation persists.** Two independent mitigations now apply — contrastive labelling (§3) and 3× start-frame diversity from the shot-size ladder (68 → 204 distinct renderings). Both are still hypotheses until measured. | **Medium → Low** | G-F is the direct test. §10.1 if it fails. |
| 9.3 | **Four emotions may be harder than one.** v5 taught one arc with 28 clips; v6 asks for four discriminable arcs. 272 clips is 10× the data but 4× the task. | Medium | Gate early at ep 2; if separation is weak but present, train longer before concluding failure. |
| 9.4 | **Expression legibility at 0.55× wide.** The face is ~55 % linear size before the 8× VAE; subtle brow movement may not survive. Flagged as a risk in the v5 programme plan (§9.4). | Medium | **G-Z** is the test. Fallback: drop the ladder to 1.00×/0.75× and re-run prep — no code change, one constant. |
| 9.9 | **1024² is off Wan's canonical aspect list** (720×1280 / 1280×720 / 480×832 / 832×480) — a known quality tax inherited from square source art. | Low, known | Held constant for comparability with v5. Revisit only as a deliberate experiment. |
| 9.5 | **Canonical-view drift** — ¾ angles pull toward front-on (v2 §07, v5 §4.7). | Low, known | v6 has 9 camera clauses vs v5's 7; measure whether it improves. Real fix needs design-sheet stills. |
| 9.6 | **happy is the weak cell** (§4.2) — fewer angles, shortest hold. | Low | Expect it; read its scores in context. |
| 9.7 | **Over-baking.** v5's motion decayed from ep04 → ep18. | Low–Medium | Save every epoch, sweep on G-M. The golden may be early. |
| 9.8 | **Box loss.** The v5 box was destroyed with its data on it. | Medium | Mirror weights **during** the run, not after. Push each checkpoint as it lands. |

---

## 10. If a gate fails

**10.1 — G-C or G-F fails (expression still not promptable).** The contrastive thesis is
wrong. Fall back to v5's design **B**: train expression as a **standalone LoRA on a frozen
base**, stacked at inference, rather than continue-training the golden. Costs the v2 identity
prior but fully decouples the trigger. v5 §4.6 named this as the fallback and it has never
been run.

**10.2 — G-L fails (length predicts emotion).** Apply §4.1's common-bucket fix and re-train.
Do **not** try to patch it at inference by always generating one length — that hides the
defect and leaves the model unusable for the other three.

**10.3 — G-M fails (motion collapsed).** Take an earlier checkpoint first (evidence #5).
If every checkpoint collapses, the low-noise expert is absorbing motion it should not; drop
the LR to 2e-5 and re-run. Do **not** reach for LoRA scale (evidence #6).

**10.4 — G-P fails (Polly lags).** Most likely a data-volume issue, not a recipe one: Polly
has never been trained on more than 33 solo clips. Oversample Polly via a per-character
`num_repeats` split and re-run — but only *after* confirming the captions rendered pink.

---

## 11. Open questions

1. **Is `neutral` a real class or an absence?** If the model treats "neutral expression" as
   "no LoRA effect", the counter-example logic in §3 weakens. G-C's neutral-vs-others pairwise
   SSIM answers it.
2. **Does more angle data reduce canonical-view drift?** v6 has 9 camera clauses vs v5's 7 —
   the first chance to measure this.
3. **Which mitigation actually broke the frame trigger** — contrastive labelling or the 3×
   scale diversity? G-F tests them jointly. Separating them needs a `--no-zoom` ablation, which
   is one flag and ~7 min of prep but a second full training run. Worth it only if G-F is
   marginal.
4. **Does the v2 prior survive four emotions?** v5's "emergent pink hearts" showed the
   continue-train preserved prior knowledge. G-R checks whether that holds under 4× the task.
5. **Is 3e-5 still right at 10× the data?** 5e-5 (v2's low-noise LR) may fit better. Decide at
   the ep-2 gate on observed under/over-fitting, not in advance.

---

## 12. Where this sits

- **Programme:** [`../v5/Training_Approach_v5.md`](../v5/Training_Approach_v5.md) — the
  primitives-first taxonomy. v6 executes its **T3 (expression)** tier, on 4 of 9 primitives.
- **Precedent:** [`../v5/Training_Approach_v5_Happy_Expression_LoRA.md`](../v5/Training_Approach_v5_Happy_Expression_LoRA.md)
  and its report [`../../docs/training_reports/v5/REPORT_happy_pilot.md`](../../docs/training_reports/v5/REPORT_happy_pilot.md).
- **Base validation:** [`../v2/Training_Approach_v2.md`](../v2/Training_Approach_v2.md) (Gate G1 PASS).
- **Parallel track:** [`../LTX-2.5/Experiment_alpha_v-alpha.md`](../LTX-2.5/Experiment_alpha_v-alpha.md)
  ports the **LTX** line to 2.5. v6 is the **Wan** line. They are independent and can run
  concurrently on separate boxes.
- **Data:** `Data/processed/v6_expressions_272/README.md` · sources at
  `Data/raw/iteration_3/` (`CHANGELOG.md` records what arrived when).
