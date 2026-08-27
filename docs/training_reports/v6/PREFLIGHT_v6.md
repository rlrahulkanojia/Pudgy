# v6 preflight — dataset, preprocessing, and the exact condition at training start

**Status:** ⏸ **built, verified, not yet trained.** Everything below is the state of the
box at the moment training was ready to launch, recorded *before* the run so the run's
result can be attributed to a known input.
**Plan:** [`Training_Approach_v6.md`](../../../training_approach/v6/Training_Approach_v6.md)
**Precedent:** [`v5/REPORT_happy_pilot.md`](../v5/REPORT_happy_pilot.md) (the pilot v6 supersedes)

> Written because v5's report had to reconstruct its own inputs after the fact and
> found two material discrepancies against its plan doc (the clips were not
> white-background, and 1024² was forced rather than chosen). Recording the input
> state up front is cheaper than re-deriving it later.

---

## 1. What v6 is

A **separate experiment**, not a continuation of the v5 run. It takes:

| | |
|---|---|
| Base | Wan2.2-**I2V-A14B**, fp16 DiT · Wan2.1 **8×** VAE · UMT5-XXL — unchanged since v2 |
| Init | **`lora_lownoise_GOLDEN_ep40`** from v2, loaded via `--network_weights` (fresh optimiser — a continue-train, *not* `--resume`) |
| Frozen | **`lora_highnoise_GOLDEN_ep40`** from v2 — loaded at inference, never trained |
| Data | all **272** clips: 2 characters × 4 emotions × 3 shot sizes × 4 grounds |
| Trains | **one** LoRA, low-noise expert only |
| Produces | `pudgy-expr-v6-lownoise` → Azure `pudgy/v6/` |

v5's weights are **not** inputs. v6 branches from v2 directly; the v5 pilot (7 Pax/happy
clips) is superseded, and v6's data is a strict superset of it in content.

**Why low-noise only:** v5 §4.6 settled it by A/B. Continue-training the *high-noise*
expert destroyed prompt control — opposite prompts produced SSIM **0.9692**, i.e. the
same video. Training low-noise gave **0.9340**, i.e. the prompt actually did something.
The high-noise expert carries the G1-validated motion prior; touching it is what broke
v5, so it stays frozen.

---

## 2. Provenance chain — verified end to end

Every link was checked on this box, not assumed:

```
Azure pudgy/raw/iteration_3/03_expression_clips/     68 ProRes-4444 alpha clips
        │   ✅ all 68 md5 match manifest.json
        ▼   prep_expressions_v6.py  (alpha composite + shot-size ladder)
Azure pudgy/processed/v6_expressions_272/            272 mp4 + 4 jsonl + toml + manifest
        │   ✅ 272/272 pass full ffprobe spec sweep
        ▼   wan_cache_latents.py + wan_cache_text_encoder_outputs.py
/workspace/wan_cache/latents_expr/{f21,f29,f37,f57}  latents + T5 embeddings
            ✅ latent shapes match the plan's expected values
```

| Check | Result |
|---|---|
| Raw source integrity | **68/68 md5 match** the manifest that built the processed set |
| Clip spec sweep (all 272) | **0 violations** — every clip 1024×1024, 24 fps, silent, correct frame count |
| Frame-count legality | all four lengths are 4N+1 (21, 29, 37, 57) ✅ |
| jsonl path resolution | **272/272** resolve on the box |
| Character balance | **136 Pax / 136 Polly** — exactly even |
| Colour grounding | **136/136** Polly captions say "pink"; **136/136** Pax say "blue" |
| Shot-size balance | 91 wide / 91 medium / 90 close-up |
| Golden init | 400 modules · rank 16 · α 32 · 320 attn + 80 FFN · **0 zeroed `lora_up`** |
| Pre-cache | **272/272 latents + 272/272 T5 embeddings**, all four buckets |
| Latent shapes | f21 `(16,6,128,128)` · f29 `(16,8,…)` · f37 `(16,10,…)` · f57 `(16,15,…)` — all match plan |
| Dataset config | parses cleanly in musubi's own `BlueprintGenerator` (4 blocks) |
| Prep reproducibility | regenerating all 272 from raw gives **272/272 identical zoom assignments** and 51–56 dB PSNR vs the shipped clips (byte differences are ffmpeg-build noise; the shipped set was encoded on macOS) |

The last line matters: a from-scratch LoRA initialises `lora_up` to zeros, so 0/400
zeroed confirms this is a genuine trained checkpoint being continued, not a fresh init.

---

## 3. The dataset

`Data/processed/v6_expressions_272` — 272 clips from 68 source performances.

| Emotion | Frames | Pax | Polly | Sources | Training clips | Latent frames | Tokens/clip |
|---|---|---|---|---|---|---|---|
| happy | 21 | 7 | 7 | 14 | 56 | 6 | 24,576 |
| surprised | 29 | 9 | 9 | 18 | 72 | 8 | 32,768 |
| angry | 37 | 9 | 9 | 18 | 72 | 10 | 40,960 |
| neutral | 57 | 9 | 9 | 18 | 72 | 15 | 61,440 |
| | | **34** | **34** | **68** | **272** | | |

Token counts verified from the cached latents: f21 is `(16, 6, 128, 128)` → 6·64·64 =
24,576 after 2×2 patchify, exactly as planned.

**Character balance is even for the first time in the programme.** v4's corpus was 59
solo-Pax to 33 solo-Polly, and v4 §5.3 traced Polly rendering *blue* to captions that
never said "pink". Both are fixed here.

### 3.1 Frame budget — why not the delivered lengths exactly

Wan's VAE needs **4N+1** frames, so each delivery is truncated to the largest legal
count that fits. This is the only truncation applied and it is architectural, not a
choice:

| Emotion | Delivered | Trained | Dropped | Why |
|---|---|---|---|---|
| happy | 21 | 21 | 0 | already legal |
| surprised | 30 | 29 | 1 | next legal above 29 is 33 > 30 |
| angry | 40 | 37 | 3 | next legal above 37 is 41 > 40 |
| neutral | 60 | 57 | 3 | next legal above 57 is 61 > 60 |

Each performance therefore trains at its **maximum legal original length**. No clip is
shortened for any other reason, and no clip is windowed, jittered or sub-sampled.

---

## 4. Preprocessing — what `prep_expressions_v6.py` actually does

Source clips are ProRes 4444, 1080×1080, 24 fps, **with a real alpha channel** (~62% of
the frame is transparent). Two consequences drive the whole pipeline:

**1. Alpha compositing onto 4 flat grounds (68 → 272).**
A naive decode composites onto *black*, which would teach a black void. Instead each
performance is composited onto white / pastel blue / peach / mint. This is the
mechanism that teaches the expression rather than the expression-on-one-background —
v5 measured **2/255** background drift on lavender, a ground it never trained on.

**2. A shot-size ladder — 1.00× close-up / 0.75× medium / 0.55× wide.**
The client spec asked for a mix of framings and none arrived; every delivered clip is
the same fixed framing. It is synthesised from the alpha channel and **named in the
caption**, so framing becomes promptable. Three properties worth knowing:

- **Distributed, not multiplied.** One zoom per (clip, background) pair, so the set
  stays 272 clips and epoch cost is unchanged. Tripling to 816 would have tripled the
  GPU bill *and* the near-duplicate-frame memorisation pressure that broke v5.
- **Zoom-out only, and measured.** The character's alpha bbox is a median 654×928 of
  1024 (91% of frame height) and 10 clips already touch the top edge, so the largest
  centre-anchored zoom-*in* keeping the body whole is ~1.15× — indistinguishable from
  1.0×. A real tight shot would have to cut the body.
- **Balance is asserted, not assumed.** Zoom is assigned round-robin over a stable
  enumeration and then checked against emotion, character, background and angle; the
  script exits non-zero if any level deviates >34% from even. Clip length already
  predicts emotion perfectly (§6) and a second such shortcut would be self-inflicted.

**Resolution: 1024², not the source 1080².** Wan's 8× VAE plus 2×2 patchify needs an
even latent side and 1080/8 = 135 is odd. Each zoom re-decodes from the 1080 source
*straight to the target size* — one resample, not 1080→1024→target — so outlines are
not softened twice, and the flat ground is regenerated rather than resampled, so there
are no edge halos.

**Captions** use the verbatim v1/v2 identity anchors ("Pax, a short round blue
penguin"), **not** v4's rare token `pxngn0`. v6 continue-trains the v2 lineage, so the
captions must sit in the distribution that lineage already saw. Shape:

```
A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines
and flat pastel colors, showing Pax, a short round blue penguin, breaking into a big
happy smile, beak opening into a wide joyful grin, eyebrows lifting, cheeks lifting;
happy expression; static wide shot, eye level, facing the camera directly, front view;
plain white studio background.
```

### 4.1 Augmentations deliberately rejected

| Rejected | Why |
|---|---|
| Horizontal flip | The client shot both directions, so L/R are complete; flipping fills exactly 1 gap in 68 sources. Doubling the set for ~1.5% new information is a bad trade. |
| Colour jitter | Exact brand colour **is** the deliverable — v4 already rendered Polly blue once. This attacks what we are protecting. |
| Random crop / scale | The ladder covers scale deterministically and balanced; random crops would cut the character. |
| Temporal jitter / sliding windows | Captions describe an *arc*; a window starting mid-arc is mislabelled. `head` extraction is correct, not lazy. |
| Rotation | Off-model. |

**What augmentation cannot fix:** each cell is *one performance from N angles*, not N
performances. Three shot sizes × four grounds × nine angles of a single take is still a
single take. Within-cell variety is the thin axis and only client data reaches it.

### 4.2 Not in this set

The 240-frame Pax turnaround (identity reference, not a performance — feeding it to an
expression LoRA would teach rotation as part of the expression). No `sad`, `scared`,
`confused`, `laughing`, `crying`, `affectionate`. No motion or interaction clips. No
design-sheet stills, so joint image+video training remains impossible and cross-angle
consistency stays untaught.

---

## 5. Environment as built

| | |
|---|---|
| GPU | NVIDIA **A100-SXM4-80GB**, driver 535.230.02, host CUDA 12.6 |
| Disk | 250 GB (**not** a persistent volume — `workspace_is_volume: false`) |
| Trainer | musubi-tuner **0.3.4** (`/workspace/musubi-tuner`) |
| Venv | `/workspace/Pudgy/.venv-wan`, Python 3.12 |
| torch | **2.13.0+cu126** |
| Pins | diffusers 0.32.1 · transformers 4.57.6 · accelerate 1.6.0 · bitsandbytes 0.50.1 |
| Weights | 65 GB: both fp16 DiT experts, Wan2.1 8× VAE, UMT5-XXL |
| Tracking | wandb project **`rlrahulkanojia/pudgy`** (same project as v2/v4/v5) |

**One deliberate deviation from v2.** v2's stand-up used torch cu**128** on a driver
advertising CUDA 13.0. This box's driver is older (12.6), so cu**126** wheels were
installed to match the host minor version exactly. CUDA minor-version compatibility
would have allowed cu128, but matching avoids the PTX-JIT edge case entirely; A100 is
sm_80 and fully covered either way. Verified with a real fp16 matmul on-device, not
just an import.

`setup_wan_env.sh` now exists and reproduces all of the above (it did not before —
the v2 stand-up lived only as prose in `v2/actions_done.md`, which is a poor recovery
plan for a box that has already been lost once).

---

## 6. ⚠️ The length↔emotion confound is shipping UNFIXED — read this before reading results

Verified on the actual data, and it is a **perfect bijection**, not a tendency:

```
f21 → happy 56      f29 → surprised 72
f37 → angry 72      f57 → neutral 72
```

Every emotion occupies exactly one frame bucket; no bucket holds two emotions. **Clip
length predicts emotion with 100% accuracy.** musubi buckets by frame count, so the
model sees "6 latent frames" and "happy" co-occur in every single example.

This matters because v5 proved this exact model takes the most predictive shortcut
available: all 28 v5 clips started from ~one frame and always ended happy, and that
frame became a near-deterministic trigger — motion from it collapsed to **0.8 px** while
the same weights moved 221–267 px from a novel frame. The length confound is the same
class of bug one level up. The failure mode at inference is concrete: **request 21
frames, get happy no matter what you prompt.**

**Why it is not fixed.** The only no-new-data fix was emitting a second copy of each
longer emotion truncated to a common length, so the common bucket carries all four
emotions. That was **ruled out** — the requirement is original lengths only, no
clipping. The alternatives are worse: padding shorter clips teaches frozen motion and
poisons the sustained-hold signal, which `neutral` exists to provide.

**What we do instead — detection, not prevention:**

1. **Gate G-L** generates every emotion at every length and checks emotion tracks the
   *prompt*, not the frame count. It runs in the ~epoch-2 set (~9 h), **before** the
   full 40–50 h is committed.
2. If G-L fails, the recovery is `prep_expressions_v6.py --common-bucket 21` + retrain
   (plan §10.2). That flag **is now implemented** (it was documented as "STILL TO
   IMPLEMENT"), so the recovery is one flag rather than new code written under pressure.
3. Do **not** patch it at inference by always generating one length — that hides the
   defect and leaves the model unusable at the other three.

A second, smaller known defect: **`happy` is the weak cell** — 7 angles against 9 for
the others (56 clips vs 72), and the shortest hold. Expect it to score worst; read its
numbers in that light. This is the opposite of v5, where happy was the *only* cell.

---

## 7. The exact training condition

```bash
bash finetune/wan/train_pudgy_expr_v6.sh
```

| Knob | Value | Why |
|---|---|---|
| Expert | **low-noise only**, timesteps **0–900** | v5 §4.6 A/B — settled, not re-litigated |
| Init | `lora_lownoise_GOLDEN_ep40` (`--network_weights`) | continue-train, fresh optimiser |
| Rank / α | **16 / 32** | *forced* — the init is rank 16; rank-8 tensors cannot load it |
| LR | **3e-5** | v2 used 5e-5 on 75 clips, v5 3e-5 on 28; 1e-4 would wreck the prior |
| Optimiser | `adamw8bit` | v2/v5 |
| Precision | **fp16** | the Comfy-Org 14B DiT ships fp16-only; musubi forbids bf16-on-fp16 |
| Timestep sampling | `shift`, `--discrete_flow_shift 5.0`, `--preserve_distribution_shape` | official I2V |
| Resolution | 1024×1024 | inherited from square source art |
| Buckets | 4 blocks (21/29/37/57), `batch_size = 1` | musubi buckets by frame count |
| `num_repeats` | **1** | 272 clips is ample; repeats only inflate epoch time |
| Steps/epoch | **272** | = clips × repeats ÷ batch |
| Epochs | **11** (~3,000 steps) | v2's golden was 3,000 steps; sweep inside that band |
| Save cadence | **every epoch** + `--save_state` | the golden is often early (v5: ep04 moved more than ep18) |
| Block-swap | **32 of 40** | ⚠️ **contradicts the plan** — measured, see §7.1 |
| Seed | 42 | comparability with v2/v5 |
| Mirroring | **on, during the run** | risk 9.8 — the v5 box was destroyed *with* its weights |

**Estimated cost.** v5 measured 26.7 s/it at 24,576 tokens. Scaling by token count (a
floor — attention is super-linear in sequence length):

| Bucket | Clips | Tokens | Est. s/it | Est. epoch |
|---|---|---|---|---|
| f21 | 56 | 24,576 | 26.7 (measured) | 25 min |
| f29 | 72 | 32,768 | ~36 | 43 min |
| f37 | 72 | 40,960 | ~45 | 53 min |
| f57 | 72 | 61,440 | ~67 | 80 min |
| | **272** | | | **≈ 3.4 h/epoch (floor)** |

Plan for 3.5–4.5 h/epoch → **~40–50 h** for 11 epochs. `neutral` alone is ~40% of the
compute; it earns it as both the contrastive class and the only sustained-hold data.

### 7.1 ⚠️ Block-swap 32, not 0 — the plan's setting OOMs

**The first launch died on step 0 with `blocks_to_swap: 0`, the value the plan specifies.**

The plan's reasoning (§5, evidence #10) is sound but rests on a measurement that does not
generalise: v5 §4.7 found block-swap to be a 2× throughput tax when VRAM is free
(27.5 → 13.7 s/it), so the plan set it to 0. **But v5 only ever trained 21-frame clips.**
v6 introduces a 57-frame bucket at 61,440 tokens — 2.5× the sequence length — and it does
not fit. The plan's compute table (§6) projected *time* per bucket and never *memory*.

The failure is precise: `torch.OutOfMemoryError: Tried to allocate 1.17 GiB` inside
`WanRMSNorm._norm`, which upcasts activations to fp32. At f57 that tensor is
61,440 × 5120 × 4 B ≈ **1.26 GB**, matching the failed allocation almost exactly. So the
pressure is **activations, not weights** — which is also why it is unaffected by the fact
that LoRA training keeps only 400 small modules trainable.

Probed on the f57 bucket alone (each probe run to ≥4 real steps on a cleared card):

| `blocks_to_swap` | f57 peak VRAM | headroom | f57 rate | Verdict |
|---|---|---|---|---|
| 0 (plan) | OOM at 76.4 / 79.2 GB | — | — | ❌ dies on step 0 |
| 24 | 76.1 GB (93%) | 5.8 GB | 72.7 s/it | fits, thin margin |
| **32** | **70.7 GB (86%)** | **11.2 GB** | **74.3 s/it** | ✅ **chosen** |

32 costs ~2% throughput over 24 and buys double the headroom. For a ~35 h run where an OOM
forfeits up to a full epoch, that trade is obviously right.

**The feared cost to the short buckets did not materialise.** f21 with 32 blocks swapped
runs at **18.5 s/it** — still faster than v5's **26.7 s/it** on an *unswapped* card. This
box is an A100-**SXM4** where v5's was PCIe, and the extra bandwidth absorbs most of the
swap tax. Peak on f21 is only 32.6 GB.

Observed in the live run: **100% GPU utilisation sustained**, 320–430 W, with VRAM
oscillating **28 → 70 GB** as the sampler moves between buckets. The idle VRAM on short
buckets is not recoverable throughput — the SMs are already saturated. Raising
`batch_size` to consume it was rejected: the plan fixes `batch_size = 1`, and v4 §12.3
measured that a larger batch *hurts* LoRA fidelity on small datasets.

> **Carry-forward:** `blocks_to_swap: 0` is safe only while every bucket is ≤ ~21 frames at
> 1024². Any future run that lengthens clips must re-probe memory before launch.

**Do not extrapolate run time from single-bucket probes.** Weighting the probe rates by
the real bucket mix (56×f21 + 72×f29 + 72×f37 + 72×f57) predicts ~41.7 s/it. The live run
measures **61.7 s/it** — 48% higher. The probes ran *homogeneous* batches back to back;
the real run interleaves four bucket shapes, so the caching allocator and the block-swap
machinery churn on every shape change. Actuals:

| | Probe-based estimate | Plan estimate | **Measured** |
|---|---|---|---|
| s/it | 41.7 | — | **61.7** |
| per epoch | 3.2 h | 3.5–4.5 h | **4.66 h** |
| 11 epochs | 35 h | 40–50 h | **~51 h** |
| epoch-2 gate | 6.4 h | ~9 h | **~9.3 h** |

The plan's own estimate was closer than the probe extrapolation. Worth remembering next
time: probes bound *memory* reliably and *throughput* only loosely.

---

## 8. Gates — implemented and ready

All behavioural (v5 §4.3: weight drift and loss both said "fine" while the model had
stopped responding to prompts; only a behavioural test caught it). All at ≥3 seeds
(v4 §5.1: never judge on one seed). Inference is two-expert FLF2V — low = the v6
checkpoint, high = the untouched v2 golden.

| Gate | Tests | Pass |
|---|---|---|
| **G-C** | 4 emotions × 2 chars, same start frame, prompt the only variable | all 6 pairwise SSIM **< 0.95** |
| **G-L** | every emotion × every length | emotion tracks prompt, not frame count |
| **G-M** | neutral + head-turn from a **novel** frame | subject x-range **> 200 px** |
| **G-F** | same from a **training** frame, at all 3 shot sizes | x-range > 100 px, no expression fires |
| **G-B** | two unseen grounds (lavender, sky) | corner drift **≤ 5/255** |
| **G-P** | G-C read per character | Polly renders pink, within 10% of Pax |
| **G-H** | 57-frame neutral, 37-frame angry | expression sustained past f30 |
| **G-Z** | shot-size clause the only variable | framing tracks prompt; expression legible at wide |

`gates_v6.py --gates ep2` runs G-C/G-L/G-M/G-F — the set that fires at ~epoch 2.

**Eval keyframes were rebuilt** (`prep_eval_keyframes_v6.py`); the v5 set died with its
box. 18 frames: training-distribution starts per character, the shot-size ladder,
unseen grounds, and a novel frame per character drawn from the v1 skit set.

> One correction worth recording: v5 §4.4 used v1 clip `00000001` as its novel frame,
> but that clip places Pax hard against the right edge and a square crop cuts him. A
> subject clipped by the frame corrupts the x-range measurement that *is* G-M's
> pass/fail criterion, so v6 uses clips `00000010` (Pax) and `00000016` (Polly), both
> verified fully in frame with margin on all sides.

**Threshold caveat.** The plan's numeric thresholds came from v5's measurement code,
which did not survive the box. The estimators in `gates_v6.py` are re-implementations,
so absolute values are not guaranteed comparable to v5's. Treat thresholds as
provisional, read numbers relative to each other within this run, and re-anchor after
one full run. Every raw metric is written to JSON so it can be re-scored without
regenerating.

---

## 9. Deviations from the plan doc, and why

| # | Plan says | Actual | Reason |
|---|---|---|---|
| 1 | `--common-bucket 21` fixes the length confound | **not applied** | Original lengths only, no clipping (§6). Flag implemented for the §10.2 recovery path. |
| 2 | Run via `train_pudgy_happy_expr.sh` with `DATASET=` | new `train_pudgy_expr_v6.sh` | The v5 script names every artefact `pudgy-happy-expr-*`; v6 is a distinct experiment and mis-named weights get misattributed later. |
| 3 | `setup_wan_env.sh` | **written** | It was referenced but did not exist. |
| 4 | `eval_v6.sh` for G-C/G-L | **written**, plus `gates_v6.py` | The emotion × length loop did not exist; prompts are built from the same constants as the prep script so they match the training distribution by construction. |
| 5 | torch cu128 (v2 stack) | **cu126** | Host driver is CUDA 12.6, older than the v5 box. |
| 6 | Mirror to Azure after the run | **during** the run | Risk 9.8; a 40–50 h run is one preemption from total loss. |
| 8 | `blocks_to_swap: 0` | **32** | The plan's value OOMs on step 0 — the 57-frame bucket is 2.5× v5's sequence length and v5 is where the "0" came from. See §7.1. |
| 7 | Novel frame = v1 clip `00000001` | clips `00000010` / `00000016` | The original clips the subject at the frame edge, corrupting G-M's metric. |

---

## 10. Open questions carried into the run

1. **Is 3e-5 right at 10× the data?** v2 used 5e-5 on 75 clips. Decide at the ep-2 gate
   on observed under/over-fitting, not in advance.
2. **Is `neutral` a real class or an absence?** If the model treats "neutral expression"
   as "no LoRA effect", the contrastive logic weakens. G-C's neutral-vs-others pairwise
   SSIM answers it.
3. **Does the v2 prior survive four emotions?** v5's "emergent pink hearts" showed the
   continue-train preserved prior knowledge; G-R checks it under 4× the task.
4. **Does more angle data reduce canonical-view drift?** v6 has 9 camera clauses vs
   v5's 7 — the first chance to measure this.
5. **Which mitigation broke the frame trigger** — contrastive labelling or the 3× scale
   diversity? G-F tests them jointly; separating them needs a `--no-zoom` ablation and
   a second full run.

---

## 11. Where things live

```
/workspace/data_v6/expressions_train/     272 training clips
/workspace/data_v6/*.workspace.{toml,jsonl}   musubi configs (paths pre-written)
/workspace/data_raw/iteration_3/          68 raw alpha sources (md5-verified)
/workspace/wan_models/                    65 GB base weights
/workspace/wan_output/v2_golden/          the v2 goldens (init + frozen partner)
/workspace/wan_cache/latents_expr/{f21,f29,f37,f57}   cached latents + T5 embeddings
/workspace/eval_v6/keyframes/             18 rebuilt eval keyframes
/workspace/.env                           secrets, untracked, 0600
```

Azure account `pudgytraining`, container `pudgy`: sources at `raw/iteration_3/`,
training set at `processed/v6_expressions_272/`, run output will land at `v6/`.
