# Training Approach v7 — Motion + Multi-Expression LoRAs on the v2 Wan2.2 goldens

**One sentence:** continue-train the **v2 goldens** on `Data/processed/v7_primitives_2272`,
producing **two LoRAs** — a motion LoRA and a five-expression LoRA — each on the expert that
carries its kind of signal, so that action *and* expression *and* clip length all become
promptable rather than baked in.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B (fp16 DiT, Wan2.1 **8×** VAE, UMT5-XXL) — unchanged since v2 |
| **Init** | `v2/weights/curated/lora_{low,high}noise_GOLDEN_ep40.safetensors` (Azure, 306.8 MB each, rank 16 / α 32) |
| **Data** | `Data/processed/v7_primitives_2272` — **2,272 clips, 226 sources**, 2 characters × **13 labels** × 3 shot sizes × 1–5 lengths |
| **Trains** | **two** LoRAs: expression on low-noise, motion on high-noise *(expert open — see §4)* |
| **Cost** | expression ≈ **11.7 h/epoch** (1,200 steps) · motion ≈ **7.5 h/epoch** (1,072 steps) · gate-driven sweep to a ≈3,500-step ceiling ≈ **59 h total** |
| **Supersedes** | [v6](../v6/Training_Approach_v6.md) (expressions only, never run) |
| **Handover** | [`GPU_HANDOFF_v7.md`](GPU_HANDOFF_v7.md) — copy-paste execution for the GPU box |

> ### Launch order — all three runs are ready
>
> | Run | Steps/epoch | h/epoch | Start | Init |
> |---|---|---|---|---|
> | **Expression** (low-noise) | 1,200 | 11.7 | **now** | `lora_lownoise_GOLDEN_ep40` |
> | **Motion A/B** — high vs low, 2 epochs each | 1,072 | 7.5 | **now**, in parallel | both v2 goldens |
> | **Motion full run** | 1,072 | 7.5 | after G-X picks the arm | the winning arm |
>
> The motion run is no longer blocked: motion now has a counter-class (§3.3), so G-N is
> testable rather than a predicted failure. Expression and the A/B touch **different
> experts and different configs** and can run concurrently on two cards.
>
> **Corrupt source policy for this run:** 3 clips are unusable and are dropped
> (`PAX_MOTION_WALKING_QF1_R`, `PAX_MOTION_CONFUSED_QF1_R`, `PAX_MOTION_CONFUSED_QF3_L`);
> the other 17 damaged files are recovered by decode-retry and **pixel-verified clean in
> the built clips**. Training does not wait on re-exports. When clean files arrive, re-run
> the prep — it is incremental, so only the replaced clips re-encode.

---

## 1. Provenance — the v2 goldens, verified

This is an unbroken continue-train lineage, not a fresh LoRA. Verified against Azure and
against both prior runs on 2026-09-05:

| Asset | Azure path | Size | Used by |
|---|---|---|---|
| Low-noise golden | `pudgy/v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors` | 306.8 MB | v5 run 2 (**the A/B winner**), v6 init, **v7 expression init** |
| High-noise golden | `pudgy/v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors` | 306.8 MB | v5 run 1, v6 inference partner, **v7 motion init** |

Both are rank 16 / α 32, 400 modules, all-linear including FFN.

**Did v6 use these too? Yes — identically.** `Training_Approach_v6.md` names
`lora_lownoise_GOLDEN_ep40` as its init and `lora_highnoise_GOLDEN_ep40` as the frozen
inference partner. v5 used the same pair, one per run, and its report confirms the load
(`load network weights from …GOLDEN_ep40.safetensors: <All keys matched successfully>`) plus
weight-space evidence that the golden really was the starting point (0/60 zero `lora_up`
blocks at epoch 1, cos-sim 0.9985 — a from-scratch LoRA would read 60/60).

**Not used as init:** `v5/weights/lora_happy_lownoise_GOLDEN_ep18.safetensors`. It is itself
a v2-golden continue-train, but specialised on Pax/happy alone and carrying the
conditioning-frame memorisation v5 §4.4 diagnosed. Starting from v2 avoids inheriting it —
the same call v6 made.

> ⚠️ **There is no GPU box.** The v5 box was destroyed and its disk was not persistent.
> Step 0 re-provisions and re-downloads ~65 GB of base weights. Everything else is pullable
> from Azure — both `raw/` and `processed/` are mirrored — so §8 is `az` commands.

---

## 2. What the evidence forces

Every decision below is inherited from a measured result.

| # | Observation | Source | → Decision here |
|---|---|---|---|
| 1 | **Expression belongs on low-noise.** Continue-training high-noise destroyed prompt control: opposite prompts gave SSIM **0.9692** (identical video) vs **0.9340** on low-noise. | v5 §4.5–4.6 | **Expression run trains low-noise only.** Settled; not re-litigated. |
| 2 | **The expert carries a *kind* of signal.** High-noise = global composition and motion; low-noise = identity and fine texture. v5's happy delta was "beak shape and brow lines — low-noise territory". | v2 G1; v5 §4.6 | Motion is whole-body and temporal → **high-noise is the hypothesis**, tested by A/B (§4). |
| 3 | **Rank cannot change.** The goldens are rank 16 / α 32; rank-8 tensors cannot load them. | v5 §3 | **rank 16 / α 32**, fixed. |
| 4 | **1e-4 wrecks the prior** on a small set. v2 used 5e-5 on 75 clips; v5 used 3e-5 on 28. | v2 recipe; v5 §3 | **3e-5**, 5e-5 as documented fallback if under-fitting at the gate. |
| 5 | **All 28 v5 clips began from ~one frame → a deterministic "happy" trigger.** Motion from it collapsed to **0.8 px**; from a novel frame the same weights moved 221–267 px. | v5 §4.4 | Attacked in data: contrastive labelling, the shot-size ladder, and now the duration ladder. G-F is the test. |
| 6 | **Motion responsiveness decays with steps** — ep04 moved 259 px, ep18 moved 221 px. | v5 §4.4 | **Sweep checkpoints; never assume the last.** See §5 on save cadence. |
| 7 | **LoRA strength is not a mitigation.** 1.0 → 0.5 moved 221 → 208 px. | v5 §4.4 | Don't tune scale to fix behaviour. Fix it in data. |
| 8 | **Weight-space distance is not a proxy for behaviour.** 13.7 % drift read as "safely preserved"; only the behavioural test caught the regression. | v5 §4.3 | **All gates are behavioural.** Loss and drift are logged, never gating. |
| 9 | **Alpha-compositing generalises.** Unseen-background drift **2/255**. | v5 §4.2 | Keep 4 grounds; lavender + sky-blue reserved for G-B. |
| 10 | **Never judge on one seed.** | v4 §5.1; v5 §5.1 | **≥3 seeds** on every gate. |
| 11 | **Polly renders blue when the caption omits her colour.** | v4 §5.3 | ✅ 100 % of Polly captions say "pink", 100 % of Pax say "blue" (verified in the built set). |
| 12 | **Block-swap is a throughput tax when VRAM is free**: 27.5 → 13.7 s/it. | v5 §4.7 | `BLOCKS_TO_SWAP=0` unless a measured OOM forces it. |
| 13 | **Sequence length is a shortcut if a label sits at one length.** | v6 §4.1 | Fixed in data — see §3. G-L is the test. |

---

## 3. The data, and what is new about it

`Data/processed/v7_primitives_1776` — one folder, two sets sharing `clips/`, each with its
own jsonl, musubi config and latent cache.

| Kind | Sources | Clips | Buckets | Labels | Expert |
|---|---|---|---|---|---|
| motion | 106 | 1,072 | f13=424 · f17=72 · f21=288 · f25=144 · f33=144 | 6 | high (open — §4) |
| expression | 120 | 1,200 | f21=480 · f25=144 · f29=288 · f37=216 · f57=72 | **7** | low |

Expression covers **happy, surprised, angry, laughing, neutral, confused, crying** — 7 of
the 10 originally requested; `sad`, `scared` and `affectionate` remain outstanding.
Motion covers **walking, running, waving, sitting, jumping** plus a derived `standing idle`
(§3.3). Character balance 111 Pax / 115 Polly.

`f13` carries all 6 motion labels and `f21` all 7 expression labels, so sequence length
predicts nothing. Two single-label buckets remain (`motion f17` = running, `expression f57`
= neutral); both are harmless because the common bucket carries everything.

**Three axes are promptable by construction**, because each is named in the caption and
balanced against every other factor:

1. **Label** — 5 actions, 5 expressions. Contrastive: the same start frame maps to five
   outcomes per kind, so `frame → label` cannot be memorised.
2. **Shot size** — 1.00× / 0.75× / 0.55×, synthesised from alpha (the client never delivered
   zoom variety), assigned round-robin and **asserted balanced** against label, character,
   background and angle. Prep exits non-zero on >34 % deviation.
3. **Duration** — a ladder of 3–5 frame buckets per kind, from the kind's floor up to each
   label's **original delivered duration**. No frames are invented: the top rung is exactly
   what the client shot and shorter rungs are head truncations, captioned
   `"<label>, opening frames only"`.

> **Data correction, 2026-09-05.** An audit found `neutral`'s quarter-angles mislabelled.
> The client used two conflicting conventions and the digit cannot tell them apart —
> `QF_L/QF_L2/QF_L3` means index 1/2/3, but Pax-neutral's `QF_R/QF_R1/QF_R2` means 1/2/**3**
> (the suffix is an offset). Verified against the footage: Pax-neutral `QF_R1` is the same
> angle as `angry QF2_R`. **24 clips carried a wrong angle clause** — a near-profile shot
> captioned "turned slightly" — and the clash also faked a duplicate-take collision that
> hid the problem. Bare quarter names are now re-indexed by order, not by digit;
> `neutral` went from 1–3 sources per angle to an even 2 across all nine.

**The length↔label confound is fixed**, which is the thing v6 §4.1 specified and shipped
without: `f13` carries all five motion labels and `f21` all five expression labels, so
sequence length predicts nothing. `verify_length_balance` exits non-zero otherwise.

### 3.1 What the data still cannot teach

- **No *living* idle.** Motion's counter-class is derived (§3.3) from footage whose body is
  frozen. It teaches "stop", not "idle naturally". A breathing idle is requested in
  [Round 4](../../docs/documents/Client_Data_Request_Round4.md) §3.
- **No `turning`, `head_turn` or `bouncing`** — 3 of the 9 requested actions never arrived.
- **No interaction clips**, so two-character motion and interaction stay untaught. Asked
  for twice, never delivered; this is the largest remaining gap in the programme.
- **Three expressions missing** — sad, scared, affectionate.
- **Each cell is one performance from N angles, not N performances.** Augmentation cannot
  reach within-cell variety; only client data can.
- **`running` does not loop** (MAE 14.72 vs 0.37–2.02), so no seamless run cycle.
- **`happy` is angle-poor** (7 angles, one length) and **`confused` lost 2 sources** to
  unrecoverable corruption. Expect both to be the weakest cells; read their scores that way.
- **17 sources are recovered from damaged files by decode-retry.** Pixel-verified clean in
  the built clips, but the source files remain corrupt.

### 3.2 Duration ladder

Each label is emitted at up to 3 lengths, from its kind's floor up to its **original
delivered duration** — always the top rung. Shorter rungs are head truncations captioned
`"<label>, opening frames only"`. **Nothing is loop-tiled or time-stretched**, so the
duration ceiling is the client's: 1.38 s motion, 2.38 s expression.

### 3.3 ⚠️ The motion idle is DERIVED, not delivered

`standing idle` reuses the `neutral` expression footage under a second, also-true
description: the character is standing and not acting. **The client has never delivered an
idle class.** Two consequences that must not be lost:

1. **It is a frozen stand-still.** Measured across all 18 sources the body barely moves
   (MAE 0.028–0.064) while the face does (0.139–0.255) — the only motion is a blink. So it
   teaches "stop" as "freeze", and G-N must be read with that in mind.
2. **The same footage is in both sets** — as `neutral expression` on the low-noise expert
   and as `standing idle` on the high-noise one. Coherent (separate LoRAs, separate
   experts, both captions true of the pixels) but it must be stated wherever the set is
   described, or it reads as data we do not have.

Controlled by `DERIVE_IDLE_FROM_NEUTRAL` in `prep_v7.py`. **Set it to `False` when a real
breathing idle is delivered**, or the frozen version will dilute the real one.

---

## 4. The one open question — which expert gets motion

v6 §5.1 **explicitly rejected** training the high-noise expert, citing v5 §4.6: *"re-running
it would spend ~40 h to reproduce a known negative."* That rejection is correct for the
experiment v5 ran, and v7 does not overturn it by assertion.

**But v5's negative was expression-on-high-noise.** It put fine facial detail into the
expert that carries global composition, and the consequence was that the expression fired
regardless of the prompt. Motion-on-high-noise is a different proposition — it puts
whole-body temporal signal into the expert that already carries exactly that — and **it has
never been tested.** Evidence #2 makes it the hypothesis; it does not make it a fact.

So the motion run is an **A/B, decided at ~2 epochs (≈ 12 h), not a commitment**:

| Arm | Expert | Init | Rationale |
|---|---|---|---|
| **M-high** | high-noise, ts 900–1000 | `lora_highnoise_GOLDEN_ep40` | motion is global/temporal — evidence #2 |
| **M-low** | low-noise, ts 0–900 | `lora_lownoise_GOLDEN_ep40` | v5's only *demonstrated*-safe expert |

**G-X (§7) decides it.** Whichever arm wins continues to ~3,000 steps; the loser is kept for
the record, exactly as v5 kept its rejected high-noise run.

> ⚠️ **If M-high wins, both experts end up trained, and v5's protective mechanism is gone.**
> That also removes the ability to attribute a regression: with one LoRA changed you can
> always A/B against the frozen partner, with two you cannot. **Keep the unit of rollback
> explicit** — the two LoRAs are separate files, trained in separate runs, from separate
> v2 goldens, and every gate before G-S is run against the *untouched* partner. So a
> regression is always bisectable: swap one LoRA back to its v2 golden and re-run the gate.
> Never ship a combined checkpoint that cannot be decomposed this way.
> v5's stated reason low-noise worked was that it *"never touches the high-noise expert, so
> the G1-validated motion prior stays intact."* With two LoRAs there is no frozen partner.
> Each is therefore gated **independently against the untouched v2 golden partner first**,
> and the pair is gated separately in **G-S**. Do not skip G-S.

---

## 5. Recipe

Identical to the v5 golden run except the dataset, the schedule and the save cadence —
deliberately, so any result is attributable to the data rather than the knobs.

| Knob | Value | Why |
|---|---|---|
| Expert | expression **low** (ts 0–900); motion **A/B** (§4) | Evidence #1, #2 |
| Init | the matching v2 golden via `--network_weights` | continue-train, not fresh |
| Rank / α | **16 / 32** | Evidence #3 — must match the init |
| LR | **3e-5** | Evidence #4 |
| Optimiser / precision | `adamw8bit` · fp16 | v2/v5; Comfy-Org ships fp16-only 14B |
| Timestep sampling | `shift`, `--discrete_flow_shift 5.0`, `--preserve_distribution_shape` | official I2V |
| Resolution | 1024×1024 | inherited from square source art — a known cost, §9 |
| Buckets | 5 blocks per kind, `batch_size = 1` | musubi buckets by frame count |
| `num_repeats` | **1** | 1,776 clips is ample |
| **Save cadence** | **`--save_every_n_steps 250`** | ⚠️ **changed from v5/v6** — see below |
| Block-swap | **0** | Evidence #12 |
| Seed | 42 | comparability with v2/v5 |
| Target steps | **gate-driven, not fixed** — see §5.2 | The inherited 3,000 no longer means what it did |

> ⚠️ **Save cadence must change, and this is easy to miss.** v5 saved every epoch because an
> epoch was 56 steps. An epoch here is **856–920 steps**, so "every epoch" would yield only
> 3–4 checkpoints across the whole run — far too coarse to find a golden that evidence #6
> says is probably early. Use `--save_every_n_steps 250` (≈12 checkpoints), keeping all of
> them. **Verify the flag exists on the pinned musubi build before launching**
> (`wan_train_network.py --help`); if it does not, fall back to `--save_every_n_epochs 1`
> and accept a coarse sweep, or shorten the epoch with a subset.

### 5.2 ⚠️ The 3,000-step target is inherited and no longer justified

v6 and earlier drafts of this plan carried "~3,000 steps" from v2's golden. That number came
from a completely different regime:

| Run | Steps/epoch | Steps | **Epochs** |
|---|---|---|---|
| v2 golden | 75 | 3,000 | **40** |
| v5 golden | 28 | 1,008 | **36** |
| v6 (planned) | 272 | 3,000 | 11 |
| **v7 expression** | 920 | 3,000 | **3.3** |
| **v7 motion** | 856 | 3,000 | **3.5** |

At 3,000 steps v7 sees each clip roughly **three times**, against 36–40 passes for both
existing goldens. Matching the lineage's epoch count would take ~35,000 steps ≈ **15 days**,
which is not on the table.

**Do not resolve this by picking a bigger number.** Evidence #6 cuts the other way: v5
measured motion responsiveness *decaying* with steps (ep04 moved 259 px, ep18 moved 221 px)
and concluded the golden was probably early. More steps is a measured risk, not a default.

**So treat step count as an output of the gates, not an input:**

1. Save every 250 steps, keep everything (§5).
2. First gate at **~500 steps** — G-C and G-M. If nothing has moved by then, the LR is
   wrong (evidence #4 fallback), not the step count.
3. Sweep every saved checkpoint against G-C / G-M / G-F and plot the curve. Stop when the
   gates **plateau or regress**, not when a step counter is reached.
4. Hard budget ceiling: **~3,500 steps** (≈4 epochs, ≈30 h expression / ≈20 h motion). If
   the gates are still improving at the ceiling, that is a finding worth reporting — and a
   reason to extend deliberately, with a number.

### 5.1 Explicitly rejected

- **One LoRA for both kinds.** Merging motion and expression onto a single expert re-creates
  the v5 failure directly.
- **Per-label LoRAs.** Destroys the contrastive signal §3 depends on and cannot compose.
- **Lowering rank.** Impossible (evidence #3) and unnecessary at 1,776 clips.
- **Tuning LoRA scale to restore motion.** Evidence #7: measured, does not work.
- **Loop-wrapping clips to lengthen them.** Would manufacture undelivered footage; the
  duration ceiling stays the client's.

---

## 6. Compute — derived from v5's measurement

v5 measured **26.7 s/it** on low-noise at 1024² × 21 frames = 6 latent frames = 24,576
tokens. Scaling by token count (a **floor** — attention is super-linear in sequence length):

| Kind | Steps/epoch | Epoch time | To a 3,500-step ceiling |
|---|---|---|---|
| motion | 1,072 | **7.5 h** | 3.3 epochs ≈ **25 h** |
| expression | 1,200 | **11.7 h** | 2.9 epochs ≈ **34 h** |
| | | | **≈ 59 h** |

Plus the motion A/B: 2 epochs × 2 arms ≈ **30 h** before committing to the full motion run.

⚠️ **At 11.7 h/epoch the first gate matters more than it used to.** The ~500-step check
lands about 5 hours in; getting it wrong costs a third of a day, not a couple of hours. Do
not skip it to "save time".

`f57` neutral alone is 14 % of the expression epoch and `f13` motion is 30 % of the motion
epoch. If budget forces a cut, drop the **middle** rung of the duration ladder first
(`LADDER_STEPS = 2` in `prep_v7.py`) — that returns roughly the v6-shaped set at ~1,272
clips. **Cut the ladder before cutting the common floor bucket**, which is load-bearing
against the length confound.

---

## 7. Gates

All behavioural (evidence #8), all at **≥3 seeds** (evidence #10). Inference is two-expert
FLF2V matching v5 §4: the expert under test carries the v7 LoRA, its partner carries the
**untouched v2 golden** — until G-S, which tests the pair.

| Gate | When | Test | Passes when |
|---|---|---|---|
| **G-X** — motion expert A/B | motion, ep 2 (**before committing**) | M-high vs M-low: same start frame, action prompt is the only variable, 5 actions × 2 characters | The arm whose pairwise SSIM between action outputs is **< 0.95** across all pairs *and* which retains prompt response. v5 calibration: 0.9692 = prompt ignored, 0.9340 = prompt works. **Decides §4.** |
| **G-C** — controllability | ep 2, then each sweep point | Same start frame, prompt the only variable, 5 labels × 2 characters, per kind | Pairwise SSIM **< 0.95** for all 10 pairs, and the five are visually distinguishable |
| **G-L** — length de-confound | after G-C | Generate **every label at every bucket** of its kind | Label tracks the **prompt**, not the frame count. Failure = the confound survived → §10.2 |
| **G-D** — duration control | after G-L | Same label + character, **requested length is the only variable**, across the label's ladder | Output length tracks the request, and quality does not degrade at the short rungs. This is the gate the duration ladder exists to pass. |
| **G-M** — motion preserved | ep 2 and final | Neutral prompt + "turn head slowly", from a **novel** frame (v1 skit frame, Azure `training_v1`) | Subject x-range **> 200 px**. v5: 221–267 px healthy, 0.8 px = collapse |
| **G-F** — no frame trigger | ep 2 and final | As G-M but from a **training** start frame, at each of the 3 shot sizes | x-range **> 100 px**, and a neutral prompt fires no expression. The joint test of all three §3 mitigations |
| **G-N** — idle is promptable | final | "standing still, not moving", motion LoRA active | Subject x-range **< 50 px** and no action fires. **Without `standing_idle` this is expected to FAIL** (§3.1 — no idle class exists). Once it lands this becomes a real pass/fail gate, and should additionally show *life*: some frame-to-frame change, not a frozen character. Calibration: `neutral`'s frozen body reads ~0.03–0.06 MAE — an idle that trains correctly should sit above that |
| **G-B** — background invariance | final | Two **unseen** grounds (lavender, sky-blue) | Corner drift **≤ 5/255**. v5: 2/255 |
| **G-P** — Polly parity | final | Every G-C test, per character | Polly renders **pink**, on-model, within **10 %** of Pax |
| **G-H** — hold | final | 57-frame neutral, 37-frame angry, 37-frame laughing | Expression sustained past f30 without relaxing. v5 relaxed at f11 on 21 f. `laughing` is the first real emotive hold in the programme |
| **G-Z** — shot size | final | Same label + character, shot-size clause the only variable | Framing tracks the prompt, and the label stays legible at 0.55× wide. If not, restrict the ladder to 1.00×/0.75× and retrain |
| **G-S** — stacking | final, **only if M-high won** | Both LoRAs loaded together, motion prompt × expression prompt | Both axes respond independently; no identity blending; motion x-range still > 200 px from a novel frame. **The test that v5's frozen-partner protection has not been lost** |
| **G-A** — unseen angle *(only if the §9 holdout is taken)* | final | Generate the held-out `QF2_R` pose per label, both characters | The angle renders correctly and scores within **15 %** of the trained neighbours `QF1_R`/`QF3_R`. Failure = the model interpolates angles it saw rather than generalising, and the nine-angle coverage is doing less than assumed |
| **G-R** — no regression | final | v2's 10-scene showcase prompts, re-run | Temporal SSIM ≥ **0.94**, struct-stability ≥ **0.87**, no identity blending in 2-char shots. v2 baseline: 0.949 / 0.880 |

**Most of this is already implemented — in the v6 gate harness, not the v5 scripts:**

| Script | Implements |
|---|---|
| `finetune/wan/gates_v6.py` (497 ln) | G-C controllability matrix + verdict |
| `finetune/wan/gate_gl_v6.py` (215 ln) | G-L length de-confound |
| `finetune/wan/gate_gf_v6.py` (212 ln) | G-F training-frame trigger |
| `finetune/wan/gates_remaining_v6.py` (277 ln) | G-F multi-seed, G-B, G-Z, G-H, G-L for Polly — batched into one model load |
| `finetune/wan/eval_v6.sh` | two-expert FLF2V generation primitive |
| `finetune/wan/sample_epochs_v6.py` | per-checkpoint visual quality curve |

⚠️ **Not drop-in.** They need adapting on three axes: **7 expression labels + 6 motion
labels** (sized for v6's four emotions); **two experts** (v6 trained low-noise only with a
frozen partner — G-S needs a two-LoRA load); and **G-D is new**, because v6 had one length
per label so duration control could not be asked. Budget roughly a day.

`gates_v6.py` shares caption constants with `prep_expressions_v6.py` so eval prompts sit in
the training distribution by construction — preserve that by importing from `prep_v7.py`
rather than retyping strings.

Execution steps, credentials, hardware and report-back checklist:
**[`GPU_HANDOFF_v7.md`](GPU_HANDOFF_v7.md)**.

---

## 8. Runbook

```bash
# 0. Provision — A100-80GB or H100-80GB. The v5 box is gone; disk is NOT persistent.
bash setup_wan_env.sh                       # venv + musubi 0.3.4 + ~65 GB base weights

# 1. Pull the v2 goldens — the init for both runs, and the frozen inference partner
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace/wan_output/v2_golden \
   --pattern "v2/weights/curated/*GOLDEN_ep40*"

# 2. Pull the whole v7 delivery — ONE command, both kinds
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace \
   --pattern "processed/v7_primitives_2272/*"
mkdir -p /workspace/data_v7 && mv /workspace/processed/v7_primitives_2272/* /workspace/data_v7/
ls /workspace/data_v7/clips/*.mp4 | wc -l        # expect 2272

# 3. Precompute latents + text embeddings, once per kind
for KIND in motion expression; do
  python .../wan_cache_latents.py \
      --dataset_config /workspace/data_v7/dataset_config_${KIND}_v7.workspace.toml \
      --vae wan_2.1_vae.safetensors --i2v
  python .../wan_cache_text_encoder_outputs.py \
      --dataset_config /workspace/data_v7/dataset_config_${KIND}_v7.workspace.toml \
      --t5 umt5-xxl --batch_size 16
done

# 4. Verify --save_every_n_steps exists on this musubi build BEFORE launching (section 5).
python /workspace/musubi-tuner/src/musubi_tuner/wan_train_network.py --help \
  | grep -q save_every_n_steps && echo "OK" || echo "MISSING -> fall back to SAVE_EVERY=1"

# 5. EXPRESSION — settled, no A/B. EPOCHS is a CEILING; stop where the gates plateau.
DATASET=/workspace/data_v7/dataset_config_expression_v7.workspace.toml \
NAME=pudgy-v7-expr-lownoise EXPERT=low EPOCHS=3 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh

# 6. MOTION A/B — 2 epochs each, then STOP and run G-X. Runs concurrently with step 5
#    on a second card: different expert, different config, no shared output.
DATASET=/workspace/data_v7/dataset_config_motion_v7.workspace.toml \
NAME=pudgy-v7-motion-highnoise EXPERT=high EPOCHS=2 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh
DATASET=/workspace/data_v7/dataset_config_motion_v7.workspace.toml \
NAME=pudgy-v7-motion-lownoise  EXPERT=low  EPOCHS=2 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh

# 7. FIRST GATE at ~500 steps (checkpoint 2) — do NOT wait for the runs to finish.
#    G-C and G-M. If nothing has moved, raise LR to 5e-5; do not just add steps.

# 8. G-X picks the motion arm -> continue it to the ceiling. Then the remaining gates.

# 9. Upload
python finetune/wan/azure_upload_v7.py       # weights + eval + logs + this doc -> pudgy/v7/
```

**`finetune/wan/train_pudgy_v7.sh` is ready** — a copy of the v5 script with the two changes
v7 needed: `NAME` is overridable (without it the two A/B arms share an output directory and
overwrite each other) and `SAVE_EVERY_STEPS` adds `--save_every_n_steps`. Everything else —
`DATASET`, `EXPERT`, `LR`, `RANK`, `ALPHA`, `NETWORK_WEIGHTS`, `BLOCKS_TO_SWAP` — is taken
from the environment as before.

## 9. Known costs carried forward

- **Square 1024²** is off the base's canonical aspect list (720×1280 / 1280×720 / 480×832 /
  832×480) — a quality cost inherited from square source art, unchanged since v5.
- **Canonical-view bias**: ¾ angles drift toward front-on (v2 showcase §07, v5 §4.7). v7 has
  9 camera clauses where v5 had 7 — measurable, not yet fixed.
- **`happy` is angle-poor** — 7 angles against 9, so 344 clips carry it at one length only.
  Expect it to be the weakest expression cell; read its scores in that light.
- **No formal holdout on pose.** Two *backgrounds* (lavender, sky-blue) are reserved and
  test background-invariance via G-B — but every one of the 174 source performances is in
  training, at all nine angles. Nothing currently distinguishes "it generalises" from "it
  memorised 174 performances at 12 variations each".

  A **performance**-level holdout is impossible: each cell contains exactly one take, so
  holding it out deletes the class. An **angle** holdout is possible and cheap:

  > **Recommended:** reserve `QF2_R` from training for every (character, label) — **20 of
  > 174 sources, ~11 %** — and add gate **G-A** below. It leaves 8 angles per cell, and it
  > makes "does it generalise to a viewing angle it never saw" answerable for the first time
  > in the programme. That matters here specifically because **canonical-view bias is a
  > known, unfixed v2/v5 defect** (§9) and v7 is the first run with enough angles to measure
  > it.
  >
  > Cost: 11 % less data, ~11 % less epoch time, and it must be decided **before** launch —
  > retro-fitting a holdout after training proves nothing.
  > Not yet implemented; it is a `--holdout-angle QF2_R` flag in `prep_v7.py`.

## 10. Fallbacks

1. **G-X says neither motion arm is controllable** → motion is not learnable as a LoRA on
   this data. Most likely cause is §3.1 (no idle counter-class), not the recipe. Do **not**
   spend the remaining ~18 h; request the four missing actions from the client first.
2. **G-L fails (length still predicts the label)** → the common bucket was insufficient.
   Raise `LADDER_STEPS` so every label spans more lengths, and re-prep — cheap, because the
   build is incremental and only new rungs are encoded.
3. **G-D fails (length not controllable)** → the ladder is not enough on its own; add the
   requested length to the caption text explicitly and re-caption with
   `prep_v7.py --captions-only`, which costs seconds and no re-encode.
4. **G-N fails (no promptable idle)** → expected until Round 4 lands. Document it as a data
   gap and treat "stand still" as out of scope. **Do not** patch it by relabelling `neutral`
   into the motion set (§3.1). If it still fails *after* `standing_idle` arrives, check
   first whether the delivered clips actually contain breathing motion — a held pose would
   teach a frozen idle and would show up as a near-zero MAE in the G-N output.
5. **Under-fitting at the ep-2 gates** → raise LR to 5e-5 (evidence #4's documented
   fallback) before changing anything structural.
