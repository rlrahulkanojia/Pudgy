# v9b-1 — skit scenes on `pudgy-v8-merged`: run report

**Run** 2026-10-27 06:45 → 2026-10-28 08:01 UTC · **Hardware** 1× A100-SXM4 80GB (Vast.ai
C.57924132, not persistent) · **Status** **stopped at step 200 of 400; failed the gate: no
checkpoint passes, so the stage is not kept.** Tier 2 was not run. Written 2026-10-28.
Plan: [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md) §0.1 ·
handover: [`HANDOVER_v9b.md`](../../../training_approach/v9/HANDOVER_v9b.md) ·
runbook and dated log: [`RUNBOOK_v9b.md`](../../../training_approach/v9/RUNBOOK_v9b.md) §9 ·
audit: [`SKIT_AUDIT_v9b.md`](SKIT_AUDIT_v9b.md) · previous run: [`REPORT_v9c_phone.md`](REPORT_v9c_phone.md).

---

## 1. Headline

| | |
|---|---|
| **Outcome** | **FAIL.** `gate_v9b.py`: `NO CHECKPOINT PASSES`, winner `null` (steps 100 and 200 scored) |
| **Stopped** | after step **200 of 400**. HANDOVER_v9b §4 Step 8: "If step 200 fails R1, stop and report." Steps 201–400 were not run |
| **Learned (L1, held-out scene loss)** | **0.881×** base at step 100 and **0.866×** at step 200 (loss 0.01161 → 0.01028 → 0.01008): **11–13% lower** ✓ |
| **Regression, expressions (R1)** | held-out accuracy **13 → 11** (step 100) and **13 → 12** (step 200) of 16; margin CI low bound −0.044 / −0.046 against a bar of −0.003 ✗ |
| **Regression, motion (R1)** | held-out accuracy **9 → 6** and **9 → 7** of 12; margin CI entirely below zero at both steps ✗ |
| **Regression, per noise level (R2)** | v8-label loss **+6.6%** (step 100) and **+6.1%** (step 200) at t = 0.85, against a bar of +5% ✗ |
| **Uses the caption (P2)** | text-attributable gain **−0.0008** at both steps, 95% CI spans zero ✗ |
| **Generated clips (Tier 2)** | **not run**: the scene suite runs only on a gate winner, so there are no sample videos |

**In one line:** the LoRA fits the skit shots (scene loss −13%), but the gain doesn't come through
the captions, and it costs v8's expressions and motion. The damage was already complete at step
100, the earliest checkpoint scored.

---

## 2. What was trained

| | |
|---|---|
| Base | Wan2.2-I2V-A14B; LoRA on the **low-noise expert** only (t ∈ [0, 900]) |
| Start | `--base_weights pudgy-v8-merged` (md5 `ec333748284410bb341f431b88f5026f`, verified) |
| LoRA | rank 32 / alpha 32, adamw8bit, lr 5e-5, cosine → ×0.1 (`cosine_with_min_lr`), warmup 20, grad-accum 8, shift 5.0, fp16, seed 42 |
| Steps | **400 planned, 200 run** (8 clips each = 1,600 samples ≈ **3.0 epochs** of 535 clips); saved every 50 |
| Data | `processed/v9b_535` (all 571 SHA256SUMS verified; registry md5 `3757257814f5588603317d747f2e250e`): **348 scene clips** + **187 v8 replay clips (35%)** |
| Buckets | scene, 480×848 (9:16): f21 **160** · f33 **72** · f49 **82** · f81 **34**. Replay, 1024²: f13 34 · f17 7 · f21 74 · f25 15 · f29 22 · f33 2 · f37 13 · f41 4 · f49 13 · f57 3 |
| Holdout | 6 whole skits, never trained (`scenes/holdout_v9b.json`); Tier 1 scores **15 held-out scene windows**, plus the v8 label clips (held out: 16 expression, 12 motion; seen: 16, 11) |
| Captions | compiled per span from the curators' beats (`scenes/captions_v9b.py`), linted against v8's action clauses (D10) |
| Memory / speed | smoke test on the heaviest bucket (`replay_f57`, 61,440 tokens): `BLOCKS_TO_SWAP=24`, **peak 77,967 MiB**, **57.3 s/it**, 0 mapping-OOM lines, avr_loss 0.000552 |
| Pace in the run | **187 s per optimiser step** (≈ 23 s per clip). v9c-phone: 487 s. The plan's estimate for 400 steps was 30–45 h; this pace gives ≈ 21 h |
| Loss (avr_loss) | 0.0061–0.0073 (steps 1–10) → 0.00691 (50) → 0.00712 (100) → 0.00639 (150) → **0.00625** (200). Scene and replay clips mixed, so not comparable with v9c's 0.001–0.002 |
| Memory in the run | 0 mapping-OOM, 0 OOM over 200 steps; 77.5 GB in use |

The data, registry, recipe and gate were fixed before the run and were not changed.

---

## 3. Results

### 3.1 Tier 1 — diffusion classifier (k = 2, paired noise, same clips as the baseline)

Accuracy (correct / n) and mean margin of the true label over the others. *Scene* = the 15
held-out skit windows, true caption against 2–3 distractors (beats reversed, another window's
action, leads swapped), so chance is 0.25–0.33.

| config | holdout expr | holdout motion | seen expr | seen motion | holdout scene | scene loss |
|---|---|---|---|---|---|---|
| v8merged (base) | **0.812** (13/16) · +0.0366 | **0.750** (9/12) · +0.0107 | 1.000 (16/16) · +0.0631 | 0.909 (10/11) · +0.0127 | 0.400 (6/15) · −0.0013 | 0.01161 |
| v9b@100 | 0.688 (11/16) · +0.0183 | **0.500** (6/12) · +0.0032 | 1.000 (16/16) · +0.0380 | 0.909 (10/11) · +0.0085 | 0.467 (7/15) · +0.0013 | 0.01028 |
| v9b@200 | 0.750 (12/16) · +0.0189 | 0.583 (7/12) · +0.0014 | 0.938 (15/16) · +0.0305 | 0.818 (9/11) · +0.0084 | 0.400 (6/15) · +0.0022 | **0.01008** |

- **Scene loss fell 11% by step 100 and 13% by step 200.** Most of the gain came in the first
  half.
- **Picking the right scene caption didn't improve:** 6/15 → 7/15 → 6/15, near chance at every
  step. The lower loss holds under any caption.
- **v8 margins roughly halved.** Held-out expressions: +0.0366 → +0.018. Held-out motion:
  +0.0107 → +0.003 / +0.001, so the model barely prefers the true motion any more. The *seen*
  split drops too (expression margin +0.063 → +0.031), so this isn't only a held-out effect.
- **Step 100 is no better than step 200.** Held-out motion is lowest at step 100 (6/12).

### 3.2 Gate (`gate_v9b.py`, rule fixed before the data)

| check | v9b@100 | v9b@200 | bar |
|---|---|---|---|
| R1 expression: Δmargin [95% CI]; accuracy | −0.0182 [−0.0445, −0.0010]; 13 → 11 | −0.0176 [−0.0465, +0.0019]; 13 → 12 | CI low ≥ −0.003 **and** ≤ 1 clip lost |
| R1 motion | −0.0075 [−0.0111, −0.0042]; 9 → 6 | −0.0093 [−0.0134, −0.0052]; 9 → 7 | same |
| R2 loss ratio at t = 0.65 / 0.85 | 1.0383 / **1.0659** | 1.0297 / **1.0611** | each ≤ 1.05 |
| L1 scene loss ratio | **0.881** ✓ | **0.866** ✓ | < 1.00 |
| P2 text-attributable gain [95% CI] | −0.0008 [−0.0063, +0.0051] | −0.0008 [−0.0070, +0.0062] | CI high < 0 |
| **verdict** | **fail** (R1 ×2, R2, P2) | **fail** (R1 ×2, R2, P2) | |

**Per-label Δmargin** (reported, not gated; n = 4–7 clips per label):

| | step 100 | step 200 |
|---|---|---|
| **motion** | jumping −0.0062 · running −0.0076 · sitting −0.0078 · walking −0.0101 · waving −0.0053 | jumping −0.0128 · running −0.0115 · sitting −0.0050 · walking −0.0114 · waving −0.0099 |
| **expression** | **crying −0.1010** · confused −0.0281 · happy −0.0204 · laughing −0.0071 · blushing −0.0057 · angry −0.0035 · surprised +0.0017 · **neutral +0.0185** | **crying −0.1113** · confused −0.0307 · happy −0.0167 · angry −0.0079 · blushing −0.0009 · laughing +0.0004 · surprised +0.0011 · **neutral +0.0249** |

- **Motion: all five labels drop, at both steps** (−0.005 to −0.013). D10's warning sign, one
  clause absorbing the others, would show as one label falling far more than the rest. Here the
  drop is broad.
- **Expression: crying drops by −0.10 to −0.11**, about 3× the next label (confused, −0.03).
  Neutral is the only label that rises (+0.02).

### 3.3 Tier 2 — not run

`v9b_box.sh eval` generates the scene suite (8 jobs; 49 frames at 480×848; seeds 42/43/44) only
for the gate winner. With no winner there are no Tier-2 clips or contact sheets. The 15
evaluation windows and 8 suite jobs are built and on Azure, ready for a later stage.

---

## 4. Wall clock (UTC)

| phase | time | duration |
|---|---|---|
| setup (Step 2) + archive copy from the old box (Step 3) | 27 Oct 06:45–07:05 | 20 min (2.66 GB in 9 min 48 s) |
| cache (Step 4) | 07:05–07:35 | 30 min |
| evalprep (Step 5) | 07:35–07:36 | 1 min |
| smoke test (Step 6) | 07:36–07:49 | 13 min |
| baseline (Step 7; v9c's regression rows reused, diff 0.00e+00) | 07:49–08:08 | 19 min |
| **training, steps 1–200** (Step 8) | 08:08–18:38 | **10 h 30 min** |
| Tier 1 + gate on v9b@200 | 18:38–20:05 | 87 min (5,219 s) |
| idle, waiting for a decision | 20:05–06:36 | 10 h 31 min |
| Tier 1 + gate on v9b@100 (user request) | 28 Oct 06:36–08:01 | 85 min (5,107 s) |

**≈ 25.3 h on the box, ≈ 14.4 h of it with the GPU busy.** Steps 4–8 and the step-200 check ran
back to back from a script (`logs_v9b/run_chain_v9b.sh`), so the GPU didn't wait between steps.

---

## 5. Problems and deviations

| item | effect | handling |
|---|---|---|
| torch **2.14.1+cu130** (the handover lists a CUDA 12.x driver) | none. `v9_box.sh setup` matches the wheel to the driver (CUDA 13.0); A100 is supported | noted in the runbook; not a fixed item |
| no git author identity on the new box | first commit failed | repo-local identity, matching the old box's commits |
| sync-loop PID file not written (transient "No such file") | none; the loop was running | PID file written by hand |
| `Killed` line in `train.log` at step 200 | expected. `pause` waits until the step-200 state stops growing, sends SIGINT, then SIGKILL after 30 s | state complete: 5 files, 884 MB, pushed to Azure with size checks |
| the monitor mirrors `.safetensors` only, not training states | the step-200 state was not on Azure | pushed by hand (28 Oct) |
| memory | none: 0 mapping-OOM and 0 OOM in the smoke test and over 200 steps | — |

No data, recipe or gate change was made. The old box was used read-only, only to copy the
handover archive.

---

## 6. Where everything is (Azure `pudgytraining`, container `pudgy`)

| what | where |
|---|---|
| checkpoints 50 / 100 / 150 / 200 (613.6 MB each) | `v9b/weights/pudgy-v9b-lownoise/` |
| step-200 training state (resume point, 884 MB) | `v9b/weights/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000200-state/` |
| Tier-1 results and gate | `v9b/eval/runs/pudgy-v9b-lownoise/` (`dcls/v8merged_k2.json`, `dcls/v9b@100_k2.json`, `dcls/v9b@200_k2.json`, `winner_v9b.json`) |
| eval windows, suite jobs, latents | `v9b/eval/` |
| box logs (train, tier1, chain, smoke, monitor, sync) | `v9b/logs/box/` |
| training set | `processed/v9b_535` |
| code, docs, reports | `v9b/work/` |
| on the box (stopped, so the disk is kept but billed) | `/workspace/wan_output/pudgy-v9b-lownoise/`, `/workspace/eval_v9b/`, `/workspace/logs_v9b/` |

The repo changes (runbook log, this report, the plan's status) are committed on the box's local
`main` and not pushed to GitHub. A git bundle of those commits is on Azure at
`v9b/work/git/v9b-1_main.bundle`.

---

## 7. Findings and options

**Findings** (measured):
1. **The fixed recipe trades v8 skills for scene fit early.** All of the regression is there by
   step 100. A shorter run of this recipe doesn't rescue it, at least down to step 100; step 50
   was not scored.
2. **The scene gain doesn't go through the text** (P2 ≈ 0; choosing the right caption stays at
   6–7/15). This is the same pattern as v9c-phone, where 99.6% of the gain didn't depend on the
   caption: the model learns how the clips look, not what the words say.
3. **35% replay didn't protect motion** (v9c-phone had 20%). Held-out motion falls to 0.58 at
   step 200, the same as v9c-phone's 0.58, and the drop is spread across all five labels.
4. **R2 is worst at t = 0.85**, the noisy end of the low-noise expert, as in v9c-phone
   (1.079 at t = 0.85).

**Hypotheses** (not tested):
- Neutral faces rising while crying falls suggests the skit shots pull expressions toward neutral.
  The test is to count expression words, and how often each appears, in the 348 scene captions.
- The broad motion drop may come from the scene clips' static camera and slow movement rather
  than from any one caption clause.

**Options for the next decision** (each changes the fixed recipe or data, so it needs the user's
sign-off):
- a. Score v9b@50 (≈ 85 min on a GPU) to see whether any point of this run passes.
- b. The embedding-first route planned for v9c2: the network stays frozen, so v8 prompts can't
   change. It may not be enough for two-character layout.
- c. A weaker or shorter LoRA (lower lr or rank, fewer steps) or a different replay mix, each
   gated at step 50 and step 100.
- d. Review the scene captions against D10 for expression and motion words before any re-run.
