# v10 Stage 1: skit scenes on the high-noise expert, run report

**Run:** 2026-11-10 14:35 → 2026-11-13 14:21 UTC. The box was stopped on 2026-11-14.

**Hardware:** 1× A100-SXM4 80 GB on Vast.ai, instance 60069260. Its `/workspace` is not a separate volume.

**Status:** **trained 300 of 300 steps. Tier 1 holds at every checkpoint, so the candidate is v10@300. Not promoted.** On 2026-11-12 the user skipped the candidate gates, G1 and Phase 0 #1 and used the GPU for full-length skit remakes instead, so production is still v8merged.

Written 2026-11-21.

Related documents:
- Plan: [`Training_Approach_v10.md`](../../../training_approach/v10/Training_Approach_v10.md), §0 (decisions), §4 (recipe), §5 (gates)
- Handover: [`HANDOVER_v10.md`](../../../training_approach/v10/HANDOVER_v10.md)
- Previous run: [`REPORT_v9b.md`](../v9/REPORT_v9b.md)
- Client write-up: ["Pudgy v10 training run: report"](https://claude.ai/artifact/C3fQCnpQGzPBDRuWiijhxL)

**Glossary.** Run-specific terms, defined in the plan:

| Term | Meaning |
|---|---|
| v10@N | the v10 LoRA at training step N, used on top of v8merged |
| Tier 1 | the loss-based checks run on each checkpoint |
| G1–G5 | the promotion gates (§3.2) |
| R2 | the per-noise-level v8-label check |
| Phase 0 | evaluation runs: #1 compares the old v9b checkpoints; #2 is the v8merged baseline |
| E5 | the frozen phone embedding |
| golden | the v2 golden high-noise LoRA (`lora_highnoise_GOLDEN_ep40`) |
| solo / solo_watch | the box's automatic pipeline |
| FLF | generating a shot from its first and last frames |

---

## 1. Headline

| | |
|---|---|
| **Outcome** | **Tier 1 holds at all 6 checkpoints** (steps 50–300). No early stop. `tier1/summary.json` names **v10@300** as the candidate |
| **Fits held-out skit scenes** | Loss on the true captions is **0.943× v8merged's** at step 300 (**−5.7%**). It improved at every checkpoint: 0.982 → 0.959 → 0.952 → 0.946 → 0.944 → 0.943 |
| **v8 prompts still work (G4 fast check)** | Loss ratio on v8's label clips, at t = 0.95 / 0.92: expressions **0.908 / 0.935**, motion **0.936 / 0.931** (bar ≤ 1.05). v10 is **better** on v8's own clips, not just no worse ✓ |
| **Uses the caption (G3)** | Gain **−0.0008**, 95% CI [−0.0096, +0.0083]. **No change** at any step. True-caption accuracy is 0.510 for v8merged and 0.510 for v10@300 |
| **Full-length remakes** | 10 held-out skits, 99 s each way: **55 of 59 parts pass QA for both v10@300 and v8merged**. Both needed 72 generations. By eye they look alike |
| **Promotion gates** | **Not run:** G1 (blind preference), G2, the full G4, G5 and Phase 0 #1. The user skipped them on 2026-11-12 |

**In one line:** v10 learned the skits' look without costing v8's expressions and motion, the
opposite of what happened to v9b-1. But the gain is in fitting the footage, not in following
captions, and on full-length remakes it isn't yet visible by eye.

---

## 2. What was trained

| | |
|---|---|
| Base model | Wan2.2-I2V-A14B. One new LoRA on the **high-noise expert** only (t ∈ [900, 1000]). The low-noise expert is `pudgy-v8-merged`, frozen |
| Starting point | `--base_weights lora_highnoise_GOLDEN_ep40` × 1.0, merged into the high-noise expert, as v8 did with v7fixed |
| LoRA settings | Rank 16, alpha 16, all linear layers. Optimizer adamw8bit, learning rate 3e-5, 20 warmup steps, then cosine decay to 0.1× (`cosine_with_min_lr`). Gradient accumulation 8, max grad norm 1.0, shift 5.0, fp16, seed 42, 24 blocks swapped to CPU |
| Steps | **300 of 300**, each over 8 samples (2,397 samples ≈ **1.57 epochs** of 1,525 clips). Saved every 25 steps, with optimizer state |
| Preservation loss | λ = 1 on replay-marked samples. The target is the golden-merged expert's own prediction with the LoRA turned off (`asset_embed_v9.py train --preserve --preserve-marker key`) |
| Phone | E5 (`pudgy-v9c2-asset-dual-step00000150.asset`), frozen and spliced in, as at inference |
| Data | `processed/v10_1525`: **991 scene windows** (643 iteration_7 windows from 138 skits + 348 v9b windows from 78 skits) and **534 replay clips** from deliveries 3–6 (35%). The cache check passed: `replay=534 scene=991 phone=80 errors=0` |
| Scene clip lengths | 480×848, 24 fps: 467 at 21 frames, 213 at 33, 221 at 49, 90 at 81. No 121-frame clips (Q3) |
| Held out | 16 whole iteration_7 skits (71 windows) plus 6 skits from earlier deliveries |
| Samples seen | `PRESERVE_STATS` at step 300: total 2,397 = 1,552 scene + 845 replay. 119 of them carried the phone embedding |
| Final loss | Average training loss 0.017 |

**Weight trajectory (Tier 0, `logs/pudgy-v10-highnoise/trajectory.json`).**
- The LoRA's weight norm grew as follows: 0.35 at step 25, 1.10 at 100, 1.47 at 200, 1.55 at 300.
- Its cosine similarity to the final weights was 0.84 at step 100 and 0.99 at step 200.
- So the direction of the update was settled by step 200, and the last 100 steps mostly grew it a little.

---

## 3. Results

### 3.1 Tier 1, every 50 steps (k = 2, paired noise, held-out clips; `tier1/gate_step*.json`)

| Step | G4 fast: expression, t 0.95 / 0.92 | G4 fast: motion, t 0.95 / 0.92 | G3 gain [95% CI] | True-caption loss ratio | Caption accuracy (v8merged → v10) | Holds |
|---|---|---|---|---|---|---|
| 50 | 0.976 / 0.982 | 0.983 / 0.966 | +0.0006 [−0.0019, +0.0032] | 0.982 | 0.510 → 0.549 | ✓ |
| 100 | 0.930 / 0.948 | 0.955 / 0.937 | +0.0009 [−0.0067, +0.0088] | 0.959 | 0.510 → 0.588 | ✓ |
| 150 | 0.920 / 0.944 | 0.948 / 0.936 | +0.0008 [−0.0077, +0.0098] | 0.952 | 0.510 → 0.529 | ✓ |
| 200 | 0.913 / 0.938 | 0.943 / 0.937 | −0.0012 [−0.0096, +0.0073] | 0.946 | 0.510 → 0.529 | ✓ |
| 250 | 0.912 / 0.938 | 0.938 / 0.934 | −0.0008 [−0.0094, +0.0079] | 0.944 | 0.510 → 0.529 | ✓ |
| **300** | **0.908 / 0.935** | **0.936 / 0.931** | **−0.0008 [−0.0096, +0.0083]** | **0.943** | 0.510 → 0.510 | ✓ |

**How to read the columns:**
- **G4 fast check:** v10's high-noise loss divided by v8merged's, on v8's held-out label clips (16 expression clips, 12 motion clips). The pass bar is ≤ 1.05 for each kind and each t.
- **G3:** run on 51 held-out scene windows. It compares the loss with the true caption against distractor captions (beat order swapped, leads' actions exchanged, another shot's action).
  - The gain is the improvement attributable to the text.
  - It is reported, not gating (Q13). It would stop the run only if the gain were > 0 with a CI that excludes 0.
  - The true-caption loss ratio is v10 against v8merged on the same windows.
- **Early stop** at step 100 (G4 fast check fails at both 50 and 100, or G3 worse): **not triggered**.

G3 at step 300, by subset. Every interval spans 0:

| Subset | n | Gain [95% CI] |
|---|---|---|
| v10 windows | 36 | +0.0023 [−0.0089, +0.0142] |
| v9b windows | 15 | −0.0082 [−0.0230, +0.0062] |
| phone | 17 | +0.0044 [−0.0182, +0.0277] |
| camera move | 5 | +0.0017 [−0.0066, +0.0100] |

Most of the gain arrives by step 100–150 and flattens after 200. The expression ratio was still
edging down at step 300, by 0.004 per 50 steps.

### 3.2 Gates and phases not run

The plan (§5.1) runs the full G2, G4 and G5 on the candidate, then G1 on a candidate that passes
them. The user skipped all of these on 2026-11-12 (20:50 UTC) to free the GPU for full-length
remakes. So the following are **not measured**:

| Gate or phase | What it would have measured |
|---|---|
| **G1** | Blind human preference, 102 pairs, pass ≥ 60 |
| **G2** | Duplicate leads, zoom, motion, identity and rendered text on generated clips |
| **G4, full** | The 52-prompt label check and compose motion |
| **G5** | Phone identity, ≤ 1 of 24 |
| **Phase 0 #1** | Tier 2 on v9b@100/200, which informs Stage 2 |

`solo_watch` was stopped right after the baseline. Its completion flags are kept, so a restart
resumes at the candidate gates (§7).

### 3.3 Baseline (Phase 0 #2): done, v8merged side only

The baseline ran from 2026-11-11 22:35 to 2026-11-12 23:06 UTC, minus a 1 h 40 min pause for
the preview below. It produced v8merged's I2V clips on the held-out G1 shots (3 seeds), the FLF
report clips, 42 contact sheets, and the v8merged side of G4 (`g4/core`, `g4/compose`,
`g4/scores.json`).

The v10 side was never generated, so these are the "before" half only. They were uploaded to
Azure on 2026-11-14, just before the box was stopped. Azure holds 246 files under the run's eval
prefix, all size-checked.

### 3.4 Preview (2026-11-12)

The baseline was paused once, from 22:45 to 00:26 UTC, for an early look:
- 4 start frames × (v10@300, v8merged), at 480×848, 81 frames each; 8 of 8 completed.
- The clips were labelled, given side-by-sides, and published to the dashboard under Generate videos → Saved videos.
- They are also on Azure at `v10/preview/2026-11-12/`.

### 3.5 Full-length remakes of 10 held-out skits

The 10 skits were sleepless, leftovers, highlighter, beach_activities, working_in_bed, blossom,
charging, gentle_one, security and dyli. Each was remade end to end with v10@300, then with
v8merged, using the `longform_client/remake.py` method:
- Every shot is generated FLF, using the skit's own frames.
- Each shot is colour-matched to the original, joined at the original cuts, and laid over the original soundtrack.
- A part uses its training caption where one exists, including curation captions for spans that were dropped from training. The other 20 of the 59 parts got prompts written by hand from the frames. Batch 2 joined spans of 1–4 frames directly instead of generating them.

The scripts are `remake_v10.py`, `remake_run.sh` and `remake_run2.sh`, under Azure `v10/remakes/2026-11-13/`.

**QA per part.** A part passes when all three hold:
- frames showing a duplicate Pax or Polly ≤ the original's count + 2;
- the largest frame-to-frame jump (÷ median) < max(5, 1.5 × the original's);
- the landing distance to the original's next frame < 12.

Seeds 42, 1042 and 2042 are tried in turn until one passes. If none passes, the best take is kept.

| Skit | Length | Parts | v10@300 passing | v8merged passing |
|---|---|---|---|---|
| sleepless | 6.1 s | 2 | 2/2 | 1/2 |
| leftovers | 14.1 s | 7 | 7/7 | 7/7 |
| highlighter | 5.6 s | 4 | 4/4 | 4/4 |
| beach_activities | 7.2 s | 5 | 4/5 | 4/5 |
| working_in_bed | 9.1 s | 4 | 3/4 | 4/4 |
| blossom | 13.7 s | 9 | 8/9 | 8/9 |
| charging | 9.2 s | 7 | 7/7 | 7/7 |
| gentle_one | 16.6 s | 9 | 8/9 | 8/9 |
| security | 8.3 s | 5 | 5/5 | 5/5 |
| dyli | 9.1 s | 7 | 7/7 | 7/7 |
| **Total** | **98.9 s** | **59** | **55/59** | **55/59** |

**Generations.** Each model needed **72 generations** for the 59 parts: 13 extra seeds spread over 9
parts. Generation time was 7.7 h for v10@300 and 7.4 h for v8merged.

**Failed parts:**

| Part | v10@300 | v8merged | Cause |
|---|---|---|---|
| beach_activities 0058–0072 | jump 5.6 | jump 7.2 | a 0.6 s whip pan (15 frames), with a hand-written prompt. The original's jump is 1.2 |
| blossom 0117–0142 | jump 8.7 | jump 6.9 | a time-lapse, with a hand-written prompt |
| gentle_one 0065–0120 | landing 17.5, 23 extra-Polly frames | landing 16.7, 22 extra-Polly frames | a street walk: Pax slips behind Polly and the street changes mid-shot. All 3 seeds failed for both models |
| working_in_bed 0000–0072 | 33 second-Pax frames (original: 12) | passes | the detector also fires on the original; judged by eye a false positive |
| sleepless 0000–0073 | passes | jump 6.6 | failed with all three seeds |

**By eye** (side-by-sides of original | v8merged | v10):
- The two are alike overall: leads on model, sets and props in place, original timing kept.
- **Blossom:** v8merged's Polly turns briefly see-through; v10's stays solid.
- **working_in_bed:** v10's Pax looks at the camera, and an extra notebook appears. v8merged is closer to the original here.
- **Small invented details from both models:** the phone screen lights up in sleepless, and a stray pink mark appears in highlighter.

**Dashboard (PR #24).** The main page makes v10 the latest experiment, "Skit Scenes Wan 2.2". It
shows 9 remakes with sound and 9 three-way side-by-sides. gentle_one is left out by the user's
choice, because of the street-walk shot.

---

## 4. Wall clock (UTC)

| When | What |
|---|---|
| 11-10 14:35 → 16:01 | Cache (`solo_watch` restarted at 14:58 with the keep-busy changes; the interrupted cache resumed by itself) |
| 16:01 → 16:30 | Eval prep and the Tier-1 baseline rows (`v8merged_k2.json`). The smoke test failed straight away (the `.env` problem, §5) |
| 16:30 → 16:39 | **4 failed training starts, each followed by a failed smoke retry.** Same cause |
| 16:47 → 16:58 | Restart on `0281c96`, then the smoke test |
| 16:58 → 11-11 01:52 | **Train to step 100**: 8 h 46 min of steps, 316 s per step |
| 01:52 → 02:46 | Tier 1 at steps 50 and 100, 27 min each. No early stop |
| 02:47 → 20:47 | **Train steps 101–300**, resumed from the step-100 state: 17 h 55 min of steps, 323 s per step |
| 20:47 → 22:35 | Tier 1 at steps 150, 200, 250 and 300, 27 min each |
| 22:35 → 11-12 23:06 | Baseline (Phase 0 #2), with the preview between 22:45 and 00:26 |
| 11-12 23:06 → 11-13 05:56 | Remakes, batch 1 (5 skits, 22 parts, both models) |
| 11-13 05:56 → 14:21 | Remakes, batch 2 (5 skits, 37 parts, both models) |
| 11-13 14:21 → 11-14 ~14:40 | Idle, waiting on the user. Then the baseline upload and the box stop |

**Training totals.** Steps took **26 h 42 min** in all, about **320 s per step**. The plan
estimated 16–20 h from v9b's 187 s per step and left the preservation pass unmeasured. The most
likely cause of the extra time is that pass: one more no-grad forward on each replay sample. This
was not profiled.

**A misleading speed reading.** After the resume, the progress bar and `status.json` show
"300/300 [17:55:14, 215.05 s/it]". That rate divides the second session's time by all 300 steps,
so it understates the true speed of about 320 s per step.

---

## 5. Problems and deviations

| Problem or deviation | Detail |
|---|---|
| **One GPU instead of two** (the user's choice on 2026-11-10, when the box had one A100) | The plan (Q6) assumed GPU 0 trains while GPU 1 evaluates. With one A100, `solo` ran everything in sequence: train to 100, Tier 1 at 50 and 100, train to 300, Tier 1 at 150–300, then the baseline. Phase 0 #1, planned to run alongside training (Q18), never ran |
| **`.env` sourcing killed training starts** (11-10 16:03–16:40) | `train_pudgy_v10.sh` sourced the repo `.env`. That file is a notes file, not a shell file, and under `set -e` it killed every smoke test and training start. Fixed in `0281c96`: it now reads `NAME=value` lines only |
| The same commit fixed a silent failure | `[ -f done ] \|\| { …; cmd && touch done; }` disables `set -e`, so a failed command was skipped without an error. It now uses explicit `if`s |
| **A credential reached the logs** | Before the fix, the W&B key was echoed into `smoke.log` and `train.log`, which are synced to Azure `v10/logs/box/`. The user was told to rotate it |
| **The preview's inference server failed once** (11-11 22:38) | The baseline resumed, was paused again at 22:45 once the server was fixed, and the preview then ran to 00:26 |
| **Change of plan by the user** (11-12 20:50) | Skipped the candidate gates, G1, Phase 0 #1 and the dashboard go-live. Instead: full-length remakes of 5 held-out skits, extended to 10 on 11-13. A stopper ended `solo_watch` once the baseline finished, at 11-12 23:06 |
| **gentle_one left off the client dashboard** | The user chose this; it is still in the QA totals above |
| **No git bundle in `v10/git`** | The `git` step never ran. Nothing was lost: the box's repo was clean at `0281c96`, which is on GitHub (PR #21, merged 11-13) |

---

## 6. Where everything is (Azure container `pudgy`)

| What | Path |
|---|---|
| Weights: steps 25–300 plus final, each with `.asset` and `-state` | `v10/weights/pudgy-v10-highnoise/`. Candidate: `pudgy-v10-highnoise-step00000300.safetensors` + `.asset.safetensors` |
| Tier 1 | `v10/eval/runs/pudgy-v10-highnoise/tier1/` (`summary.json`, `gate_step{50..300}.json`, `v10_{step}_k2.json`, `v8merged_k2.json`) |
| Baseline (v8merged) | `v10/eval/runs/pudgy-v10-highnoise/suite/` and `g4/` |
| Remakes | `v10/remakes/2026-11-13/`: `final/{v10_300,v8merged}/`, `compare/`, `takes/`, `qa/`, `remake.log`, scripts |
| Preview | `v10/preview/2026-11-12/` |
| Training logs and trajectory | `v10/logs/pudgy-v10-highnoise/` (`train.log`, `status.json`, `trajectory.json`, tensorboard) |
| Box logs | `v10/logs/box/` (`solo.log`, `train.log`, `smoke.log`, `gpu1_gen.log`, `sync.log`, …) |
| Code and docs snapshot | `v10/work/` |
| Dashboard videos | container `pudgy-dashboard`: `v10/Full-length_remakes/`, `v10/Side_by_side/`, and Saved videos under `inference/jobs/` |

**Only on the stopped box:** the small helper scripts in `/workspace/infer_api` (preview,
go-live, stopper and `serve_main.sh`). A stop keeps them; a recycle or destroy would delete them.

---

## 7. Findings and options

**Findings:**
1. **The high-noise route works where v9b-1's low-noise route failed.** Training scenes into the high-noise expert, with replay and the preservation loss, made v8's own label clips fit *better* (0.91–0.94). v9b-1 lost held-out expressions and motion by step 100.
2. **The gain is in fit, not in caption following.** The true-caption loss is 5.7% lower, but G3 shows no change at any step and in every subset. Captions are not being followed more closely than with v8merged.
3. **At this sample size the gain isn't visible by eye.** The remakes pass the same 55 of 59 QA checks. The differences go both ways and are small.
4. **The hard shots are hard for both models:** whip pans, time-lapses, and a two-character walk in which the background changes.

**Options. The user decides; none has been started:**
- **Finish the promotion checks on v10@300.**
  - Restart instance 60069260 from the Vast console, then run `tmux new-session -d -s solo "bash /workspace/Pudgy/finetune/wan/v10_box.sh solo_watch; echo SOLO_WATCH_EXIT; exec bash"`. It resumes at the candidate gates (full G2, G4 and G5), then builds the G1 page.
  - The baseline is already done. Generating the v10 side should take about as long as the baseline did, roughly 23 h (an estimate, not measured).
  - This is the only route to replacing v8merged in production (Q1).
- **A longer Stage 1.** The ratios were still edging down at step 300, but the weight direction had settled by step 200 (cosine 0.99), so expect little more from extra steps.
- **Stage 2 (low-noise).** Plan §4 makes it conditional: Stage 1 must pass, a detail gap must be shown, and the user must approve. It needs G1 or Phase 0 #1 evidence first.
- **Caption following.** If text following is the goal, it needs a different lever; v10's data and recipe did not move G3.
