# v10 handover: Stage 1 (high-noise) on a new 2-GPU machine

**Written** 2026-11-10 on the Mac that holds the data. **For** whoever runs v10 on the GPU box, a
person or a Claude session. This file is the single entry point (decision Q9). Read §0 and §1
first; the plan's "why" is [`Training_Approach_v10.md`](Training_Approach_v10.md), and its §0
lists every decision (Q1–Q20) this run follows.

---

## 0. The project in one page

**Goal.** A video model that animates the client's 2D characters **Pax** (a short round blue
penguin) and **Polly** (a short round pink penguin) in the Pudgy Penguins style: thick clean
black outlines, flat pastel colours, hand-drawn timing. Output is 9:16 shots, assembled into
reels by a separate pipeline (`finetune/wan/longform_client/`). The client is the Pudgy
Penguins team (contact: Charlie, on the shared Slack); they deliver the footage.

**Who decides.** The user decides anything outside the written plan. A Mac session curates data,
writes plans and builds training sets. A GPU-box session (you) runs training and evaluation
and reports.

**The model line** (details in [`../README.md`](../README.md); reports under
`docs/training_reports/`):

| version | what | outcome |
|---|---|---|
| v1 | CogVideoX 5B, one LoRA | characters drift mid-clip; superseded |
| **v2** | **Wan2.2-I2V-A14B**: a LoRA per expert (high-noise = layout and motion, low-noise = detail) | passed; its **high-noise "golden" LoRA** (`lora_highnoise_GOLDEN_ep40`) is still production |
| v3, v4 | AniSora and LTX-2.3 tracks | hedges, not production |
| v5 | expressions on the high-noise expert | **collapsed prompt control**: the main risk this run guards against |
| v6, v7 | expressions and motion on the low-noise expert | v7fixed became the low-noise base |
| **v8** | one joint low-noise LoRA (5 motions, 8 expressions) | Tier-1 win; **`pudgy-v8-merged`** (v7fixed + v8@550) is the production low-noise expert |
| **v9c2** | the iFin phone | solved by a registry phrase plus a learned embedding **E5**, adopted |
| v9b-1 | scenes from 99 skits, a **low-noise** LoRA | **failed**: v8 motion and expressions regressed by step 100 and the gain ignored the caption ([`REPORT_v9b.md`](../../docs/training_reports/v9/REPORT_v9b.md)) |
| **v10** | **this run**: scenes from 186 new clean skits + v9b's, a **high-noise** LoRA with the low-noise expert frozen | — |

Isolated experiments outside the line (LTX-2.5, Kandinsky 5, MiniMax H3) didn't beat v8.

**Production today:** low-noise `pudgy-v8-merged`, high-noise v2 golden, E5 on the phone phrase;
generation through `finetune/wan/eval_v9/gen_asset_v9.py`. If v10 fails its gate, production
stays exactly this.

**Where things live.**

| what | where |
|---|---|
| code and docs | GitHub `rlrahulkanojia/Pudgy` (`main` once the v10 PR is merged; until then branch `v10/curation-and-captions`). The Wan line is `finetune/wan/`; scene curation is `finetune/wan/scenes/` |
| client footage | `Data/raw/iteration_1`–`7` on the Mac, mirrored to Azure `raw/` (never edited) |
| training sets | `Data/processed/<set>` (each with a README); v10's is `processed/v10_1525` |
| weights, eval, logs | Azure storage account `pudgytraining`, container **`pudgy`**, one prefix per run (`v8/`, `v9b/`, `v9c2/`, **`v10/`**) |
| secrets | `/workspace/Pudgy/.env` (`AZURE_STORAGE_CONNECTION_STRING`, optional `WANDB_API_KEY`), from the user; git-ignored |

**What v10 is.** v9b-1 showed that putting scene behaviour into the low-noise expert erodes v8's
labels, and v9c2's probe showed a shot's layout is settled in the high-noise phase. So v10 trains
a fresh **high-noise** LoRA on production skit shots and leaves the low-noise expert (where v8's
labels live) untouched. The question it answers: does the client prefer v10's shots to today's,
blind, without losing v8's prompts or the phone?

---

## 1. What is fixed

**Don't re-curate, re-caption, rebuild the set or change the recipe or the gates.** All of it was
decided with the user (plan §0). If something looks wrong, stop and ask.

| | |
|---|---|
| Model | Wan2.2-I2V-A14B. A fresh LoRA on the **high-noise expert only**, t ∈ [900, 1000], single-expert musubi mode. Low-noise stays `pudgy-v8-merged`, frozen |
| Base | `lora_highnoise_GOLDEN_ep40` merged into the high-noise DiT with `--base_weights`, so the saved LoRA is a pure v10 task vector. At inference: high = golden + v10, low = v8merged |
| Recipe (Q16) | rank 16 / α 16, all linear layers, adamw8bit, lr 3e-5, warmup 20, cosine to ×0.1, grad-accum 8, **≤ 300 optimizer steps**, save every 25, `BLOCKS_TO_SWAP=24`, fp16, seed 42 |
| Preservation (Q10) | on replay samples the loss is ‖v_θ − v_base‖² (λ = 1), where v_base is the golden-merged expert with the LoRA off. Replay samples are marked in their text-encoder cache |
| Phone (Q17) | E5 frozen and spliced into the phone phrase during training, as at inference |
| Data | `processed/v10_1525`: **991 scene windows** at 480 wide × 848 tall, 24 fps (643 from iteration_7 + v9b's 348; f21 467 · f33 213 · f49 221 · f81 90; 88 camera moves) + **534 v8 replay clips** at 1024² (35%) |
| Holdout | 16 iteration_7 skits (`finetune/wan/scenes/holdout_v10.json`) + v9b's 6. None of their frames is in the set |
| Gates (Q1, Q12–Q15, Q19, Q21–Q24) | plan §5.1. G1 blind human preference decides; G2, G4, G5 guard; G3 is reported and can stop a run |
| Not in this run (Q20) | Stage 2 (low-noise) unless the user approves it after Stage 1; the keyframe LoRA (v10-K); end-to-end reels |

---

## 2. The machine

| need | why |
|---|---|
| **2 GPUs with 80 GB** (A100 or H100) (Q6), **or one** (Q25: the `solo` sequence, §4b) | with two, GPU 0 trains while GPU 1 runs the baseline, Phase 0 #1 and each checkpoint's Tier 1; with one, everything runs in turn. A clean training run reserves almost the whole 80 GB |
| RAM ≥ 128 GB | block swap, data loaders, caching |
| Disk ≥ 250 GB free on `/workspace` | ~65 GB base weights, caches, 12 checkpoints with states, eval clips |
| CUDA 12.x/13.x driver, ffmpeg, internet to Hugging Face, GitHub and Azure | weights from HF, code from GitHub, everything else from Azure |

`/workspace` is **not persistent** on Vast. Azure is the only backup, so push as you go (§5).

---

## 3. What to download

Everything comes from **Hugging Face** (base weights, fast), **GitHub** (code) and **Azure**
(container `pudgy`). Azure → box downloads measured only **0.2–0.4 MB/s** on 27 Oct, so the Azure
items are kept small (~2.5 GB needed first, ~1.2 GB later). `v10_box.sh pull` and `pull_v9b` do
all of it; the table is for checking.

| # | item | source | goes to | check |
|---|---|---|---|---|
| 1 | code and docs | GitHub `rlrahulkanojia/Pudgy`, `main` (after the v10 PR is merged; until then `v10/curation-and-captions`) | `/workspace/Pudgy` | `git log` shows the v10 commits |
| 2 | base weights (~65 GB): Wan2.2 I2V high- and low-noise DiT fp16, VAE, umT5 | Hugging Face, by `v9_box.sh setup` | `/workspace/wan_models` | the setup script checks |
| 3 | musubi-tuner **v0.3.4** + `finetune/wan/patches/musubi-v0.3.4-resume-step.patch` | GitHub, by `v9_box.sh setup` | `/workspace/musubi-tuner` | "patch already applied" on a re-run |
| 4 | merged v8 base (1.5 GB) | `v9b/base/pudgy-v8-merged.safetensors` | `/workspace/wan_output/v8_merged/` | md5 `ec333748284410bb341f431b88f5026f` |
| 5 | golden high-noise LoRA (0.3 GB) | `v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors` | `/workspace/wan_output/v2_golden/` | `[asset] base weights: 400/400 LoRA modules match` at smoke |
| 6 | E5 phone embedding (0.4 MB) | `v9c2/weights/pudgy-v9c2-asset-dual/pudgy-v9c2-asset-dual-step00000150.asset.safetensors` | `/workspace/wan_output/e5/` | shape (25, 4096) at smoke |
| 7 | **training set** `v10_1525` (217 MB, 1,558 files) | `processed/v10_1525/` | `/workspace/Data/processed/v10_1525`, linked as `/workspace/data_v10` | `SHA256SUMS`; 1,525 clips |
| 8 | **eval set** `v10_eval` (24 MB, 174 files) | `processed/v10_eval/` | `/workspace/Data/processed/v10_eval`, linked as **`/workspace/eval_v10`** (its paths are rooted there) | `SHA256SUMS` |
| 9 | v9 regression set with latents (G4's fast check) | `v9b/handover/eval_for_v9b.tar` (405 MB) | extracted in `/workspace` → `/workspace/eval_v9/holdout` | present |
| 10 | v9b@100, v9b@200 (Phase 0 #1 only; 1.2 GB) | `v9b/weights/pudgy-v9b-lownoise/pudgy-v9b-lownoise-step00000{100,200}.safetensors` | `/workspace/wan_output/pudgy-v9b-lownoise/` | present |
| 11 | secrets `.env` | **the user** (never in git or Azure) | `/workspace/Pudgy/.env`, mode 600 | `AZURE_STORAGE_CONNECTION_STRING` set |
| 12 | GitHub push access for `v10/box-run` (Q7) | **the user** (a deploy key or token) | the box's git credentials | `git push` works; without it the bundle on Azure is the record |

**Not needed:** raw footage (the set is final: prep ran on the Mac, Q11), `v9_box.sh restore` and
`merge`, the v8 and v9 keyframes (they are inside the eval set).

**The paths are fixed.** The scripts and the eval set hard-code `/workspace/...`.

---

## 4. Steps

Each step ends with what you must see. Log every step with UTC time and numbers in a new
`docs/training_reports/v10/RUNLOG_v10.md` (create it; the report in Step 12 builds on it).
All commands: `B="bash /workspace/Pudgy/finetune/wan/v10_box.sh"`.

### Step 1. Code and secrets
```bash
git clone https://github.com/rlrahulkanojia/Pudgy.git /workspace/Pudgy     # main, once the v10 PR is merged
git -C /workspace/Pudgy log --oneline -5                                    # the v10 commits are there
cp <the .env from the user> /workspace/Pudgy/.env && chmod 600 /workspace/Pudgy/.env
md5sum /workspace/Pudgy/finetune/wan/assets/registry.toml                 # 3757257814f5588603317d747f2e250e
```

### Step 2. Environment and base weights (~5 min)
```bash
$B setup        # v9_box.sh setup (venv, musubi 0.3.4 + patch, HF weights) + rapidocr + HF prefetch for eval
```
**Must see:** musubi at v0.3.4 with the patch applied; the four weight files under `/workspace/wan_models`;
`prefetch` finishing (umT5 tokenizer, DINOv2-base, CLIP ViT-L/14).

### Step 3. Data and weights from Azure
```bash
$B pull                                                  # items 4-9; links; checks
nohup bash /workspace/Pudgy/finetune/wan/v10_box.sh pull_v9b > /workspace/logs_v10/pull_v9b.log 2>&1 &   # item 10, in the background
```
**Must see:** merged base md5 `ec333748…026f`; "training set: SHA256SUMS all match"; "eval set:
SHA256SUMS all match"; `clips: 1525`; `common_v10.py configs v8merged` resolving every file. Note the
download speed in the run log.

### Step 4. Cache the training set (GPU 0, ~1.5 h)
```bash
$B cache
```
**Must see:** `verify_cache_v8` OK; every `scene_f*` at H×W **(106, 60)** and every `replay_f*` at
**(128, 128)**, all `ok` (a wrong shape means silent re-bucketing: stop and ask); the last line
`CACHE_MARK_CHECK OK replay=534 scene=991 phone=80 errors=0` (80 = 52 v10 + 28 v9b scene windows naming the phone; counted from `manifest.json`).

### Step 5. Cache the eval windows (GPU 1, minutes)
```bash
$B evalprep        # latents for the 51 held-out Tier-1 windows -> /workspace/eval_v10/scenes/cache/f33
```

### Step 6. Smoke test (GPU 0, ~10–15 min)
```bash
$B smoke
```
`SMOKE=1` trains 3 steps on 3 samples: the heaviest replay clip (f57 at 1024², 61,440 tokens), the
heaviest phone window and the heaviest plain window, with preservation, E5 and block swap 24.
**Pass = `SMOKE OK`**, which requires: exit 0; `== done:`; **0** `memory mapping failed with OOM` lines;
a `PRESERVE_STATS` line with `preserve=1` and `asset=1` (expected `total=3 preserve=1 data=2 asset=1`);
`[asset] base weights: N/N LoRA modules match the DiT`.
- **Peak memory near 80 GB is normal**: the allocator reserves the card. Never gate on it.
- If the OOM count is above 0: `BLOCKS_TO_SWAP=26 $B smoke`, then 28. Train with the passing value
  (`BLOCKS_TO_SWAP=… $B train`) and record it.
- Write down the s/it.

### Step 7. Start GPU 1 (background, runs for ~1.5 days)
```bash
nohup bash /workspace/Pudgy/finetune/wan/v10_box.sh gpu1 > /dev/null 2>&1 &     # log: /workspace/logs_v10/gpu1.log
```
The scheduler, in priority order each round:
1. the Tier-1 base rows for v8merged + golden (once, ~20–30 min);
2. **Tier 1 on each new checkpoint at a multiple of 50** (G3 caption use + G4's fast check, the
   high-noise v8-label R2; ~20–30 min each), applying the **early-stop rule**;
3. the **baseline** (Phase 0 #2) in 10-clip slices: v8merged on the 102 G1 clips + 8 FLF, and on the
   G4 core 52 + compose 18 (~17–20 h in all);
4. **Phase 0 #1** in slices, only while training runs: v9b@100/200 vs v8merged on v9b's scene jobs and
   the core 52 (~12 h).

It exits when training is over, every checkpoint is scored and the baseline is complete, printing
"GPU 1 is free".

### Step 8. Train Stage 1 (GPU 0, ~18–23 h)
```bash
$B train          # background trainer + Azure monitor (MONITOR_PREFIX=v10)
$B status         # training progress, PRESERVE_STATS, GPU 1 state, Tier-1 results
```
**Watch for:**
- the startup lines `CACHE_MARK_CHECK OK …`, `[asset] train: single-expert, preserve x1, marker key,
  E frozen offset (25, 4096)` and the base-weights match;
- a `PRESERVE_STATS` line every 5 steps: `preserve/total` should sit near **35%**, and `preserve_mse`
  stays small;
- checkpoints `pudgy-v10-highnoise-step000000{25,50,…}.safetensors` reaching Azure `v10/weights/`;
- no mapping-OOM lines piling up (one in a long run is survivable).

**ETA** = 300 × (seconds per optimizer step over steps 5–10). v9b measured 187 s/step on an A100; the
preservation pass adds ~10–15%. Tell the user the ETA.

Resume after any stop: `$B train` again. It picks the newest **complete** state and continues the step
counter (the log says `continuing at global step N`).

### Step 9. Tier 1 and the early stop (automatic)
`gpu1` scores steps 50, 100, 150, … as they land (`$R/tier1/gate_step<S>.json`, where
`R=/workspace/eval_v10/runs/pudgy-v10-highnoise`). **Early stop** (plan §5.1): at step 100, if the
high-noise v8-label R2 failed at both 50 and 100, or G3 shows caption use measurably worse, the
scheduler writes `/workspace/logs_v10/EARLY_STOP` and pauses the trainer. Then:
1. `$B archive`, `MSG="v10 box: early stop at step 100" $B git`;
2. write the report (Step 12) with the Tier-1 numbers;
3. tell the user. **Stage 1 is not kept; don't retry anything on your own.**

GPU 1 still finishes the baseline and Phase 0 #1 afterwards: Phase 0 #1 informs the user's next
decision.

### Step 10. The candidate: G2, G4, G5 (both GPUs, ~16 h)
After `== done:` in the training log **and** "GPU 1 is free" in `gpu1.log`:
```bash
$B candidate      # picks the latest Tier-1 checkpoint; G5 on GPU 0, G2 + G4 on GPU 1; falls back on a FAIL
EVAL_GPU=0 $B phase0     # once G5 has finished on GPU 0 (~6.6 h in), if gpu1 left Phase 0 #1 incomplete
```
`candidate` exits **0** (all pass), **11** (G2 needs an eye check), **30** (no checkpoint holds Tier 1:
Stage 1 not kept, go to Step 12), **2** (missing input: fix and re-run). On a G2/G4/G5 **FAIL** it tries
the previous Tier-1 checkpoint by itself (plan §5.1). The verdicts are in
`/workspace/logs_v10/candidate_verdicts.txt`; the candidate's name in `/workspace/logs_v10/CANDIDATE`.

**On 11 (REVIEW):** look at every clip `$R/suite/g2_v10_<S>.json` lists under `eye_check` (more
duplicate-lead frames than the real shot, or text absent from frame 0 and the real shot; Q14, Q21),
against its real shot and the v8merged clip, and record a verdict per clip in the run log. G2's other
bars are automatic: zoom ≤ 3% (smoothed), median motion ratio 0.7–1.3, identity not measurably worse
than v8merged (Q22). The candidate passes G2 only if every flagged clip is fine by eye; if one isn't, treat G2 as
FAIL and run `candidate` again after noting it (it then moves to the previous checkpoint).

### Step 11. G1: blind preference (the user judges)
```bash
$B g1             # builds the blind pair page (102 pairs) -> Azure v10/eval/g1/pages/<name>/; the key goes to v10/eval/g1/keys/
```
Tell the user the page is ready. **They** download the page folder, open `index.html` locally, judge
every pair (left / right / can't tell, with each clip's caption shown) and export the answers JSON. The
Mac session then runs
`python3 finetune/wan/eval_v10/g1_pairs_v10.py tally --key <name>.key.json <answers>.json`: exit 0 = G1
PASS. With 102 pairs, v10 must win **≥ 60** (the exact one-sided binomial at α = 0.05; "can't tell" counts
as half). Never open the key or the pair order yourself.

### Step 12. Archive, git, report
```bash
$B archive                                    # weights (+ states), run logs, eval outputs -> Azure v10/
MSG="v10 box: Stage 1 done, candidate <S>" $B git     # commit, push to v10/box-run, bundle -> Azure v10/git
```
Write `docs/training_reports/v10/REPORT_v10.md`, following `REPORT_v9b.md`: headline, what was trained,
the smoke and timing numbers, the Tier-1 table per checkpoint, the early-stop decision, the candidate's
G2/G4/G5 tables with contact sheets, Phase 0 #1's reading, and what G1 needs from the user. Update the
plan's status line. Send the user a short summary with sample clips. Then run `git` again.

---

### 4b. On a one-GPU box: `solo` (this run, decided 2026-11-10, Q25)

The box rented for this run has **one** A100 80 GB. Steps 1–3 are the same (on one GPU the box script
already uses GPU 0 for evaluation). Then, instead of Steps 4–11, run the whole sequence on GPU 0 in tmux:
```bash
tmux new-session -d -s solo "bash /workspace/Pudgy/finetune/wan/v10_box.sh solo_watch; exec bash"   # log: /workspace/logs_v10/solo.log
tmux new-session -d -s sync "while true; do bash /workspace/Pudgy/finetune/wan/v10_box.sh sync >> /workspace/logs_v10/sync.log 2>&1; sleep 1800; done"
```
`solo` runs, one job at a time, each stage with the same checks as Steps 4–11:
1. cache + marks, evalprep, smoke (stops on `SMOKE FAILED`), the Tier-1 base rows;
2. **train to step 100** (with `MAX_STEPS` still 300, so the learning-rate schedule is the planned one), pause,
   **Tier 1 at 50 and 100** and the early-stop rule (~1 h);
3. resume and **finish training** to step 300, then Tier 1 at 150–300;
4. the **baseline** (Phase 0 #2), then the **candidate's G5, G2, G4** in turn (with the fallback on a FAIL);
5. the **G1 page** to Azure for the user, archive and git; then **Phase 0 #1** while the user judges.

**Keeping the GPU busy** (2026-11-10): each generation phase runs in one model load (no 10-clip slices on one
GPU); `solo` polls every 30 s; at any stop that waits for a person (early stop, no checkpoint holds, a G2 eye
check) it first runs Phase 0 #1, which needs no decision; and `solo_watch` re-runs a failed stage (exit 1)
after 2 min, up to 3 times, before giving up. Use `tmux` (a bare `nohup … &` over ssh dies with the session).

It is **resumable**: after any stop, fix the cause and run the same tmux command again; finished stages are
skipped (flag files in `/workspace/logs_v10`, Tier-1 results, existing clips) and training resumes from its
newest complete state. Its exit codes are the stops to report:

| exit | meaning | do |
|---|---|---|
| 0 | done: G1 page on Azure, Phase 0 #1 done | Step 12 (report); tell the user the page is ready |
| 20 | early stop at step 100 (`EARLY_STOP`); Phase 0 #1 ran afterwards | archived and pushed already; report (Step 9) |
| 11 | G2 needs an eye check; the G1 page and Phase 0 #1 are already done meanwhile (the page must not be judged yet) | Step 10's eye check. All fine → `rm /workspace/logs_v10/G2_EYE_CHECK_PENDING`, run `solo_watch` again (exit 0). One not fine → `rm /workspace/logs_v10/{CANDIDATE,g1.done,G2_EYE_CHECK_PENDING}`, then `CANDIDATE_BEFORE=<S> bash v10_box.sh candidate` (the previous checkpoint), then `solo_watch` again |
| 30 | no checkpoint holds Tier 1; Phase 0 #1 ran afterwards | archived and pushed already; report |
| 1 | a stage failed 4 times in a row (`solo_watch` already retried it 3 times) | read `solo.log` and that stage's log; fix; run `solo_watch` again |

**Timeline on one A100** (estimates): setup and downloads ~3–6 h; cache, smoke, base ~2.5 h; training to 100
~6 h + Tier 1 ~1 h; training to 300 ~13–15 h; Tier 1 at 150–300 ~2 h; baseline ~17–20 h; candidate
(G5 ~6.6 h, G2 ~13 h, G4 ~3 h) ~23 h; the G1 page; Phase 0 #1 ~12 h. About **4–4.5 days** to the G1 page,
plus Phase 0 #1. Same GPU-hours as two GPUs.

## 5. Keeping things safe

- **Azure while it runs:** the monitor mirrors checkpoints; loop the sync for code, docs and logs:
  `nohup sh -c 'while true; do bash /workspace/Pudgy/finetune/wan/v10_box.sh sync; sleep 1800; done' > /workspace/logs_v10/sync.log 2>&1 &`
- **Prefixes you own:** `v10/` only. **Read-only for you:** `processed/` (never re-upload the sets),
  `raw/`, `v2/`, `v8/`, `v9b/`, `v9c/`, `v9c2/`. Never run `scripts/mirror_data_to_azure.py` on the box.
- **Git (Q7):** commit on the box and push only to `v10/box-run`, never `main`. `$B git` also keeps a
  bundle on Azure `v10/git/`.
- **If the machine dies:** set up a new one from this file, pull the latest `*-step*-state` from Azure
  `v10/weights/pudgy-v10-highnoise/` into `/workspace/wan_output/pudgy-v10-highnoise/`, and `$B train`
  resumes. Eval outputs already on Azure (`v10/eval/runs`) can be pulled back the same way; generation
  skips clips that exist.

## 6. Pitfalls already paid for

| don't | because |
|---|---|
| `pkill -f <pattern>` | it kills your own shell. Kill by PID; `pause` uses a pattern that matches only this run's trainer |
| gate the smoke on a memory number | the allocator reserves almost all 80 GB on a healthy run; the mapping warning is the signal |
| run two GPU jobs on one GPU | each takes the whole 80 GB. GPU 0 = training (then G5, Phase 0 #1); GPU 1 = the scheduler (then G2, G4). `candidate` refuses to start while `gpu1` runs |
| treat scene clips as 848 wide | they are **480 wide × 848 tall**; latent H×W (106, 60); generation takes `--video_size 848 480` |
| edit `registry.toml`, captions, curation or the sets | they are final and checksummed; any change means a rebuild on the Mac |
| re-run T5 caching without `cache_mark_v10.py check` after it | the marks survive a same-caption re-cache, but `check` is what proves it; the trainer refuses to start without the marks |
| reuse a cache directory after rows change | musubi trains on every latent in a cache directory, whatever the jsonl says |
| drop `HF_HUB_OFFLINE=1` | the box steps set it; musubi re-fetches the tokenizer otherwise (HTTP 429 in v7) |
| trust `azure_sync.py push`'s size check after an in-place edit | it skips same-size blobs; use `push --force` and `finetune/wan/azure_verify_md5.py` |
| more than 16 parallel ffmpeg decodes | 128 concurrent decodes deadlocked ffmpeg 6.1 once |
| show the G1 key or pair order to anyone judging | G1 is blind |

## 7. After this run

- **G1 passes** (and G2, G4, G5 held): the user decides whether v10 becomes production (high-noise =
  golden + v10 LoRA; low-noise and E5 unchanged), and whether Stage 2 (low-noise, plan §4) is worth
  running for detail. Neither starts on its own.
- **G1 fails, or nothing holds:** production stays v8merged + golden + E5. The report says why, with
  Phase 0 #1's reading.
- **Later handovers (Q20):** the keyframe LoRA (v10-K) and end-to-end sample reels.

## 8. Inventory

| what | where |
|---|---|
| plan (why, decisions Q1–Q20, gates) | `training_approach/v10/Training_Approach_v10.md` |
| box script | `finetune/wan/v10_box.sh` |
| prep (ran on the Mac) | `finetune/wan/prep_v10.py`, `scenes/rows_v10.py`, `scenes/choose_holdout_v10.py`, `scenes/holdout_v10.json`, `scenes/v9b_535_scene_windows.json` |
| training | `finetune/wan/train_pudgy_v10.sh`, `cache_mark_v10.py`, `asset_embed_v9.py` (`--preserve-marker key`); CPU tests in `finetune/wan/tests/v10_train/` |
| evaluation | `finetune/wan/eval_v10/` (each script's docstring has its command, outputs and exit codes; Mac tests in `eval_v10/tests/`) |
| curation (iteration_7) | `finetune/wan/scenes/curation/iteration_7__*.json`, brief `CURATION_v10.md`, `VERIFY_v10.md` |
| training set | Azure `processed/v10_1525` (README, `manifest.json` with every clip's source and caption, `SHA256SUMS`) |
| eval set | Azure `processed/v10_eval` (README, `manifest.json`, `SHA256SUMS`) |
| outputs | Azure `v10/weights/pudgy-v10-highnoise/`, `v10/eval/`, `v10/logs/`, `v10/work/`, `v10/git/` |
