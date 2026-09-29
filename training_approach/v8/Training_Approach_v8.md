# Training Approach v8 — one joint LoRA on top of v7 (correctly combined), measured by likelihood

**One sentence:** train **one fresh low-noise LoRA** on the joint motion + 8-expression set,
on top of a DiT with **both v7 LoRAs merged in, correctly combined** (v7 expression +
v7 motion − v2 golden, so the golden counts once), with gradient accumulation and a decaying
LR — and select checkpoints with a **diffusion classifier on a held-out camera angle**
instead of pixel SSIM.

| | |
|---|---|
| **Base** | Wan2.2-I2V-A14B low-noise DiT (fp16) **+ `pudgy-v7-fixed-combined` merged** (`--base_weights`) = v7 expr step 2500 + v7 motion final − v2 low golden, one exact rank-48 LoRA; high-noise expert = v2 high golden, frozen |
| **Trains** | one fresh LoRA, rank 32 / α 32, low-noise expert (t ∈ [0, 0.9]) |
| **Data** | `processed/v8_joint_2096` — 2,096 clips, 194 sources, 13 labels (5 motion + 8 expression), 26 sources held out at `QF2_R` |
| **Optimiser** | adamw8bit, LR 5e-5, warmup 20, cosine → 10%, grad-accum 8, clip 1.0 |
| **Budget** | ≤ 600 optimizer steps (4,800 clips ≈ 2.3 epochs); save every 50; the gates, not the counter, decide the stop |
| **Supersedes** | [v7](../v7/Training_Approach_v7.md) — see [`ANALYSIS_v7.md`](../../docs/training_reports/v7/ANALYSIS_v7.md) |

---

## 1. What the v7 evidence forces

Every row is a measurement from [`ANALYSIS_v7.md`](../../docs/training_reports/v7/ANALYSIS_v7.md)
or from the v8 data build, not a preference.

| # | Measured | → v8 decision |
|---|---|---|
| 1 | Both v7 LoRAs contain the golden G; stacking them gives W + **2G** + τ_e + τ_m — a **76%** error on the intended update | The base is their **correct** sum G + τ_e + τ_m, merged via `--base_weights` (§1.1); the trained file is a **pure task vector** τ on top |
| 2 | τ_e and τ_m share a common component: cos **0.33** overall, **0.54–0.58** on self-attn q/k, input-subspace overlap **0.67** (random: 0.0016) | **One joint LoRA** learns the shared "v7 data" component once, instead of twice in two files |
| 3 | G-X: motion on low-noise separates (0.944), high-noise blurs (0.962). ANALYSIS §4: in-place actions have ≈ 0 between-label variance at high noise, so this is data-specific but holds for this data | Both kinds on the **low-noise** expert; high-noise golden stays frozen (still bisectable) |
| 4 | Batch 1 + constant LR ⇒ updates mostly noise: ‖τ‖ ∝ t^0.55, consecutive-update cos 0.10–0.24, drift share of ‖τ‖² **5% @500, 19% @2500** | **Grad-accum 8** (gradient noise variance ÷ 8 per update) + **cosine decay to 10%** (anneals the random-walk term) |
| 5 | Effective rank of τ ≈ **8.9 / 32** per kind | Rank 32 for a joint LoRA carrying both kinds; α 32 (scale 1). LR 5e-5 × scale 1 ≈ v7's 3e-5 × scale 2 per-step ΔW |
| 6 | Face-SSIM recovers the **seed** 67–81%, the label at chance; CLIP recovers the label 57–71% | G-C is scored semantically (CLIP/DINOv2); checkpoint selection uses the **diffusion classifier** (§5) |
| 7 | A stretched request plays one cycle slowly (v7 G-D probe) | **Loop-tiled f49 rung** for the 4 cyclic actions (seam verified per source) |
| 8 | 21 sources short of real frames; ffmpeg pads with duplicates; v7 trained across the gaps | Rungs **end before the first missing frame**; gapped sources are never tiled |
| 9 | No holdout existed, so "generalises" and "memorised 174 performances" were indistinguishable | **`QF2_R` held out** (26 sources) from v8's training. ⚠️ With the v7 base (§1.1) the base has seen them; only `blushing` is fully unseen |
| 10 | `blocks_to_swap 0` OOMs on f57 (v7 defect 1); swap 16 thrashes at rank 32 (81.2 GB, 31 allocator retries) | **swap 20**: 0 retries, and faster (58.5 vs ~77 s/clip) |

**Considered and not done, with the reason:**

- **16 fps resample** (Wan2.2 A14B's native rate). The footage is animated **on ones with
  eased in-betweens** (measured); 24→16 alternates 1- and 2-frame steps (judder) or, if
  blended, ghosts the thick outlines. Stays 24 fps; eval plays back at 24.
- **Orthogonal adaptation** (Po et al., CVPR 2024) for two separate composable LoRAs.
  Needs a custom network module and gives up the shared component (#2). Phase-2 option if
  per-kind rollback turns out to matter more than composition.
- **Diffusion-DPO** on same-start-frame pairs (Wallace et al., 2023). The dataset's
  contrastive structure makes the pairs free, but it is a second training stage — only
  worth it if Tier 1 shows a controllability ceiling.
- **Caption dropout** for label-axis CFG. musubi 0.3.4 has none; the inference-time
  alternative (label-contrastive negative prompt, `suite guide`) needs no training.

### 1.1 Starting point: v7, not the bare golden (decision 2026-09-26)

The first draft of this plan started from the v2 golden alone. **It now starts from v7**,
so v8 builds on what v7 already learned about the 7 expressions and 5 actions instead of
relearning it. The two v7 LoRAs cannot simply be loaded together — each contains the
golden, and stacking them counts it twice (§1 #1, a 76% error). So the base is their
**correct sum**, built as one file:

    pudgy-v7-fixed-combined = L_e + L_m − G = G + τ_e + τ_m          (rank 16+16+16 = 48)

`lora_tools_v8.py combine` concatenates the factors with each file's scale and sign folded
into the up-matrices; verified against the definition to a relative error of **2.5·10⁻⁸**
(fp32 round-off), ‖·‖ = 41.50. One merge instead of three fp16 round-trips that add and
then remove a whole golden. Inputs are the v7 report's goldens: expression step 2500,
motion final (step 2144).

What this changes:

- **v8 at step 0 *is* `v7fixed`.** Every gate bar is now "better than `v7fixed`", not
  "better than the golden". The golden stays in the baseline set as the floor.
- **The QF2_R holdout is no longer unseen for the model as a whole.** v7 trained on every
  angle, so the base already saw those 26 sources. The v8 LoRA still never trains on them,
  so Tier 1 on QF2_R measures what **v8's training** adds on clips it was not fitted to —
  but it can no longer test "generalises to a never-seen camera angle" (gate G-A is void
  for any v7-derived model). **Exception: `blushing`** — v7 never saw it at all, so the 2
  QF2_R blushing sources are genuinely unseen by base and LoRA alike.
- **The v7 stutter clips are baked into the base.** v7 trained sitting rungs across the
  duplicate-padded gaps (ANALYSIS §5); v8's gap-aware data cannot remove that, only
  outweigh it.
- **The v7 interference (τ_e · τ_m, cos 0.33) is in the base too** — the joint fine-tune is
  the chance to correct it, and `compose` is the test.
- Smoke test on this base (f57, 2 steps): merge logged, 57 s/clip, 0 allocator retries,
  loss **0.00082–0.00089 vs 0.00106** on the golden-only base for the same clips — the v7
  base already explains this data better, as it should.

`BASE_WEIGHTS=/workspace/wan_output/v2_golden/lora_lownoise_GOLDEN_ep40.safetensors`
restores the golden-only start if the v7 base turns out to cap what v8 can learn.

#### 1.2 Is the v7 base a head start or a trap? — thresholds fixed before step 100

Tier-1 baselines on the 26 held-out clips (k = 2, paired noise), per kind:

| start | expr loss | expr margin | motion loss | motion margin |
|---|---|---|---|---|
| v2 golden | 0.00564 | −0.0137 | 0.00604 | −0.0128 |
| **v7fixed (v8 step 0)** | **0.00499** | **+0.0227** | **0.00467** | **−0.0047** |
| v7expr alone | 0.00323 | +0.0250 | — | — |
| v7motion alone | — | — | 0.00283 | −0.0039 |

v7fixed beats the golden on 75% (expr) / 100% (motion) of clips, paired, and keeps ~90% of
the expression margin — but it keeps only **27%** (expr) / **43%** (motion) of each v7
specialist's loss reduction over the golden: combining the two LoRAs re-adds their shared
component (§1 #2). So v8 has to *undo interference* before it adds anything. (v7 trained on
QF2_R, so its lead here is partly memory; `blushing`, unseen by all, ranks 2nd/1st for
v7fixed vs 6th/7th for the golden — n = 2.)

Both starts can reach the same place in principle (the rank-32 v8 LoRA can represent
either the two label task vectors, effective rank ≈ 9 each, or the interference correction,
which lives in the same rank-16+16 subspaces); the v7 start buys ≈ 500–1,000 clips of the
4,800-clip budget. The decision rule, written down **before** the data exists:

| Gate | v7 base is working if (holdout split, per kind) | Otherwise |
|---|---|---|
| **step 100** | expr loss **< 0.0040** and motion loss **< 0.0035** (well below v7fixed's 0.0050 / 0.0047, toward the specialists' 0.0032 / 0.0028), **and** margins ≥ v7fixed's (+0.0227 expr / −0.0047 motion) | note it; continue — one gate is not enough evidence to discard ~9 h |
| **step 300** | loss keeps falling past the step-100 values toward the specialists, margins hold or grow | if loss is still **≥ 0.0045 (expr) / ≥ 0.0042 (motion)** — within ~10% of v7fixed — the v7 base is a trap: **restart from the golden** (`BASE_WEIGHTS=…/lora_lownoise_GOLDEN_ep40.safetensors`, fresh output dir) |

Margins and losses are read from `/workspace/eval_v8/dcls/v8@<step>_k2.json` (`summary`),
compared paired against `v7fixed_k2.json`. Accuracy is reported but not used for the
decision: at n = 16 / 10 it moves in steps of 0.06 / 0.10.

---

## 2. Data — `processed/v8_joint_2096`

Built by [`prep_v8.py`](../../finetune/wan/prep_v8.py) from `raw/iteration_3`, `_4`, `_5`
(md5 de-dup drops 288 re-sent files).

| | Sources | Clips | Buckets |
|---|---|---|---|
| motion (walking, running, waving, sitting, jumping) | 78 | 896 | f13 · f17 · f21 · f25 · f33 · **f49 tiled** |
| expression (happy, surprised, angry, confused, crying, neutral, laughing, **blushing**) | 116 | 1,200 | f21 · f25 · f29 · f37 · f41 · f57 |
| **held out (QF2_R)** | 26 | — | used by Tier 1 and G-A only |

Joint buckets: f13 308 · f17 96 · f21 616 · f25 212 · f29 304 · f33 64 · f37 192 · f41 44 ·
f49 196 · f57 64. 4 grounds × 3 shot sizes, balance asserted as in v7. Colour anchors
verified: no Pax caption contains "pink" (the blushing caption says "rosy").

**What changed vs `v7_primitives_2272`:**
- **+ blushing** (iteration_5): 12 sources after holdout; one (Pax QF2_L) is 8 frames short
  and ships at f21–f29 only.
- **− derived `standing idle`** — its footage *is* `neutral`, which is now in the same LoRA.
- **+ f49 loop-tiled rung** for walking / waving / jumping / sitting. Tiled only when the
  seam (last→first frame, relative to a median frame step) is ≤ 1.5: walking 0.62–1.04,
  sitting 1.2–1.45, waving 0.17–0.35 (near-hold kept), jumping ≈ 0 (duplicate dropped);
  running 2.3–2.7 never tiled. Measured after the build: the worst step in a tiled clip is
  no larger than the worst natural step in untiled clips of the same action.
- **Gap-aware rungs.** 13 of 16 sitting sources lose their upper rungs (gaps at frames
  13–25); 4 confused and 1 walking source are dropped (gap below the floor), 1 more
  sitting source by v7's tear check.

**Label imbalance, kept deliberately.** `confused` and `happy` were delivered at 21
frames only, so they get one rung and **48 clips each vs 192** for the other expressions
(≈ ¼ of the per-label gradient share). Upweighting them (`num_repeats`) was considered
and **declined** (2026-09-26): they are read as the two cells to watch at each gate break.

**Known limits carried forward:** no living idle, no interaction clips in a
single-character set (12 two-character clips arrived in iteration_5 — they need their own
design), `confused` is now the thinnest class (12 sources), `happy` and `blushing` are
7-angle, square 1024² is off Wan's canonical aspects.

---

## 3. Recipe — [`train_pudgy_v8.sh`](../../finetune/wan/train_pudgy_v8.sh)

| Knob | v7 | **v8** | Why |
|---|---|---|---|
| init | `--network_weights` golden (continue-train) | **`--base_weights` v7-combined (merged) + fresh LoRA** | #1, §1.1 |
| LoRAs | 2 (+1 A/B loser) | **1 joint** | #2, #3 |
| rank / α | 16 / 32 | **32 / 32** | #5 |
| LR / schedule | 3e-5 constant | **5e-5, warmup 20, cosine → 0.1×** | #4 |
| batch | 1 | **1 × accum 8** | #4 |
| steps | 3,600 / 2,144 | **≤ 600 optimizer (= 4,800 clips)** | gate-driven |
| save | every 250 | **every 50 optimizer steps + state** | 12 points for Tier 0/1 |
| blocks_to_swap | 16 / 0 | **20** (measured: 16 thrashes at f57) | #10 |
| timesteps | ts 0–900, shift 5, preserve shape | unchanged | |
| precision / optimiser / seed | fp16 · adamw8bit · 42 | unchanged | attributable to the data + the 4 changes |

---

## 4. Compute

Measured on this box (A100 80GB PCIe): **58.5 s per clip at f57** (15 latent frames,
61k tokens, swap 20) — the worst bucket. Cost per clip grows at least linearly in latent
frames L = 1 + (n−1)/4, so scaling linearly from f57 is an **upper bound** for the shorter
buckets (attention is super-linear, so short clips are cheaper than this predicts):

| | mean L over the set | ≤ s/clip | 1 epoch (2,096 clips = 262 steps) | 600 steps (4,800 clips) |
|---|---|---|---|---|
| v8 joint | 7.54 | ≈ 29 | ≈ 17 h | **≈ 39 h** (upper bound) |

Plus ~3 min per restart (DiT load + base merge) and ~50 min of GPU per Tier-1 gate
break. The monitor reports the true s/it from the trainer's own progress bar within the
first hour — re-plan the gate breaks from that, not from this table.

For scale: v7 was 7,888 steps (batch 1) ≈ 75 h across three LoRAs. v8 sees 4,800 clips
in one LoRA.

---

## 5. Evaluation — three tiers, cheapest first

Training holds ~67 GB of the 80 GB card, so **nothing that loads a second DiT can run
during training** (v7 correction 5). In-run monitoring is CPU-only; everything else runs
at **gate breaks** (stop → gate → resume from the saved state; the scheduler continues
exactly because the state carries it).

| Tier | What | When | Cost | Script |
|---|---|---|---|---|
| **0** | Weight trajectory: ‖τ(t)‖, drift share, consecutive-update cosine → tensorboard `diag/*` | every checkpoint, automatic | CPU, seconds | `eval_v8/lora_tools_v8.py traj` (run by the monitor) |
| **1** | **Diffusion classifier** on held-out QF2_R clips: is the true label's caption the best denoiser of the real clip, vs every other label's caption, under shared noise? | baselines once; each gate break | ~50 min / checkpoint (26 clips × ~7 captions × 2 draws × ~8 s) | `eval_v8/dcls_v8.py` |
| **2** | Generated suites scored with CLIP/DINOv2: `core` (controllability, FRONT + holdout start frame), `compose` (motion × expression), `gb` (unseen grounds), `gd` (duration/repetition), `guide` (label-contrastive negative prompt) | final + 1–2 candidates, and baselines | core ≈ 4 h / config | `eval_v8/suite_v8.py` |

**Tier 1 design notes.** Flow-matching loss under caption c is an ELBO proxy for
−log p(x₀|c, y); with the same (t, ε) for every candidate the argmin is a Bayes classifier
under a uniform prior (Li et al. 2023). Latent frame 0 is the conditioning image — its
error is label-independent — so it is excluded, and the error is weighted by where the
*true* clip changes (|x₀[f] − x₀[0]|), which removes the static background no caption can
explain; the mask depends on the ground truth only, never on a candidate. The noise draws
depend only on (clip, k), so they are **identical across checkpoints**: two checkpoints are
compared paired, not as independent samples. t ∈ {0.65, 0.85} first — content is decided
there.

**Baselines (run once, before or at the first gate break):** `golden` (v2 only — the
floor), `v7expr`, `v7motion`, `v7fixed` (v7 pair with the golden subtracted), `v7naive`
(the 2G stack, to measure what §1 costs). **`v7fixed` is v8 at step 0** (§1.1), so it is
the bar. v7 trained on QF2_R, so the "holdout" split measures what v8's own training adds
on clips its LoRA never fitted — not unseen-angle generalisation (except `blushing`).

### Gate breaks and pass bars

| Break | Tier 1 | Tier 2 | Proceed if |
|---|---|---|---|
| **step 100** (~800 clips) | v8@100 + baselines | — | holdout margin and loss move beyond `v7fixed` (step 0) — **base thresholds in §1.2** (expr loss < 0.0040, motion < 0.0035, margins ≥ v7fixed). **If nothing moved at all: LR → 1e-4** (fresh LoRA, α/r = 1 — the documented fallback), not more steps |
| **step 300** | v8@300 | — | still improving; drift share (Tier 0) above v7's 19% ⚠️ *invalid bar: the drift fit assumes a constant LR; under cosine use the consecutive-update cosine — see REPORT_v8 §3.3*; **§1.2 trap test** — loss still ≥ 0.0045 / 0.0042 ⇒ restart from the golden |
| **final / plateau** | all saved checkpoints ≥ 100, + `swa` of the plateau | `core`, `compose`, `gb`, `gd`, `guide` on the Tier-1 winner vs `golden` and `v7fixed` | ship bars below |

**Ship bars** (Tier 2, n per cell is small — read as ±0.1):
- **Controllability:** CLIP 1-NN label accuracy ≥ v7's 0.57–0.71 on FRONT, and **no more
  than 0.15 lower on the QF2_R holdout start frame** than on FRONT (the generalisation
  test v7 could not run).
- **Composition:** `compose` motion *and* expression zero-shot accuracy both ≥ `v7fixed`,
  and motion energy ≥ 80% of motion-only clips.
- **G-B:** corner drift ≤ 5/255 on both unseen grounds.
- **G-D:** tiled actions at f49 show periodicity (repeat ≥ 0.5) — a cycle, not one slow arc.
- **Parity:** Pax/Polly accuracy within 10%.
- G-F / G-M / G-H / G-N from v7's harness remain valid questions; run them on the winner
  with `gates_v7.py`-style prompts once the v8 LoRA stack is plumbed there (not yet done).

---

## 6. Readiness — what was verified on this box (2026-09-26)

| Check | Result |
|---|---|
| Environment | musubi-tuner **v0.3.4** (pinned by tag), torch 2.14 cu126 on the A100, diffusers 0.32.1 · transformers 4.57.6 · accelerate 1.6.0; 65 GB weights |
| Dataset | 2,096 clips; **0** frame-count mismatches (ffprobe on every file); tiled joins no rougher than natural steps; colour anchors clean outside the ground clause |
| Caches | latents + T5 for every row, checked by `verify_cache_v8.py` (shape = 1 + (n−1)/4 latent frames, I2V conditioning present, readable) |
| Training smoke (f57, the worst bucket) | 3 optimizer steps × accum 2 with the real script: loss 0.00126 → 0.00096, step checkpoint + resume state + final file written. The saved LoRA is a **pure τ**: 400 modules, rank 32, scale 1, ‖τ‖ = 0.6% of ‖G‖, cos(τ, G) = −0.0003 — the golden went into the base, not the file |
| Memory | swap 16: 81.2 GB peak, 31 allocator retries, ~77 s/clip → **swap 20**: 0 retries, 58.5 s/clip |
| **Tier 1 validation** | on 8 seen expression clips, K = 2: `golden` acc **0.125** (= chance), mean rank 4.0/8; `v7expr` acc **0.875**, rank 1.25, margin +0.036. v7expr's only miss is *blushing* — a label v7 never trained. P(≥ 7/8 at chance) ≈ 3·10⁻⁶. The classifier separates a trained from an untrained model with 8 clips |
| Tier 2 | suites expand to 110 prompts/config (dry run); the scorer reproduces ANALYSIS_v7 §3 exactly on the v7 clips (CLIP 1-NN 0.571 / 0.714, DINOv2 0.381 / 0.476) |
| Tier 0 | SWA re-compression exact (⟨SWA, mean⟩ = ‖mean‖², rel. err 2.6e-7) |
| Services | `pudgy-v8-train` (auto-resume from newest state, no auto-restart) and `pudgy-v8-monitor` registered in supervisor, **stopped**; tensorboard already serving `/workspace` |
| Mirror | monitor dry-run lists the v8 blobs; nothing uploaded yet |

A side result of the validation: v7's largest margin is on **confused** (+0.107) — the
label REPORT_v7 §3.2 called its weak cell on the strength of the SSIM metric.

---

## 7. Runbook

```bash
# 0. Everything below is already done on the current box; on a fresh one:
bash setup_wan_env.sh                                   # musubi pinned to v0.3.4 first:
#   git clone https://github.com/kohya-ss/musubi-tuner /workspace/musubi-tuner && git -C /workspace/musubi-tuner checkout v0.3.4
python /workspace/pull_raw.py raw/iteration_3/ raw/iteration_4/ raw/iteration_5/   # raw data
python finetune/wan/prep_v8.py --workers 32             # -> Data/processed/v8_joint_2096
ln -sfn /workspace/Data/processed/v8_joint_2096 /workspace/data_v8
/workspace/cache_v8.sh                                  # latents + T5 (~2.5 h)
python finetune/wan/eval_v8/build_assets_v8.py          # keyframes + classifier set
#   then cache the classifier set: wan_cache_latents.py --dataset_config /workspace/eval_v8/holdout/holdout.toml --i2v ...

# 1. Baselines (Tier 1), GPU, ~4 h
python finetune/wan/eval_v8/dcls_v8.py --configs golden v7expr v7motion v7fixed v7naive

# 2. Train (supervisor: survives the shell, auto-resumes from the newest state)
supervisorctl start pudgy-v8-monitor pudgy-v8-train
tail -f /var/log/portal/pudgy-v8-train.log
#   tensorboard: already running (portal) over /workspace - run dir logs/ + logs/diag

# 3. Gate breaks at steps 100 and 300 - automated (stop -> dcls + traj -> restart -> verify)
nohup finetune/wan/eval_v8/gate_break_v8.sh 100 300 > /workspace/eval_v8/gate_break.log 2>&1 &

# 4. After the run
python finetune/wan/eval_v8/lora_tools_v8.py traj /workspace/wan_output/pudgy-v8-joint-lownoise
python finetune/wan/eval_v8/dcls_v8.py --configs v8@100 v8@200 v8@300 v8@400 v8@500 v8@600
python finetune/wan/eval_v8/lora_tools_v8.py swa /workspace/wan_output/pudgy-v8-joint-lownoise --from 300 --to 600
python finetune/wan/eval_v8/dcls_v8.py --configs v8@swa300-600
python finetune/wan/eval_v8/suite_v8.py --suite all --configs golden v7fixed v8@<winner>
```

**Resuming (found 2026-09-26, fixed before the first gate break).** musubi v0.3.4's
`--resume` restores the LoRA, optimizer, LR scheduler and RNG, but restarts the step
counter at 0 (`# TODO skip until initial step`). Unpatched, a resume at step 100 would save
its step 150 as `…-step00000050` (overwriting the real one), run 600 *more* steps, and
push the restored cosine past its end, where it rises again.
`patches/musubi-v0.3.4-resume-step.patch` continues the count from the state
directory's step; `setup_wan_env.sh` applies it; `gate_break_v8.sh` verifies after every
restart that the log says `continuing at global step N` and that the bar passes N, and
stops training if not. Remaining v0.3.4 behaviour: the data loader starts a fresh,
reshuffled epoch on resume, so the clips of a partial epoch are not skipped — at two
breaks this is ≤ 300 clips seen out of order, not a bias. The W&B run continues under the
same id (`WANDB_RUN_ID` + `WANDB_RESUME=allow`, set by the service wrapper).

Inference with a v8 checkpoint always loads its base **and** τ:
`--lora_weight pudgy-v7-fixed-combined.safetensors pudgy-v8-…-stepN.safetensors --lora_multiplier 1 1`
(`eval_v8/common.py: lora_config("v8@N")`).

## 8. Fallbacks

1. **Tier 1 flat at step 100** → LR 1e-4 (restart from step 0; the schedule is part of the
   state). Not more steps — v7 showed steps without signal are a random walk.
2. **Drift share not above v7 (Tier 0)** → accumulation 16. The noise term scales ∝ 1/batch.
3. **Holdout accuracy ≪ seen accuracy** → the model memorises angles: consider holding out
   less and adding angle-balanced sampling; report it — v7 could not have seen it.
4. **Composition fails while each kind passes alone** → Phase 2: orthogonal-adaptation
   LoRAs, or a composition caption set (needs client clips of combined actions).
5. **OOM at f57 / f49** → swap 24 (20 is the measured default; §6).
