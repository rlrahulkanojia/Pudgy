# expv2 runbook: a fresh GPU box, from zero to the verdict

**Audience:** whoever runs expv2 (a person or an agent) on a new GPU box, without the author
present. Every step is copy-paste. Each **Report back** names what to send to the author.
The author works on a Mac with no CUDA.

**Read first:** [`README.md`](README.md) (what this is), [`docs/PLAN.md`](docs/PLAN.md) (the
design and the pre-registered bars), [`docs/MODEL_NOTES.md`](docs/MODEL_NOTES.md) §2 (the
gotchas the scripts work around). File layouts: [`docs/INTERFACES.md`](docs/INTERFACES.md).

**Isolation:** everything below runs from `expv2-minimax-h3/`. Nothing imports from
`finetune/`, `ltx25/`, `prep/` or `expv1-*`, and no Wan, LTX or Kandinsky weights are loaded,
except the two measurement references (G0.1's Wan VAEs, §8's v8@550 clips).

**Stop rules:** **G0.1 fails → stop expv2 and report.** A G0.2 hard fail → stop and report.
No pilot passes the §6.3 sanity gate → stop and re-check G0.2b before touching the LR.

---

## 0. The box

| | |
|---|---|
| GPU | **1× A100-80GB** (or H100-80GB). No FP8 is used |
| Host RAM | **≥ 128 GB** (stage 1 loads Qwen3-VL-32B) |
| Disk | **≥ 300 GB** under `/workspace`: weights ≈ 119 GB (+ 66 GB for the G0.2a parity DiT, deleted after), training cache ≈ 15–20 GB, corpus ≈ 5 GB, checkpoints ≈ 0.3 GB each |
| Network | HuggingFace and ModelScope (weights), Azure Blob (data and results) |
| Persistence | **Assume `/workspace` is not persistent** (C1). Step 5 starts the Azure mirror; do it in the first hour |
| Time | ≈ 28–37 A100-h for arm A (PLAN §10), re-planned after G0.6 |

Use `tmux` for everything long: `tmux new -s expv2`, detach with `Ctrl-b d`, reattach with
`tmux attach -t expv2`.

---

## 1. Credentials

The Azure connection string is **not in git**. Get the repo's `.env` from the author and
copy it to the box:

```bash
scp .env <box>:/workspace/.env
grep -c '^AZURE_STORAGE_CONNECTION_STRING' /workspace/.env      # expect 1
```

`configs/env.sh` exports it from there; no script prints it. The HuggingFace and ModelScope
repos used here are public and not gated, so no HF token is needed.

## 2. Repo and environment

```bash
git clone https://github.com/rlrahulkanojia/Pudgy /workspace/Pudgy && cd /workspace/Pudgy
bash expv2-minimax-h3/setup/setup_box.sh      # apt, venv, torch, DiffSynth @ acf2ad2 + patches, deps
source expv2-minimax-h3/configs/env.sh        # every new shell, before anything below
```

`setup_box.sh` is idempotent; re-run it after any interruption. It pins DiffSynth-Studio to
`acf2ad2`, applies `patches/*.patch` (and refuses if they do not apply), and writes
`$EXPV2_WORK/eval/phase0/env.json`.

> **Report back:** the tail of the `setup_box.sh` output and `nvidia-smi`.

## 3. Weights

```bash
python expv2-minimax-h3/setup/fetch_models.py --set core --verify    # ~119 GB, then sha256 of each big file
```

This writes `$EXPV2_WORK/eval/phase0/pins.json` (HF revisions `42ed227` / `e5eb578`, file
sizes and sha256) and links `MiniMax/MiniMax-H3 → MiniMaxAI/MiniMax-H3`. From then on
DiffSynth runs with `DIFFSYNTH_SKIP_DOWNLOAD=True` and never touches the network. The G0.2a
parity DiT (`--set parity`) and the arm-D adapter (`--set adapter`, ModelScope) are fetched
only when their step comes.

## 4. Corpus

The corpus is built from raw on CPU. First check whether it is already on Azure:

```bash
python expv2-minimax-h3/scripts/azure_sync.py pull corpus --dry      # "0 blobs" = not mirrored yet
```

**If it is mirrored**, pull it and skip to the loader check:

```bash
python expv2-minimax-h3/scripts/azure_sync.py pull corpus
```

**Otherwise** build it here:

```bash
python expv2-minimax-h3/scripts/azure_sync.py pull raw                # raw/iteration_{3,4,5}, ~4 GB
python expv2-minimax-h3/prep/prep_expv2.py --dry-run                  # prints the PLAN 5.2 table, actual vs expected
python expv2-minimax-h3/prep/prep_expv2.py --workers 32 --exact-tokens
python expv2-minimax-h3/scripts/azure_sync.py push corpus             # so the next box can skip this
```

`prep_expv2.py` exits non-zero on any PLAN §5.6 assertion, including per-label counts that
disagree with §5.2. **Do not loosen an assertion to get past it.** Report the printed
difference instead.

Then, either way, check the loader:

```bash
python expv2-minimax-h3/prep/check_loader.py      # every mp4: N+1 frames, first N == PNGs, DiffSynth's LoadVideo yields N
```

> **Report back:** the `--dry-run` table and the `check_loader.py` summary line.

## 5. Start the Azure mirror (first hour)

```bash
tmux new -d -s mirror "source expv2-minimax-h3/configs/env.sh && \
  python expv2-minimax-h3/scripts/azure_sync.py push eval weights logs --loop 900 2>&1 | tee -a \$EXPV2_WORK/logs/mirror.log"
```

Results land in `pudgy/expv2-minimax-h3/{eval,weights,logs}/` every 15 minutes.

---

## The fast path: `run_expv2.sh`

Sections 6–8 are also chained by one runner, so the GPU never waits for a person between
steps. Each stage is exactly the command written below it; finished stages leave a marker
and are skipped on re-run:

```bash
bash expv2-minimax-h3/run_expv2.sh status
tmux new -s expv2 "bash expv2-minimax-h3/run_expv2.sh auto"
```

`auto` runs corpus → G0.1 → cache → G0.2 → G0.5 → G0.6 → encodings → base → pilots →
calibration → main → selection → Tier 2. It **stops** on any failure, and at every decision a
person must make: the G0.1 eye check (record it with `vae_gate.py --eye`, then run `auto`
again), a calibration STOP, and the end, where the two rubrics are filled in by hand
(section 8). The log is `$EXPV2_WORK/logs/run_expv2.log`. `run_expv2.sh <stage>` re-runs one
stage. The sections below are the same steps by hand, with what to look at after each.

---

## 6. Phase 0

### G0.1 — VAE gate (stop gate)

```bash
python expv2-minimax-h3/eval/vae_gate.py
```

The script writes `eval/phase0/vae_gate/report.json` and contact sheets at 4× zoom. It
checks the metric bars itself (exit code 2 = FAIL) and leaves `gate: PENDING_EYE`. It
**cannot** check the eye bar: open `eval/phase0/vae_gate/sheets/*_768.png` and judge beak
outline, eye highlights, flipper outline and blush shapes at 768² against the Wan2.1
reference. Then record the judgement:

```bash
python expv2-minimax-h3/eval/vae_gate.py --eye PASS --eye-notes "outlines and highlights intact vs wan21"
# or: --eye FAIL --eye-notes "<what dissolved>"
```

**Metric FAIL or eye FAIL → stop expv2 and report.** Nothing trains until `gate` is `PASS`.

> **Report back:** `report.json` and 2–3 contact sheets.

### Stage-1 training cache

```bash
bash expv2-minimax-h3/configs/cache.sh              # f22 (the trained bucket, PLAN 5.1); ~1 h
```

If it runs out of memory (Qwen3-VL 66.7 GB + VAE on one card, MODEL_NOTES 2.21):
`CACHE_OFFLOAD=1 bash expv2-minimax-h3/configs/cache.sh`.

### G0.2 — pipeline checks (stop on a hard fail)

```bash
python expv2-minimax-h3/setup/fetch_models.py --set parity           # +66 GB, for G0.2a only
python expv2-minimax-h3/eval/pipeline_checks.py --checks a,b,c,d,e,f,g,h
python expv2-minimax-h3/setup/fetch_models.py --delete parity        # once G0.2a has passed
```

Pre-declared outcomes (PLAN §4): **a** fails → stop (the build question comes first).
**f** fails → train f22 only: `mv $EXPV2_WORK/cache/train/f39 $EXPV2_WORK/cache/train/f56 $EXPV2_WORK/cache/` and
note it in the report. **b, d, e, g, h** fail → stop and report.

> **Report back:** `eval/phase0/pipeline_checks/report.json`.

### G0.5 / G0.6 — source baseline and memory probe

```bash
python expv2-minimax-h3/eval/source_baseline.py     # CPU; can run in parallel with anything
python expv2-minimax-h3/eval/mem_probe.py --frames 22,39,56
```

Re-plan PLAN §10 from `mem_probe.json` (s/sample at each length). Tell the author if the
arm-A estimate moves outside 28–37 A100-h.

### Pass-1 encodings, then G0.3 / G0.4 — the untrained control

```bash
python expv2-minimax-h3/eval/dcls.py --encode --set holdout,calib      # Qwen3-VL + VAE, once
python expv2-minimax-h3/eval/generate.py --encode --suites core,native,gb,probe
python expv2-minimax-h3/eval/dcls.py --ckpt base --set holdout,calib --sigmas grid    # G0.3
python expv2-minimax-h3/eval/generate.py --ckpt base --suites core,native             # G0.4
python expv2-minimax-h3/eval/score.py --ckpt base --suites core,native
```

> **Report back:** `tier1/base.json` summary, `scores/base/core.json` summary, and a few
> base clips (does the base animate Pudgy on twos, drift to 3D, or lose frame 0?).

---

## 7. Phase 2 — pilots, the shift, the main run

```bash
bash expv2-minimax-h3/configs/train.sh expv2-h3-s12       # ~1.5 h
bash expv2-minimax-h3/configs/train.sh expv2-h3-s222      # ~1.5 h
python expv2-minimax-h3/eval/dcls.py --ckpt expv2-h3-s12@last,expv2-h3-s222@last --set holdout,calib --sigmas grid
python expv2-minimax-h3/eval/dcls.py --calibrate --pilots expv2-h3-s12@last,expv2-h3-s222@last     # CPU
```

`--calibrate` freezes the two Tier-1 sigmas, runs the §6.3 sanity gate and picks the shift.
It writes everything to `eval/tier1/calibration.json`; the chosen shift is
`shift.winner_train_flow_shift`, which `train.sh` reads by itself. **If neither pilot passes the sanity
gate, stop** (re-check G0.2b; do not touch the LR). If both pilots sit *above* the base on
most clips, re-run both pilots at `LR=5e-5`.

```bash
bash expv2-minimax-h3/configs/train.sh expv2-h3-main      # ~6 h, 10 checkpoints; SHIFT from calibration.json
python expv2-minimax-h3/eval/tier0.py --run expv2-h3-main                                  # CPU, any time during the run
```

---

## 8. Phase 3 — evaluate and decide

```bash
python expv2-minimax-h3/eval/dcls.py --ckpt base,expv2-h3-main@all --set holdout --sigmas calibrated
python expv2-minimax-h3/eval/generate.py --ckpt expv2-h3-main@all --suites probe
python expv2-minimax-h3/eval/score.py --ckpt expv2-h3-main@all --suites probe
python expv2-minimax-h3/eval/select_winner.py --run expv2-h3-main            # PLAN 7.3 -> reports/winner.json

W=<winner name from reports/winner.json, e.g. expv2-h3-main@250>
python expv2-minimax-h3/eval/generate.py --ckpt "$W" --suites core,native,gb
python expv2-minimax-h3/eval/score.py --ckpt "$W" --suites core,native,gb
```

**Two manual rubrics.** Both are filled in by eye; the scripts only write the templates.

1. **V2:** `python expv2-minimax-h3/eval/verdict.py --rubric-template --ckpt "$W"` writes
   `eval/reports/rubric_<ckpt>.json` with one entry per `core` `QF2_R` clip. For each clip,
   judge flat fills, intact black outlines, no 3D shading, on-model shape and correct colour,
   and record "washed out / foggy" when that is why it fails. Arm D's trigger reads that
   last field.
2. **§8, blinded:** the `h2h_v8.py --rubric-template` command below writes all 96 clips (H3 and
   v8, 48 each) as same-size mp4s under shuffled ids, with the key kept apart. Rate
   `eval/reports/h2h_rubric.json` without opening the key.

```bash
python expv2-minimax-h3/scripts/azure_sync.py pull v8                                     # v8@550's 52 core clips
python expv2-minimax-h3/eval/h2h_v8.py --ckpt "$W" --v8-dir "$EXPV2_WORK/v8_550" --rubric-template
#   ... rate eval/reports/h2h_rubric.json (blind) ...
python expv2-minimax-h3/eval/h2h_v8.py --ckpt "$W" --v8-dir "$EXPV2_WORK/v8_550"            # PLAN 8
python expv2-minimax-h3/eval/verdict.py --arm A                                           # PLAN 9 -> reports/verdict.{json,md}
```

### Arm D, only if PLAN §6.6 fires

`verdict.py` prints whether the arm-D trigger fired: V11 failed, ≥ 20% of `core` `QF2_R`
clips are washed out, or the probe saturation stayed < 0.9× source.

```bash
python expv2-minimax-h3/setup/fetch_models.py --set adapter --verify
bash expv2-minimax-h3/configs/train.sh expv2-h3-d          # same SHIFT, cfg 1 + the DeCFG adapter
# then section 8 again with expv2-h3-d in place of expv2-h3-main
# (select_winner.py --run expv2-h3-d, ...), and finally: verdict.py --arm both
```

---

## 9. Hand back

- `eval/reports/verdict.md` and `verdict.json`, `winner.json`, `h2h_v8.json`
- `eval/tier1/*.json`, `eval/scores/**`, `eval/phase0/**`
- the winner's LoRA (`output/<run>/step-<samples>.safetensors`) and the run's `expv2_run.json`, `loss.csv`, `train.log`

All of it is under `pudgy/expv2-minimax-h3/` once the mirror has run a final pass:
`python expv2-minimax-h3/scripts/azure_sync.py push eval weights logs`. The author writes
`docs/REPORT.md` from it.

## 10. Troubleshooting

| Symptom | Cause, fix |
|---|---|
| `patches: does NOT apply` | DiffSynth is not at `acf2ad2`, or has local edits. `git -C $DIFFSYNTH_DIR status`; re-run `setup_box.sh` |
| Stage 1 CUDA OOM | `CACHE_OFFLOAD=1 bash configs/cache.sh` (MODEL_NOTES 2.21) |
| Stage 2 `requires unconditional inputs` | The cache was built without `--training_cfg_scale 4`. Delete `cache/train/` and re-run `cache.sh` (2.19) |
| `--num_frames must be 17n+5` | A hand-written stage-2 command is missing `--num_frames 22` (2.21) |
| Far more checkpoints than expected | `--save_steps` was given in optimizer steps; it counts samples (2.18) |
| `check_loader.py`: a clip yields fewer than N frames | The mp4 lost its tail frame (2.17). Rebuild that clip with `prep_expv2.py`; never retrain on it |
| `FileNotFoundError` for a model | Run `fetch_models.py --set core` again. Paths are `$DIFFSYNTH_MODEL_BASE_PATH/<repo id>/<file>` |
| A run dir already holds checkpoints | `train.sh` refuses to mix runs. Move the old dir away, or `FORCE=1` |
