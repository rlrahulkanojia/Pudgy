# GPU Handover — Training Approach v7

**Audience:** whoever runs v7 on an 80 GB GPU box. Written to be executed **without the
author present** — every step is copy-paste, and every "report back" box names the artifact
to send back.

**The author is on a Mac with no CUDA** and cannot run any of this.

**Read first:** [`Training_Approach_v7.md`](Training_Approach_v7.md) — the plan, the
evidence behind each decision, and the gates. This document is only the executable part.

---

## 0. Hardware and prerequisites

| | |
|---|---|
| GPU | **1× 80 GB** (A100-80GB or H100-80GB) minimum. **2× is strongly preferred** — the expression run and the motion A/B use different experts and different configs, so they run concurrently and halve wall-clock. |
| Disk | **~200 GB free.** ~65 GB base weights + ~1 GB v2 goldens + 236 MB data + checkpoints (each is 307 MB, and we keep every one — see §5) + resume states if enabled. |
| CUDA | 12.x, driver recent enough for bf16/fp16 on the card. |
| Time | expression ≈ 34 h · motion A/B ≈ 30 h · motion full run ≈ 25 h. |
| ⚠️ Disk persistence | **`/workspace` is NOT persistent.** The v5 box was destroyed with its whole environment on it, and its training data was lost because only the weights had been uploaded. Upload artifacts as you go (§7), do not leave them only on the box. |

---

## 1. Credentials — do this first, everything else depends on it

The box needs the Azure connection string to pull data and push results.

```bash
# Copy the repo's .env to the box (it is NOT in git — get it from the author)
scp .env <box>:/workspace/.env
grep -c AZURE_STORAGE_CONNECTION_STRING /workspace/.env    # expect 1
```

`azure_upload_v7.py` reads `/workspace/.env` directly. For the `az` CLI commands below,
either export it or use `az login`:

```bash
export AZURE_STORAGE_CONNECTION_STRING="$(grep AZURE_STORAGE_CONNECTION_STRING /workspace/.env | cut -d= -f2- | tr -d '"')"
```

> **Report back:** confirm `az storage blob list --account-name pudgytraining -c pudgy --prefix processed/v7_primitives_2272/ --num-results 1` returns a blob.

---

## 2. Environment — one script, idempotent

```bash
git clone <repo> /workspace/Pudgy && cd /workspace/Pudgy
bash setup_wan_env.sh              # venv + musubi-tuner 0.3.4 + ~65 GB base weights
```

Safe to re-run after an interruption — every step checks before doing.
`SKIP_WEIGHTS=1` if the weights are already present.

> **Report back:** the tail of `setup_wan_env.sh` output, plus `nvidia-smi`.

---

## 3. Pull the goldens and the data

```bash
# the v2 goldens — the init for both runs, and the frozen inference partner
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace/wan_output/v2_golden \
   --pattern "v2/weights/curated/*GOLDEN_ep40*"
ls -la /workspace/wan_output/v2_golden/v2/weights/curated/   # 2 files, 306.8 MB each

# the v7 dataset — ONE command, both kinds
az storage blob download-batch --account-name pudgytraining \
   --source pudgy --destination /workspace \
   --pattern "processed/v7_primitives_2272/*"
mkdir -p /workspace/data_v7 && mv /workspace/processed/v7_primitives_2272/* /workspace/data_v7/
ls /workspace/data_v7/clips/*.mp4 | wc -l                    # expect 2272
```

The `*.workspace.toml` / `*.workspace.jsonl` in that folder already contain `/workspace/`
paths — **nothing needs editing.** They expect clips at `/workspace/data_v7/clips` and
caches under `/workspace/wan_cache/latents_v7/<kind>`.

> **Report back:** the two `wc -l` / `ls` counts.

---

## 4. Pre-flight — check this BEFORE committing 11 hours

```bash
python /workspace/musubi-tuner/src/musubi_tuner/wan_train_network.py --help \
  | grep -q save_every_n_steps && echo "OK" || echo "MISSING"
```

- **OK** → proceed with `SAVE_EVERY_STEPS=250` below.
- **MISSING** → drop `SAVE_EVERY_STEPS` and use `SAVE_EVERY=1` (per epoch) instead, and
  **tell the author** — an epoch is 1,072–1,200 steps, so per-epoch saving yields only 3–4
  checkpoints for a whole run, and v5 showed the best checkpoint is usually *early*. The
  run is still valid, but the golden sweep will be coarse.

---

## 5. Precompute caches

```bash
cd /workspace/musubi-tuner
for KIND in motion expression; do
  python src/musubi_tuner/wan_cache_latents.py \
      --dataset_config /workspace/data_v7/dataset_config_${KIND}_v7.workspace.toml \
      --vae /workspace/wan_models/wan_2.1_vae.safetensors --i2v
  python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
      --dataset_config /workspace/data_v7/dataset_config_${KIND}_v7.workspace.toml \
      --t5 /workspace/wan_models/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16
done
```

Adjust the `--vae` / `--t5` paths to wherever `setup_wan_env.sh` put them.

> **Report back:** `du -sh /workspace/wan_cache/latents_v7/*`.

---

## 6. Train

Two runs. **On a 2-GPU box run them concurrently** (`CUDA_VISIBLE_DEVICES=0` / `=1`) —
different experts, different configs, no shared output directory.

```bash
cd /workspace/Pudgy

# A. EXPRESSION — settled, no A/B. 1,200 steps/epoch, ~11.7 h/epoch.
CUDA_VISIBLE_DEVICES=0 \
DATASET=/workspace/data_v7/dataset_config_expression_v7.workspace.toml \
NAME=pudgy-v7-expr-lownoise EXPERT=low EPOCHS=3 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh 2>&1 | tee /workspace/train_v7_expr.log

# B. MOTION A/B — 2 epochs each, then STOP. 1,072 steps/epoch, ~7.5 h/epoch.
CUDA_VISIBLE_DEVICES=1 \
DATASET=/workspace/data_v7/dataset_config_motion_v7.workspace.toml \
NAME=pudgy-v7-motion-highnoise EXPERT=high EPOCHS=2 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh 2>&1 | tee /workspace/train_v7_motion_high.log

CUDA_VISIBLE_DEVICES=1 \
DATASET=/workspace/data_v7/dataset_config_motion_v7.workspace.toml \
NAME=pudgy-v7-motion-lownoise EXPERT=low EPOCHS=2 SAVE_EVERY_STEPS=250 \
  bash finetune/wan/train_pudgy_v7.sh 2>&1 | tee /workspace/train_v7_motion_low.log
```

`EPOCHS` is a **ceiling, not a target** — see plan §5.2. Stop where the gates plateau.

**`BLOCKS_TO_SWAP=0`** (the default) unless you hit a measured OOM. v5 found block-swap
costs 2× throughput when VRAM is free (27.5 → 13.7 s/it).

### ⚠️ First gate at ~500 steps — about 5 hours in. Do not skip it.

Run **G-C** and **G-M** on checkpoint 2 while training continues. Calibration from v5:

| Signal | Healthy | Broken |
|---|---|---|
| Pairwise SSIM between opposite prompts | **0.9340** — prompt works | **0.9692** — prompt ignored |
| Subject x-range, novel start frame | **221–267 px** | **0.8 px** — collapse |

If nothing has moved: **raise LR to 5e-5** (plan evidence #4). Do not just add steps —
v5 measured motion responsiveness *decaying* with training.

> **Report back:** the ~500-step gate numbers for both runs, and the loss curve.

---

## 7. Gates

**The v6 gate harness is the starting point — it already implements most of what v7 needs.**
These run against a checkpoint, drive the real inference path, and measure the produced
video (nothing is inferred from loss — v5 §4.3 showed loss and weight-drift both said
"fine" while the model had stopped responding to prompts):

| Script | Implements |
|---|---|
| `finetune/wan/gates_v6.py` | G-C controllability matrix + verdict |
| `finetune/wan/gate_gl_v6.py` | G-L length de-confound |
| `finetune/wan/gate_gf_v6.py` | G-F training-frame trigger |
| `finetune/wan/gates_remaining_v6.py` | G-F multi-seed, G-B, G-Z, G-H, G-L for Polly — batched into ONE model load |
| `finetune/wan/eval_v6.sh` | single two-expert FLF2V generation primitive |
| `finetune/wan/sample_epochs_v6.py` | per-checkpoint visual quality curve |

⚠️ **They need adapting for v7 — they are not drop-in.** Three differences:

1. **7 expression labels, not 4** (adds `confused`, `crying`) and **6 motion labels**. The
   gate matrices and their caption constants are sized for v6's four emotions.
2. **Two experts.** v6 trained low-noise only and loaded the v2 golden as the frozen
   partner. In v7 the motion LoRA may also be trained (pending G-X), so gates need a
   two-LoRA load — and **G-S** (plan §7) exists specifically to test the pair.
3. **G-D is new** — duration control across the ladder. v6 had one length per label so the
   question could not be asked.

Budget roughly a day of adaptation. `gates_v6.py` shares its caption constants with
`prep_expressions_v6.py` so eval prompts sit in the training distribution by construction;
keep that property by importing from `prep_v7.py` rather than retyping strings.

Full gate table with pass criteria: plan **§7**.

---

## 8. Upload — do this as you go, not only at the end

```bash
python finetune/wan/azure_upload_v7.py --dry-run
python finetune/wan/azure_upload_v7.py
```

Pushes weights, eval, logs and docs to `pudgy/v7/`. Idempotent — re-run after more eval and
it only sends what is new. `--with-states` adds resume states (large; excluded by default,
same policy as v2/v5).

> ⚠️ **The v5 box was lost with its training data on it** because only weights had been
> uploaded. Do not let artifacts exist in one place.

> **Report back:** the final `DONE — v7/ now holds N blobs` line.

---

## 9. Things that will bite you

- **`/workspace` is not persistent.** Assume the box can vanish.
- **`NAME=` must differ per run.** The two motion A/B arms would otherwise share an output
  directory and overwrite each other's checkpoints. `train_pudgy_v7.sh` makes it
  overridable; the v5 script did not, which is why v7 has its own copy.
- **Rank/alpha must stay 16/32.** The v2 goldens are rank 16; rank-8 tensors cannot load
  them, and the trainer's error message is not obvious about why.
- **Keep every checkpoint.** v5's golden was mid-run, and v5 §4.4 measured motion
  responsiveness decaying with steps. Do not keep only the last.
- **Square 1024² is off the base's canonical aspect list** (720×1280 / 1280×720 / 480×832 /
  832×480). This is a known, accepted quality cost inherited from square source art — not a
  bug to fix on the box.
- **3 source clips are unusable and already excluded** from the dataset; 17 more were
  recovered from damaged files. Nothing to do on the box — flagged so the counts make sense
  (`confused` has 16 sources, `walking` 16, where others have 18).

---

## 10. Report-back checklist

- [ ] §1 blob list returns a result
- [ ] §2 `setup_wan_env.sh` tail + `nvidia-smi`
- [ ] §3 clip count = 2272, goldens = 2 files
- [ ] §4 pre-flight OK or MISSING
- [ ] §5 cache sizes
- [ ] §6 **~500-step gate numbers** + loss curves ← the most important one
- [ ] §7 gate verdicts per checkpoint
- [ ] §8 upload confirmation
