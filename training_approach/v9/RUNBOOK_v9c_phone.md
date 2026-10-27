# Runbook: v9c-phone, the first prop stage

How to run the phone stage on a fresh GPU box, start to finish. Each step is one command
from `finetune/wan/v9_box.sh`, safe to re-run, and each lists what "done" looks like. The
plan behind it is [Training_Approach_v9.md](Training_Approach_v9.md) §4–§6 and
[Assets_v9.md](Assets_v9.md); the data is
`raw/iteration_6/CHANGELOG.md` (in `Data/`, outside the repo; Azure `pudgy/raw/iteration_6/CHANGELOG.md`).

**What this stage trains.** A fresh low-noise LoRA (rank 16) on top of the v8 production
model, merged into one base file. It learns the iFin phone from the client's 36 texting
clips (32 trained, 4 held out), while 20% of every batch replays v8's data so nothing
earlier is lost. One phrase names the phone everywhere: *"the iFin phone, a small pale grey
phone with a black screen, three camera lenses on the back and a little tuft on top"*
(`finetune/wan/assets/registry.toml`).

| | |
|---|---|
| **Box** | 1× A100 80 GB, ≥ 64 GB RAM, ≥ 300 GB disk, CUDA 12.x driver, `ffmpeg` installed |
| **Branch** | `v9/phone-stage` |
| **Inputs (Azure `pudgy`)** | `raw/iteration_3–6`, `v8/base/pudgy-v7-fixed-combined.safetensors`, `v8/weights/…/step00000550`, `v2/weights/curated/*GOLDEN_ep40*` (all present, checked 2026-10-24) |
| **Outputs (Azure)** | `v9c/weights/`, `v9c/base/`, `v9c/eval/`, `v9c/logs/`, `processed/v9c_phone_<N>/` |
| **Time** | setup ~1 h · prep ~0.5–1 h · cache ~1 h · baseline ~1 h · **train ~20 h** · eval ~4 h |
| **Measured (1× A100-SXM4 80GB, 26 Oct)** | prep ~1 min with `PREP_WORKERS=16` · baseline 3 h · **train 27 h** (487 s/step) · Tier 1 ~90 min per config · Tier 2 ~16.5 min per clip (phone suite 4.4 h) |

---

## 0. Before you start

```bash
git clone https://github.com/rlrahulkanojia/Pudgy /workspace/Pudgy
cd /workspace/Pudgy && git checkout v9/phone-stage
cp <your .env> /workspace/Pudgy/.env      # AZURE_STORAGE_CONNECTION_STRING (+ WANDB_API_KEY, optional)
```

The `.env` is never in git. `setup_wan_env.sh` downloads weights with the `hf` CLI at
`/venv/main/bin/hf` (vast.ai image). On another provider, `pip install -U huggingface_hub`
and run with `HF_BIN=$(which hf)`.

Run long steps in `tmux` so a dropped SSH session does not kill them.

## 1. Environment, data, base

| step | command | done when |
|---|---|---|
| setup | `bash finetune/wan/v9_box.sh setup` | prints `environment ready`; torch sees the GPU |
| restore | `bash finetune/wan/v9_box.sh restore` | `/workspace/Data/raw/iteration_{3,4,5,6}` present; `ls /workspace/wan_output/*/` shows v7fixed, v8 step 550, both goldens |
| merge | `bash finetune/wan/v9_box.sh merge` | `/workspace/wan_output/v8_merged/pudgy-v8-merged.safetensors` (~1.5 GB) |

`merge` adds v7fixed and v8@550 into one exact LoRA file (`lora_tools_v8.py combine`).
That file is the base this stage trains on, and the base every later stage builds on.

## 2. Training set

```bash
bash finetune/wan/v9_box.sh prep
```

This lints the registry, then runs `prep_v9.py`, which prints what it found before building
anything. **Check these lines in `/workspace/logs_v9c/prep.log`:**

- `iteration_6: 230 flat clips are md5 re-sends`, with **no** `! NEW iteration_6 file` lines
- `superseded by CONFUSED/V2: 18` (the clean re-export replaces the old confused clips)
- `interactions (v9b, not built): 21 (2 filename overrides applied)`, including the
  client-approved `INTERACTION_WAVING_SIDE_R → walking SIDE_R`
- `holdout ['QF2_R']: 32 sources (4 prop)`
- `props: 32 training sources`. The 11 damaged Polly clips show their clean window,
  e.g. `window 6+58`, and shorter rungs.
- `replay: 95 clips … over 26 (character, label) groups`, each group 3–4
- `wrote 475 clips` and `built: /workspace/Data/processed/v9c_phone_475`

The numbers above come from a full local build on 2026-10-24. Different numbers mean the
data or the code differs; stop and find out why. The set is linked to `/workspace/data_v9c`.

```bash
bash finetune/wan/v9_box.sh cache       # latents + T5, then verify_cache_v8.py
```

Done when the verify step reports every row complete.

## 3. Evaluation assets and the baseline (before any training)

```bash
bash finetune/wan/v9_box.sh evalprep    # /workspace/eval_v9/{holdout,keyframes}
bash finetune/wan/v9_box.sh baseline    # Tier-1 on v8merged and on v8@550
```

`baseline` must end with **`merge check: … -> OK`**. That proves the merged file is the
same model as v8@550. On `MISMATCH`, do not train; rebuild the merge.

The `v8merged` numbers are the "before" every checkpoint is compared with:

- the v8-label losses (regression gate);
- the held-out phone loss;
- whether the phone's name beats the distractors ("binding").

## 4. Smoke test, then the run

```bash
bash finetune/wan/v9_box.sh smoke
```

This runs 2 steps on the longest bucket only (f61; v8's longest was f57). It reports peak
GPU memory, allocator-retry lines and s/it, and must print `SMOKE OK`.

- **Peak at the limit (≳ 79 GB) or any allocator retries:** set `BLOCKS_TO_SWAP=24`
  (v8: more swap was faster at the limit) and smoke again.
- **Expected speed:** roughly 45–65 s per clip at f61.

```bash
bash finetune/wan/v9_box.sh train       # background; resumes from the newest state if one exists
bash finetune/wan/v9_box.sh status      # step, loss, checkpoints
tail -f /workspace/logs_v9c/train.log
```

The run is ≤ 200 optimizer steps of 8 clips, saving every 25 (about 20 h). The monitor
mirrors checkpoints and logs to Azure `v9c/` every 30 min, because the box is not
persistent.

**If the box restarts:** run `train` again. It finds the newest `*-state` folder and
resumes; musubi is patched to continue the step count. Check that the log says
`continuing at global step N`.

## 5. Gate and choice (after training)

```bash
bash finetune/wan/v9_box.sh eval
```

1. **Tier-1** runs the diffusion classifier at steps 100–200, on the same clips and with
   the same noise as the baseline, so every comparison is paired.
2. **Gate** (`eval_v9/gate_v9.py`). The rule was fixed before the data. Measured on the
   held-out angle (QF2_R):

   | check | bar |
   |---|---|
   | v8-label loss vs `v8merged` (regression) | ≤ 1.05× |
   | held-out phone loss vs `v8merged` | < 1.00× |
   | binding accuracy (true phrase vs generic phone / book / cup / remote) | reported; breaks ties |

   The winner is the passing checkpoint with the lowest phone loss. It is written to
   `/workspace/eval_v9/runs/<RUN_NAME>/winner_v9c.json`. Results for each run go in their own folder there.
3. **Tier-2** (`suite_v9.py`) generates 16 clips each on `v8merged` and the winner:
   - **phone:** phone start frames, including held-out ones;
   - **textcall:** no phone in the first frame, the prompt asks for it;
   - **leak:** v8 prompts, where a phone must not appear.

   It scores phone presence with CLIP plus pop/freeze flags, and writes a contact sheet per
   clip. **Look at the sheets.** The numbers only point at what to look at.

**If no checkpoint passes** (`NO CHECKPOINT PASSES`), do not keep the stage. Re-run it with
more replay:

```bash
rm /workspace/data_v9c && PUDGY_DATA=/workspace/Data .venv-wan/bin/python finetune/wan/prep_v9.py \
  --replay-frac 0.35 --workers $(nproc) --out /workspace/Data/processed
ln -sfn /workspace/Data/processed/v9c_phone_<N> /workspace/data_v9c
```

Then run `cache`, and the remaining steps under a new run name. Every step reads
`RUN_NAME`:

```bash
export RUN_NAME=pudgy-v9c-phone-r35
bash finetune/wan/v9_box.sh train
# … after it finishes
bash finetune/wan/v9_box.sh eval && bash finetune/wan/v9_box.sh archive
```

## 6. Archive, then hand back

```bash
bash finetune/wan/v9_box.sh archive
```

This uploads to Azure:

- all checkpoints, plus the merged base (checkpoints are useless without it);
- logs and eval results (classifier JSONs, gate, suite scores and sheets);
- the processed training set.

**Report back:**

- `winner_v9c.json` and the gate table from `/workspace/logs_v9c/eval_gate.log`;
- the Tier-2 contact sheets for `phone`, `textcall` and `leak`;
- the smoke numbers (memory, s/clip) and the total wall-clock time.

**Do not delete the box until `archive` reports every size check matching.**

## What is deliberately not in this stage

- **Phone stills.** In image-to-video a single frame is conditioned on itself, so it
  carries almost no training signal. The 24 views are the inference keyframe pixels and
  the eval reference instead.
- **Plates and interactions.** These belong to v9b (composites and scenes), which needs
  the plate floor annotations first.
- **Fast motion.** v9a is still waiting on the client's Round 5 Tier A footage.
- **The v9 camera-clause caption change.** Captions stay byte-identical to v8's, so the
  only new variable in this stage is the phone.
