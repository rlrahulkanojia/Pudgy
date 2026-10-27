# v9b-1 handover: long-form scene training on a new GPU machine

**Written** 2026-10-27, ~06:20 UTC, on the old box (A100-SXM4 80GB). **For** whoever runs v9b-1 on
the new machine, a person or a Claude session. Read this file first, then
[`RUNBOOK_v9b.md`](RUNBOOK_v9b.md) §5–§7 (commands, gate, failure handling) and
[`Training_Approach_v9.md`](Training_Approach_v9.md) §0.1 (decisions D1–D10).

**Why the move.** v9b-1 starts from v8 and doesn't depend on the phone experiments. The user
moved it to its own GPU so the old box can keep working on the phone (v9c2) at the same time.

---

## 0. In short

**Where things come from:** code and docs from GitHub (branch `v9b-handover`); data and weights
from the handover archive on the old box, or from Azure (§3).

Everything up to the GPU steps is **done and verified on the old box**:
- the curation of all 99 skits;
- the training set, rebuilt on 27 Oct with the approved phone phrase;
- the held-out eval set;
- the scripts, with the gate tested on synthetic results.

Your job:
1. Set up the machine.
2. Download the bundle.
3. Verify the checksums.
4. Run the GPU steps: cache → evalprep → smoke → baseline → train (400 steps) → eval → archive.
5. Write the report.

**Don't re-curate, re-caption, rebuild the set or change the recipe.** Every decision below was
made by the user or checked against data. If something looks wrong, stop and ask the user.

---

## 1. What is fixed

| | |
|---|---|
| Model | Wan2.2-I2V-A14B. A LoRA on the **low-noise expert only**, timestep boundary 0.9, t ∈ [0, 900] |
| Base (D1) | `pudgy-v8-merged` = v7fixed + v8@550 combined exactly. Merged into the frozen DiT with `--base_weights`, so the saved LoRA is a pure v9b task vector |
| Recipe | `finetune/wan/train_pudgy_v9b.sh`: rank 32 / alpha 32, adamw8bit, lr 5e-5, cosine to ×0.1, warmup 20, grad-accum 8, **400 optimizer steps**, checkpoint every 50, `BLOCKS_TO_SWAP=24`, fp16, plain MSE (D8) |
| Data | `processed/v9b_535`:<br>• **348 scene windows**, 480 wide × 848 tall, 24 fps (D2): f21 160 · f33 72 · f49 82 · f81 34. 158 are two-shots; static camera only (D3); from 78 skits<br>• **187 v8 replay clips** at 1024² (35%, with the D10 label anchors) |
| Holdout | 6 whole skits: BackToBed, TirePressure, Drinker (it1), BackTogether, MorningFart, BubbleWrap (it2). None of their frames is in the set |
| Phone phrase | the approved identity-only phrase "the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top". 28 scene captions contain it |
| Registry | `finetune/wan/assets/registry.toml`, **md5 `3757257814f5588603317d747f2e250e`**. The bundle carries this version; the old box writes the identical file at ~07:50 UTC |
| Not in v9b-1 | the high-noise run (D7: only if E1a or the travel bar calls for it), camera moves, composites (v9b-2) |

**How the set was rebuilt** (27 Oct, old box; RUNBOOK_v9b §9):
- `prep_v9b.py` ran with the approved registry. The old set was kept as `v9b_535_blackscreen`.
- `handoff_verify.py` passed, comparing the two sets:
  - the same 348 windows;
  - exactly 28 captions changed, each by only the phrase;
  - replay rows and dataset configs identical;
  - all 535 clips byte-identical.

  The earlier checks on the clips therefore still hold: size, frame count, fps, no held-out leak, dupcheck.

---

## 2. The machine

| need | why |
|---|---|
| **GPU with 80 GB** (A100 80GB or H100 80GB) | v9c's smoke test peaked at **80,420 MiB** with 24 blocks swapped; v9b's heaviest bucket is slightly smaller. `setup_wan_env.sh` refuses less than 70 GB |
| RAM ≥ 128 GB | block swap + data loaders + caching (the old box has 503 GB) |
| Disk ≥ 200 GB free on `/workspace` | ~70 GB weights + caches + 8 checkpoints with optimizer states |
| CUDA 12.x driver, ffmpeg, internet to Hugging Face and Azure | weights come from HF, everything else from Azure |

`/workspace` is **not persistent** on Vast. Azure (container `pudgy`) is the only backup, so push
to it as you go (§5).

---

## 3. What to download

**Fastest: copy one archive straight from the old box.** Azure downloads to the old box measured
only **0.2–0.4 MB/s** on 27 Oct, while Hugging Face ran at 73 MB/s on the same machine. Through
Azure, the 2.7 GB below could take hours.

The old box holds everything in **`/workspace/handover_v9b/v9b_handover_all.tar`**:
- 2.66 GB; its SHA-256 is in `v9b_handover_all.tar.sha256` next to it (copy both);
- items 1–5 of the table below, already laid out at their `/workspace`-relative paths, plus
  `SHA256SUMS` (571 files) and this document.

Old box at the time of writing: Vast instance C.57545515, IP `80.188.223.202`, SSH port `11312`,
Jupyter port `11350`. Ask the user for access, either an SSH key added to the old box or its
Jupyter token.
```bash
cd /workspace
# either SSH (the new box's public key in the old box's /root/.ssh/authorized_keys):
rsync -aP -e "ssh -p 11312" root@80.188.223.202:/workspace/handover_v9b/v9b_handover_all.tar\* .
# or the old box's Jupyter (token from the user / Vast console):
for f in v9b_handover_all.tar v9b_handover_all.tar.sha256; do
  curl -fL -o $f "http://80.188.223.202:11350/files/workspace/handover_v9b/$f?token=<TOKEN>"; done

sha256sum -c v9b_handover_all.tar.sha256
tar -xf v9b_handover_all.tar -C /workspace && sha256sum -c --quiet SHA256SUMS && echo "ALL 571 OK"
# Code and docs come from GitHub (Step 1). The archive's Pudgy_code tarball is only a fallback copy.
# Then do Step 1 (clone + .env) and Step 2, then Step 3's links, eval-tar extract and md5 check,
# skipping Step 3's downloads.
```
The same archive was test-extracted on the old box, and all 571 checksums passed. If the old box
is gone, use Azure: the same files, the table below and Steps 1 and 3.

The Azure container is **`pudgy`**. The connection string is in `/workspace/Pudgy/.env`
(`AZURE_STORAGE_CONNECTION_STRING`, plus optional `WANDB_API_KEY`). **The `.env` is not in the
bundle**, because it holds secrets. Get it from the user or copy it from the old box.

| # | item | Azure | goes to | check |
|---|---|---|---|---|
| 1 | Code and docs: the repo with the approved registry and this file | **GitHub `rlrahulkanojia/Pudgy`, branch `v9b-handover`** (fallback: `v9b/handover/Pudgy_code_2026-10-27.tar.gz`) | `/workspace/Pudgy` | registry md5 |
| 2 | Merged v8 base (1.5 GB) | `v9b/base/pudgy-v8-merged.safetensors` | `/workspace/wan_output/v8_merged/` | md5 `ec333748284410bb341f431b88f5026f` |
| 3 | Golden LoRAs (the high-noise one runs every generation) | `v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors`, `…lownoise…` | `/workspace/restore/…`, linked into `/workspace/wan_output/v2_golden/` | present |
| 4 | Training set | `processed/v9b_535/` (566 files) | `/workspace/Data/processed/v9b_535`, linked as `/workspace/data_v9b` | `SHA256SUMS` |
| 5 | Eval sets: the regression set **with its latent cache**, the v9c base Tier-1 rows, and the v9b held-out scene set | `v9b/handover/eval_for_v9b.tar` (405 MB) | extract in `/workspace` → `eval_v9/holdout`, `eval_v9/runs/pudgy-v9c-phone-lownoise/dcls`, `eval_v9b` | `SHA256SUMS` |
| 6 | Base weights (~65 GB): low- and high-noise DiT fp16, VAE, T5 | Hugging Face, via `setup_wan_env.sh` | `/workspace/wan_models` | the script checks |
| 7 | musubi-tuner **v0.3.4 + `finetune/wan/patches/musubi-v0.3.4-resume-step.patch`** | GitHub, via `setup_wan_env.sh` | `/workspace/musubi-tuner` | "patch already applied" on a re-run |

**Not needed:**
- the raw skits and v8 sources: the set is final, so prep doesn't run here;
- `v9_box.sh restore`: it pulls raw iteration_3–6 and the v8 parts;
- `v9_box.sh merge`: item 2 is the merged file.

**The paths are fixed.** The scripts hard-code `/workspace/...`, so keep exactly the layout above.

---

## 4. Steps

Each step ends with what you must see. Log every step, with UTC time and numbers, in
`RUNBOOK_v9b.md` §9 under a "new machine" heading.

### Step 1. Code and secrets
```bash
cd /workspace
git clone -b v9b-handover https://github.com/rlrahulkanojia/Pudgy.git /workspace/Pudgy
git -C /workspace/Pudgy log --oneline -3
md5sum /workspace/Pudgy/finetune/wan/assets/registry.toml    # 3757257814f5588603317d747f2e250e
cp <the .env from the user / old box> /workspace/Pudgy/.env  # git-ignored; never commit it
```
Fallback, only if GitHub is unreachable: fetch the same code from Azure.
```bash
cd /workspace
# First copy the .env (from the user / old box) to /workspace/Pudgy.env, then fetch the code
# bundle with a throwaway client (azure_sync.py is inside the bundle):
pip install -q azure-storage-blob
set -a; . /workspace/Pudgy.env; set +a
python - <<'EOF'
import os
from azure.storage.blob import BlobServiceClient
c = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
for n in ("v9b/handover/Pudgy_code_2026-10-27.tar.gz", "v9b/handover/SHA256SUMS"):
    open(n.split("/")[-1], "wb").write(c.download_blob(n).readall())
EOF
grep Pudgy_code SHA256SUMS | sha256sum -c -
tar -xzf Pudgy_code_2026-10-27.tar.gz          # -> /workspace/Pudgy
cp Pudgy.env /workspace/Pudgy/.env
md5sum /workspace/Pudgy/finetune/wan/assets/registry.toml   # 3757257814f5588603317d747f2e250e
```

### Step 2. Environment and base weights
```bash
bash /workspace/Pudgy/finetune/wan/v9_box.sh setup     # = setup_wan_env.sh: venv, musubi 0.3.4 + patch, weights
```
**Must see:**
- the VRAM preflight passing;
- `musubi-tuner` at v0.3.4 with the patch applied;
- these files present:
  - `wan_models/comfy22/split_files/diffusion_models/wan2.2_i2v_{low,high}_noise_14B_fp16.safetensors`
  - `wan_models/comfy21/split_files/vae/wan_2.1_vae.safetensors`
  - `wan_models/t5/models_t5_umt5-xxl-enc-bf16.pth`

### Step 3. Bundle
`azure_sync.py pull --dest D <blob>` writes to `D/<full blob path>`. The paths below follow that
rule, and they are the paths `SHA256SUMS` lists.
```bash
cd /workspace/Pudgy/finetune/wan; PY=/workspace/Pudgy/.venv-wan/bin/python
$PY azure_sync.py pull --dest /workspace/restore v9b/base/pudgy-v8-merged.safetensors \
    v2/weights/curated/lora_highnoise_GOLDEN_ep40.safetensors v2/weights/curated/lora_lownoise_GOLDEN_ep40.safetensors
$PY azure_sync.py pull --dest /workspace/Data processed/v9b_535/          # -> /workspace/Data/processed/v9b_535/
$PY azure_sync.py pull --dest /workspace v9b/handover/eval_for_v9b.tar    # -> /workspace/v9b/handover/eval_for_v9b.tar

mkdir -p /workspace/wan_output/v8_merged /workspace/wan_output/v2_golden
ln -sf /workspace/restore/v9b/base/pudgy-v8-merged.safetensors /workspace/wan_output/v8_merged/
ln -sf /workspace/restore/v2/weights/curated/lora_{high,low}noise_GOLDEN_ep40.safetensors /workspace/wan_output/v2_golden/
ln -sfn /workspace/Data/processed/v9b_535 /workspace/data_v9b
cd /workspace && sha256sum -c --quiet SHA256SUMS        # BEFORE extracting; no output = every file matches
tar -xf /workspace/v9b/handover/eval_for_v9b.tar -C /workspace
md5sum /workspace/wan_output/v8_merged/pudgy-v8-merged.safetensors   # ec333748284410bb341f431b88f5026f
```
**Must see:**
- the merged base's md5 matches;
- `sha256sum -c` prints nothing;
- `/workspace/data_v9b/manifest.json` exists;
- `ls /workspace/data_v9b/clips | wc -l` = **535**.

**Don't run `v9b_box.sh check` or `prep`.** Both need the raw skits. They ran on the old box, and
the checksums prove the files are the ones they checked.

### Step 4. Cache the training set (GPU)
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh cache
```
**Must see:**
- `verify_cache_v8` OK;
- the shape check: every `scene_f*` directory at **H×W (106, 60)** and every `replay_f*` directory
  at **(128, 128)**, all marked `ok`. A wrong shape means silent re-bucketing, so stop and ask.

Old-box estimate: ~15–25 min.

### Step 5. Cache the eval scene set
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh evalprep
```
`build_eval_v9b.py` finds every clip and keyframe already present, so it decodes nothing. Then it
caches latents.

**Must see:**
- 15 Tier-1 windows and 8 Tier-2 jobs in `/workspace/eval_v9b/scenes/index.json` and `suite_jobs.json`;
- latents under `/workspace/eval_v9b/scenes/cache/f33`.

### Step 6. Smoke test (2 steps on the heaviest bucket)
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh smoke
grep -ac "memory mapping failed with OOM" /workspace/logs_v9b/smoke.log
```
**Pass:**
- it prints `SMOKE OK`;
- the grep count is **0**;
- the loss is finite.

**Peak memory near 80 GB is normal.** nvidia-smi shows the allocator's reservation:
- v9c: 80,420 MiB at swap 24, clean;
- v9c at swap 22: 81.2 GB *with* mapping warnings.

**Do not gate on a memory number.** If the grep count is above 0, or the log has `out of memory`:
`BLOCKS_TO_SWAP=26 bash .../v9b_box.sh smoke`, then 28. Use the passing value for training
(`BLOCKS_TO_SWAP=… bash v9b_box.sh train`) and record it.

Write down the s/it.
- v9c reference, A100: 63–64 s/it in its smoke and **487 s per optimizer step** in the run. v9c's
  clips have ~2× v9b's average tokens, but block swapping doesn't shrink with them.
- Plan estimate for v9b-1 on an A100: **≈ 30–45 h** for training plus Tier 1. An H100 should be faster.

### Step 7. Baseline (Tier 1 on v8merged)
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh baseline
```
`tier1_v9b.py` re-scores the first 6 regression clips and compares them with the v9c base rows,
which were scored on the old A100.
- **If they match to 1e-3,** it reuses the rows and scores only the scene set.
- **If not,** it re-scores everything: ~3 h on an A100, which is expected on a different GPU and fine.

Record which one happened.

**Must see:** a `v8merged` result under `/workspace/eval_v9b/runs/pudgy-v9b-lownoise/dcls/`.

### Step 8. Train
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh train      # background trainer + Azure monitor (MONITOR_PREFIX=v9b)
bash /workspace/Pudgy/finetune/wan/v9b_box.sh status
```
**Watch for:**
- the first optimizer steps completing;
- a finite `avr_loss` (v9c went 0.0021 → 0.0011; v9b's scale will differ);
- no mapping warnings piling up (one in a 27 h run is survivable, as in v9c);
- checkpoints `pudgy-v9b-lownoise-step000000{50,100,…}.safetensors` appearing every 50 steps and
  reaching Azure `v9b/weights/` through the monitor.

**ETA** = 400 × (seconds per optimizer step over steps 5–10). Tell the user.

**Recommended mid-run check at step 200** (RUNBOOK_v9b §6):
1. `v9b_box.sh pause 200`: waits until step 200's state is fully written, then stops the trainer by
   its script name;
2. `v9b_box.sh tier1 200`, then read the gate output;
3. `v9b_box.sh train`: resumes from the latest `*-step*-state`.

If step 200 fails R1 (regression), stop and report. Don't burn the second half.

### Step 9. Evaluate, pick, archive
```bash
bash /workspace/Pudgy/finetune/wan/v9b_box.sh eval      # Tier 1 on v8merged + 100/200/300/400, gate, Tier-2 scene suite on the winner
bash /workspace/Pudgy/finetune/wan/v9b_box.sh archive   # weights, logs, eval, set -> Azure v9b/ + processed/v9b_535
```
**The gate** (`gate_v9b.py`; plan §5.4):
- R1: per-kind paired Δmargin, CI lower bound ≥ −0.003, and accuracy down by ≤ 1 clip;
- R2: per-t loss ratio ≤ 1.05;
- L1: scene loss ratio < 1;
- P2: text-attributable gain, CI upper bound < 0.

The winner is the passing checkpoint with the lowest L1. **If nothing passes,** `eval` stops with
"NO CHECKPOINT PASSES": the stage isn't kept, and you report (RUNBOOK_v9b §7).

Tier 2 generates the held-out skits' first frames at `--video_size 848 480` (height width), seeds
42/43/44. It scores duplicate characters, drift, frozen frames, jumps and the motion ratio, and it
makes contact sheets. Plan estimate for the whole eval: ≈ 20 h.

### Step 10. Report
Write `docs/training_reports/v9/REPORT_v9b.md`, following `REPORT_v9c_phone.md`: headline, what was
trained, Tier 1 table, gate, Tier 2 with sheets, wall clock, problems. Update the plan's status line
and §0.1. Send the user a short summary with sample videos.

---

## 5. Keeping things safe

- **Azure while it runs:** loop the sync, e.g.
  `nohup sh -c 'while true; do bash /workspace/Pudgy/finetune/wan/v9b_box.sh sync; sleep 1800; done' > /workspace/logs_v9b/sync.log 2>&1 &`
  It pushes curation, code, docs and logs to `v9b/work` and `v9b/logs/box`. The monitor pushes checkpoints.
- **Prefixes you own:** `v9b/` and `processed/v9b_535`.
  - **Never write under `v9c/`**: the phone track owns it.
  - The old box's v9b logs up to the handover are frozen under `v9b/logs/box_a100_pre_handover/`.
- **If the machine dies,** set up a new one with this file. Pull the latest `*-step*-state` from Azure (or the last
  checkpoint), and `v9b_box.sh train` resumes from it. The resume patch in musubi makes the step
  counter continue.

## 6. Pitfalls already paid for

| don't | because |
|---|---|
| `pkill -f <pattern>` | it matches and kills your own shell. Kill by PID; `pause` uses `pgrep -f src/musubi_tuner/wan_train_network.py` |
| more than 16 parallel ffmpeg decodes, or decodes without `-threads 4` | 128 concurrent decodes deadlocked ffmpeg 6.1 |
| gate the smoke test on a memory number | the allocator reserves almost all 80 GB on a healthy run; the mapping warning is the signal |
| treat scene clips as 848 wide | they are **480 wide × 848 tall** (portrait skits); latent H×W (106, 60); generation takes `--video_size 848 480` (height width) |
| edit `registry.toml`, captions or curation | the set is final and verified; any change means a rebuild and re-verification (RUNBOOK_v9b §5). The registry is shared with the phone track, which owns it |
| run a second GPU job next to the trainer | it uses the whole 80 GB |
| drop `HF_HUB_OFFLINE=1` in eval | the box steps set it; without it, transformers calls home |
| trust `azure_sync.py push`'s size check after editing a file in place | it skips any blob whose size already matches, so a same-size edit never uploads (27 Oct: `SHA256SUMS` went stale this way). Use `push --force` for files you edit in place, and `finetune/wan/azure_verify_md5.py PREFIX=PATH` to compare by content |

## 7. The old box (phone track) and merging later

- The old box keeps its GPU for the phone work (v9c2: the asset embedding and the phone-turn fixes).
  It owns `registry.toml` and the phone docs (`RESEARCH_v9c_fixes.md`, `RUNBOOK_v9c_phone.md`, `REPORT_v9c_phone.md`).
- **v9b-1 is unaffected by later registry edits**, because its captions are already built. Don't pull
  registry changes into this run.
- Both machines edit the repo. To merge cleanly, edit on this machine only `RUNBOOK_v9b.md`,
  `HANDOVER_v9b.md`, the v9b rows of `Training_Approach_v9.md` (§0.1, §4, status line), `REPORT_v9b.md`, and
  `finetune/wan/{eval_v9b,scenes}/`, `prep_v9b.py`, `train_pudgy_v9b.sh` and `v9b_box.sh` if a fix
  is needed. Log each fix in the runbook.
- Git: on 27 Oct the old box's work was committed to branch **`v9b-handover`** and pushed, in three
  commits: the phone track (v9c report, v9c2 research and tooling, the approved registry), v9b-1, and
  the `azure_sync --force` fix.
  `main` is unchanged. Commit further v9b work on that branch, and push or merge when the user asks.

## 8. Inventory

`SHA256SUMS` (on Azure at `v9b/handover/`, and inside the archive) lists SHA-256 for 571 files.
The paths are relative to `/workspace`, exactly where the archive or steps 1 and 3 put them:

| path | what |
|---|---|
| `Pudgy_code_2026-10-27.tar.gz` | the code |
| `v9b/handover/eval_for_v9b.tar` | the eval sets |
| `restore/v9b/base/pudgy-v8-merged.safetensors` | the merged base |
| `restore/v2/weights/curated/lora_{high,low}noise_GOLDEN_ep40.safetensors` | the golden LoRAs |
| `Data/processed/v9b_535/…` | the training set, 566 files |
