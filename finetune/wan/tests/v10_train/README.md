# v10 training-side tests (CPU only)

Tests for the v10 Stage 1 training code (Training_Approach_v10.md Q10, Q16, Q17):
`finetune/wan/asset_embed_v9.py` (preservation loss with an explicit replay marker),
`finetune/wan/cache_mark_v10.py` (marks musubi's text-encoder caches) and
`finetune/wan/train_pudgy_v10.sh` (the launcher). They run on a Mac or any machine without a GPU,
against the real musubi-tuner v0.3.4 code and the real umT5 tokenizer.

## Set up and run

```bash
bash finetune/wan/tests/v10_train/setup_env.sh    # once; ~40 s with a warm uv cache, needs uv + git + network
bash finetune/wan/tests/v10_train/run_all.sh      # ~2.5 min on an M-series Mac
VERBOSE=1 bash finetune/wan/tests/v10_train/run_all.sh   # also lists every check
```

Everything goes into a work dir, `$V10_TEST_WORK` (default `~/.cache/pudgy_v10_tests`):

| path | what |
|---|---|
| `venv/` | Python 3.11.10 (uv), torch 2.7.1 + torchvision 0.22.1 CPU, `requirements.txt` (pinned) |
| `musubi/` | kohya-ss/musubi-tuner at tag v0.3.4 (`30c658c`), read-only: imported, never modified |
| `hf/` | HF cache holding the umT5 tokenizer (`Wan-AI/Wan2.1-T2V-14B`, subfolder `google/umt5-xxl`) |
| `tmp/run.XXXX/` | temp files of one run; deleted when all tests pass, kept on failure |

The repo path is derived from the scripts' location. Nothing is written to the repo, and no bytecode
is written either. Optional: `PUDGY_DATA` (default `<repo>/../../Data`) for the v10 caption check below.

## Expected output

```
syntax (python): OK
syntax (bash -n): OK
test_call_dit.py: PASS (31 ok)
test_cache_mark.py: PASS (25 ok)
test_spans_real.py: PASS (4 ok)      # 2 ok + a SKIP line if the v10 captions file is not found
test_parser.py: PASS (5 ok)
test_train_sh.sh: PASS (35 ok)
ALL TESTS PASS
```

## What each test covers

**`test_call_dit.py`** checks the patched training step. It uses musubi's real `WanNetworkTrainer.call_dit`/`_call_dit`,
its `lora_wan` network and its `--base_weights` merge path, with a tiny stand-in DiT that has WanModel's call
signature and uses the text context. It checks:
- **Replay sample** (`preserve` = 1):
  - target = the base prediction with the LoRA at multiplier 0, with golden still merged in;
  - prediction = the LoRA-on prediction;
  - block swap is switched to forward-only around the base pass;
  - the multiplier is restored afterwards;
  - gradients reach the LoRA.
- **Scene window without the phone:** plain data target, no base pass.
- **Phone window:** E5 is added to exactly rows [s, e) of the T5 context, and the model sees the spliced context.
- **Stop conditions:** the run stops on an unmarked sample, a replay sample with a phone span, a wrong span length,
  or 100 samples without a replay sample.
- **Logging and resume:** the `PRESERVE_STATS` line every N steps, the tensorboard extras, the `.asset.safetensors`
  metadata, counters continuing after `--resume`, and `--strict-base-weights` rejecting a base LoRA module that
  the DiT doesn't have.
- **Backward compatibility:** with v9c2's flags (`--preserve` without `--preserve-marker`), prediction, target
  and block-swap calls are bit-identical to the committed original.

**`test_cache_mark.py`** runs `cache_mark_v10.py` on synthetic caches written with musubi's own cache writers
(`fixtures.py`: 6 + 2 scene windows with real repo captions, 3 of them naming the phone, plus 3 replay samples at 1024²). It checks:
- **Marking:**
  - `mark` writes 11 files and `check` passes;
  - a dry run writes nothing;
  - T5 rows stay bit-identical and musubi's metadata is kept;
  - a second `mark` is a no-op;
  - no temp files are left behind.
- **musubi reads the marks:** musubi's real training-time dataset pipeline (BlueprintGenerator → dataset group)
  delivers `preserve`/`assetspan` as per-sample lists in every batch.
- **Re-caching:** a musubi T5 re-cache with the same caption keeps the marks; a changed caption fails `check`
  until re-marked.
- **Smoke set:** `smoke` builds the 3-sample set (heaviest replay, phone and plain windows) and musubi sees 3 batches.
- **Exit 1 on inconsistencies:** a replay caption naming the phone, a missing `_te` file, a stale latent, a jsonl
  caption changed without re-caching, an E5 row count ≠ the phrase's token count, a badly named cache dir, and
  a cache dir outside `--cache-root`.

**`test_spans_real.py`** checks the phone-phrase token span on real captions with the real tokenizer. It must be
found exactly where the text contains the registry phrase, never twice, and every caption must fit in 512 tokens.
- It always runs on the v9b windows in the repo (`scenes/v9b_535_scene_windows.json`): 28/28 found.
- It runs on the v10 windows when `$PUDGY_DATA/processed/v10_scenes_748/captions_v10_windows.jsonl` exists:
  74/74 found on 2026-11-10.

**`test_parser.py`** checks the launcher's arguments:
- the exact musubi argument list from `train_pudgy_v10.sh` (dry run) parses with musubi v0.3.4's own parser,
  with 29 Stage 1 values as specified (high-noise single expert, golden via `--base_weights`, rank/alpha 16,
  lr 3e-5, t 900–1000, swap 24, …);
- the wrapper flags parse with `asset_embed_v9.py`'s argparse.

**`test_train_sh.sh`** runs `train_pudgy_v10.sh` with a fake `/workspace` and a fake `accelerate`, under macOS's bash 3.2. It checks:
- **Refusals:** an unmarked cache, `PRESERVE_WEIGHT=0`, two visible GPUs, and cache dirs outside `CACHE`.
- **The dry run:** every Stage 1 flag is present, and there is no `--dit_high_noise`.
- **Resume:**
  - auto-resume picks the newest *complete* state and skips an incomplete one;
  - `RESUME=none` starts fresh;
  - a state at or past MAX_STEPS exits 0;
  - an explicit incomplete state is refused.
- **Smoke mode:** 3 steps × 1 sample, its own output name, never resumes.
- **Smoke gate:** the gate lines given to the box script pass on a good log and fail on a bad one.
- **Pause pattern:** `pgrep -f 'asset_embed_v9\.py train .*--output_name pudgy-v10-highnoise( |$)'` matches the
  launcher and the trainer, but not a smoke run or the box shell.

Helpers: `common.py` holds the paths, env defaults and the CPU-only musubi import patch.
`fixtures.py` builds the synthetic caches. `build_ws.py` builds the fake `/workspace`.
`run_py.py` runs a repo script with the patch applied.

**The v9c2 original isn't stored here.** `test_call_dit.py` needs it for the backward-compatibility check, so it
reads it from git history (`git show 6450000:finetune/wan/asset_embed_v9.py`, the last commit before the v10
port). It writes it to the run's temp dir with its one CUDA-only line (`torch.device("cuda")`) pointed at the
CPU. This needs a clone that contains that commit (not a shallow clone without it).

## Not covered

- **A GPU run of musubi's real training loop:**
  - accelerate with fp16, bitsandbytes adamw8bit, and the real block-swap offloader on a 14B WanModel;
  - peak memory, the `memory mapping failed with OOM` count, and seconds per step with the extra base pass.
  - That is the box smoke test (`SMOKE=1 bash finetune/wan/train_pudgy_v10.sh`).
  - That musubi calls `on_post_optimizer_step` and `extra_step_logs` inside its real loop was checked by reading
    the source only (trainer_base.py).
- **The real golden LoRA file:** whether all its modules match the high-noise DiT (expected 400/400). The smoke test
  checks this through `--strict-base-weights`.
- **The box's HF cache:** these tests download the tokenizer into their own cache. On the box, `cache_mark_v10.py mark`
  must use the HF cache the T5 caching step used.
- **End-to-end resume:** only the choice of state dir and the counter restore are tested, not accelerate restoring a real
  state or the resume-step patch.
- **The real v10 training set and caches:** the box's `cache_mark_v10.py check` covers them.
