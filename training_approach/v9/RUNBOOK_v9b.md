# v9b-1 runbook: long-form scenes from the client skits

How v9b-1 is built, trained and judged, step by step, with the commands. Plan and decisions:
[`Training_Approach_v9.md`](Training_Approach_v9.md) §0.1 (D1–D10), §3.2–§3.4, §4, §5.
Audit of the source footage:
[`SKIT_AUDIT_v9b.md`](../../docs/training_reports/v9/SKIT_AUDIT_v9b.md). The log at the end
records what actually happened, with dates.

## 1. What v9b-1 is

A fresh **low-noise LoRA (rank 32)** on **`pudgy-v8-merged`**, trained on curated spans of
the 99 client skits, plus 35% v8 replay. It teaches what v8 never saw:
- two characters in one shot;
- real and flat-ground sets;
- entrances and exits;
- props in use;
- several actions in order, described beat by beat.

Each window is at most 81 frames (3.4 s). Long videos are chains of such shots (§3.4 of
the plan).

**Decisions** (Training_Approach_v9 §0.1). D1–D4 were confirmed by the user on 2026-10-26;
D5–D10 follow the plan as written.

| # | decision |
|---|---|
| D1 | base = `pudgy-v8-merged`, not v9c-phone |
| D2 | 480×848 (9:16), unless the scene-scale VAE check shows the faces breaking (then 576×1024) |
| D3 | locked-camera spans only; pans, zooms and pushes are dropped |
| D4 | leads + distinct side characters only; lookalikes and lead copies dropped |
| D5 | outfits named in the caption |
| D6 | buckets f21 · f33 · f49 · f81 |
| D7 | low-noise first; a separate high-noise run only if E1a / the travel bar call for it |
| D8 | plain MSE (no motion-weighted loss in v9b-1) |
| D9 | camera clause "fixed camera, locked-off shot, the character(s) stay(s) the same size" on scene clips; v8 wording on replay |
| D10 | no v8 action clause in scene captions; v8 replay anchors every label at ≥ 1 clip per 4 scene windows using its words |

## 2. Pipeline

```
raw skits (Data/raw/iteration_1, _2)
  │  scenes/curate_tools.py   profile / strip / compare  (signals + frame-numbered pictures)
  ▼
curator pass (12 parallel curators, brief scenes/CURATION_v9b.md)
  │  scenes/curation/<skit>.json: spans tiling every frame, keep/drop + reason, beats with frame ranges
  │  scenes/curate_check.py   enforces D3–D6, D10 and the schema
  ▼
verification: a second look at a random 10% of kept spans (+ every flagged one)
  ▼
scenes/choose_holdout.py → scenes/holdout_v9b.json (6 whole skits, never trained)
  ▼
prep_v9b.py   windows (f81/49/33/21, non-overlapping, lead visible ≥ half)
  │           captions compiled by scenes/captions_v9b.py from beats + registry, then linted
  │           35% v8 replay (prep_v9 builders) with D10 label anchors
  ▼
v9b_box.sh cache → smoke (heaviest bucket) → train (+ Azure monitor)
  ▼
eval_v9b/: tier1_v9b.py (regression + scene sets) → gate_v9b.py → suite_v9b.py (Tier 2)
```

## 3. Curation

- **Tools** (`finetune/wan/scenes/curate_tools.py`):
  - `profile` gives hard-cut candidates (frame-diff spikes), gradual-change candidates
    (dissolves, fades, camera moves) and a camera hint, *keep12*: the share of edges still
    in place 12 frames later. It was calibrated on known shots: locked shots have a median
    ≥ 0.82, and the Economic pan scores 0.52. Corner-tracking camera estimates were tried
    first and dropped, because the corners land on the moving characters.
  - `strip` renders numbered frame grids.
  - `compare` overlays two frames' edges: locked background edges coincide.
- **Schema and rules**: `scenes/CURATION_v9b.md`.
  - Spans tile every frame of a skit, so nothing is skipped silently.
  - Every drop has a reason.
  - Kept spans carry who is on screen, where and in what outfit, the setting and view, and
    **beats**: one action each, with frame ranges, so a window's caption can take only the
    beats it shows.
- **Validator** `scenes/curate_check.py` (also run by prep). It checks:
  - tiling;
  - kept ⇒ static camera, ≥ 21 frames and a lead present;
  - no overlays or lookalikes;
  - beats contiguous and covering the span;
  - no "blue"/"pink" outside the anchors;
  - no camera words;
  - no transcribed text;
  - no 6-word run from a v8 action clause (D10);
  - the frame count matches the decoder.
- **Pilot**: the Chair skit, curated by hand to test the tools, schema, validator and
  compiler end to end. It caught four bugs before the curators started:
  - lowercased names after "Then";
  - "characters" used for a single character;
  - stale screen positions within a span, fixed with `moves`;
  - two-action beats that a window could show only half of, fixed with the one-action-per-beat
    rule.
- **Parallel pass**: 12 curators, about 8 skits each (`scenes/curation_batches.json`,
  balanced by frame count). Each writes only its own files and must reach 0 validator errors.
- **ffmpeg**: every decode uses `-threads 4`. About 128 concurrent default-threaded ffmpeg
  6.1 decodes deadlocked on this box during v9c prep.

## 4. Holdout

`scenes/choose_holdout.py --write`: 3 skits per iteration. Eligible skits have:
- at least 2 kept spans of ≥ 33 frames;
- at least 1 kept span of ≥ 49 frames;
- kept time between the 25th and 75th percentile.

Among them it picks greedily to cover a two-shot, a one-character span, a flat ground, a
full set and a prop event. Seeded and recorded.

## 5. Prep, cache, smoke

```bash
bash finetune/wan/v9b_box.sh check     # registry lint, 99 valid curation files, holdout chosen
bash finetune/wan/v9b_box.sh prep      # prep_v9b.py -> /workspace/Data/processed/v9b_<N>, linked /workspace/data_v9b
bash finetune/wan/v9b_box.sh cache     # VAE latents + T5, then verify_cache_v8.py
bash finetune/wan/v9b_box.sh smoke     # 2 steps on the heaviest bucket: peak memory and s/clip
```

- Scene windows: 1080×1920 → 480×854 (Lanczos) → centre crop 480×848, x264 crf 12, 24 fps.
- Replay: prep_v9's builders, so the clips are exactly the distribution v8 learned (1024²,
  flat grounds, v8 captions). The QF2_R angle stays out of training, because it is the
  regression set.
- The heaviest sample is a replay f57 at 1024² (≈ 61k tokens), not a scene f81 (≈ 33k).
  The smoke test runs that bucket.

## 6. Training and evaluation

```bash
bash finetune/wan/v9b_box.sh baseline  # Tier 1 on v8merged (regression rows reused from v9c after a spot check; scene set scored)
bash finetune/wan/v9b_box.sh train     # background; monitor mirrors weights + logs to Azure v9b/ every 30 min
bash finetune/wan/v9b_box.sh status
bash finetune/wan/v9b_box.sh pause 200 # waits for step 200's state, stops cleanly
bash finetune/wan/v9b_box.sh tier1 200 # Tier 1 + gate on that checkpoint, then: v9b_box.sh train (resumes)
bash finetune/wan/v9b_box.sh eval      # after the run: Tier 1 on every STEPS_EVAL, gate, Tier-2 suite
bash finetune/wan/v9b_box.sh archive
```

- **Recipe** (`train_pudgy_v9b.sh`): rank 32 / alpha 32, adamw8bit, lr 5e-5, cosine to
  10%, warmup 20, grad-accum 8, t ∈ [0, 900] (low-noise expert), blocks to swap 24, save
  every 50.
- **Steps: 400, not the plan's 800.** The curated set is ~392 scene windows + ~211 replay
  clips (~565 after the holdout), so 400 × 8 samples ≈ 5.7 epochs; 800 would be ~11, on
  footage that is only ~10 minutes long. v9c-phone plateaued after ~2 epochs. Tier 1 at
  100 / 200 / 300 / 400 picks the checkpoint.
- **Gate** (`eval_v9b/gate_v9b.py`, fixed before the data). The winner is the passing
  checkpoint with the lowest L1. A checkpoint passes when all of these hold:
  - **R1**: per-kind paired margin change, bootstrap CI lower bound ≥ −0.003, and accuracy
    down ≤ 1 clip;
  - **R2**: v8-label loss ratio ≤ 1.05 at each t;
  - **L1**: held-out scene loss < base;
  - **P2**: text-attributable gain, CI upper bound < 0.
- **Tier 2** (`eval_v9b/suite_v9b.py`): held-out span first frames, true captions, 49
  frames at 480×848, seeds 42/43/44, base vs winner. It reports:
  - duplicate-character frames;
  - border drift;
  - frozen and pop flags;
  - motion relative to the real span;
  - contact sheets with the real span on top.

## 7. If it fails

- **R1/R2 fail** (regression): re-run with more anchors (`ANCHOR_RATIO` 3) or a lower step
  count. Read the per-label Δmargin to find the absorbed clause, and fix its wording in the
  curation.
- **L1 or P2 fail** (no learning, or caption ignored): check the caption lint and the
  beat/window alignment before touching the recipe.
- **Tier 2 shows duplicates or zoom**: compare with the base first. Then check the windows
  that taught it (manifest → window → curation span).

## 8. Where everything is

| what | where |
|---|---|
| curation, tools, brief | `finetune/wan/scenes/` (synced to Azure `v9b/work/scenes` every 30 min) |
| prep, training, box script | `finetune/wan/prep_v9b.py`, `train_pudgy_v9b.sh`, `v9b_box.sh` |
| eval | `finetune/wan/eval_v9b/` → results under `/workspace/eval_v9b/runs/pudgy-v9b-lownoise/` |
| training set | `/workspace/Data/processed/v9b_<N>/` (manifest.json) → Azure `processed/` at archive |
| weights | `/workspace/wan_output/pudgy-v9b-lownoise/` → Azure `v9b/weights/` (monitor) |
| logs | `/workspace/logs_v9b/` → Azure `v9b/logs/box` |

## 9. Log

- **2026-10-26.**
  - The user confirmed D1–D4 and asked to start the curator pass, then training once the
    v9c2 GPU experiments finish (ETA ~03:45 UTC 27 Oct).
  - Built the tools, schema, validator and caption compiler, and piloted them on Chair (4
    bugs fixed).
  - Launched 12 curators.
  - Wrote prep_v9b, the training and box scripts, and the eval package.
  - Started the scene-scale VAE check on CPU.
- **2026-10-26, curation findings.**
  - **The camera hint misses slow moves.** Curators 01, 02, 08, 09 and 11 found slow push-ins,
    zooms and pans that keep12 marked "static?". Examples: Burrito is one 14% zoom across the
    whole skit; EatingStages is +9% over 240 frames; 4amshow and BringWater are push-ins.
  - **The pilot example was wrong**, and curator-01 found it. Chair [0,86) is a +4% push-in.
    It is now dropped, and the running curators were told to check every kept span's first
    vs last frame.
  - **New check, `scenes/span_motion_check.py`.** It fits a similarity transform between a
    span's first and last frames (and first and middle) from ORB features on the background
    only, with the leads masked. Calibration: locked shots measure exactly 1.000 and 0 px;
    every known move measures 1.02–1.10 or a shift. Its flags go to the by-eye verification
    pass. Props, bellies and outfits can still carry features, so a flag is not a verdict.
  - **Rulings that apply to every curator** (all prior decisions already conform, except
    BestWeekend [0,85), flipped to dropped):
    - **Lead depictions:** any recognisable depiction of Pax or Polly ⇒ `lead_copy`, whatever
      its size. That covers photos, posters, face mugs and headbands, plushes, charms and
      figures in a lead's look, and screens showing a lead. A tiny generic penguin charm
      (GroundhogDay) is a prop.
    - **Lighting:** a steady night or dim-room lighting is kept, with `time` set. Whole-picture
      effects (vignettes, colour washes, flashes, time-lapses, fly-ins) are dropped.
    - **Backgrounds behind a locked car:** a background that visibly scrolls past (Savings) is a
      camera move, so dropped; a few pixels of scenery drift (LifeWithHim) is kept.
  - **Open question for the client.** In several skits the iFin carries a lead charm in place
    of its tuft (Maps, Vibing, PMActivities). Those spans are dropped as `lead_copy`, and the
    registry phrase ("a little tuft on top") wouldn't describe them anyway. Is the charm part
    of the iFin's design?
  - **Verification, pass 1** (3 independent verifiers, 77 kept spans: the 52 flagged by
    `span_motion_check` plus a seeded random 25). Result: 58 ok, 14 fixed, 5 dropped, plus 4
    more drops from the depiction ruling below.
    - **Camera moves inside a span**, which first-vs-last comparisons miss:
      - an impact shake at f58–61 (ActingTough);
      - a drift starting 4 frames before the cut (Germs);
      - an eased push-in that snaps back (WakingUp);
      - a snap push-in on a plain ground (Dust).

      So `span_motion_check.py` now also fits the first frame against every 2nd frame. That
      catches all four, while the known locked shots stay clean.
    - **Depictions:**
      - a Pax-style penguin printed on a shopping bag, in the pilot Chair skit again;
      - jersey crests drawn as the leads (Tone, Champ), which the depiction rule drops;
      - a Polly keychain charm by an ignition (TirePressure).
    - **Beats:** eyes said closed when open, still poses during a sway, a missed second
      jump, misnamed props, and a missed `size_change`.
    - **About 1 span in 4 needed a change**, so pass 2 covers **every remaining kept span**
      (170): 6 verifiers, each with the per-frame motion hints. Fixes are applied with
      `scenes/apply_verify.py`, then the validator is run again.
    - **Validator hole:** "grey-blue" got past the colour check, because hyphenated words
      weren't split. Prep's caption lint caught it; it is fixed in `curate_check.py` (two
      setting descriptions).
  - **Verification complete.**
    - Every kept span was seen by at least one independent verifier: pass 1 (77), pass 2 (170),
      and a pass 3 (32) for spans that pass 2's set builder had skipped by mistake. The builder
      matched spans by start *or* end frame, so a span ending where a pass-1 span began was
      skipped; pass 3 matched by overlap.
    - Verdicts over all passes: 173 ok, 94 fixed, 12 dropped. The rulings dropped 5 more
      (jersey crests, face mugs).
    - Most fixes were beats that held an expression or pose after it changed: eyes closed
      for 2 frames written across a 20-frame beat.
    - Other depictions found: a sticker on a bottle (Drinker), playing cards (Disaster),
      phone charms (Follow, PMActivities) and a phone screen showing lead thumbnails (BF).
    - Camera: motion inside a span in 5 spans (WakingUp, Dust, ActingTough, Germs,
      LifeWithHim's scrolling scenery). Of the motion check's flags, every other one was
      bellies, props, clouds or steam.
    - Final: 99 files valid, **266 kept spans, 9.7 min**.
  - **Holdout** (`scenes/choose_holdout.py --write`, seed 9):
    - iteration_1: BackToBed, TirePressure, Drinker;
    - iteration_2: BackTogether, MorningFart, BubbleWrap.

    926 of 14,015 kept frames (7%). Together they cover two-shots, one-character spans,
    flat and full sets, and prop events.
  - **Eval set** (`eval_v9b/build_eval_v9b.py`):
    - 15 held-out Tier-1 windows (f33) with 37 distractor captions (15 beat order, 7 leads
      swapped, 15 another window's action). n = 15 is small, so P2's CI will be wide.
    - 8 Tier-2 jobs (f49).
  - **Training set built: `processed/v9b_535`** (`v9b_box.sh prep`, 22:42 UTC).
    - 348 scene windows from 78 skits (8.7 min, 158 two-shots): 160 at f21, 72 at f33, 82 at f49,
      34 at f81.
    - 187 v8 replay clips (35%), with the D10 anchors met (e.g. happy-words in 98 windows → ≥ 25
      happy replay clips).
    - Checked, all clean:
      - all 535 clips match their block's size, frame count and 24 fps;
      - cache directories are unique;
      - no window comes from a held-out skit;
      - no unresolved `{ref}`;
      - the longest caption is 237 words (≈ 310 T5 tokens, under the 512 limit).
    - `longform/dupcheck.py` on all 348 scene windows flagged 62. All 62 were inspected by eye
      and all are false positives: blue or pink props, sinks, igloos and water, or a lead's flipper
      counted as a second body. MessUp's mirror shows only droplets, no reflection. No duplicate
      leads remain.
    - Backed up to Azure `processed/v9b_535`; the eval set is at `v9b/eval`.
  - **GPU timing.** The v9c2 queue slipped to ~13:20 UTC 27 Oct (~08:05 if the user cuts its
    tail). v9b training starts when that session reports the GPU free.
  - **Phase 0 #4, the scene-scale VAE check (D2 decided: 480×848).**
    - Method (`eval_v9b/phase0_vae_v9b.py`, CPU): VAE round trip of 17 frames from 8 random kept skit shots
      at each resolution, plus 3 v8 training clips at 1024² as the reference.
    - PSNR inside the lead's bounding box:
      - 480×848: mean 32.8 dB (min 26.1);
      - 576×1024: mean 33.8 dB (min 26.9);
      - v8 1024² reference: mean 34.9 dB (min 30.1).
    - By eye (`/workspace/eval_v9b/phase0_vae/vae_check_sheet.png`): at 480×848 eyes, beaks, tufts, outlines
      and blush marks survive in every shot, including the smallest lead (~170 px) and the lowest-PSNR
      shot (Dust, whose low score comes from dust texture, not the face).
    - 576×1024 gains ~1 dB that isn't visible, at ≥ 1.4× the tokens. So: 480×848.
  - **The iFin phrase (open, user decision).** 29 of 373 scene windows name the iFin, and
    the registry phrase says "with a black screen". Several skit shots show the screen lit
    (an alarm, a call), so the phrase is false there. The v9c session recommends an
    identity-only phrase: "the iFin phone, a small pale grey phone with three camera lenses
    on the back and a little tuft on top". Screen state and viewpoint would go in per-shot
    clauses.
    - The registry stays frozen until the v9c2 GPU queue ends, because its running jobs read
      the phrase live.
    - v9b is built with the current phrase. If the user approves the change before the T5
      cache step, the captions are rebuilt from the registry first.
  - **Process slips.**
    - All curators shared one scratchpad, so helper scripts overwrote each other. Each curator
      now uses its own subfolder. No curation file was affected: the scripts that ran by
      mistake errored out.
    - My own `pkill -f <pattern>` killed the shell that ran it, which the v9c session had
      warned about. Kill by PID from now on.
- **27 Oct, 05:40 UTC: the phone phrase approved; hand-off automated.**
  - **Decisions (user, relayed by the v9c session):**
    - the identity-only phrase: "the iFin phone, a small pale grey phone with three camera lenses on the back and a little tuft on top";
    - the v9c2 screen tier is cut after E1b; V2, C0b and E3 are skipped. The GPU frees at ~07:50 UTC.
  - **Evidence for the phrase** (v9c2 screen tier: 41 frames, standing FRONT, 3 seeds × 2 IDs):
    - control: 5/6 clips show the screen-flip defect;
    - no screen words in the phrase or the texting clause (E2c): 0/6;
    - negative prompt (E2a): 3/6;
    - guidance 3.5 (E2b): 5/6.
  - **Effect on v9b:** 28 of the 348 training windows contain the phrase (the earlier 29-of-373 count included held-out windows).
    - The eval set contains no iFin phrase, so it is unchanged.
    - Replay captions never name the phone.
    - `captions_v9b.py` reads only the phone's `phrase` and `name`, not the texting `actions` clauses the v9c session may also edit.
  - **Hand-off runner:** `/workspace/logs_v9b/handoff_go.sh`, PID in `handoff_go.pid`, stage lines in `handoff.log`.
    1. Waits for both:
       - the v9c2 stop script's "screen tier (41 f) done" line, with the GPU idle;
       - the new phrase in `registry.toml`, with the file unchanged for 2 min.
    2. `check`: registry lint and curation.
    3. Moves the old set to `/workspace/Data/aside_v9b/v9b_535_blackscreen`. That is outside `processed/`, so the `v9b_[0-9]*` glob in the prep step can't pick it up. `prep.log` is kept as `prep_blackscreen.log`.
    4. Runs `prep` again: the same code path, a deterministic rebuild.
    5. `handoff_verify.py` requires all of these:
       - the registry differs only in `phone.phrase` and `phone.actions`;
       - the same 348 windows;
       - exactly 28 captions changed, each equal to `old.replace(old_phrase, new_phrase)`;
       - identical replay rows and dataset configs;
       - the same jsonl rows apart from that replacement;
       - clips byte-identical, or else re-probed for size, frame count and fps.
       It was tested on the current set: comparing the set with itself fails on every phrase check, and a simulated correct rebuild passes.
    6. Pushes the new set to Azure `processed/v9b_535`.
    7. `cache`, `evalprep`, `smoke`. The smoke must pass with peak memory < 78 GB and no NaN or inf loss.
    8. `baseline`. If it fails, training still starts; the eval is fixed separately.
    9. `train`.
    It stops at the first failure.
  - **Check caught while writing the verifier:** skit frames are portrait (1080 wide), so scene clips are 480 wide × 848 tall. The `(width, height)` check had them backwards; fixed and confirmed with ffprobe (480×848 scene, 1024×1024 replay).
- **27 Oct, 06:00–06:30 UTC: v9b-1 moves to a new GPU machine (user decision).** The old box keeps its GPU for the phone track (v9c2).
  - **Old-box runner:** `handoff_go.sh` (PID 2298960) was stopped at 05:59:32 UTC before it reached any GPU step. The v9c session confirmed it.
  - **The set was rebuilt on CPU with the approved registry:**
    - both approved edits: the phrase, plus "the screen turned toward itself" → "holding it with its back toward the viewer" in the two phone texting actions;
    - `registry.toml` stayed untouched while the v9c2 jobs read it live; prep ran on a copy, `/workspace/logs_v9b/registry_new_phrase.toml` (md5 `3757257814f5588603317d747f2e250e`), through `/workspace/logs_v9b/prep_with_registry.py`, which points `registry.load()` at the copy;
    - the v9c session's write, armed for ~07:50 UTC, was pre-checked byte-identical to the copy (same md5);
    - a first rebuild with the phrase only was discarded, because the manifest records the registry's `actions`.
  - **`handoff_verify.py`: OK** (`/workspace/logs_v9b/handoff_verify.log`).
    - registry: the phrase and the texting clauses changed, nothing else;
    - 348 windows, of which 28 captions changed, each only by the phrase;
    - 1,070 jsonl rows checked;
    - 535/535 clips byte-identical.
    - The old set is at `/workspace/Data/aside_v9b/v9b_535_blackscreen` and `prep_blackscreen.log`.
  - **Azure:**
    - `processed/v9b_535` was re-pushed (566/566 size-checked; 9 changed files uploaded);
    - `v9b/base/pudgy-v8-merged.safetensors` (md5 `ec333748284410bb341f431b88f5026f`; `v9c/base/` was empty);
    - `v9b/handover/`: the code bundle, `eval_for_v9b.tar` (the regression set with its latent cache, the v9c base Tier-1 rows, the v9b eval set), `SHA256SUMS`.
  - **Handover doc:** [`HANDOVER_v9b.md`](HANDOVER_v9b.md), the single entry point for the new machine.
  - **Caught while writing it:** the old runner's smoke check required peak memory < 78,000 MiB, but v9c's clean smoke peaked at 80,420 MiB (the allocator reserves nearly the whole card). That would have failed a healthy run.
    - The correct signal is the allocator warning `expandable_segments: memory mapping failed with OOM`. v9c: 0 at swap 24; at swap 22, warnings at 81.2 GB; 1 at step 120 of the training run, survived.
    - The handover gates on that warning instead.
  - **Azure is slow to download from (06:15–06:25 UTC).**
    - Measured from the old box: 0.2 MB/s on one stream, ~0.4 MB/s with azure_sync's parallel streams. The 37 MB code bundle took 4 min.
    - Hugging Face ran at 73 MB/s on the same machine, so the link is fine; Azure→box transfer is the bottleneck. Uploads were fast.
    - The end-to-end Azure download test was stopped for this reason.
    - Instead, a single archive, `/workspace/handover_v9b/v9b_handover_all.tar` (code, eval tar, merged base, golden LoRAs, set, `SHA256SUMS`), is the primary route to the new machine, by rsync over SSH or the old box's Jupyter. It was test-extracted on the old box: all 571 checksums OK. Azure is the fallback.
  - **GitHub (27 Oct, user request):** the work is committed to branch `v9b-handover` (`main` was the checked-out default, so per policy a branch was cut) and pushed to `rlrahulkanojia/Pudgy`.
    - Three commits: the phone track (v9c report, v9c2 research and tooling, the approved `registry.toml`), then v9b-1 (curation, prep, captions, eval, box scripts, runbook, handover), then the `azure_sync --force` fix.
    - The new machine clones that branch for code and docs (HANDOVER_v9b §4 Step 1). Data comes from the handover archive or Azure.
    - The registry was committed with the approved content (md5 `3757…250e`) without touching the working file: the v9c2 jobs still read the old one until their write at ~07:50, which makes the working tree match the commit.
  - **Azure upload trap found and fixed (06:45 UTC).**
    - `azure_sync.py push` skips any blob whose size already matches. `SHA256SUMS` and `v9b_handover_all.tar.sha256` (same length after a hash changed) had gone stale on Azure.
    - Found with `finetune/wan/azure_verify_md5.py`, which compares Content-MD5. Both files were re-uploaded and now match.
    - Every other handover blob matches by content: all 566 set files, the code bundle and the doc. The eval tar and the merged base have no stored MD5; each was uploaded once and is in `SHA256SUMS`.
    - `azure_sync.py push` gained an opt-in `--force`; the default behaviour is unchanged.
  - **Merged into `main` (user request, 27 Oct).** `origin/main` was still `ab559fb`, so `main` was fast-forwarded to `cdc8dda`; `v9b-handover` points at the same commit. The handover now clones `main`.
