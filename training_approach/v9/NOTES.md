# v9 notes: dataset and the numbers behind it

Companion to the [v9 README](README.md) (how to prompt), the plan
[`Training_Approach_v9.md`](Training_Approach_v9.md), and
[`Assets_v9.md`](Assets_v9.md). Run results:
[`REPORT_v9c_phone.md`](../../docs/training_reports/v9/REPORT_v9c_phone.md).

## 1. The v9c-phone dataset

**`processed/v9c_phone_475`**: 475 clips at 1024×1024 and 24 fps, silent, with the character
placed on a flat colour background. Built by `finetune/wan/prep_v9.py` from client deliveries 3–6
(iteration_6 adds the phone).

| | Sources | Clips | Labels |
|---|---|---|---|
| Phone (new) | 32 | 380 | texting, sitting · texting, standing (Pax + Polly) |
| Replay (v8) | 81 | 95 (20%) | 26 (character, label) groups across all v8 movements and expressions |
| Held out | 32 (4 phone) | — | the `QF2_R` angle |

- **Phone clips:** each source is used on all 4 backgrounds (white, pastel blue, peach, mint)
  at 21, 41 and 61 frames (0.9 / 1.7 / 2.5 s).
- **Replay:** spread evenly across character × label so no v8 label disappears. It fills the
  shorter lengths too (13–57 frames).
- **Length buckets:** f13 15 · f17 3 · f21 168 · f25 8 · f29 20 · f33 20 · f37 19 · f41 94 ·
  f45 12 · f49 14 · f57 10 · f61 92.
- **Captions are compiled from the asset registry** (`finetune/wan/assets/registry.toml`), so
  the phone phrase is byte-identical in every clip and at inference.

### Issues found in the data

| Issue | Effect | What we did |
|---|---|---|
| **230 iteration_6 files are re-sends** of iterations 3–5 (same MD5) | Would double-count old clips | Detected by MD5 and skipped. No genuinely new lines beyond the phone |
| **18 older `confused` sources superseded** by `CONFUSED/V2` | Two versions of the same label | Only V2 used |
| **21 two-character interaction clips** (2 needed filename overrides) | Need the v9b caption design | Not used in this stage |
| **One replay source missing its first frame** (`PAX_MOTION_WALKING_QF1_R.mov`) | Can't cut even the shortest (f13) clip | Dropped from replay |
| **11 Polly phone clips + 2 old files** need re-export (Assets_v9 §9) | Usable windows only | Windows used; re-export requested, not blocking |

## 2. The numbers behind the v9c choices, and what the run measured

| # | What we measured | What it meant | What v9c does / should do |
|---|---|---|---|
| 1 | v7fixed + v8@550 combined by rank concatenation (rank 80): weight error **≤ 1.5×10⁻⁷**. The eval check read 1.5×10⁻³ (bar 1×10⁻³), but its Tier-1 tables match v8@550 to 4 d.p. | The merge is exact; the eval gap is fp16 noise | Train on the single merged base. The check's bar is too tight for fp16 |
| 2 | Before training, the base scored the phone phrase **0/4** against "a phone / a small red book / a white cup / a black TV remote control" (mean rank 3.75 of 5) | The name means nothing to the base model | The binding check measures whether training teaches the name |
| 3 | After 200 steps: held-out phone loss **0.613×** the base; binding still **0/4** (1/4 at steps 125–175) | It fits the phone footage but doesn't learn the name as a concept | The look must come from reference pixels (Assets_v9 §2) |
| 4 | The v9c weight change is **4.7%** the size of the v8 base add-on | A small nudge | It shows in loss, not in video |
| 5 | Generated clips, base vs winner, same seed: whole-frame difference ~1/255, but **11–25/255 inside the moving phone/flipper region**; defect verdict identical on all 8 | v9c redraws the phone area but doesn't fix the defect | A loss gate alone can't keep a prop stage. Visual bar ID1 added (validated: 0/380 false alarms) |
| 6 | The phone **turns round to show its screen** in 5/8 clips in **both** models: FRONT 4/4, QF2_R 1/4, onsets at frames 2–27; sometimes drawn with a call screen or "iFi…" lettering | The words "phone … black screen" pull a camera-facing phone to its screen side. Whether it is set before the high-noise → low-noise switch is unverified | Inference fixes first (negative prompt, CFG, phrase, FLF), then the asset embedding ([RESEARCH_v9c_fixes](RESEARCH_v9c_fixes.md) §2) |
| 7 | Held-out **motion** accuracy 0.75 → 0.58 (margin CI [−0.012, −0.003]); the **sitting caption** moved closest to other motions' true captions (−0.0078); regression uncorrelated with motion energy (ρ = −0.01) | The phone-sitting caption reuses the v8 sitting wording, 190 : 7 | Anchor the sitting wording, or use the embedding route (v8 prompts untouched) |
| 8 | Phone loss: step 125 → 200 changes only 0.00073 → 0.00070 | It plateaued early | ~125 steps is enough. Saves ~10 h of a 27 h run |
| 9 | Speed on one A100 80GB: training **487 s per step** (8 clips); Tier 1 **~90 min per config**; generation **~16.5 min per 41-frame 1024² clip** (25 steps × 35 s) | The runbook's eval estimates were 2–3× low | Budget eval at these rates |
| 10 | `prep_v9 --workers $(nproc)` on 128 cores: 128 parallel ffmpeg 6.1 decodes **deadlocked** | Prep sat for 3 h | Capped at 16 workers (`PREP_WORKERS`); prep takes ~1 min |

**Final (26 Oct):** training finished at step 200 after 27 h. All of steps 125–200 pass the
gate, and the winner is **step 200** (lowest held-out phone loss, 0.613×). In generated clips the
phone turns round to show its screen in 5/8 clips, the same as v8merged. `textcall`: the name draws a
generic screen-out smartphone. `leak`: no phone in v8 prompts. Full report:
[`REPORT_v9c_phone.md`](../../docs/training_reports/v9/REPORT_v9c_phone.md).

## 3. v9c2 (2026-10-27 → 30): what changed and the numbers behind it

| # | Measured | Meant | Done |
|---|---|---|---|
| 1 | Removing the screen words from the phrase and clause: flips **12/24 → 1/24** (11 fixed, 0 broken, exact McNemar p = 9.8×10⁻⁴), on both v9c@200 and v8merged | The words caused the turn | Registry wording changed 2026-10-27 |
| 2 | Decoding the model's clean-video estimate at the expert switch matched the final clip **12/12** | The turn is set in the high-noise phase, which v9c never trained | Phone training now goes through both experts |
| 3 | **`processed/v9c_phone_475`, recaptioned → `data_v9c2`**: 380 phone captions from the new registry (each checked against the old one plus the two text changes); clips and latents unchanged | Training must use the inference wording | `recaption_v9.py` |
| 4 | Text cache rebuilt on CPU: emulated-bf16 T5 drifted 3–7%; **fp32** matches a fresh encode to 1.7×10⁻³. The old GPU-bf16 cache had per-token errors up to 3.8× | — | `te_cache_cpu_fp32_v9.py`; T5 precision at inference tested: fp32 = fp8 for the turn (13/24 vs 12/24) |
| 5 | **E5** phone embedding (25×4096, both experts, 150 steps, 264 phone clips): flips **0/24**, held-out phone loss **×0.899**, name binding **3/4** held-out (v9c 0/4), steadier texting | It generalises, ties the name to the phone, and can't touch v8 prompts | **Adopted** |
| 6 | Pivotal LoRA on frozen E5 (cross-attn k/v, preservation loss): gate PASS at both noise levels, flips 0/24, generation ≈ E5 | No visible gain for extra complexity | Kept on Azure, unused |
| 7 | Musubi trains on every latent in a cache directory, whatever the jsonl lists | Replay clips in E5's first smoke run gave an embedding-free step | Phone-only cache dirs per group |

Full story, corrections and the introspection: [`RESEARCH_v9c_fixes.md`](RESEARCH_v9c_fixes.md),
[`RESEARCH_asset_memory.md`](RESEARCH_asset_memory.md).
