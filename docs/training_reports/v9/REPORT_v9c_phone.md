# v9c-phone — the first prop stage (the iFin phone): run report

**Run** 2026-10-25 02:50 → 2026-10-26 06:03 UTC · **Hardware** 1× A100-SXM4 80GB (Vast.ai, not
persistent) · **Status** training complete, Tier 1 + gate complete, Tier 2 complete (`phone`,
`textcall`, `leak`). Updated 2026-10-26 19:30 UTC; §3.3 corrected after a full-resolution review
(see [`RESEARCH_v9c_fixes.md`](../../../training_approach/v9/RESEARCH_v9c_fixes.md), corrections C1–C3).
Plan: [`Training_Approach_v9.md`](../../../training_approach/v9/Training_Approach_v9.md) ·
assets: [`Assets_v9.md`](../../../training_approach/v9/Assets_v9.md) ·
runbook: [`RUNBOOK_v9c_phone.md`](../../../training_approach/v9/RUNBOOK_v9c_phone.md) ·
plain-language: [`README.md`](../../../training_approach/v9/README.md),
[`NOTES.md`](../../../training_approach/v9/NOTES.md) · previous run: [`REPORT_v8.md`](../v8/REPORT_v8.md).

---

## 1. Headline

| | |
|---|---|
| **Winner** | `pudgy-v9c-phone-lownoise-step00000200.safetensors` (rank 16 LoRA, low-noise expert) |
| **Load it as** | `--lora_weight pudgy-v8-merged.safetensors <winner> --lora_multiplier 1 1`; high-noise: `lora_highnoise_GOLDEN_ep40` × 1 |
| **Gate** | all four evaluated checkpoints (125–200) **PASS**; the winner has the lowest held-out phone loss |
| **Learned (held-out phone loss)** | **0.613×** v8merged (loss 0.00116 → 0.00070 on the 4 QF2_R phone clips) |
| **Regression (held-out v8 labels)** | 1.032× v8merged loss (bar ≤ 1.05×) — but held-out **motion accuracy 0.75 → 0.58** (§3.1) |
| **Name binding** | **0/4** (base 0/4; steps 125–175 1/4). The phrase does not out-score generic objects |
| **Generated clips** | the phone **turns round to show its screen** in **5/8** clips for **both** models (FRONT 4/4, QF2_R 1/4); v9c changes the phone/flipper region (11–25/255) but not the outcome (§3.3) |

**In one line:** the stage passed its gate and lowered phone loss by ~39%, but it learned the
phone clips without using the name, and it doesn't fix the visible defect. Whenever the phone
faces the camera, it turns round to show its screen in both models.

---

## 2. What was trained

| | |
|---|---|
| Base | Wan2.2-I2V-A14B; LoRA on the **low-noise expert** only |
| Start | `--base_weights pudgy-v8-merged` (v7fixed + v8@550 combined exactly by rank concatenation, rank 80; weight error ≤ 1.5×10⁻⁷) |
| LoRA | rank 16 / alpha 16, adamw8bit, lr 5e-5, cosine → 0.1, warmup 20, grad-accum 8 |
| Steps | 200 optimiser steps (8 clips each), saved every 25 |
| Data | `processed/v9c_phone_475`: **380 phone clips** (32 sources × 4 backgrounds × f21/f41/f61) + **95 replay clips** (20%, 81 sources over 26 character × label groups) |
| Holdout | the QF2_R angle (32 sources, 4 of them phone) |
| Caption | the registry phrase, verbatim: "the iFin phone, a small pale grey phone with a black screen, three camera lenses on the back and a little tuft on top" |
| Memory / speed | `BLOCKS_TO_SWAP=24`: peak 80.4 GB, 64 s/it in the smoke test; **487 s per optimiser step** in the run (~61 s per clip) |
| Wall clock | prep ~1 min (after the fix in §4) · cache · baseline 3 h · **training 27 h 03 min** · Tier 1 6 h · phone suite 4 h 26 min |
| Loss | 0.00211 (start) → 0.00128 (step 79) → 0.00113 (step 108) → **0.00109** (step 200) |

---

## 3. Results

### 3.1 Tier 1 — diffusion classifier (k = 2, paired noise, same clips as the baseline)

Held-out = the QF2_R angle; *seen* = training angles. Accuracy / mean rank of the true label.

| config | holdout expr (16) | holdout motion (12) | holdout prop (4) | seen expr (16) | seen motion (11) | seen prop (4) | holdout prop loss |
|---|---|---|---|---|---|---|---|
| v8merged (base) | 0.812 / 1.31 | **0.750 / 1.42** | 0.00 / 3.75 | 1.000 / 1.00 | **0.909 / 1.09** | 0.25 / 2.50 | 0.00116 |
| v9c@125 | 0.875 / 1.19 | 0.583 / 1.83 | 0.25 / 2.50 | 0.938 / 1.06 | 0.818 / 1.36 | 0.50 / 2.75 | 0.00073 |
| v9c@150 | 0.875 / 1.12 | 0.583 / 1.83 | 0.25 / 2.50 | 0.938 / 1.06 | 0.818 / 1.36 | 0.50 / 2.50 | 0.00072 |
| v9c@175 | 0.875 / 1.12 | 0.583 / 1.83 | 0.25 / 1.75 | 1.000 / 1.00 | 0.818 / 1.36 | 0.50 / 2.00 | 0.00071 |
| **v9c@200** | 0.875 / 1.12 | 0.583 / 1.83 | 0.00 / 2.75 | 1.000 / 1.00 | 0.818 / 1.36 | 0.50 / 2.25 | **0.00070** |

- **Phone loss fell 39–40% at every checkpoint**, almost all of it by step 125. Later steps
  add very little (0.00073 → 0.00070).
- **Expressions held or improved** (holdout 0.81 → 0.88).
- **Motion recognition dropped** on both splits (holdout 0.75 → 0.58, seen 0.91 → 0.82) and
  stayed flat from step 125 onwards. The motion *loss* barely moved (0.00240 → 0.00249, which is why
  the 1.05× regression bar passes), but the margin between the true motion and the others
  shrank from +0.0107 to +0.0033. The 20% replay did not fully protect motion. n = 12, so
  this is 2 clips, but it is in the same direction on both splits.
- The **binding** row (holdout prop accuracy) is the phrase scored against "a phone", "a small
  red book", "a white cup" and "a black TV remote control". With n = 4 it moves by one clip
  between checkpoints. No checkpoint makes the name win reliably.

### 3.2 Gate (`gate_v9.py`, rule fixed before the data)

```
config          regress   phone  bind bind_b v8 acc  verdict
v9c@125           1.036   0.632  0.25   0.00   0.75  PASS
v9c@150           1.033   0.626  0.25   0.00   0.75  PASS
v9c@175           1.033   0.619  0.25   0.00   0.75  PASS
v9c@200           1.032   0.613  0.00   0.00   0.75  PASS
winner: v9c@200
```

`regress` = held-out v8-label loss ÷ v8merged (bar ≤ 1.05) · `phone` = held-out phone loss ÷
v8merged (bar < 1.00) · `bind` = binding accuracy (reported, breaks ties) · `v8 acc` = held-out
v8-label accuracy (base 0.786). Only steps 125–200 were evaluated (`STEPS_EVAL`), to save ~3 h
of Tier 1.

### 3.3 Tier 2 — `phone` suite (8 clips per config, seed 42, 41 frames, 1024², 25 steps)

Start frame = frame 0 of a phone clip (FRONT = trained angle, QF2_R = held out), both
characters, sitting and standing; prompt = the training caption.

| clip | p_phone base / v9c | jump base / v9c | frozen base / v9c |
|---|---|---|---|
| pax sitting FRONT | 0.989 / 0.986 | 7.28 / 7.64 | 0.05 / 0.08 |
| pax sitting QF2_R | 0.986 / 0.986 | 3.37 / 3.37 | 0.18 / 0.15 |
| pax standing FRONT | 0.982 / 0.987 | 2.48 / 2.25 | 0.08 / 0.05 |
| pax standing QF2_R | 0.985 / 0.989 | 7.24 / 6.81 | 0.00 / 0.00 |
| polly sitting FRONT | 0.993 / 0.994 | 6.29 / 5.30 | 0.03 / 0.03 |
| polly sitting QF2_R | 0.991 / 0.991 | 2.10 / 2.93 | 0.10 / 0.08 |
| polly standing FRONT | 0.994 / 0.994 | 5.36 / 5.24 | 0.28 / **0.40** |
| polly standing QF2_R | 0.994 / 0.993 | **9.66** / **8.66** | 0.00 / 0.00 |

POP flag (jump > 8): 1 / 1 · FROZEN flag (> 0.35): 0 / 1.

**The contact sheets** (base row above winner row for each clip: [`phone_pairs_a.png`](phone_pairs_a.png)
Pax, [`phone_pairs_b.png`](phone_pairs_b.png) Polly):

- **Same defects, different details.** Over the whole frame the mean difference is ~1/255, but
  inside the moving region (phone + flippers, 2.5–9% of the frame) it is **11–25/255**. v9c redraws
  the phone area, but every clip keeps the same defect verdict.
- **The phone turns round to show its screen** (reviewed at full resolution, then confirmed by the
  ID1 detector, `eval_v9/phone_id_v9.py`, which agrees on 16/16):
  - *Pax sitting FRONT:* turns through edge-on (an earlier low-resolution read called this a
    "pen"), then screen out with the tuft lost; onset frame 13;
  - *Pax standing FRONT:* screen out from frame 2, tuft kept (the same object, turned);
  - *Polly sitting FRONT:* turns at frame 27; the screen shows a call UI with green/red buttons;
  - *Polly standing FRONT:* screen out from frame 5, with a photo on it;
  - *Pax standing QF2_R:* turns at frame 25; call UI with "iFi…" lettering;
  - the other three QF2_R clips keep the phone's back to the viewer.
- **FRONT fails 4/4**, and those start frames come from a *trained* angle. In training, the screen
  always faces the character. Call screens and the name drawn as lettering point at the text
  ("phone … with a black screen") being rendered literally.
- Character identity, outlines and colours are stable in both. Eyes closing is the only
  expression change.

**Why v9c doesn't fix it** (what is known, and what is still a hypothesis):
1. **It learned the clips, not the name** (verified): 99.6% of the phone-loss gain is
   caption-independent, and its text sensitivity on phone clips fell to ×0.30–0.68.
2. **The text pulls toward a screen-out phone** (strong evidence): drawn call screens and lettering,
   and failure concentrated on the frontal view.
3. **Is the turn set before v9c acts?** (hypothesis) Steps 0–8 of 25 run on the high-noise expert,
   which v9c doesn't touch. But at the switch the SNR is only ≈ 0.012, so a lot is still open.
   Experiment E1 measures it with an x₀ probe at the switch.

### 3.4 Tier 2 — `textcall` and `leak`

- **`textcall`** (neutral FRONT start, no phone; texting prompt): both models draw a phone
  (CLIP 0.98), but it is a **generic smartphone with its screen to the viewer**, popping in at frame 1
  (POP 2/2 in both). The name doesn't summon the iFin phone.
- **`leak`** (v8 prompts): no phone appears in either model (CLIP phone score 0.25 base, 0.23 v9c).
  The stage didn't leak the phone into other prompts.

---

## 4. Defects found and fixed during the run

| defect | effect | fix |
|---|---|---|
| `prep_v9.py --workers $(nproc)` on a 128-core box | 128 concurrent ffmpeg 6.1 decodes deadlocked in the replay preflight (futex waits); prep sat for **3 h** | Capped with `PREP_WORKERS` (default 16) in `v9_box.sh`. Prep then took ~1 min |
| `pkill -f <pattern>` to clear the hung ffmpegs | the pattern matched the killing shell itself (exit 144) | `pgrep -x ffmpeg \| xargs kill -9` |
| smoke-config split in `v9_box.sh` | `AttributeError: NoneType.group`. prep_v9's header *comment* contains "[[datasets]]" | Split only on the table header at line start: `re.split(r"(?m)^\[\[datasets\]\]", text)` |
| baseline merge check reported **MISMATCH** (1.5×10⁻³ > 1×10⁻³ over 63 clips) | per the runbook, "do not train on this base" | Direct weight comparison: ≤ 1.5×10⁻⁷ relative error, so the merge is exact and the gap is fp16 eval noise. v8merged and v8@550 Tier-1 tables are identical to 4 d.p. Proceeded; flagged |
| `BLOCKS_TO_SWAP=22` | OOM-mapping warnings at 81.2 GB peak; the smoke "retr" grep doesn't catch this warning type | Used 24: same 63 s/it, 80.4 GB, no warnings |
| monitor exits on `finished` | the step-200 checkpoint and the final file were not confirmed mirrored | Ran `monitor_v8.py --once` by hand. All 8 checkpoints + final are on Azure, sizes matched |
| runbook timings | Tier 1 was "~4 h" (actual ~90 min per config); Tier-2 "4–6 min per clip" (actual **~16.5 min**: 25 steps × 35 s at 1024² + load) | Budgets corrected in this report |

`v9_box.sh` changes (the first two fixes) are uncommitted in the working tree, for review.

---

## 5. Where everything is (Azure `pudgytraining`, container `pudgy`)

| what | where |
|---|---|
| checkpoints 25–200 + final | `v9c/weights/pudgy-v9c-phone-lownoise/` (mirrored, sizes verified) |
| logs, status | `v9c/` (monitor mirror) |
| merged base, eval results, training set | **not yet**: `bash finetune/wan/v9_box.sh archive` after the suites finish |
| on the box | `/workspace/wan_output/pudgy-v9c-phone-lownoise/`, `/workspace/eval_v9/runs/pudgy-v9c-phone-lownoise/` (dcls/, suites/, winner), `/workspace/logs_v9c/` |

---

## 6. Still running / not done

- `archive`. Do not release the box until it reports every size check matching.
- Custom-background inference clips: waiting on the background, characters and actions.

---

## 7. Findings and next steps

Root-cause research and the v9c2 plan: [`RESEARCH_v9c_fixes.md`](../../../training_approach/v9/RESEARCH_v9c_fixes.md).

1. **The phone loss gate is too easy to pass for what we want.** A 39% loss drop that leaves the
   defect untouched means the loss gate alone can't keep a prop stage. The visual bar ID1 now exists
   and is validated (0/380 false alarms on real clips).
2. **Test inference-only fixes before any retraining**: a negative prompt for screens and text,
   lower CFG, a phrase without "black screen", and FLF. Then the asset embedding. Train the
   high-noise expert only if the x₀ probe shows the turn is set before the switch.
3. **Motion recognition dropped (0.75 → 0.58 held out)**, mainly through the **sitting caption**,
   which the phone-sitting caption repeats at 190 : 7. Anchor that wording (or use the embedding
   route, which leaves v8 prompts untouched), and gate on paired margins, not only loss.
4. **Fewer steps would have done.** Steps 125–200 differ by < 5% in phone loss. A ~125-step stage
   saves ~10 h of a 27 h run.
5. **Binding by text is not happening at this scale** (n = 4, 0–1/4). Keep relying on reference
   pixels for the look (the "one name, one look" rule), and treat `textcall` as informational.
