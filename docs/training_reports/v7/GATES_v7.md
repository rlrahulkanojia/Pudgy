# v7 Gates — what each one is for, and which ones ran

**Status as of 2026-09-11.** Training is complete and the GPU has been released. This
document is the gate register: what every gate tests, why it exists, whether it ran, and
what it found. It is the companion to [`REPORT_v7.md`](REPORT_v7.md) (results) and
[`Training_Approach_v7.md`](../../../training_approach/v7/Training_Approach_v7.md) (plan).

---

## 0. What a gate is — and is not

A gate is a **behavioural pass/fail test run against a frozen checkpoint**. It generates
video through the real inference path and measures the produced pixels.

**Gates do not train or improve the model.** Every gate runs `wan_generate_video.py`,
never `wan_train_network.py`. Running more gates produces more evidence about the same
weights; it does not make the output better. This is worth stating plainly because the
name suggests a pipeline stage that improves something — it does not.

The reason the programme gates behaviourally at all is v5 §4.3: **loss and weight-drift
both said "fine" while the model had stopped responding to prompts.** Nothing here is
inferred from training curves.

### The metric, and its direction

Most gates compare two clips that differ only in their prompt, using SSIM over the face
crop. **Lower SSIM = more different = better controllability.** The bar is 0.92 on the
face crop (v6 calibration); v5's whole-frame calibration was 0.9692 = prompt ignored,
0.9340 = prompt works.

---

## 1. Register

| Gate | Tests | Why it exists | Status |
|---|---|---|---|
| **G-X** | Motion expert A/B: high-noise vs low-noise | The plan's one genuinely open question (§4) | ✅ **PASS — M-low** |
| **G-C** | Controllability: prompt is the only variable | The core claim — is the label promptable? | ✅ **expression + motion** |
| **G-P** | Polly parity | v4 rendered Polly blue; parity is not assumed | ✅ **PASS** (1.5–1.7% gap) |
| **G-D** | Duration control across the ladder | The gate the duration ladder exists to pass | ⚠️ **partial** (probe only) |
| **G-L** | Length is not a shortcut for the label | v6 §4.1 confound; the ladder's whole purpose | ❌ **not run** |
| **G-F** | A *training* start frame does not trigger the label | **The exact v5 failure** | ❌ **not run** |
| **G-M** | Motion survives from a *novel* start frame | v5 collapsed to 0.8 px from a trained frame | ❌ **not run** |
| **G-N** | Idle is promptable ("stop" is a state) | Needed for beat chaining | ❌ **not run** — expected to FAIL |
| **G-B** | Background invariance on unseen grounds | Alpha compositing must generalise | ❌ **not run** |
| **G-H** | Expression holds past frame 30 | v5 relaxed by f11 | ❌ **not run** |
| **G-Z** | Shot size is promptable | The zoom ladder must be real | ❌ **not run** |
| **G-S** | Two trained LoRAs stacked | **Not required** — only if M-high won G-X | ⬜ **N/A** |
| **G-A** | Unseen camera angle | **Not applicable** — no pose holdout was taken | ⬜ **N/A** |
| **G-R** | No regression vs the v2 showcase | Guards the existing product | 🚫 **BLOCKED** |

Legend: ✅ ran and passed · ⚠️ partial · ❌ implemented, not run · ⬜ not applicable · 🚫 blocked

---

## 2. The gates that ran

### G-X — which expert carries motion  ✅ PASS
**Question.** v6 rejected training the high-noise expert, citing v5. But v5's negative was
*expression*-on-high-noise — fine facial detail in the expert that carries global
composition. Motion-on-high-noise is a different proposition and had never been tested.
Both arms were trained (2 epochs each) and compared.

| Arm | worst-pair SSIM | Distinct | Animation |
|---|---|---|---|
| M-high | 0.9618 | ❌ | 2.9× static |
| **M-low** | **0.9444** | ✅ | **4.3× static** |

**Verdict: motion goes on the low-noise expert.** M-high's actions blur together — 0.9618
is close to v5's 0.9692 "prompt ignored" mark.

**Consequence, and it is the good one:** the high-noise v2 golden stays **frozen**, so it
remains a fixed reference. A later regression can still be bisected by swapping one LoRA
back. Had M-high won, both experts would have changed and that ability would be gone —
which is why G-S exists and why it is now unnecessary.

> ⚠️ **This gate was wrong on its first run and reported both arms failing.** It reused
> v5's ">200 px subject x-range" bar, which belongs to **G-M** — a gate that asks the
> character to *turn its head*, so the subject's centre of mass sweeps sideways. Every v7
> motion label is deliberately *in place* ("walking **in place**", "hopping straight
> **up**"), so the centre of mass barely moves even when the animation is perfect: it
> scored 5–16 px against a 200 px bar on clips that visibly animate. It now measures
> frame-to-frame MAE against a **generated** static reference (0.0049, measured). The
> lesson generalises: **a threshold is only valid for the motion it was calibrated on.**

### G-C — controllability  ✅ PASS (with one known weak pair)

**Expression** — 7 labels, 21 pairs per character, 3 seeds, on the golden:

| Checkpoint | mean | worst | failing |
|---|---|---|---|
| step 500 | 0.8118 | 0.9581 | 1/126 |
| **step 2500** (golden) | **0.8092** | **0.9504** | 1/126 |

**125 of 126 pairs distinct.** The single failure is `confused|neutral` for **Polly on
seed 42 only** — seeds 43 and 44 pass. `confused` is the thinnest class in the set (64
clips against 216 for mature labels, 7 source angles against 9) and the subtlest
expression. Reads as a **data gap, not a training failure**.

For scale: v6 separated 4 emotions at 0.83–0.85 typical. v7 sits at 0.809 across nearly
four times as many pairs — **the vocabulary roughly tripled without controllability
degrading.**

**Motion** — 5 actions, 10 pairs per character, 4 checkpoints:

| Checkpoint | mean | worst | Pass |
|---|---|---|---|
| 500 | 0.8056 | 0.8475 | ✅ |
| 1000 | 0.8028 | 0.8491 | ✅ |
| 1500 | 0.8029 | 0.8595 | ✅ |
| 2000 | 0.8021 | 0.8504 | ✅ |

**All pairs distinct at every checkpoint**, with more headroom than expression.

### G-P — Polly parity  ✅ PASS
Pax 0.8022 / Polly 0.8162 → **1.7% gap** against a 10% bar (1.5% at step 500). Computed
inside G-C. v4 rendered Polly blue because none of its solo-Polly captions said "pink";
every v7 caption names the colour, and parity confirms it worked.

### G-D — duration control  ⚠️ PARTIAL
The full gate (every label × every ladder rung × 3 seeds, 192 clips) **did not run**. An
informal probe covered 4 motion actions at their natural length and at 49 frames (2.04 s):

| Action | Stretch | Length exact | tail/head |
|---|---|---|---|
| running | **2.9×** | ✅ | **1.46** |
| jumping | 1.5× | ✅ | 0.75 |
| waving | 2.0× | ✅ | 0.79 |
| sitting | 2.0× | ✅ | 0.07 |

**All 8 returned exactly the requested length**, so duration is controllable. `running`
sustains at 2.9× beyond its trained length. `sitting` is a *seated idle* by design, so
near-zero movement is correct. `tail/head` compares movement in the last third against
the first — it catches an action that completes early and then freezes, which a
whole-clip average hides.

> ⚠️ The animation metric is **too blunt for move-then-hold gestures**. `waving` scored
> below the animation floor even at its trained length, yet the filmstrip shows the
> gesture rendering correctly — the flipper raises and holds. A few frames of large change
> average away. Do not use this metric as a pass/fail bar for gestures without checking
> the frames.

---

## 3. The gates that did not run

All are **implemented and dry-run verified** in `finetune/wan/gates_v7.py`; they need GPU
time only. Costs are measured (6.4 min/clip at 21 frames, 4.0 at 13, scaling with
sequence length), assuming the batched runner.

| Gate | Clips (3 seeds) | Est. | Why it matters |
|---|---|---|---|
| **G-F** | 18 | 2.0 h | **The highest-value gate left.** v5's failure was that the *training start frame* became a deterministic trigger: the expression fired regardless of the prompt, and motion from it collapsed to 0.8 px. v7's mitigations (common bucket, shot ladder, 4 grounds) are designed to prevent it. **Untested.** |
| **G-L** | 51 expr / 45 motion | 12.7 h | Whether clip length secretly encodes the label. In v6 each emotion shipped at exactly one length, making length a perfect predictor; the v7 duration ladder exists to break that. The ladder is built and captioned, but **the confound has not been re-tested**. |
| **G-D** (full) | 102 expr / 90 motion | 25.3 h | Duration control across the whole ladder, not the 4-action probe. |
| **G-B** | 36 expr / 36 motion | 6.4 h | Unseen grounds (lavender, sky-blue, both reserved and never trained). v5 measured 2/255 drift. |
| **G-M** | 6 expr / 6 motion | 1.1 h | Motion from a *novel* frame. Keyframes are built (a v1 bathroom-sink scene — different pose, palette, framing). |
| **G-N** | 6 | 0.4 h | **Expected to FAIL, by construction.** `standing_idle` is DERIVED from `neutral` footage whose body is frozen (measured MAE 0.028–0.064), so it teaches "stop" as "freeze". Running it documents the size of the gap; fixing it needs delivered idle footage (Round 4 §3). |
| **G-H** | 9 | 2.3 h | Whether an expression holds past frame 30. v5 relaxed by f11. `laughing` is the programme's first real emotive hold. |
| **G-Z** | 9 expr / 9 motion | 1.6 h | Shot size promptable, label legible at 0.55× wide. |

**Total to a complete table: ~52 h at 3 seeds, ~26 h with G-D and G-L at 1 seed.**
G-D and G-L are 60% of the cost and are *coverage* matrices — 3 seeds buys less there
than on a behavioural pass/fail gate like G-F or G-N.

### G-R — blocked, not skipped  🚫
G-R re-runs **v2's 10-scene showcase prompts** and checks temporal SSIM ≥ 0.94 and
struct-stability ≥ 0.87 against v2's published 0.949 / 0.880.

**The prompt text does not exist in this repo or in the Azure container.**
`pudgy/v2/eval/showcase/` holds the rendered mp4s (`01_construction` … `10_doorway_mug`)
and `v2/docs/REPORT_showcase_analysis.md`, but neither carries the strings.
`gates_v7.py --gate gr` therefore **refuses to run without `--prompts`** rather than
falling back to `sample_prompts.txt`, which is v6's in-training sampler prompts — scoring
those and reporting "no regression" would be a confident, meaningless pass.

**Action required: obtain the 10 prompt strings from the author.**

---

## 4. How to run the rest

```bash
set -a; . /workspace/.env; set +a
cd /workspace/Pudgy/finetune/wan
GOLDEN=/workspace/wan_output/pudgy-v7-expr-lownoise/pudgy-v7-expr-lownoise-step00002500.safetensors

python gates_v7.py --gate gf --kind expression --seeds 42 43 44 --ckpt $GOLDEN
python gates_v7.py --gate gl --kind expression --seeds 42 --ckpt $GOLDEN
python gates_v7.py --gate gn --kind motion --side low --seeds 42 43 44 \
    --ckpt /workspace/wan_output/pudgy-v7-motion-lownoise/pudgy-v7-motion-lownoise.safetensors
```

Gates batch by default — one model load per LoRA configuration instead of one per clip
(measured: 501 clips need 15 loads, not 501; ~33 h of reloading saved). `--no-batch`
restores the per-clip path. Clips already on disk are skipped, so re-running after a
metric fix costs no GPU. `--dry` prints the matrix and writes nothing.

**The weights are unchanged and complete** — every checkpoint is on Azure under
`pudgy/v7/weights/`, so the gates can be finished on any box without retraining.
