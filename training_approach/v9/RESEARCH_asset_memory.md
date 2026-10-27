# Storing an asset once and calling it by name: research for v9c2

Addendum to [`RESEARCH_v9c_fixes.md`](RESEARCH_v9c_fixes.md). Question asked: *are we using the
right mechanism to store the phone, so that a name maps to its look and the model learns it
better?* Short answer: **no, not yet.** This note explains what we do today, what the literature
does, and the mechanism we recommend for v9c2. Written 2026-10-26.

---

## 1. What v9 does today

| Part of the asset | Where it lives | Problem |
|---|---|---|
| **Name** | A 30-token English phrase ("the iFin phone, a small pale grey phone with a black screen, …") in `assets/registry.toml`, encoded by the frozen umT5 text encoder on every use | The words keep their ordinary meaning. "A phone with a black screen" pulls toward the base model's generic iPhone with the screen facing out, which is what the clips flip into (F2) |
| **Look** | The reference PNG, pasted into **frame 0 only** | Nothing refers back to it after frame 0; the model is free to drift |
| **What training learned** | The LoRA weights, shared by every prompt | Learned without using the caption (F1: 99.6% of the gain is caption-independent), so it leaks into every prompt (F3: motion regression) |

**There is no stored embedding.** "One name, one look" is enforced only by convention: the same
words and the same pixels. The model never gets a representation that *is* the iFin phone.

---

## 2. What the research does: three places to store a concept

| Mechanism | Stored as | Key papers | Effect on other prompts | Fit for us |
|---|---|---|---|---|
| **A. Learned token** (textual inversion, P+, pivotal tuning) | A few embedding vectors bound to a keyword; the network stays frozen | Textual Inversion (Gal et al. 2022); [P+ / XTI](https://arxiv.org/abs/2303.09522); pivotal tuning = token + LoRA; [DreamVideo](https://arxiv.org/abs/2312.04433) builds its identity step this way (token first, then an adapter) | **Exactly zero** when the keyword is absent: the vectors only enter through that keyword | **High**: works on our base and checkpoints today (§3) |
| **B. Reference-image conditioning** | The asset image, encoded and injected as tokens that every frame can attend to | [VACE](https://arxiv.org/abs/2503.07598) (Wan2.1-VACE-14B; [Wan2.2-VACE-Fun-A14B](https://huggingface.co/qqceqqq/wan2.2-vace-fun-a14b), built on T2V-A14B); [Phantom](https://huggingface.co/papers/2502.11079) (Phantom-Wan 14B, `bytedance-research/Phantom`); [SkyReels-A2](https://arxiv.org/pdf/2504.02436) (A2-Wan2.1-14B); [Video Alchemist](https://arxiv.org/abs/2501.06187) and [ConceptMaster](https://arxiv.org/abs/2501.04698) **bind each reference's embedding to its word in the prompt**, which is your idea exactly | No weight change at all; the reference is used only when supplied | **High value, high cost**: these are *different* checkpoints (Wan2.1, or Wan2.2 **T2V**+VACE). Our v7–v9 add-ons were trained on Wan2.2 **I2V** and don't carry over. A base migration (§4) |
| **C. Weights** (DreamBooth, LoRA, Custom Diffusion) | Weight deltas | DreamBooth; [Custom Diffusion](https://arxiv.org/abs/2212.04488) (cross-attention K/V only) | Leaks unless constrained ([LoRA forgets less, but still forgets](https://arxiv.org/abs/2405.09673v2)) | What we do now. Keep it, but constrained (cross-attention + preservation) |

Two related results bear on our motion regression:
- [**Still-Moving**](https://arxiv.org/abs/2407.08674) (Chefer et al., TOG 2024) learns a concept from **static** clips. It
  trains the identity part alongside a separate *motion adapter* that soaks up the "nothing moves"
  bias, then **throws the motion adapter away** at inference. That is exactly our situation:
  80% of v9c's data is near-static.
- **DreamVideo** splits the job the same way, with an identity adapter (token, then adapter) and a
  separate motion adapter.

---

## 3. Recommended: a learned asset embedding, spliced in after the text encoder

### Mechanism

The trainer doesn't run the text encoder; it reads cached umT5 outputs, one **[L × 4096]** matrix
per caption (L = 143 for a phone caption, `*_wan_te.safetensors`), and feeds them to every
cross-attention layer (`wan_train_network.py:692`). So the stored asset can be a small matrix
**E ∈ ℝ^{K × 4096}** that replaces the rows of the keyword's span:

    registry.toml   [assets.phone]  phrase    = "the iFin phone, a small pale grey phone …"  (29 tokens)
                                    embedding = "assets/ifin_te.safetensors"   # E, 29 × 4096
    caption         "… holding the iFin phone, a small pale grey phone …, up in front of its chest …"
    context         T5(caption) with the phrase's 29 rows replaced by E (replace) or shifted by E (offset)

Two modes, both implemented (`finetune/wan/asset_embed_v9.py`):

- **offset:** context rows = T5 rows + E, with E starting at 0. Step 0 is **exactly** the base, so any
  change is E's alone. This is the validation mode.
- **replace:** context rows = E, starting from the mean of the phrase's rows over the 380 captions.
  This is a true stored token, independent of the surrounding words. Each caption's own rows sit 22%
  (max 33%) from that mean, because T5 is contextual, but V1a measured the functional effect as
  ×1.0002 (max 0.4%) on phone-clip loss: effectively the base.

- **Training:** a patch at the batch's `t5` list splices E (an `nn.Parameter`) into phone samples.
- **Inference:** the same splice on `t5_embeds`. The prompt compiler finds the keyword and
  splices the matching E, one file per asset. That gives "store once, call by name" for every
  future prop.

### Why it targets our findings

| Finding | How the embedding addresses it | Why it holds |
|---|---|---|
| F3: leakage into other prompts | **Zero change on any prompt without the keyword** | E enters the network only at the keyword's rows. With the DiT frozen (pure embedding training), every non-phone prompt gives exactly v8merged's output. v8 regression is then 0 by construction, not by bar |
| F1: caption-blind learning | All learning *must* go through text | With the network frozen, the only free parameters are in the context, so every bit of the phone gain is text-attributable. The binding check becomes meaningful: "the iFin phone" vs "a white cup" now differ by exactly what E learned |
| F2: generic-phone prior | The keyword no longer means "phone with a black screen" | Initialise E from the visible-side phrase chosen in experiment E2, minus the screen words. Training then moves E towards whatever keeps *this* object stable through motion |

### What it can't do (being explicit)

- **It can't make the phone appear from text alone.** In image-to-video, I(x₀ ; c | img) = 0 still
  holds while the phone is always in frame 0 (RESEARCH_v9c_fixes §1). E learns to *hold* the
  object, not to *produce* it. Producing it needs entrance clips (client ask A1) or reference
  conditioning (§4).
- **Its capacity is limited.** 29 × 4096 ≈ 119k numbers, against the rank-16 LoRA's ~77M (40 blocks × [8 attention projections at 16·(5120+5120) + 2 FFN at 16·(5120+13824)]).
  It suits "what object to keep drawing", not new motion. So use **pivotal tuning**: stage 1 trains
  E with the network frozen; stage 2 trains a cross-attention LoRA with E frozen, with the
  preservation loss on replay. Stage 2 is the only part that can touch other prompts, and the
  preservation term pins it.
- **Its leverage at mid noise is small.** V1a: moving the phrase rows by 22% changed phone-clip loss
  by only 0.4% at t ∈ {0.65, 0.85}. The frozen low-noise expert barely listens to the phrase on phone
  clips there. Its leverage may be larger at high noise, where text sets the layout. That is one
  more reason to train E through **both** experts (below), and to judge it by generation (ID1), not
  only Tier 1.
- **It feeds both experts.** The same context goes to the high-noise expert (t > 0.9) at inference,
  so an E trained through the low-noise expert alone acts *untrained* on the high-noise steps. Either
  train it through both (musubi `--dit_high_noise`, plus a ~30-line patch so v8merged merges into the
  low-noise expert and the golden LoRA into the high-noise one; musubi's dual mode merges
  `--base_weights` into one expert only), or splice it only on low-noise steps (a generator patch).
  V2 tests "E on both" first, since that's what the T5 hook does.
- **Prompts that combine the phone with other actions are untested.** E is trained only on
  texting poses; "walking while holding the iFin phone" may carry texting stillness with it. There
  is no data for that yet (client ask A1 helps).
- **The splice works on contextual outputs.** The tokens around the keyword were encoded with the
  placeholder phrase in place. That's fine as long as training and inference use the same placeholder
  (the compiler guarantees this).

### Experiment E5 (adds to §4 of the fixes plan)

| | |
|---|---|
| Setup | E only (DiT and v8merged frozen), 29 rows; `replace` initialised from the E2c phrase if E2c wins, otherwise `offset`; lr 5e-3 (the text rows' element std is 0.088, so Adam moves each element ≤ 6% of that per step, gentler than standard textual inversion); 150 steps; through both experts once the merge patch exists |
| Cost | ≈ 6 h training (short clips; replay samples carry no E gradient and can be dropped) + Tier 1 on phone clips only (v8 rows unchanged by construction) + ID1 confirm tier (24 clips at 41 frames ≈ 6.6 h) |
| Read | **ID1** defect rate vs base (the decider); P1 held-out phone loss; P2 holds by construction |
| Patches | done: `asset_embed_v9.py` (cache spans, in-memory trainer patch), `eval_v9/dcls_asset_v9.py` (Tier 1), `eval_v9/gen_asset_v9.py` (generation via a T5 hook). To do: the dual-expert merge patch |

### Validation status (2026-10-26)

| stage | result |
|---|---|
| V0: phrase spans and an exact splice | ✅ 380/380 phone captions, 0 others; cache rows = tokens; splice bit-exact |
| V1a: `offset` E = 0 vs base | ✅ 0.0 difference on all 40 clip × caption losses |
| V1a: `replace` init vs base | ✅ ×1.0002 mean, max 0.4% |
| V1b: embedding-only training, 40 steps | running |
| V1c: Tier 1 with the trained E | after V1b |
| V2: generation + ID1 | next |

---

## 4. Longer term: reference conditioning (mechanism B)

This puts the iFin PNG *and* the characters in as references that every frame attends to, with
each reference bound to its word. That gives text-called props with no frame-0 pasting, which is
what `textcall` asks for.

| Option | Base | Open weights | Cost to adopt |
|---|---|---|---|
| Wan2.2-VACE-Fun-A14B | Wan2.2 **T2V**-A14B + VACE | `alibaba-pai/Wan2.2-VACE-Fun-A14B` | Re-train the Pudgy add-ons on T2V (v7–v9 don't transfer) |
| Wan2.1-VACE-14B | Wan2.1 | `Wan-AI/Wan2.1-VACE-14B` | Same, on an older base |
| Phantom-Wan 14B | Wan2.1 | `bytedance-research/Phantom` (also in `Kijai/WanVideo_comfy`) | Same; up to 4 references |
| SkyReels-A2 | Wan2.1-14B | A2-Wan2.1-14B-Preview (480×832) | Lower resolution; preview |

**Zero-shot pilot first, no training:** give VACE-Fun-A14B (or Phantom) the iFin PNG and a Pax
reference with the texting prompt. Score it with the same ID1 metric. If the phone holds better
zero-shot than our trained I2V stack, that's the case for migrating, and it goes in the v10 plan
rather than v9.

---

## 5. Updated v9c2 order

1. Inference experiments E0–E3 (fixes plan §4, rev 2), plus a VACE/Phantom zero-shot look (§4).
2. **V1 → V2 → E5**: the asset embedding on its own, through both experts.
3. Pivotal stage 2, only if E isn't enough: a cross-attention k/v LoRA with E frozen, sitting
   anchors and preservation. The high-noise LoRA only if the E1a x₀ probe points there.
4. Client data (A1 entrance clips, A2 turnaround, A3 layers) feeds the next pass, and the
   reference-conditioning route.

No new client asks beyond A1–A6. The turnaround sheet (A2) becomes the **reference image set**
for mechanism B, and the layered exports (A3) supply clean asset crops for it.
