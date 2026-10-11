# MiniMax H3: can it be the base for the Pudgy Penguins LoRA work?

**Date:** 2026-10-11
**Status:** desk research only. No GPU job was run and nothing was paid for. Revised the same day: the per-label lengths (§4.1) were re-counted from the raw files, and the shift test (§6.3, §6.5) was corrected. The experiment plan is [`expv2-minimax-h3/`](../../../expv2-minimax-h3/README.md).
**Licence:** set aside at the user's request (2026-10-11). The verdict, pros/cons and gates below are **technical only**. The licence findings are kept, for the record only, in Appendix A.
**Scope:** MiniMax H3 (open weights released 2026-07-28 to 08-02), compared with the bases and recipes of v1–v9 and the standalone LTX-2.5 run. The Kandinsky experiment is left out on purpose.

**What was read**
- The Hugging Face repo: model card, LICENSE, license Q&A, every config, the VAE code, and the safetensors headers. The parameter counts below are computed from tensor shapes.
- The MiniMax blog, design.minimax.io/h3, and the Notion overview page.
- GitHub `MiniMax-AI/MiniMax-H3`: all 82 issues and PRs.
- All 113 Hugging Face discussions.
- The diffusers port, the SGLang cookbook and the vLLM recipe.
- Trainer code and issues for DiffSynth-Studio, musubi-tuner, ai-toolkit, SimpleTuner and diffusion-pipe.
- The LoRA model cards on Hugging Face and ModelScope.
- The Artificial Analysis and LMArena leaderboards.
- This repo's v1–v9 and `ltx25/` reports.

**What could not be read:** Reddit (blocked to every tool), and an H3 technical report, which does not exist yet.

**How claims are labelled**

| Label | Meaning |
|---|---|
| **V** | Verified: read in code, a config, a weight header or a primary document |
| **R** | Reported: someone said it in an issue, a model card or a discussion, and it is not reproduced here |
| **I** | Inferred: our own arithmetic or reasoning |

Lesson tags such as A6, B11 or D4 point to the index of v1–v9 lessons, each with its source, in `expv1-kandinsky-50-video-pro/docs/LEARNINGS_CARRIED.md`. The tags are used only as references; this document does not compare against that experiment.

---

## 0. Verdict

| Question | Answer | Confidence |
|---|---|---|
| Is H3 a better *base model* than what v2–v9 used? | **Probably yes on general I2V quality.** LMArena I2V ranks it #1 (Elo 1495). Wan 2.2 A14B scores 1170 there, which works out to an **≈87% expected preference** for H3. On Artificial Analysis silent I2V the margin is 1356 vs 1107, **≈81%**. It is the top open-weight model on both boards. | Medium. These are general-content votes, not 2D-cartoon votes, and the AA entry was probably run through MiniMax's hosted pipeline (§3) |
| Does it fit our current data? | **Partly.** H3's shortest legal clip is 22 frames. Of 227 single-character sources, **145 (64%) reach 22 frames, 67 (30%) reach 39** (angry, laughing, blushing, neutral) **and 18 (8%, `neutral`) reach 56**. Every label is present only at 22, so a run on today's data is in practice a **0.92 s (22-frame) model**. **4 of 13 labels have no clip that long**: walking (16 f), running (17 f), happy (21 f), confused (21 f). Sitting has missing frames in 13 of 18 sources, which leaves **5**. Walking can be loop-tiled as in v8, and 21-frame clips padded by one frame; running needs Round 5's 3–5 s takes (§4.1) | High. Counted from the raw files' decoded frame timestamps, de-duplicated by MD5 |
| Can it be fine-tuned on our hardware? | **Yes, on 1× A100 80 GB.** Use the pruned AdaLN base (40 GB bf16 / 21 GB int8). One public first-frame I2V LoRA was trained on exactly that card at **12.5 s/step at 768²×73 frames** (§6.4) | Medium–high |
| Will a LoRA hold its quality? | **Unknown, and this is the main technical risk.** Only **CFG-distilled** weights are public. Plain LoRA training "de-distils" them: washed-out output and lost prompt adherence, reported by every trainer. Workarounds exist, but degradation is still reported at 3–5k steps (§6.1) | Medium |
| Will the VAE keep Pudgy's thin outlines and flat grounds? | **Unknown, and risky.** Its spatial compression is 16×, leaving **2.1–2.7× fewer latent values per pixel than Wan's VAE**. There are open reports of line art fading (#73) and colour shift on plain backgrounds (HF #78). It must be measured on our frames before anything else (§4.4) | Medium |

**Recommendation (technical only)**
- **H3 is a credible candidate, worth one isolated experiment now:** `expv2-minimax-h3`, following the `expvN-<model>` convention. It is **not** a v-line replacement until it beats v8@550 on v8's held-out `QF2_R` clips with re-calibrated tiers.
- Run it in this order; each step stops the experiment if it fails:
  1. **VAE gate (G0.1):** Pax and Polly round-trip through H3's VAE at 768² and hold their outlines at 4× zoom (the ltx25 §1 method), with flat-ground drift ≤ 2/255. This is the cheapest test, needs no training, and decides the most.
  2. **Zero-shot probe (G0.4):** on unseen start frames, at 22/39/56 frames and 3 seeds, measure cadence (on ones or twos?), 3D drift and first-frame drift.
  3. **First LoRA on today's data:** a 22-frame corpus covering 12 of 13 labels (running excluded; sitting thin at 7 usable sources).
  4. **Round 5 re-run:** a second run once Round 5's longer takes arrive. This is where H3's 39–56-frame buckets and its native 4–15 s output can actually be used.
- Expected cost: ≈6–14 A100-hours for a v8-sized run, against v8's 57.5 h (§6.4). That is an estimate, to be re-measured in G0.6.

---

## 1. What H3 is (verified numbers)

| | Value | Source |
|---|---|---|
| Transformer | **33.123B params**, dense, single-stream. 50 blocks, hidden 5376, attention 56×128 (inner dim 7168), gated FFN 14336, 2 token-refiner blocks | V: `transformer/config.json`, safetensors headers |
| Where the params sit | **13.010B (39.3%)** in AdaLN modulation (`adaln_proj` 96768×2688 per block = 6 × 5376 × 3 modalities). **19.27B** in the 50 blocks' attention and FFN. **0.77B** in the refiner | V, header arithmetic |
| AdaLN | Depends only on (timestep, modality), so it can be cached. Community "pruned" checkpoints replace the 13B with a `[1025, 8]` table: **66.3 → 40.2 GB bf16** | V: model card l.142; R: Comfy-Org and DiffSynth file sizes |
| Text encoder | Qwen3-VL-32B (64 layers, uses layer 50's hidden states), 66.7 GB bf16. The **start frame also goes through it** as vision tokens | V: config; V: diffusers `encoders.py` (agent read) |
| Video VAE | f16t4d24: 16× spatial, 24 channels. Causal conv encoder plus a **36-layer ViT decoder**. 2.604B params, fp32, 10.4 GB. Encodes 17-frame chunks into 5 latents (≈3.4 frames per latent) | V: config, `klvae.py` l.100–106, headers |
| Patchify | 1×2×2, so **one token per 32×32 px** (Wan: per 16×16 px) | V |
| Checkpoints | **FL2VA**: text, first frame, last frame, or first+last → audio+video. **Ref2VA**: ≤9 images, ≤3 videos, ≤3 audio refs | V: README |
| First-frame conditioning | **In-context tokens**, not channel concatenation. The keyframe is VAE-encoded into extra rows held at t = 0.999, at frame-0 RoPE time. The generated frame 0 is *not* overwritten, so it only approximates the input | V: diffusers `before_denoise.py` (agent read) |
| Guidance | **CFG-distilled**. No guidance input and no negative prompt in the official pipelines. One forward per step | V: README l.181; no guidance tensor in headers |
| Scheduler | Rectified flow, shift **12 (video) / 3 (audio)**, 50 steps by default | V: `scheduler_config.json` |
| Output | **4–15 s**, 24 fps, short edge 768 by default (trained canvas 1344×768 / 768×1344), 32 kHz stereo audio on every generation. 2K is API-only | V: README |
| Frame grid | **17n+5** frames → 5n+2 latents. Legal lengths here are **22 / 39 / 56 / 73** | V: diffusers and DiffSynth checks |
| Not released | Context-IR prompt rewriter ("critical to the quality", README l.80), Regenerate-2K, sparse attention, un-distilled weights, training code, technical report | V |
| Download (FL2VA) | ≈144 GB in bf16/fp32. The pruned int8 route is ≈42 GB: DiT 21 + text encoder 15.7 (nvfp4) + VAE 5.2 (fp16) | V: HF API sizes |

---

## 2. Side by side with the bases v1–v9 used

| | v1 CogVideoX1.5-5B | v2, v5–v9 Wan2.2-I2V-A14B | v4 LTX-2.3-22B | ltx25 LTX-2.5-22B | **H3 FL2VA** |
|---|---|---|---|---|---|
| Params doing the work per token | 5B | 14B (one of two experts) | 22B | 22B | **19.3B** (33.1B with AdaLN) |
| VAE (spatial / temporal / channels) | 8× / 4× / 16 | 8× / 4× / 16 | 32× / 8× / 128 | 32× / 8× / 128 | **16× / ≈3.4× / 24** |
| Latent values per pixel per frame | 0.0625 | 0.0625 | 0.0156 | 0.0156 | **0.023–0.030** |
| Tokens per 1024² latent frame | 4,096 | 4,096 | 1,024 | 1,024 | **1,024** |
| First-frame mechanism | channel concat | channel concat + mask | first-frame latent tokens kept clean | same as LTX-2.3 | **in-context tokens + VLM vision tokens** |
| Last-frame / FLF used by us | — | FLF2V at inference (v2, v5) | — | — | **native, first only, last only, or both** |
| Native length | 81 f @ 16 fps | 81 f @ 16 fps | v4: style held to 97 f, 249 f with IC-LoRA | 17–33 f trained | **4–15 s @ 24 fps** |
| Our clip lengths accepted as-is | 33 f | 13–57 f (4k+1) | 49 f (8k+1) | 17–33 f (8k+1) | **only 22 / 39 / 56** |
| CFG at inference | yes | yes (+ negative prompt) | yes | yes | **no, distilled in** |
| Licence | custom THUDM | **Apache-2.0** | LTX-2 Community | LTX-2 Community | **MiniMax H3 Community, excludes US/EU/UK/KR** |
| Our best measured result | subject vanished f24–29 | **v8@550**: Tier-1 acc 0.81 expr / 0.80 motion; compose motion 0.44 | G1 4.5/5, but seed basins 40–66% | waving motion decayed 1.48× → 0.52× of source | none (untested) |

Sources:
- Our numbers: `docs/training_reports/v8/REPORT_v8.md` §1, §3; `ltx25/docs/FINDINGS_run1.md` §1; `training_approach/v4/README.md`; `training_approach/FINDINGS.md`.
- H3 values per pixel: 24 / (16·16·4) = 0.0234 by the nominal ratio, and 0.0298 at the measured 22 frames → 7 latents (I).

**What the table means for us**
- **Per-pixel latent budget.** H3's budget lies between Wan's (the base whose outlines never failed in v2–v8) and LTX-2.x's. The LTX-2.x VAE passed our outline gate only with its *diffusion* decoder; the conv decoder "visibly dissolves outlines" despite better PSNR (ltx25 §1). H3's ViT decoder is a third design that nobody has tested on flat line art.
- **Compute per clip.** At equal resolution H3 has **4× fewer tokens than Wan**. That outweighs its ~1.6× more params per token, so per-clip compute is **0.29–0.46× Wan's** (§6.4).

---

## 3. Public quality evidence

| Board (snapshot) | H3 | Comparators | Expected preference for H3 (I) |
|---|---|---|---|
| **LMArena Image-to-Video** (2026-09-21; 2.08M votes, 48 models) | **#1, 1495 ±5** (57,112 votes) | wan-v2.2-a14b 1170; ltx-2-19b 1159; hunyuan-video-1.5 1198; hailuo-2.3 1262 | vs Wan 2.2 **0.87**; vs LTX-2 **0.87** |
| **Artificial Analysis I2V-Silent v1.0** (fetched 2026-10-11, 83 models) | #4 overall, **#1 open-weight, 1356 ±11** | Wan 2.2 A14B 1106.7 (#68); Wan 2.1 14B 1000 (anchor); HunyuanVideo-1.5 1133.9 | vs Wan 2.2 **0.81**; vs Wan 2.1 **0.89** |
| AA I2V v1.0 (with audio) | #2, 1181 ±8 (#1 is "H3 Max", fal's API-only post-train, not open) | LTX-2.3 Pro 949; Wan 2.6 889 | — |
| AA T2V-Silent, "Animation & Gaming" slice | **#2, 1114.8 ±37** (n = 423); top open-weight | LTX-2.5 ≈880; LTX-2.3 Pro 827 | vs LTX-2.5 ≈0.79 |
| Academic | Sci-VBench (2608.09873): best open model, 2.61 vs HunyuanVideo-1.5 2.43. Physical reasoning (2609.18323): 41.97% success over 517 cases | | |

Source: V (agent snapshots saved under the session scratchpad, spot-checked).

**Caveats, which matter more than the numbers**
1. The AA I2V entry is listed at $0.13/s, the 2K API price. It was probably generated by the **hosted pipeline (Context-IR + Regenerate-2K)**, not by the open 768p weights we would fine-tune (I).
2. These are untuned, general-content votes. Our question is how a **LoRA-tuned** model holds **flat 2D penguins**. On that there is only anecdote, and it cuts both ways (§4.3–4.4).
3. Elo gaps measure preference rate, not production pass rate. v4 is the warning here: LTX-2.3's G1 averaged 4.5/5, yet 40–66% of seeds fell into "raccoon" or "3D claymation" basins (`training_approach/v4/README.md:135–136`).
4. The client has already rejected hosted outputs from Seedance, Wan 2.7 and OpenArt on quality, and asked for custom training. A strong leaderboard rank does not replace a LoRA.

---

## 4. Fit with Pudgy's data and deliverable

### 4.1 Clip length: the largest technical mismatch

H3 was trained on 4–15 s clips, which musubi gives as 124–362 frames. Every official pipeline refuses durations under 4–5 s (V). Trainers accept 17n+5 frames, and community LoRAs do train at 22–39 frames (R: alvdansen, uwcc, bubu_yier). Nobody has published quality evidence below 4 s.

Applied to our data:

| | Count | Note |
|---|---|---|
| v8 clips (2,096) shorter than 22 frames: f13, f17, f21 rungs | **1,020 (48.7%)** | unusable as cut; f21 misses the floor by one frame |
| Re-cut to H3's grid | 772 → 22 f, 240 → 39 f, 64 → 56 f | (I, `prep_v8` bucket counts) |
| walking: longest clip 0.67 s = 16 f | **no legal length** | loop-tiling is possible (v8 seam 0.62–1.04) |
| running: longest 0.71 s = 17 f | **no legal length** | not loopable (seam 2.3–2.7×) |
| happy, confused: delivered at 21 f only | **no legal length** | one frame short |
| waving, sitting, jumping (26 / 28 / 33 f) | 22 f only | 39 f needs ≥ 1.63 s |

Sources: `training_approach/v9/Training_Approach_v9.md:30–34`; `training_approach/v8/Training_Approach_v8.md:135–154`.

**Per label, from the raw files.**
- Method: `Data/raw/iteration_3–5`, 240 unique files by MD5. The turnaround and the 12 two-character clips are excluded, leaving 227 sources.
- "Usable" means the frames before the first missing one, read from decoded frame timestamps (the `prep_v8.real_frame_gap` method). That method finds the same 21 damaged files v8 found.
- An earlier draft of this table used the v7 manifest's `source_frames`. Those values are v7's 4k+1 rungs, not true lengths, and v7 never detected missing frames. That draft therefore under-counted angry, laughing and neutral and over-counted sitting.

| Label | Sources | Usable length (f) | ≥22 | ≥39 | ≥56 | Share of the performance kept at 22 f |
|---|---|---|---|---|---|---|
| neutral | 18 | 60 | 18 | 18 | 18 | 37% (56 f keeps 93%) |
| angry | 18 | 40 | 18 | 18 | 0 | 55% (39 f keeps 98%) |
| laughing | 18 | 39 | 18 | 18 | 0 | 56% (39 f keeps all) |
| blushing | 14 | 41 (one source 32) | 14 | 13 | 0 | 54% (39 f keeps 95%) |
| jumping | 18 | 33 | 18 | 0 | 0 | 67% (the landing may be cut) |
| crying, surprised | 18 + 18 | 31, 30 | 36 | 0 | 0 | 71%, 73% |
| waving | 18 | 26 | 18 | 0 | 0 | 85% |
| sitting | 18 | 28, but **13 sources have gaps** at frames 13–25 | **5** (7 at ≥ 21) | 0 | 0 | 79% |
| happy | 14 | 21 | 0 | 0 | 0 | — (one frame short) |
| confused | 18 | 21 (4 damaged: 5, 8, 17, 17) | 0 (14 at 21) | 0 | 0 | — (one frame short) |
| running | 18 | 17 | 0 | 0 | 0 | — |
| walking | 19 | 16 (one file undecodable) | 0 | 0 | 0 | — |
| **Total** | **227** | | **145 (64%)** | **67 (30%)** | **18 (8%)** | |

Two consequences:
- **The 22-frame floor is the only length every trained label shares.** The 39- and 56-frame rungs exist for four expressions only. That is allowed, since length must not predict the label only at the shared floor (B11), but it means every motion trains at 22 frames.
- **Sitting is the new thin class.** It has 5 sources at 22 frames, or 7 if a single held frame is allowed. Neither sitting `QF2_R` clip (17 and 21 frames) reaches 22.

**Ways round it, and what each costs**
- **Padding with held frames:** teaches "stop moving", the failure we keep fighting (ltx25 waving decay; v8 compose motion 0.44). **One** held frame (happy, confused and two sitting sources, 21 → 22) is negligible next to the source's own 37.9% median frozen share. Five held frames (running, 17 → 22) are not.
- **Loop-tiling:** works only for walking.
- **Waiting for Round 5:** the client was asked for 3–5 s clips with ≥3 cycles (`docs/documents/Client_Data_Request_Round5.md:48–58`). Those clips would land in H3's range (73–124 frames). Until then H3 trains almost entirely at 22 frames, about **5.6× below its 124-frame trained floor**.

At inference, 13–21-frame outputs mean generating 22 frames and trimming. The cost is negligible, but diffusers' 5-second check has to be bypassed (V).

### 4.2 Resolution and aspect

- **768×1344 (9:16) is H3's trained canvas** (V), and 9:16 is the client's main format (`Client_Data_Request_Round3.md:61`). That is a better match than v5–v8, which trained Wan at **1024² square, off Wan's canonical shapes** (`docs/training_reports/v5/REPORT_happy_pilot.md:287`).
- **512²** is valid in code but below the documented 768 short edge. HF #65 says adherence is *better* at 352–576p (R).
- **1024²** is just over the 768×1344 area cap (1.05 MP vs 1.03 MP).
- **768²** is the closest official square.

### 4.3 Frame cadence and style priors

- **On twos.** H3 tends to animate anime-style output **on twos**: "likely to be 12 FPS instead of 24 (each frame repeated twice)". The reporter got ~1 in 10 true-24 fps clips (V: HF #74; the musubi docs call it "a learned production convention").
  - The client's animation is **true 24 fps on ones** (client-confirmed 2026-07-01; v8 plan §1).
  - So the base prior conflicts with the art. A LoRA can plausibly override it: alvdansen's keyframe LoRA standardised cadence (R). Measure the duplicated-frame rate at Phase 0.
- **3D drift.** "the base H3 model's 2D anime style leaned too heavily towards 3D" (V, RunningHub 2D slider LoRA card, 2026-10-10). This is the same failure class as v4's "3D claymation" basin, which hit ~40% of Pax seeds.
- **Motion not learned.** One 2D LoRA author says anime motion timing is not learned; it stays base-H3 (V, akhaliq card).

### 4.4 VAE vs thin outlines and flat grounds

H3's latent budget is 2.1–2.7× smaller than Wan's per pixel (§2). Pudgy art is thin dark outlines on flat fills over flat grounds, which is the worst case for every report below:

| Report | Status |
|---|---|
| "线稿消失虚化" (line art disappears and blurs) | V, GitHub #73, no maintainer reply |
| Colour shift on a plain VAE round-trip, "serious on a pure background" | V, HF #78. Our flat-ground drift bar was 1.1–1.7/255 in v8 |
| "fine details in distant backgrounds noticeably lost" | V, DiffSynth maintainer, issue #1585 |
| Small subjects pixelate; MiniMax called it system-level | R, AMA summary |
| Periodic brightness dips on a 17-frame cadence | V, GitHub #85, traced to a dtype cast in ComfyUI's time embedder (a runtime bug, not the VAE), but worth checking |

**One trap to avoid in our own evaluation.** musubi found that all its H3 mp4 writers encoded at ~1 Mbps, "crushing fine detail… masquerading as generation artifacts" (V, musubi #1057). Our VAE gate must write lossless frames.

### 4.5 Captions and prompts

- H3 tokenises the prompt verbatim; there is no chat template. The text and vision tokens sit **inside the DiT sequence**, so a ~1.8k-token prompt is ~30% of a 512²×56-frame sequence (V/I).
- The official format has three fields: `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`. The I2V instruction line is "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced." (V). It is "extremely sensitive to the prompt format" (R).
- Our v8 caption grammar would have to be re-authored. The **content rules** still carry over:
  - colour words in every caption (A10: Polly renders blue without "pink");
  - length must not predict the label;
  - a contrastive `neutral` label.
- There is **no negative prompt**. That removes the lever, and also removes the failure: the default-negative zoom-out of −11.4% in v8 cannot happen.

### 4.6 Audio

- Audio is generated on every call (80 rows/s, ~3.5% of a 512² sequence). Training can zero its loss weight (`--audio_loss_weight 0`), and the maintainer says this "markedly speeds up visual convergence" (R, DiffSynth #1735).
- Silent training data is slightly off-distribution. Strip the audio at export.
- The client does not need audio now, but "will ask for audio when we move to longer narrative shorts" (`final_docs/Data_Requirements.md:59`). Native audio is a future plus, not a current one.

### 4.7 Two characters and props

- Ref2VA takes ≤9 reference images, so characters plus props work natively, with no re-architecture.
- Our scope ceiling is **frame control**, decided 2026-07-01 for CogVideoX and flagged as needing re-confirmation since the move to Wan. Ref2VA is therefore informational unless that decision is reopened.
- FL2VA (first/last frame) is inside scope and is the checkpoint to use. The AMA reports Ref2VA as "softer" (R).

---

## 5. Licence

Set aside at the user's request (2026-10-11); not part of this verdict. Notes kept in Appendix A.

---

## 6. How we would train it

### 6.1 CFG distillation: the central training risk

The public weights are CFG-distilled, and there is no un-distilled checkpoint. Requests for one (GitHub #75, HF #88, HF #97) have gone unanswered, and MiniMax answers "Not decided" on training code. What a plain flow-matching LoRA does to it:

| Evidence | Source |
|---|---|
| "regress to its pre-distilled state — normal performance when CFG is enabled, but poor quality when disabled" | V, DiffSynth maintainer, #1585 |
| "Video training then washes out and loses prompt adherence… image training breaks structurally within about 50 steps. Plain flow training is therefore not offered" | V, musubi docs |
| LineartAnime LoRA: improvements by ~step 500 at batch 4 (~2,000 samples), **degrading by step 3000** (~12,000 samples); rescued with EMA | V, #1585 |
| Ghosting in movement at ~5k steps | R, diffusion-pipe #533 |
| De-distill adapter plus guidance loss at scale 4: best of three runs, but "looks covered in white fog" at 10k steps | R, DiffSynth #1735 |

**The countermeasures** are all the same idea. Each trains against `uncond + s·(v − uncond)` using the model's own no-grad unconditional prediction, at **s = 3.5–4**:
- musubi `--h3_guidance_loss_scale 4 --h3_guidance_loss_sigma_min 0.15`;
- DiffSynth `--training_cfg_scale 4`;
- ai-toolkit "Contrastive Guidance 3.5";
- diffusion-pipe `cfg = 4`.

The guidance loss costs **~1.5× step time** (V, musubi PR #1045). Alternatively, use a "DeCFG" training adapter that is merged during training and dropped at inference (DiffSynth TrainingAdapter, ostris v3). Sources disagree on stacking the adapter with the guidance loss, so don't stack them without an A/B.

**What this means against our past budgets:**
- v8 saw 4,800 clips (600 steps × accumulation 8), and v7 ran 7,888 batch-1 steps.
- Both are **inside or past** the range where H3 LoRAs start to degrade (2,000–12,000 samples).
- Expect the best checkpoint early, as in v8 (plateau at step ~300) and ltx25 (best at 500–1000). Save every 50 steps and let Tier 1 choose.
- Our Tier-1 diffusion classifier still works on a distilled model, but its calibration (which t band, margins) must be re-derived. The conditional prediction is itself the guided one.

### 6.2 Trainer choice

| Trainer | H3 support | First-frame-only I2V training | Short clips | Verdict for us |
|---|---|---|---|---|
| **DiffSynth-Studio** | Since 2026-08-03; pruned, FP8, int8-ConvRot and NF4 bases | **Yes** (`--extra_inputs input_image`); keyframe goes to both Qwen3-VL and the DiT, in training *and* inference | Any 17n+5 | **Primary** |
| ai-toolkit | Since 2026-08-03; default is pruned int8 + nvfp4 text encoder | Yes (`do_i2v`) | 39 f default | **Fallback.** Proven on 1× A100 for I2V. Possible train/infer mismatch: the keyframe may skip the text path in training (I) |
| musubi-tuner | **v0.3.5 (2026-09-16)**, experimental; our pinned v0.3.4 has none | **No**: video FL2VA trains first+last only (`minimax_h3_cache_latents.py:246`) | Needs `--allow_experimental_duration` below 5 s | Familiar, and has the best guidance-loss tooling. But first+last training vs first-only use is a train/inference mismatch, the class of bug we already know |
| SimpleTuner | Yes; 24/32/48/80 GB presets | Yes | 39 f preset | Alternative (`h3_drift` loss) |
| diffusion-pipe | T2V/T2I only | No | — | Not for I2V |

### 6.3 A Pudgy-adapted recipe

| Setting | Value | Why / source |
|---|---|---|
| Base | Pruned FL2VA, bf16 40 GB (or int8-ConvRot 21 GB; A100 has int8 tensor cores but **no FP8**) | V |
| Lengths | 22 / 39 / 56 frames, cut **before the first missing frame** (A6). Walking loop-tiled to 22 (v8 seam 0.62–1.04); 21-frame sources (happy, confused, two sitting) padded by one held frame; running excluded until Round 5 | V grid; §4.1 |
| Resolution | **768²**, or 768×1344 for 9:16 work; 512² only as an explicit off-distribution arm | §4.2 |
| Grounds and holdout | As v8: 4 flat grounds composited from alpha, 2 held out; `QF2_R` held out; written lossless | B9, B15; alvdansen "composite onto white", "encode losslessly" |
| Captions | Official three-field format + I2V line, "2D-animated" style word, colour words, `overall_soundscape: N/A` and `non_diegetic_music: N/A`; frozen between train and eval | §4.5 |
| LoRA | **Rank 32** on `qkv_proj, out_proj, fc1, fc2`, no AdaLN = **155.1M params** (our arithmetic matches the 155M reported in DiffSynth PR #1680) | V |
| Optimiser | adamw8bit, LR **5e-5 → test 1e-4**, cosine to 10%, accumulation 8 | v8 D4 lesson; DiffSynth and pablodawson used 1e-4 |
| Distillation | `--training_cfg_scale 4` in both stages, *or* the DeCFG adapter (A/B); never both untested | §6.1 |
| Timestep shift | **Unresolved:** DiffSynth trains at 2.22 ("empirical"; one user argues "at least 5"); musubi and ai-toolkit at the official 12/3 | Settle it with two short pilot LoRAs (2.22 vs 12) scored by Tier 1. Base-model likelihood cannot settle it: the per-timestep loss does not depend on the shift, only the weighting over timesteps does |
| Audio | `--audio_loss_weight 0` | §4.6 |
| Steps | Budget in **exposures per performance** (v8 ≈185 plateaued; ltx25 ≈500 memorised); save every 50; expect best ≤ 2,000 samples | v8, ltx25, #1585 |
| Inference | CFG 1, shift 12/3, ≥50 steps (the default 50-step grid has only one evaluation below σ = 0.2, which matters for thin lines), generate 22/39/56 and trim, strip audio, write at CRF ≤ 16 | V; musubi #1057 |

### 6.4 Compute and memory against v8

| | v8 (Wan2.2, measured) | H3 (estimate) | Basis |
|---|---|---|---|
| Forward FLOPs, 1024² clip | 4.59 PF (f57, 61,440 tokens) | 1.34 PF (f56, 19,944 tokens) → **0.29×** | I: header-derived params, token formula |
| Forward FLOPs, 768² / 512² clip | 1.82 / 0.57 PF | 0.63 / 0.25 PF → 0.35× / 0.44× | I |
| GPU memory, weights | 28.6 GB fp16 per expert; needed block-swap 20 (16 thrashed at 81.2 GB) | 66.3 GB full (≈78 GB at 124 f/480p: no headroom); **pruned 40.2 GB → ≈53 GB training**; pruned int8 21 GB | V: v8 report; R: DiffSynth #1624 |
| A100 time per clip | **39–43 s** (block-swap 20) | ≈4–10 s at 768², 22–56 f, with guidance loss | Anchor: **12.5 s/step at 768²×73 f** on 1× A100, pruned int8 + adapter + guidance loss (pablodawson card, R), scaled by tokens (I) |
| 4,800-clip budget | **57.5 h** | **≈6–14 h** | I |
| Text-encoder stage | UMT5 11.4 GB | Qwen3-VL-32B 51.5–66.7 GB. **Cannot sit on the GPU with the DiT**, so text and keyframe embeddings are cached in a separate pass (DiffSynth "sft:data_process"). Plan on ≥128 GB host RAM | V/R |
| Inference per clip | 6.4 min (v7 gates, 21 f); v2 ~14 min | Order of 1–3 min at 768²×56 f, 50 steps (anchor: A100 0.5 MP × 10 s × 20 steps = 3 min 40 s sampling, HF #35) | I |
| Disk | 65 GB of weights | ≈144 GB full or ≈42 GB pruned/quantised | V |

These H3 timings are **extrapolations, not measurements**. G0.6 must re-measure them on the box (C6: re-plan from the first hour's s/it).

### 6.5 Phase-0 gates before any training, in order

1. **VAE gate (G0.1):**
   - Pax and Polly round-tripped at 768² and 512². Inspect a **4× outline zoom by eye** plus positive-gradient edge metrics, the ltx25 §1 method, because metrics there picked the decoder that dissolves outlines.
   - Ground drift must stay **≤ 2/255** on flat grounds (v5/v8 bar) to rule out HF #78's colour shift.
   - Check the 17-frame-cadence brightness for #85.
   - **Stop if it fails.** This needs only the encoder and ViT decoder (2.6B params, 5.2 GB in fp16), so it can run on a CPU box or a small GPU.
2. **Zero-shot probe (G0.4):**
   - Unseen start frames, 3 seeds, 22/39/56 frames.
   - Measure the duplicated-frame rate (on twos?), 3D drift, outline fidelity, first-frame drift (frame 0 is *not* pinned), and motion energy against source (G0.5).
3. **Shift A/B:** two 100-step pilot LoRAs, one at shift 2.22 and one at 12, compared by Tier 1 on the held-out clips. Comparing the base model's likelihood under the two shifts would only compare the two timestep weightings.
4. **Memory and throughput probe (G0.6):** the longest bucket, with guidance loss on.

---

## 7. Pros and cons compared with v1–v9

### Pros

1. **Strongest public I2V preference of any base we have used.** LMArena I2V #1 (1495 vs Wan 2.2's 1170, ≈87% expected preference). AA I2V-silent top open-weight model (≈81% vs Wan 2.2, ≈89% vs Wan 2.1). AA's Animation & Gaming slice #2. *Caveat: general content, probably the hosted pipeline.*
2. **Much cheaper to train per clip than v8.** 2.2–3.4× fewer FLOPs at equal resolution (0.29–0.46×), and no block-swap with a pruned base. Estimated wall-clock is 4–10× lower: a v8-sized run is ≈6–14 h, not 57.5 h, and the 50-minute gate breaks shrink with it. *Estimate, not measured.*
3. **One transformer.** There is no high/low-noise expert split. v2, v5 and v9 had to train, route and merge two experts, and v7 → v8 needed merge algebra to undo double-counted goldens.
4. **Native 9:16 at its trained canvas (768×1344).** That is the client's main format. v5–v8 trained Wan at off-canonical 1024² square, and v9b planned 480×848.
5. **Native 4–15 s single-pass output, multi-shot.** The client wants up to 30 s. The v-line produces ≤2.4 s clips, and v8's long-form scenes had to be stitched; two-character layout broke at ~0.8 s.
6. **Native first-only / last-only / first+last in one checkpoint.** v2 relied on FLF2V at inference; v9 plans a first/last fallback.
7. **Finer temporal latents:** ≈3.4 frames per latent vs Wan's 4. A running stride (~8 frames) gets ≈2.4 latent steps instead of 2 (v9 F1 names this as part of why fast motion is worst).
8. **A 32B vision-language encoder sees the start frame.** Plausibly this gives stronger identity and colour grounding than UMT5 text alone (the Polly-renders-blue problem, A10). *Hypothesis, untested.*
9. **No default negative prompt to fight the captions.** v8's −11.4% zoom-out came from Wan's "static" negative.
10. **Native audio and Ref2VA multi-reference** if the client later wants voiced shorts or prop references. Ref2VA is outside the current frame-control scope.
11. **Fast-moving tooling.** Five trainers had H3 support within a week of release (musubi's formal release came 2026-09-16), 126 adapters on HF, and existing 2D LoRAs: flat-cel "STUDIO 1939", Rough-2D-Cartoon, and a 2D cartoon character LoRA at 39 frames.

### Cons

1. **CFG-distilled weights only.** Every LoRA recipe is a workaround, and degradation is reported at 3k–5k steps even with them. The v-line trained plain flow-matching LoRAs for 600 × 8 and 7,888 steps without that failure mode.
2. **Our clips are mostly below its length floor.** 48.7% of v8's clips are under 22 frames. Walking, running, happy and confused have **no** legal-length clip, and only 5 of 18 sitting sources reach 22 frames before a missing frame. Trained range 4–15 s vs our 0.5–2.5 s. Wan took f13–f57 as delivered.
3. **More compressive VAE on the art it is weakest at.** It has 2.1–2.7× fewer latent values per pixel than Wan, and there are open reports of line-art fading, flat-background colour shift, fine-detail loss and 17-frame-cadence dips. Wan's VAE showed no outline problems in v2–v8.
4. **Style priors against the brief:** anime animated on twos (~12 fps) vs the client's true 24 fps on ones; 2D output "leaning 3D", the v4 claymation risk again.
5. **Resolution off-distribution at our usual sizes.** 512² is below the 768 short edge, and 1024² is over the area cap. Only 768² or 768×1344 are on-canvas.
6. **Prompt dependence on a closed component.** Context-IR is "critical to quality" and is not released. The format is long, structured and "extremely sensitive", and the text tokens add sequence cost. Our caption grammar and its frozen train/eval wording have to be rebuilt.
7. **Heavier stack:** ≈144 GB of weights (vs 65 GB), a 52–67 GB text encoder that cannot share the GPU with the DiT, ≥128 GB host RAM recommended, no FP8 on A100, and audio generated whether wanted or not.
8. **Immature, inconsistent tooling.**
   - Two and a half months old, with no technical report.
   - musubi support is experimental and trains first+last only.
   - Trainers disagree on the shift (2.22 vs 12).
   - Weight-layout traps: fused-QKV interleave, fc1 swap, LoRAs for pruned vs full bases that won't load across.
   - No published A100 training numbers beyond one model card.
9. **Starts from zero.** No v-line LoRA, merged golden, eval calibration or Tier-1 baseline transfers; only knowledge does. The first H3 result cannot be read as "better than v8" until it is scored on v8's held-out clips with re-calibrated tiers.

---

## 8. Restrictions H3 would impose on our approach

1. **Frame grid 17n+5:** buckets become 22/39/56; the duration ladder and the "length must not predict the label" floor (B11) are re-designed around a 22-frame floor.
2. **No CFG and no negative prompt at inference:** every control has to come through the prompt, the keyframes or the LoRA.
3. **Distillation-aware training only:** guidance loss or adapter; shorter runs; early checkpoints; ~1.5× step cost.
4. **Two-stage pipeline:** cache text and keyframe embeddings with the 32B encoder first. A start frame cannot be swapped at evaluation without re-encoding it through Qwen3-VL.
5. **Frame 0 is not pinned.** Unlike Wan's concatenation, the output's first frame only approximates the supplied frame. A "first-frame fidelity" gate becomes necessary.
6. **Audio is always produced** and must be stripped. A silent corpus trains slightly off-distribution.

---

## 9. What would change this verdict

| If… | Then |
|---|---|
| The VAE gate fails on outlines or flat grounds | Stop. No LoRA recovers what the VAE discards (expv-style G0.1 rule) |
| Round 5 delivers ≥22-frame clips for every label | The data mismatch largely disappears; 39–56-frame buckets become possible |
| MiniMax releases un-distilled weights or official training code | The main training risk drops sharply |
| A public flat-2D, short-clip H3 I2V LoRA appears with numbers | It would replace much of Phase 0's uncertainty |

---

## 10. Sources

**H3 primary**
- Model card, LICENSE, `docs/QA-about-License.md`, configs and safetensors headers: https://huggingface.co/MiniMaxAI/MiniMax-H3 (snapshot 2026-10-11).
- Blog, 2026-07-31: https://www.minimax.io/blog/minimax-h3
- https://design.minimax.io/h3
- The Notion overview, read through Notion's public API; "Latest version — August 11, 2026".
- GitHub: https://github.com/MiniMax-AI/MiniMax-H3 issues #73 (line art), #75 (un-distilled weights), #85 (17-frame dips).

**Hugging Face discussions** (https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions)
- #12 (licence Q&A), #35 (A100 inference time), #65 (adherence vs resolution), #74 (anime at 12 fps), #78 (VAE colour shift), #88 and #97 (un-distilled weights).

**Trainers**
- DiffSynth-Studio issues #1585, #1557, #1583, #1624, #1735, PR #1650, PR #1680: https://github.com/modelscope/DiffSynth-Studio
- musubi-tuner PR #1045, releases v0.3.5/v0.3.6, `docs/minimax_h3.md`, issue #1057: https://github.com/kohya-ss/musubi-tuner
- ai-toolkit (ostris).
- The LoRA card for first-frame I2V on 1× A100: https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA
- diffusers `minimax_h3` docs and PR #14371; the SGLang cookbook (docs.sglang.io, MiniMax-H3); the vLLM recipe (recipes.vllm.ai/MiniMaxAI/MiniMax-H3).

**Leaderboards**
- https://lmarena.ai/leaderboard/image-to-video (2026-09-21 snapshot).
- https://artificialanalysis.ai/video/leaderboard/image-to-video (fetched 2026-10-11).

**This repo**
- `docs/training_reports/v5–v8/`
- `training_approach/v1–v9/`
- `ltx25/docs/FINDINGS_run1.md`
- `docs/documents/Client_Data_Request_Round5.md`
- `docs/documents/final_docs/Data_Requirements.md`

Raw downloads and the arithmetic scripts are in this session's scratchpad. They were not committed.

---

## Appendix A: licence notes (set aside 2026-10-11, not part of the verdict)

Kept so the facts are not lost if the question returns. The licence was dated 2026-08-02 and has not changed since; the HF and GitHub copies are byte-identical (V).

| Clause | Text (abridged) | Effect on this project |
|---|---|---|
| §I.3, §I.5 | Applicable Territory = worldwide **excluding the EU, UK, Republic of Korea and USA** | Igloo Inc. (parent of Pudgy Penguins) is **headquartered in Miami** (V: Dealroom; Refresh Miami 2024-07-23). The client sits in an excluded territory |
| §V.4, AUP #1 | No use, reproduction, modification, distribution or **display of the Works or their Outputs** outside the territory | Taken literally, delivering H3 clips to a US client for US channels (GIPHY, Instagram) is unauthorised. MiniMax staff said informally (HF #12) that creators outside excluded territories may distribute outputs worldwide. The literal question was asked on 2026-10-10 and is unanswered |
| §II, licence Q&A | Excluded-territory users "can apply for a formal license" (platform.minimax.io/h3-license) | The form asks for country, company and revenue over $20M, plus five undertakings: safeguards, flow-down, confidentiality, indemnity. Approvals reported 2026-08-13 and 08-31; no rejections reported (R) |
| §IV.1 | Separate written authorisation if commercial products exceed **$20M yearly revenue** | Pudgy's CEO projected ~$50M for 2025 (R, Decrypt). Whether this means company-wide revenue or H3-related revenue is unanswered (HF #12) |
| §IV.2 | "prominently display 'MiniMax H3'" on commercial product UI | Staff say it applies to software, not finished videos (R) |
| **§V.3** | **No using the Works or their Outputs to improve any other AI model** | **H3 clips can never be training or augmentation data for the Wan line.** That shuts the obvious "synthetic data for running" idea for v9 |
| §I.11 | A LoRA is a "Model Derivative" | Handing a LoRA to the client brings the licence, the NOTICE file and the territory rule with it |
| AUP #12 | Machine-generated content in public must be **clearly disclosed** | Every GIPHY or Instagram post would need an AI label |
| §VI.3, §IX | Licensee indemnifies MiniMax; Hong Kong law and courts | — |
| Context | The licensor, Nanonoble Pte. Ltd., is a defendant in **Disney/Universal/WBD v. MiniMax** (C.D. Cal., filed 2025-09-16; motions to dismiss denied) | Reputational and IP-risk question for a US consumer brand |

**Our own location.** Our location and the GPU host's are part of "use" too:
- Vast.ai boxes run in many countries, including the US. An H3 run would have to pin the host to a non-excluded country.
- The same goes for where weights are stored. The Azure region for `pudgytraining` is not recorded in the repo.

**Compare:** Wan is **Apache-2.0**, with no territory, no AI-disclosure duty, no revenue threshold and no ban on downstream training.

**Hosted API.** The MiniMax API is globally available under separate API terms ($0.08/s at 768p, 4 s minimum = $0.32/clip). But it takes no custom LoRA, it would send client IP to MiniMax, and it runs the hosted pipeline rather than the open weights. That makes it a probe at most, never the product.


Sources: Dealroom, Igloo Inc. (https://app.dealroom.co/companies/igloo_); Refresh Miami, 2024-07-23; Disney Enterprises et al. v. MiniMax et al., C.D. Cal., filed 2025-09-16; HF discussion #12.
