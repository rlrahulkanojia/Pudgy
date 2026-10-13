# Kandinsky 5.0 Video Pro — what it is, and what its code does

Read from source, not from the model card: `kandinskylab/kandinsky-5` at commit `3a74261`
(2026-01-19), the official LoRA trainer `kandinskylab/kandinsky-5-lora-train` (2025-11-25),
musubi-tuner tag **v0.3.4**, and the technical report (arXiv
[2511.14993](https://arxiv.org/abs/2511.14993), 80 pages). Nothing below has been run on our
box. Items marked **verify** are checked in Phase 0 ([`PLAN.md`](PLAN.md) §4).

---

## 1. The model

| | |
|---|---|
| Size | **19B** DiT: 60 blocks, model dim 4096, FFN 16384, head dim 128, 4 text-refiner blocks |
| Objective | flow matching (velocity = noise − x₀) |
| Text | **Qwen2.5-VL-7B** (last hidden layer, behind a fixed system-prompt template, **256-token cap**), refined by 4 text-only blocks, fed in by cross-attention; plus **CLIP ViT-L/14** pooled, added to the time embedding |
| VAE | **HunyuanVideo 3D VAE**: 8× spatial, 4× temporal, 16 channels. Kandinsky ships its own faster tiling code for it ([report §7.1](https://arxiv.org/abs/2511.14993)) |
| Patching / RoPE | 1×2×2 patches; 3D RoPE, rescaled per tier (`scale_factor` 1/2/2 at SD, 1/3.16/3.16 at HD) |
| Frame rate | **24 fps** native; latent frames = (n − 1)/4 + 1, so a clip has n = 4k + 1 frames; 5 s = 121 frames |
| Sampling | 50 steps, CFG 5 (= 100 forward passes); `scheduler_scale` (the timestep shift) 10 at SD, 5 at HD for I2V |
| Licence | **MIT** (repo and weights). The VAE is HunyuanVideo's, under Tencent's own licence (flagged as more restrictive in `docs/documents/research/Wan2.2_Evaluation.md`). **Check it before commercial use** |

**Released Pro checkpoints:** text-to-video SFT and pretrain, 5 s and 10 s each, and
**image-to-video SFT 5 s only**. The report describes distilled "Flash" and RL-tuned
versions; the repo TODO lists the Pro distilled checkpoint as **not released**.

**Two resolution tiers on the same weights:** SD (512-class: 512×512, 512×768, 768×512;
full attention) and HD (1024-class, including **1024×1024**; NABLA sparse attention).

**Vendor-reported cost** (one H100, offloading on): Pro 5 s at 512×768 takes 560 s and
47 GB; at 768×1280 it takes 1,241 s and 53 GB (report Table 5).

### How it was trained (report §4, §6)

- **Pretraining:** ~250M clips, scene-split. Scenes judged **too static or too dynamic were
  filtered out**. Captions came from a VLM.
- **Task mix:** text-to-image / text-to-video / image-to-video at **2% / 77% / 21%** for
  Pro.
- **Image-to-video regime:** *"The first frame of the video sequence remains unnoised and,
  unlike the other frames, is accompanied by a single-channel mask of ones."*
- **SFT:** ~2.8k hand-picked videos plus 45k images, split by a VLM into 9 domains (one is
  **cartoons**). One model was fine-tuned per domain and the nine were averaged with equal
  weights. The selection criteria favoured realism and "natural, non-AI appearance".

### What the vendor claims (their own side-by-side tests)

- Beats Wan2.2 A14B on **visual quality and motion dynamics**, in both text-to-video and
  image-to-video.
- **Weaker at prompt following** than Wan and Veo 3. The report blames the 256-token text
  encoder (§11).

That last point matters here: every axis we control (label, character, shot size, ground)
is controlled through the caption.

---

## 2. Gotchas found in the code

Each would make a first run look like a model failure when it is a setup failure.

### 2.1 The default negative prompt is anti-Pudgy

`kandinsky/i2v_pipeline.py:125`, and `test.py:68` / `test_lora.py`:

```text
"Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
```

With CFG 5, this pushes every step away from 2D cartoons *and* away from stillness. It is
the mechanism behind the v8 zoom-out (Wan's negative contained "static" while every
caption said "static … shot"; [`REPORT_v8.md`](../../docs/training_reports/v8/REPORT_v8.md)
§8), only stronger. **Always pass an explicit Pudgy negative prompt.** musubi's generator
defaults to an empty negative (`kandinsky5_generate_video.py:30`).

### 2.2 Prompt rewriting is on by default

`i2v_pipeline.py:126` sets `expand_prompts=True`: Qwen2.5-VL looks at the start frame and
rewrites the prompt under the instruction *"Make prompt dynamic"*
(`models/text_embedders.py:37–44`). That breaks the fixed caption grammar the LoRA is
trained on. **Off everywhere.** musubi's generator does not call the rewriter (**verify**).

### 2.3 Both LoRA trainers set up the first frame differently from inference

| Path | Where the clean first-frame latent goes | `visual_cond` channels | mask |
|---|---|---|---|
| **Official inference** (`generation_utils.py:159–168`) | **overwrites the noisy input** at frame 0, every step | all zeros | 1 at frame 0 |
| **musubi inference** (`kandinsky5/generation_utils.py:369–383`) | overwrites the noisy input at frame 0 | **first frame** (corrected 2026-10-12: the code also writes `visual_cond[0]`; G0.2 dump confirms). Off-distribution: see §2.12 | 1 at frame 0 |
| **Official LoRA trainer** (`train/train_utils.py:398–416`) | into the **`visual_cond` channels**; the noisy input stays noisy | first frame | 1 at frame 0 |
| **musubi trainer** (`kandinsky5_train_network.py:796–840`) | into the **`visual_cond` channels** | see §2.4 | see §2.4 |

The report's description of image-to-video pretraining matches the inference rows. So a
LoRA trained with either stock trainer learns under a different input layout from the one it
is used with. expv1 patches the trainer to the inference layout and **leaves frame 0 out of
the loss** (its target would be meaningless: the model is handed a clean latent but asked
for a noise velocity). G0.2 dumps the tensors, and G0.3 asks the base checkpoint which
layout it was trained on (it should explain held-out clips better under that layout).

### 2.4 musubi's first-frame mode also feeds in the last frame

- `kandinsky5_cache_latents.py:78–80` always caches **[first, last]** frame latents.
- In the default `--i2v_mode first`, the trainer pads that pair to the clip length:
  position 0 = first frame, **position 1 = last frame**, the rest zeros.
- It then marks frames with `torch.rand(F) < visual_cond_prob`. `visual_cond_prob` is fixed
  at **1.0** (line 109; there is no command-line flag), so **every** frame's mask is 1.

As read, a "first frame only" run shows the model the clip's last frame at latent index 1
and flags all frames as given. The same code is on musubi `main` (checked 2026-10-07).
**Verified** by the G0.2 dump (2026-10-12): last frame at latent index 1, mask 1 on every frame,
frame 0 in the loss. The expv1 patch replaces this path (`--i2v_layout A|A1|B`).

### 2.5 The official LoRA trainer encodes captions with the image template

`encode/encode_captions.py` calls `text_embedder.encode([caption])`, whose default is
`type_of_content="image"` (`models/text_embedders.py:208`). Inference uses the **video**
template, which has a different system prompt and crop offset (41 vs 129 tokens). musubi
caches with the video template (`kandinsky5_cache_text_encoder_outputs.py:45`), which is
one more reason to use musubi.

### 2.6 The official LoRA trainer only covers the 2B model

It ships only `configs/model/dit_2b_video.yaml` (Lite), hard-codes 512, and defaults to
8 GPUs with FSDP and a 1,000-step warmup. **musubi v0.3.4 already supports Pro**
(`--task k5-pro-i2v-5s-sd`, `networks.lora_kandinsky`), and it is the trainer this repo
pins. Its Kandinsky support is labelled *experimental*, and only SD tasks are defined; HD
needs `--force_nabla_attention` and `--nabla_resize`.

### 2.7 The start frame's size is chosen by the pipeline, not by you

Both pipelines rescale the start frame to a pixel budget (SD: 512×768 = 0.39 MP;
HD: 1024×1024). The output size follows the start frame:

- The **official** pipeline also *upscales*: a square frame becomes **624×624** at SD,
  which is not one of the SD training sizes.
- **musubi** never upscales (`k = min(1, …)`), so a 512×512 start frame stays 512×512.

**Pre-size every start frame to exactly 512×512, and assert the output size.**

### 2.8 Kandinsky's VAE drops frames unless the clip is 8k+1 (measured)

`AutoencoderKLHunyuanVideo.encode` (kandinsky-5 @ 3a74261) picks its tiling with
`get_enc_optimal_tiling`, which returns 17-frame temporal tiles at **stride 8**. Above 17
frames, only lengths of the form **8k+1** come back whole. The others are cut short with no
error. Measured on the A100 box (2026-10-12), frames in → latent frames → frames out:

| in | 13 | 17 | **21** | 25 | **29** | 33 | **37** | 41 | 49 | 57 | 121 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| latent | 4 | 5 | **3** | 7 | **7** | 9 | **9** | 11 | 13 | 15 | 31 |
| out | 13 | 17 | **9** | 25 | **25** | 33 | **33** | 41 | 49 | 57 | 121 |

**Effect on the plan:** PLAN §5.1 uses 4k+1 lengths, and §7.4's `core` suite runs at
**f21**. Any clip that goes through this encode path, whether a VAE gate clip, a Tier-1
clip encoded with Kandinsky's code, or a start-and-end anchor, must be 13, 17 or 8k+1.
musubi caches latents with its own VAE code, so check its behaviour separately (G0.2).
`eval/vae_gate.py` uses `k5_len()` and asserts that the frame count out equals the count in.
**musubi's cacher has the same truncation** when `--vae` is the VAE *directory* (Kandinsky's
class): a 21-frame clip was cached as 3 latent frames. The expv1 patch makes the cacher refuse
any clip whose latent count is wrong (G0.2).

### 2.9 musubi's `--vae` path picks the VAE implementation

`kandinsky5/models/vae.py:build_vae`: a **file** path loads HunyuanVideo's own VAE code
(`hunyuan_model.vae.load_vae`); a **directory** loads musubi's copy of Kandinsky's class. They
differ: frame-0 latents 4.2% apart on the same clip. The directory form matches Kandinsky's
own encode to 0.5% (bf16). **Pass the directory** (`.../HunyuanVideo`), not the
`.safetensors` file musubi's docs show.

### 2.10 Training saw a different first-frame latent from the one inference gives

musubi caches the first-frame latent as frame 0 of the **whole-clip** encode. Both generators
encode the start image **on its own** (`opt_tiling=False`). Even in Kandinsky's own fp16 code
those differ by **4.4%** (sampling noise is 0.07%). The expv1 cacher flag
`--first_frame_image_encode` encodes frame 0 as the generator does: 0.08% from Kandinsky's
image encode (G0.2).

### 2.11 musubi's generator `--frames` counts latent frames

`kandinsky5_generate_video.py --frames 17` produced a **65-frame** video. Video frames =
4 × (latent − 1) + 1, so a 17-frame clip is `--frames 5`, 25 frames is `--frames 7`. Assert the
output frame count on every generation.

### 2.12 The checkpoint expects layout A, and musubi's generator feeds it A1 (measured)

G0.3 (2026-10-13) scored the base model's flow-matching loss on the 26 held-out `QF2_R` clips.
It used the true caption only, frames ≥ 1 with the change mask, t ∈ {0.65, 0.85}, and k = 2
noise draws shared across layouts.

| Layout | Mean loss | vs A |
|---|---|---|
| **A** (official inference: clean latent in the noisy input, `visual_cond` zero) | **0.0342** | — |
| A1 (musubi generator: A + `visual_cond[0]`) | 0.0395 | +15.6%, worse on 26/26 |
| B (`visual_cond[0]` only) | 0.0423 | +23.6%, worse on 26/26 |
| stock musubi trainer (§2.4) | 0.2495 | +630%, worse on 26/26 |

All three Wilcoxon p = 3·10⁻⁸. This matches the report's description (§1) and the official
pipeline (§2.3).

- **musubi's generator is off-distribution for this checkpoint.** It writes the first frame into
  `visual_cond` as well (§2.3).
- **The expv1 patch adds `--i2v_layout A`** to `kandinsky5_generate_video.py` and to
  `generate_sample_latents_only`. G0.2's `gen_A` check shows that the DiT input then matches
  trainer layout A.
- **expv1 trains with `--i2v_layout A`, generates with `--i2v_layout A`, and scores Tier 1 in A.**

Text length: musubi's text cacher defaults to a Qwen `max_length` of 512, and the generator
uses the task's 256. With right padding and a causal encoder, the valid-token embeddings are
the same for any caption under 256 tokens. Keep captions ≤ 200 tokens (PLAN §5.5).

---

## 3. What already fits the footage, and what does not

| Fits | Does not |
|---|---|
| 24 fps native, which is the client's frame rate (Wan was 16 fps native) | Admitted weakness at prompt following, and the prompt is our only control |
| 1:1 is a supported size at both tiers | No expert split: Wan-line expert routing does not apply |
| A cartoons domain in fine-tuning | Static scenes filtered out of pretraining; fine-tuning favoured realism |
| An 8×/4×/16-channel VAE, the same class that round-tripped this art at ~38 dB on CogVideoX | This specific VAE has **never been measured** on Pudgy art |
| musubi support in the pinned version | Only first-frame I2V is official; the first+last mode is experimental and musubi-only |
