# MiniMax H3 — the facts this plan relies on, and the gotchas

The long-form evaluation is
[`../../docs/documents/research/MiniMax_H3_Evaluation.md`](../../docs/documents/research/MiniMax_H3_Evaluation.md)
(cited as **Eval §n**). This file keeps only what the plan depends on. Labels: **V** = read in code,
config, weight header or primary doc; **R** = reported by someone, not reproduced; **I** = our
inference. Nothing here has been run.

---

## 1. Facts

| | Value | Label, source |
|---|---|---|
| Transformer | 33.123B params, dense single-stream, 50 blocks, hidden 5376, 56×128 heads, FFN 14336. 13.010B of it is AdaLN modulation | V, Eval §1 |
| Build used here | `Comfy-Org/MiniMax-H3: diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`, 40.2 GB. The AdaLN MLP is replaced by a timestep lookup table | V file name and size; R behaviour, Eval §1 |
| Text encoder | Qwen3-VL-32B (layer 50 hidden states), 66.7 GB bf16. The **start frame also goes through it** as vision tokens | V, Eval §1 |
| Video VAE | f16t4d24, causal conv encoder + 36-layer ViT decoder, 2.6B params fp32. 17-frame chunks → 5 latents | V, Eval §1 |
| Tokens | patch 1×2×2 → one token per 32×32 px. 768² = 576 tokens per latent frame | V |
| Frame grid | 17n+5 frames → 5n+2 latents: **22 → 7, 39 → 12, 56 → 17** | V, DiffSynth doc |
| First-frame conditioning | keyframe VAE-encoded into extra rows held at t = 0.999 with frame-0 RoPE; the generated frame 0 is **not** overwritten | V (diffusers code, agent read), Eval §1 |
| Guidance | CFG-distilled: one forward per step; DiffSynth defaults `cfg_scale` 1.0 and `negative_prompt` `" "` and says the negative "takes no effect" | V, DiffSynth doc l.98, l.105 |
| Inference schedule | `flow_shift` 12 (video), `audio_flow_shift` 3, two independent sigma schedules, 50 steps default | V, DiffSynth doc l.106–107 |
| Training (DiffSynth) | `examples/minimax_h3/model_training/train.py`; stage 1 `--task sft:data_process` (text encoder loaded), stage 2 `--task sft:train` (DiT only). LoRA default targets `qkv_proj,out_proj`, rank 32 | V, DiffSynth doc l.229 |
| H3-specific flags | `--training_cfg_scale` (> 1 adds a no-grad unconditional branch), `--audio_loss_weight` (0 = video term only, audio still noised and forwarded), `--silent_on_missing_audio`, `--processor_path`, `--initialize_model_on_cpu` | V, DiffSynth doc l.217–221 |
| FL2VA training inputs | `--extra_inputs "input_image,end_image"` takes the **first and last** frame of each training video; no extra metadata column | V, DiffSynth doc l.233 |
| DeCFG adapter | `DiffSynth-Studio/MiniMax-H3-TrainingAdapter`: `model_for_comfy_dit.safetensors` for the Comfy (pruned) layout, passed as `--preset_lora_path … --preset_lora_model dit`. "Training only — do not load it at inference" | V, DiffSynth FP8-pruned example script |
| Prompt (I2V) | `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` then `integrated_multimodal_description: [Shot 1] …`, `overall_soundscape: …`, `non_diegetic_music: …` | V, model card I2V example |
| Output | 4–15 s officially, 24 fps, short edge 768 default, 32 kHz stereo audio on every call | V, Eval §1 |

---

## 2. Gotchas — each one would silently corrupt a first run

| # | Gotcha | Consequence if missed | What the plan does |
|---|---|---|---|
| 2.1 | **Only CFG-distilled weights exist.** Plain flow-matching LoRA training "regresses to its pre-distilled state": washed-out output, lost prompt adherence. Degradation is reported at 3k–5k steps even with countermeasures (Eval §6.1) | A run that looks fine on loss produces foggy, off-prompt clips | Guidance loss `--training_cfg_scale 4` in arm A; DeCFG adapter as pre-declared arm D; wash-out bar V11; short budget, early checkpoints |
| 2.2 | **Frame 0 is not pinned.** The output's first frame only approximates the supplied frame (Wan pins it by concatenation) | "On-model" start frames drift, and every metric that uses frame 0 as reference is off | Bar V9 (first-frame fidelity); Tier 1 excludes the frame-0 latent; motion metrics reference the *supplied* frame |
| 2.3 | **The pipeline snaps `num_frames` up to the next 17n+5**, and retake pads a short source by repeating its last frame (V, DiffSynth doc l.101, l.134). Whether the training loader pads a short file the same way is not documented | A 21-frame file could train as 22 frames with a silent duplicate: the A6 bug class again | Prep writes files of exactly 22/39/56 frames and asserts it; G0.2 hashes the frames the loader actually reads |
| 2.4 | **The FL2VA training example trains first + last frame** (`input_image,end_image`). musubi's video FL2VA trains first + last only (Eval §6.2) | Train/inference mismatch: the LoRA learns to expect an end anchor we never give | `--extra_inputs input_image` only; G0.2 dumps the conditioning rows in training and inference and asserts no end-frame rows |
| 2.5 | **DiffSynth trains at shift 2.22 for both modalities and infers at 12/3.** The maintainers chose 2.22 empirically because the training shift was not released (V, DiffSynth #1583). There is no documented flag | An untested hyperparameter that changes which noise levels the LoRA learns | Two 100-step pilots, 2.22 vs 12, Tier 1 decides (PLAN §6.3). The 12 arm needs `patches/diffsynth-h3-train-shift.patch` |
| 2.6 | **The pruned build uses ComfyUI's QKV layout**, and LoRAs and adapters are layout-specific (V, example script comment). Pruned AdaLN is a lookup over discrete timesteps (R) | A LoRA trained on one build will not load, or loads wrongly, on the other; a lookup table can differ from the MLP between grid points | Train, evaluate and score on the **same pruned build**; G0.2 parity test against the full bf16 DiT at three t values |
| 2.7 | **The DiT and Qwen3-VL cannot be resident on one GPU together** (V, DiffSynth doc l.229). Every new start frame must pass through Qwen3-VL | Naive per-clip evaluation reloads 107 GB of weights per clip (C4) | Two-pass evaluation: encode every (caption, start frame) pair first, then load the DiT once per checkpoint. G0.2 checks the two-pass output equals the one-pass pipeline's |
| 2.8 | **The start frame enters twice:** Qwen3-VL vision tokens *and* DiT in-context latents. ai-toolkit may skip the text path in training (I, Eval §6.2) | Train/inference mismatch on the identity signal | DiffSynth only; G0.2 asserts both paths are present in the stage-1 cache and at inference |
| 2.9 | **Audio is always generated and the audio stream is always forwarded.** Our clips are silent | Missing audio inputs can crash stage 1, or a garbage audio target leaks into the loss | `--silent_on_missing_audio`, `--audio_loss_weight 0`; G0.2 logs the audio loss term as exactly 0; audio stripped at export |
| 2.10 | **The prompt format is long, structured and "extremely sensitive"** (R). The Context-IR prompt rewriter that is "critical to the quality" is not released. Text tokens sit inside the DiT sequence | Our v8-style caption may be read as off-format; long captions cost sequence length | Official I2V line + three fields, our content in `integrated_multimodal_description`; one frozen template; ≤ 220 Qwen tokens asserted (PLAN §5.5) |
| 2.11 | **mp4 writers crush fine detail.** musubi found all its H3 writers at ~1 Mbps, "masquerading as generation artifacts" (V, musubi #1057) | The VAE gate and every Tier-2 metric would measure the encoder, not the model | All scoring reads PNG frames from memory or disk; mp4 only for viewing, CRF ≤ 16 |
| 2.12 | **Anime-style output is often animated on twos** (~12 fps effective; ~1 in 10 clips true 24 fps for the reporter, V, HF #74). The client's art is true 24 fps on ones (A2) | Clips look correct frame by frame but move at half rate | Cadence measured as the "twos index" against the source (G0.5), bar V10 |
| 2.13 | **H3's 2D output "leaned too heavily towards 3D"** (V, RunningHub LoRA card) | The v4 claymation basin again (~40% of Pax seeds there) | Manual rubric V2 checks shading; automated fill-flatness in V11; ≥ 3 seeds (B2) |
| 2.14 | **VAE reports against our art type:** line art fading (V, GitHub #73), colour shift on plain backgrounds (V, HF #78), fine background detail lost (V, DiffSynth #1585), brightness dips on a 17-frame cadence (V, #85, traced to a ComfyUI dtype bug, not the VAE) | Damage no LoRA can undo | G0.1 is a stop gate with a by-eye 4× check, a 2/255 ground-drift bar and a chunk-boundary brightness check |
| 2.15 | **DiffSynth's documented flags include no LR scheduler and no optimizer-state resume** (V for the documented list; the code is unchecked) | Constant LR at batch 1 gave mostly-noise updates on Wan (D4); stopping for gate breaks would lose the optimizer moments | G0.2 checks `--help` and the code. If absent, `patches/diffsynth-cosine-lr.patch` adds cosine decay; gate breaks are replaced by pilots + post-hoc checkpoint scoring (PLAN §6) |
| 2.16 | **What the guidance loss's unconditional branch drops is undocumented:** text only, or text and start frame? Sources also disagree on stacking the DeCFG adapter with guidance loss (Eval §6.1) | The loss would sharpen the image condition along with the text condition, or not, without us knowing | G0.2 records the unconditional branch's inputs. The adapter and guidance loss are never stacked: arm A uses the loss, arm D the adapter |

Not a gotcha for this plan, but worth knowing: diffusers' H3 pipeline refuses durations under
4–5 s (V). expv2 uses DiffSynth's pipeline for inference, which only snaps to 17n+5.
