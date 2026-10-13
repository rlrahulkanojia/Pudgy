# expv2-minimax-h3 — G0.1 report: the VAE gate failed, and expv2 ended there

**Status:** final, 2026-10-12 run, written up 2026-10-13. **expv2 stopped at Phase 0, gate G0.1
(PLAN §4), with a FAIL.** No stage-1 cache, no G0.2–G0.6, no LoRA and no generation ran. The
MiniMax H3 weights have since been deleted from the box at the user's request. Everything
below is read from the gate's own JSON files, logs and git history. Each section names its
sources, and no number was re-measured for this report.

Primary evidence (local box copy and Azure mirror; §10 lists them all):
`eval/phase0/vae_gate/report.json`, `eye_proxy.json`, `eye_proxy_centerline.json`, `sheets/`.

---

## 1. Verdict

| | |
|---|---|
| **Gate** | `report.json` → `"gate": "FAIL"`, `eye_check.result = "FAIL"` (recorded 2026-10-12T11:35:11) |
| **Metric bars** (ground drift, chunk dip, flat fill) | **all pass** (`verdict.metric_pass = true`) |
| **Outline/highlight bar** (PLAN G0.1 bar 1, decided on the pre-stated proxy rule; §5) | **fails**: H3 breaks thin strokes on the angry clips |
| **Key number** | H3 outline-centerline survival, minimum frame: **0.8613–0.8913** on the 4 angry clips vs **0.9996–0.9997** for Wan2.1 on the same clips. **13–16 broken stroke gaps (≥ 6 px) in the worst frame**, vs **0** for Wan2.1 |
| **What is lost** | eyebrows, crest strokes and the mouth line inside the beak (clearest at `polly_angry`, frame 18) |
| **What survives** | eye-highlight cores (0.9708–0.9793 kept), ground colour (drift ≤ 0.996/255), chunk boundaries, flat fills |
| **Is it the pipeline?** | No. The fp32 re-run is the same (centerline 0.86, 15 gaps). Wan2.2-TI2V-5B, also 16× spatial, keeps every stroke. **It is H3's decoder** (§6) |
| **Consequence** | PLAN §1: *"FAIL at V1: the VAE cannot represent the art. Stop; no LoRA recovers it."* V1 cannot pass, so §9 cannot reach PASS or PARTIAL. The §8 head-to-head and the Round 5 re-run (§12) are not triggered |

---

## 2. What was tested

### 2.1 Clips

Built by `prep/prep_expv2.py` into `Data/processed/expv2_h3/vae_gate/png{768,512}/`. Each clip
is a directory of PNGs; there is no mp4 anywhere in the gate (MODEL_NOTES 2.11).

| Clip id | Character | Action | Ground | Frames |
|---|---|---|---|---|
| `pax_angry_FRONT__{white,mint}_f39` | Pax | angry (expression) | white, mint | 39 |
| `pax_waving_FRONT__{white,mint}_f22` | Pax | waving (motion) | white, mint | 22 |
| `pax_walking_FRONT__{white,mint}_f22` | Pax | walking (16 frames loop-tiled to 22, PLAN §5.1) | white, mint | 22 |
| `polly_angry_FRONT__{white,mint}_f39` | Polly | angry | white, mint | 39 |
| `polly_waving_FRONT__{white,mint}_f22` | Polly | waving | white, mint | 22 |
| `polly_walking_FRONT__{white,mint}_f22` | Polly | walking (tiled) | white, mint | 22 |
| `flatfill_mint_f22` | — | constant mint fill, ground RGB (206, 235, 219) | mint | 22 |

That is 12 character clips plus 1 flat fill = **13 clips per size**. All frame counts are on
H3's 17n+5 grid (22 → 7 latents, 39 → 12 latents; recorded `latent_shape` `[1, 24, 7, 48, 48]`
and `[1, 24, 12, 48, 48]` at 768²).

### 2.2 Resolutions

- **768²**, the training size. Every bar is judged here.
- **512²**, for the record only.

### 2.3 VAEs

| Key | Model | Spatial ratio | Role | Frames seen | Wall time (load + 26 round trips) |
|---|---|---|---|---|---|
| `h3` | MiniMax H3 FL2VA `video_vae/source/model.safetensors` (HF `MiniMaxAI/MiniMax-H3` @ `42ed227`, sha256 `5f0c2e16…`) | 16× (f16t4d24) | under test | all 22 / 39 | 170 s |
| `wan21` | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` `vae/` (the Wan2.1 / Wan2.2-A14B VAE used by v1–v9) | 8× | same-clip reference for the eye bar | first 21 / 37 (Wan needs 4k+1) | 148 s |
| `wan22_5b` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` `vae/` | 16× | informational: is a failure about 16× or about H3? | first 21 / 37 | 100 s |

(Wall times are from `report.json` → `vae_status`.)

### 2.4 Code path

`eval/vae_gate.py` at repo commit `78ae368` (recorded in `report.json` → `git_sha`) with the
`4832f98` fixes applied (§7). DiffSynth-Studio `acf2ad29c65f…` plus the two expv2 patches, on
venv `/workspace/venv-expv2` (torch 2.11.0+cu128, diffusers 0.35.2, transformers 5.19.0,
1× A100-SXM4-80GB). The environment is recorded in `eval/phase0/env.json`.

- **H3:** DiffSynth's own `MiniMaxH3Pipeline`, VAE only (`h3lib.load_pipe(("vae",))`,
  `torch_dtype=bfloat16`). Encode is `video_vae.encode_video` (the trainer's
  `MiniMaxH3Unit_InputVideoEmbedder` path). Decode is `video_vae.decode_video(...,
  tiled=True, tile_size=256, tile_overlap=64)` (the sampler's path). Pre- and
  post-processing are `pipe.preprocess_video` / `pipe.vae_output_to_video`.
- **Wan:** diffusers `AutoencoderKLWan`, fp32. Slicing is on. Tiling is on for Wan2.1 and
  **off for Wan2.2-5B** (§7, bug 1). Encode uses `latent_dist.mode()`, so it is deterministic.
- Every decoded array is saved to `decoded/<vae>/<size>/<clip>.npz` (H3 at 768 also as PNGs in
  `decoded/h3_png768/`). Metrics are computed from those arrays against the source PNGs.

---

## 3. Metric definitions

### 3.1 Gate metrics (`eval/vae_gate.py`, written into `report.json`)

`src` and `dec` are `[F, H, W, 3]` uint8 arrays, truncated to the shorter of the two. The
first six are copied verbatim from `ltx25/eval/vae_gate_ltx25.py` (`measure()`).

| Metric | Definition | Direction |
|---|---|---|
| **PSNR** | `20·log10(255) − 10·log10(MSE)` over all pixels, channels and frames (99 dB if MSE ≤ 1e-12) | higher is better |
| **SSIM** | `skimage.metrics.structural_similarity(src[i], dec[i], channel_axis=2, data_range=255)`, mean over frames | higher is better |
| **edge-PSNR** | PSNR restricted to the **edge mask**: pixels whose Sobel-style gradient magnitude (central differences on the channel-mean grey image) is ≥ the 75th percentile of the positive gradients. These are the outlines themselves | higher is better |
| **edge cov** | fraction of pixels in the edge mask (sanity check: about 0.01–0.10 is expected). It is 0 on the flat fill, so edge-PSNR there is NaN (undefined) | — |
| **flat-band** | on pixels where the source gradient is exactly 0: mean \|horizontal neighbour difference\| of the decode minus that of the source, floored at 0 (grey levels). Measures banding or texture added to flat regions | lower is better |
| **shimmer** (`shimmer_abs`) | on pixels whose source changes by < 1 grey level between consecutive frames: mean frame-to-frame \|Δ\| of the decode minus that of the source, floored at 0 (grey levels). Measures temporal flicker on static areas | lower is better |
| **ground drift** | the ground colour is the modal RGB of frame 0's border. On every pixel that is exactly that colour in the source: mean \|dec − src\| per channel, in grey levels (/255). `ground_drift_max_frame` is the worst single frame | **bar ≤ 2.0** on every 768 clip |
| **chunk dip** | `e[i]` = per-frame mean-luminance error (dec − src). Take the largest \|Δe\| at a VAE chunk start (frames 17 and 34: H3 chunks 17 frames from frame 0, `minimax_h3_video_vae.py:439-443`) minus the largest \|Δe\| anywhere else. Negative means the chunk boundary is quieter than the rest of the clip. `chunk_dip_plan_frames` is the same at PLAN's original frames 5/22/39, kept for the record | **bar ≤ 1.0** on every 768 clip |
| **flat fill**: `fill_mae`, `fill_std`, `grid_ratio` | on `flatfill_*`: mean \|dec − fill colour\|, mean spatial std per frame and channel, and the FFT energy on the 16-px latent lattice divided by all AC energy (luminance, DC removed) | **bars**: `fill_std ≤ 1.0`, `fill_mae ≤ 2.0`, `flat_band ≤ 0.5`; `grid_ratio` recorded |

**Means.** In `report.json` → `means`, PSNR, SSIM, edge-PSNR, flat-band and shimmer average
**all 13 clips, including the flat fill** (edge-PSNR skips the NaN). Ground drift averages
the 12 character clips only. That is why the H3 768² PSNR mean is 34.63 dB with the flat fill
(43.19 dB) and 33.91 dB without it. §4.3 gives both.

### 3.2 Eye-bar proxies (the deciding measurements)

PLAN G0.1 bar 1 is an eye judgement: "outlines and highlights intact at 768² by eye, judged
against the Wan2.1 reference". At the user's request it was decided on a rule stated before
the numbers were read (§5). Two scripts measured it on the 768² decodes. They ran from the
session scratchpad (now checked in as `eval/eye_proxy.py`, `eval/eye_proxy_centerline.py`); their definitions follow exactly. Luminance is
`0.299R + 0.587G + 0.114B`.

**Pass 1, `eye_proxy.json`** (all three VAEs):

| Measure | Definition |
|---|---|
| `aligned`, `psnr_shift` | mean frame PSNR of `src[i]` vs `dec[i+k]` for k ∈ {−1, 0, +1}. Aligned if k = 0 is best, which rules out a frame-offset artefact |
| `outline_recall` | outline mask = source luminance < 64. Per frame, the fraction of those pixels whose decode luminance is < 128. Reported as the **worst frame** (`outline_recall`) and the mean |
| `highlight_recall` | highlight mask = connected components of source luminance > 230 with 4–600 px, whose 2-px surrounding ring is > 60% dark (< 80). These are eye highlights. Per frame, the fraction whose decode luminance is > 200. Worst frame. `highlight_px_per_frame` is the mask size (0 on the waving clips: no highlight passed the filter) |
| `sharpness_retention` | on source pixels with gradient magnitude > 20: mean decode gradient / mean source gradient, averaged over frames |

**Pass 2, `eye_proxy_centerline.json`** (H3 and Wan2.1). Pass 2 refines pass 1 so that
anti-aliased stroke rims and highlight rims do not count against a VAE:

| Measure | Definition |
|---|---|
| `centerline` | skeletonise the source outline mask (luminance < 64; `skimage.morphology.skeletonize`). Per frame, the fraction of skeleton pixels whose decode luminance is < 128. This asks whether the **middle of each stroke** is still dark. `centerline_min` is the worst frame and `worst_frame` its index |
| `broken_strokes_max_per_frame` | skeleton pixels lost in the decode, grouped into 8-connected components. Count the components of **≥ 6 px**: each one is a visible gap in a line. Worst frame |
| `highlight_core_min` | the pass-1 highlight mask eroded by 1 px (the core, without its rim). Fraction of core pixels with decode luminance > 200. Worst frame |

---

## 4. Results

### 4.1 Headline means (`report.json` → `means`; 13 clips, ground drift over 12)

| VAE | Size | PSNR | SSIM | edge-PSNR | edge cov | flat-band | shimmer | ground drift |
|---|---|---|---|---|---|---|---|---|
| **H3** | 768 | **34.63** | 0.9880 | **17.99** | 0.0133 | 0.186 | 0.246 | 0.549 |
| H3 | 512 | 32.46 | 0.9825 | 17.07 | 0.0176 | 0.207 | 0.289 | 0.460 |
| Wan2.1 | 768 | 43.31 | 0.9981 | 25.24 | 0.0133 | 0.037 | 0.103 | 0.078 |
| Wan2.1 | 512 | 41.60 | 0.9972 | 24.53 | 0.0176 | 0.054 | 0.126 | 0.102 |
| Wan2.2-5B | 768 | 47.95 | 0.9990 | 28.29 | 0.0133 | 0.030 | 0.047 | 0.026 |
| Wan2.2-5B | 512 | 46.07 | 0.9985 | 27.55 | 0.0176 | 0.042 | 0.063 | 0.041 |

At 768², H3 is **8.7 dB below Wan2.1** in PSNR and **7.2 dB below it on the outlines**
(edge-PSNR). The Wan2.2-5B VAE, at the same 16× spatial ratio as H3, is **13.3 dB above H3**
and above Wan2.1 on every metric. H3 is also worse at 512² than at 768² (−2.2 dB PSNR), so
smaller frames do not help.

### 4.2 Character clips only (12 clips): min – max, mean

| VAE | Size | PSNR | SSIM | edge-PSNR | flat-band | shimmer | ground drift | chunk dip |
|---|---|---|---|---|---|---|---|---|
| **H3** | 768 | 33.33–34.65, **33.91** | 0.9842–0.9899, 0.9871 | 17.18–18.94, 17.99 | 0.126–0.252, 0.192 | 0.170–0.336, 0.258 | 0.146–0.996, 0.549 | −0.321 – +0.055 |
| H3 | 512 | 30.90–32.55, 31.57 | 0.9776–0.9848, 0.9811 | 16.27–18.14, 17.07 | 0.147–0.282, 0.216 | 0.204–0.394, 0.304 | 0.184–0.781, 0.460 | −0.404 – +0.086 |
| Wan2.1 | 768 | 40.74–42.68, 41.90 | 0.9973–0.9987, 0.9980 | 24.49–26.05, 25.24 | 0.026–0.053, 0.039 | 0.063–0.173, 0.107 | 0.021–0.176, 0.078 | −0.252 – −0.038 |
| Wan2.1 | 512 | 38.30–41.11, 40.05 | 0.9959–0.9980, 0.9970 | 23.11–25.39, 24.53 | 0.040–0.075, 0.057 | 0.080–0.206, 0.131 | 0.038–0.200, 0.102 | −0.287 – −0.036 |
| Wan2.2-5B | 768 | 44.02–46.55, 45.40 | 0.9987–0.9993, 0.9990 | 27.15–29.44, 28.29 | 0.025–0.041, 0.032 | 0.037–0.070, 0.051 | 0.013–0.040, 0.026 | −0.051 – −0.012 |
| Wan2.2-5B | 512 | 41.49–44.61, 43.33 | 0.9978–0.9988, 0.9983 | 25.90–28.70, 27.55 | 0.035–0.059, 0.046 | 0.050–0.094, 0.068 | 0.022–0.060, 0.041 | −0.062 – −0.002 |

**By ground (768², 6 clips each, means).** For H3 the mint ground is clearly worse than white
on every flat-area measure. The gate's ground bar still passes with about 2× margin.

| VAE | Ground | PSNR | ground drift | flat-band | shimmer |
|---|---|---|---|---|---|
| H3 | mint | 34.18 | **0.933** | 0.247 | 0.314 |
| H3 | white | 33.65 | 0.165 | 0.136 | 0.202 |
| Wan2.1 | mint | 42.04 | 0.131 | 0.047 | 0.133 |
| Wan2.1 | white | 41.76 | 0.026 | 0.031 | 0.081 |
| Wan2.2-5B | mint | 45.57 | 0.036 | 0.036 | 0.055 |
| Wan2.2-5B | white | 45.23 | 0.017 | 0.029 | 0.047 |

**By action (768², 4 clips each, means of PSNR / edge-PSNR).** H3: angry 33.72 / 17.91,
walking 33.67 / 17.81, waving 34.35 / 18.26. Wan2.1: angry 41.20 / 24.78, walking 42.04 / 25.29,
waving 42.47 / 25.65. Wan2.2-5B: angry 44.45 / 27.43, walking 45.45 / 28.32, waving
46.29 / 29.13. The angry clips are not much worse in PSNR. They fail on thin strokes (§4.5),
which PSNR averages away.

### 4.3 Per clip, 768² (the gate size)

Clip ids are shortened: `pax_angry/mint_f39` = `pax_angry_FRONT__mint_f39`. "frames" gives the
frames H3 / Wan were compared on.

| Clip | frames (H3 / Wan) | PSNR H3 | PSNR Wan2.1 | PSNR Wan2.2-5B | SSIM H3 | SSIM Wan2.1 | SSIM 5B | edge-PSNR H3 | edge-PSNR Wan2.1 | edge-PSNR 5B | edge cov |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `flatfill/mint_f22` | 22 / 21 | 43.19 | 60.24 | 78.62 | 0.9987 | 0.9997 | 1.0000 | n/a | n/a | n/a | 0.0000 |
| `pax_angry/mint_f39` | 39 / 37 | 33.75 | 41.00 | 44.36 | 0.9857 | 0.9974 | 0.9987 | 18.36 | 24.85 | 27.50 | 0.0144 |
| `pax_angry/white_f39` | 39 / 37 | 33.44 | 40.74 | 44.02 | 0.9868 | 0.9977 | 0.9988 | 17.86 | 24.49 | 27.15 | 0.0144 |
| `pax_walking/mint_f22` | 22 / 21 | 33.93 | 42.22 | 45.59 | 0.9842 | 0.9973 | 0.9987 | 18.39 | 25.67 | 28.58 | 0.0145 |
| `pax_walking/white_f22` | 22 / 21 | 33.39 | 41.91 | 45.20 | 0.9851 | 0.9976 | 0.9988 | 17.61 | 25.21 | 28.10 | 0.0145 |
| `pax_waving/mint_f22` | 22 / 21 | 34.62 | 42.68 | 46.43 | 0.9857 | 0.9978 | 0.9989 | 18.94 | 26.05 | 29.44 | 0.0143 |
| `pax_waving/white_f22` | 22 / 21 | 34.17 | 42.27 | 45.97 | 0.9868 | 0.9980 | 0.9990 | 18.22 | 25.52 | 28.85 | 0.0143 |
| `polly_angry/mint_f39` | 39 / 37 | 34.07 | 41.62 | 44.83 | 0.9880 | 0.9980 | 0.9989 | 18.01 | 25.03 | 27.67 | 0.0147 |
| `polly_angry/white_f39` | 39 / 37 | 33.61 | 41.44 | 44.59 | 0.9894 | 0.9984 | 0.9991 | 17.41 | 24.73 | 27.39 | 0.0147 |
| `polly_walking/mint_f22` | 22 / 21 | 34.04 | 42.13 | 45.65 | 0.9871 | 0.9980 | 0.9990 | 18.06 | 25.32 | 28.48 | 0.0145 |
| `polly_walking/white_f22` | 22 / 21 | 33.33 | 41.89 | 45.36 | 0.9882 | 0.9982 | 0.9990 | 17.18 | 24.95 | 28.11 | 0.0145 |
| `polly_waving/mint_f22` | 22 / 21 | 34.65 | 42.60 | 46.55 | 0.9886 | 0.9984 | 0.9992 | 18.36 | 25.69 | 29.34 | 0.0142 |
| `polly_waving/white_f22` | 22 / 21 | 33.96 | 42.33 | 46.21 | 0.9899 | 0.9987 | 0.9993 | 17.50 | 25.32 | 28.87 | 0.0142 |

| Clip | flat-band H3 | Wan2.1 | 5B | shimmer H3 | Wan2.1 | 5B | ground drift H3 | Wan2.1 | 5B | drift max-frame H3 | chunk dip H3 | chunk dip (PLAN frames) H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flatfill/mint_f22` | 0.117 | 0.014 | 0.001 | 0.108 | 0.053 | 0.001 | 1.663 | 0.061 | 0.001 | 1.786 | −0.026 | −0.130 |
| `pax_angry/mint_f39` | 0.248 | 0.053 | 0.041 | 0.336 | 0.173 | 0.070 | 0.950 | 0.176 | 0.040 | 1.030 | −0.150 | −0.207 |
| `pax_angry/white_f39` | 0.142 | 0.034 | 0.032 | 0.230 | 0.108 | 0.062 | 0.171 | 0.032 | 0.025 | 0.315 | −0.113 | −0.211 |
| `pax_walking/mint_f22` | 0.247 | 0.049 | 0.036 | 0.318 | 0.128 | 0.052 | 0.939 | 0.109 | 0.032 | 1.006 | 0.008 | −0.021 |
| `pax_walking/white_f22` | 0.143 | 0.035 | 0.030 | 0.212 | 0.084 | 0.045 | 0.187 | 0.030 | 0.017 | 0.225 | 0.055 | −0.093 |
| `pax_waving/mint_f22` | 0.244 | 0.045 | 0.033 | 0.295 | 0.118 | 0.047 | 0.892 | 0.112 | 0.037 | 0.948 | −0.088 | −0.047 |
| `pax_waving/white_f22` | 0.134 | 0.031 | 0.027 | 0.183 | 0.074 | 0.039 | 0.170 | 0.029 | 0.018 | 0.214 | −0.001 | −0.075 |
| `polly_angry/mint_f39` | 0.250 | 0.048 | 0.038 | 0.326 | 0.151 | 0.064 | 0.937 | 0.165 | 0.039 | 1.015 | −0.321 | −0.345 |
| `polly_angry/white_f39` | 0.136 | 0.029 | 0.029 | 0.215 | 0.086 | 0.055 | 0.154 | 0.023 | 0.017 | 0.221 | −0.244 | −0.284 |
| `polly_walking/mint_f22` | 0.252 | 0.045 | 0.034 | 0.318 | 0.121 | 0.050 | 0.996 | 0.110 | 0.030 | 1.052 | −0.034 | −0.020 |
| `polly_walking/white_f22` | 0.137 | 0.030 | 0.028 | 0.201 | 0.073 | 0.043 | 0.161 | 0.023 | 0.013 | 0.197 | 0.051 | −0.100 |
| `polly_waving/mint_f22` | 0.244 | 0.041 | 0.031 | 0.290 | 0.109 | 0.045 | 0.883 | 0.111 | 0.035 | 0.955 | −0.116 | −0.059 |
| `polly_waving/white_f22` | 0.126 | 0.026 | 0.025 | 0.170 | 0.063 | 0.037 | 0.146 | 0.021 | 0.013 | 0.178 | 0.002 | −0.041 |

The chunk-dip and ground-drift columns are computed for every VAE by the same function. Only
H3's values are gated, and the Wan columns are context.

**Flat fill, all VAEs (768²).** H3: `fill_mae` 1.663, `fill_std` 0.288, flat-band 0.117,
`grid_ratio` 0.2654. Wan2.1: 0.061 / 0.190 / 0.014 / 0.0159. Wan2.2-5B: 0.001 / 0.021 / 0.001 /
0.1013. H3 darkens the flat mint by a near-constant luminance error of −1.61 to −1.88 grey
levels on every frame (`lum_err_per_frame`). That uses 83% of the `fill_mae ≤ 2.0` bar. It
passes, but only just.

### 4.4 Per clip, 512² (for the record)

| Clip | PSNR H3 | Wan2.1 | 5B | SSIM H3 | Wan2.1 | 5B | edge-PSNR H3 | Wan2.1 | 5B | flat-band H3 | shimmer H3 | ground drift H3 | chunk dip H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flatfill/mint_f22` | 43.17 | 60.14 | 78.92 | 0.9989 | 0.9997 | 1.0000 | n/a | n/a | n/a | 0.103 | 0.109 | 1.670 | −0.021 |
| `pax_angry/mint_f39` | 31.49 | 38.52 | 41.77 | 0.9791 | 0.9959 | 0.9978 | 17.38 | 23.37 | 26.19 | 0.273 | 0.394 | 0.776 | −0.215 |
| `pax_angry/white_f39` | 31.07 | 38.30 | 41.49 | 0.9801 | 0.9964 | 0.9979 | 16.86 | 23.11 | 25.90 | 0.175 | 0.297 | 0.241 | −0.140 |
| `pax_walking/mint_f22` | 31.70 | 40.44 | 43.72 | 0.9776 | 0.9961 | 0.9981 | 17.41 | 25.05 | 28.08 | 0.269 | 0.362 | 0.781 | −0.076 |
| `pax_walking/white_f22` | 31.14 | 40.19 | 43.47 | 0.9783 | 0.9965 | 0.9982 | 16.75 | 24.69 | 27.75 | 0.173 | 0.264 | 0.247 | 0.086 |
| `pax_waving/mint_f22` | 32.55 | 40.78 | 44.42 | 0.9799 | 0.9966 | 0.9983 | 18.14 | 25.23 | 28.69 | 0.259 | 0.324 | 0.753 | −0.129 |
| `pax_waving/white_f22` | 31.98 | 40.39 | 43.98 | 0.9808 | 0.9969 | 0.9984 | 17.45 | 24.76 | 28.18 | 0.158 | 0.223 | 0.225 | −0.018 |
| `polly_angry/mint_f39` | 31.51 | 39.44 | 42.61 | 0.9824 | 0.9971 | 0.9984 | 16.86 | 23.92 | 26.76 | 0.265 | 0.375 | 0.659 | −0.404 |
| `polly_angry/white_f39` | 30.94 | 39.32 | 42.30 | 0.9834 | 0.9976 | 0.9985 | 16.27 | 23.71 | 26.39 | 0.159 | 0.266 | 0.206 | −0.289 |
| `polly_walking/mint_f22` | 31.59 | 40.73 | 43.84 | 0.9813 | 0.9972 | 0.9985 | 17.03 | 25.17 | 28.06 | 0.282 | 0.370 | 0.610 | −0.120 |
| `polly_walking/white_f22` | 30.90 | 40.55 | 43.60 | 0.9824 | 0.9976 | 0.9986 | 16.35 | 24.86 | 27.73 | 0.161 | 0.244 | 0.201 | 0.037 |
| `polly_waving/mint_f22` | 32.33 | 41.11 | 44.61 | 0.9836 | 0.9977 | 0.9987 | 17.53 | 25.39 | 28.70 | 0.266 | 0.328 | 0.635 | −0.142 |
| `polly_waving/white_f22` | 31.58 | 40.88 | 44.14 | 0.9848 | 0.9980 | 0.9988 | 16.78 | 25.08 | 28.14 | 0.147 | 0.204 | 0.184 | −0.035 |

Flat fill at 512²: H3 `fill_mae` 1.670, `fill_std` 0.261, `grid_ratio` 0.4016. Wan2.1 0.063 /
0.195 / 0.0153. Wan2.2-5B 0.001 / 0.024 / 0.0171.

### 4.5 Outlines and highlights, 768² (the deciding measurements)

**Pass 1, `eye_proxy.json`.** All 36 clip × VAE pairs are aligned: shift 0 has the best PSNR
in every case, so there is no frame offset.

| Clip | outline recall min H3 | Wan2.1 | 5B | mean H3 | highlight recall min H3 | Wan2.1 | 5B | highlight px/frame | sharpness H3 | Wan2.1 | 5B | PSNR shift −1 / 0 / +1 (H3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pax_angry/mint_f39` | **0.9064** | 0.9985 | 0.9969 | 0.9894 | **0.8150** | 0.9855 | 0.9855 | 70 | 0.872 | 0.969 | 0.986 | 28.23 / **34.35** / 28.17 |
| `pax_angry/white_f39` | **0.8946** | 0.9988 | 0.9967 | 0.9865 | **0.8050** | 0.9855 | 0.9855 | 70 | 0.869 | 0.966 | 0.985 | 28.08 / **34.18** / 28.02 |
| `pax_walking/mint_f22` | 0.9886 | 0.9996 | 0.9999 | 0.9957 | **0.8235** | 0.9902 | 1.0000 | 99 | 0.852 | 0.975 | 0.989 | 23.16 / **33.90** / 22.60 |
| `pax_walking/white_f22` | 0.9815 | 0.9993 | 0.9999 | 0.9926 | **0.8173** | 0.9902 | 1.0000 | 99 | 0.842 | 0.972 | 0.988 | 22.74 / **33.37** / 22.10 |
| `pax_waving/mint_f22` | 0.9896 | 0.9998 | 0.9997 | 0.9957 | — | — | — | 0 | 0.869 | 0.975 | 0.990 | 24.75 / **34.54** / 24.78 |
| `pax_waving/white_f22` | 0.9856 | 0.9995 | 0.9997 | 0.9929 | — | — | — | 0 | 0.863 | 0.971 | 0.990 | 24.22 / **34.08** / 24.19 |
| `polly_angry/mint_f39` | **0.8732** | 0.9983 | 0.9957 | 0.9885 | **0.7850** | 0.9855 | 0.9855 | 70 | 0.871 | 0.962 | 0.990 | 28.76 / **34.73** / 28.64 |
| `polly_angry/white_f39` | **0.8597** | 0.9978 | 0.9959 | 0.9842 | **0.7900** | 0.9855 | 0.9855 | 70 | 0.863 | 0.960 | 0.990 | 28.60 / **34.41** / 28.48 |
| `polly_walking/mint_f22` | 0.9857 | 0.9996 | 0.9999 | 0.9934 | **0.8137** | 0.9902 | 1.0000 | 99 | 0.848 | 0.966 | 0.991 | 22.92 / **34.00** / 22.32 |
| `polly_walking/white_f22` | 0.9740 | 0.9993 | 0.9999 | 0.9886 | **0.8235** | 0.9902 | 1.0000 | 99 | 0.832 | 0.963 | 0.991 | 22.58 / **33.30** / 21.91 |
| `polly_waving/mint_f22` | 0.9802 | 0.9995 | 0.9998 | 0.9917 | — | — | — | 0 | 0.864 | 0.965 | 0.993 | 24.34 / **34.56** / 24.37 |
| `polly_waving/white_f22` | 0.9690 | 0.9995 | 0.9998 | 0.9865 | — | — | — | 0 | 0.850 | 0.961 | 0.992 | 23.98 / **33.88** / 23.98 |

Summary (`eye_proxy.json` → `summary`): outline min H3 **0.8597** vs Wan2.1 0.9978. Highlight
min H3 **0.785** vs Wan2.1 0.9855. `outline_pass: false`, `highlight_pass: false`,
`eye_proxy_pass: false`. Sharpness retention: H3 0.832–0.872, Wan2.1 0.960–0.975, Wan2.2-5B
0.985–0.993.

**Pass 2, `eye_proxy_centerline.json`** (H3 vs Wan2.1). It asks whether a stroke's middle
survives and whether a highlight's core survives, separately from their soft rims.

| Clip | centerline min H3 | mean H3 | worst frame H3 | broken gaps/frame (max) H3 | centerline min Wan2.1 | gaps Wan2.1 | highlight core min H3 | Wan2.1 | same rule (≥ 0.95, within 0.03 of Wan2.1) |
|---|---|---|---|---|---|---|---|---|---|
| `pax_angry/mint_f39` | **0.8913** | 0.9905 | 10 | **16** | 0.9997 | 0 | 0.9781 | 1.0000 | **FAIL** |
| `pax_angry/white_f39` | **0.8823** | 0.9887 | 10 | **15** | 0.9996 | 0 | 0.9781 | 1.0000 | **FAIL** |
| `pax_walking/mint_f22` | 0.9925 | 0.9985 | 20 | 1 | 0.9996 | 0 | 0.9793 | 1.0000 | pass |
| `pax_walking/white_f22` | 0.9920 | 0.9976 | 14 | 1 | 0.9996 | 0 | 0.9793 | 1.0000 | pass |
| `pax_waving/mint_f22` | 0.9849 | 0.9979 | 14 | 2 | 1.0000 | 0 | — | — | pass |
| `pax_waving/white_f22` | 0.9861 | 0.9971 | 1 | 2 | 1.0000 | 0 | — | — | pass |
| `polly_angry/mint_f39` | **0.8848** | 0.9917 | 18 | **13** | 0.9996 | 0 | 0.9708 | 1.0000 | **FAIL** |
| `polly_angry/white_f39` | **0.8613** | 0.9889 | 18 | **13** | 0.9996 | 0 | 0.9708 | 1.0000 | **FAIL** |
| `polly_walking/mint_f22` | 0.9917 | 0.9978 | 20 | 1 | 0.9996 | 0 | 0.9793 | 1.0000 | pass |
| `polly_walking/white_f22` | 0.9887 | 0.9965 | 20 | 2 | 0.9996 | 0 | 0.9793 | 1.0000 | pass |
| `polly_waving/mint_f22` | 0.9780 | 0.9966 | 1 | 3 | 1.0000 | 0 | — | — | pass |
| `polly_waving/white_f22` | 0.9726 | 0.9951 | 1 | 2 | 1.0000 | 0 | — | — | pass |

Reading the two passes together:

- **Outlines fail, and the failure is concentrated.** On the 4 angry clips, centerline min is
  0.8613–0.8913, with 13–16 gaps of ≥ 6 px in the worst frame. On the other 8 clips it is
  0.9726–0.9925 with 1–3 gaps. Wan2.1 is ≥ 0.9996 with 0 gaps on all 12. The *mean* over frames
  stays at 0.984–0.999 for H3. The damage sits in particular frames and particular strokes
  (the angry brow and crest lines, and the mouth line), so an average hides it.
- **Highlights pass once the rim is excluded.** Pass 1's highlight failure (0.785–0.824)
  disappears on the 1-px-eroded core (0.9708–0.9793, within 0.03 of Wan2.1's 1.0). H3 softens
  the highlight's edge, but the highlight itself is intact.
- **The worst frames** are 10 (Pax angry) and 18 (Polly angry). Frame 18 is one frame after
  the VAE chunk start at 17. Frame 10 is mid-chunk, so the loss is not a chunk-boundary effect,
  and the chunk-dip bar agrees (all values ≤ 0.055).

The by-eye crop made at the time (`polly_angry_FRONT__white_f39`, frame 18: source | H3 |
Wan2.1 | lost skeleton pixels in red) shows the mouth line inside the beak turned from black to
a faint grey-brown in H3 and kept black in Wan2.1. The contact sheets
`sheets/polly_angry_FRONT__*_768.png` and `sheets/pax_angry_FRONT__*_768.png` show the same at
4× (rows: frame × {eyes, beak, blush, flipper}; columns: source | h3 | wan21 | wan22_5b |
|src − h3| × 5).

---

## 5. The pre-stated rule and how it was applied

**PLAN G0.1, pass when all of:**

| # | Bar | Result at 768² | Pass |
|---|---|---|---|
| 1 | outlines and highlights intact at 768² by eye, judged against Wan2.1 (B7: the eye decides; metrics are recorded) | outlines broken on 4/12 clips (§4.5) | **FAIL** |
| 2 | ground drift ≤ 2/255 on every clip | max 0.996 (`polly_walking/mint`); every clip passes | pass |
| 3 | chunk-boundary dip ≤ 1/255 | max 0.055 (`pax_walking/white`); every clip passes | pass |
| 4 | no grid or banding on the flat-fill clip | `fill_std` 0.288 ≤ 1.0, `fill_mae` 1.663 ≤ 2.0, flat-band 0.117 ≤ 0.5 | pass |

`vae_gate.py` checks bars 2–4 itself and left `gate: PENDING_EYE`. Bar 1 was decided **on
metrics at the user's request**, with the rule written into `eye_proxy.json` before the
numbers were read:

> **"outline & highlight recall ≥ 0.95 every clip and within 0.03 of wan21; alignment shift 0 best"**

How it was applied:

1. **Pass 1** (§4.5): alignment passes. Outline recall fails (min 0.8597 < 0.95). Highlight
   recall fails (min 0.785).
2. **Pass 2** applied the same thresholds to the stroke centerline and the highlight core.
   This removes the one way the rule could be unfair to H3: soft anti-aliased rims counting as
   lost ink. Highlights then **pass** (core min 0.9708 ≥ 0.95, within 0.03 of 1.0). Outlines
   still **fail** on all four angry clips (0.8613–0.8913 < 0.95, and 0.11–0.14 below Wan2.1).
3. One failing clip fails the rule (*every clip*), so bar 1 fails and the gate fails. It was
   recorded with
   `vae_gate.py --eye FAIL --eye-notes "…"`. The note is kept verbatim in
   `report.json` → `eye_check.notes`:

> Decided on metrics at the user's request (pre-stated rule, eye_proxy*.json). Thin strokes
> dissolve on angry clips: outline centerline survival min 0.86-0.89 vs wan21 0.9996, 13-16
> broken stroke gaps (>=6px) per frame vs 0; eyebrows/crest/mouth line lost (polly_angry f18).
> Identical in fp32 (0.86, 15 gaps) so not precision. Highlights intact (core 0.97-0.98).
> wan22_5b (also 16x) keeps all strokes -> H3 decoder, not compression ratio. Metric bars
> (drift, chunk dip, flat fill) passed.

PLAN §1 maps this to **FAIL at V1** ("the VAE cannot represent the art. Stop; no LoRA recovers
it"). RUNBOOK §6 says the same: *Metric FAIL or eye FAIL → stop expv2 and report.*

---

## 6. Why this is H3's decoder, not our pipeline

| Alternative explanation | Test | Result |
|---|---|---|
| **bf16 precision** (the gate's H3 pipe runs `torch_dtype=bfloat16`) | Re-ran the H3 round trip with every dtype fp32 (load, offload, compute) on `polly_angry_FRONT__white_f39`, `pax_angry_FRONT__white_f39` and `polly_walking_FRONT__white_f22`, using the same `h3_roundtrip` and centerline/gap measure | **Same failure**: centerline min 0.86, 15 broken gaps (recorded in `eye_check.notes`). bf16 gave 0.8613 / 13 and 0.8823 / 15 on the two angry clips. The per-clip fp32 printout was not saved; the summary in the note is the record |
| **The 16× compression ratio itself** | Wan2.2-TI2V-5B VAE, also 16× spatial, same clips | It keeps every stroke: outline recall min 0.9957, highlight recall 0.9855–1.0, sharpness 0.985–0.993, edge-PSNR 28.29 (vs H3 17.99), PSNR 47.95 (vs H3 34.63). (The centerline pass was run on H3 and Wan2.1 only. For Wan2.2-5B the evidence is pass 1 and the sheets) |
| **A frame offset** (misaligned comparison would look like lost strokes) | `psnr_shift` at k = −1, 0, +1 | shift 0 is best on all 36 clip × VAE pairs; H3 is 5–11 dB better at 0 than at ±1 |
| **A chunk-boundary artefact** | chunk dip; worst-frame index | dip ≤ 0.055 everywhere; worst frames 10 and 18 (not both at the 17 boundary) |
| **Wrong code path** (diffusers instead of DiffSynth) | by construction | H3 ran through DiffSynth's `MiniMaxH3Pipeline` at the pinned `acf2ad2`: the encode the trainer uses and the decode the sampler uses (§2.4). A LoRA trained in this setup would sit on exactly this decoder |
| **Our source art at 768²** (1080 → 768 thins outlines to 0.71×, PLAN §11) | Wan2.1 and Wan2.2-5B on the same 768² PNGs | both keep the strokes (centerline ≥ 0.9996 for Wan2.1), so the strokes are there to be kept |

**Not isolated:** the fp32 re-run kept DiffSynth's **tiled decode** (`tile_size=256,
tile_overlap=64`), so tiled vs untiled decode was not tested as a variable. Tiling seams would
be expected to show as a lattice on the flat fill (H3 `grid_ratio` 0.2654 at 768²; it has no
bar) rather than as gaps along thin strokes, but this was not measured. See §9.

---

## 7. Bugs found and fixed on the box

All five are in commit **`4832f98`** ("expv2: fix the G0.1 VAE gate and add an early v8
comparison", 2026-10-12, on top of `78ae368`). Diff: 4 files, +17 / −5.

| # | File | Bug | Symptom | Fix |
|---|---|---|---|---|
| 1 | `eval/vae_gate.py` `load_wan` | diffusers 0.35.2's `AutoencoderKLWan.tiled_encode` skips `patchify`, so a `patch_size` VAE (Wan2.2-5B) crashes when tiled | `RuntimeError: Given groups=1, weight of size [160, 12, 3, 3, 3], expected input[1, 3, 3, 258, 258] to have 12 channels, but got 3`. The runner then printed `STOP: G0.1 metric bars FAILED`, a **false FAIL** caused by the crash (`logs/run_expv2.log`, 10:46:08) | tiling only when `vae.config.patch_size in (None, 1)`; Wan2.2-5B runs untiled (fits in 80 GB at 768²) |
| 2 | `eval/vae_gate.py` `main` (means) | the flat-fill clip has no edges, so its `edge_psnr` is NaN, and `np.mean` over clips returned NaN | edge-PSNR means poisoned | means skip `None` **and** NaN (`x == x`) |
| 3 | `setup/setup_box.sh` | `scikit-image` not installed | SSIM silently `None` (`ssim_fn = None` fallback) | added `scikit-image` to the pip line |
| 4 | `eval/h2h_v8.py` | report path hard-coded to `reports/h2h_v8.json` | an early read would overwrite the final §8 report | new `--out` flag |
| 5 | `run_expv2.sh` `stage_base` | — (addition) | — | after G0.3/G0.4, pull v8@550 and write an early, non-gating base-vs-v8 read to `reports/h2h_v8_base.json` (the §8 label stays on the trained winner) |

**Corpus decode tears (not a code bug).** The first box build of the corpus
(`logs/prep.log`, 64 workers) lost 4 sources to random ffmpeg decode failures (exit 245):
`Pax/angry/PAX_ANGRY_FRONT.mov`, `Polly/angry/POLLY_ANGRY_QF2_R.mov`,
`Polly/surprised/POLLY_EXPRESSION_SURPRISE_QF3_R.mov` and
`Polly/surprised/POLLY_EXPRESSION_SURPRISE_SIDE_L.mov`. Prep refused to swap anything in
("tears are non-deterministic"). The re-run (`logs/prep2.log`, 32 workers) was clean: 800
training clips, 2.24 GB, 321 s, verified (frame counts, PNG counts, hashes, holdout
disjointness). It matches PLAN §5.2 exactly: 225 unique sources discovered, 191 usable at f22,
**168 training sources, 672 f22 clips** (+ 64 f39 + 64 f56 built, not trained), **23
held-out** `QF2_R` clips, 156 start frames, bank 209, calib 23, `vae_gate` 12 + flat fill,
`count_deviations: []`, captions ≤ 203 Qwen3-VL tokens. `check_loader.py` then passed 846/846
(f22 718/718, f39 64/64, f56 64/64), and the corpus was pushed to Azure
`processed/expv2_h3/`.

---

## 8. Head-to-head smoke test (§8 pipeline check, not a result)

Before G0.1 the §8 scorer was smoke-tested with **v8@550 against itself**: its 52 `core` clips
pulled from `pudgy/v8/eval/suites/core/v8@550/`, with the H3 side pointed at the same clips
(`h3_ckpt: "_smoke"`, run 2026-10-12T10:46). Both sides being identical is the point: b and c
must be 0, and the scorer's numbers for v8 are a useful baseline.

| | |
|---|---|
| Pairs | **48** (12 shared labels × 2 characters × FRONT/`QF2_R`, seed 42), 0 missing. Windows: 21 frames ×32 (expression), 13 frames ×16 (motion) |
| McNemar, CLIP 1-NN | b = 0, c = 0, both correct 24, neither 24, exact two-sided *p* = 1.0. **v8 accuracy 0.500 (24/48)** |
| McNemar, DINOv2 1-NN | b = 0, c = 0, both 18, neither 30, *p* = 1.0. **v8 accuracy 0.375 (18/48)** |
| Rubric | none (no blinded rubric was filled for a smoke run) |
| Medians (v8) | motion ratio **1.947**, corner drift **2.022**/255, zoom (relative to the label's source) **−3.5%** |
| Median saturation (v8) | 0.157 |

v8@550 CLIP 1-NN, broken down (correct / pairs):

| By | | | | |
|---|---|---|---|---|
| kind | expression 12/32 (0.375) | motion 12/16 (0.750) | | |
| angle | FRONT 13/24 | `QF2_R` 11/24 | | |
| character | Pax 12/24 | Polly 12/24 | | |
| label, 4/4 | crying, laughing, waving, jumping | | | |
| label, 3/4 | angry, walking | | | |
| label, 1/4 | surprised, sitting | | | |
| label, 0/4 | happy, confused, neutral, blushing | | | |

DINOv2 1-NN by kind: expression 8/32, motion 10/16. By angle: FRONT 11/24, `QF2_R` 7/24.

For scale only (PLAN §9's bars are for H3's `QF2_R` clips, not for v8): v8's expression
accuracy (0.375) sits below V4's 0.45 expression bar, and its motion accuracy (0.75) above the
0.40 motion bar. The smoke run confirmed that the scorer, the McNemar arithmetic, the v8 pull
and the 48-pair join all work. Commit `4832f98` then added `--out` and the early base read
(§7). The smoke report is on Azure at `expv2-minimax-h3/eval/reports/h2h_smoke.json` (uploaded 2026-10-13). The stated §8 confounds (resolution 1024² → 768², prompt wording, v8 seeing
`QF2_R` via v7, trimming, n = 1 seed, start-frame rendering size) are recorded in it and in
PLAN §8.

---

## 9. What it means, and what would reopen expv2

**What it means.**

- On this art, H3's VAE does not keep thin black strokes in expressive frames. The strokes it
  loses carry the label: on an angry face, the brows, crest and mouth line *are* the
  expression. A LoRA trains the DiT in latent space and decodes through this same decoder, so
  no amount of fine-tuning can put those strokes back (PLAN §1, FAIL at V1).
- The rest of the VAE is acceptable: ground colour, chunk boundaries and flat fills pass, and
  highlight cores survive. The problem is narrow (thin strokes), but it is on the feature that
  matters most for Pudgy.
- It is not 16× compression as such. Wan2.2-5B at 16× is the best of the three VAEs on every
  measure. A future high-compression base is not ruled out by this result; H3's decoder is.
- PLAN §9 can no longer reach PASS or PARTIAL (both need V1). §8 and the Round 5 re-run (§12)
  are not triggered. v8@550 stays the line's reference.
- Cost: about one A100 session for setup, the corpus build and the gate. Nothing was spent on
  caching, training or generation (PLAN §10 budgeted ≈ 28–37 A100-h for arm A).

**What would reopen expv2.** Any of the following, re-measured with the same rule on all 12
clips (`vae_gate.py`, then both proxy passes; ≤ 1 h of GPU per PLAN §10):

1. **A new H3 video VAE or decoder** (an official fix, or a decoder fine-tuned for line art)
   whose centerline min is ≥ 0.95 and within 0.03 of Wan2.1 on the angry clips. G0.1 would
   then be re-run as written, and the plan would continue from G0.2 unchanged.
2. **An untested decode setting that changes the result.** The one left open is **untiled**
   decode (§6). If `decode_video(..., tiled=False)` at 768² brings the angry clips to the bar,
   G0.1 passes on that setting, and inference and every eval must then use it too.
3. **Different source art.** Thicker strokes, or a working size above 768² inside H3's area
   cap, would be a PLAN amendment (it changes the fixed resolution, §2), not a re-run.

What would **not** reopen it: a better LoRA recipe, arm D, more data, or Round 5 footage. None
of them touch the decoder.

**Housekeeping.** The two proxy scripts (§3.2) and the fp32 check ran from the box session's
scratch space; they are now checked in as-is as `eval/eye_proxy.py`, `eval/eye_proxy_centerline.py`
and `eval/eye_proxy_fp32.py` (paths inside are the box paths of that run). The MiniMax weights
(DiT, text encoder, VAEs; about 119 GB) were deleted from the box at the user's request. The
pins in `eval/phase0/pins.json` (HF revisions `42ed227` / `e5eb578` and sha256) are enough to
re-fetch exactly the same files.

---

## 10. Artifacts

**Azure**, storage account `pudgytraining`, container `pudgy`:

| Path | Content | Files | Size |
|---|---|---|---|
| `expv2-minimax-h3/eval/phase0/vae_gate/report.json` | full gate report: per-clip, per-VAE, per-size metrics, means, bars, eye check, `gate: FAIL` | 1 | 111 KB |
| `expv2-minimax-h3/eval/phase0/vae_gate/eye_proxy.json` | pass 1: outline / highlight recall, sharpness, alignment, rule and summary | 1 | 12 KB |
| `expv2-minimax-h3/eval/phase0/vae_gate/eye_proxy_centerline.json` | pass 2: centerline, broken gaps, highlight cores | 1 | 4 KB |
| `expv2-minimax-h3/eval/phase0/vae_gate/sheets/` | 4× contact sheets, `<clip>_{768,512}.png` | 26 | 9.4 MB |
| `expv2-minimax-h3/eval/phase0/vae_gate/decoded/` | `{h3,wan21,wan22_5b}/{768,512}/<clip>.npz` (key `dec`, uint8 `[F,H,W,3]`) + `h3_png768/<clip>/NNNN.png` | 432 | 419 MB |
| `expv2-minimax-h3/eval/phase0/env.json`, `pins.json` | box environment (pip freeze, SHAs, dirty DiffSynth files); model revisions and sha256 | 2 | 4 KB |
| `expv2-minimax-h3/eval/scores/_bank/` | real-clip embedding bank (`bank_emb_w13.npz`, `bank_emb_w21.npz`), built for the §8 smoke | 2 | 2.4 MB |
| `expv2-minimax-h3/logs/` | `setup_box`, `fetch_core`, `pull_raw`, `prep` (the torn build), `prep2` (the clean build), `run_expv2` (incl. the bug-1 crash), `vae_gate`, `mirror` | 8 | 50 KB |
| `processed/expv2_h3/` | the expv2 corpus: `clips/` (800 mp4), `png/`, `holdout/`, `calib/`, `bank/`, `start_frames/`, `vae_gate/`, `manifest.json`, `metadata_f{22,39,56}.csv`, `check_loader.json`, … | 30,094 | 2.24 GB |
| `v8/eval/suites/core/v8@550/` | v8@550 `core` clips used by the §8 smoke (pre-existing, read only) | — | — |

There is nothing under `expv2-minimax-h3/weights/`, because no LoRA was trained.

**Box** (not persistent): `/workspace/expv2/` (the same eval and log trees, plus `v8_550/` and
`stages/corpus.done`) and `/workspace/Data/processed/expv2_h3/`.

**Git:** `78ae368` (scripts the gate ran), `4832f98` (the fixes in §7, with the result in its
message), this report.
