# Training Approach — index

The Pudgy Penguins 2D-animation video-model effort, organised by version. Each version is a distinct base-model track; v4 is the current lead.

## Versions
- **[v1/](v1/Training_Approach_v1.md)** — CogVideoX1.5-5B-I2V, single character/style LoRA. Executed baseline; mid-clip character drift → superseded.
- **[v2/](v2/Training_Approach_v2.md)** — Wan2.2-A14B, decouple identity/motion (two-expert LoRA + FLF2V). **Validated (Gate G1 PASS).** Also: [actions_done.md](v2/actions_done.md) (env stand-up log).
- **[v3/](v3/Training_Approach_v3.md)** — AniSora V3.2 (anime-native). The v2 thesis on an anime-native base; the parallel hedge track.
- **[v4/](v4/Training_Approach_v4.md)** — LTX-2.3-22B (current lead): stylized-2D base, IC-LoRA control, Claude-driven prompt system. Plus **[GPU_HANDOFF_iteration_2_v4.md](v4/GPU_HANDOFF_iteration_2_v4.md)** — how the GPU box consumes the `iteration_2_v4` dataset (LTX re-encode → preprocess → train).

## Shared reference material (cross-version)
- **[FINDINGS.md](FINDINGS.md)** — consolidated run-v1 + Phase-0 diagnostics + base-model exploration results.
- **[phase0_diagnostics.md](phase0_diagnostics.md)** — VAE round-trip and Phase-0 gate results.
- **[base_model_exploration.md](base_model_exploration.md)** — base-model comparison (Wan / AniSora / …) feeding Gate G0.
- **[docs/](docs/)** — GPU handover notes; **[scripts/vae_roundtrip.py](scripts/vae_roundtrip.py)** — multi-VAE round-trip tool; **[assets/](assets/)** — montages referenced by FINDINGS.

> The `iteration_2_v4` **dataset** (clips, prompts, catalog, `prep_ltx.py`) lives outside this repo under `Data/iteration_2_v4/` and is transferred to the GPU box manually — see the v4 GPU handoff doc above.
