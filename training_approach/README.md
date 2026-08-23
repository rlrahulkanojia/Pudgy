# Training Approach — index

The Pudgy Penguins 2D-animation video-model effort, organised by version. v1–v4 are distinct base-model tracks (v4 is the current trained lead); **v5 changes the data contract rather than the base**. Two current plans run in parallel on independent tracks: **[v6](v6/Training_Approach_v6.md)** continues the **Wan** line on the full expression set, and **[`alpha v-alpha`](LTX-2.5/Experiment_alpha_v-alpha.md)** moves the **LTX** line to 2.5.

## Versions
- **[v1/](v1/Training_Approach_v1.md)** — CogVideoX1.5-5B-I2V, single character/style LoRA. Executed baseline; mid-clip character drift → superseded.
- **[v2/](v2/Training_Approach_v2.md)** — Wan2.2-A14B, decouple identity/motion (two-expert LoRA + FLF2V). **Validated (Gate G1 PASS).** Also: [actions_done.md](v2/actions_done.md) (env stand-up log).
- **[v3/](v3/Training_Approach_v3.md)** — AniSora V3.2 (anime-native). The v2 thesis on an anime-native base; the parallel hedge track.
- **[v4/](v4/Training_Approach_v4.md)** — LTX-2.3-22B (current trained lead): stylized-2D base, IC-LoRA control, Claude-driven prompt system. Plus **[GPU_HANDOFF_iteration_2_v4.md](v4/GPU_HANDOFF_iteration_2_v4.md)** — how the GPU box consumes the `iteration_2_v4` dataset (LTX re-encode → preprocess → train) — and **[CURATION_REPORT_iteration_2_v4.md](v4/CURATION_REPORT_iteration_2_v4.md)** — the data curation report (initial → work → final + validation).
- **v5/** — back to the **Wan2.2-A14B** line, primitives-first. Two documents at different scopes, both current:
  - **[Training_Approach_v5_Happy_Expression_LoRA.md](v5/Training_Approach_v5_Happy_Expression_LoRA.md)** — the **executable pilot**: continue-train the v2 golden high-noise expert on the 7 Pax/happy clips delivered so far. Runnable now.
  - **[Training_Approach_v5.md](v5/Training_Approach_v5.md)** — the **programme plan** the pilot sits inside: a closed taxonomy of motion / expression / interaction-moment primitives, trained as a curriculum (T0–T4), then composed into full video. Data-gated on the [Round 3 request](../docs/documents/Client_Data_Request_Round3.md).

- **[LTX-2.5/](LTX-2.5/Experiment_alpha_v-alpha.md)** — **experiment `alpha v-alpha`**: ports the v4 LTX track to **LTX-2.5-22B** and, in the same harness, runs the data-contract ablation v4 left open (does a small clean corpus beat the big dirty one?). Five gated arms; A1/A0/A2/A3 need **no new data**, and A4's 60-clip T1 gate was largely met by the 2026-08-20 delivery (68 clips in; `sad` and the turnaround stills still open).

- **[v6/](v6/Training_Approach_v6.md)** — **the current Wan-line plan**: continue-train the **v2 low-noise golden** on all 272 clips of `Data/processed/v6_expressions_272` (2 characters × 4 emotions). Teaches expression *contrastively* so it becomes promptable — the fix for v5's conditioning-frame memorisation. Low-noise expert only; the v2 high-noise motion golden stays frozen. Runnable now, no new client data.

## Shared reference material (cross-version)
- **[FINDINGS.md](FINDINGS.md)** — consolidated run-v1 + Phase-0 diagnostics + base-model exploration results.
- **[phase0_diagnostics.md](phase0_diagnostics.md)** — VAE round-trip and Phase-0 gate results.
- **[base_model_exploration.md](base_model_exploration.md)** — base-model comparison (Wan / AniSora / …) feeding Gate G0.
- **[docs/](docs/)** — GPU handover notes; **[scripts/vae_roundtrip.py](scripts/vae_roundtrip.py)** — multi-VAE round-trip tool; **[assets/](assets/)** — montages referenced by FINDINGS.

> The `iteration_2_v4` **dataset** (clips, prompts, catalog, `prep_ltx.py`) lives outside this repo under `Data/processed/v4_ltx_249clip/` and is transferred to the GPU box manually — see the v4 GPU handoff doc above.
