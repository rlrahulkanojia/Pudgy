# Training Approach — index

The Pudgy Penguins 2D-animation video-model effort, organised by version. v1–v4 are distinct base-model tracks (v4 is the current trained lead); **v5 changes the data contract rather than the base**. The current plan in this line is **[v6](v6/Training_Approach_v6.md)**, continuing the **Wan** line on the full expression set. The **LTX-2.5** work is *not* part of this line at all — it is a standalone, isolated experiment in [`../ltx25/`](../ltx25/README.md) (see the note below).

## Versions
- **[v1/](v1/Training_Approach_v1.md)** — CogVideoX1.5-5B-I2V, single character/style LoRA. Executed baseline; mid-clip character drift → superseded.
- **[v2/](v2/Training_Approach_v2.md)** — Wan2.2-A14B, decouple identity/motion (two-expert LoRA + FLF2V). **Validated (Gate G1 PASS).** Also: [actions_done.md](v2/actions_done.md) (env stand-up log).
- **[v3/](v3/Training_Approach_v3.md)** — AniSora V3.2 (anime-native). The v2 thesis on an anime-native base; the parallel hedge track.
- **[v4/](v4/Training_Approach_v4.md)** — LTX-2.3-22B (current trained lead): stylized-2D base, IC-LoRA control, Claude-driven prompt system. Plus **[GPU_HANDOFF_iteration_2_v4.md](v4/GPU_HANDOFF_iteration_2_v4.md)** — how the GPU box consumes the `iteration_2_v4` dataset (LTX re-encode → preprocess → train) — and **[CURATION_REPORT_iteration_2_v4.md](v4/CURATION_REPORT_iteration_2_v4.md)** — the data curation report (initial → work → final + validation).
- **v5/** — back to the **Wan2.2-A14B** line, primitives-first. Two documents at different scopes, both current:
  - **[Training_Approach_v5_Happy_Expression_LoRA.md](v5/Training_Approach_v5_Happy_Expression_LoRA.md)** — the **executable pilot**: continue-train the v2 golden high-noise expert on the 7 Pax/happy clips delivered so far. Runnable now.
  - **[Training_Approach_v5.md](v5/Training_Approach_v5.md)** — the **programme plan** the pilot sits inside: a closed taxonomy of motion / expression / interaction-moment primitives, trained as a curriculum (T0–T4), then composed into full video. Data-gated on the [Round 3 request](../docs/documents/Client_Data_Request_Round3.md).

- **[v7/](v7/Training_Approach_v7.md)** — **executed 2026-09-05 → 09-11.** Motion + 7 expressions on the v2 goldens; gate G-X put motion on the **low-noise** expert, so v2's motion prior stays frozen. Training complete, gate coverage partial — see [`GATES_v7.md`](../docs/training_reports/v7/GATES_v7.md) and [`REPORT_v7.md`](../docs/training_reports/v7/REPORT_v7.md).
- **[v6/](v6/Training_Approach_v6.md)** — **the current Wan-line plan**: continue-train the **v2 low-noise golden** on all 272 clips of `Data/processed/v6_expressions_272` (2 characters × 4 emotions). Teaches expression *contrastively* so it becomes promptable — the fix for v5's conditioning-frame memorisation. Low-noise expert only; the v2 high-noise motion golden stays frozen. Runnable now, no new client data.

## Shared reference material (cross-version)
- **[FINDINGS.md](FINDINGS.md)** — consolidated run-v1 + Phase-0 diagnostics + base-model exploration results.
- **[phase0_diagnostics.md](phase0_diagnostics.md)** — VAE round-trip and Phase-0 gate results.
- **[base_model_exploration.md](base_model_exploration.md)** — base-model comparison (Wan / AniSora / …) feeding Gate G0.
- **[docs/](docs/)** — GPU handover notes; **[scripts/vae_roundtrip.py](scripts/vae_roundtrip.py)** — multi-VAE round-trip tool; **[assets/](assets/)** — montages referenced by FINDINGS.

> The `iteration_2_v4` **dataset** (clips, prompts, catalog, `prep_ltx.py`) lives outside this repo under `Data/processed/v4_ltx_249clip/` and is transferred to the GPU box manually — see the v4 GPU handoff doc above.

> **Note:** the LTX-2.5 work is **not** part of this version line and does not live here. It is a standalone clean-slate experiment in [`../ltx25/`](../ltx25/README.md) — its own docs, prep, config and evaluation, sharing **no code, data, config or recipe** with v1–v7. That separation is a requirement of the experiment, not an accident: a result there has to mean something on its own rather than relative to a chain of prior runs. Measured facts about the art and the client carry over; none of the Wan- or v4-specific structure does. An earlier LTX-2.5 plan (`alpha v-alpha`) that framed the move as a port of the v4 track was **removed** for exactly that reason — it is recoverable from git history at `f42e3f2` if the analysis is ever wanted.
