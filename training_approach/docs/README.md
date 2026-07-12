# Training Approach — Pudgy Penguins video model

Planning + handover for the Pax & Polly character/style video model.

| File | What it is | For |
|---|---|---|
| **`Training_Approach_v1.md`** | The executed CogVideoX1.5 single-LoRA baseline + why it fell short | context / the bar to beat |
| **`Training_Approach_v2.md`** | The new plan: decouple identity↔motion, switch base, phases + gates + rubric | strategy |
| **`Training_Approach_v3.md`** | Commits the base to **AniSora V3.2** (anime-native Wan2.2-A14B); native keyframe-interpolation identity-pinning; pros + expected gains over v2 | strategy |
| **`Training_Approach_v4.md`** | Alternative base track: **LTX-2.3-22B** (stylized-2D-native, IC-LoRA control, official looping, <1h LoRA); unconstrained dataset rebuild; **Claude-driven prompt system**. Baked off against v3. | strategy (current) |
| **`GPU_HANDOVER.md`** | ⭐ Copy-paste runbook for the GPU team (Phase 0 + Phase 1 setup) | execution |
| **`scripts/vae_roundtrip.py`** | Phase 0.1 VAE reconstruction diagnostic (decides the base model) | execution |

**Start here:** GPU team → read v1, then v2, then run `GPU_HANDOVER.md`. Author is on a Mac (no CUDA) — all training/diagnostics run on the GPU box.

**Related (repo root):** `Data_Readiness_Gap_Analysis.md`, `Data_Discrepancies.md`, `training_dataset/`, `pudgy-lora-repo/`. Full evidence + sources in the *Model & Architecture Reassessment* memo.
