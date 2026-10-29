# v9c2 GPU queue jobs

Copies of the job scripts run by the GPU queue (`finetune/wan/gpu_queue.sh`, service `gpu-queue`) on 2026-10-28/29.
Each one states its pre-registered decision rules at the top. Results are logged to `/workspace/research_v9/v2/v2.log`.

| job | does |
|---|---|
| `010_e5_asset_dual.sh` | E5: base decision (R vs Rb), smoke gate, phone-embedding training through both experts, Tier 1, ID1 confirm tier |
| `015_e5_select.sh` | scores every E5 checkpoint on held-out and seen phone clips; selects on held-out phone loss (used only if < 1.00) |
| `020_t5_precision.sh` | T5 fp32 vs fp8 at inference, old wording (control C0old 12/24) |
| `030_pivotal_lora.sh` | pivotal stage 2: cross-attn k/v LoRA through both experts on the frozen selected E, preservation loss, gates (low- and high-noise), ID1 |

`te_cache_cpu_fp32_v9.py` (one level up) rebuilds musubi text caches on CPU in fp32, saved in bf16 (RESEARCH_v9c_fixes 6b).
