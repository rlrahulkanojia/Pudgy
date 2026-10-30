# v8 box services

Copies of what runs on the GPU box outside the repo, so a fresh box can be rebuilt:

| File | Install to | Does |
|---|---|---|
| `pudgy-v8-train.sh` / `.conf` | `/opt/supervisor-scripts/`, `/etc/supervisor/conf.d/` | training under supervisor; resumes from the newest `*-stepN-state` and appends to the same W&B run |
| `pudgy-v8-monitor.sh` / `.conf` | same | `monitor_v8.py`: status, Tier-0 diagnostics, Azure mirror |
| `pull_raw.py` | `/workspace/` | download `raw/iteration_*` from Azure |
| `cache_v8.sh` | `/workspace/` | latents + T5 cache for the v8 set |
| `gpu-queue.sh` / `.conf` | `/opt/supervisor-scripts/`, `/etc/supervisor/conf.d/` | `finetune/wan/gpu_queue.sh`: runs job scripts from `/workspace/gpu_queue/pending/` one at a time once the GPU has been idle 3 min (no compute processes, < 2 GB used); `done/`, `failed/`, `logs/`, `queue.log`; `touch /workspace/gpu_queue/PAUSE` holds new starts. `autostart=true` |

After copying: `supervisorctl reread && supervisorctl update`. Both programs are
`autostart=false`; start with `supervisorctl start pudgy-v8-monitor pudgy-v8-train`.
