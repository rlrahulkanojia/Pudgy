# v8 box services

Copies of what runs on the GPU box outside the repo, so a fresh box can be rebuilt:

| File | Install to | Does |
|---|---|---|
| `pudgy-v8-train.sh` / `.conf` | `/opt/supervisor-scripts/`, `/etc/supervisor/conf.d/` | training under supervisor; resumes from the newest `*-stepN-state` and appends to the same W&B run |
| `pudgy-v8-monitor.sh` / `.conf` | same | `monitor_v8.py`: status, Tier-0 diagnostics, Azure mirror |
| `pull_raw.py` | `/workspace/` | download `raw/iteration_*` from Azure |
| `cache_v8.sh` | `/workspace/` | latents + T5 cache for the v8 set |

After copying: `supervisorctl reread && supervisorctl update`. Both programs are
`autostart=false`; start with `supervisorctl start pudgy-v8-monitor pudgy-v8-train`.
