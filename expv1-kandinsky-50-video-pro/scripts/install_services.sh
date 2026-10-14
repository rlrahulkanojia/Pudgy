#!/bin/bash
# Supervisor services for expv1 (base.md §7 pattern): expv1-train (configs/train_expv1.sh,
# auto-resumes) and expv1-mirror (Azure every 15 min). Not autostarted: start them by hand.
#   scripts/install_services.sh && supervisorctl start expv1-mirror expv1-train
#   ARM=b RANK=16 ALPHA=16 scripts/install_services.sh      # arm B (PLAN §6.4)
set -e
E=/workspace/Pudgy/expv1-kandinsky-50-video-pro
mkdir -p /workspace/expv1/logs
cat > /etc/supervisor/conf.d/expv1-train.conf <<CONF
[program:expv1-train]
command=/bin/bash $E/configs/train_expv1.sh
environment=ARM="${ARM:-a}",RANK="${RANK:-32}",ALPHA="${ALPHA:-32}"
directory=/workspace/musubi-tuner
autostart=false
autorestart=false
stopsignal=INT
stopwaitsecs=120
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=/workspace/expv1/logs/train.log
stdout_logfile_maxbytes=0
CONF
cat > /etc/supervisor/conf.d/expv1-mirror.conf <<CONF
[program:expv1-mirror]
command=/venv/main/bin/python $E/scripts/mirror_expv1.py --interval 900
autostart=false
autorestart=true
redirect_stderr=true
stdout_logfile=/workspace/expv1/logs/mirror.log
stdout_logfile_maxbytes=0
CONF
supervisorctl reread && supervisorctl update
