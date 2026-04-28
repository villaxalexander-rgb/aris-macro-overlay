#!/usr/bin/env bash
# scheduler-entrypoint.sh — Sets up cron to run the ARIS pipeline daily.
#
# Default schedule: 16:30 ET Mon-Fri (30 minutes after US futures close).
# Override via PIPELINE_SCHEDULE env var.
set -e

SCHEDULE="${PIPELINE_SCHEDULE:-30 16 * * 1-5}"

echo "============================================"
echo " A.R.I.S Scheduler"
echo " Schedule: ${SCHEDULE} (cron format)"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# Export all env vars so cron jobs can access them
printenv | grep -E '^(IBKR_|FRED_|LSEG_|PRIMARY_|ANTHROPIC_|PIPELINE_|PATH)' > /app/.env.cron

# Write crontab
cat > /etc/cron.d/aris-pipeline << EOF
# A.R.I.S Daily Pipeline
SHELL=/bin/bash
${SCHEDULE} root cd /app && set -a && source /app/.env.cron && set +a && python main.py >> /app/logs/cron.log 2>&1

# Jeffrey briefing 5 min after pipeline (let signals settle)
$(echo "$SCHEDULE" | awk '{$1=$1+5; print}') root cd /app && set -a && source /app/.env.cron && set +a && python jeffrey_briefing/briefing.py --stdout >> /app/logs/cron.log 2>&1

# LSEG key rotation reminder — check every Monday at 09:00 UTC
0 9 * * 1 root cd /app && python3 -c "
import json, os
from datetime import datetime, timedelta
path = 'state/lseg_key_rotation.json'
if not os.path.exists(path):
    print('[LSEG-KEY] WARNING: No rotation history found. Rotate your LSEG app key!')
    print('[LSEG-KEY] Run: ./docker/scripts/rotate-lseg-key.sh NEW_KEY')
else:
    with open(path) as f:
        history = json.load(f)
    last = datetime.fromisoformat(history[-1]['rotated_at'].rstrip('Z'))
    days_since = (datetime.utcnow() - last).days
    if days_since >= 90:
        print(f'[LSEG-KEY] WARNING: Key is {days_since} days old — ROTATE NOW!')
        print('[LSEG-KEY] Run: ./docker/scripts/rotate-lseg-key.sh NEW_KEY')
    elif days_since >= 75:
        print(f'[LSEG-KEY] NOTICE: Key is {days_since} days old — rotation due in {90-days_since} days.')
    else:
        print(f'[LSEG-KEY] OK: Key rotated {days_since} days ago.')
" >> /app/logs/cron.log 2>&1
EOF

chmod 0644 /etc/cron.d/aris-pipeline

echo "[scheduler] Crontab installed. Starting cron daemon..."
echo "[scheduler] Next run: $(date -d 'tomorrow 16:30' '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'check schedule')"

# Run cron in foreground
exec cron -f
