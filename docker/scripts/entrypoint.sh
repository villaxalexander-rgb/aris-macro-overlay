#!/usr/bin/env bash
# entrypoint.sh — ARIS container entry point.
# 1. Waits for IB Gateway to be ready
# 2. Runs the daily pipeline (or a specific command passed as args)
set -e

GATEWAY_HOST="${IBKR_HOST:-ibgateway}"
GATEWAY_PORT="${IBKR_PORT:-4002}"

echo "============================================"
echo " A.R.I.S Macro Overlay System"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# Wait for Gateway
/app/docker/scripts/wait-for-gateway.sh "$GATEWAY_HOST" "$GATEWAY_PORT" 120

# If arguments passed, run those; otherwise run the daily pipeline
if [ $# -gt 0 ]; then
    echo "[entrypoint] Running: $*"
    exec "$@"
else
    echo "[entrypoint] Running daily pipeline..."
    exec python main.py
fi
