#!/usr/bin/env bash
# wait-for-gateway.sh — block until IB Gateway API port is accepting connections.
set -e

HOST="${1:-ibgateway}"
PORT="${2:-4002}"
TIMEOUT="${3:-120}"

echo "[wait-for-gateway] Waiting for IB Gateway at ${HOST}:${PORT} (timeout ${TIMEOUT}s)..."

elapsed=0
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "[wait-for-gateway] ERROR: Gateway not ready after ${TIMEOUT}s — aborting."
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $((elapsed % 10)) -eq 0 ]; then
        echo "[wait-for-gateway] Still waiting... (${elapsed}s elapsed)"
    fi
done

echo "[wait-for-gateway] Gateway is up at ${HOST}:${PORT} (took ${elapsed}s)."
