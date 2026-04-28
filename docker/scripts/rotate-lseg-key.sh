#!/usr/bin/env bash
# rotate-lseg-key.sh — Validate a new LSEG app key and hot-swap it into .env.
#
# Usage:
#   ./docker/scripts/rotate-lseg-key.sh NEW_APP_KEY_HERE
#
# What it does:
#   1. Validates the new key by making a test API call via Python
#   2. Backs up current .env
#   3. Swaps LSEG_APP_KEY in .env
#   4. Restarts the ARIS + scheduler containers (ibgateway untouched)
#   5. Records rotation timestamp in state/lseg_key_rotation.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
STATE_FILE="$PROJECT_ROOT/state/lseg_key_rotation.json"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <NEW_LSEG_APP_KEY>"
    echo ""
    echo "Steps to get a new key:"
    echo "  1. Go to https://developers.lseg.com/en/api-catalog"
    echo "  2. Open App Key Generator (or MyAccount → App Keys)"
    echo "  3. Create new key or regenerate existing"
    echo "  4. Run this script with the new key"
    exit 1
fi

NEW_KEY="$1"

echo "============================================"
echo " A.R.I.S — LSEG App Key Rotation"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ── 1. Validate the new key ──────────────────────────────────
echo ""
echo "[1/5] Validating new key..."

# Load current env for LSEG session config
set -a
source "$ENV_FILE" 2>/dev/null || true
set +a

VALIDATION_RESULT=$(python3 -c "
import os, sys
try:
    # Attempt a lightweight LSEG API call with the new key
    os.environ['LSEG_APP_KEY'] = '$NEW_KEY'
    session_type = os.environ.get('LSEG_SESSION_TYPE', 'platform')

    if session_type == 'desktop':
        # Desktop: try opening a desktop session (requires Workspace running)
        import lseg.data as ld
        ld.open_session(app_key='$NEW_KEY')
        ld.close_session()
        print('OK:desktop_session')
    else:
        # Platform: test with RDP credentials
        import lseg.data as ld
        ld.open_session(
            app_key='$NEW_KEY',
            grant=ld.session.platform.GrantPassword(
                username=os.environ.get('LSEG_USERNAME', ''),
                password=os.environ.get('LSEG_PASSWORD', ''),
            ),
        )
        ld.close_session()
        print('OK:platform_session')
except ImportError:
    # lseg-data not installed locally — skip validation, trust the key
    print('SKIP:lseg_not_installed')
except Exception as e:
    print(f'FAIL:{e}')
" 2>&1)

echo "  Validation: $VALIDATION_RESULT"

if [[ "$VALIDATION_RESULT" == FAIL:* ]]; then
    echo ""
    echo "  ERROR: New key failed validation."
    echo "  Reason: ${VALIDATION_RESULT#FAIL:}"
    echo "  .env NOT updated. Aborting."
    exit 1
fi

if [[ "$VALIDATION_RESULT" == SKIP:* ]]; then
    echo "  WARNING: Could not validate (lseg-data not installed locally)."
    echo "  Proceeding with key swap — verify manually after restart."
fi

# ── 2. Backup current .env ───────────────────────────────────
echo ""
echo "[2/5] Backing up .env..."
BACKUP="$ENV_FILE.bak.$(date +%Y%m%d_%H%M%S)"
cp "$ENV_FILE" "$BACKUP"
echo "  Saved to: $BACKUP"

# ── 3. Swap the key ─────────────────────────────────────────
echo ""
echo "[3/5] Updating LSEG_APP_KEY in .env..."
OLD_KEY=$(grep '^LSEG_APP_KEY=' "$ENV_FILE" | cut -d'=' -f2- || echo "")

if grep -q '^LSEG_APP_KEY=' "$ENV_FILE"; then
    sed -i "s|^LSEG_APP_KEY=.*|LSEG_APP_KEY=${NEW_KEY}|" "$ENV_FILE"
else
    echo "LSEG_APP_KEY=${NEW_KEY}" >> "$ENV_FILE"
fi
echo "  Old key: ${OLD_KEY:0:8}...${OLD_KEY: -4} (masked)"
echo "  New key: ${NEW_KEY:0:8}...${NEW_KEY: -4} (masked)"

# ── 4. Restart ARIS containers ───────────────────────────────
echo ""
echo "[4/5] Restarting ARIS containers..."
cd "$PROJECT_ROOT"
if command -v docker &>/dev/null && docker compose ps --quiet 2>/dev/null; then
    docker compose restart aris scheduler
    echo "  Containers restarted."
else
    echo "  Docker not running or compose not available — skip restart."
    echo "  Run manually: cd $PROJECT_ROOT && docker compose restart aris scheduler"
fi

# ── 5. Record rotation ──────────────────────────────────────
echo ""
echo "[5/5] Recording rotation..."
mkdir -p "$(dirname "$STATE_FILE")"
python3 -c "
import json, os
from datetime import datetime
path = '$STATE_FILE'
history = []
if os.path.exists(path):
    with open(path) as f:
        history = json.load(f)
history.append({
    'rotated_at': datetime.utcnow().isoformat() + 'Z',
    'key_prefix': '${NEW_KEY:0:8}',
    'validation': '$VALIDATION_RESULT',
})
with open(path, 'w') as f:
    json.dump(history, f, indent=2)
print(f'  Rotation #{len(history)} recorded.')
"

echo ""
echo "============================================"
echo " Key rotation complete."
echo " Next rotation due: ~$(date -d '+90 days' '+%Y-%m-%d' 2>/dev/null || echo '90 days from now')"
echo "============================================"
