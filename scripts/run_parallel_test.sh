#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────
# A.R.I.S — 3-Day Parallel Hybrid Test
# Runs the daily signal pipeline twice per invocation:
#   1. USE_HYBRID_ALPHA=true   (T6 hybrid: 50% TSMOM + 50% BSV)
#   2. USE_HYBRID_ALPHA=false  (BSV-only baseline)
# Outputs land in data/parallel_test/{hybrid,baseline}/ for comparison.
#
# Usage (on Hetzner):
#   cd ~/Desktop/aris-macro-overlay
#   chmod +x scripts/run_parallel_test.sh
#   bash scripts/run_parallel_test.sh          # single run (both modes)
#
#   Schedule via cron for 3 consecutive trading days:
#   crontab -e →
#   30 20 * * 1-5 cd ~/Desktop/aris-macro-overlay && bash scripts/run_parallel_test.sh
#   (20:30 UTC = 16:30 ET, Mon-Fri)
#
#   After 3 days, generate the comparison report:
#   bash scripts/run_parallel_test.sh --compare
# ───────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Desktop/aris-macro-overlay}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

DATE=$(date +%Y-%m-%d)
HYBRID_DIR="data/parallel_test/hybrid"
BASELINE_DIR="data/parallel_test/baseline"
LOG_DIR="data/parallel_test/logs"
REPORT_FILE="data/parallel_test/comparison_report.txt"

mkdir -p "$HYBRID_DIR" "$BASELINE_DIR" "$LOG_DIR"

# ── Compare mode ──────────────────────────────────────────────────
if [[ "${1:-}" == "--compare" ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo " A.R.I.S Parallel Test — Comparison Report"
    echo " Generated: $(date)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    python3 - <<'PYEOF'
import json, glob, os, sys
from collections import defaultdict

hybrid_files = sorted(glob.glob("data/parallel_test/hybrid/*_signals.json"))
baseline_files = sorted(glob.glob("data/parallel_test/baseline/*_signals.json"))

if not hybrid_files or not baseline_files:
    print("ERROR: No signal files found. Run the pipeline for at least 1 day first.")
    sys.exit(1)

print(f"Days collected: hybrid={len(hybrid_files)}, baseline={len(baseline_files)}")
print("")

# Parse all signal files
def load_signals(files):
    days = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        date = data.get("date", os.path.basename(f)[:10])
        signals = {}
        for s in data.get("signals", []):
            canonical = s.get("canonical", s.get("ticker", "?"))
            signals[canonical] = {
                "target": s.get("target_weight", s.get("weight", 0)),
                "direction": s.get("direction", "flat"),
                "score": s.get("composite_score", s.get("score", 0)),
            }
        days[date] = signals
    return days

hybrid = load_signals(hybrid_files)
baseline = load_signals(baseline_files)

# Compare each overlapping day
all_dates = sorted(set(hybrid.keys()) & set(baseline.keys()))
print(f"Overlapping days: {len(all_dates)}")
print("")

for date in all_dates:
    h = hybrid[date]
    b = baseline[date]
    all_assets = sorted(set(h.keys()) | set(b.keys()))

    print(f"── {date} ──")
    print(f"  {'Asset':<6} {'Hybrid Wt':>10} {'BSV Wt':>10} {'Delta':>10} {'H-Dir':<6} {'B-Dir':<6}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*6}")

    total_gross_h, total_gross_b = 0, 0
    flips = 0
    for asset in all_assets:
        hw = h.get(asset, {}).get("target", 0)
        bw = b.get(asset, {}).get("target", 0)
        hd = h.get(asset, {}).get("direction", "flat")
        bd = b.get(asset, {}).get("direction", "flat")
        delta = hw - bw
        total_gross_h += abs(hw)
        total_gross_b += abs(bw)
        if hd != bd:
            flips += 1
        flag = " <<<" if abs(delta) > 0.05 else ""
        print(f"  {asset:<6} {hw:>10.4f} {bw:>10.4f} {delta:>+10.4f} {hd:<6} {bd:<6}{flag}")

    print(f"  Gross exposure: hybrid={total_gross_h:.4f}, baseline={total_gross_b:.4f}")
    print(f"  Direction flips: {flips}/{len(all_assets)}")
    print("")

print("═══════════════════════════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════════════════════════")
print(f"  Total days compared: {len(all_dates)}")
if len(all_dates) < 3:
    print("  ⚠ Less than 3 days — keep running before merging PR #1")
else:
    print("  ✓ 3+ days collected — review the deltas above and decide")
PYEOF

    exit 0
fi


# ── Run mode (default) ────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " A.R.I.S Parallel Test — $DATE"
echo "═══════════════════════════════════════════════════════════════"

# --- Run 1: Hybrid ON ---
echo ""
echo "[1/2] Running with USE_HYBRID_ALPHA=true ..."
export USE_HYBRID_ALPHA=true
export TSMOM_WEIGHT=0.50
export SIGNAL_OUTPUT_PATH="$HYBRID_DIR/"
python3 -c "
from signal_engine.daily_signals import run_daily_signals, save_daily_signals
import os
os.environ['SIGNAL_OUTPUT_PATH'] = '$HYBRID_DIR/'
output = run_daily_signals()
save_daily_signals(output)
print(f'Hybrid signals saved: {len(output.get(\"signals\",[]))} assets')
" >> "$LOG_DIR/${DATE}_hybrid.log" 2>&1

HYBRID_EXIT=$?
if [ $HYBRID_EXIT -eq 0 ]; then
    echo "  ✓ Hybrid run complete — see $LOG_DIR/${DATE}_hybrid.log"
else
    echo "  ✗ Hybrid run FAILED (exit $HYBRID_EXIT) — check $LOG_DIR/${DATE}_hybrid.log"
fi

# --- Run 2: Baseline (BSV-only) ---
echo "[2/2] Running with USE_HYBRID_ALPHA=false ..."
export USE_HYBRID_ALPHA=false
export SIGNAL_OUTPUT_PATH="$BASELINE_DIR/"
python3 -c "
from signal_engine.daily_signals import run_daily_signals, save_daily_signals
import os
os.environ['SIGNAL_OUTPUT_PATH'] = '$BASELINE_DIR/'
output = run_daily_signals()
save_daily_signals(output)
print(f'Baseline signals saved: {len(output.get(\"signals\",[]))} assets')
" >> "$LOG_DIR/${DATE}_baseline.log" 2>&1

BASELINE_EXIT=$?
if [ $BASELINE_EXIT -eq 0 ]; then
    echo "  ✓ Baseline run complete — see $LOG_DIR/${DATE}_baseline.log"
else
    echo "  ✗ Baseline run FAILED (exit $BASELINE_EXIT) — check $LOG_DIR/${DATE}_baseline.log"
fi


# --- Quick diff ---
echo ""
echo "── Quick diff for $DATE ──"
H_FILE="$HYBRID_DIR/${DATE}_signals.json"
B_FILE="$BASELINE_DIR/${DATE}_signals.json"
if [ -f "$H_FILE" ] && [ -f "$B_FILE" ]; then
    python3 -c "
import json
h = json.load(open('$H_FILE'))
b = json.load(open('$B_FILE'))
hs = {s.get('canonical', s.get('ticker','?')): s for s in h.get('signals',[])}
bs = {s.get('canonical', s.get('ticker','?')): s for s in b.get('signals',[])}
assets = sorted(set(hs)|set(bs))
flips = sum(1 for a in assets
            if hs.get(a,{}).get('direction','flat') != bs.get(a,{}).get('direction','flat'))
print(f'  Assets: {len(assets)}  |  Direction flips: {flips}')
for a in assets:
    hw = hs.get(a,{}).get('target_weight', hs.get(a,{}).get('weight',0))
    bw = bs.get(a,{}).get('target_weight', bs.get(a,{}).get('weight',0))
    d = hw - bw
    if abs(d) > 0.01:
        print(f'    {a:<6} hybrid={hw:+.4f}  bsv={bw:+.4f}  Δ={d:+.4f}')
"
else
    echo "  (signal files not found — check logs for errors)"
fi

echo ""
echo "Done. Run 'bash scripts/run_parallel_test.sh --compare' after 3 days."
deactivate 2>/dev/null || true
