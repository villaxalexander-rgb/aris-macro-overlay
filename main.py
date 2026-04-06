"""
A.R.I.S Macro Overlay System - Main Entry Point
"""
import json
import os
from datetime import datetime

from signal_engine.daily_signals import run_daily_signals, save_daily_signals
from risk_layer.risk_checks import run_all_checks


def run_daily_pipeline():
    print(f"\n{'='*60}")
    print(f"A.R.I.S MACRO OVERLAY - Daily Run {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1: Signals (always runs)
    print("[1/5] Running signal engine...")
    signals = run_daily_signals()
    signal_file = save_daily_signals(signals)
    print(f"Signal file: {signal_file}")

    # Step 2: Risk checks
    print("\n[2/5] Running risk checks...")
    nav = 250000  # TODO: Replace with real NAV from IBKR
    daily_pnl = 0
    risk_result = run_all_checks(proposed_notional=5000, nav=nav, daily_pnl=daily_pnl)

    # Log risk check results
    risk_status = "ALL PASS" if risk_result["all_pass"] else "BLOCKED"
    print(f"Risk checks: {risk_status}")
    for name, check in risk_result["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  {name}: {status} (value: {check.get('value', 'N/A')})")

    # Save risk check results alongside signals
    risk_log = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "risk_status": risk_status,
        "checks": risk_result["checks"],
    }
    os.makedirs("data/signals", exist_ok=True)
    risk_file = f"data/signals/{risk_log['date']}_risk.json"
    with open(risk_file, "w") as f:
        json.dump(risk_log, f, indent=2, default=str)
    print(f"Risk log: {risk_file}")

    if not risk_result["all_pass"]:
        print("\nExecution blocked by risk layer. Signals saved, skipping execution.")
        print(f"\n{'='*60}")
        print("Daily pipeline complete (signals only).")
        print(f"{'='*60}\n")
        return

    # Step 3: Execution
    print("\n[3/5] Execution engine...")
    print("TODO: Connect IBKR executor (Module 3)")

    # Step 4: Logging
    print("\n[4/5] Logging trades...")
    print("TODO: Log actual trades once execution is live")

    # Step 5: Jeffrey briefing
    print("\n[5/5] Generating fund note...")
    print("TODO: Connect Jeffrey briefing (Module 5)")

    print(f"\n{'='*60}")
    print("Daily pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_daily_pipeline()
