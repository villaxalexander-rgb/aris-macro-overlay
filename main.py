"""
A.R.I.S Macro Overlay System - Main Entry Point
"""
from datetime import datetime

from signal_engine.daily_signals import run_daily_signals, save_daily_signals
from risk_layer.risk_checks import run_all_checks


def run_daily_pipeline():
    print(f"\n{'='*60}")
    print(f"A.R.I.S MACRO OVERLAY - Daily Run {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1: Signals
    print("[1/5] Running signal engine...")
    signals = run_daily_signals()
    save_daily_signals(signals)

    # Step 2: Risk checks
    print("\n[2/5] Running risk checks...")
    nav = 250000  # TODO: Replace with real NAV from IBKR
    daily_pnl = 0
    risk_result = run_all_checks(proposed_notional=5000, nav=nav, daily_pnl=daily_pnl)
    print(f"Risk checks: {'ALL PASS' if risk_result['all_pass'] else 'BLOCKED'}")

    if not risk_result["all_pass"]:
        print("Execution blocked by risk layer. Logging and exiting.")
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
