"""
A.R.I.S Macro Overlay System — Main Entry Point
Daily orchestration: signals → risk checks → execution → logging → briefing
"""
import json
from datetime import datetime

from signal_engine.daily_signals import run_daily_signals, save_daily_signals
from risk_layer.risk_checks import run_all_checks
from logging_audit.trade_logger import log_trade, generate_pre_trade_thesis
from jeffrey_briefing.briefing import generate_briefing, save_briefing
from risk_layer.drawdown_tracker import update as update_drawdown, get_summary as dd_summary


def run_daily_pipeline():
    """
    Full daily pipeline:
    1. Generate signals + classify regime
    2. For each signal, run risk checks
    3. Execute qualifying trades via IBKR
    4. Log everything
    5. Generate fund note (Jeffrey)
    """
    print(f"\n{'='*60}")
    print(f"A.R.I.S MACRO OVERLAY — Daily Run {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1: Signals
    print("[1/5] Running signal engine...")
    signals = run_daily_signals()
    save_daily_signals(signals)

    # Step 2: Risk checks (using dummy NAV/PnL until IBKR connected)
    print("\n[2/5] Running risk checks...")
    # TODO: Replace with real NAV from IBKR
    nav = 250000
    daily_pnl = 0

    # Update drawdown tracker
    dd_state = update_drawdown(nav)
    print(f"  {dd_summary()}")
    risk_result = run_all_checks(
        proposed_notional=5000,
        nav=nav,
        daily_pnl=daily_pnl,
    )
    print(f"Risk checks: {'ALL PASS' if risk_result['all_pass'] else 'BLOCKED'}")

    if not risk_result["all_pass"]:
        print("⚠ Execution blocked by risk layer. Logging and exiting.")
        # Still log signals and risk state
        return

    # Step 3: Execution
    print("\n[3/5] Execution engine...")
    print("TODO: Connect IBKR executor (Module 3)")
    # executor = IBKRExecutor()
    # executor.connect()
    # ... place orders based on signals ...

    # Step 4: Logging
    print("\n[4/5] Logging trades...")
    print("TODO: Log actual trades once execution is live")

    # Step 5: Jeffrey briefing
    print("\n[5/5] Generating fund note...")
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        note = generate_briefing(today)
        path = save_briefing(note, today)
        print(f"Fund note saved to {path}")
    except Exception as e:
        print(f"Jeffrey briefing failed: {e}")

    print(f"\n{'='*60}")
    print("Daily pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_daily_pipeline()
