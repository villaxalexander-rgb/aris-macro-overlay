"""
Module 2 - Risk Layer
Hardcoded risk controls that gate all execution.
"""
import yfinance as yf
from datetime import datetime
from config.settings import (
    MAX_POSITION_PCT,
    VIX_HALT_THRESHOLD,
    DAILY_LOSS_LIMIT_PCT,
    NO_TRADE_MINUTES_BEFORE_CLOSE,
)


def check_vix():
    """Check if VIX is below halt threshold."""
    vix = yf.Ticker("^VIX")
    current_vix = vix.info.get("regularMarketPrice", 0)
    ok = current_vix < VIX_HALT_THRESHOLD
    return ok, current_vix


def check_position_size(proposed_notional, nav):
    """Check if proposed position is within 2% NAV limit."""
    pct = proposed_notional / nav if nav > 0 else 1.0
    ok = pct <= MAX_POSITION_PCT
    return ok, pct


def check_daily_loss(daily_pnl, nav):
    """Check if daily loss exceeds kill switch threshold."""
    loss_pct = daily_pnl / nav if nav > 0 else 0.0
    ok = loss_pct > -DAILY_LOSS_LIMIT_PCT
    return ok, loss_pct


def check_market_hours():
    """Check if we are not in the final 30 minutes before close."""
    now = datetime.now()
    close_hour, close_min = 17, 0
    minutes_to_close = (close_hour * 60 + close_min) - (now.hour * 60 + now.minute)
    ok = minutes_to_close > NO_TRADE_MINUTES_BEFORE_CLOSE
    return ok, minutes_to_close


def run_all_checks(proposed_notional, nav, daily_pnl):
    """Run all risk checks. ALL must pass for execution to proceed."""
    vix_ok, vix_level = check_vix()
    size_ok, size_pct = check_position_size(proposed_notional, nav)
    loss_ok, loss_pct = check_daily_loss(daily_pnl, nav)
    hours_ok, mins_left = check_market_hours()

    all_pass = all([vix_ok, size_ok, loss_ok, hours_ok])

    return {
        "all_pass": all_pass,
        "checks": {
            "vix": {"pass": vix_ok, "value": vix_level, "threshold": VIX_HALT_THRESHOLD},
            "position_size": {"pass": size_ok, "value": size_pct, "threshold": MAX_POSITION_PCT},
            "daily_loss": {"pass": loss_ok, "value": loss_pct, "threshold": -DAILY_LOSS_LIMIT_PCT},
            "market_hours": {"pass": hours_ok, "minutes_to_close": mins_left},
        },
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    result = run_all_checks(proposed_notional=5000, nav=250000, daily_pnl=-2000)
    print(f"All checks pass: {result['all_pass']}")
    for name, check in result["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  {name}: {status}")
