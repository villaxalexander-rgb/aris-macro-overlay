"""
Module 2 — Risk Layer
Hardcoded risk controls that gate all execution.
Every check must pass before any order is sent.

Phase 2 changes:
  - VIX/MOVE/DXY now flow through DualSourceRouter (FRED primary
    where LSEG isn't entitled, LSEG primary for MOVE/DXY which
    ARE entitled).
  - MOVE added as a hard halt (rates vol regime).
  - DXY added as a soft warning indicator (USD regime).
  - Each check returns its provider/source for the HealthRecord.
"""
from datetime import datetime
from typing import Optional

from signal_engine.data_sources.dual_source import DualSourceRouter
from signal_engine.resilience import log
from config.settings import (
    MAX_POSITION_PCT,
    VIX_HALT_THRESHOLD,
    MOVE_HALT_THRESHOLD,
    DXY_WARN_HIGH,
    DXY_WARN_LOW,
    DAILY_LOSS_LIMIT_PCT,
    NO_TRADE_MINUTES_BEFORE_CLOSE,
)

# ---- Lazy router singleton ------------------------------------------------
_router: Optional[DualSourceRouter] = None


def _get_router() -> DualSourceRouter:
    """Lazy DualSourceRouter — FRED primary for macro since VIX is FRED-only."""
    global _router
    if _router is None:
        _router = DualSourceRouter(primary_macro="fred")
        if _router.lseg is not None:
            try:
                _router.lseg.open()
            except Exception as e:
                log.warning(f"LSEG session could not open in risk layer ({e})")
                _router.lseg = None
    return _router


# ---- Macro vol / FX checks ------------------------------------------------
def check_vix() -> tuple[bool, float]:
    """Check if VIX is below halt threshold (equity vol regime).
    Routes via FRED VIXCLS — LSEG .VIX is not entitled on this account.
    Returns (ok, current_vix). On fetch failure, returns (False, NaN) — fail-safe.
    """
    try:
        router = _get_router()
        current_vix = router.fetch_vix()
        ok = current_vix < VIX_HALT_THRESHOLD
        return ok, current_vix
    except Exception as e:
        log.error(f"check_vix failed: {e}. Failing safe (halt).")
        return False, float("nan")


def check_move() -> tuple[bool, float]:
    """Check if MOVE is below halt threshold (rates vol regime).
    Routes via LSEG .MOVE primary, no FRED equivalent.
    Returns (ok, current_move). On fetch failure, returns (False, NaN).
    """
    try:
        router = _get_router()
        s = router.fetch_macro("move", lookback_months=1)
        current_move = float(s.dropna().iloc[-1])
        ok = current_move < MOVE_HALT_THRESHOLD
        return ok, current_move
    except Exception as e:
        log.error(f"check_move failed: {e}. Failing safe (halt).")
        return False, float("nan")


def check_dxy() -> tuple[bool, float, str]:
    """Soft check on DXY — never halts, but tags the regime so the
    daily JSON / Jeffrey briefing can flag dollar-driven setups.

    Returns (ok, current_dxy, regime_tag) where regime_tag is one of
    'strong_usd', 'weak_usd', 'neutral_usd'. ok is always True unless
    fetch fails (in which case 'unknown').
    """
    try:
        router = _get_router()
        s = router.fetch_macro("dxy", lookback_months=1)
        current_dxy = float(s.dropna().iloc[-1])
        if current_dxy > DXY_WARN_HIGH:
            tag = "strong_usd"
            log.warning(f"DXY = {current_dxy:.2f} > {DXY_WARN_HIGH} — strong USD regime")
        elif current_dxy < DXY_WARN_LOW:
            tag = "weak_usd"
            log.warning(f"DXY = {current_dxy:.2f} < {DXY_WARN_LOW} — weak USD regime")
        else:
            tag = "neutral_usd"
        return True, current_dxy, tag
    except Exception as e:
        log.warning(f"check_dxy failed: {e}. Skipping (non-halting).")
        return True, float("nan"), "unknown"


# ---- Position / loss / hours checks ---------------------------------------
def check_position_size(proposed_notional: float, nav: float) -> tuple[bool, float]:
    """Check if proposed position is within 2% NAV limit."""
    pct = proposed_notional / nav if nav > 0 else 1.0
    ok = pct <= MAX_POSITION_PCT
    return ok, pct


def check_daily_loss(daily_pnl: float, nav: float) -> tuple[bool, float]:
    """Check if daily loss exceeds kill switch threshold."""
    loss_pct = daily_pnl / nav if nav > 0 else 0.0
    ok = loss_pct > -DAILY_LOSS_LIMIT_PCT
    return ok, loss_pct


def check_market_hours() -> tuple[bool, int]:
    """Check if we're not in the final 30 minutes before close.
    NOTE: this still uses local time — needs ET conversion before go-live.
    """
    now = datetime.now()
    close_hour, close_min = 17, 0
    minutes_to_close = (close_hour * 60 + close_min) - (now.hour * 60 + now.minute)
    ok = minutes_to_close > NO_TRADE_MINUTES_BEFORE_CLOSE
    return ok, minutes_to_close


# ---- Orchestrator ---------------------------------------------------------
def run_all_checks(proposed_notional: float, nav: float, daily_pnl: float) -> dict:
    """
    Run all risk checks. ALL hard halts must pass for execution to proceed.
    DXY is a soft indicator and never blocks execution.
    """
    vix_ok, vix_level = check_vix()
    move_ok, move_level = check_move()
    dxy_ok, dxy_level, dxy_tag = check_dxy()
    size_ok, size_pct = check_position_size(proposed_notional, nav)
    loss_ok, loss_pct = check_daily_loss(daily_pnl, nav)
    hours_ok, mins_left = check_market_hours()

    # DXY is soft — does NOT gate execution
    all_pass = all([vix_ok, move_ok, size_ok, loss_ok, hours_ok])

    return {
        "all_pass": all_pass,
        "checks": {
            "vix": {
                "pass": vix_ok, "value": vix_level,
                "threshold": VIX_HALT_THRESHOLD, "provider": "fred",
            },
            "move": {
                "pass": move_ok, "value": move_level,
                "threshold": MOVE_HALT_THRESHOLD, "provider": "lseg",
            },
            "dxy": {
                "pass": dxy_ok, "value": dxy_level, "regime": dxy_tag,
                "warn_high": DXY_WARN_HIGH, "warn_low": DXY_WARN_LOW,
                "provider": "lseg", "soft": True,
            },
            "position_size": {
                "pass": size_ok, "value": size_pct, "threshold": MAX_POSITION_PCT,
            },
            "daily_loss": {
                "pass": loss_ok, "value": loss_pct, "threshold": -DAILY_LOSS_LIMIT_PCT,
            },
            "market_hours": {
                "pass": hours_ok, "minutes_to_close": mins_left,
                "threshold": NO_TRADE_MINUTES_BEFORE_CLOSE,
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    result = run_all_checks(
        proposed_notional=5000,
        nav=250000,
        daily_pnl=-2000,
    )
    print(f"All checks pass: {result['all_pass']}")
    for name, check in result["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        val = check.get("value", "—")
        extra = f" [{check['regime']}]" if "regime" in check else ""
        print(f"  {name}: {status}  value={val}{extra}")
