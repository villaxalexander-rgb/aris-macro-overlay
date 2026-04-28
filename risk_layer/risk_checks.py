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

Phase 3 additions:
  - Sector gross cap (SECTOR_GROSS_CAP_PCT, default 40% of NAV).
  - Per-contract notional guard (MAX_PER_CONTRACT_NOTIONAL_PCT, default 3%)
    drops contracts whose single-unit notional would dominate NAV
    (SB, HE, CT, KC etc. on $1M paper).
  - ET-aware market hours via zoneinfo.America/New_York.
  - check_portfolio(intents, nav, daily_pnl) — portfolio-level orchestrator
    returning an OK/HALT status and per-check details for audit.
  - flatten_positions(ib, contracts) — hard kill-switch that emits
    closing MarketOrders for any non-zero position.
"""
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

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

# ---- Phase 3 constants ----------------------------------------------------
SECTOR_GROSS_CAP_PCT = 0.40
MAX_PER_CONTRACT_NOTIONAL_PCT = 0.03
ET = ZoneInfo("America/New_York")

# Status tokens
OK = "OK"
HALT = "HALT"

# Sector map — kept in sync with execution/execute_signals.py
SECTOR_MAP: dict[str, str] = {
    "CL": "energy", "BZ": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "precious_metals", "SI": "precious_metals",
    "PL": "precious_metals", "PA": "precious_metals",
    "HG": "industrial",
    "ZC": "grains", "ZW": "grains", "ZS": "grains", "ZM": "grains", "ZL": "grains",
    "CT": "softs", "KC": "softs", "SB": "softs", "CC": "softs",
    "LE": "livestock", "GF": "livestock", "HE": "livestock",
}

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
    """Check if we're not in the final NO_TRADE_MINUTES_BEFORE_CLOSE window
    before the CME Globex daily settlement (17:00 ET).

    Futures trade nearly 24h, but we avoid the settlement window because
    fill quality degrades and the overlay is long-horizon.
    """
    now_et = datetime.now(ET)
    close_hour, close_min = 17, 0  # 5pm ET, CME daily settlement
    minutes_to_close = (close_hour * 60 + close_min) - (now_et.hour * 60 + now_et.minute)
    if minutes_to_close < 0:
        # After settlement — treat as "fresh session" tomorrow, OK to trade
        minutes_to_close += 24 * 60
    ok = minutes_to_close > NO_TRADE_MINUTES_BEFORE_CLOSE
    return ok, minutes_to_close


# ---- Phase 3: sector caps, per-contract notional, portfolio orchestrator --
def check_sector_caps(intents, nav: float) -> tuple[bool, dict]:
    """Verify no sector's gross notional exceeds SECTOR_GROSS_CAP_PCT of NAV.

    `intents` is any iterable of objects exposing .sector, .target_contracts,
    .price, .multiplier (matches execution.execute_signals.OrderIntent).
    Returns (all_pass, {sector: {"gross": usd, "pct": fraction, "pass": bool}}).
    """
    if nav <= 0:
        return False, {"error": "non-positive nav"}

    gross_by_sector: dict[str, float] = {}
    for i in intents:
        sector = getattr(i, "sector", None) or SECTOR_MAP.get(
            getattr(i, "canonical", ""), "other"
        )
        # Accept either `target_contracts` (Phase 3 spec) or `target_qty`
        # (current OrderIntent dataclass in execute_signals.py).
        target = getattr(i, "target_contracts", None)
        if target is None:
            target = getattr(i, "target_qty", 0)
        gross = abs(int(target)) * float(getattr(i, "price", 0)) \
            * float(getattr(i, "multiplier", 0))
        gross_by_sector[sector] = gross_by_sector.get(sector, 0.0) + gross

    report: dict[str, dict] = {}
    all_pass = True
    for sector, gross in gross_by_sector.items():
        pct = gross / nav
        passed = pct <= SECTOR_GROSS_CAP_PCT
        report[sector] = {"gross": gross, "pct": pct, "pass": passed}
        if not passed:
            all_pass = False
            log.warning(
                f"Sector {sector} gross {pct*100:.1f}% > cap "
                f"{SECTOR_GROSS_CAP_PCT*100:.0f}%"
            )
    return all_pass, report


def check_per_contract_notional(
    canonical: str,
    price: float,
    multiplier: float,
    nav: float,
) -> tuple[bool, float]:
    """Flag if ONE contract already exceeds MAX_PER_CONTRACT_NOTIONAL_PCT.

    At $1M NAV with a 3% cap this drops SB (~$95k/contract), HE (~$40k),
    CT (~$35k), KC (~$80k) and similar lumpy singletons from the
    tradeable subset — use micros or skip.
    """
    if nav <= 0:
        return False, float("inf")
    single = price * multiplier
    pct = single / nav
    ok = pct <= MAX_PER_CONTRACT_NOTIONAL_PCT
    if not ok:
        log.warning(
            f"{canonical}: 1-contract notional ${single:,.0f} "
            f"({pct*100:.2f}%) exceeds {MAX_PER_CONTRACT_NOTIONAL_PCT*100:.0f}% cap"
        )
    return ok, pct


def check_portfolio(intents, nav: float, daily_pnl: float = 0.0) -> dict:
    """Portfolio-level risk orchestrator. Returns a structured report with
    a top-level `status` (OK|HALT) and `all_pass` boolean. Any hard halt
    (VIX, MOVE, daily loss, sector cap, market hours) sets HALT.

    Soft warnings (DXY regime, per-contract notional on individual legs)
    do not halt but are surfaced in the report.
    """
    vix_ok, vix_level = check_vix()
    move_ok, move_level = check_move()
    dxy_ok, dxy_level, dxy_tag = check_dxy()
    loss_ok, loss_pct = check_daily_loss(daily_pnl, nav)
    hours_ok, mins_left = check_market_hours()
    sectors_ok, sector_report = check_sector_caps(intents, nav)

    # Per-leg notional — advisory only (the executor already picked micros
    # where it could). Flag individual legs that still blow the cap.
    per_leg: dict[str, dict] = {}
    for i in intents:
        canonical = getattr(i, "canonical", "?")
        ok, pct = check_per_contract_notional(
            canonical,
            float(getattr(i, "price", 0)),
            float(getattr(i, "multiplier", 0)),
            nav,
        )
        per_leg[canonical] = {"pass": ok, "pct": pct}

    hard_halts = [vix_ok, move_ok, loss_ok, hours_ok, sectors_ok]
    all_pass = all(hard_halts)
    status = OK if all_pass else HALT
    flatten_required = not loss_ok  # only the daily-loss breach triggers flatten

    return {
        "status": status,
        "all_pass": all_pass,
        "flatten_required": flatten_required,
        "checks": {
            "vix": {"pass": vix_ok, "value": vix_level, "threshold": VIX_HALT_THRESHOLD},
            "move": {"pass": move_ok, "value": move_level, "threshold": MOVE_HALT_THRESHOLD},
            "dxy": {"pass": dxy_ok, "value": dxy_level, "regime": dxy_tag, "soft": True},
            "daily_loss": {
                "pass": loss_ok, "value": loss_pct,
                "threshold": -DAILY_LOSS_LIMIT_PCT,
            },
            "market_hours": {
                "pass": hours_ok, "value": mins_left,
                "threshold": NO_TRADE_MINUTES_BEFORE_CLOSE,
            },
            "sector_caps": {
                "pass": sectors_ok, "value": sector_report,
                "threshold": SECTOR_GROSS_CAP_PCT,
            },
            "per_contract_notional": {
                "pass": all(v["pass"] for v in per_leg.values()) if per_leg else True,
                "value": per_leg,
                "threshold": MAX_PER_CONTRACT_NOTIONAL_PCT,
                "soft": True,
            },
        },
        "timestamp": datetime.now(ET).isoformat(),
    }


def flatten_positions(ib, contracts) -> list[dict]:
    """Kill-switch: submit closing MarketOrders for any non-zero position.

    `ib` is an ib_async.IB instance; `contracts` is an iterable of Future
    contract objects to consider. Positions not in `contracts` are ignored
    (caller should pass the full resolved set).
    """
    from ib_async import MarketOrder  # local import to avoid hard dep at import

    locals_of_interest = {c.localSymbol for c in contracts if c is not None}
    fills: list[dict] = []
    for pos in ib.positions():
        if pos.contract.secType != "FUT":
            continue
        if pos.contract.localSymbol not in locals_of_interest:
            continue
        qty = int(pos.position)
        if qty == 0:
            continue
        action = "SELL" if qty > 0 else "BUY"
        order = MarketOrder(action, abs(qty))
        trade = ib.placeOrder(pos.contract, order)
        ib.sleep(1.0)
        fills.append({
            "local": pos.contract.localSymbol,
            "action": action,
            "qty": abs(qty),
            "status": trade.orderStatus.status,
            "fill_price": trade.orderStatus.avgFillPrice,
        })
        log.warning(
            f"[FLATTEN] {pos.contract.localSymbol} {action} {abs(qty)} "
            f"status={trade.orderStatus.status}"
        )
    return fills


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
