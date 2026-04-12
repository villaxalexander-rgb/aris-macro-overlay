"""
execute_signals.py — A.R.I.S Phase 3 Orchestrator

Reads a daily signal JSON, resolves front-month futures contracts via
contract_resolver, computes target contract counts from `target_positions`
(signal strength on [-1, +1]) scaled by MAX_POSITION_PCT * NAV, diffs against
current IBKR positions, and places market orders for the deltas.

Safety features
---------------
- Defaults to --dry-run. Live execution requires --live.
- Idempotency guard: refuses to execute the same signal file twice in one
  trading day. State stored in logs/executed_signals.json.
- Sector concentration warning (precious_metals tilt today).
- Refuses to run if pipeline_status in signal JSON is "FAILED".
- Logs every order intent + fill to logs/trade_log.csv.

Position sizing
---------------
    notional_per_asset = target_position * MAX_POSITION_PCT * NAV
    contract_qty       = round(notional_per_asset / (price * multiplier))

target_position is the regime-weighted BSV composite, range roughly [-1, +1].
A composite of 1.0 with NAV $1M and MAX_POSITION_PCT=0.02 means $20k notional
on that asset.

CLI
---
    python -m execution.execute_signals                  # dry-run today
    python -m execution.execute_signals --date 2026-04-06
    python -m execution.execute_signals --live           # actually place orders
"""
from __future__ import annotations


import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from ib_async import IB, MarketOrder

from config.settings import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID,
    MAX_POSITION_PCT, SIGNAL_OUTPUT_PATH, TRADE_LOG_PATH,
)

# Sizing interpretation
# ---------------------
# `target_positions` from daily_signals.py is the regime-weighted BSV composite,
# range roughly [-1, +1]. Treat it as a fraction of NAV (so target=0.20 means
# 20% NAV notional in that asset). The 2% MAX_POSITION_PCT in settings.py is a
# RISK-LAYER cap, not a sizing scalar — applied as a hard ceiling in the
# risk_layer module, not here. On a $1M paper account a single CL contract is
# already ~$113k notional (11.3% of NAV), so the 2% cap is incompatible with
# trading full-size CL — that's a known design issue tracked for Phase 3
# (resolution: switch to micro contracts MCL/MGC/MZC, or relax cap, or run
# at higher NAV).
SIZING_NAV_FRACTION = 1.0  # multiplier on target_pct → notional
from execution.contract_resolver import resolve_all, CONTRACT_SPECS, MICRO_SPECS
from risk_layer.risk_checks import check_portfolio, HALT, flatten_positions

EXECUTED_LOG_PATH = "logs/executed_signals.json"
SECTOR_MAP = {
    "CL": "energy", "BZ": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "precious_metals", "SI": "precious_metals", "PL": "precious_metals", "PA": "precious_metals",
    "HG": "industrial",
    "ZC": "grains", "ZW": "grains", "ZS": "grains", "ZM": "grains", "ZL": "grains",
    "CT": "softs", "KC": "softs", "SB": "softs", "CC": "softs",
    "LE": "livestock", "GF": "livestock", "HE": "livestock",
}
SECTOR_GROSS_WARN_PCT = 0.40   # warn if any sector exceeds 40% of gross


@dataclass
class OrderIntent:
    canonical: str
    local_symbol: str
    target_qty: int
    current_qty: int
    delta_qty: int
    action: str          # BUY | SELL | NONE
    price: float
    multiplier: float
    notional_target: float
    target_pct: float

    def to_row(self):
        return {
            "timestamp": datetime.now().isoformat(),
            **asdict(self),
        }


def load_signals(date_str: str) -> dict:
    path = Path(SIGNAL_OUTPUT_PATH) / f"{date_str}_signals.json"
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    with open(path) as f:
        return json.load(f)


def check_idempotency(date_str: str, force: bool) -> None:
    """Refuse to execute the same signal date twice unless --force."""
    Path("logs").mkdir(exist_ok=True)
    if not Path(EXECUTED_LOG_PATH).exists():
        return
    with open(EXECUTED_LOG_PATH) as f:
        executed = json.load(f)
    if date_str in executed and not force:
        raise RuntimeError(
            f"Signal date {date_str} already executed at {executed[date_str]}. "
            f"Use --force to override (e.g. after a crash mid-execution)."
        )


def mark_executed(date_str: str) -> None:
    Path("logs").mkdir(exist_ok=True)
    executed = {}
    if Path(EXECUTED_LOG_PATH).exists():
        with open(EXECUTED_LOG_PATH) as f:
            executed = json.load(f)
    executed[date_str] = datetime.now().isoformat()
    with open(EXECUTED_LOG_PATH, "w") as f:
        json.dump(executed, f, indent=2)


def get_or_pin_nav_at_open(nav_now: float) -> float:
    """Read locally-pinned NAV at the start of today's ET trading session.

    If today's snapshot doesn't exist yet, create it from current NAV.
    Subsequent calls the same ET date return the pinned value.

    This is the trustworthy anchor for the daily-loss kill switch — IBKR's
    PnL.dailyPnL field can equal total unrealized PnL on currently-held
    positions (i.e. anchored at position open, not session start), which
    would false-positive every morning the book is underwater from
    cumulative drawdown. Pinning NAV locally at first call of the ET day
    makes the kill switch test intra-day discipline only.
    """
    et_today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    p = Path("state/nav_at_open.json")
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        snap = json.loads(p.read_text())
        if snap.get("et_date") == et_today:
            return float(snap["nav_at_open"])

    snap = {
        "et_date": et_today,
        "nav_at_open": nav_now,
        "written_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "source": "ib.accountValues NetLiquidationByCurrency BASE",
    }
    p.write_text(json.dumps(snap, indent=2))
    print(f"  [STATE] nav_at_open pinned: ${nav_now:,.2f} for {et_today}")
    return nav_now


def fetch_snapshot_price(ib: IB, contract) -> Optional[float]:
    """Best-effort spot price for a futures contract.

    Order of preference: live tick (if market open) -> last historical close.
    Returns None if both fail.
    """
    try:
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(2)
        for px in (ticker.last, ticker.close, ticker.marketPrice()):
            if px and not math.isnan(px) and px > 0:
                ib.cancelMktData(contract)
                return float(px)
        ib.cancelMktData(contract)
    except Exception as e:
        print(f"  [WARN] reqMktData failed for {contract.localSymbol}: {e}")

    # Fallback: last close from 1-day historical bar
    try:
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="2 D",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if bars:
            return float(bars[-1].close)
    except Exception as e:
        print(f"  [WARN] reqHistoricalData failed for {contract.localSymbol}: {e}")
    return None


def get_current_positions(ib: IB) -> dict[str, int]:
    """Returns canonical -> signed contract count for our universe.

    Reverse-maps BOTH full-size (CL, GC, ...) and micro/mini IB symbols
    (MCL, MGC, SIL, XC, ...) back to the canonical so positions opened
    via micros are not invisible on the next run. Without the micro
    side of this map, yesterday's MCL fills would be reported as flat
    and today's signal run would double the book.
    """
    positions = ib.positions()
    out: dict[str, int] = {}
    sym_to_canonical: dict[str, str] = {}
    for c, spec in CONTRACT_SPECS.items():
        sym_to_canonical[spec[0]] = c
    for c, spec in MICRO_SPECS.items():
        sym_to_canonical[spec[0]] = c
    for p in positions:
        canonical = sym_to_canonical.get(p.contract.symbol)
        if canonical:
            out[canonical] = out.get(canonical, 0) + int(p.position)
    return out


def build_order_intents(
    ib: IB,
    nav: float,
    target_positions: dict[str, float],
    contracts: dict,
    current_positions: dict[str, int],
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []
    for canonical, target_pct in target_positions.items():
        if canonical not in contracts:
            print(f"  [SKIP] {canonical}: no resolved contract")
            continue
        contract = contracts[canonical]
        multiplier = float(contract.multiplier or CONTRACT_SPECS.get(canonical, ("", "", "", 1))[3])
        price = fetch_snapshot_price(ib, contract)
        if price is None:
            print(f"  [SKIP] {canonical}: no price")
            continue
        notional_target = target_pct * SIZING_NAV_FRACTION * nav
        raw_qty = notional_target / (price * multiplier)
        target_qty = int(round(raw_qty))
        current_qty = current_positions.get(canonical, 0)
        delta = target_qty - current_qty
        action = "BUY" if delta > 0 else ("SELL" if delta < 0 else "NONE")
        intents.append(OrderIntent(
            canonical=canonical,
            local_symbol=contract.localSymbol,
            target_qty=target_qty,
            current_qty=current_qty,
            delta_qty=delta,
            action=action,
            price=price,
            multiplier=multiplier,
            notional_target=notional_target,
            target_pct=target_pct,
        ))
    return intents


def check_sector_concentration(intents: list[OrderIntent]) -> None:
    sector_gross: dict[str, float] = {}
    total_gross = 0.0
    for it in intents:
        sec = SECTOR_MAP.get(it.canonical, "other")
        gross = abs(it.notional_target)
        sector_gross[sec] = sector_gross.get(sec, 0.0) + gross
        total_gross += gross
    if total_gross == 0:
        return
    for sec, g in sorted(sector_gross.items(), key=lambda x: -x[1]):
        pct = g / total_gross
        flag = "  [WARN]" if pct > SECTOR_GROSS_WARN_PCT else "        "
        print(f"{flag} sector {sec:<16s} {pct*100:5.1f}% of gross  (${g:,.0f})")


def print_preview(intents: list[OrderIntent], nav: float) -> None:
    print(f"\n{'='*84}")
    print(f"  ORDER PREVIEW   NAV ${nav:,.0f}   sizing NAV-fraction {SIZING_NAV_FRACTION:.1f}x")
    print(f"{'='*84}")
    print(f"  {'sym':<5}{'local':<10}{'tgt%':>7}{'price':>10}{'mult':>8}"
          f"{'cur':>6}{'tgt':>6}{'delta':>7}  action")
    print(f"  {'-'*82}")
    for it in sorted(intents, key=lambda x: -abs(x.delta_qty)):
        marker = " " if it.action == "NONE" else "*"
        print(f" {marker}{it.canonical:<5}{it.local_symbol:<10}"
              f"{it.target_pct*100:>6.1f}%{it.price:>10.2f}{it.multiplier:>8.0f}"
              f"{it.current_qty:>6d}{it.target_qty:>6d}{it.delta_qty:>+7d}  {it.action}")
    n_orders = sum(1 for i in intents if i.action != "NONE")
    gross = sum(abs(i.notional_target) for i in intents)
    print(f"  {'-'*82}")
    leverage = gross / nav if nav else 0
    flag = "  [WARN] HIGH LEVERAGE" if leverage > 1.5 else ""
    print(f"  {n_orders} orders to place    gross notional ${gross:,.0f}    "
          f"({leverage*100:.1f}% of NAV){flag}")
    if leverage > 1.5:
        print(f"  Tradeable subset uses micro contracts or scaled-down NAV "
              f"(Phase 3 sizing punchlist).")


def append_trade_log(intents: list[OrderIntent], filled: dict[str, dict]) -> None:
    Path(TRADE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(TRADE_LOG_PATH).exists()
    fieldnames = [
        "timestamp", "canonical", "local_symbol", "action", "delta_qty",
        "target_qty", "current_qty", "price", "multiplier", "notional_target",
        "target_pct", "fill_status", "fill_price", "order_id",
    ]
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for it in intents:
            row = {
                "timestamp": datetime.now().isoformat(),
                "canonical": it.canonical,
                "local_symbol": it.local_symbol,
                "action": it.action,
                "delta_qty": it.delta_qty,
                "target_qty": it.target_qty,
                "current_qty": it.current_qty,
                "price": it.price,
                "multiplier": it.multiplier,
                "notional_target": it.notional_target,
                "target_pct": it.target_pct,
                "fill_status": filled.get(it.canonical, {}).get("status", "DRY_RUN"),
                "fill_price": filled.get(it.canonical, {}).get("fill_price", ""),
                "order_id": filled.get(it.canonical, {}).get("order_id", ""),
            }
            w.writerow(row)


def place_orders(ib: IB, intents: list[OrderIntent], contracts: dict) -> dict[str, dict]:
    filled: dict[str, dict] = {}
    for it in intents:
        if it.action == "NONE" or it.delta_qty == 0:
            continue
        contract = contracts[it.canonical]
        order = MarketOrder(it.action, abs(it.delta_qty))
        try:
            trade = ib.placeOrder(contract, order)
            ib.sleep(1)
            filled[it.canonical] = {
                "status": trade.orderStatus.status,
                "fill_price": trade.orderStatus.avgFillPrice or 0.0,
                "order_id": trade.order.orderId,
            }
            print(f"  [ORDER] {it.action} {abs(it.delta_qty)} {it.local_symbol} "
                  f"-> {trade.orderStatus.status}")
        except Exception as e:
            print(f"  [ERROR] {it.canonical}: {e}")
            filled[it.canonical] = {"status": f"ERROR: {e}", "fill_price": 0.0, "order_id": 0}
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="Signal file date YYYY-MM-DD (default: today)")
    ap.add_argument("--live", action="store_true",
                    help="Actually place orders. Default is dry-run.")
    ap.add_argument("--force", action="store_true",
                    help="Override idempotency guard")
    args = ap.parse_args()

    print(f"\n[execute_signals] date={args.date}  mode={'LIVE' if args.live else 'DRY-RUN'}\n")

    # Load signals
    sig = load_signals(args.date)

    # Hard gate: refuse to trade against a degraded signal file.
    health_info = sig.get("health", {})
    health_ok = health_info.get("healthy", True)  # boolean from HealthRecord
    if not health_ok and not args.force:
        degradations = health_info.get("degradations", [])
        print(f"[BLOCKED] Signal health check failed.")
        if degradations:
            for d in degradations:
                print(f"  - {d}")
        print(f"Pass --force to override.")
        return 4

    target_positions = sig.get("target_positions", {})
    if not target_positions:
        print("  [ABORT] Empty target_positions in signal file")
        return
    health = sig.get("health", {})
    print(f"  Regime: {sig.get('regime', {}).get('regime', '?')}")
    print(f"  Pipeline health: {'OK' if health.get('healthy', True) else 'DEGRADED'}")
    if health.get("disagreements"):
        print(f"  Cross-validation disagreements: {len(health['disagreements'])}")
    print(f"  Targets: {len(target_positions)} assets\n")

    if args.live:
        check_idempotency(args.date, args.force)


    # Connect IBKR
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
    print(f"  Connected to IBKR {IBKR_HOST}:{IBKR_PORT}")

    try:
        nav = 0.0
        for s in ib.accountSummary():
            if s.tag == "NetLiquidation":
                nav = float(s.value); break
        print(f"  NAV: ${nav:,.2f}\n")

        # Resolve contracts
        print("  Resolving front-month contracts...")
        contracts = resolve_all(ib, list(target_positions.keys()))
        print(f"  Resolved {len(contracts)}/{len(target_positions)}\n")

        # Current positions
        current_positions = get_current_positions(ib)
        if current_positions:
            print(f"  Current positions: {current_positions}\n")
        else:
            print("  Current positions: (none)\n")

        # Build intents
        intents = build_order_intents(ib, nav, target_positions, contracts, current_positions)
        print_preview(intents, nav)

        # Sector concentration
        print(f"\n  Sector concentration:")
        check_sector_concentration(intents)

        # Daily PnL — anchored locally to today's NAV at first run of the ET
        # day. IBKR's PnL.dailyPnL is kept only as a cross-check; see
        # get_or_pin_nav_at_open() for rationale.
        nav_at_open = get_or_pin_nav_at_open(nav)
        daily_pnl_local = nav - nav_at_open
        daily_pnl_pct = daily_pnl_local / nav_at_open if nav_at_open > 0 else 0.0
        print(f"  NAV at open (ET): ${nav_at_open:,.2f}")
        print(f"  Daily PnL (local): ${daily_pnl_local:+,.2f} ({daily_pnl_pct*100:+.2f}%)")

        # IBKR's number kept as a cross-check, not as the gate
        ib_daily_pnl = 0.0
        try:
            accounts = ib.managedAccounts()
            if accounts:
                ib.reqPnL(accounts[0])
                ib.sleep(1.5)
                pnls = ib.pnl()
                if pnls:
                    ib_daily_pnl = float(pnls[0].dailyPnL or 0.0)
                ib.cancelPnL(accounts[0])
        except Exception:
            pass
        if abs(ib_daily_pnl - daily_pnl_local) > 1000:
            print(f"  [INFO] IBKR dailyPnL=${ib_daily_pnl:+,.2f} diverges from local "
                  f"by ${ib_daily_pnl - daily_pnl_local:+,.2f} — using local")

        # Risk layer gate (fed the trustworthy local number)
        risk_report = check_portfolio(intents, nav=nav, daily_pnl=daily_pnl_local)
        print(f"\nRisk layer: {'PASS' if risk_report['all_pass'] else 'HALT'}")
        for name, r in risk_report["checks"].items():
            status = "PASS" if r.get("pass", True) else "FAIL"
            print(f"  {name:<22} {status}  {r.get('value', '—')}")

        if risk_report["status"] == HALT:
            print("\n[HALT] Risk layer blocked execution. No orders placed.")
            if args.live and risk_report.get("flatten_required"):
                print("[KILL-SWITCH] Daily loss limit breached — flattening book.")
                open_contracts = [
                    contracts[i.canonical]
                    for i in intents
                    if i.current_qty != 0 and i.canonical in contracts
                ]
                flatten_positions(ib, open_contracts)
            return 3

        # Execute or dry-run
        if args.live:
            print(f"\n  [LIVE] Placing {sum(1 for i in intents if i.action != 'NONE')} orders...")
            filled = place_orders(ib, intents, contracts)
            append_trade_log(intents, filled)
            mark_executed(args.date)
            print(f"\n  [DONE] Logged to {TRADE_LOG_PATH}")
        else:
            append_trade_log(intents, {})
            print(f"\n  [DRY-RUN] No orders placed. Log: {TRADE_LOG_PATH}")
            print(f"  Re-run with --live to actually execute.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
