"""
Offline tests for the Phase 3 additions to risk_layer/risk_checks.py.

These tests do NOT touch IB Gateway, FRED, or LSEG. They exercise the pure
functions (sector caps, per-contract notional, ET market hours) and
monkey-patch the macro checks so check_portfolio can be tested
deterministically.

Run from the repo root:
    python -m tests.test_risk_layer_phase3
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

# Allow `python -m tests.test_risk_layer_phase3` from repo root
# with no installed package.
sys.path.insert(0, ".")

import pytest

from risk_layer import risk_checks as rc


NAV = 1_000_000.0


# ---- Pytest fixture -------------------------------------------------------
@pytest.fixture
def monkeypatch_time():
    """Swap rc.datetime for a frozen ET time; restore after the test."""
    original_dt = rc.datetime

    def _patch(hour: int, minute: int):
        class _FrozenDT:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 4, 8, hour, minute, tzinfo=tz or ET)
        rc.datetime = _FrozenDT  # type: ignore

    yield _patch
    rc.datetime = original_dt
ET = ZoneInfo("America/New_York")


def _intent(canonical, sector, target_contracts, price, multiplier):
    """Minimal stand-in for execution.execute_signals.OrderIntent."""
    return SimpleNamespace(
        canonical=canonical,
        sector=sector,
        target_contracts=target_contracts,
        price=price,
        multiplier=multiplier,
    )


# ---- check_sector_caps ----------------------------------------------------
def test_sector_caps_pass():
    # One MCL contract at $108 with multiplier 100 = $10,800 notional.
    intents = [
        _intent("CL", "energy", 10, 108.0, 100),   # $108k, 10.8% NAV
        _intent("GC", "precious_metals", 5, 3200.0, 10),  # $160k, 16% NAV
    ]
    ok, report = rc.check_sector_caps(intents, NAV)
    assert ok is True, report
    assert report["energy"]["pass"] is True
    assert report["precious_metals"]["pass"] is True
    print("  sector_caps_pass: OK")


def test_sector_caps_fail_energy():
    # 50 MCL + 30 MGC sized to blow the energy cap.
    intents = [
        _intent("CL", "energy", 50, 108.0, 100),   # $540k, 54% NAV
        _intent("BZ", "energy", 20, 103.0, 100),   # $206k, 20.6% NAV
    ]
    ok, report = rc.check_sector_caps(intents, NAV)
    assert ok is False
    assert report["energy"]["pass"] is False
    assert report["energy"]["pct"] > rc.SECTOR_GROSS_CAP_PCT
    print(f"  sector_caps_fail_energy: OK (energy {report['energy']['pct']*100:.1f}%)")


def test_sector_caps_empty():
    ok, report = rc.check_sector_caps([], NAV)
    assert ok is True
    assert report == {}
    print("  sector_caps_empty: OK")


# ---- check_per_contract_notional ------------------------------------------
def test_per_contract_notional_pass_micro():
    # MCL: 108 * 100 = $10,800 = 1.08% — well under 3%
    ok, pct = rc.check_per_contract_notional("CL", 108.0, 100, NAV)
    assert ok is True
    assert pct < rc.MAX_PER_CONTRACT_NOTIONAL_PCT
    print(f"  per_contract_notional_pass_micro: OK (pct={pct*100:.2f}%)")


def test_per_contract_notional_fail_full_he():
    # Full HE (lean hogs): $0.85/lb * 40,000 lbs = $34,000 = 3.4% NAV > 3% → fail
    ok, pct = rc.check_per_contract_notional("HE", 0.85, 40_000, NAV)
    assert ok is False, f"expected fail, got pct={pct*100:.2f}%"
    assert pct > rc.MAX_PER_CONTRACT_NOTIONAL_PCT
    print(f"  per_contract_notional_fail_he: OK (pct={pct*100:.2f}%)")


def test_per_contract_notional_nav_zero():
    ok, pct = rc.check_per_contract_notional("CL", 108.0, 100, 0)
    assert ok is False
    assert pct == float("inf")
    print("  per_contract_notional_nav_zero: OK")


# ---- check_market_hours ---------------------------------------------------
def test_market_hours_pre_settlement(monkeypatch_time):
    # 10:00 ET should be plenty of minutes before 17:00 close
    monkeypatch_time(10, 0)
    ok, mins = rc.check_market_hours()
    assert ok is True
    assert mins == 7 * 60  # 420
    print(f"  market_hours_pre_settlement: OK (mins_to_close={mins})")


def test_market_hours_blackout(monkeypatch_time):
    # 16:45 ET — within 30-min no-trade window
    monkeypatch_time(16, 45)
    ok, mins = rc.check_market_hours()
    assert ok is False
    assert mins == 15
    print(f"  market_hours_blackout: OK (mins_to_close={mins})")


def test_market_hours_after_settlement(monkeypatch_time):
    # 17:30 ET — after daily settlement, treated as fresh next session
    monkeypatch_time(17, 30)
    ok, mins = rc.check_market_hours()
    assert ok is True
    assert mins > rc.NO_TRADE_MINUTES_BEFORE_CLOSE
    print(f"  market_hours_after_settlement: OK (mins_to_close={mins})")


# ---- check_portfolio orchestrator -----------------------------------------
def test_portfolio_all_pass(monkeypatch_time):
    monkeypatch_time(10, 0)
    # Patch the macro checks that would otherwise hit network
    rc.check_vix = lambda: (True, 18.0)
    rc.check_move = lambda: (True, 95.0)
    rc.check_dxy = lambda: (True, 102.0, "neutral_usd")

    intents = [
        _intent("CL", "energy", 10, 108.0, 100),
        _intent("GC", "precious_metals", 5, 3200.0, 10),
    ]
    report = rc.check_portfolio(intents, nav=NAV, daily_pnl=-5_000)
    assert report["status"] == rc.OK, report
    assert report["all_pass"] is True
    assert report["flatten_required"] is False
    assert report["checks"]["sector_caps"]["pass"] is True
    print(f"  portfolio_all_pass: OK (status={report['status']})")


def test_portfolio_halt_on_daily_loss(monkeypatch_time):
    monkeypatch_time(10, 0)
    rc.check_vix = lambda: (True, 18.0)
    rc.check_move = lambda: (True, 95.0)
    rc.check_dxy = lambda: (True, 102.0, "neutral_usd")

    intents = [_intent("CL", "energy", 5, 108.0, 100)]
    report = rc.check_portfolio(intents, nav=NAV, daily_pnl=-50_000)  # -5% > 3% limit
    assert report["status"] == rc.HALT
    assert report["checks"]["daily_loss"]["pass"] is False
    assert report["flatten_required"] is True
    print(f"  portfolio_halt_daily_loss: OK (flatten={report['flatten_required']})")


def test_portfolio_halt_on_sector_cap(monkeypatch_time):
    monkeypatch_time(10, 0)
    rc.check_vix = lambda: (True, 18.0)
    rc.check_move = lambda: (True, 95.0)
    rc.check_dxy = lambda: (True, 102.0, "neutral_usd")

    intents = [
        _intent("CL", "energy", 60, 108.0, 100),  # $648k = 64.8% NAV
    ]
    report = rc.check_portfolio(intents, nav=NAV, daily_pnl=0)
    assert report["status"] == rc.HALT
    assert report["checks"]["sector_caps"]["pass"] is False
    assert report["flatten_required"] is False  # daily loss is fine
    print("  portfolio_halt_sector_cap: OK")


def test_portfolio_halt_on_vix(monkeypatch_time):
    monkeypatch_time(10, 0)
    rc.check_vix = lambda: (False, 42.0)
    rc.check_move = lambda: (True, 95.0)
    rc.check_dxy = lambda: (True, 102.0, "neutral_usd")

    intents = [_intent("CL", "energy", 5, 108.0, 100)]
    report = rc.check_portfolio(intents, nav=NAV, daily_pnl=0)
    assert report["status"] == rc.HALT
    assert report["checks"]["vix"]["pass"] is False
    print("  portfolio_halt_vix: OK")


# ---- Runner ---------------------------------------------------------------
def _make_time_monkeypatcher():
    """Returns a function that swaps rc.datetime for a frozen ET time."""
    def _patch(hour: int, minute: int):
        class _FrozenDT:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 4, 8, hour, minute, tzinfo=tz or ET)
        rc.datetime = _FrozenDT  # type: ignore
    return _patch


def main():
    monkeypatch_time = _make_time_monkeypatcher()

    # Save originals so we can restore (the portfolio tests overwrite them)
    original = {
        "check_vix": rc.check_vix,
        "check_move": rc.check_move,
        "check_dxy": rc.check_dxy,
        "datetime": rc.datetime,
    }

    tests = [
        ("test_sector_caps_pass",              lambda: test_sector_caps_pass()),
        ("test_sector_caps_fail_energy",       lambda: test_sector_caps_fail_energy()),
        ("test_sector_caps_empty",             lambda: test_sector_caps_empty()),
        ("test_per_contract_notional_pass",    lambda: test_per_contract_notional_pass_micro()),
        ("test_per_contract_notional_fail_he", lambda: test_per_contract_notional_fail_full_he()),
        ("test_per_contract_notional_nav_0",   lambda: test_per_contract_notional_nav_zero()),
        ("test_market_hours_pre_settlement",   lambda: test_market_hours_pre_settlement(monkeypatch_time)),
        ("test_market_hours_blackout",         lambda: test_market_hours_blackout(monkeypatch_time)),
        ("test_market_hours_after_settlement", lambda: test_market_hours_after_settlement(monkeypatch_time)),
        ("test_portfolio_all_pass",            lambda: test_portfolio_all_pass(monkeypatch_time)),
        ("test_portfolio_halt_daily_loss",     lambda: test_portfolio_halt_on_daily_loss(monkeypatch_time)),
        ("test_portfolio_halt_sector_cap",     lambda: test_portfolio_halt_on_sector_cap(monkeypatch_time)),
        ("test_portfolio_halt_vix",            lambda: test_portfolio_halt_on_vix(monkeypatch_time)),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        try:
            print(f"[{name}]")
            fn()
            passed += 1
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
            print(f"  FAIL: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR: {type(e).__name__}: {e}")

    # Restore originals
    for k, v in original.items():
        setattr(rc, k, v)

    print(f"\n=== {passed}/{len(tests)} passed ===")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
