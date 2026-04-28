"""
Contract Resolver — maps A.R.I.S canonical commodity symbols to qualified
ib_async Future contracts with automatic front-month rolling.

Why this exists:
  Calling Future('CL', exchange='NYMEX') with no expiry will fail to qualify
  because the WTI chain has 130+ contracts. Every futures order needs an
  explicit lastTradeDateOrContractMonth. Hardcoding expiries means manual
  intervention every month at roll. This module discovers the front month
  at runtime via ib.reqContractDetails() and skips contracts within
  ROLL_BUFFER_DAYS of expiry to avoid delivery risk and roll-week chop.

Usage:
    from ib_async import IB
    from execution.contract_resolver import resolve_all
    ib = IB(); ib.connect('127.0.0.1', 4002, clientId=1)
    contracts = resolve_all(ib, ['CL', 'BZ', 'GC'])
    # contracts['CL'] is a fully-qualified Future ready for placeOrder
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from ib_async import IB, Future, ContractDetails

# Days before expiry to skip a contract. Front-month gets rolled to next
# month if it's within this window. Standard practice is 5-7 days.
# Bumped from 5 to 7 after HE Apr'26 came within the buffer's edge on 2026-04-07.
ROLL_BUFFER_DAYS = 7

# Micro toggle. When True, resolve_front_month prefers micro/mini contracts
# (MCL, MGC, SIL, etc.) over their full-size counterparts. This is a hard
# requirement for running the strategy on a $1M paper account — a single
# full-size CL contract is ~11% of NAV and blows through the 2% MAX_POSITION_PCT
# cap on its own, making the risk layer impossible to honor without micros.
# Set via env var USE_MICROS (default "true"). Set to "false" for production
# run at higher NAV where full-size contracts fit the sizing envelope.
USE_MICROS = os.getenv("USE_MICROS", "true").lower() in ("true", "1", "yes")

# Canonical -> IBKR contract specs.
# Format: (ib_symbol, exchange, currency, multiplier_hint)
# multiplier_hint is informational only — IBKR returns the real multiplier
# in ContractDetails. We just use it for sanity-checking position sizing.
CONTRACT_SPECS: dict[str, tuple[str, str, str, float]] = {
    # Energy — NYMEX
    "CL": ("CL", "NYMEX", "USD", 1000),     # WTI Crude, 1000 bbl
    "BZ": ("BZ", "NYMEX", "USD", 1000),     # Brent (financial), 1000 bbl
    "NG": ("NG", "NYMEX", "USD", 10000),    # Natural Gas, 10000 mmBtu
    "HO": ("HO", "NYMEX", "USD", 42000),    # Heating Oil, 42000 gal
    "RB": ("RB", "NYMEX", "USD", 42000),    # RBOB Gasoline, 42000 gal

    # Precious + industrial metals — COMEX (GC, SI, HG) / NYMEX (PL, PA)
    "GC": ("GC", "COMEX", "USD", 100),      # Gold, 100 oz
    "SI": ("SI", "COMEX", "USD", 5000),     # Silver, 5000 oz
    "PL": ("PL", "NYMEX", "USD", 50),       # Platinum, 50 oz
    "PA": ("PA", "NYMEX", "USD", 100),      # Palladium, 100 oz
    "HG": ("HG", "COMEX", "USD", 25000),    # Copper, 25000 lb

    # Grains — CBOT
    "ZC": ("ZC", "CBOT", "USD", 5000),      # Corn, 5000 bu
    "ZW": ("ZW", "CBOT", "USD", 5000),      # Wheat, 5000 bu
    "ZS": ("ZS", "CBOT", "USD", 5000),      # Soybeans, 5000 bu
    "ZM": ("ZM", "CBOT", "USD", 100),       # Soybean Meal, 100 short tons
    "ZL": ("ZL", "CBOT", "USD", 60000),     # Soybean Oil, 60000 lb
}

# Softs — ICE US (IBKR exchange code: NYBOT)
CONTRACT_SPECS.update({
    "CT": ("CT", "NYBOT", "USD", 50000),    # Cotton, 50000 lb
    "KC": ("KC", "NYBOT", "USD", 37500),    # Coffee, 37500 lb
    "SB": ("SB", "NYBOT", "USD", 112000),   # Sugar #11, 112000 lb
    "CC": ("CC", "NYBOT", "USD", 10),       # Cocoa, 10 metric tons

    # Livestock — CME
    "LE": ("LE", "CME", "USD", 40000),      # Live Cattle, 40000 lb
    "GF": ("GF", "CME", "USD", 50000),      # Feeder Cattle, 50000 lb
    "HE": ("HE", "CME", "USD", 40000),      # Lean Hogs, 40000 lb
})


# Canonical -> micro/mini contract specs.
# Same format as CONTRACT_SPECS. KEPT MINIMAL ON PURPOSE: every entry must
# be verified as tradeable on the connected IBKR gateway before landing
# here. With the symmetric fallback in resolve_front_month, an unverified
# entry no longer crashes resolution — but it would silently route the
# canonical through the full-size fallback every day, which is the wrong
# default for any leg whose full-size per-contract notional blows the
# 3% MAX_PER_CONTRACT cap (e.g. SI is ~$160k/contract on $925k NAV).
#
# Verified resolving on IBKR paper as of 2026-04-08:
#   MCL  - Micro WTI Crude,    100 bbl    (1/10 of CL)  - NYMEX
#   QG   - E-mini Natural Gas, 2500 mmBtu (1/4  of NG)  - NYMEX
#   MGC  - Micro Gold,         10 oz      (1/10 of GC)  - COMEX
#   MHG  - Micro Copper,       2500 lb    (1/10 of HG)  - COMEX
#
# Removed 2026-04-08 (do not re-add without TWS contract-search confirmation):
#   SIL (SI) - COMEX returned 0 contractDetails on the paper gateway. CME
#              still lists Micro Silver, but until IBKR confirms availability
#              on this account, leaving it absent means SI is not traded
#              (preferable to silently routing to full-size 5000-oz SI).
#   XC (ZC), XW (ZW), XK (ZS) - CME e-mini grains, delisted ~2018. These
#              will never resolve on any venue. Do not re-add.
#
# No micros exist for: BZ, HO, RB, PL, PA, ZM, ZL, CT, KC, SB, CC, LE, GF, HE
# These stay full-size. The risk_layer is responsible for dropping any asset
# whose per-contract notional exceeds MAX_PER_CONTRACT_PCT of NAV.
MICRO_SPECS: dict[str, tuple[str, str, str, float]] = {
    "CL": ("MCL", "NYMEX", "USD", 100),
    "NG": ("QG",  "NYMEX", "USD", 2500),
    "GC": ("MGC", "COMEX", "USD", 10),
    "HG": ("MHG", "COMEX", "USD", 2500),
}


def _spec_for(canonical: str) -> Optional[tuple[str, str, str, float]]:
    """Return the IBKR spec for a canonical, preferring micros when enabled."""
    if USE_MICROS and canonical in MICRO_SPECS:
        return MICRO_SPECS[canonical]
    return CONTRACT_SPECS.get(canonical)


def is_micro(canonical: str) -> bool:
    """True if this canonical is currently being traded as a micro/mini."""
    return USE_MICROS and canonical in MICRO_SPECS


def _parse_expiry(s: str) -> Optional[datetime]:
    """IBKR returns lastTradeDateOrContractMonth as YYYYMMDD or YYYYMM."""
    if not s:
        return None
    try:
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        if len(s) == 6:
            # Month-only — treat as last day of month for safety
            dt = datetime.strptime(s, "%Y%m")
            # Move to last day of that month
            if dt.month == 12:
                return dt.replace(day=31)
            return (dt.replace(month=dt.month + 1)) - timedelta(days=1)
    except ValueError:
        pass
    return None


def _try_resolve(
    ib: IB,
    spec: tuple[str, str, str, float],
    cutoff: datetime,
) -> Optional[Future]:
    """Attempt to find the front-month contract for a single (symbol, exchange)
    spec. Returns None for ANY failure mode — either the venue returned no
    contract definitions at all (symbol not listed on this gateway), or the
    venue returned definitions but none survived the secType / strict-symbol /
    expiry-cutoff filtering. Caller decides whether to attempt a fallback spec.
    """
    ib_symbol, exchange, currency, _ = spec
    template = Future(symbol=ib_symbol, exchange=exchange, currency=currency)
    details: list[ContractDetails] = ib.reqContractDetails(template)
    if not details:
        return None

    candidates: list[tuple[datetime, Future]] = []
    for d in details:
        c = d.contract
        # ib_async returns base Contract with secType='FUT', not Future subclass.
        # Filter on secType, not isinstance.
        if getattr(c, "secType", None) != "FUT":
            continue
        # CRITICAL: strict symbol match. IBKR's reqContractDetails does a
        # prefix match, so symbol='SI' returns both SI (5000oz silver) and
        # SIL (1000oz micro silver). Without this filter, we'd accidentally
        # trade the wrong-sized contract. Same hazard for GC/MGC, ZC/MZC,
        # CL/QM, etc.
        if c.symbol != ib_symbol:
            continue
        # Multiplier-integrity guard: even when c.symbol matches, IBKR's
        # COMEX silver chain has been observed to return Micro Silver
        # (SILJ6, mult=1000) under base symbol 'SI'. Compare the contract's
        # reported multiplier to the spec we requested (spec[3]) and reject
        # mismatches so we never silently route into the wrong contract size.
        try:
            c_mult = float(c.multiplier) if c.multiplier else 0.0
        except (ValueError, TypeError):
            c_mult = 0.0
        if c_mult != float(spec[3]):
            continue
        expiry = _parse_expiry(c.lastTradeDateOrContractMonth)
        if expiry is None or expiry < cutoff:
            continue
        candidates.append((expiry, c))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def resolve_front_month(
    ib: IB,
    canonical: str,
    roll_buffer_days: int = ROLL_BUFFER_DAYS,
) -> Optional[Future]:
    """Find the front-month Future contract for a canonical symbol.

    Returns a fully-qualified Future ready for placeOrder, or None if no
    suitable contract was found within the chain. When USE_MICROS is True
    and the canonical has a micro spec, the micro is tried first; on ANY
    failure mode (venue returns nothing, or returns details but all get
    filtered out) the resolver falls back to the full-size contract. This
    keeps the fallback symmetric — the previous early-return on empty
    `details` was a dead-code path that left SI/ZC/ZW/ZS unresolved on
    2026-04-08 even though full-size contracts exist on the venue.
    """
    spec = _spec_for(canonical)
    if spec is None:
        raise ValueError(f"No contract spec for canonical symbol: {canonical}")
    using_micro = is_micro(canonical)
    cutoff = datetime.now() + timedelta(days=roll_buffer_days)

    contract = _try_resolve(ib, spec, cutoff)
    if contract is not None:
        return contract

    # Fallback to full-size when the micro attempt yielded nothing — fires
    # for BOTH "venue didn't list the micro at all" and "venue listed it
    # but all candidates were filtered out". The risk layer will then
    # decide whether the full-size notional is acceptable for current NAV.
    if using_micro and canonical in CONTRACT_SPECS:
        full_spec = CONTRACT_SPECS[canonical]
        if full_spec != spec:
            print(f"  [INFO] {canonical}: micro {spec[0]} unresolved, "
                  f"falling back to full-size {full_spec[0]}")
            return _try_resolve(ib, full_spec, cutoff)
    return None


def resolve_all(
    ib: IB,
    canonicals: list[str],
    roll_buffer_days: int = ROLL_BUFFER_DAYS,
) -> dict[str, Future]:
    """Resolve a list of canonicals to qualified Future contracts.

    Skips canonicals that fail to resolve and prints a warning. Returns
    a dict of canonical -> Future for successful resolutions only.
    """
    out: dict[str, Future] = {}
    failures: list[str] = []
    for c in canonicals:
        try:
            f = resolve_front_month(ib, c, roll_buffer_days)
            if f is None:
                failures.append(c)
            else:
                out[c] = f
        except Exception as e:
            print(f"  [ERROR] {c}: {e}")
            failures.append(c)
    if failures:
        print(f"  [WARN] Failed to resolve: {failures}")
    return out


if __name__ == "__main__":
    # Smoke test: resolve all 22 GSCI canonicals against IBKR paper
    from config.settings import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
    from config.tickers import get_canonical_list

    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
    print(f"Connected to IBKR at {IBKR_HOST}:{IBKR_PORT}\n")
    print(f"Resolving {len(get_canonical_list())} canonicals...\n")
    contracts = resolve_all(ib, get_canonical_list())
    print(f"\nResolved {len(contracts)}/{len(get_canonical_list())}:")
    for canonical, c in sorted(contracts.items()):
        print(
            f"  {canonical:4s} -> {c.localSymbol:10s} "
            f"exp={c.lastTradeDateOrContractMonth:8s} "
            f"exch={c.exchange:8s} mult={c.multiplier}"
        )
    ib.disconnect()
