"""
Module 1 — BSV Signal Engine
Multi-factor commodity momentum model: momentum, carry, value, reversal
across the GSCI universe.

Phase 2 changes:
  - Prices now flow through DualSourceRouter (LSEG primary,
    yfinance secondary, cross-validated).
  - Carry is now REAL annualized roll yield computed from the LSEG
    full futures curve (not the old short-term momentum proxy).
  - Curve fetching is best-effort: if LSEG is down or a particular
    asset has no curve, we fall back to the legacy momentum proxy
    for that asset only.

Output: signal scores per asset (-1 to +1)
"""
from typing import Optional

import pandas as pd
import numpy as np

from signal_engine.resilience import log
from signal_engine.data_sources.dual_source import DualSourceRouter

# ---- Lazy router singleton ----------------------------------------------
_router: Optional[DualSourceRouter] = None


def _get_router() -> DualSourceRouter:
    """Lazily build a DualSourceRouter and open the LSEG session if available."""
    global _router
    if _router is None:
        _router = DualSourceRouter()
        if _router.lseg is not None:
            try:
                _router.lseg.open()
            except Exception as e:
                log.warning(
                    f"LSEG session could not open ({e}); router will use yfinance only"
                )
                _router.lseg = None
    return _router


# ---- Public fetchers ----------------------------------------------------
def fetch_commodity_prices(
    canonical: list[str], lookback_days: int = 365
) -> pd.DataFrame:
    """Fetch commodity prices via DualSourceRouter (LSEG primary, yfinance secondary)."""
    router = _get_router()
    df = router.fetch_prices(canonical, lookback_days=lookback_days)
    log.info(
        f"BSV prices: shape={df.shape}, "
        f"nan_pct={df.isna().mean().mean():.1%}, "
        f"used={router.last_used_source.get('prices', '?')}"
    )
    return df


def fetch_commodity_curves(canonical: list[str]) -> dict[str, pd.DataFrame]:
    """
    Best-effort curve fetch for the carry signal. Returns a dict
    canonical -> curve DataFrame, skipping any asset whose curve fails.
    LSEG-only — yfinance has no curve data.
    """
    router = _get_router()
    curves: dict[str, pd.DataFrame] = {}
    if router.lseg is None:
        log.warning("No LSEG session — curves unavailable, will use carry proxy")
        return curves
    for c in canonical:
        try:
            curves[c] = router.fetch_curve(c)
        except Exception as e:
            log.warning(f"curve fetch failed for {c}: {type(e).__name__}: {e}")
    log.info(f"Curves fetched for {len(curves)}/{len(canonical)} assets")
    return curves


# ---- Factor functions ---------------------------------------------------
def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """12-month price momentum (return over lookback period)."""
    returns = prices.pct_change(window).iloc[-1]
    return returns.rank(pct=True) * 2 - 1


def compute_curve_carry(curves: dict[str, pd.DataFrame]) -> pd.Series:
    """
    REAL carry signal — annualized roll yield from the full futures curve.

    For each asset:
        roll_yield = (front_settle - next_settle) / front_settle
        annualized = roll_yield * (365 / days_between_expiries)

    Convention:
        backwardation (front > next) -> positive carry -> long bias
        contango     (front < next) -> negative carry -> short bias

    Returns a rank-normalised Series in [-1, 1]. Assets with no usable
    curve are silently dropped — caller must handle missing entries.
    """
    raw: dict[str, float] = {}
    for canonical, curve in curves.items():
        if curve is None or len(curve) < 2:
            continue
        if "settle" not in curve.columns or "expiry" not in curve.columns:
            continue
        c0 = curve.iloc[0]
        c1 = curve.iloc[1]
        s0, s1 = c0.get("settle"), c1.get("settle")
        if pd.isna(s0) or pd.isna(s1) or s0 in (0, None):
            continue
        try:
            days = (pd.Timestamp(c1["expiry"]) - pd.Timestamp(c0["expiry"])).days
        except Exception:
            continue
        if days <= 0:
            continue
        roll_yield = (float(s0) - float(s1)) / float(s0)
        annualized = roll_yield * (365.0 / days)
        raw[canonical] = annualized

    if not raw:
        return pd.Series(dtype=float)

    s = pd.Series(raw, dtype=float)
    log.info(
        f"Real curve carry computed for {len(s)} assets "
        f"(median annualized roll yield: {s.median():.2%})"
    )
    return s.rank(pct=True) * 2 - 1


def compute_carry_proxy(prices: pd.DataFrame) -> pd.Series:
    """
    Legacy carry proxy — short-term return rank. Used only as a per-asset
    fallback when the real curve is missing.
    """
    short_ret = prices.pct_change(21).iloc[-1]
    return short_ret.rank(pct=True) * 2 - 1


def compute_value(prices: pd.DataFrame, window: int = 252 * 5) -> pd.Series:
    """Value signal — mean reversion over 5-year horizon."""
    if len(prices) < window:
        window = len(prices)
    long_mean = prices.rolling(window).mean().iloc[-1]
    current = prices.iloc[-1]
    deviation = (long_mean - current) / long_mean
    return deviation.rank(pct=True) * 2 - 1


def compute_reversal(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """Short-term reversal — 1-month contrarian signal."""
    short_ret = prices.pct_change(window).iloc[-1]
    return (-short_ret).rank(pct=True) * 2 - 1


# ---- Composite ----------------------------------------------------------
def generate_bsv_signals(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Combine all four factors into a composite BSV signal.

    Args:
        prices:  DataFrame of commodity prices (columns = canonical names)
        curves:  optional dict canonical -> curve DataFrame for real carry.
                 If None or empty for an asset, falls back to carry proxy.
        weights: factor weights dict
    """
    if weights is None:
        weights = {
            "momentum": 0.40,
            "carry": 0.25,
            "value": 0.20,
            "reversal": 0.15,
        }

    momentum = compute_momentum(prices)
    value = compute_value(prices)
    reversal = compute_reversal(prices)

    # Real carry where available, proxy where not
    real_carry = compute_curve_carry(curves) if curves else pd.Series(dtype=float)
    proxy_carry = compute_carry_proxy(prices)
    carry = real_carry.reindex(prices.columns).combine_first(proxy_carry)
    carry_source = pd.Series(
        ["real_curve" if c in real_carry.index else "proxy" for c in prices.columns],
        index=prices.columns,
    )

    signals = pd.DataFrame({
        "momentum": momentum,
        "carry": carry,
        "carry_source": carry_source,
        "value": value,
        "reversal": reversal,
    })

    # Composite is numeric only
    signals["composite"] = (
        signals["momentum"].astype(float) * weights["momentum"]
        + signals["carry"].astype(float) * weights["carry"]
        + signals["value"].astype(float) * weights["value"]
        + signals["reversal"].astype(float) * weights["reversal"]
    )

    return signals


if __name__ == "__main__":
    from config.tickers import get_canonical_list

    canonical = get_canonical_list()
    print(f"Fetching prices for {len(canonical)} assets via DualSourceRouter...")
    prices = fetch_commodity_prices(canonical, lookback_days=365 * 2)
    print(f"Got {prices.shape[1]} cols, {prices.shape[0]} rows")

    print("Fetching curves...")
    curves = fetch_commodity_curves(canonical)

    signals = generate_bsv_signals(prices, curves=curves)
    print("\nBSV Signals:")
    print(signals.sort_values("composite", ascending=False))
