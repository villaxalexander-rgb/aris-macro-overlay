"""
Module 1 — BSV Signal Engine
Phase 2: DualSourceRouter prices + REAL curve carry from LSEG full futures curves.
"""
from typing import Optional

import pandas as pd
import numpy as np

from signal_engine.resilience import log
from signal_engine.data_sources.dual_source import DualSourceRouter

_router: Optional[DualSourceRouter] = None


def _get_router() -> DualSourceRouter:
    global _router
    if _router is None:
        _router = DualSourceRouter()
        if _router.lseg is not None:
            try:
                _router.lseg.open()
            except Exception as e:
                log.warning(f"LSEG session could not open ({e}); router will use yfinance only")
                _router.lseg = None
    return _router


def fetch_commodity_prices(canonical: list[str], lookback_days: int = 365) -> pd.DataFrame:
    router = _get_router()
    df = router.fetch_prices(canonical, lookback_days=lookback_days)
    log.info(
        f"BSV prices: shape={df.shape}, "
        f"nan_pct={df.isna().mean().mean():.1%}, "
        f"used={router.last_used_source.get('prices', '?')}"
    )
    return df


def fetch_commodity_curves(canonical: list[str]) -> dict[str, pd.DataFrame]:
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


def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    # ffill to survive sparse LSEG history; pct_change then asks for the
    # trailing-window return which is NaN only if an asset truly has no
    # usable data at all.
    p = prices.ffill()
    returns = p.pct_change(window, fill_method=None).iloc[-1]
    return returns.rank(pct=True) * 2 - 1


def compute_curve_carry(curves: dict[str, pd.DataFrame]) -> pd.Series:
    """REAL carry — annualized roll yield from full futures curve.
    backwardation = positive carry = long bias.
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
    log.info(f"Real curve carry computed for {len(s)} assets (median: {s.median():.2%})")
    return s.rank(pct=True) * 2 - 1


def compute_carry_proxy(prices: pd.DataFrame) -> pd.Series:
    """Legacy carry proxy — short-term return rank. Per-asset fallback only."""
    p = prices.ffill()
    short_ret = p.pct_change(21, fill_method=None).iloc[-1]
    return short_ret.rank(pct=True) * 2 - 1


def compute_value(prices: pd.DataFrame, window: int = 252 * 5) -> pd.Series:
    """Value signal — 5y mean reversion, NaN-robust.

    LSEG price panels can be 10-15% sparse; without ffill + min_periods
    a single NaN anywhere in the 5y window collapses the rolling mean
    to NaN and every composite becomes NaN.
    """
    if len(prices) < window:
        window = len(prices)
    p = prices.ffill()
    min_p = max(60, window // 4)
    long_mean = p.rolling(window, min_periods=min_p).mean().iloc[-1]
    current = p.iloc[-1]
    deviation = (long_mean - current) / long_mean
    return deviation.rank(pct=True) * 2 - 1


def compute_reversal(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """Short-term reversal — 1-month contrarian signal. NaN-robust."""
    p = prices.ffill()
    short_ret = p.pct_change(window, fill_method=None).iloc[-1]
    return (-short_ret).rank(pct=True) * 2 - 1


def generate_bsv_signals(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 0.40, "carry": 0.25, "value": 0.20, "reversal": 0.15}

    momentum = compute_momentum(prices)
    value = compute_value(prices)
    reversal = compute_reversal(prices)

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


# ---- Hybrid composite (Phase A) ----------------------------------------
def generate_hybrid_signals(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    bsv_weights: dict | None = None,
    tsmom_lookbacks: tuple[int, ...] = (63, 126, 252),
    tsmom_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Phase A hybrid signal: blend TSMOM ensemble with BSV composite.

    Args:
        prices:          DataFrame of commodity prices
        curves:          optional curves dict for real carry (legacy BSV)
        bsv_weights:     BSV factor weights (passed to generate_bsv_signals)
        tsmom_lookbacks: TSMOM lookback windows in trading days
        tsmom_weight:    weight on TSMOM side of hybrid, BSV gets (1 - this)

    Returns:
        DataFrame with columns: momentum, carry, carry_source, value,
        reversal, bsv_composite, tsmom, composite
        where `composite` is the hybrid signal used downstream.
    """
    from signal_engine.tsmom import compute_tsmom_ensemble

    assert 0.0 <= tsmom_weight <= 1.0, f"tsmom_weight must be in [0,1], got {tsmom_weight}"

    # BSV side (reuse existing function, grab its composite column)
    bsv_signals = generate_bsv_signals(prices, curves=curves, weights=bsv_weights)
    bsv_composite = bsv_signals["composite"].astype(float)

    # TSMOM side
    tsmom_signal = compute_tsmom_ensemble(prices, lookbacks=tsmom_lookbacks)
    tsmom_signal = tsmom_signal.reindex(prices.columns).fillna(0.0)

    # Blend
    hybrid = tsmom_weight * tsmom_signal + (1.0 - tsmom_weight) * bsv_composite

    # Preserve all BSV columns, add tsmom + hybrid composite (replaces old composite)
    out = bsv_signals.copy()
    out = out.rename(columns={"composite": "bsv_composite"})
    out["tsmom"] = tsmom_signal
    out["composite"] = hybrid

    log.info(
        f"Hybrid signal: BSV mean={bsv_composite.mean():.2f}, "
        f"TSMOM mean={tsmom_signal.mean():.2f}, "
        f"blend tsmom_w={tsmom_weight}, "
        f"hybrid abs_mean={hybrid.abs().mean():.2f}"
    )
    return out
