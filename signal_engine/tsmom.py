"""
Module 1b — Time-Series Momentum (TSMOM) ensemble
Moskowitz/Ooi/Pedersen 2012 style per-asset momentum with vol targeting.

Unlike BSV (cross-sectional rank), TSMOM is per-asset: each instrument makes
its own long/short/flat decision based on its own recent return. This gives
it a very different failure mode and pairs well with BSV as a hybrid.

Output convention:
  - Signal in [-1, +1] via tanh(return / (k * asset_vol))
  - Per-asset vol-target sizing handled by daily_signals; this module
    returns raw signal, not position size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.resilience import log

# Defaults pulled from config.settings at call-site; these are fallbacks only
DEFAULT_LOOKBACKS = (63, 126, 252)     # 3m, 6m, 12m
DEFAULT_VOL_WINDOW = 63                 # trailing 3m for vol estimate
DEFAULT_TANH_K = 2.0                    # higher = softer saturation


def _compute_continuous_signal(
    prices: pd.DataFrame,
    lookback: int,
    vol_window: int = DEFAULT_VOL_WINDOW,
    k: float = DEFAULT_TANH_K,
) -> pd.Series:
    """
    Continuous TSMOM for a single lookback, end-of-series.
    Signal = tanh( return_lookback / (k * realized_vol_over_lookback) )

    Vol is annualized daily-stdev scaled back to the lookback horizon.
    Returns a Series indexed by asset.
    """
    if len(prices) < max(lookback, vol_window) + 1:
        return pd.Series(dtype=float)

    ret = prices.pct_change(lookback).iloc[-1]
    daily = prices.pct_change().iloc[-vol_window:]
    vol_annualised = daily.std() * np.sqrt(252)
    vol_over_lookback = vol_annualised * np.sqrt(lookback / 252)

    # Guard: assets with zero vol (dead series) get signal 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ret / (k * vol_over_lookback)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    return np.tanh(z)


def compute_tsmom_ensemble(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    weights: tuple[float, ...] | None = None,
    vol_window: int = DEFAULT_VOL_WINDOW,
    k: float = DEFAULT_TANH_K,
) -> pd.Series:
    """
    Average TSMOM signal across lookbacks.
    Returns a Series in [-1, +1] indexed by asset.
    """
    if weights is None:
        weights = tuple(1.0 / len(lookbacks) for _ in lookbacks)
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"TSMOM ensemble weights must sum to 1, got {sum(weights)}")

    components = []
    for L, w in zip(lookbacks, weights):
        sig = _compute_continuous_signal(prices, L, vol_window=vol_window, k=k)
        if not sig.empty:
            components.append(w * sig)

    if not components:
        log.warning("TSMOM ensemble: insufficient price history for any lookback")
        return pd.Series(dtype=float)

    combined = sum(components)
    log.info(
        f"TSMOM ensemble: {len(combined)} assets, "
        f"mean={combined.mean():.2f}, abs_mean={combined.abs().mean():.2f}, "
        f"nonzero={int((combined.abs() > 0.05).sum())}"
    )
    return combined


def compute_asset_volatility(
    prices: pd.DataFrame,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """
    Annualized realized volatility per asset, used for vol-target sizing
    in daily_signals.
    """
    daily = prices.pct_change().iloc[-vol_window:]
    return daily.std() * np.sqrt(252)


if __name__ == "__main__":
    # Smoke test
    import numpy as np
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    fake = pd.DataFrame(
        rng.normal(0.0002, 0.015, size=(300, 5)).cumsum(axis=0) + 100,
        index=dates, columns=["A", "B", "C", "D", "E"],
    )
    sig = compute_tsmom_ensemble(fake)
    print("Signal:\n", sig.sort_values(ascending=False))
    vol = compute_asset_volatility(fake)
    print("\nVolatility:\n", vol.sort_values(ascending=False))
