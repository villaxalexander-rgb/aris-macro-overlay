"""
Module 1 - Volatility Targeting Layer

Scales raw signal scores by inverse realized volatility so that each position
contributes a roughly equal amount of risk to the portfolio.

Without this layer, a score of 0.5 in NG=F (~50% annualized vol) and a score
of 0.5 in GC=F (~15% annualized vol) would get the same NAV%, meaning NG
contributes ~3x the dollar risk despite having the same conviction. After
this layer, dollar risk is roughly equalized across the book.

Position sizing formula:
    position_pct = signal_score * (VOL_TARGET_PCT / realized_vol)
    position_pct = clip(position_pct, -MAX_POSITION_PCT, +MAX_POSITION_PCT)

Where realized_vol is the rolling 60-day annualized standard deviation of
daily log returns.
"""
import numpy as np
import pandas as pd

from config.settings import VOL_TARGET_PCT, VOL_LOOKBACK_DAYS, MAX_POSITION_PCT


def compute_realized_vol(prices: pd.DataFrame,
                         window: int = VOL_LOOKBACK_DAYS) -> pd.Series:
    """Annualized realized volatility per asset.

    Args:
        prices: DataFrame indexed by DatetimeIndex, columns = tickers, values = prices
        window: Rolling window in trading days (default 60)

    Returns:
        Series indexed by ticker. Values = annualized stdev of daily log returns.
        Assets with insufficient history return NaN.
    """
    log_returns = np.log(prices / prices.shift(1))
    daily_vol = log_returns.rolling(window).std().iloc[-1]
    annualized = daily_vol * np.sqrt(252)
    return annualized


def compute_vol_target_size(score: float,
                             asset_vol: float,
                             target_vol: float = VOL_TARGET_PCT,
                             max_pct: float = MAX_POSITION_PCT) -> float:
    """Compute vol-targeted position size as a fraction of NAV.

    Args:
        score: Signal score in [-1, +1] (regime-adjusted composite)
        asset_vol: Annualized realized volatility of the asset
        target_vol: Target vol per position (default 10%)
        max_pct: Hard cap on position size (default 2% of NAV)

    Returns:
        Position size as fraction of NAV in [-max_pct, +max_pct].
        Returns 0.0 if asset_vol is missing, zero, or negative.
    """
    if asset_vol is None or pd.isna(asset_vol) or asset_vol <= 0:
        return 0.0
    raw_size = score * (target_vol / asset_vol)
    return float(np.clip(raw_size, -max_pct, max_pct))


def apply_vol_targeting(signals: pd.DataFrame,
                         prices: pd.DataFrame,
                         target_vol: float = VOL_TARGET_PCT,
                         max_pct: float = MAX_POSITION_PCT) -> pd.DataFrame:
    """Apply vol-targeting to a signals DataFrame.

    Reads from regime_adjusted_composite if present, else falls back to composite.
    Adds 3 new columns to the returned DataFrame:
        realized_vol_60d: float — annualized realized vol per asset
        vol_scalar: float       — target_vol / realized_vol (capped at 10x)
        position_pct: float     — final position size as fraction of NAV

    Args:
        signals: DataFrame indexed by ticker (output of generate_bsv_signals or
                 apply_regime_weights)
        prices: DataFrame of prices used to compute realized vol
        target_vol: Per-position vol target
        max_pct: Hard position size cap

    Returns:
        Copy of signals with realized_vol_60d, vol_scalar, position_pct columns added.
    """
    out = signals.copy()
    vols = compute_realized_vol(prices)

    # Decide which composite to scale: prefer regime-adjusted if it exists
    score_col = "regime_adjusted_composite" if "regime_adjusted_composite" in out.columns else "composite"

    out["realized_vol_60d"] = [vols.get(t, np.nan) for t in out.index]
    out["vol_scalar"] = [
        (target_vol / v) if (v is not None and not pd.isna(v) and v > 0) else 0.0
        for v in out["realized_vol_60d"]
    ]
    # Cap vol_scalar at 10x to prevent extreme leverage on suspiciously low-vol assets
    out["vol_scalar"] = out["vol_scalar"].clip(upper=10.0)

    out["position_pct"] = [
        compute_vol_target_size(score, vol, target_vol, max_pct)
        for score, vol in zip(out[score_col], out["realized_vol_60d"])
    ]

    return out


if __name__ == "__main__":
    from signal_engine.bsv_signals import fetch_commodity_prices, generate_bsv_signals
    from signal_engine.regime_classifier import classify_regime, apply_regime_weights
    from config.settings import GSCI_ASSETS

    print("Fetching prices and computing signals...")
    prices = fetch_commodity_prices(GSCI_ASSETS, lookback_days=365 * 5)
    signals = generate_bsv_signals(prices)
    regime = classify_regime()
    signals = apply_regime_weights(signals, regime["regime"])
    signals = apply_vol_targeting(signals, prices)

    print(f"\nRegime: {regime['regime']}")
    print(f"\nVol-targeted positions (sorted by abs size):")
    sorted_signals = signals.reindex(
        signals["position_pct"].abs().sort_values(ascending=False).index
    )
    cols = ["regime_adjusted_composite", "realized_vol_60d", "vol_scalar", "position_pct"]
    print(sorted_signals[cols].to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nTotal gross exposure: {signals['position_pct'].abs().sum():.2%} of NAV")
    print(f"Total net exposure: {signals['position_pct'].sum():+.2%} of NAV")
