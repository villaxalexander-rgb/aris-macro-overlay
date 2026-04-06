"""
Module 1 - Macro Regime Classifier
Growth/Inflation quadrant using FRED API (ISM + CPI).
"""
import pandas as pd
from fredapi import Fred
from config.settings import FRED_API_KEY, FRED_ISM_SERIES, FRED_CPI_SERIES, SECTOR_MAP


def get_fred_data(series_id, lookback_months=24):
    """Fetch a FRED series."""
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series(series_id)
    return data.tail(lookback_months)


def compute_trend(series, window=6):
    """Determine if a series is trending up or down."""
    ma = series.rolling(window).mean()
    if ma.iloc[-1] > ma.iloc[-2]:
        return "up"
    return "down"


def classify_regime():
    """Classify the current macro regime based on ISM and CPI trends."""
    ism = get_fred_data(FRED_ISM_SERIES)
    cpi = get_fred_data(FRED_CPI_SERIES)
    cpi_yoy = cpi.pct_change(12) * 100

    growth_trend = compute_trend(ism)
    inflation_trend = compute_trend(cpi_yoy)

    if growth_trend == "up" and inflation_trend == "down":
        regime = "Goldilocks"
    elif growth_trend == "up" and inflation_trend == "up":
        regime = "Reflation"
    elif growth_trend == "down" and inflation_trend == "up":
        regime = "Stagflation"
    else:
        regime = "Deflation"

    return {
        "regime": regime,
        "growth_trend": growth_trend,
        "inflation_trend": inflation_trend,
        "growth_value": float(ism.iloc[-1]),
        "inflation_value": float(cpi_yoy.iloc[-1]),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


# Regime → sector multiplier table.
# These are MULTIPLICATIVE scalars applied to BSV composite scores.
# 1.0 = neutral, >1.0 = overweight, <1.0 = underweight.
# Theory:
#   Goldilocks  → balanced; growth assets get small boost
#   Reflation   → growth + inflation up → energy, base metals, ags rip
#   Stagflation → growth down, inflation up → precious metals + energy hedge,
#                 livestock collapses (consumer weakness)
#   Deflation   → growth + inflation down → everything underweight, precious
#                 metals slight overweight (defensive)
REGIME_WEIGHTS = {
    "Goldilocks":  {"energy": 1.1, "metals": 1.2, "precious_metals": 0.9,
                    "agriculture": 1.0, "livestock": 1.0},
    "Reflation":   {"energy": 1.3, "metals": 1.4, "precious_metals": 1.0,
                    "agriculture": 1.1, "livestock": 0.9},
    "Stagflation": {"energy": 1.2, "metals": 0.8, "precious_metals": 1.4,
                    "agriculture": 1.0, "livestock": 0.6},
    "Deflation":   {"energy": 0.6, "metals": 0.7, "precious_metals": 1.1,
                    "agriculture": 0.8, "livestock": 0.7},
}


def get_sector_weight(ticker: str, regime: str) -> float:
    """Look up the regime-conditional sector multiplier for a ticker.
    Returns 1.0 (neutral) if ticker has no sector mapping or regime is unknown.
    """
    sector = SECTOR_MAP.get(ticker)
    if sector is None:
        return 1.0
    regime_table = REGIME_WEIGHTS.get(regime)
    if regime_table is None:
        return 1.0
    return regime_table.get(sector, 1.0)


def apply_regime_weights(signals: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Apply regime-conditional sector multipliers to BSV composite scores.

    Args:
        signals: DataFrame indexed by ticker with at least a 'composite' column.
        regime: One of 'Goldilocks', 'Reflation', 'Stagflation', 'Deflation'.

    Returns:
        New DataFrame with two added columns:
            sector_weight: the multiplier applied (per ticker)
            regime_adjusted_composite: composite * sector_weight
        Original 'composite' column is preserved for transparency.
    """
    out = signals.copy()
    out["sector"] = [SECTOR_MAP.get(t, "unknown") for t in out.index]
    out["sector_weight"] = [get_sector_weight(t, regime) for t in out.index]
    out["regime_adjusted_composite"] = out["composite"] * out["sector_weight"]
    return out


if __name__ == "__main__":
    regime = classify_regime()
    print(f"Current Regime: {regime['regime']}")
    print(f"  Growth: {regime['growth_trend']} ({regime['growth_value']:.1f})")
    print(f"  Inflation: {regime['inflation_trend']} ({regime['inflation_value']:.1f}%)")
    print(f"\nSector weights for {regime['regime']}:")
    for sector, weight in REGIME_WEIGHTS[regime['regime']].items():
        print(f"  {sector:<18} {weight:.2f}")
