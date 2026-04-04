"""
Module 1 — Macro Regime Classifier
Growth/Inflation quadrant using FRED API (ISM + CPI).

Regimes:
  - Goldilocks:  Growth up, Inflation down
  - Reflation:   Growth up, Inflation up
  - Stagflation: Growth down, Inflation up
  - Deflation:   Growth down, Inflation down
"""
import pandas as pd
from fredapi import Fred
from config.settings import FRED_API_KEY, FRED_ISM_SERIES, FRED_CPI_SERIES


def get_fred_data(series_id: str, lookback_months: int = 24) -> pd.Series:
    """Fetch a FRED series."""
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series(series_id)
    return data.tail(lookback_months)


def compute_trend(series: pd.Series, window: int = 6) -> str:
    """Determine if a series is trending up or down."""
    ma = series.rolling(window).mean()
    if ma.iloc[-1] > ma.iloc[-2]:
        return "up"
    return "down"

def classify_regime() -> dict:
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

REGIME_WEIGHTS = {
    "Goldilocks":  {"energy": 1.0, "metals": 1.0, "agriculture": 1.0, "livestock": 1.0},
    "Reflation":   {"energy": 1.2, "metals": 1.3, "agriculture": 1.1, "livestock": 0.9},
    "Stagflation": {"energy": 0.8, "metals": 1.4, "agriculture": 1.0, "livestock": 0.7},
    "Deflation":   {"energy": 0.6, "metals": 0.8, "agriculture": 0.9, "livestock": 0.8},
}


if __name__ == "__main__":
    regime = classify_regime()
    print(f"Current Regime: {regime['regime']}")
    print(f"  Growth: {regime['growth_trend']} ({regime['growth_value']:.1f})")
    print(f"  Inflation: {regime['inflation_trend']} ({regime['inflation_value']:.1f}%)")