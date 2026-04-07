"""
Module 1 — Macro Regime Classifier
Growth/Inflation quadrant via DualSourceRouter.

Phase 2 changes:
  - All FRED calls now flow through DualSourceRouter (FRED primary,
    LSEG secondary where entitled, cross-validated).
  - Series identifiers come from config/tickers.MACRO_SERIES — single
    source of truth shared with the data_sources package.

Regimes:
  - Goldilocks:  Growth ↑, Inflation ↓
  - Reflation:   Growth ↑, Inflation ↑
  - Stagflation: Growth ↓, Inflation ↑
  - Deflation:   Growth ↓, Inflation ↓
"""
from typing import Optional

import pandas as pd

from signal_engine.resilience import log
from signal_engine.data_sources.dual_source import DualSourceRouter

# Lazy router singleton — shared with bsv_signals via the module
_router: Optional[DualSourceRouter] = None


def _get_router() -> DualSourceRouter:
    global _router
    if _router is None:
        # FRED is primary for macro regime data — VIX/SPX/Treasuries are
        # not entitled on this LSEG account, and FRED is the canonical
        # source for ISM/CPI anyway.
        _router = DualSourceRouter(primary_macro="fred")
    return _router


def compute_trend(series: pd.Series, window: int = 6) -> str:
    """Determine if a series is trending up or down (6-month moving average)."""
    ma = series.rolling(window).mean()
    if ma.iloc[-1] > ma.iloc[-2]:
        return "up"
    return "down"


def classify_regime() -> dict:
    """
    Classify the current macro regime using router-fetched ISM + CPI.

    Returns dict with keys: regime, growth_trend, inflation_trend,
        growth_value, inflation_value, ism_source, cpi_source,
        ism_provider, cpi_provider, timestamp.
    """
    router = _get_router()

    ism = router.fetch_macro("ism_manufacturing", lookback_months=24)
    cpi = router.fetch_macro("cpi_yoy", lookback_months=36)

    # CPI: compute YoY % change
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
        "ism_source": ism.attrs.get("source", "fresh"),
        "cpi_source": cpi.attrs.get("source", "fresh"),
        "ism_provider": ism.attrs.get("provider", "fred"),
        "cpi_provider": cpi.attrs.get("provider", "fred"),
        "router_disagreements": dict(router.last_disagreements),
    }


# Regime-based signal adjustments
REGIME_WEIGHTS = {
    "Goldilocks": {
        "energy": 1.0, "metals": 1.0, "agriculture": 1.0, "livestock": 1.0,
        "precious_metals": 1.0,
    },
    "Reflation": {
        "energy": 1.2, "metals": 1.3, "agriculture": 1.1, "livestock": 0.9,
        "precious_metals": 1.1,
    },
    "Stagflation": {
        "energy": 0.8, "metals": 1.4, "agriculture": 1.0, "livestock": 0.7,
        "precious_metals": 1.5,
    },
    "Deflation": {
        "energy": 0.6, "metals": 0.8, "agriculture": 0.9, "livestock": 0.8,
        "precious_metals": 0.9,
    },
}


if __name__ == "__main__":
    regime = classify_regime()
    print(f"Current Regime: {regime['regime']}")
    print(f"  Growth: {regime['growth_trend']} ({regime['growth_value']:.1f})")
    print(f"  Inflation: {regime['inflation_trend']} ({regime['inflation_value']:.1f}%)")
    print(f"  ISM provider: {regime['ism_provider']} ({regime['ism_source']})")
    print(f"  CPI provider: {regime['cpi_provider']} ({regime['cpi_source']})")
    if regime["router_disagreements"]:
        print(f"  Disagreements: {regime['router_disagreements']}")
