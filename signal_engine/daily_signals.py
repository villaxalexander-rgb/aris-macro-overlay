"""
Module 1 - Daily Signal Pipeline
"""
import json
import os
from datetime import datetime
import pandas as pd

from signal_engine.bsv_signals import fetch_commodity_prices, generate_bsv_signals
from signal_engine.regime_classifier import (
    classify_regime,
    REGIME_WEIGHTS,
    apply_regime_weights,
)
from signal_engine.vol_target import apply_vol_targeting
from config.settings import GSCI_ASSETS, SIGNAL_OUTPUT_PATH


def run_daily_signals():
    """Run the full daily signal pipeline.

    Pipeline:
      1. Classify regime from FRED (ISM + CPI)
      2. Compute raw BSV signals (4-factor composite, cross-sectionally ranked)
      3. Apply regime-conditional sector multipliers (regime_adjusted_composite)
      4. Apply vol-targeting to scale positions by inverse realized vol
      5. Use position_pct (final, vol-targeted, capped) as target_positions
    """
    regime = classify_regime()
    print(f"Regime: {regime['regime']} "
          f"(Growth {regime['growth_trend']}, Inflation {regime['inflation_trend']})")

    prices = fetch_commodity_prices(GSCI_ASSETS, lookback_days=365 * 5)
    bsv = generate_bsv_signals(prices)

    # Step 3: regime-conditional sector multipliers
    bsv = apply_regime_weights(bsv, regime["regime"])

    # Step 4: vol-targeting — equalizes risk contribution across positions
    bsv = apply_vol_targeting(bsv, prices)

    # Report top positions by absolute size (not score)
    top_by_size = bsv.reindex(bsv["position_pct"].abs().sort_values(ascending=False).index)
    print(f"Top 3 longs (vol-targeted): "
          f"{bsv.nlargest(3, 'position_pct').index.tolist()}")
    print(f"Top 3 shorts (vol-targeted): "
          f"{bsv.nsmallest(3, 'position_pct').index.tolist()}")
    print(f"Gross exposure: {bsv['position_pct'].abs().sum():.2%} of NAV | "
          f"Net: {bsv['position_pct'].sum():+.2%}")

    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "signals": bsv.to_dict(orient="index"),
        "target_positions": bsv["position_pct"].to_dict(),         # final sized positions
        "regime_adjusted_composite": bsv["regime_adjusted_composite"].to_dict(),
        "raw_composite": bsv["composite"].to_dict(),               # transparency
        "gross_exposure": float(bsv["position_pct"].abs().sum()),
        "net_exposure": float(bsv["position_pct"].sum()),
    }
    return output


def save_daily_signals(output):
    """Save daily signal output as JSON."""
    os.makedirs(SIGNAL_OUTPUT_PATH, exist_ok=True)
    filename = f"{SIGNAL_OUTPUT_PATH}{output['date']}_signals.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved signals to {filename}")
    return filename


if __name__ == "__main__":
    output = run_daily_signals()
    save_daily_signals(output)
