"""
Module 1 - Daily Signal Pipeline
"""
import json
import os
from datetime import datetime
import pandas as pd

from signal_engine.bsv_signals import fetch_commodity_prices, generate_bsv_signals
from signal_engine.regime_classifier import classify_regime, REGIME_WEIGHTS
from config.settings import GSCI_ASSETS, SIGNAL_OUTPUT_PATH


def run_daily_signals():
    """Run the full daily signal pipeline."""
    regime = classify_regime()
    print(f"Regime: {regime['regime']} (Growth {regime['growth_trend']}, Inflation {regime['inflation_trend']})")

    prices = fetch_commodity_prices(GSCI_ASSETS, lookback_days=365 * 5)
    bsv = generate_bsv_signals(prices)

    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "signals": bsv.to_dict(orient="index"),
        "target_positions": bsv["composite"].to_dict(),
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
