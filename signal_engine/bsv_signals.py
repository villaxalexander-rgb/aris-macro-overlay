"""
Module 1 - BSV Signal Engine
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def fetch_commodity_prices(tickers, lookback_days=365):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data


def compute_momentum(prices, window=252):
    if len(prices) < window:
        window = len(prices) - 1
    returns = prices.pct_change(window).iloc[-1]
    return returns.rank(pct=True) * 2 - 1


def compute_carry(prices):
    short_ret = prices.pct_change(21).iloc[-1]
    return short_ret.rank(pct=True) * 2 - 1


def compute_value(prices, target_window=1200):
    """Value signal - compute per column to handle different data lengths."""
    results = {}
    for col in prices.columns:
        col_data = prices[col].dropna()
        if len(col_data) < 60:
            results[col] = 0.0
            continue
        window = min(target_window, len(col_data) - 1)
        long_mean = col_data.rolling(window).mean().iloc[-1]
        current = col_data.iloc[-1]
        if long_mean != 0:
            results[col] = (long_mean - current) / long_mean
        else:
            results[col] = 0.0
    result = pd.Series(results)
    return result.rank(pct=True) * 2 - 1


def compute_reversal(prices, window=21):
    short_ret = prices.pct_change(window).iloc[-1]
    return (-short_ret).rank(pct=True) * 2 - 1


def generate_bsv_signals(prices, weights=None):
    if weights is None:
        weights = {"momentum": 0.40, "carry": 0.25,
                   "value": 0.20, "reversal": 0.15}
    signals = pd.DataFrame({
        "momentum": compute_momentum(prices),
        "carry": compute_carry(prices),
        "value": compute_value(prices),
        "reversal": compute_reversal(prices),
    })
    signals["composite"] = (
        signals["momentum"] * weights["momentum"]
        + signals["carry"] * weights["carry"]
        + signals["value"] * weights["value"]
        + signals["reversal"] * weights["reversal"]
    )
    return signals


if __name__ == "__main__":
    from config.settings import GSCI_ASSETS
    print("Fetching commodity prices...")
    prices = fetch_commodity_prices(GSCI_ASSETS, lookback_days=365*5)
    print(f"Got data for {len(prices.columns)} assets")
    signals = generate_bsv_signals(prices)
    print("\nBSV Signals:")
    print(signals.sort_values("composite", ascending=False))
