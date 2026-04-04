"""
Module 1 — BSV Signal Engine
Multi-factor commodity momentum model: momentum, carry, value, reversal
across 24 GSCI assets.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def fetch_commodity_prices(tickers: list[str], lookback_days: int = 365) -> pd.DataFrame:
    """Fetch historical prices for commodity futures from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data


def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """12-month price momentum (return over lookback period)."""
    returns = prices.pct_change(window).iloc[-1]
    return returns.rank(pct=True) * 2 - 1


def compute_carry(prices: pd.DataFrame) -> pd.Series:
    """
    Carry signal — proxy using roll yield.
    TODO: Replace with actual futures curve data from IBKR.
    """
    short_ret = prices.pct_change(21).iloc[-1]    return short_ret.rank(pct=True) * 2 - 1


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


def generate_bsv_signals(prices: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Combine all four factors into a composite BSV signal."""
    if weights is None:
        weights = {
            "momentum": 0.40,
            "carry": 0.25,
            "value": 0.20,
            "reversal": 0.15,
        }
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
    prices = fetch_commodity_prices(GSCI_ASSETS, lookback_days=365 * 5)
    print(f"Got data for {len(prices.columns)} assets")
    signals = generate_bsv_signals(prices)
    print("\nBSV Signals:")
    print(signals.sort_values("composite", ascending=False))