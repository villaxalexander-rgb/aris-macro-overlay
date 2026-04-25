import numpy as np
import pandas as pd
import pytest
from signal_engine.tsmom import (
    compute_tsmom_ensemble, compute_asset_volatility, _compute_continuous_signal
)

def _make_prices(returns_per_asset, n_days=400):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    df = {}
    for asset, mu in returns_per_asset.items():
        rets = rng.normal(mu, 0.01, n_days)
        df[asset] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(df, index=dates)

def test_uptrend_positive():
    p = _make_prices({"UP": 0.002}, n_days=400)
    sig = _compute_continuous_signal(p, 252)
    assert sig["UP"] > 0.5

def test_downtrend_negative():
    p = _make_prices({"DOWN": -0.002}, n_days=400)
    sig = _compute_continuous_signal(p, 252)
    assert sig["DOWN"] < -0.5

def test_flat_near_zero():
    p = _make_prices({"FLAT": 0.0}, n_days=400)
    sig = _compute_continuous_signal(p, 252)
    assert abs(sig["FLAT"]) < 0.7

def test_ensemble_weights_must_sum_to_one():
    p = _make_prices({"A": 0.001}, n_days=400)
    with pytest.raises(ValueError):
        compute_tsmom_ensemble(p, lookbacks=(63, 126), weights=(0.4, 0.4))

def test_insufficient_history_returns_empty():
    p = _make_prices({"A": 0.001}, n_days=30)
    sig = compute_tsmom_ensemble(p, lookbacks=(252,))
    assert sig.empty

def test_volatility_positive():
    p = _make_prices({"A": 0.001, "B": -0.001}, n_days=200)
    vol = compute_asset_volatility(p)
    assert (vol > 0).all()
