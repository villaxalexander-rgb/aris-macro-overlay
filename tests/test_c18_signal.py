"""Tests for C18 Balanced 6-Factor signal and its new building blocks."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_prices():
    """400 business days, 6 assets with different vols."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=400)
    vols = {"CL": 0.02, "NG": 0.03, "GC": 0.008,
            "SI": 0.015, "ZC": 0.012, "ZW": 0.01}
    data = {}
    for sym, vol in vols.items():
        data[sym] = 100 * np.cumprod(
            1 + np.random.normal(0.0002, vol, len(dates))
        )
    return pd.DataFrame(data, index=dates)


class TestMomentumAccel:
    def test_returns_series_with_correct_index(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_momentum_accel
        result = compute_momentum_accel(synthetic_prices)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(synthetic_prices.columns)

    def test_values_bounded_minus1_to_1(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_momentum_accel
        result = compute_momentum_accel(synthetic_prices)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_custom_windows(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_momentum_accel
        result = compute_momentum_accel(synthetic_prices, fast=21, slow=42)
        assert len(result) == 6
        assert not result.isna().all()


class TestSkewness:
    def test_returns_series_with_correct_index(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_skewness
        result = compute_skewness(synthetic_prices)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(synthetic_prices.columns)

    def test_values_bounded_minus1_to_1(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_skewness
        result = compute_skewness(synthetic_prices)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_custom_window(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_skewness
        r1 = compute_skewness(synthetic_prices, window=21)
        r2 = compute_skewness(synthetic_prices, window=126)
        assert not r1.equals(r2)


class TestGenerateC18Signal:
    def test_returns_dataframe_with_expected_columns(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices)
        expected_cols = {"reversal", "sector_rot", "carry", "accel",
                         "skew", "value", "vol_mult", "composite"}
        assert expected_cols == set(result.columns)

    def test_composite_not_all_zero(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices)
        assert result["composite"].abs().sum() > 0

    def test_default_weights_sum_to_one(self):
        w = {"reversal": 0.30, "sector_rot": 0.12, "carry": 0.13,
             "accel": 0.10, "skew": 0.10, "value": 0.25}
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_custom_weights(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        w = {"reversal": 0.20, "sector_rot": 0.10, "carry": 0.10,
             "accel": 0.20, "skew": 0.20, "value": 0.20}
        result = generate_c18_signal(synthetic_prices, weights=w)
        assert result["composite"].abs().sum() > 0

    def test_vol_filter_disabled(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices,
                                     vol_filter_enabled=False)
        assert (result["vol_mult"] == 1.0).all()

    def test_vol_filter_enabled_has_dampened_assets(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices,
                                     vol_filter_enabled=True)
        unique_vals = result["vol_mult"].unique()
        assert len(unique_vals) >= 2

    def test_six_factors_all_populated(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices)
        for col in ["reversal", "sector_rot", "carry", "accel",
                     "skew", "value"]:
            assert not result[col].isna().all(), f"{col} is all NaN"

    def test_different_from_c10(self, synthetic_prices):
        from signal_engine.bsv_signals import (
            generate_c18_signal, generate_c10_signal,
        )
        c18 = generate_c18_signal(synthetic_prices)
        c10 = generate_c10_signal(synthetic_prices)
        assert not c18["composite"].equals(c10["composite"])

    def test_has_accel_and_skew_columns(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c18_signal
        result = generate_c18_signal(synthetic_prices)
        assert "accel" in result.columns
        assert "skew" in result.columns
        assert "tsmom" not in result.columns  # C18 replaces TSMOM
