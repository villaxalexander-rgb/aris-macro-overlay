"""
Tests for C8 'Kitchen Sink VF' signal — Phase A.1
Uses synthetic price data so tests are deterministic and fast.
"""
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """200-day synthetic price panel for 6 assets across 3 sectors."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=400)
    assets = ["CL", "NG", "GC", "SI", "ZC", "ZW"]
    # Random walks with different vols so vol-filter has something to bite on
    vols = [0.35, 0.50, 0.15, 0.18, 0.22, 0.20]
    data = {}
    for asset, v in zip(assets, vols):
        rets = np.random.normal(0, v / np.sqrt(252), len(dates))
        data[asset] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_sector_rotation
# ---------------------------------------------------------------------------

class TestSectorRotation:
    def test_returns_series_with_correct_index(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_sector_rotation
        result = compute_sector_rotation(synthetic_prices, lookback=63)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(synthetic_prices.columns)

    def test_values_bounded(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_sector_rotation
        result = compute_sector_rotation(synthetic_prices, lookback=63)
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_different_lookbacks_differ(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_sector_rotation
        r1 = compute_sector_rotation(synthetic_prices, lookback=21)
        r2 = compute_sector_rotation(synthetic_prices, lookback=252)
        # With random walks and different lookbacks, signals should differ
        assert not r1.equals(r2)


# ---------------------------------------------------------------------------
# compute_vol_filter
# ---------------------------------------------------------------------------

class TestVolFilter:
    def test_returns_series_with_correct_index(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_vol_filter
        result = compute_vol_filter(synthetic_prices, vol_window=63, dampen_low_vol=0.3)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(synthetic_prices.columns)

    def test_values_are_either_dampen_or_one(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_vol_filter
        dampen = 0.3
        result = compute_vol_filter(synthetic_prices, vol_window=63, dampen_low_vol=dampen)
        for v in result.values:
            assert v == pytest.approx(1.0) or v == pytest.approx(dampen), \
                f"Expected 1.0 or {dampen}, got {v}"

    def test_splits_assets_into_two_groups(self, synthetic_prices):
        """Half the assets should get 1.0, the other half should get dampen."""
        from signal_engine.bsv_signals import compute_vol_filter
        dampen = 0.3
        result = compute_vol_filter(synthetic_prices, vol_window=63, dampen_low_vol=dampen)
        n_full = (result == 1.0).sum()
        n_damp = (result == dampen).sum()
        assert n_full + n_damp == len(result)
        assert n_full > 0 and n_damp > 0  # both groups should be populated

    def test_dampen_zero_kills_low_vol(self, synthetic_prices):
        from signal_engine.bsv_signals import compute_vol_filter
        result = compute_vol_filter(synthetic_prices, vol_window=63, dampen_low_vol=0.0)
        # At least some assets should be zeroed out
        assert (result == 0.0).any()


# ---------------------------------------------------------------------------
# generate_c8_signal (integration)
# ---------------------------------------------------------------------------

class TestGenerateC8Signal:
    def test_returns_dataframe_with_expected_columns(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c8_signal
        result = generate_c8_signal(synthetic_prices)
        assert isinstance(result, pd.DataFrame)
        for col in ["reversal", "sector_rot", "tsmom", "value", "vol_mult", "composite"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_index_matches_assets(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c8_signal
        result = generate_c8_signal(synthetic_prices)
        assert set(result.index) == set(synthetic_prices.columns)

    def test_composite_not_all_zero(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c8_signal
        result = generate_c8_signal(synthetic_prices)
        assert result["composite"].abs().sum() > 0

    def test_vol_filter_disabled(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c8_signal
        result = generate_c8_signal(synthetic_prices, vol_filter_enabled=False)
        assert (result["vol_mult"] == 1.0).all()

    def test_custom_weights_sum_respected(self, synthetic_prices):
        from signal_engine.bsv_signals import generate_c8_signal
        w = {"reversal": 1.0, "sector_rot": 0.0, "tsmom": 0.0, "value": 0.0}
        result = generate_c8_signal(synthetic_prices, weights=w, vol_filter_enabled=False)
        # With only reversal weight, composite should equal reversal
        pd.testing.assert_series_equal(
            result["composite"], result["reversal"], check_names=False
        )

    def test_default_weights_sum_to_one(self):
        """Sanity: default C8 weights should sum to 1.0."""
        from config.settings import C8_WEIGHTS
        assert sum(C8_WEIGHTS.values()) == pytest.approx(1.0)
