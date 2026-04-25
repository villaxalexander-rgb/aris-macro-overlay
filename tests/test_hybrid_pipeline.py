import numpy as np
import pandas as pd
from signal_engine.bsv_signals import generate_hybrid_signals, generate_bsv_signals

def _fake_prices():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", periods=600, freq="B")
    df = pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(600, 6)).cumsum(axis=0) + 100,
        index=dates, columns=["CL", "GC", "ZC", "ZS", "HG", "SI"],
    )
    return df

def test_hybrid_has_expected_columns():
    p = _fake_prices()
    out = generate_hybrid_signals(p, curves=None)
    for col in ["momentum", "carry", "value", "reversal",
                "bsv_composite", "tsmom", "composite"]:
        assert col in out.columns, f"missing {col}"

def test_hybrid_zero_weight_equals_bsv():
    p = _fake_prices()
    out = generate_hybrid_signals(p, curves=None, tsmom_weight=0.0)
    bsv = generate_bsv_signals(p, curves=None)
    pd.testing.assert_series_equal(
        out["composite"], bsv["composite"], check_names=False
    )

def test_hybrid_full_weight_equals_tsmom():
    from signal_engine.tsmom import compute_tsmom_ensemble
    p = _fake_prices()
    out = generate_hybrid_signals(p, curves=None, tsmom_weight=1.0)
    tsmom = compute_tsmom_ensemble(p).reindex(p.columns).fillna(0.0)
    pd.testing.assert_series_equal(
        out["composite"], tsmom, check_names=False
    )

def test_hybrid_composite_in_range():
    p = _fake_prices()
    out = generate_hybrid_signals(p, curves=None)
    assert out["composite"].abs().max() <= 1.5  # allow small overshoot from blending
