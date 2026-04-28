"""
Module 1 — BSV Signal Engine
Phase 2: DualSourceRouter prices + REAL curve carry from LSEG full futures curves.
"""
from typing import Optional

import pandas as pd
import numpy as np

from signal_engine.resilience import log
from signal_engine.data_sources.dual_source import DualSourceRouter

_router: Optional[DualSourceRouter] = None


def _get_router() -> DualSourceRouter:
    global _router
    if _router is None:
        _router = DualSourceRouter()
        if _router.lseg is not None:
            try:
                _router.lseg.open()
            except Exception as e:
                log.warning(f"LSEG session could not open ({e}); router will use yfinance only")
                _router.lseg = None
    return _router


def fetch_commodity_prices(canonical: list[str], lookback_days: int = 365) -> pd.DataFrame:
    router = _get_router()
    df = router.fetch_prices(canonical, lookback_days=lookback_days)
    log.info(
        f"BSV prices: shape={df.shape}, "
        f"nan_pct={df.isna().mean().mean():.1%}, "
        f"used={router.last_used_source.get('prices', '?')}"
    )
    return df


def fetch_commodity_curves(canonical: list[str]) -> dict[str, pd.DataFrame]:
    router = _get_router()
    curves: dict[str, pd.DataFrame] = {}
    if router.lseg is None:
        log.warning("No LSEG session — curves unavailable, will use carry proxy")
        return curves
    for c in canonical:
        try:
            curves[c] = router.fetch_curve(c)
        except Exception as e:
            log.warning(f"curve fetch failed for {c}: {type(e).__name__}: {e}")
    log.info(f"Curves fetched for {len(curves)}/{len(canonical)} assets")
    return curves


def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    # ffill to survive sparse LSEG history; pct_change then asks for the
    # trailing-window return which is NaN only if an asset truly has no
    # usable data at all.
    p = prices.ffill()
    returns = p.pct_change(window, fill_method=None).iloc[-1]
    return returns.rank(pct=True) * 2 - 1


def compute_curve_carry(curves: dict[str, pd.DataFrame]) -> pd.Series:
    """REAL carry — annualized roll yield from full futures curve.
    backwardation = positive carry = long bias.
    """
    raw: dict[str, float] = {}
    for canonical, curve in curves.items():
        if curve is None or len(curve) < 2:
            continue
        if "settle" not in curve.columns or "expiry" not in curve.columns:
            continue
        c0 = curve.iloc[0]
        c1 = curve.iloc[1]
        s0, s1 = c0.get("settle"), c1.get("settle")
        if pd.isna(s0) or pd.isna(s1) or s0 in (0, None):
            continue
        try:
            days = (pd.Timestamp(c1["expiry"]) - pd.Timestamp(c0["expiry"])).days
        except Exception:
            continue
        if days <= 0:
            continue
        roll_yield = (float(s0) - float(s1)) / float(s0)
        annualized = roll_yield * (365.0 / days)
        raw[canonical] = annualized

    if not raw:
        return pd.Series(dtype=float)
    s = pd.Series(raw, dtype=float)
    log.info(f"Real curve carry computed for {len(s)} assets (median: {s.median():.2%})")
    return s.rank(pct=True) * 2 - 1


def compute_carry_proxy(prices: pd.DataFrame) -> pd.Series:
    """Legacy carry proxy — short-term return rank. Per-asset fallback only."""
    p = prices.ffill()
    short_ret = p.pct_change(21, fill_method=None).iloc[-1]
    return short_ret.rank(pct=True) * 2 - 1


def compute_value(prices: pd.DataFrame, window: int = 252 * 5) -> pd.Series:
    """Value signal — 5y mean reversion, NaN-robust.

    LSEG price panels can be 10-15% sparse; without ffill + min_periods
    a single NaN anywhere in the 5y window collapses the rolling mean
    to NaN and every composite becomes NaN.
    """
    if len(prices) < window:
        window = len(prices)
    p = prices.ffill()
    min_p = max(60, window // 4)
    long_mean = p.rolling(window, min_periods=min_p).mean().iloc[-1]
    current = p.iloc[-1]
    deviation = (long_mean - current) / long_mean
    return deviation.rank(pct=True) * 2 - 1


def compute_reversal(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """Short-term reversal — 1-month contrarian signal. NaN-robust."""
    p = prices.ffill()
    short_ret = p.pct_change(window, fill_method=None).iloc[-1]
    return (-short_ret).rank(pct=True) * 2 - 1


def generate_bsv_signals(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 0.40, "carry": 0.25, "value": 0.20, "reversal": 0.15}

    momentum = compute_momentum(prices)
    value = compute_value(prices)
    reversal = compute_reversal(prices)

    real_carry = compute_curve_carry(curves) if curves else pd.Series(dtype=float)
    proxy_carry = compute_carry_proxy(prices)
    carry = real_carry.reindex(prices.columns).combine_first(proxy_carry)
    carry_source = pd.Series(
        ["real_curve" if c in real_carry.index else "proxy" for c in prices.columns],
        index=prices.columns,
    )

    signals = pd.DataFrame({
        "momentum": momentum,
        "carry": carry,
        "carry_source": carry_source,
        "value": value,
        "reversal": reversal,
    })

    signals["composite"] = (
        signals["momentum"].astype(float) * weights["momentum"]
        + signals["carry"].astype(float) * weights["carry"]
        + signals["value"].astype(float) * weights["value"]
        + signals["reversal"].astype(float) * weights["reversal"]
    )
    return signals


if __name__ == "__main__":
    from config.tickers import get_canonical_list
    canonical = get_canonical_list()
    print(f"Fetching prices for {len(canonical)} assets via DualSourceRouter...")
    prices = fetch_commodity_prices(canonical, lookback_days=365 * 2)
    print(f"Got {prices.shape[1]} cols, {prices.shape[0]} rows")
    print("Fetching curves...")
    curves = fetch_commodity_curves(canonical)
    signals = generate_bsv_signals(prices, curves=curves)
    print("\nBSV Signals:")
    print(signals.sort_values("composite", ascending=False))


# ---- Hybrid composite (Phase A) ----------------------------------------
def generate_hybrid_signals(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    bsv_weights: dict | None = None,
    tsmom_lookbacks: tuple[int, ...] = (63, 126, 252),
    tsmom_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Phase A hybrid signal: blend TSMOM ensemble with BSV composite.

    Args:
        prices:          DataFrame of commodity prices
        curves:          optional curves dict for real carry (legacy BSV)
        bsv_weights:     BSV factor weights (passed to generate_bsv_signals)
        tsmom_lookbacks: TSMOM lookback windows in trading days
        tsmom_weight:    weight on TSMOM side of hybrid, BSV gets (1 - this)

    Returns:
        DataFrame with columns: momentum, carry, carry_source, value,
        reversal, bsv_composite, tsmom, composite
        where `composite` is the hybrid signal used downstream.
    """
    from signal_engine.tsmom import compute_tsmom_ensemble

    assert 0.0 <= tsmom_weight <= 1.0, f"tsmom_weight must be in [0,1], got {tsmom_weight}"

    # BSV side (reuse existing function, grab its composite column)
    bsv_signals = generate_bsv_signals(prices, curves=curves, weights=bsv_weights)
    bsv_composite = bsv_signals["composite"].astype(float)

    # TSMOM side
    tsmom_signal = compute_tsmom_ensemble(prices, lookbacks=tsmom_lookbacks)
    tsmom_signal = tsmom_signal.reindex(prices.columns).fillna(0.0)

    # Blend
    hybrid = tsmom_weight * tsmom_signal + (1.0 - tsmom_weight) * bsv_composite

    # Preserve all BSV columns, add tsmom + hybrid composite (replaces old composite)
    out = bsv_signals.copy()
    out = out.rename(columns={"composite": "bsv_composite"})
    out["tsmom"] = tsmom_signal
    out["composite"] = hybrid

    log.info(
        f"Hybrid signal: BSV mean={bsv_composite.mean():.2f}, "
        f"TSMOM mean={tsmom_signal.mean():.2f}, "
        f"blend tsmom_w={tsmom_weight}, "
        f"hybrid abs_mean={hybrid.abs().mean():.2f}"
    )
    return out


# ---- C8 Multi-Factor Composite (Phase A.1) --------------------------------

def compute_sector_rotation(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Cross-sectional sector rotation signal.

    Ranks sectors by trailing return, then maps the sector rank back to
    each constituent asset.  Long the best-performing sector, short the
    worst. Returns values in [-1, 1].
    """
    from config.tickers import SECTOR_MAP

    sector_assets: dict[str, list[str]] = {}
    for asset in prices.columns:
        sec = SECTOR_MAP.get(asset, "other")
        sector_assets.setdefault(sec, []).append(asset)

    # Average trailing return per sector
    sector_ret: dict[str, pd.Series] = {}
    for sec, assets in sector_assets.items():
        avail = [a for a in assets if a in prices.columns]
        if avail:
            sector_ret[sec] = prices[avail].pct_change(lookback).iloc[-1].mean()

    if not sector_ret:
        return pd.Series(0.0, index=prices.columns)

    sr = pd.Series(sector_ret)
    sr_rank = sr.rank(pct=True)
    sr_signal = (sr_rank - 0.5) * 2  # map [0,1] -> [-1,1]

    # Map sector signal back to individual assets
    out = pd.Series(0.0, index=prices.columns)
    for asset in prices.columns:
        sec = SECTOR_MAP.get(asset, "other")
        if sec in sr_signal.index:
            out[asset] = sr_signal[sec]
    return out


def compute_vol_filter(
    prices: pd.DataFrame,
    vol_window: int = 63,
    dampen_low_vol: float = 0.3,
) -> pd.Series:
    """Per-asset vol filter: 1.0 if current realized vol > expanding median,
    else `dampen_low_vol`.  Used to concentrate exposure in volatile regimes
    where mean-reversion (reversal) is strongest.
    """
    returns = prices.pct_change()
    rv = returns.rolling(vol_window).std().iloc[-1] * np.sqrt(252)
    rv_hist = returns.rolling(vol_window).std() * np.sqrt(252)
    rv_median = rv_hist.expanding(min_periods=126).median().iloc[-1]

    high_vol = rv > rv_median
    mult = high_vol.astype(float) * 1.0 + (~high_vol).astype(float) * dampen_low_vol
    return mult


def generate_c8_signal(
    prices: pd.DataFrame,
    curves: dict[str, pd.DataFrame] | None = None,
    weights: dict | None = None,
    reversal_window: int = 10,
    sector_rot_lookback: int = 252,
    tsmom_lookback: int = 252,
    vol_filter_enabled: bool = True,
    vol_dampen_low: float = 0.3,
) -> pd.DataFrame:
    """
    C8 'Kitchen Sink VF' multi-factor composite with vol filter.

    Research battery #3 winner:  Sharpe 0.92, OOS Sharpe 0.75,
    Sharpe@10bp 0.59, MaxDD -20%, Calmar 0.58.

    Default weights: 40% reversal(10d) + 25% sector_rotation(252d)
                   + 15% TSMOM(252d) + 20% value(5y)

    Args:
        prices:              DataFrame of commodity prices (index=dates, cols=assets)
        curves:              optional curves dict (not used by C8 but kept for API compat)
        weights:             factor weights dict; keys: reversal, sector_rot, tsmom, value
        reversal_window:     lookback for cross-sectional reversal (default 10d)
        sector_rot_lookback: lookback for sector rotation momentum (default 252d)
        tsmom_lookback:      lookback for time-series momentum (default 252d)
        vol_filter_enabled:  apply per-asset vol filter (default True)
        vol_dampen_low:      multiplier for low-vol assets (default 0.3)

    Returns:
        DataFrame with columns: reversal, sector_rot, tsmom, value,
        vol_mult, composite (the final signal used downstream).
    """
    from signal_engine.tsmom import compute_tsmom_ensemble

    if weights is None:
        weights = {"reversal": 0.40, "sector_rot": 0.25, "tsmom": 0.15, "value": 0.20}

    # Factor 1: Cross-sectional reversal (10d)
    rev = compute_reversal(prices, window=reversal_window)

    # Factor 2: Sector rotation (252d)
    sec_rot = compute_sector_rotation(prices, lookback=sector_rot_lookback)

    # Factor 3: TSMOM (252d) — time-series momentum, sign of trailing return
    tsmom_sig = compute_tsmom_ensemble(prices, lookbacks=(tsmom_lookback,))
    tsmom_sig = tsmom_sig.reindex(prices.columns).fillna(0.0)

    # Factor 4: Value (5y mean reversion)
    val = compute_value(prices)

    # Raw composite
    raw = (
        weights["reversal"] * rev.astype(float)
        + weights["sector_rot"] * sec_rot.astype(float)
        + weights["tsmom"] * tsmom_sig.astype(float)
        + weights["value"] * val.astype(float)
    )

    # Vol filter
    if vol_filter_enabled:
        vol_mult = compute_vol_filter(prices, dampen_low_vol=vol_dampen_low)
    else:
        vol_mult = pd.Series(1.0, index=prices.columns)

    composite = raw * vol_mult

    out = pd.DataFrame({
        "reversal": rev,
        "sector_rot": sec_rot,
        "tsmom": tsmom_sig,
        "value": val,
        "vol_mult": vol_mult,
        "composite": composite,
    })

    log.info(
        f"C8 signal: rev_mean={rev.abs().mean():.3f}, "
        f"secrot_mean={sec_rot.abs().mean():.3f}, "
        f"tsmom_mean={tsmom_sig.abs().mean():.3f}, "
        f"val_mean={val.abs().mean():.3f}, "
        f"vol_filt={'ON' if vol_filter_enabled else 'OFF'}, "
        f"composite abs_mean={composite.abs().mean():.3f}"
    )
    return out


def compute_carry_proxy(prices: pd.DataFrame, short_window: int = 21,
                        long_window: int = 63) -> pd.Series:
    """Roll-yield proxy via short vs long return differential.

    Positive spread ≈ backwardation (buy signal).
    Returns cross-sectionally ranked values in [-1, 1].

    Gorton & Rouwenhorst (2006) and Koijen et al. (2018) document
    Sharpe 0.5-0.8 on commodity carry alone. This proxy uses price
    returns as a stand-in until real futures curve data (Phase B) is live.
    """
    ret_short = prices.pct_change(short_window).iloc[-1]
    ret_long = prices.pct_change(long_window).iloc[-1]
    raw = ret_short - ret_long / (long_window / short_window)
    ranked = raw.rank(pct=True)
    return (ranked - 0.5) * 2


def generate_c10_signal(prices, curves=None, weights=None,
                        reversal_windows=(5, 10, 21),
                        sector_rot_lookback=252,
                        tsmom_lookback=252,
                        carry_short=21, carry_long=63,
                        vol_filter_enabled=True,
                        vol_dampen_low=0.3) -> pd.DataFrame:
    """C10 'Multi-Rev Carry' composite — Battery #5 winner.

    Default weights: rev=0.40, sec_rot=0.15, carry=0.15, tsmom=0.10, value=0.20
    Key improvements over C8:
      - Multi-window reversal ensemble (5d+10d+21d) instead of single 10d
      - Roll-yield carry proxy as 5th factor
    Sharpe 1.068 (0bp) / 0.713 (10bp), OOS 1.313, MaxDD -21.3%.
    """
    from signal_engine.tsmom import compute_tsmom_ensemble

    if weights is None:
        weights = {
            "reversal": 0.40, "sector_rot": 0.15, "carry": 0.15,
            "tsmom": 0.10, "value": 0.20,
        }

    # Factor 1: Multi-window reversal ensemble
    rev_signals = [compute_reversal(prices, window=w) for w in reversal_windows]
    rev = sum(rev_signals) / len(rev_signals)

    # Factor 2: Sector rotation
    sec_rot = compute_sector_rotation(prices, lookback=sector_rot_lookback)

    # Factor 3: Carry proxy (roll yield)
    carry = compute_carry_proxy(prices, short_window=carry_short,
                                long_window=carry_long)

    # Factor 4: TSMOM
    tsmom_sig = compute_tsmom_ensemble(prices, lookbacks=(tsmom_lookback,))
    tsmom_sig = tsmom_sig.reindex(prices.columns).fillna(0.0)

    # Factor 5: Value (5y mean reversion)
    val = compute_value(prices)

    # Raw composite
    raw = (
        weights["reversal"] * rev.astype(float)
        + weights["sector_rot"] * sec_rot.astype(float)
        + weights["carry"] * carry.astype(float)
        + weights["tsmom"] * tsmom_sig.astype(float)
        + weights["value"] * val.astype(float)
    )

    # Vol filter
    if vol_filter_enabled:
        vol_mult = compute_vol_filter(prices, dampen_low_vol=vol_dampen_low)
    else:
        vol_mult = pd.Series(1.0, index=prices.columns)

    composite = raw * vol_mult

    out = pd.DataFrame({
        "reversal": rev,
        "sector_rot": sec_rot,
        "carry": carry,
        "tsmom": tsmom_sig,
        "value": val,
        "vol_mult": vol_mult,
        "composite": composite,
    })

    log.info(
        f"C10 signal: rev_mean={rev.abs().mean():.3f} "
        f"(windows={reversal_windows}), "
        f"secrot={sec_rot.abs().mean():.3f}, "
        f"carry={carry.abs().mean():.3f}, "
        f"tsmom={tsmom_sig.abs().mean():.3f}, "
        f"val={val.abs().mean():.3f}, "
        f"vf={'ON' if vol_filter_enabled else 'OFF'}, "
        f"composite={composite.abs().mean():.3f}"
    )
    return out
