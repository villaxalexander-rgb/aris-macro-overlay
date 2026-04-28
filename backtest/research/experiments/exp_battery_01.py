"""
A.R.I.S Alpha Experiment Battery
Run on frozen price_cache.pkl (22 commodities, 2007-2026)
Tests ~10 strategy candidates. Saves results to JSON.
"""
import pickle
import json
import numpy as np
import pandas as pd
from collections import OrderedDict

# ── Load data ──
with open("/sessions/upbeat-adoring-goldberg/mnt/Project: A.R.I.S Macro Overlay System/backtest/research/price_cache.pkl", "rb") as f:
    cache = pickle.load(f)

prices = cache["prices"].ffill()
regime = cache["regime_monthly"]
returns = prices.pct_change().dropna(how="all")

SECTORS = {
    "energy": ["CL", "BZ", "NG", "HO", "RB"],
    "precious": ["GC", "SI", "PL", "PA"],
    "industrial": ["HG"],
    "grains": ["ZC", "ZW", "ZS", "ZM", "ZL"],
    "softs": ["CT", "KC", "SB", "CC"],
    "livestock": ["LE", "GF", "HE"],
}

results = OrderedDict()

def backtest_signal(signal_df, label, vol_target=0.12, rebal="weekly"):
    """
    Generic long/short backtest from a signal DataFrame.
    signal_df: columns = assets, index = dates, values = signal strength [-1, 1]
    Returns dict of metrics.
    """
    sig = signal_df.reindex(returns.index).ffill().fillna(0)

    # Normalize to unit gross exposure each day
    gross = sig.abs().sum(axis=1).replace(0, 1)
    weights = sig.div(gross, axis=0)

    # Vol-target scaling
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    rolling_vol = port_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    vol_scale = vol_target / rolling_vol.replace(0, vol_target)
    vol_scale = vol_scale.clip(0.1, 3.0)

    scaled_ret = port_ret * vol_scale.shift(1)

    # Weekly rebalance simulation (only rebal on Wednesdays)
    if rebal == "weekly":
        is_rebal = pd.Series(sig.index.weekday == 2, index=sig.index)
        held_weights = weights.copy()
        for i in range(1, len(held_weights)):
            if not is_rebal.iloc[i]:
                held_weights.iloc[i] = held_weights.iloc[i-1]
        port_ret_weekly = (held_weights.shift(1) * returns).sum(axis=1)
        rolling_vol_w = port_ret_weekly.rolling(63, min_periods=21).std() * np.sqrt(252)
        vol_scale_w = vol_target / rolling_vol_w.replace(0, vol_target)
        vol_scale_w = vol_scale_w.clip(0.1, 3.0)
        scaled_ret = port_ret_weekly * vol_scale_w.shift(1)

    # Metrics
    scaled_ret = scaled_ret.dropna()
    if len(scaled_ret) < 252:
        return {"label": label, "error": "insufficient data"}

    cum = (1 + scaled_ret).cumprod()
    years = len(scaled_ret) / 252
    cagr = cum.iloc[-1] ** (1/years) - 1
    vol = scaled_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Turnover
    daily_turnover = weights.diff().abs().sum(axis=1).mean()

    # Win rate
    win_rate = (scaled_ret > 0).mean()

    # Yearly breakdown
    yearly_ret = scaled_ret.groupby(scaled_ret.index.year).apply(lambda x: (1+x).prod()-1)

    return {
        "label": label,
        "cagr": round(cagr * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "daily_turnover": round(daily_turnover, 4),
        "win_rate": round(win_rate * 100, 1),
        "years": round(years, 1),
        "worst_year": round(yearly_ret.min() * 100, 2),
        "best_year": round(yearly_ret.max() * 100, 2),
        "pct_positive_years": round((yearly_ret > 0).mean() * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: TSMOM variants — different lookback combos
# ══════════════════════════════════════════════════════════════════
print("EXP 1: TSMOM lookback variants...")

def tsmom_signal(prices, lookback):
    """Time-series momentum: sign of trailing return."""
    ret = prices.pct_change(lookback)
    return ret.apply(np.sign)

for lookbacks in [(21,), (63,), (126,), (252,), (63, 126, 252), (21, 63, 126, 252), (63, 252)]:
    sigs = [tsmom_signal(prices, lb) for lb in lookbacks]
    combined = sum(sigs) / len(sigs)
    label = f"TSMOM_{'_'.join(str(l) for l in lookbacks)}"
    results[label] = backtest_signal(combined, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Cross-sectional momentum (relative strength)
# ══════════════════════════════════════════════════════════════════
print("\nEXP 2: Cross-sectional momentum...")

for window in [63, 126, 252]:
    ret = prices.pct_change(window)
    # Rank cross-sectionally each day -> long top quartile, short bottom
    ranked = ret.rank(axis=1, pct=True)
    signal = (ranked - 0.5) * 2  # map [0,1] -> [-1,1]
    label = f"XSMOM_{window}d"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Carry proxy variants (no real curve data, but
# we test different return-based proxies)
# ══════════════════════════════════════════════════════════════════
print("\nEXP 3: Carry proxy variants...")

for window in [5, 10, 21, 63]:
    short_ret = prices.pct_change(window)
    # Cross-sectional rank — high short-term return ≈ backwardation
    ranked = short_ret.rank(axis=1, pct=True)
    signal = (ranked - 0.5) * 2
    label = f"CARRY_PROXY_{window}d"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Volatility breakout — go long assets with
# expanding vol (trend-following confirmation), short contracting
# ══════════════════════════════════════════════════════════════════
print("\nEXP 4: Volatility breakout...")

for short_w, long_w in [(10, 63), (21, 126), (5, 21)]:
    short_vol = returns.rolling(short_w).std()
    long_vol = returns.rolling(long_w).std()
    vol_ratio = short_vol / long_vol.replace(0, np.nan)
    # Combine with TSMOM: vol expanding + positive momentum = strong long
    mom_63 = tsmom_signal(prices, 63)
    signal = mom_63 * (vol_ratio.rank(axis=1, pct=True) - 0.5) * 2
    label = f"VOL_BREAKOUT_{short_w}v{long_w}"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Mean reversion — short-term reversal signals
# ══════════════════════════════════════════════════════════════════
print("\nEXP 5: Mean reversion / reversal...")

for window in [5, 10, 21]:
    short_ret = prices.pct_change(window)
    # Contrarian: short recent winners, long recent losers (cross-sectional)
    ranked = (-short_ret).rank(axis=1, pct=True)
    signal = (ranked - 0.5) * 2
    label = f"REVERSAL_{window}d"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Sector rotation — rank sectors by momentum,
# go long best sector, short worst
# ══════════════════════════════════════════════════════════════════
print("\nEXP 6: Sector rotation...")

for lookback in [63, 126, 252]:
    sector_ret = {}
    for sector, assets in SECTORS.items():
        avail = [a for a in assets if a in prices.columns]
        if avail:
            sector_ret[sector] = prices[avail].pct_change(lookback).mean(axis=1)
    sector_df = pd.DataFrame(sector_ret)
    sector_rank = sector_df.rank(axis=1, pct=True)

    # Map sector rank back to individual assets
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for sector, assets in SECTORS.items():
        if sector in sector_rank.columns:
            for a in assets:
                if a in signal.columns:
                    signal[a] = (sector_rank[sector] - 0.5) * 2

    label = f"SECTOR_ROT_{lookback}d"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Regime-conditional TSMOM — adjust exposure by regime
# ══════════════════════════════════════════════════════════════════
print("\nEXP 7: Regime-conditional TSMOM...")

# Map monthly regime to daily
regime_daily = regime.reindex(prices.index, method="ffill")

base_signal = sum(tsmom_signal(prices, lb) for lb in [63, 126, 252]) / 3

for regime_rule_name, multipliers in [
    ("expansion_boost", {"Expansion": 1.2, "Slowdown": 0.5, "Stagflation": 0.8, "Recovery": 1.0}),
    ("stagflation_hedge", {"Expansion": 1.0, "Slowdown": 0.7, "Stagflation": 1.5, "Recovery": 1.0}),
    ("risk_on_off", {"Expansion": 1.5, "Slowdown": 0.3, "Stagflation": 0.3, "Recovery": 1.2}),
]:
    regime_mult = regime_daily.map(multipliers).fillna(1.0)
    signal = base_signal.multiply(regime_mult, axis=0)
    label = f"REGIME_TSMOM_{regime_rule_name}"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 8: Dual momentum — combine TSMOM + XSMOM
# ══════════════════════════════════════════════════════════════════
print("\nEXP 8: Dual momentum (TSMOM + XSMOM blend)...")

for lookback in [63, 126, 252]:
    ts = tsmom_signal(prices, lookback)
    xs_ret = prices.pct_change(lookback)
    xs = (xs_ret.rank(axis=1, pct=True) - 0.5) * 2

    for ts_w in [0.3, 0.5, 0.7]:
        signal = ts_w * ts + (1 - ts_w) * xs
        label = f"DUAL_MOM_{lookback}d_tsw{int(ts_w*100)}"
        results[label] = backtest_signal(signal, label)
        print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 9: Trend strength filter — only trade when trend is
# strong (absolute momentum above threshold)
# ══════════════════════════════════════════════════════════════════
print("\nEXP 9: Trend strength filter...")

base = sum(tsmom_signal(prices, lb) for lb in [63, 126, 252]) / 3
trend_strength = prices.pct_change(126).abs()

for threshold_pct in [0.10, 0.20, 0.30]:
    # Only trade assets where |126d return| > threshold
    strong = trend_strength > threshold_pct
    signal = base * strong.astype(float)
    label = f"TREND_FILTER_{int(threshold_pct*100)}pct"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 10: Multi-factor composite variants
# ══════════════════════════════════════════════════════════════════
print("\nEXP 10: Multi-factor composite weight variants...")

# Precompute factors
mom_126 = tsmom_signal(prices, 126)
mom_252 = tsmom_signal(prices, 252)
tsmom_avg = (mom_126 + mom_252) / 2

xs_ret_126 = prices.pct_change(126)
xsmom = (xs_ret_126.rank(axis=1, pct=True) - 0.5) * 2

carry = prices.pct_change(21)
carry_xs = (carry.rank(axis=1, pct=True) - 0.5) * 2

value_5y = prices.rolling(252*5, min_periods=252).mean()
val_dev = (value_5y - prices) / value_5y
value_sig = (val_dev.rank(axis=1, pct=True) - 0.5) * 2

reversal = (-prices.pct_change(5)).rank(axis=1, pct=True)
rev_sig = (reversal - 0.5) * 2

factor_combos = [
    ("TSMOM60_CARRY20_VAL20", {"tsmom": 0.6, "carry": 0.2, "value": 0.2, "reversal": 0.0, "xsmom": 0.0}),
    ("TSMOM40_XSMOM20_CARRY20_VAL20", {"tsmom": 0.4, "carry": 0.2, "value": 0.2, "reversal": 0.0, "xsmom": 0.2}),
    ("TSMOM50_CARRY30_REV20", {"tsmom": 0.5, "carry": 0.3, "value": 0.0, "reversal": 0.2, "xsmom": 0.0}),
    ("EQUAL_5FACTOR", {"tsmom": 0.2, "carry": 0.2, "value": 0.2, "reversal": 0.2, "xsmom": 0.2}),
    ("MOM_HEAVY", {"tsmom": 0.5, "carry": 0.1, "value": 0.1, "reversal": 0.0, "xsmom": 0.3}),
    ("CARRY_HEAVY", {"tsmom": 0.3, "carry": 0.4, "value": 0.2, "reversal": 0.1, "xsmom": 0.0}),
]

for label, w in factor_combos:
    signal = (
        w["tsmom"] * tsmom_avg +
        w["carry"] * carry_xs +
        w["value"] * value_sig +
        w["reversal"] * rev_sig +
        w["xsmom"] * xsmom
    )
    label = f"COMPOSITE_{label}"
    results[label] = backtest_signal(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("EXPERIMENT RESULTS — SORTED BY SHARPE")
print("="*80)

valid = {k: v for k, v in results.items() if "error" not in v}
ranked = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

print(f"\n{'Strategy':<45} {'Sharpe':>7} {'CAGR%':>7} {'Vol%':>6} {'MaxDD%':>7} {'WinR%':>6} {'Turnover':>9}")
print("-"*90)
for name, m in ranked:
    print(f"{name:<45} {m['sharpe']:>7.3f} {m['cagr']:>7.2f} {m['vol']:>6.1f} {m['max_dd']:>7.2f} {m['win_rate']:>6.1f} {m['daily_turnover']:>9.4f}")

print(f"\nTotal experiments: {len(results)}")
print(f"Positive Sharpe: {sum(1 for v in valid.values() if v['sharpe'] > 0)}/{len(valid)}")

# Top 5
print("\n── TOP 5 CANDIDATES ──")
for i, (name, m) in enumerate(ranked[:5], 1):
    print(f"  {i}. {name}")
    print(f"     Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2f}%, Vol={m['vol']:.1f}%, MaxDD={m['max_dd']:.2f}%")
    print(f"     Best year={m['best_year']:.1f}%, Worst={m['worst_year']:.1f}%, %Pos years={m['pct_positive_years']:.0f}%")

# Save
output = {
    "generated": pd.Timestamp.now().isoformat(),
    "data": "price_cache.pkl (22 assets, 2007-05-22 to 2026-04-10)",
    "vol_target": "12%",
    "rebalance": "weekly (Wednesday)",
    "results": dict(results),
    "ranking": [name for name, _ in ranked],
}

with open("/sessions/upbeat-adoring-goldberg/experiment_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nResults saved to experiment_results.json")
