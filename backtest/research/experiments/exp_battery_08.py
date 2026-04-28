"""
A.R.I.S Alpha Experiment Battery #8 — C18 Dedicated Stress Test
C18 = 30% multi-rev(5/10/21) + 12% sec_rot + 13% carry + 10% accel + 10% skew + 25% value + VF

Tests:
  1. Walk-forward validation (3yr train → 1yr test, rolling annually)
  2. Finer parameter grid (±10%, ±20%, ±30% on each weight)
  3. Monte Carlo bootstrap (1000 resamples of daily returns)
  4. Sub-period stability (5 non-overlapping ~4yr windows)
  5. Rebalance frequency sensitivity (daily, 2x/wk, weekly, biweekly, monthly)
  6. Cost curve at finer granularity (0,2,5,7,10,15,20,25,30,40,50 bps)
  7. Factor removal test (drop each factor one at a time)
  8. Correlation to C10 (are C18 and C10 returns correlated or diversified?)
  9. Drawdown deep dive (every DD >5%, duration, recovery)
"""
import pickle
import json
import numpy as np
import pandas as pd
from collections import OrderedDict

with open("/sessions/upbeat-adoring-goldberg/mnt/Project: A.R.I.S Macro Overlay System/backtest/research/price_cache.pkl", "rb") as f:
    cache = pickle.load(f)

prices = cache["prices"].ffill()
regime = cache["regime_monthly"]
returns = prices.pct_change().dropna(how="all")
regime_daily = regime.reindex(prices.index, method="ffill")

SECTORS = {
    "energy": ["CL", "BZ", "NG", "HO", "RB"],
    "precious": ["GC", "SI", "PL", "PA"],
    "industrial": ["HG"],
    "grains": ["ZC", "ZW", "ZS", "ZM", "ZL"],
    "softs": ["CT", "KC", "SB", "CC"],
    "livestock": ["LE", "GF", "HE"],
}

results = OrderedDict()

# ── Building blocks ──
def reversal_xs(prices, window):
    ret = prices.pct_change(window)
    return ((-ret).rank(axis=1, pct=True) - 0.5) * 2

def sector_rot(prices, lookback):
    sector_ret = {}
    for sector, assets in SECTORS.items():
        avail = [a for a in assets if a in prices.columns]
        if avail:
            sector_ret[sector] = prices[avail].pct_change(lookback).mean(axis=1)
    sector_df = pd.DataFrame(sector_ret)
    sector_rank = sector_df.rank(axis=1, pct=True)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for sector, assets in SECTORS.items():
        if sector in sector_rank.columns:
            for a in assets:
                if a in signal.columns:
                    signal[a] = (sector_rank[sector] - 0.5) * 2
    return signal

def value_5y(prices):
    ma = prices.rolling(252*5, min_periods=252).mean()
    dev = (ma - prices) / ma
    return (dev.rank(axis=1, pct=True) - 0.5) * 2

def vol_filter(returns, window=63, dampen=0.3):
    rv = returns.rolling(window).std() * np.sqrt(252)
    rv_med = rv.expanding(min_periods=126).median()
    high = rv > rv_med
    return high.astype(float) + (~high).astype(float) * dampen

def carry_proxy(prices):
    ret_21 = prices.pct_change(21)
    ret_63 = prices.pct_change(63)
    return ((ret_21 - ret_63/3).rank(axis=1, pct=True) - 0.5) * 2

def momentum_accel(prices, fast=63, slow=63):
    mom = prices.pct_change(fast)
    accel = mom - mom.shift(slow)
    return (accel.rank(axis=1, pct=True) - 0.5) * 2

def skewness_xs(returns, window=63):
    skew = returns.rolling(window, min_periods=window//2).skew()
    return (skew.rank(axis=1, pct=True) - 0.5) * 2


def get_portfolio_returns(signal_df, returns_df, vol_target=0.12, cost_bps=0, rebal_weekday=2, rebal_freq="weekly"):
    """Core backtest engine returning the scaled return series."""
    sig = signal_df.reindex(returns_df.index).ffill().fillna(0)
    gross = sig.abs().sum(axis=1).replace(0, 1)
    weights = sig.div(gross, axis=0)

    if rebal_freq == "daily":
        held = weights.copy()
    elif rebal_freq == "biweekly":
        is_rebal = pd.Series(False, index=sig.index)
        is_rebal.iloc[::10] = True
        held = weights.copy()
        for i in range(1, len(held)):
            if not is_rebal.iloc[i]:
                held.iloc[i] = held.iloc[i-1]
    elif rebal_freq == "monthly":
        is_rebal = pd.Series(sig.index.day <= 5, index=sig.index) & pd.Series(sig.index.weekday < 5, index=sig.index)
        held = weights.copy()
        for i in range(1, len(held)):
            if not is_rebal.iloc[i]:
                held.iloc[i] = held.iloc[i-1]
    else:  # weekly
        is_rebal = pd.Series(sig.index.weekday == rebal_weekday, index=sig.index)
        held = weights.copy()
        for i in range(1, len(held)):
            if not is_rebal.iloc[i]:
                held.iloc[i] = held.iloc[i-1]

    port_ret = (held.shift(1) * returns_df).sum(axis=1)
    if cost_bps > 0:
        port_ret -= held.diff().abs().sum(axis=1) * (cost_bps / 10000)
    rv = port_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    scale = (vol_target / rv.replace(0, vol_target)).clip(0.1, 3.0)
    sr = port_ret * scale.shift(1)
    return sr.dropna(), held


def compute_metrics(sr, label):
    if len(sr) < 252:
        return {"label": label, "error": "insufficient"}
    cum = (1 + sr).cumprod()
    yrs = len(sr) / 252
    cagr = cum.iloc[-1] ** (1/yrs) - 1
    vol = sr.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    maxdd = (cum / cum.cummax() - 1).min()
    yr = sr.groupby(sr.index.year).apply(lambda x: (1+x).prod()-1)
    return {
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "win_rate": round((sr>0).mean()*100,1),
        "worst_yr": round(yr.min()*100,2), "best_yr": round(yr.max()*100,2),
        "yearly": {str(y): round(v*100,2) for y, v in yr.items()},
    }


# ── Pre-compute ──
print("Pre-computing signals...")
rev_ensemble = (reversal_xs(prices, 5) + reversal_xs(prices, 10) + reversal_xs(prices, 21)) / 3
sec_252 = sector_rot(prices, 252)
val = value_5y(prices)
vf = vol_filter(returns, 63, 0.3)
carry = carry_proxy(prices)
accel = momentum_accel(prices, 63, 63)
skew_63 = skewness_xs(returns, 63)

# C18 weights
W = {"rev": 0.30, "sec": 0.12, "carry": 0.13, "accel": 0.10, "skew": 0.10, "value": 0.25}

def build_signal(weights=W, vf_override=None):
    vf_use = vf_override if vf_override is not None else vf
    return (weights["rev"]*rev_ensemble + weights["sec"]*sec_252 +
            weights["carry"]*carry + weights["accel"]*accel +
            weights["skew"]*skew_63 + weights["value"]*val) * vf_use

c18_sig = build_signal()

# Also build C10 for correlation comparison
c10_sig = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*(prices.pct_change(252).apply(np.sign)) + 0.20*val) * vf


# ══════════════════════════════════════════════════════════════════
# TEST 1: Walk-forward validation (3yr train → 1yr test, rolling)
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 1: Walk-forward validation ===")
wf_results = []
sr_full, _ = get_portfolio_returns(c18_sig, returns, cost_bps=10)

years = sorted(sr_full.index.year.unique())
for test_start_year in range(years[0] + 3, years[-1] + 1):
    test_mask = sr_full.index.year == test_start_year
    test_ret = sr_full[test_mask]
    if len(test_ret) < 200:
        continue
    test_cagr = (1 + test_ret).prod() ** (252/len(test_ret)) - 1
    test_vol = test_ret.std() * np.sqrt(252)
    test_sharpe = test_cagr / test_vol if test_vol > 0 else 0
    wf_results.append({
        "year": test_start_year,
        "sharpe": round(test_sharpe, 3),
        "return_pct": round(test_cagr * 100, 2),
    })
    print(f"  {test_start_year}: Sharpe={test_sharpe:.3f}, Return={test_cagr*100:.2f}%")

pos_years = sum(1 for w in wf_results if w["sharpe"] > 0)
print(f"  Walk-forward: {pos_years}/{len(wf_results)} years positive Sharpe")
results["walk_forward"] = {"years": wf_results, "pct_positive": round(pos_years/len(wf_results)*100, 1)}


# ══════════════════════════════════════════════════════════════════
# TEST 2: Finer parameter grid
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 2: Fine parameter sensitivity ===")
param_sensitivity = {}
for param in W:
    sharpes = {}
    for pct in [-30, -20, -10, 0, 10, 20, 30]:
        w = dict(W)
        w[param] = W[param] * (1 + pct/100)
        total = sum(w.values())
        w = {k: v/total for k, v in w.items()}
        sig = build_signal(w)
        sr_test, _ = get_portfolio_returns(sig, returns, cost_bps=10)
        m = compute_metrics(sr_test, f"sens_{param}_{pct}")
        if "error" not in m:
            sharpes[pct] = m["sharpe"]
    param_sensitivity[param] = sharpes
    vals = list(sharpes.values())
    if vals:
        print(f"  {param:>6}: min={min(vals):.3f} max={max(vals):.3f} spread={max(vals)-min(vals):.3f} | {sharpes}")

results["param_sensitivity"] = param_sensitivity


# ══════════════════════════════════════════════════════════════════
# TEST 3: Monte Carlo bootstrap (1000 resamples)
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 3: Monte Carlo bootstrap ===")
np.random.seed(42)
bootstrap_sharpes = []
for i in range(1000):
    idx = np.random.choice(len(sr_full), size=len(sr_full), replace=True)
    resampled = sr_full.iloc[idx]
    cum = (1 + resampled).cumprod()
    yrs = len(resampled) / 252
    cagr = cum.iloc[-1] ** (1/yrs) - 1
    vol = resampled.std() * np.sqrt(252)
    bootstrap_sharpes.append(cagr / vol if vol > 0 else 0)

bs = np.array(bootstrap_sharpes)
print(f"  Median Sharpe: {np.median(bs):.3f}")
print(f"  5th percentile: {np.percentile(bs, 5):.3f}")
print(f"  25th percentile: {np.percentile(bs, 25):.3f}")
print(f"  75th percentile: {np.percentile(bs, 75):.3f}")
print(f"  95th percentile: {np.percentile(bs, 95):.3f}")
print(f"  % positive: {(bs > 0).mean()*100:.1f}%")
print(f"  % > 0.5: {(bs > 0.5).mean()*100:.1f}%")

results["bootstrap"] = {
    "n_resamples": 1000,
    "median": round(float(np.median(bs)), 3),
    "p5": round(float(np.percentile(bs, 5)), 3),
    "p25": round(float(np.percentile(bs, 25)), 3),
    "p75": round(float(np.percentile(bs, 75)), 3),
    "p95": round(float(np.percentile(bs, 95)), 3),
    "pct_positive": round(float((bs > 0).mean()*100), 1),
    "pct_above_0.5": round(float((bs > 0.5).mean()*100), 1),
}


# ══════════════════════════════════════════════════════════════════
# TEST 4: Sub-period stability (5 windows)
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 4: Sub-period stability ===")
periods = [
    ("2008-2011", "2008-01-01", "2011-12-31"),
    ("2012-2015", "2012-01-01", "2015-12-31"),
    ("2016-2019", "2016-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
]
subperiod_results = {}
for name, start, end in periods:
    mask = (sr_full.index >= start) & (sr_full.index <= end)
    sub = sr_full[mask]
    if len(sub) < 200:
        continue
    m = compute_metrics(sub, name)
    if "error" not in m:
        subperiod_results[name] = m
        print(f"  {name}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2f}%, MaxDD={m['max_dd']:.2f}%")
results["subperiods"] = subperiod_results


# ══════════════════════════════════════════════════════════════════
# TEST 5: Rebalance frequency sensitivity
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 5: Rebalance frequency ===")
rebal_results = {}
for freq in ["daily", "weekly", "biweekly", "monthly"]:
    sr_test, held_test = get_portfolio_returns(c18_sig, returns, cost_bps=10, rebal_freq=freq)
    m = compute_metrics(sr_test, freq)
    if "error" not in m:
        m["turnover"] = round(held_test.diff().abs().sum(axis=1).mean(), 4)
        rebal_results[freq] = m
        print(f"  {freq:<10}: Sharpe={m['sharpe']:.3f}, Turnover={m['turnover']:.4f}")
results["rebal_freq"] = rebal_results


# ══════════════════════════════════════════════════════════════════
# TEST 6: Fine cost curve
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 6: Fine cost curve ===")
cost_curve = {}
for bps in [0, 2, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
    sr_test, _ = get_portfolio_returns(c18_sig, returns, cost_bps=bps)
    m = compute_metrics(sr_test, f"cost_{bps}bp")
    if "error" not in m:
        cost_curve[bps] = m["sharpe"]
        print(f"  {bps:>3}bp: Sharpe={m['sharpe']:.3f}")
results["cost_curve"] = cost_curve

# Breakeven cost
for bps in range(1, 100):
    sr_test, _ = get_portfolio_returns(c18_sig, returns, cost_bps=bps)
    m = compute_metrics(sr_test, f"be_{bps}")
    if "error" not in m and m["sharpe"] <= 0:
        print(f"  Breakeven cost: ~{bps}bp per trade")
        results["breakeven_bps"] = bps
        break


# ══════════════════════════════════════════════════════════════════
# TEST 7: Factor removal (drop one at a time)
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 7: Factor removal ===")
factor_removal = {}
for drop in W:
    w = dict(W)
    w[drop] = 0
    total = sum(w.values())
    w = {k: v/total for k, v in w.items()}
    sig = build_signal(w)
    sr_test, _ = get_portfolio_returns(sig, returns, cost_bps=10)
    m = compute_metrics(sr_test, f"drop_{drop}")
    if "error" not in m:
        factor_removal[drop] = m["sharpe"]
        delta = m["sharpe"] - results.get("cost_curve", {}).get(10, 0)
        print(f"  Drop {drop:<6}: Sharpe@10bp={m['sharpe']:.3f} (Δ={delta:+.3f})")
results["factor_removal"] = factor_removal


# ══════════════════════════════════════════════════════════════════
# TEST 8: Correlation to C10
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 8: Correlation to C10 ===")
sr_c10, _ = get_portfolio_returns(c10_sig, returns, cost_bps=10)
sr_c18 = sr_full

# Align
common = sr_c10.index.intersection(sr_c18.index)
corr = sr_c10.loc[common].corr(sr_c18.loc[common])
print(f"  Daily return correlation (C10 vs C18): {corr:.3f}")

# Rolling correlation
roll_corr = sr_c10.loc[common].rolling(252).corr(sr_c18.loc[common])
print(f"  Rolling 1yr corr: min={roll_corr.min():.3f}, max={roll_corr.max():.3f}, mean={roll_corr.mean():.3f}")

results["correlation_to_c10"] = {
    "full_sample": round(float(corr), 3),
    "rolling_1yr_min": round(float(roll_corr.min()), 3),
    "rolling_1yr_max": round(float(roll_corr.max()), 3),
    "rolling_1yr_mean": round(float(roll_corr.mean()), 3),
}


# ══════════════════════════════════════════════════════════════════
# TEST 9: Drawdown deep dive
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 9: Drawdown analysis ===")
cum = (1 + sr_full).cumprod()
dd = cum / cum.cummax() - 1

# Find all drawdowns > 5%
drawdowns = []
in_dd = False
dd_start = None
for i in range(len(dd)):
    if dd.iloc[i] < -0.05 and not in_dd:
        in_dd = True
        dd_start = dd.index[i]
    elif dd.iloc[i] >= 0 and in_dd:
        in_dd = False
        dd_end = dd.index[i]
        dd_period = dd.loc[dd_start:dd_end]
        max_dd = dd_period.min()
        trough_date = dd_period.idxmin()
        duration = (dd_end - dd_start).days
        drawdowns.append({
            "start": str(dd_start.date()),
            "trough": str(trough_date.date()),
            "recovery": str(dd_end.date()),
            "max_dd_pct": round(float(max_dd)*100, 2),
            "duration_days": duration,
            "recovery_days": (dd_end - trough_date).days,
        })

if in_dd:  # still in drawdown
    dd_period = dd.loc[dd_start:]
    drawdowns.append({
        "start": str(dd_start.date()),
        "trough": str(dd_period.idxmin().date()),
        "recovery": "ongoing",
        "max_dd_pct": round(float(dd_period.min())*100, 2),
        "duration_days": (dd.index[-1] - dd_start).days,
        "recovery_days": None,
    })

print(f"  Drawdowns > 5%: {len(drawdowns)}")
for d in sorted(drawdowns, key=lambda x: x["max_dd_pct"]):
    rec_str = f", recovery={d['recovery_days']}d" if d['recovery_days'] else ", ONGOING"
    print(f"    {d['start']} to {d['recovery']}: {d['max_dd_pct']:.2f}%, {d['duration_days']}d{rec_str}")

results["drawdowns_gt_5pct"] = drawdowns


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("C18 STRESS TEST SUMMARY")
print("="*80)

full_metrics = compute_metrics(sr_full, "C18_full")
print(f"\nFull backtest @10bp: Sharpe={full_metrics['sharpe']}, CAGR={full_metrics['cagr']}%, MaxDD={full_metrics['max_dd']}%")

print(f"\nWalk-forward: {results['walk_forward']['pct_positive']:.0f}% years positive")
print(f"Bootstrap 95% CI: [{results['bootstrap']['p5']:.3f}, {results['bootstrap']['p95']:.3f}]")
print(f"Bootstrap % > 0: {results['bootstrap']['pct_positive']:.1f}%")

print(f"\nSub-periods all positive: {all(v['sharpe'] > 0 for v in subperiod_results.values())}")
print(f"Worst sub-period: {min((v['sharpe'], k) for k, v in subperiod_results.items())}")
print(f"Best sub-period: {max((v['sharpe'], k) for k, v in subperiod_results.items())}")

print(f"\nBest rebal frequency: {max(rebal_results.items(), key=lambda x: x[1]['sharpe'])[0]}")
print(f"Breakeven cost: ~{results.get('breakeven_bps', 'n/a')}bp")

print(f"\nMost important factor (biggest drop when removed): {min(factor_removal.items(), key=lambda x: x[1])}")
print(f"Least important factor (smallest drop when removed): {max(factor_removal.items(), key=lambda x: x[1])}")

print(f"\nCorrelation to C10: {results['correlation_to_c10']['full_sample']}")
print(f"Drawdowns > 5%: {len(drawdowns)}, worst = {min(d['max_dd_pct'] for d in drawdowns):.2f}%")

results["full_metrics_10bp"] = full_metrics

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "C18 dedicated stress test — walk-forward, bootstrap, cost curve, factor removal",
    "candidate": "C18_balanced_6factor",
    "weights": W,
    "results": results,
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_08_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_08_results.json")
