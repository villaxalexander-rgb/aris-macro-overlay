"""
A.R.I.S Alpha Experiment Battery #6 — Advanced Price-Derived Factors
Baseline: C10 Multi-Rev Carry (Sharpe 1.068 @0bp / 0.713 @10bp)

New factor experiments:
  1. Skewness factor — short negatively-skewed assets, long positively-skewed
  2. Momentum acceleration — rate of change of momentum (2nd derivative)
  3. Dispersion timing — scale reversal exposure based on cross-sectional dispersion
  4. Seasonal patterns — monthly return seasonal factors
  5. Regime-corrected C10 — fix the regime overlay (correct regime names)
  6. Combined best from all batteries
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

# ── Building blocks (reuse from previous batteries) ──
def reversal_xs(prices, window):
    ret = prices.pct_change(window)
    return ((-ret).rank(axis=1, pct=True) - 0.5) * 2

def tsmom(prices, lb):
    return prices.pct_change(lb).apply(np.sign)

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
    return high.astype(float) * 1.0 + (~high).astype(float) * dampen

def carry_proxy(prices):
    ret_21 = prices.pct_change(21)
    ret_63 = prices.pct_change(63)
    return ((ret_21 - ret_63/3).rank(axis=1, pct=True) - 0.5) * 2


def backtest(signal_df, label, vol_target=0.12, cost_bps=0):
    sig = signal_df.reindex(returns.index).ffill().fillna(0)
    gross = sig.abs().sum(axis=1).replace(0, 1)
    weights = sig.div(gross, axis=0)
    is_rebal = pd.Series(sig.index.weekday == 2, index=sig.index)
    held = weights.copy()
    for i in range(1, len(held)):
        if not is_rebal.iloc[i]:
            held.iloc[i] = held.iloc[i-1]
    port_ret = (held.shift(1) * returns).sum(axis=1)
    if cost_bps > 0:
        port_ret -= held.diff().abs().sum(axis=1) * (cost_bps / 10000)
    rv = port_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    scale = (vol_target / rv.replace(0, vol_target)).clip(0.1, 3.0)
    sr = port_ret * scale.shift(1)
    sr = sr.dropna()
    if len(sr) < 252:
        return {"label": label, "error": "insufficient"}
    cum = (1 + sr).cumprod()
    yrs = len(sr) / 252
    cagr = cum.iloc[-1] ** (1/yrs) - 1
    vol = sr.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    maxdd = (cum / cum.cummax() - 1).min()
    yr = sr.groupby(sr.index.year).apply(lambda x: (1+x).prod()-1)

    oos_mask = sr.index >= "2019-01-01"
    is_ret, oos_ret = sr[~oos_mask], sr[oos_mask]
    is_sharpe = oos_sharpe = None
    if len(is_ret) > 252:
        is_c = (1+is_ret).prod() ** (252/len(is_ret)) - 1
        is_v = is_ret.std() * np.sqrt(252)
        is_sharpe = round(is_c / is_v if is_v > 0 else 0, 3)
    if len(oos_ret) > 252:
        oos_c = (1+oos_ret).prod() ** (252/len(oos_ret)) - 1
        oos_v = oos_ret.std() * np.sqrt(252)
        oos_sharpe = round(oos_c / oos_v if oos_v > 0 else 0, 3)

    return {
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "turnover": round(held.diff().abs().sum(axis=1).mean(),4),
        "win_rate": round((sr>0).mean()*100,1),
        "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "worst_yr": round(yr.min()*100,2), "best_yr": round(yr.max()*100,2),
    }


# ── Pre-compute ──
print("Pre-computing signals...")
rev_5 = reversal_xs(prices, 5)
rev_10 = reversal_xs(prices, 10)
rev_21 = reversal_xs(prices, 21)
rev_ensemble = (rev_5 + rev_10 + rev_21) / 3
sec_252 = sector_rot(prices, 252)
ts_252 = tsmom(prices, 252)
val = value_5y(prices)
vf = vol_filter(returns, 63, 0.3)
carry = carry_proxy(prices)


# ── C10 baseline ──
c10 = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*ts_252 + 0.20*val) * vf
results["C10_BASELINE"] = backtest(c10, "C10_BASELINE")
results["C10_BASELINE_10bp"] = backtest(c10, "C10_BASELINE_10bp", cost_bps=10)
print(f"C10 baseline: Sharpe={results['C10_BASELINE']['sharpe']} / @10bp={results['C10_BASELINE_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# EXP 1: Skewness factor
# Rolling skewness of returns — long positive skew, short negative
# ══════════════════════════════════════════════════════════════════
print("\nEXP1: Skewness factor...")

for skew_window in [63, 126, 252]:
    skew = returns.rolling(skew_window, min_periods=skew_window//2).skew()
    skew_xs = (skew.rank(axis=1, pct=True) - 0.5) * 2  # long positive skew

    # Standalone
    sig = skew_xs * vf
    results[f"E1_skew_{skew_window}d"] = backtest(sig, f"E1_skew_{skew_window}d")
    print(f"  skew_{skew_window}d standalone: Sharpe={results[f'E1_skew_{skew_window}d']['sharpe']}")

    # Blend with C10 (replace sector_rot with skew)
    for sw in [0.10, 0.15, 0.20]:
        remainder = 1.0 - sw
        sig = (remainder*0.40/0.85)*rev_ensemble + sw*skew_xs + (remainder*0.15/0.85)*carry + (remainder*0.10/0.85)*ts_252 + (remainder*0.20/0.85)*val
        sig = sig * vf
        key = f"E1_c10+skew{skew_window}_{int(sw*100)}"
        results[key] = backtest(sig, key)
        results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    candidates_10 = [(k, v['sharpe']) for k, v in results.items() if k.startswith(f'E1_c10+skew{skew_window}') and k.endswith('_10bp') and 'error' not in v]
    best = max(candidates_10, default=('none', 0), key=lambda x: x[1])
    print(f"  best blend@10bp: {best}")


# ══════════════════════════════════════════════════════════════════
# EXP 2: Momentum acceleration
# 2nd derivative: change in momentum. If momentum is accelerating,
# stronger signal. Decelerating momentum = reduce.
# ══════════════════════════════════════════════════════════════════
print("\nEXP2: Momentum acceleration...")

mom_252 = prices.pct_change(252)
mom_63 = prices.pct_change(63)
mom_accel = mom_63 - mom_63.shift(63)  # momentum acceleration
accel_xs = (mom_accel.rank(axis=1, pct=True) - 0.5) * 2

# Standalone
sig = accel_xs * vf
results["E2_accel_standalone"] = backtest(sig, "E2_accel_standalone")
print(f"  accel standalone: Sharpe={results['E2_accel_standalone']['sharpe']}")

# Use accel to scale TSMOM component of C10
for accel_w in [0.10, 0.15]:
    # Replace tsmom weight with accel
    sig = 0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + accel_w*accel_xs + (0.20-accel_w+0.10)*val/(0.30)*0.20
    # Simpler: just add accel as replacement for some tsmom
    sig = 0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + (0.10-accel_w)*ts_252 + accel_w*accel_xs + 0.20*val
    sig = sig * vf
    key = f"E2_c10+accel_{int(accel_w*100)}"
    results[key] = backtest(sig, key)
    results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    print(f"  {key}: Sharpe={results[key]['sharpe']} / @10bp={results[f'{key}_10bp']['sharpe']}")

# Replace tsmom entirely with accel
sig = 0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*accel_xs + 0.20*val
sig = sig * vf
results["E2_c10_accel_replace"] = backtest(sig, "E2_c10_accel_replace")
results["E2_c10_accel_replace_10bp"] = backtest(sig, "E2_c10_accel_replace_10bp", cost_bps=10)
print(f"  accel replaces tsmom: Sharpe={results['E2_c10_accel_replace']['sharpe']} / @10bp={results['E2_c10_accel_replace_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# EXP 3: Dispersion timing
# When cross-sectional return dispersion is high, reversal
# should work better (more mean reversion opportunities).
# Scale reversal weight based on dispersion.
# ══════════════════════════════════════════════════════════════════
print("\nEXP3: Dispersion timing...")

xs_dispersion = returns.rolling(21).std().std(axis=1)  # cross-sectional vol of vols
disp_high = xs_dispersion > xs_dispersion.expanding(min_periods=126).median()

for rev_high, rev_low, name in [
    (0.55, 0.30, "disp_aggressive"),    # tilt heavily to rev in high dispersion
    (0.50, 0.35, "disp_moderate"),
    (0.45, 0.38, "disp_mild"),
]:
    rev_w = pd.Series(rev_low, index=prices.index)
    disp_aligned = disp_high.reindex(prices.index, fill_value=False)
    rev_w.loc[disp_aligned] = rev_high

    # Rescale other weights proportionally
    other_total = 0.15 + 0.15 + 0.10 + 0.20  # sec + carry + tsmom + val = 0.60
    other_scale = (1.0 - rev_w) / other_total

    sig_df = (rev_w.values[:, None] * rev_ensemble.values +
              (other_scale.values[:, None] * 0.15) * sec_252.values +
              (other_scale.values[:, None] * 0.15) * carry.values +
              (other_scale.values[:, None] * 0.10) * ts_252.values +
              (other_scale.values[:, None] * 0.20) * val.values)
    sig = pd.DataFrame(sig_df, index=prices.index, columns=prices.columns) * vf

    results[f"E3_{name}"] = backtest(sig, f"E3_{name}")
    results[f"E3_{name}_10bp"] = backtest(sig, f"E3_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E3_{name}']['sharpe']} / @10bp={results[f'E3_{name}_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# EXP 4: Seasonal patterns
# Commodities have well-documented seasonality (heating oil winter,
# grains planting/harvest). Use trailing 5yr average monthly return
# as a seasonal overlay.
# ══════════════════════════════════════════════════════════════════
print("\nEXP4: Seasonal patterns...")

# Simple seasonal: historical average return by calendar month (full sample)
monthly_ret = returns.resample("ME").sum()
seasonal_mean = monthly_ret.groupby(monthly_ret.index.month).mean()  # 12 x 22
# Map each trading day to its month's seasonal signal
seasonal_daily = prices.copy() * 0
for i, dt in enumerate(prices.index):
    m = dt.month
    if m in seasonal_mean.index:
        seasonal_daily.iloc[i] = seasonal_mean.loc[m]
seasonal_xs = (seasonal_daily.rank(axis=1, pct=True) - 0.5) * 2

# Standalone
sig = seasonal_xs.fillna(0) * vf
results["E4_seasonal_standalone"] = backtest(sig, "E4_seasonal_standalone")
print(f"  seasonal standalone: Sharpe={results['E4_seasonal_standalone']['sharpe']}")

# Blend with C10
for sw in [0.05, 0.10, 0.15]:
    scale = 1.0 - sw
    sig = scale*(0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*ts_252 + 0.20*val) + sw*seasonal_xs.fillna(0)
    sig = sig * vf
    key = f"E4_c10+seasonal_{int(sw*100)}"
    results[key] = backtest(sig, key)
    results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    print(f"  {key}: Sharpe={results[key]['sharpe']} / @10bp={results[f'{key}_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# EXP 5: Regime-corrected overlay
# Battery #2 regime overlay failed because regime names were wrong.
# Now using correct names: Goldilocks, Deflation, Stagflation, Reflation
# ══════════════════════════════════════════════════════════════════
print("\nEXP5: Regime overlay (corrected names)...")

for rule_name, mults in [
    ("amp_goldilocks", {"Goldilocks": 1.3, "Deflation": 0.7, "Stagflation": 1.0, "Reflation": 0.8}),
    ("amp_stagflation", {"Goldilocks": 0.8, "Deflation": 0.8, "Stagflation": 1.5, "Reflation": 1.0}),
    ("amp_reflation", {"Goldilocks": 0.8, "Deflation": 0.7, "Stagflation": 1.0, "Reflation": 1.5}),
    ("reduce_deflation", {"Goldilocks": 1.0, "Deflation": 0.3, "Stagflation": 1.0, "Reflation": 1.0}),
    ("kill_deflation", {"Goldilocks": 1.0, "Deflation": 0.0, "Stagflation": 1.0, "Reflation": 1.0}),
    ("optimized", {"Goldilocks": 1.2, "Deflation": 0.5, "Stagflation": 1.3, "Reflation": 1.0}),
]:
    mult = regime_daily.map(mults).fillna(1.0)
    sig = c10.multiply(mult, axis=0)
    results[f"E5_{rule_name}"] = backtest(sig, f"E5_{rule_name}")
    results[f"E5_{rule_name}_10bp"] = backtest(sig, f"E5_{rule_name}_10bp", cost_bps=10)
    print(f"  {rule_name}: Sharpe={results[f'E5_{rule_name}']['sharpe']} / @10bp={results[f'E5_{rule_name}_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# EXP 6: Kitchen sink — combine best findings from all batteries
# ══════════════════════════════════════════════════════════════════
print("\nEXP6: Combined best-of-all candidates...")

# C13: C10 + dispersion timing (moderate) + regime correction (optimized)
disp_rev_w = pd.Series(0.35, index=prices.index)
disp_aligned_c13 = disp_high.reindex(prices.index, fill_value=False)
disp_rev_w.loc[disp_aligned_c13] = 0.50
other_s = (1.0 - disp_rev_w) / 0.60
sig_c13 = (disp_rev_w.values[:, None] * rev_ensemble.values +
           (other_s.values[:, None] * 0.15) * sec_252.values +
           (other_s.values[:, None] * 0.15) * carry.values +
           (other_s.values[:, None] * 0.10) * ts_252.values +
           (other_s.values[:, None] * 0.20) * val.values)
sig_c13 = pd.DataFrame(sig_c13, index=prices.index, columns=prices.columns) * vf
regime_mult = regime_daily.map({"Goldilocks": 1.2, "Deflation": 0.5, "Stagflation": 1.3, "Reflation": 1.0}).fillna(1.0)
sig_c13 = sig_c13.multiply(regime_mult, axis=0)
results["C13_disp_regime"] = backtest(sig_c13, "C13_disp_regime")
results["C13_disp_regime_10bp"] = backtest(sig_c13, "C13_disp_regime_10bp", cost_bps=10)
print(f"  C13 (disp+regime): Sharpe={results['C13_disp_regime']['sharpe']} / @10bp={results['C13_disp_regime_10bp']['sharpe']}")

# C14: C10 + accel replaces tsmom + regime
sig_c14_raw = 0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*accel_xs + 0.20*val
sig_c14 = sig_c14_raw * vf
sig_c14 = sig_c14.multiply(regime_mult, axis=0)
results["C14_accel_regime"] = backtest(sig_c14, "C14_accel_regime")
results["C14_accel_regime_10bp"] = backtest(sig_c14, "C14_accel_regime_10bp", cost_bps=10)
print(f"  C14 (accel+regime): Sharpe={results['C14_accel_regime']['sharpe']} / @10bp={results['C14_accel_regime_10bp']['sharpe']}")

# C15: C10 + skewness factor (small allocation) + regime
skew_126 = returns.rolling(126, min_periods=63).skew()
skew_xs = (skew_126.rank(axis=1, pct=True) - 0.5) * 2
sig_c15_raw = 0.35*rev_ensemble + 0.10*sec_252 + 0.15*carry + 0.10*ts_252 + 0.10*skew_xs + 0.20*val
sig_c15 = sig_c15_raw * vf
sig_c15 = sig_c15.multiply(regime_mult, axis=0)
results["C15_skew_regime"] = backtest(sig_c15, "C15_skew_regime")
results["C15_skew_regime_10bp"] = backtest(sig_c15, "C15_skew_regime_10bp", cost_bps=10)
print(f"  C15 (skew+regime): Sharpe={results['C15_skew_regime']['sharpe']} / @10bp={results['C15_skew_regime_10bp']['sharpe']}")

# C16: Full kitchen sink — 7 factors
sig_c16_raw = (0.30*rev_ensemble + 0.10*sec_252 + 0.12*carry + 0.08*ts_252 +
               0.05*accel_xs + 0.05*skew_xs + 0.15*val + 0.05*seasonal_xs.fillna(0))
# Note: weights sum to 0.90, the remaining 0.10 is implicit scaling
sig_c16_raw = sig_c16_raw / 0.90  # renormalize
sig_c16 = sig_c16_raw * vf
results["C16_7factor"] = backtest(sig_c16, "C16_7factor")
results["C16_7factor_10bp"] = backtest(sig_c16, "C16_7factor_10bp", cost_bps=10)
print(f"  C16 (7-factor): Sharpe={results['C16_7factor']['sharpe']} / @10bp={results['C16_7factor_10bp']['sharpe']}")

# C17: 7-factor + regime
sig_c17 = sig_c16.multiply(regime_mult, axis=0)
results["C17_7factor_regime"] = backtest(sig_c17, "C17_7factor_regime")
results["C17_7factor_regime_10bp"] = backtest(sig_c17, "C17_7factor_regime_10bp", cost_bps=10)
print(f"  C17 (7-factor+regime): Sharpe={results['C17_7factor_regime']['sharpe']} / @10bp={results['C17_7factor_regime_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*120)
print("BATTERY #6 — SORTED BY SHARPE (0bp)")
print("="*120)

valid = {k: v for k, v in results.items() if "error" not in v and not k.endswith("_10bp")}
ranked = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

print(f"\n{'Strategy':<40} {'Sharpe':>7} {'CAGR%':>7} {'Vol%':>6} {'MaxDD%':>7} {'IS':>6} {'OOS':>6} {'WR%':>5} {'T/O':>7}")
print("-"*100)
for name, m in ranked:
    is_s = f"{m['is_sharpe']:.3f}" if m.get('is_sharpe') is not None else "  n/a"
    oos_s = f"{m['oos_sharpe']:.3f}" if m.get('oos_sharpe') is not None else "  n/a"
    print(f"{name:<40} {m['sharpe']:>7.3f} {m['cagr']:>7.2f} {m['vol']:>6.1f} {m['max_dd']:>7.2f} {is_s:>6} {oos_s:>6} {m['win_rate']:>5.1f} {m['turnover']:>7.4f}")

# Cost comparison
print(f"\n{'Strategy':<40} {'@0bp':>7} {'@10bp':>7}")
print("-"*55)
for name, m in ranked[:15]:
    key_10 = f"{name}_10bp"
    s10 = results[key_10]["sharpe"] if key_10 in results and "error" not in results[key_10] else None
    s10_str = f"{s10:.3f}" if s10 is not None else "  n/a"
    print(f"{name:<40} {m['sharpe']:>7.3f} {s10_str:>7}")

c10_sharpe = results.get("C10_BASELINE", {}).get("sharpe", 0)
better = [(n,m) for n,m in ranked if m["sharpe"] > c10_sharpe and n != "C10_BASELINE"]
print(f"\n── C10 Baseline Sharpe: {c10_sharpe:.3f} ──")
print(f"── Strategies that beat C10: {len(better)}/{len(valid)-1} ──")

if better:
    print("\n── TOP CANDIDATES (beat C10) ──")
    for i, (name, m) in enumerate(better[:10], 1):
        key_10 = f"{name}_10bp"
        s10 = results[key_10]["sharpe"] if key_10 in results and "error" not in results[key_10] else None
        s10_str = f"{s10:.3f}" if s10 is not None else "n/a"
        print(f"  {i}. {name}: Sharpe={m['sharpe']:.3f} (@10bp={s10_str}), "
              f"CAGR={m['cagr']:.2f}%, MaxDD={m['max_dd']:.2f}%, "
              f"IS={m.get('is_sharpe','n/a')}, OOS={m.get('oos_sharpe','n/a')}")

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "Advanced factors: skewness, accel, dispersion timing, seasonal, regime correction",
    "baseline": "C10_BASELINE",
    "results": dict(results),
    "ranking": [name for name, _ in ranked],
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_06_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_06_results.json")
