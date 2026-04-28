"""
A.R.I.S Alpha Experiment Battery #7 — Clean 6-Factor Stress Test
Baseline: C10 Multi-Rev Carry (Sharpe 1.068 @0bp / 0.713 @10bp)

Agenda:
  1. Fix seasonal: expanding-window monthly average (no look-ahead)
  2. Stress-test momentum acceleration as TSMOM replacement
  3. Build clean 6-factor composite (rev + sec_rot + carry + accel + skew + value)
  4. Parameter sensitivity on 6-factor
  5. Cost curves, IS/OOS, rolling stability, regime breakdown
  6. Debug regime overlay (check why it has zero effect)
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

def momentum_accel(prices, fast=63, slow=63):
    """2nd derivative of momentum: change in momentum over 'slow' days."""
    mom = prices.pct_change(fast)
    accel = mom - mom.shift(slow)
    return (accel.rank(axis=1, pct=True) - 0.5) * 2

def skewness_xs(returns, window=63):
    skew = returns.rolling(window, min_periods=window//2).skew()
    return (skew.rank(axis=1, pct=True) - 0.5) * 2

def seasonal_expanding(returns):
    """Expanding-window seasonal: for each day, average return for that
    calendar month using ONLY data up to that point (no look-ahead)."""
    monthly_ret = returns.resample("ME").sum()
    # Build expanding average per month
    seasonal_signal = pd.DataFrame(np.nan, index=monthly_ret.index, columns=monthly_ret.columns)
    for month in range(1, 13):
        mask = monthly_ret.index.month == month
        month_data = monthly_ret[mask]
        expanding_mean = month_data.expanding(min_periods=3).mean()
        seasonal_signal.loc[mask] = expanding_mean.values
    # Forward-fill to daily
    seasonal_daily = seasonal_signal.reindex(prices.index, method="ffill")
    return (seasonal_daily.rank(axis=1, pct=True) - 0.5) * 2


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
    turnover_avg = held.diff().abs().sum(axis=1).mean()

    # IS/OOS
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

    # Rolling 3yr Sharpe
    roll_3yr = sr.rolling(756, min_periods=504).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    roll_min = round(roll_3yr.min(), 3) if not roll_3yr.dropna().empty else None
    roll_pct_pos = round((roll_3yr > 0).mean()*100, 1) if not roll_3yr.dropna().empty else None

    # Regime
    regime_sharpe = {}
    for r in regime_daily.dropna().unique():
        mask = regime_daily == r
        r_ret = sr[mask.reindex(sr.index, fill_value=False)]
        if len(r_ret) > 126:
            r_cagr = (1 + r_ret).prod() ** (252/len(r_ret)) - 1
            r_vol = r_ret.std() * np.sqrt(252)
            regime_sharpe[r] = round(r_cagr / r_vol if r_vol > 0 else 0, 3)

    return {
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "turnover": round(turnover_avg,4),
        "win_rate": round((sr>0).mean()*100,1),
        "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "roll_3yr_min": roll_min, "roll_3yr_pct_pos": roll_pct_pos,
        "regime_sharpe": regime_sharpe,
        "worst_yr": round(yr.min()*100,2), "best_yr": round(yr.max()*100,2),
        "yearly": {str(y): round(v*100,2) for y, v in yr.items()},
    }


# ── Pre-compute ──
print("Pre-computing signals...")
rev_ensemble = (reversal_xs(prices, 5) + reversal_xs(prices, 10) + reversal_xs(prices, 21)) / 3
sec_252 = sector_rot(prices, 252)
ts_252 = tsmom(prices, 252)
val = value_5y(prices)
vf = vol_filter(returns, 63, 0.3)
carry = carry_proxy(prices)
accel = momentum_accel(prices, fast=63, slow=63)
skew_63 = skewness_xs(returns, window=63)


# ══════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════
print("\n=== BASELINES ===")
c10 = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*ts_252 + 0.20*val) * vf
results["C10_baseline"] = backtest(c10, "C10_baseline")
results["C10_baseline_10bp"] = backtest(c10, "C10_baseline_10bp", cost_bps=10)
print(f"C10: {results['C10_baseline']['sharpe']} / @10bp={results['C10_baseline_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# PART 0: Debug regime overlay
# ══════════════════════════════════════════════════════════════════
print("\n=== REGIME DEBUG ===")
print(f"regime_daily unique values: {regime_daily.dropna().unique()}")
print(f"regime_daily dtype: {regime_daily.dtype}")
print(f"regime_daily sample: {regime_daily.dropna().head(3).tolist()}")
# Test a manual map
test_map = {"Goldilocks": 2.0, "Deflation": 0.5, "Stagflation": 1.5, "Reflation": 0.8}
mapped = regime_daily.map(test_map)
print(f"After map: unique={mapped.dropna().unique()}, NaN count={mapped.isna().sum()}, total={len(mapped)}")
print(f"Map works: {not mapped.isna().all()}")


# ══════════════════════════════════════════════════════════════════
# PART 1: Fix seasonal (expanding window, no look-ahead)
# ══════════════════════════════════════════════════════════════════
print("\n=== PART 1: Clean seasonal ===")
seasonal = seasonal_expanding(returns)
sig = seasonal.fillna(0) * vf
results["seasonal_clean_standalone"] = backtest(sig, "seasonal_clean_standalone")
print(f"Clean seasonal standalone: Sharpe={results['seasonal_clean_standalone']['sharpe']}")

for sw in [0.05, 0.10, 0.15]:
    scale = 1.0 - sw
    sig = scale*c10 + sw * seasonal.fillna(0) * vf
    key = f"seasonal_clean_{int(sw*100)}"
    results[key] = backtest(sig, key)
    results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    print(f"  C10 + {int(sw*100)}% seasonal: {results[key]['sharpe']} / @10bp={results[f'{key}_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# PART 2: Stress-test momentum acceleration
# ══════════════════════════════════════════════════════════════════
print("\n=== PART 2: Accel stress test ===")

# Window sensitivity
for fast, slow in [(21, 21), (21, 63), (63, 63), (63, 126), (126, 126), (126, 252)]:
    acc = momentum_accel(prices, fast=fast, slow=slow)
    sig = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*acc + 0.20*val) * vf
    key = f"accel_f{fast}_s{slow}"
    results[key] = backtest(sig, key)
    results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    print(f"  fast={fast} slow={slow}: {results[key]['sharpe']} / @10bp={results[f'{key}_10bp']['sharpe']}")

# Weight sensitivity (keeping fast=63, slow=63)
for aw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    # Take from TSMOM + value proportionally
    remaining = 1.0 - 0.40 - 0.15 - 0.15 - aw  # rev + sec + carry + accel
    val_w = remaining * (0.20 / 0.30)
    ts_w = remaining * (0.10 / 0.30)
    sig = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + aw*accel + val_w*val + ts_w*ts_252) * vf
    # Actually simpler: just replace tsmom with accel at various levels
    sig = (0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + aw*accel + (0.10-aw)*ts_252 + 0.20*val) * vf
    key = f"accel_w{int(aw*100)}"
    results[key] = backtest(sig, key)
    results[f"{key}_10bp"] = backtest(sig, f"{key}_10bp", cost_bps=10)
    print(f"  accel_weight={aw:.2f}: {results[key]['sharpe']} / @10bp={results[f'{key}_10bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# PART 3: Clean 6-factor composite (no seasonal)
# rev + sec_rot + carry + accel + skew + value, with VF
# ══════════════════════════════════════════════════════════════════
print("\n=== PART 3: Clean 6-factor composites ===")

configs = [
    # (name, rev, sec, carry, accel, skew, value)
    ("C18_balanced",     0.30, 0.12, 0.13, 0.10, 0.10, 0.25),
    ("C19_rev_heavy",    0.35, 0.10, 0.13, 0.10, 0.07, 0.25),
    ("C20_accel_heavy",  0.30, 0.10, 0.12, 0.18, 0.05, 0.25),
    ("C21_skew_heavy",   0.28, 0.10, 0.12, 0.08, 0.17, 0.25),
    ("C22_carry_heavy",  0.30, 0.08, 0.22, 0.08, 0.07, 0.25),
    ("C23_value_light",  0.33, 0.12, 0.13, 0.12, 0.10, 0.20),
    ("C24_minimal_6f",   0.35, 0.10, 0.15, 0.10, 0.10, 0.20),
]

for name, rw, sw, cw, aw, skw, vw in configs:
    assert abs(rw + sw + cw + aw + skw + vw - 1.0) < 0.001, f"{name} weights don't sum to 1"
    sig = (rw*rev_ensemble + sw*sec_252 + cw*carry + aw*accel + skw*skew_63 + vw*val) * vf
    results[name] = backtest(sig, name)
    results[f"{name}_10bp"] = backtest(sig, f"{name}_10bp", cost_bps=10)
    for bps in [5, 15, 20, 30]:
        results[f"{name}_{bps}bp"] = backtest(sig, f"{name}_{bps}bp", cost_bps=bps)
    print(f"  {name}: {results[name]['sharpe']} / @10bp={results[f'{name}_10bp']['sharpe']} / @20bp={results[f'{name}_20bp']['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# PART 4: Parameter sensitivity on best 6-factor
# ══════════════════════════════════════════════════════════════════
print("\n=== PART 4: 6-factor param sensitivity ===")
# Find the best 6-factor
best_6f_name = max(
    [(n, results[n]["sharpe"]) for n in [c[0] for c in configs] if "error" not in results[n]],
    key=lambda x: x[1]
)[0]
print(f"Best 6-factor: {best_6f_name} (Sharpe={results[best_6f_name]['sharpe']})")

# Get its weights
best_config = next(c for c in configs if c[0] == best_6f_name)
_, base_rw, base_sw, base_cw, base_aw, base_skw, base_vw = best_config
base_dict = {"rev": base_rw, "sec": base_sw, "carry": base_cw,
             "accel": base_aw, "skew": base_skw, "value": base_vw}

for param_name, param_val in base_dict.items():
    sharpes = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        w = dict(base_dict)
        w[param_name] = param_val * mult
        total = sum(w.values())
        w = {k: v/total for k, v in w.items()}  # renormalize
        sig = (w["rev"]*rev_ensemble + w["sec"]*sec_252 + w["carry"]*carry +
               w["accel"]*accel + w["skew"]*skew_63 + w["value"]*val) * vf
        key = f"sens_{best_6f_name}_{param_name}_{int(mult*100)}"
        results[key] = backtest(sig, key, cost_bps=10)
        if "error" not in results[key]:
            sharpes.append(results[key]["sharpe"])
    if sharpes:
        print(f"  {param_name}: range [{min(sharpes):.3f}, {max(sharpes):.3f}], spread={max(sharpes)-min(sharpes):.3f}")


# ══════════════════════════════════════════════════════════════════
# PART 5: Vol filter dampen sensitivity on best 6-factor
# ══════════════════════════════════════════════════════════════════
print("\n=== PART 5: VF dampen sensitivity ===")
for dampen in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    vf_test = vol_filter(returns, 63, dampen)
    sig = (base_rw*rev_ensemble + base_sw*sec_252 + base_cw*carry +
           base_aw*accel + base_skw*skew_63 + base_vw*val) * vf_test
    key = f"vf_dampen_{int(dampen*10)}"
    results[key] = backtest(sig, key, cost_bps=10)
    print(f"  dampen={dampen:.1f}: Sharpe@10bp={results[key].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*130)
print("BATTERY #7 — HEADLINE RESULTS (sorted by Sharpe @0bp)")
print("="*130)

# Filter to main experiments (not _Xbp variants or sensitivity)
main_keys = [k for k in results if "error" not in results[k]
             and not any(k.endswith(f"_{b}bp") for b in [5,10,15,20,30])
             and not k.startswith("sens_") and not k.startswith("vf_dampen")]
main = sorted([(k, results[k]) for k in main_keys], key=lambda x: x[1]["sharpe"], reverse=True)

print(f"\n{'Strategy':<30} {'Sharpe':>7} {'@10bp':>7} {'CAGR%':>7} {'MaxDD%':>7} {'IS':>6} {'OOS':>6} {'Roll3yMin':>9} {'3y%Pos':>6} {'WR%':>5}")
print("-"*110)
for name, m in main:
    k10 = f"{name}_10bp"
    s10 = results[k10]["sharpe"] if k10 in results and "error" not in results[k10] else None
    s10_str = f"{s10:.3f}" if s10 is not None else "  n/a"
    is_s = f"{m['is_sharpe']:.3f}" if m.get('is_sharpe') is not None else "  n/a"
    oos_s = f"{m['oos_sharpe']:.3f}" if m.get('oos_sharpe') is not None else "  n/a"
    r3m = f"{m['roll_3yr_min']:.3f}" if m.get('roll_3yr_min') is not None else "  n/a"
    r3p = f"{m['roll_3yr_pct_pos']:.1f}" if m.get('roll_3yr_pct_pos') is not None else " n/a"
    print(f"{name:<30} {m['sharpe']:>7.3f} {s10_str:>7} {m['cagr']:>7.2f} {m['max_dd']:>7.2f} {is_s:>6} {oos_s:>6} {r3m:>9} {r3p:>6} {m['win_rate']:>5.1f}")

# Cost curve for 6-factor candidates
print("\n── Cost Curves (6-factor candidates) ──")
c_names = [c[0] for c in configs]
print(f"{'Strategy':<20}" + "".join(f"{'@'+str(b)+'bp':>8}" for b in [0,5,10,15,20,30]))
for name in c_names:
    if "error" in results.get(name, {"error":1}):
        continue
    row = f"{name:<20}"
    for b in [0,5,10,15,20,30]:
        k = name if b == 0 else f"{name}_{b}bp"
        s = results[k]["sharpe"] if k in results and "error" not in results[k] else None
        row += f"{s:>8.3f}" if s is not None else f"{'n/a':>8}"
    print(row)

# Regime breakdown for top candidates
print("\n── Regime Sharpe (top candidates) ──")
top_5 = [name for name, _ in main[:5]]
regimes = sorted(set().union(*[set(results[k].get("regime_sharpe",{}).keys()) for k in top_5]))
print(f"{'Strategy':<30}" + "".join(f"{r:<15}" for r in regimes))
for k in top_5:
    row = f"{k:<30}"
    for r in regimes:
        rs = results[k].get("regime_sharpe",{}).get(r, None)
        row += f"{rs:<15}" if rs is not None else f"{'n/a':<15}"
    print(row)

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "Clean 6-factor stress test, accel windows, seasonal fix, regime debug",
    "baseline": "C10_baseline",
    "results": dict(results),
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_07_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_07_results.json")
