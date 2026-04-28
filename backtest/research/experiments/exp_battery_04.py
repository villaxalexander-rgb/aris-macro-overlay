"""
A.R.I.S Alpha Experiment Battery #4 — C8 Enhancement Candidates
Baseline: C8 Kitchen Sink VF (Sharpe ~0.92 IS, 0.75 OOS)

New dimensions:
  1. Multi-window reversal ensemble (blend 5d+10d+21d)
  2. Momentum crash hedge — cut reversal signal when asset in deep drawdown
  3. Correlation-adjusted weighting — shrink correlated clusters
  4. Vol-of-vol timing — scale gross exposure when vol regime shifting
  5. Adaptive factor weights — tilt based on trailing 126d factor performance
  6. Carry enhancement — use roll yield proxy (front-month vs 3-month return)
  7. Combined: best enhancements layered onto C8
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

# ══════════════════════════════════════════════════════════════════
# SIGNAL BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════

def reversal_xs(prices, window):
    ret = prices.pct_change(window)
    ranked = (-ret).rank(axis=1, pct=True)
    return (ranked - 0.5) * 2

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

    # Transaction costs
    if cost_bps > 0:
        turnover = held.diff().abs().sum(axis=1)
        port_ret -= turnover * (cost_bps / 10000)

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
    turnover = held.diff().abs().sum(axis=1).mean()

    # IS/OOS split (2007-2018 / 2019+)
    oos_mask = sr.index >= "2019-01-01"
    is_ret = sr[~oos_mask]
    oos_ret = sr[oos_mask]
    is_sharpe = oos_sharpe = None
    if len(is_ret) > 252:
        is_cum = (1+is_ret).prod() ** (252/len(is_ret)) - 1
        is_vol = is_ret.std() * np.sqrt(252)
        is_sharpe = round(is_cum / is_vol if is_vol > 0 else 0, 3)
    if len(oos_ret) > 252:
        oos_cum = (1+oos_ret).prod() ** (252/len(oos_ret)) - 1
        oos_vol = oos_ret.std() * np.sqrt(252)
        oos_sharpe = round(oos_cum / oos_vol if oos_vol > 0 else 0, 3)

    # Rolling 3yr Sharpe
    roll_sharpe = sr.rolling(756, min_periods=504).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    roll_min = round(roll_sharpe.min(), 3) if not roll_sharpe.dropna().empty else None
    roll_max = round(roll_sharpe.max(), 3) if not roll_sharpe.dropna().empty else None

    return {
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "turnover": round(turnover,4), "win_rate": round((sr>0).mean()*100,1),
        "years": round(yrs,1), "worst_yr": round(yr.min()*100,2),
        "best_yr": round(yr.max()*100,2), "pct_pos_yrs": round((yr>0).mean()*100,1),
        "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "roll_3yr_min": roll_min, "roll_3yr_max": roll_max,
    }


# ══════════════════════════════════════════════════════════════════
# BASELINE: C8 Kitchen Sink VF (reproduce from battery #3)
# ══════════════════════════════════════════════════════════════════
print("BASELINE: C8 Kitchen Sink VF...")
rev_10 = reversal_xs(prices, 10)
sec_252 = sector_rot(prices, 252)
ts_252 = tsmom(prices, 252)
val = value_5y(prices)
vf = vol_filter(returns, 63, 0.3)

c8_raw = 0.40 * rev_10 + 0.25 * sec_252 + 0.15 * ts_252 + 0.20 * val
c8 = c8_raw * vf
results["C8_BASELINE"] = backtest(c8, "C8_BASELINE")
results["C8_BASELINE_10bp"] = backtest(c8, "C8_BASELINE_10bp", cost_bps=10)
print(f"  C8 baseline: Sharpe={results['C8_BASELINE'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 1: Multi-window reversal ensemble
# Instead of single 10d reversal, blend multiple windows
# ══════════════════════════════════════════════════════════════════
print("\nEXP1: Multi-window reversal ensembles...")

rev_5 = reversal_xs(prices, 5)
rev_21 = reversal_xs(prices, 21)
rev_42 = reversal_xs(prices, 42)

for name, rev_blend in [
    ("REV_5_10", 0.5*rev_5 + 0.5*rev_10),
    ("REV_5_10_21", (rev_5 + rev_10 + rev_21) / 3),
    ("REV_5_10_21_42", (rev_5 + rev_10 + rev_21 + rev_42) / 4),
    ("REV_wt_short", 0.50*rev_5 + 0.35*rev_10 + 0.15*rev_21),  # tilt to short-term
    ("REV_wt_mid", 0.20*rev_5 + 0.50*rev_10 + 0.30*rev_21),    # tilt to mid
]:
    sig = 0.40*rev_blend + 0.25*sec_252 + 0.15*ts_252 + 0.20*val
    sig = sig * vf
    results[f"E1_{name}"] = backtest(sig, f"E1_{name}")
    results[f"E1_{name}_10bp"] = backtest(sig, f"E1_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E1_{name}'].get('sharpe','err')} / @10bp={results[f'E1_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 2: Momentum crash hedge
# Cut reversal signal when asset is in deep drawdown (>20% from peak)
# Reversal says "buy losers" but if it's a real crash, that's catching knives
# ══════════════════════════════════════════════════════════════════
print("\nEXP2: Momentum crash hedge...")

cum_ret = (1 + returns).cumprod()
drawdown = cum_ret / cum_ret.cummax() - 1

for dd_thresh, name in [(-0.15, "dd15"), (-0.20, "dd20"), (-0.30, "dd30"), (-0.40, "dd40")]:
    crash_mask = drawdown < dd_thresh  # True when asset in deep drawdown
    # Zero out the reversal "buy" signal for crashing assets
    rev_hedged = rev_10.copy()
    # Only kill the long side of reversal for crashing assets
    long_mask = rev_10 > 0
    rev_hedged[crash_mask & long_mask] = 0

    sig = 0.40*rev_hedged + 0.25*sec_252 + 0.15*ts_252 + 0.20*val
    sig = sig * vf
    results[f"E2_crash_{name}"] = backtest(sig, f"E2_crash_{name}")
    results[f"E2_crash_{name}_10bp"] = backtest(sig, f"E2_crash_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E2_crash_{name}'].get('sharpe','err')} / @10bp={results[f'E2_crash_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 3: Correlation-aware weighting
# Down-weight signals for highly correlated assets within same sector
# ══════════════════════════════════════════════════════════════════
print("\nEXP3: Correlation-adjusted weighting...")

# Rolling 63-day pairwise correlation, take mean per asset
corr_window = 63
avg_corr = returns.rolling(corr_window, min_periods=21).corr().groupby(level=0).apply(
    lambda x: x.mean(axis=1).mean()
) if False else None  # Too slow for full panel — use sector-level approximation instead

# Simpler: within-sector correlation penalty
for penalty, name in [(0.5, "corr_50"), (0.7, "corr_70"), (0.3, "corr_30")]:
    sector_count = {}
    for asset in prices.columns:
        for sector, assets in SECTORS.items():
            if asset in assets:
                n = len([a for a in assets if a in prices.columns])
                sector_count[asset] = n
                break
        if asset not in sector_count:
            sector_count[asset] = 1

    # Penalty: assets in larger sectors get scaled down
    max_n = max(sector_count.values())
    corr_mult = pd.Series({
        a: 1.0 - penalty * (n - 1) / max(max_n - 1, 1)
        for a, n in sector_count.items()
    })

    sig = c8_raw * vf * corr_mult
    results[f"E3_{name}"] = backtest(sig, f"E3_{name}")
    results[f"E3_{name}_10bp"] = backtest(sig, f"E3_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E3_{name}'].get('sharpe','err')} / @10bp={results[f'E3_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 4: Vol-of-vol timing
# Scale overall exposure: when vol is rising fast (vol-of-vol high),
# reduce exposure. Mean reversion less effective in vol spikes.
# ══════════════════════════════════════════════════════════════════
print("\nEXP4: Vol-of-vol timing...")

port_vol = returns.mean(axis=1).rolling(63).std() * np.sqrt(252)
vol_change = port_vol.pct_change(21)  # 1-month vol change

for scale_down, scale_up, name in [
    (0.5, 1.0, "vov_conservative"),
    (0.3, 1.0, "vov_aggressive"),
    (0.5, 1.3, "vov_both"),  # also scale UP when vol falling
    (0.0, 1.0, "vov_halt"),  # go flat when vol spiking
]:
    q75 = vol_change.rolling(252, min_periods=63).quantile(0.75)
    q25 = vol_change.rolling(252, min_periods=63).quantile(0.25)
    vol_rising = (vol_change > q75).reindex(prices.index, fill_value=False)
    vol_falling = (vol_change < q25).reindex(prices.index, fill_value=False)

    timing_mult = pd.Series(1.0, index=prices.index)
    timing_mult.loc[vol_rising] = scale_down
    timing_mult.loc[vol_falling] = scale_up

    sig = c8.multiply(timing_mult, axis=0)
    results[f"E4_{name}"] = backtest(sig, f"E4_{name}")
    results[f"E4_{name}_10bp"] = backtest(sig, f"E4_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E4_{name}'].get('sharpe','err')} / @10bp={results[f'E4_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 5: Adaptive factor weights
# Instead of fixed 40/25/15/20, tilt toward factors that have been
# working recently (trailing 126d IC or Sharpe contribution)
# ══════════════════════════════════════════════════════════════════
print("\nEXP5: Adaptive factor weights...")

# Compute trailing factor returns (not portfolio returns, just factor * returns)
factor_signals = {
    "reversal": rev_10 * vf,
    "sector_rot": sec_252 * vf,
    "tsmom": ts_252 * vf,
    "value": val * vf,
}

base_weights = {"reversal": 0.40, "sector_rot": 0.25, "tsmom": 0.15, "value": 0.20}

for adapt_speed, name in [(126, "adapt_126d"), (63, "adapt_63d"), (252, "adapt_252d")]:
    # Compute trailing Sharpe for each factor
    factor_rets = {}
    for fname, fsig in factor_signals.items():
        gross_f = fsig.abs().sum(axis=1).replace(0, 1)
        w_f = fsig.div(gross_f, axis=0)
        factor_rets[fname] = (w_f.shift(1) * returns).sum(axis=1)

    factor_ret_df = pd.DataFrame(factor_rets)
    trail_sharpe = factor_ret_df.rolling(adapt_speed, min_periods=adapt_speed//2).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    # Softmax to get adaptive weights (with temperature to prevent extreme concentration)
    temp = 1.0
    exp_sharpe = np.exp(trail_sharpe / temp)
    adaptive_w = exp_sharpe.div(exp_sharpe.sum(axis=1), axis=0)

    # Blend: 50% base weights + 50% adaptive
    blended_w = adaptive_w * 0.5
    for fname, bw in base_weights.items():
        blended_w[fname] = blended_w[fname] + 0.5 * bw

    # Reconstruct signal with adaptive weights
    adaptive_sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for fname, fsig in factor_signals.items():
        adaptive_sig = adaptive_sig + fsig.multiply(blended_w[fname], axis=0)

    results[f"E5_{name}"] = backtest(adaptive_sig, f"E5_{name}")
    results[f"E5_{name}_10bp"] = backtest(adaptive_sig, f"E5_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E5_{name}'].get('sharpe','err')} / @10bp={results[f'E5_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 6: Roll yield / carry proxy
# Use 21d vs 63d return differential as a proxy for roll yield
# (positive = backwardation = long bias)
# ══════════════════════════════════════════════════════════════════
print("\nEXP6: Roll yield carry proxy...")

ret_21 = prices.pct_change(21)
ret_63 = prices.pct_change(63)
carry_proxy = (ret_21 - ret_63 / 3)  # Normalize 63d to same scale
carry_xs = (carry_proxy.rank(axis=1, pct=True) - 0.5) * 2

# Replace sector_rot with carry, or add as 5th factor
for combo, name in [
    # Replace sector_rot with carry
    ({"reversal": 0.40, "carry": 0.25, "tsmom": 0.15, "value": 0.20}, "carry_replace_sec"),
    # Add carry as 5th factor
    ({"reversal": 0.35, "sector_rot": 0.20, "carry": 0.15, "tsmom": 0.10, "value": 0.20}, "carry_5factor"),
    # Heavy carry
    ({"reversal": 0.30, "carry": 0.30, "tsmom": 0.15, "value": 0.25}, "carry_heavy"),
]:
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    component_map = {
        "reversal": rev_10, "sector_rot": sec_252, "tsmom": ts_252,
        "value": val, "carry": carry_xs,
    }
    for fname, fw in combo.items():
        sig = sig + fw * component_map[fname]
    sig = sig * vf

    results[f"E6_{name}"] = backtest(sig, f"E6_{name}")
    results[f"E6_{name}_10bp"] = backtest(sig, f"E6_{name}_10bp", cost_bps=10)
    print(f"  {name}: Sharpe={results[f'E6_{name}'].get('sharpe','err')} / @10bp={results[f'E6_{name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# EXP 7: Best enhancements combined
# Take the winning ideas from E1-E6 and stack them
# ══════════════════════════════════════════════════════════════════
print("\nEXP7: Combined enhancements...")

# Multi-window reversal (best from E1)
rev_ensemble = (rev_5 + rev_10 + rev_21) / 3

# Crash hedge (moderate)
rev_crash_hedged = rev_ensemble.copy()
crash_mask_20 = drawdown < -0.20
long_mask_e = rev_ensemble > 0
rev_crash_hedged[crash_mask_20 & long_mask_e] = 0

for combo_name, rev_comp, extra_factors in [
    # Multi-rev + crash hedge + carry
    ("C9_multirev_crash_carry",
     rev_crash_hedged,
     {"sector_rot": 0.20, "carry": 0.10, "tsmom": 0.10, "value": 0.20}),
    # Multi-rev + carry (no crash hedge)
    ("C10_multirev_carry",
     rev_ensemble,
     {"sector_rot": 0.15, "carry": 0.15, "tsmom": 0.10, "value": 0.20}),
    # Kitchen Sink + carry (5 factors, single rev)
    ("C11_c8_plus_carry",
     rev_10,
     {"sector_rot": 0.20, "carry": 0.10, "tsmom": 0.10, "value": 0.20}),
    # Aggressive reversal + minimal extras
    ("C12_rev_heavy_carry",
     rev_ensemble,
     {"carry": 0.15, "tsmom": 0.10, "value": 0.15}),
]:
    rev_w = 1.0 - sum(extra_factors.values())  # remainder goes to reversal
    sig = rev_w * rev_comp
    component_map = {
        "sector_rot": sec_252, "tsmom": ts_252,
        "value": val, "carry": carry_xs,
    }
    for fname, fw in extra_factors.items():
        sig = sig + fw * component_map[fname]
    sig = sig * vf

    results[combo_name] = backtest(sig, combo_name)
    results[f"{combo_name}_10bp"] = backtest(sig, f"{combo_name}_10bp", cost_bps=10)
    print(f"  {combo_name}: Sharpe={results[combo_name].get('sharpe','err')} / @10bp={results[f'{combo_name}_10bp'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*110)
print("BATTERY #4 RESULTS — SORTED BY SHARPE (0bp cost)")
print("="*110)

valid = {k: v for k, v in results.items() if "error" not in v and not k.endswith("_10bp")}
ranked = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

print(f"\n{'Strategy':<40} {'Sharpe':>7} {'CAGR%':>7} {'Vol%':>6} {'MaxDD%':>7} {'Calmar':>7} {'IS':>6} {'OOS':>6} {'Roll3yMin':>9} {'T/O':>7}")
print("-"*110)
for name, m in ranked:
    is_s = f"{m['is_sharpe']:.3f}" if m.get('is_sharpe') is not None else "  n/a"
    oos_s = f"{m['oos_sharpe']:.3f}" if m.get('oos_sharpe') is not None else "  n/a"
    r3m = f"{m['roll_3yr_min']:.3f}" if m.get('roll_3yr_min') is not None else "  n/a"
    print(f"{name:<40} {m['sharpe']:>7.3f} {m['cagr']:>7.2f} {m['vol']:>6.1f} {m['max_dd']:>7.2f} {m['calmar']:>7.3f} {is_s:>6} {oos_s:>6} {r3m:>9} {m['turnover']:>7.4f}")

# Cost-adjusted comparison
print(f"\n{'Strategy':<40} {'Sharpe@0bp':>10} {'Sharpe@10bp':>11} {'Decay%':>7}")
print("-"*70)
for name, m in ranked:
    key_10 = f"{name}_10bp"
    if key_10 in results and "error" not in results[key_10]:
        s0 = m["sharpe"]
        s10 = results[key_10]["sharpe"]
        decay = round((1 - s10/s0)*100, 1) if s0 > 0 else 0
        print(f"{name:<40} {s0:>10.3f} {s10:>11.3f} {decay:>6.1}%")

# C8 baseline comparison
c8_sharpe = results.get("C8_BASELINE", {}).get("sharpe", 0)
better = [(n,m) for n,m in ranked if m["sharpe"] > c8_sharpe and n != "C8_BASELINE"]
print(f"\n── C8 Baseline Sharpe: {c8_sharpe:.3f} ──")
print(f"── Strategies that beat C8: {len(better)}/{len(valid)-1} ──")

if better:
    print("\n── TOP CANDIDATES (beat C8) ──")
    for i, (name, m) in enumerate(better[:10], 1):
        is_s = f"IS={m['is_sharpe']:.3f}" if m.get('is_sharpe') is not None else ""
        oos_s = f"OOS={m['oos_sharpe']:.3f}" if m.get('oos_sharpe') is not None else ""
        key_10 = f"{name}_10bp"
        s10 = results[key_10]["sharpe"] if key_10 in results and "error" not in results[key_10] else None
        cost_str = f"@10bp={s10:.3f}" if s10 is not None else ""
        print(f"  {i}. {name}")
        print(f"     Sharpe={m['sharpe']:.3f} {cost_str}, CAGR={m['cagr']:.2f}%, MaxDD={m['max_dd']:.2f}%, {is_s} {oos_s}")

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "C8 enhancement: multi-rev, crash hedge, correlation, vol-timing, adaptive, carry",
    "baseline": "C8_BASELINE",
    "results": dict(results),
    "ranking": [name for name, _ in ranked],
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_04_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_04_results.json")
