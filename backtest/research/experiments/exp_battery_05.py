"""
A.R.I.S Alpha Experiment Battery #5 — Stress Test Top B4 Candidates
Candidates from Battery #4 that beat C8 (Sharpe 0.915):
  C10: multi-window reversal ensemble + carry + sec_rot + tsmom + value + VF  (1.068)
  carry_5factor: C8 + carry as 5th factor  (1.040)
  corr_70: C8 + 70% sector-size correlation penalty  (1.033)
  adapt_126d: C8 with 50% adaptive factor weighting (126d trailing Sharpe)  (1.008)

Tests:
  1. Parameter sensitivity (±20% on each weight, ±50% on key windows)
  2. Rolling 2-year and 3-year Sharpe stability & worst windows
  3. Drawdown profile: max DD duration, recovery time, underwater curve
  4. Regime breakdown by macro quadrant
  5. Cost sensitivity: 0/5/10/15/20/30 bps
  6. Yearly return table
  7. Final recommendation
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

# ── Signal building blocks ──
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
    raw = ret_21 - ret_63 / 3
    return (raw.rank(axis=1, pct=True) - 0.5) * 2

def corr_penalty(prices, penalty=0.7):
    sector_count = {}
    for asset in prices.columns:
        for sector, assets in SECTORS.items():
            if asset in assets:
                n = len([a for a in assets if a in prices.columns])
                sector_count[asset] = n
                break
        if asset not in sector_count:
            sector_count[asset] = 1
    max_n = max(sector_count.values())
    return pd.Series({
        a: 1.0 - penalty * (n - 1) / max(max_n - 1, 1)
        for a, n in sector_count.items()
    })


def full_backtest(signal_df, label, vol_target=0.12, cost_bps=0):
    """Extended backtest with regime breakdown, yearly table, drawdown analysis."""
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
    turnover_avg = held.diff().abs().sum(axis=1).mean()

    # Drawdown analysis
    dd = cum / cum.cummax() - 1
    underwater = dd < -0.01  # in drawdown
    dd_periods = underwater.astype(int).groupby((~underwater).astype(int).cumsum())
    max_dd_duration = 0
    for _, grp in dd_periods:
        if len(grp) > 0 and grp.iloc[0] == 1:
            max_dd_duration = max(max_dd_duration, len(grp))

    # Regime breakdown
    regime_stats = {}
    for r in regime_daily.dropna().unique():
        mask = regime_daily == r
        r_ret = sr[mask.reindex(sr.index, fill_value=False)]
        if len(r_ret) > 126:
            r_cagr = (1 + r_ret).prod() ** (252/len(r_ret)) - 1
            r_vol = r_ret.std() * np.sqrt(252)
            regime_stats[r] = {
                "sharpe": round(r_cagr / r_vol if r_vol > 0 else 0, 3),
                "cagr": round(r_cagr*100, 2),
                "days": len(r_ret),
            }

    # Rolling Sharpe
    for w_name, w_days in [("2yr", 504), ("3yr", 756)]:
        rs = sr.rolling(w_days, min_periods=w_days//2).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        regime_stats[f"roll_{w_name}_min"] = round(rs.min(), 3) if not rs.dropna().empty else None
        regime_stats[f"roll_{w_name}_max"] = round(rs.max(), 3) if not rs.dropna().empty else None
        regime_stats[f"roll_{w_name}_pct_pos"] = round((rs > 0).mean()*100, 1) if not rs.dropna().empty else None

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

    return {
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "turnover": round(turnover_avg,4),
        "win_rate": round((sr>0).mean()*100,1),
        "max_dd_duration_days": max_dd_duration,
        "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "regime": regime_stats,
        "yearly": {str(y): round(v*100,2) for y, v in yr.items()},
    }


# ── Pre-compute signals ──
print("Pre-computing signal components...")
rev_5 = reversal_xs(prices, 5)
rev_10 = reversal_xs(prices, 10)
rev_21 = reversal_xs(prices, 21)
sec_252 = sector_rot(prices, 252)
ts_252 = tsmom(prices, 252)
val = value_5y(prices)
vf = vol_filter(returns, 63, 0.3)
carry = carry_proxy(prices)
corr_mult = corr_penalty(prices, 0.7)

rev_ensemble = (rev_5 + rev_10 + rev_21) / 3


# ══════════════════════════════════════════════════════════════════
# CANDIDATE BUILDERS (functions so we can perturb parameters)
# ══════════════════════════════════════════════════════════════════

def build_c10(rev_w=0.40, sec_w=0.15, carry_w=0.15, ts_w=0.10, val_w=0.20):
    raw = rev_w*rev_ensemble + sec_w*sec_252 + carry_w*carry + ts_w*ts_252 + val_w*val
    return raw * vf

def build_carry5f(rev_w=0.35, sec_w=0.20, carry_w=0.15, ts_w=0.10, val_w=0.20):
    raw = rev_w*rev_10 + sec_w*sec_252 + carry_w*carry + ts_w*ts_252 + val_w*val
    return raw * vf

def build_corr70():
    raw = 0.40*rev_10 + 0.25*sec_252 + 0.15*ts_252 + 0.20*val
    return raw * vf * corr_mult

def build_c8():
    raw = 0.40*rev_10 + 0.25*sec_252 + 0.15*ts_252 + 0.20*val
    return raw * vf


# ══════════════════════════════════════════════════════════════════
# TEST 1: Full backtest at multiple cost levels
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 1: Cost sensitivity ===")
candidates = {
    "C8_baseline": build_c8(),
    "C10_multirev_carry": build_c10(),
    "carry_5factor": build_carry5f(),
    "corr_70": build_corr70(),
}

for name, sig in candidates.items():
    for bps in [0, 5, 10, 15, 20, 30]:
        key = f"{name}_{bps}bp"
        results[key] = full_backtest(sig, key, cost_bps=bps)
    s0 = results[f"{name}_0bp"]["sharpe"]
    s10 = results[f"{name}_10bp"]["sharpe"]
    s20 = results[f"{name}_20bp"]["sharpe"]
    s30 = results[f"{name}_30bp"]["sharpe"]
    print(f"  {name:<25} 0bp={s0:.3f}  10bp={s10:.3f}  20bp={s20:.3f}  30bp={s30:.3f}")


# ══════════════════════════════════════════════════════════════════
# TEST 2: Parameter sensitivity for C10 (the winner)
# Perturb each weight by ±20% from default
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 2: C10 parameter sensitivity ===")
base_w = {"rev_w": 0.40, "sec_w": 0.15, "carry_w": 0.15, "ts_w": 0.10, "val_w": 0.20}
for param in base_w:
    for mult in [0.6, 0.8, 1.0, 1.2, 1.5]:
        w = dict(base_w)
        w[param] = base_w[param] * mult
        # Renormalize
        total = sum(w.values())
        w = {k: v/total for k, v in w.items()}
        sig = build_c10(**w)
        key = f"C10_sens_{param}_{int(mult*100)}"
        results[key] = full_backtest(sig, key, cost_bps=10)
    # Report range
    sharpes = [results[f"C10_sens_{param}_{int(m*100)}"]["sharpe"]
               for m in [0.6, 0.8, 1.0, 1.2, 1.5]
               if "error" not in results.get(f"C10_sens_{param}_{int(m*100)}", {"error":1})]
    if sharpes:
        print(f"  {param}: Sharpe@10bp range [{min(sharpes):.3f}, {max(sharpes):.3f}]  spread={max(sharpes)-min(sharpes):.3f}")


# ══════════════════════════════════════════════════════════════════
# TEST 3: Parameter sensitivity for carry_5factor
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 3: carry_5factor parameter sensitivity ===")
base_w2 = {"rev_w": 0.35, "sec_w": 0.20, "carry_w": 0.15, "ts_w": 0.10, "val_w": 0.20}
for param in base_w2:
    for mult in [0.6, 0.8, 1.0, 1.2, 1.5]:
        w = dict(base_w2)
        w[param] = base_w2[param] * mult
        total = sum(w.values())
        w = {k: v/total for k, v in w.items()}
        sig = build_carry5f(**w)
        key = f"CF5_sens_{param}_{int(mult*100)}"
        results[key] = full_backtest(sig, key, cost_bps=10)
    sharpes = [results[f"CF5_sens_{param}_{int(m*100)}"]["sharpe"]
               for m in [0.6, 0.8, 1.0, 1.2, 1.5]
               if "error" not in results.get(f"CF5_sens_{param}_{int(m*100)}", {"error":1})]
    if sharpes:
        print(f"  {param}: Sharpe@10bp range [{min(sharpes):.3f}, {max(sharpes):.3f}]  spread={max(sharpes)-min(sharpes):.3f}")


# ══════════════════════════════════════════════════════════════════
# TEST 4: Vol filter dampen sensitivity
# ══════════════════════════════════════════════════════════════════
print("\n=== TEST 4: Vol filter dampen sensitivity ===")
for dampen in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    vf_test = vol_filter(returns, 63, dampen)
    raw = 0.40*rev_ensemble + 0.15*sec_252 + 0.15*carry + 0.10*ts_252 + 0.20*val
    sig = raw * vf_test
    key = f"C10_vf_dampen_{int(dampen*10)}"
    results[key] = full_backtest(sig, key, cost_bps=10)
    print(f"  dampen={dampen:.1f}: Sharpe@10bp={results[key].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*120)
print("BATTERY #5 — HEADLINE COMPARISON (10bp costs)")
print("="*120)

headline_keys = [f"{n}_10bp" for n in ["C8_baseline", "C10_multirev_carry", "carry_5factor", "corr_70"]]
print(f"\n{'Strategy':<30} {'Sharpe':>7} {'CAGR%':>7} {'Vol%':>6} {'MaxDD%':>7} {'Calmar':>7} {'IS':>6} {'OOS':>6} {'MaxDD_Days':>10} {'WR%':>5}")
print("-"*100)
for key in headline_keys:
    m = results[key]
    if "error" in m:
        continue
    is_s = f"{m['is_sharpe']:.3f}" if m.get('is_sharpe') is not None else "  n/a"
    oos_s = f"{m['oos_sharpe']:.3f}" if m.get('oos_sharpe') is not None else "  n/a"
    print(f"{key:<30} {m['sharpe']:>7.3f} {m['cagr']:>7.2f} {m['vol']:>6.1f} {m['max_dd']:>7.2f} {m['calmar']:>7.3f} {is_s:>6} {oos_s:>6} {m.get('max_dd_duration_days','?'):>10} {m['win_rate']:>5.1f}")

# Yearly return table
print("\n── Yearly Returns (%) ──")
yearly_years = sorted(set().union(*[set(results[k].get("yearly",{}).keys()) for k in headline_keys if "error" not in results[k]]))
header = f"{'Year':<6}" + "".join(f"{k.replace('_10bp',''):<25}" for k in headline_keys)
print(header)
print("-" * len(header))
for y in yearly_years:
    row = f"{y:<6}"
    for k in headline_keys:
        val_yr = results[k].get("yearly",{}).get(y, None)
        row += f"{val_yr:>8.2f}                 " if val_yr is not None else f"{'n/a':>8}                 "
    print(row)

# Regime breakdown
print("\n── Regime Breakdown (Sharpe) ──")
regimes_seen = set()
for k in headline_keys:
    if "error" not in results[k]:
        for r in results[k].get("regime", {}):
            if not r.startswith("roll_"):
                regimes_seen.add(r)
regimes_seen = sorted(regimes_seen)
print(f"{'Strategy':<30}" + "".join(f"{r:<15}" for r in regimes_seen))
for k in headline_keys:
    if "error" in results[k]:
        continue
    row = f"{k:<30}"
    for r in regimes_seen:
        rs = results[k].get("regime",{}).get(r,{})
        if isinstance(rs, dict):
            row += f"{rs.get('sharpe','n/a'):<15}"
        else:
            row += f"{'n/a':<15}"
    print(row)

# Rolling stability
print("\n── Rolling Sharpe Stability ──")
print(f"{'Strategy':<30} {'2yr_min':>8} {'2yr_max':>8} {'2yr_%pos':>8} {'3yr_min':>8} {'3yr_max':>8} {'3yr_%pos':>8}")
for k in headline_keys:
    if "error" in results[k]:
        continue
    rg = results[k].get("regime", {})
    print(f"{k:<30} {rg.get('roll_2yr_min','n/a'):>8} {rg.get('roll_2yr_max','n/a'):>8} {rg.get('roll_2yr_pct_pos','n/a'):>8} {rg.get('roll_3yr_min','n/a'):>8} {rg.get('roll_3yr_max','n/a'):>8} {rg.get('roll_3yr_pct_pos','n/a'):>8}")

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "Stress test top B4 candidates: cost sensitivity, param sensitivity, regime, stability",
    "candidates": ["C10_multirev_carry", "carry_5factor", "corr_70", "C8_baseline"],
    "results": {k: v for k, v in results.items()},
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_05_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_05_results.json")
