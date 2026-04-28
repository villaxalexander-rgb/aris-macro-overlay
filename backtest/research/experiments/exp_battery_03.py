"""
A.R.I.S Alpha Experiment Battery #3 — T6 Replacement Candidates
Stress-test the top signals from Battery #2 as production replacements.

Tests:
  1. Rolling 3-year Sharpe stability (is it consistent or one lucky period?)
  2. In-sample / out-of-sample split (2007-2018 train, 2019-2026 test)
  3. Transaction cost sensitivity (0bp, 5bp, 10bp, 20bp per trade)
  4. Drawdown profile & recovery time
  5. Final candidate comparison table
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

# ── Signal generators ──
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

def vol_filter(prices, returns, signal, zero_low_vol=True):
    """Apply vol filter: amplify signal in high-vol, reduce in low-vol."""
    rv_63 = returns.rolling(63).std() * np.sqrt(252)
    rv_median = rv_63.expanding(min_periods=126).median()
    high_vol = rv_63 > rv_median
    if zero_low_vol:
        return signal * high_vol.astype(float)
    else:
        return signal * (high_vol.astype(float) * 1.0 + (~high_vol).astype(float) * 0.3)


# ── Backtest engine with cost model ──
def backtest_full(signal_df, label, vol_target=0.12, cost_bps=0, prices_df=None):
    """Full backtest with transaction costs and detailed analytics."""
    sig = signal_df.reindex(returns.index).ffill().fillna(0)
    gross = sig.abs().sum(axis=1).replace(0, 1)
    weights = sig.div(gross, axis=0)

    # Weekly rebal (Wednesday)
    is_rebal = pd.Series(sig.index.weekday == 2, index=sig.index)
    held = weights.copy()
    for i in range(1, len(held)):
        if not is_rebal.iloc[i]:
            held.iloc[i] = held.iloc[i-1]

    port_ret = (held.shift(1) * returns).sum(axis=1)

    # Transaction costs
    turnover = held.diff().abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000)
    port_ret = port_ret - cost

    # Vol-target
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
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    yr = sr.groupby(sr.index.year).apply(lambda x: (1+x).prod()-1)
    avg_turnover = turnover.mean()

    # Drawdown recovery
    dd = cum / cum.cummax() - 1
    in_dd = dd < -0.01
    dd_runs = (~in_dd).cumsum()
    dd_lengths = in_dd.groupby(dd_runs).sum()
    avg_dd_days = dd_lengths[dd_lengths > 0].mean() if (dd_lengths > 0).any() else 0
    max_dd_days = dd_lengths.max()

    # Rolling 3-year Sharpe
    rolling_sr_3y = sr.rolling(756, min_periods=504).apply(
        lambda x: (((1+x).prod())**(252/len(x))-1) / (x.std()*np.sqrt(252)) if x.std()>0 else 0,
        raw=False
    )

    # Regime breakdown
    regime_sharpe = {}
    for r in regime_daily.dropna().unique():
        mask = regime_daily == r
        r_ret = sr[mask.reindex(sr.index, fill_value=False)]
        if len(r_ret) > 126:
            r_cagr = (1 + r_ret).prod() ** (252/len(r_ret)) - 1
            r_vol = r_ret.std() * np.sqrt(252)
            regime_sharpe[r] = round(r_cagr / r_vol if r_vol > 0 else 0, 3)

    return {
        "label": label, "cost_bps": cost_bps,
        "cagr": round(cagr*100, 2), "vol": round(vol*100, 2),
        "sharpe": round(sharpe, 3), "max_dd": round(maxdd*100, 2),
        "calmar": round(calmar, 3),
        "turnover": round(avg_turnover, 4),
        "win_rate": round((sr>0).mean()*100, 1),
        "years": round(yrs, 1),
        "worst_yr": round(yr.min()*100, 2),
        "best_yr": round(yr.max()*100, 2),
        "pct_pos_yrs": round((yr>0).mean()*100, 1),
        "avg_dd_days": round(avg_dd_days, 0),
        "max_dd_days": round(max_dd_days, 0),
        "rolling_3y_sharpe_min": round(rolling_sr_3y.min(), 3) if not rolling_sr_3y.isna().all() else None,
        "rolling_3y_sharpe_max": round(rolling_sr_3y.max(), 3) if not rolling_sr_3y.isna().all() else None,
        "rolling_3y_sharpe_median": round(rolling_sr_3y.median(), 3) if not rolling_sr_3y.isna().all() else None,
        "regime_sharpe": regime_sharpe,
        "yearly_returns": {int(k): round(v*100, 2) for k, v in yr.items()},
    }


# ══════════════════════════════════════════════════════════════════
# Define the 6 candidate strategies
# ══════════════════════════════════════════════════════════════════

def build_candidates():
    rev10 = reversal_xs(prices, 10)
    rev30 = reversal_xs(prices, 30)
    sec252 = sector_rot(prices, 252)
    ts252 = tsmom(prices, 252)
    ts_blend = (tsmom(prices, 63) + tsmom(prices, 126) + tsmom(prices, 252)) / 3

    # BSV composite (Phase A baseline ingredients)
    mom_xs = (prices.pct_change(252).rank(axis=1, pct=True) - 0.5) * 2
    carry_xs = (prices.pct_change(21).rank(axis=1, pct=True) - 0.5) * 2
    val_5y = prices.rolling(252*5, min_periods=252).mean()
    val_dev = (val_5y - prices) / val_5y
    val_xs = (val_dev.rank(axis=1, pct=True) - 0.5) * 2
    rev21_xs = reversal_xs(prices, 21)
    bsv = 0.40 * mom_xs + 0.25 * carry_xs + 0.20 * val_xs + 0.15 * rev21_xs
    t6 = 0.5 * ts_blend + 0.5 * bsv

    candidates = OrderedDict()

    # C0: Phase A baseline
    candidates["C0_T6_HYBRID"] = t6

    # C1: Pure vol-filtered reversal 10d
    candidates["C1_REV10_VOLFILT"] = vol_filter(prices, returns, rev10, zero_low_vol=True)

    # C2: Vol-filtered reversal with low-vol damping (not zeroed)
    candidates["C2_REV10_VOLDAMP"] = vol_filter(prices, returns, rev10, zero_low_vol=False)

    # C3: Reversal 10d + Sector Rotation 60/40
    candidates["C3_REV10_SECROT_60_40"] = 0.6 * rev10 + 0.4 * sec252

    # C4: Vol-filtered reversal + sector rotation
    rev10_vf = vol_filter(prices, returns, rev10, zero_low_vol=False)
    candidates["C4_REV10VF_SECROT_60_40"] = 0.6 * rev10_vf + 0.4 * sec252

    # C5: Triple blend (reversal + TSMOM252 + secrot)
    candidates["C5_TRIPLE_R60_T20_S20"] = 0.6 * rev10 + 0.2 * ts252 + 0.2 * sec252

    # C6: Reversal 30d (smoother, lower turnover)
    candidates["C6_REV30_VOLFILT"] = vol_filter(prices, returns, rev30, zero_low_vol=True)

    # C7: Rev10 + Rev30 blend (multi-horizon reversal)
    rev_blend = 0.6 * rev10 + 0.4 * rev30
    candidates["C7_REV_MULTIHORIZON"] = vol_filter(prices, returns, rev_blend, zero_low_vol=False)

    # C8: Kitchen sink — rev + secrot + tsmom252 + value, vol-filtered
    kitchen = 0.40 * rev10 + 0.25 * sec252 + 0.15 * ts252 + 0.20 * val_xs
    candidates["C8_KITCHEN_SINK_VF"] = vol_filter(prices, returns, kitchen, zero_low_vol=False)

    return candidates

candidates = build_candidates()


# ══════════════════════════════════════════════════════════════════
# TEST 1: Full-sample backtest at 0bp cost
# ══════════════════════════════════════════════════════════════════
print("="*90)
print("TEST 1: Full-sample backtest (0bp costs)")
print("="*90)
for name, sig in candidates.items():
    r = backtest_full(sig, name, cost_bps=0)
    results[f"{name}_0bp"] = r
    print(f"  {name:<40} Sharpe={r.get('sharpe','err'):>7} CAGR={r.get('cagr',''):>6}% MaxDD={r.get('max_dd',''):>7}%")


# ══════════════════════════════════════════════════════════════════
# TEST 2: Transaction cost sensitivity
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 2: Transaction cost sensitivity")
print("="*90)
print(f"\n{'Strategy':<40} {'0bp':>7} {'5bp':>7} {'10bp':>7} {'20bp':>7} {'T/O':>7}")
print("-"*75)

for name, sig in candidates.items():
    sharpes = {}
    for cost in [0, 5, 10, 20]:
        r = backtest_full(sig, f"{name}_{cost}bp", cost_bps=cost)
        sharpes[cost] = r.get("sharpe", "err")
        results[f"{name}_{cost}bp"] = r
    to = results[f"{name}_0bp"].get("turnover", 0)
    print(f"  {name:<40} {sharpes[0]:>7.3f} {sharpes[5]:>7.3f} {sharpes[10]:>7.3f} {sharpes[20]:>7.3f} {to:>7.4f}")


# ══════════════════════════════════════════════════════════════════
# TEST 3: In-sample / Out-of-sample split
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 3: In-sample (2007-2018) vs Out-of-sample (2019-2026)")
print("="*90)

split_date = "2019-01-01"
prices_is = prices[prices.index < split_date]
prices_oos = prices[prices.index >= split_date]
returns_is = returns[returns.index < split_date]
returns_oos = returns[returns.index >= split_date]

print(f"\n{'Strategy':<40} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'Delta':>7} {'Decay%':>7}")
print("-"*80)

for name, sig in candidates.items():
    # IS
    sig_is = sig[sig.index < split_date]
    r_is = backtest_full(sig_is, f"{name}_IS", cost_bps=5)

    # OOS
    sig_oos = sig[sig.index >= split_date]
    r_oos = backtest_full(sig_oos, f"{name}_OOS", cost_bps=5)

    is_s = r_is.get("sharpe", 0)
    oos_s = r_oos.get("sharpe", 0)
    delta = oos_s - is_s
    decay = ((is_s - oos_s) / abs(is_s) * 100) if is_s != 0 else 0

    results[f"{name}_IS"] = r_is
    results[f"{name}_OOS"] = r_oos

    print(f"  {name:<40} {is_s:>10.3f} {oos_s:>11.3f} {delta:>+7.3f} {decay:>7.1f}")


# ══════════════════════════════════════════════════════════════════
# TEST 4: Rolling 3-year Sharpe stability
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 4: Rolling 3-year Sharpe stability")
print("="*90)

print(f"\n{'Strategy':<40} {'Median':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
print("-"*75)

for name in candidates:
    r = results.get(f"{name}_0bp", {})
    med = r.get("rolling_3y_sharpe_median", "N/A")
    mn = r.get("rolling_3y_sharpe_min", "N/A")
    mx = r.get("rolling_3y_sharpe_max", "N/A")
    rng = mx - mn if isinstance(mx, (int,float)) and isinstance(mn, (int,float)) else "N/A"
    print(f"  {name:<40} {med:>8} {mn:>8} {mx:>8} {rng:>8}")


# ══════════════════════════════════════════════════════════════════
# TEST 5: Drawdown profile
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 5: Drawdown profile")
print("="*90)

print(f"\n{'Strategy':<40} {'MaxDD%':>8} {'AvgDD days':>11} {'MaxDD days':>11} {'Calmar':>8}")
print("-"*80)

for name in candidates:
    r = results.get(f"{name}_0bp", {})
    print(f"  {name:<40} {r.get('max_dd',''):>8} {r.get('avg_dd_days',''):>11} {r.get('max_dd_days',''):>11} {r.get('calmar',''):>8}")


# ══════════════════════════════════════════════════════════════════
# TEST 6: Year-by-year returns for top 3
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 6: Year-by-year returns (top 3 candidates at 5bp)")
print("="*90)

# Identify top 3 by OOS Sharpe at 5bp
oos_ranking = []
for name in candidates:
    r = results.get(f"{name}_OOS", {})
    if "sharpe" in r:
        oos_ranking.append((name, r["sharpe"]))
oos_ranking.sort(key=lambda x: x[1], reverse=True)

top3 = [n for n, _ in oos_ranking[:3]]
print(f"\nTop 3 by OOS Sharpe: {top3}")

for name in top3:
    r = results.get(f"{name}_5bp", {})
    yr = r.get("yearly_returns", {})
    print(f"\n  {name}:")
    for y in sorted(yr.keys()):
        bar = "+" * max(0, int(yr[y]/2)) + "-" * max(0, int(-yr[y]/2))
        print(f"    {y}: {yr[y]:>+7.2f}%  {bar}")


# ══════════════════════════════════════════════════════════════════
# FINAL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("FINAL SCORECARD")
print("="*90)

print(f"\n{'Strategy':<40} {'Full Sharpe':>11} {'OOS Sharpe':>11} {'Sharpe@10bp':>12} {'MaxDD%':>8} {'Calmar':>8}")
print("-"*95)

scorecard = []
for name in candidates:
    full = results.get(f"{name}_0bp", {})
    oos = results.get(f"{name}_OOS", {})
    at10 = results.get(f"{name}_10bp", {})
    row = {
        "name": name,
        "full_sharpe": full.get("sharpe", 0),
        "oos_sharpe": oos.get("sharpe", 0),
        "sharpe_10bp": at10.get("sharpe", 0),
        "max_dd": full.get("max_dd", 0),
        "calmar": full.get("calmar", 0),
    }
    scorecard.append(row)
    print(f"  {name:<40} {row['full_sharpe']:>11.3f} {row['oos_sharpe']:>11.3f} {row['sharpe_10bp']:>12.3f} {row['max_dd']:>8.2f} {row['calmar']:>8.3f}")

# Composite score: 40% OOS Sharpe + 30% Sharpe@10bp + 20% Calmar + 10% full Sharpe
print(f"\n── Composite score (40% OOS + 30% cost-adj + 20% Calmar + 10% full) ──")
for row in scorecard:
    row["composite"] = (
        0.40 * row["oos_sharpe"] +
        0.30 * row["sharpe_10bp"] +
        0.20 * row["calmar"] +
        0.10 * row["full_sharpe"]
    )

scorecard.sort(key=lambda x: x["composite"], reverse=True)
for i, row in enumerate(scorecard, 1):
    marker = " <<<" if i == 1 else ""
    print(f"  {i}. {row['name']:<40} composite={row['composite']:>7.3f}{marker}")

winner = scorecard[0]["name"]
print(f"\n★ RECOMMENDED T6 REPLACEMENT: {winner}")
print(f"  This strategy should be implemented as the new default alpha signal.")

# Save everything
output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "T6 replacement stress test — cost sensitivity, IS/OOS, rolling stability, drawdown",
    "split_date": split_date,
    "candidates": list(candidates.keys()),
    "results": {k: v for k, v in results.items()},
    "scorecard": scorecard,
    "recommendation": winner,
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_03_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_03_results.json")
