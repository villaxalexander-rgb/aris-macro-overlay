"""
A.R.I.S Alpha Experiment Battery #2 — Deep Dive
Focus: Reversal is the standout signal (Sharpe 0.48).
Now: stress-test it, blend with TSMOM, test regime interaction,
and find the optimal composite for live trading.
"""
import pickle
import json
import numpy as np
import pandas as pd

with open("/sessions/upbeat-adoring-goldberg/mnt/Project: A.R.I.S Macro Overlay System/backtest/research/price_cache.pkl", "rb") as f:
    cache = pickle.load(f)

prices = cache["prices"].ffill()
regime = cache["regime_monthly"]
returns = prices.pct_change().dropna(how="all")
regime_daily = regime.reindex(prices.index, method="ffill")

results = {}

def backtest(signal_df, label, vol_target=0.12):
    sig = signal_df.reindex(returns.index).ffill().fillna(0)
    gross = sig.abs().sum(axis=1).replace(0, 1)
    weights = sig.div(gross, axis=0)

    # Weekly rebal
    is_rebal = pd.Series(sig.index.weekday == 2, index=sig.index)
    held = weights.copy()
    for i in range(1, len(held)):
        if not is_rebal.iloc[i]:
            held.iloc[i] = held.iloc[i-1]

    port_ret = (held.shift(1) * returns).sum(axis=1)
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
        "label": label, "cagr": round(cagr*100,2), "vol": round(vol*100,2),
        "sharpe": round(sharpe,3), "max_dd": round(maxdd*100,2),
        "calmar": round(cagr/abs(maxdd),3) if maxdd != 0 else 0,
        "turnover": round(turnover,4), "win_rate": round((sr>0).mean()*100,1),
        "years": round(yrs,1), "worst_yr": round(yr.min()*100,2),
        "best_yr": round(yr.max()*100,2), "pct_pos_yrs": round((yr>0).mean()*100,1),
        "regime_sharpe": regime_sharpe,
    }

def tsmom(prices, lb):
    return prices.pct_change(lb).apply(np.sign)

def reversal_xs(prices, window):
    ret = prices.pct_change(window)
    ranked = (-ret).rank(axis=1, pct=True)
    return (ranked - 0.5) * 2

def sector_rot(prices, lookback, sectors):
    sector_ret = {}
    for sector, assets in sectors.items():
        avail = [a for a in assets if a in prices.columns]
        if avail:
            sector_ret[sector] = prices[avail].pct_change(lookback).mean(axis=1)
    sector_df = pd.DataFrame(sector_ret)
    sector_rank = sector_df.rank(axis=1, pct=True)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for sector, assets in sectors.items():
        if sector in sector_rank.columns:
            for a in assets:
                if a in signal.columns:
                    signal[a] = (sector_rank[sector] - 0.5) * 2
    return signal

SECTORS = {
    "energy": ["CL", "BZ", "NG", "HO", "RB"],
    "precious": ["GC", "SI", "PL", "PA"],
    "industrial": ["HG"],
    "grains": ["ZC", "ZW", "ZS", "ZM", "ZL"],
    "softs": ["CT", "KC", "SB", "CC"],
    "livestock": ["LE", "GF", "HE"],
}

# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 1: Reversal window sweep (finer grid)
# ══════════════════════════════════════════════════════════════════
print("DD1: Reversal window fine sweep...")
for w in [3, 5, 7, 10, 14, 21, 30, 42, 63]:
    sig = reversal_xs(prices, w)
    label = f"REV_{w}d"
    results[label] = backtest(sig, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 2: Reversal + TSMOM blends (the money question)
# Can we combine the best reversal with TSMOM to get diversification?
# ══════════════════════════════════════════════════════════════════
print("\nDD2: Reversal + TSMOM blends...")

rev_10 = reversal_xs(prices, 10)
ts_252 = tsmom(prices, 252)
ts_blend = (tsmom(prices, 63) + tsmom(prices, 126) + tsmom(prices, 252)) / 3

for rev_w, ts_w, ts_type in [
    (0.7, 0.3, "252"), (0.6, 0.4, "252"), (0.5, 0.5, "252"),
    (0.4, 0.6, "252"), (0.3, 0.7, "252"),
    (0.7, 0.3, "blend"), (0.6, 0.4, "blend"), (0.5, 0.5, "blend"),
    (0.4, 0.6, "blend"), (0.3, 0.7, "blend"),
]:
    ts = ts_252 if ts_type == "252" else ts_blend
    signal = rev_w * rev_10 + ts_w * ts
    label = f"REV10_{int(rev_w*100)}_TSMOM{ts_type}_{int(ts_w*100)}"
    results[label] = backtest(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 3: Reversal + Sector Rotation blend
# ══════════════════════════════════════════════════════════════════
print("\nDD3: Reversal + Sector Rotation blends...")

sec_252 = sector_rot(prices, 252, SECTORS)

for rev_w, sec_w in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]:
    signal = rev_w * rev_10 + sec_w * sec_252
    label = f"REV10_{int(rev_w*100)}_SECROT_{int(sec_w*100)}"
    results[label] = backtest(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 4: Triple blend — Reversal + TSMOM + Sector Rotation
# ══════════════════════════════════════════════════════════════════
print("\nDD4: Triple blend (Rev + TSMOM + SecRot)...")

combos = [
    (0.5, 0.3, 0.2), (0.4, 0.3, 0.3), (0.4, 0.4, 0.2),
    (0.5, 0.2, 0.3), (0.6, 0.2, 0.2), (0.3, 0.4, 0.3),
]

for rw, tw, sw in combos:
    signal = rw * rev_10 + tw * ts_252 + sw * sec_252
    label = f"TRIPLE_R{int(rw*100)}_T{int(tw*100)}_S{int(sw*100)}"
    results[label] = backtest(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 5: Reversal with regime conditioning
# Does reversal work differently in different regimes?
# ══════════════════════════════════════════════════════════════════
print("\nDD5: Reversal regime analysis...")

rev_10_bt = backtest(rev_10, "REV_10d_regime_check")
print(f"  REV_10d regime Sharpes: {rev_10_bt.get('regime_sharpe', {})}")

# Try amplifying reversal in regimes where it works best
for rule_name, mults in [
    ("amp_expansion", {"Expansion": 1.5, "Slowdown": 0.5, "Stagflation": 0.8, "Recovery": 1.2}),
    ("amp_slowdown", {"Expansion": 0.8, "Slowdown": 1.5, "Stagflation": 1.2, "Recovery": 0.8}),
    ("amp_stagflation", {"Expansion": 0.8, "Slowdown": 0.8, "Stagflation": 1.5, "Recovery": 1.0}),
    ("flat_bad_regime", {"Expansion": 1.0, "Slowdown": 0.0, "Stagflation": 1.0, "Recovery": 1.0}),
]:
    mult = regime_daily.map(mults).fillna(1.0)
    signal = rev_10.multiply(mult, axis=0)
    label = f"REV10_REGIME_{rule_name}"
    results[label] = backtest(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 6: Reversal with vol filter — only trade reversal
# when vol is elevated (mean-reversion stronger in volatile markets)
# ══════════════════════════════════════════════════════════════════
print("\nDD6: Reversal + vol filter...")

rv_63 = returns.rolling(63).std() * np.sqrt(252)
rv_median = rv_63.expanding().median()

for mult_low, mult_high in [(0.3, 1.0), (0.0, 1.0), (0.5, 1.5), (0.0, 1.5)]:
    high_vol = rv_63 > rv_median
    vol_mult = high_vol.astype(float) * mult_high + (~high_vol).astype(float) * mult_low
    signal = rev_10 * vol_mult
    label = f"REV10_VOL_lo{int(mult_low*10)}_hi{int(mult_high*10)}"
    results[label] = backtest(signal, label)
    print(f"  {label}: Sharpe={results[label].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# DEEP DIVE 7: Compare vs current T6 hybrid (Phase A baseline)
# ══════════════════════════════════════════════════════════════════
print("\nDD7: Phase A T6 hybrid baseline for comparison...")

# Reproduce T6: 50% TSMOM(63,126,252) + 50% BSV
tsmom_ensemble = (tsmom(prices, 63) + tsmom(prices, 126) + tsmom(prices, 252)) / 3

# BSV: mom 0.4, carry 0.2, value 0.2, reversal 0.15
mom_252_xs = (prices.pct_change(252).rank(axis=1, pct=True) - 0.5) * 2
carry_21_xs = (prices.pct_change(21).rank(axis=1, pct=True) - 0.5) * 2
val_5y = prices.rolling(252*5, min_periods=252).mean()
val_dev = (val_5y - prices) / val_5y
val_xs = (val_dev.rank(axis=1, pct=True) - 0.5) * 2
rev_21_xs = reversal_xs(prices, 21)

bsv = 0.40 * mom_252_xs + 0.25 * carry_21_xs + 0.20 * val_xs + 0.15 * rev_21_xs
t6 = 0.5 * tsmom_ensemble + 0.5 * bsv

results["PHASE_A_T6_HYBRID"] = backtest(t6, "PHASE_A_T6_HYBRID")
print(f"  T6 hybrid: Sharpe={results['PHASE_A_T6_HYBRID'].get('sharpe','err')}")


# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print("DEEP DIVE RESULTS — SORTED BY SHARPE")
print("="*90)

valid = {k: v for k, v in results.items() if "error" not in v}
ranked = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

print(f"\n{'Strategy':<50} {'Sharpe':>7} {'CAGR%':>7} {'Vol%':>6} {'MaxDD%':>7} {'Calmar':>7} {'WR%':>5} {'T/O':>7}")
print("-"*100)
for name, m in ranked:
    print(f"{name:<50} {m['sharpe']:>7.3f} {m['cagr']:>7.2f} {m['vol']:>6.1f} {m['max_dd']:>7.2f} {m['calmar']:>7.3f} {m['win_rate']:>5.1f} {m['turnover']:>7.4f}")

# Highlight: how do the best new candidates compare to Phase A T6?
t6_sharpe = results.get("PHASE_A_T6_HYBRID", {}).get("sharpe", 0)
better = [(n,m) for n,m in ranked if m["sharpe"] > t6_sharpe]

print(f"\n── Phase A T6 Hybrid Sharpe: {t6_sharpe:.3f} ──")
print(f"── Strategies that beat T6: {len(better)}/{len(valid)} ──")
print()

print("── TOP 10 CANDIDATES ──")
for i, (name, m) in enumerate(ranked[:10], 1):
    regime_str = ", ".join(f"{k}={v}" for k, v in m.get("regime_sharpe", {}).items())
    print(f"  {i}. {name}")
    print(f"     Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2f}%, MaxDD={m['max_dd']:.2f}%, Calmar={m['calmar']:.3f}")
    if regime_str:
        print(f"     Regime: {regime_str}")

output = {
    "generated": pd.Timestamp.now().isoformat(),
    "focus": "Deep dive on reversal, blends, regime conditioning",
    "baseline": "PHASE_A_T6_HYBRID",
    "results": dict(results),
    "ranking": [name for name, _ in ranked],
}

with open("/sessions/upbeat-adoring-goldberg/exp_battery_02_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\nSaved to exp_battery_02_results.json")
