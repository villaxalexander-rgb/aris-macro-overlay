#!/usr/bin/env python3
"""
Scenario runner — apply candidate fixes to aris_backtest and compare.

Fetches price + macro data ONCE, caches it, then runs each scenario.
"""
import json
import pickle
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make the scratch copy importable
sys.path.insert(0, str(Path(__file__).parent))
import aris_backtest_v2 as bt  # noqa

CACHE = Path(__file__).parent / "price_cache.pkl"


def fetch_and_cache():
    """Fetch prices + macro once, cache to disk."""
    if CACHE.exists():
        print(f"Loading cached data from {CACHE}")
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    import yfinance as yf
    from datetime import timedelta

    config = bt.BacktestConfig()
    fetch_start = (pd.Timestamp(config.start_date) - timedelta(days=config.warmup_days + 60)).strftime("%Y-%m-%d")

    yf_tickers = {k: v[0] for k, v in bt.TICKERS.items()}
    ticker_str = " ".join(yf_tickers.values())

    print(f"Fetching {len(yf_tickers)} futures from {fetch_start} → {config.end_date}")
    raw = yf.download(ticker_str, start=fetch_start, end=config.end_date,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(yf_tickers)]
    else:
        prices = raw
    yf_to_canonical = {v: k for k, v in yf_tickers.items()}
    prices = prices.rename(columns=yf_to_canonical)
    available = [c for c in bt.TICKERS.keys() if c in prices.columns]
    prices = prices[available].sort_index().ffill().dropna(how="all")

    # Macro proxy
    macro_raw = yf.download("SPY TIP", start=fetch_start, end=config.end_date,
                            auto_adjust=True, progress=False)
    macro_close = macro_raw["Close"] if isinstance(macro_raw.columns, pd.MultiIndex) else macro_raw
    ism = macro_close["SPY"].pct_change(126).dropna() * 100 + 50 if "SPY" in macro_close.columns else pd.Series(dtype=float)
    cpi = macro_close["TIP"].pct_change(126).dropna() * 100 + 2 if "TIP" in macro_close.columns else pd.Series(dtype=float)

    regime_monthly = bt.classify_regime_series(ism, cpi) if len(ism) > 0 and len(cpi) > 0 else pd.Series(dtype=str)

    payload = {"prices": prices, "regime_monthly": regime_monthly}
    with open(CACHE, "wb") as f:
        pickle.dump(payload, f)
    print(f"Cached to {CACHE}")
    return payload


def run_scenario(name, *, regime_weights=None, bsv_weights=None,
                 rebalance_freq="daily", gross_cap=None, deadband_pct=None,
                 data=None):
    """Run one backtest scenario with optional overrides."""
    # Snapshot & override module-level globals
    orig_regime = bt.REGIME_WEIGHTS.copy()
    orig_bsv = None

    if regime_weights:
        for reg, secs in regime_weights.items():
            bt.REGIME_WEIGHTS[reg] = {**bt.REGIME_WEIGHTS[reg], **secs}

    if bsv_weights:
        orig_bsv = bt.compute_bsv_composite
        mom_w, carry_w, val_w, rev_w = bsv_weights

        def new_composite(prices):
            momentum = bt.compute_momentum(prices)
            carry = bt.compute_carry_proxy(prices)
            value = bt.compute_value(prices)
            reversal = bt.compute_reversal(prices)
            return mom_w*momentum + carry_w*carry + val_w*value + rev_w*reversal
        bt.compute_bsv_composite = new_composite

    # Run via a lean inline simulation using cached data
    result = _simulate(data["prices"], data["regime_monthly"],
                       rebalance_freq=rebalance_freq,
                       gross_cap=gross_cap, deadband_pct=deadband_pct)

    # Restore
    bt.REGIME_WEIGHTS.clear()
    bt.REGIME_WEIGHTS.update(orig_regime)
    if orig_bsv:
        bt.compute_bsv_composite = orig_bsv

    result["name"] = name
    return result


def _simulate(prices, regime_monthly, *, rebalance_freq="daily",
              gross_cap=None, deadband_pct=None):
    """Lean sim — mirrors bt.run_backtest but uses precomputed data."""
    config = bt.BacktestConfig(rebalance_freq=rebalance_freq)
    composite = bt.compute_bsv_composite(prices)

    sim_start = pd.Timestamp(config.start_date)
    sim_dates = prices.loc[sim_start:].index
    nav = config.initial_nav
    positions = {}
    snapshots = []
    cost_rate = config.transaction_cost_bps / 10000.0

    for i, date in enumerate(sim_dates):
        today_prices = prices.loc[date]

        if positions and i > 0:
            prev_prices = prices.loc[sim_dates[i-1]]
            pnl = 0.0
            for a, exp in positions.items():
                if a in today_prices.index and a in prev_prices.index:
                    pt, pp = today_prices[a], prev_prices[a]
                    if pd.notna(pt) and pd.notna(pp) and pp != 0:
                        pnl += exp * (pt/pp - 1)
            nav += pnl

        if len(regime_monthly) > 0:
            vr = regime_monthly[regime_monthly.index <= date]
            current_regime = vr.iloc[-1] if len(vr) > 0 else "Goldilocks"
        else:
            current_regime = "Goldilocks"

        do_rebalance = False
        if rebalance_freq == "daily":
            do_rebalance = True
        elif rebalance_freq == "weekly" and date.weekday() == 0:
            do_rebalance = True
        elif rebalance_freq == "monthly" and (i == 0 or date.month != sim_dates[i-1].month):
            do_rebalance = True

        turnover = 0.0
        costs = 0.0

        if do_rebalance and date in composite.index:
            signals = composite.loc[date].dropna()
            target_pcts = {}
            rw = bt.REGIME_WEIGHTS.get(current_regime, bt.REGIME_WEIGHTS["Goldilocks"])
            for a in signals.index:
                sector = bt.SECTOR_MAP.get(a, "other")
                target_pcts[a] = float(signals[a]) * rw.get(sector, 1.0)

            target_pcts, _ = bt.apply_sector_rescale(target_pcts)

            # NEW: gross exposure cap at portfolio level
            if gross_cap is not None:
                gross = sum(abs(v) for v in target_pcts.values())
                if gross > gross_cap:
                    scale = gross_cap / gross
                    target_pcts = {k: v*scale for k, v in target_pcts.items()}

            new_positions = {}
            for a, tgt in target_pcts.items():
                if a in today_prices.index and pd.notna(today_prices[a]):
                    price = float(today_prices[a])
                    mult = bt.get_multiplier(a, config.use_micros)
                    one_lot = price * mult
                    if one_lot > config.max_per_contract_pct * nav or one_lot <= 0:
                        continue
                    notional_target = tgt * nav
                    n = round(notional_target / one_lot)
                    if n != 0:
                        new_positions[a] = n * one_lot

            # NEW: dead-band — skip leg trade if |delta| < deadband_pct of NAV
            if deadband_pct is not None:
                filtered = dict(positions)  # keep old positions
                for a, new_exp in new_positions.items():
                    old_exp = positions.get(a, 0.0)
                    if abs(new_exp - old_exp) / nav >= deadband_pct:
                        filtered[a] = new_exp
                # also drop assets no longer in new_positions if delta big enough
                for a in list(filtered.keys()):
                    if a not in new_positions:
                        if abs(filtered[a]) / nav >= deadband_pct:
                            filtered[a] = 0
                new_positions = {k: v for k, v in filtered.items() if v != 0}

            # Turnover
            all_a = set(list(positions.keys()) + list(new_positions.keys()))
            for a in all_a:
                turnover += abs(new_positions.get(a, 0.0) - positions.get(a, 0.0))
            costs = turnover * cost_rate
            nav -= costs
            positions = new_positions

        gross = sum(abs(v) for v in positions.values())
        net = sum(positions.values())
        snapshots.append(bt.DailySnapshot(
            date=date, nav=nav, gross_exposure=gross, net_exposure=net,
            regime=current_regime,
            n_positions=len([v for v in positions.values() if v != 0]),
            turnover=turnover, costs=costs, daily_return=0.0,
        ))

    metrics = bt.compute_metrics(snapshots, config)
    # Also compute arithmetic-mean Sharpe for comparison
    navs = np.array([s.nav for s in snapshots])
    rets = np.diff(navs) / navs[:-1]
    arith_sharpe = (np.mean(rets)*252 - 0.03) / (np.std(rets)*np.sqrt(252)) if np.std(rets) > 0 else 0
    metrics["sharpe_arithmetic"] = arith_sharpe
    return {"metrics": metrics, "snapshots": snapshots}


def fmt_row(r):
    m = r["metrics"]
    return (f"  {r['name']:<30s}  "
            f"CAGR={m['cagr']*100:+5.1f}%  "
            f"Vol={m['annual_volatility']*100:4.1f}%  "
            f"SharpeArith={m['sharpe_arithmetic']:+.3f}  "
            f"MaxDD={m['max_drawdown']*100:+5.1f}%  "
            f"Costs=${m['total_costs']/1000:.0f}k  "
            f"Stag={m['regime_annualized_returns'].get('Stagflation',0)*100:+5.1f}%")


def main():
    data = fetch_and_cache()

    scenarios = [
        # Baseline
        dict(name="0: baseline (current)"),
        # Regime fix alone
        dict(name="1: Stag energy 0.8→1.2",
             regime_weights={"Stagflation": {"energy": 1.2}}),
        dict(name="2: Stag energy 0.8→1.4 + livestock 0.7→1.0",
             regime_weights={"Stagflation": {"energy": 1.4, "livestock": 1.0}}),
        # Cost fix alone
        dict(name="3: weekly rebalance only",
             rebalance_freq="weekly"),
        # Combined
        dict(name="4: regime fix + weekly",
             regime_weights={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly"),
        # + gross cap
        dict(name="5: #4 + 100% gross cap",
             regime_weights={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0),
        # + composite reweight
        dict(name="6: #5 + composite (0.50/0.15/0.20/0.15)",
             regime_weights={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0,
             bsv_weights=(0.50, 0.15, 0.20, 0.15)),
        # Pure momentum strip-down
        dict(name="7: #5 + pure momentum only (1.0/0/0/0)",
             regime_weights={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0,
             bsv_weights=(1.0, 0.0, 0.0, 0.0)),
    ]

    results = []
    for i, s in enumerate(scenarios):
        print(f"\nScenario {i}: {s['name']}")
        r = run_scenario(**{k: v for k, v in s.items() if k != "name"}, name=s["name"], data=data)
        results.append(r)

    print("\n" + "="*140)
    print("SCENARIO COMPARISON (15-year backtest, $1M initial)")
    print("="*140)
    for r in results:
        print(fmt_row(r))
    print("="*140)

    # Save summary
    summary = []
    for r in results:
        m = r["metrics"]
        summary.append({
            "name": r["name"],
            "cagr": m["cagr"],
            "vol": m["annual_volatility"],
            "sharpe_arith": m["sharpe_arithmetic"],
            "sharpe_cagr": m["sharpe_ratio"],
            "max_dd": m["max_drawdown"],
            "total_costs": m["total_costs"],
            "regime_returns": m["regime_annualized_returns"],
        })
    with open(Path(__file__).parent / "scenario_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to scenario_results.json")


if __name__ == "__main__":
    main()
