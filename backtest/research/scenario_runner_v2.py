#!/usr/bin/env python3
"""
scenario_runner_v2 — frozen-data, deterministic scenario sweep.

Uses price_cache.pkl (frozen snapshot of yfinance data from 2026-04-24).
Adds two research variants to decouple signal quality from lot-size pathology:

  - ideal_nav_eligibility: use a fixed $10M NAV for the per-contract cap only,
    real NAV still drives sizing. Tests whether NAV-dependent dropouts are hurting us.
  - fractional_contracts: no integer rounding of contracts. Tests whether tiny-NAV
    rounding errors are a factor.

Runs the baseline + all 8 previous scenarios + 3 ideal-NAV variants. Saves to
scenario_results_v2.json and a markdown summary.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import aris_backtest_v2 as bt  # noqa

CACHE = Path(__file__).parent / "price_cache.pkl"


def load_cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def simulate(prices, regime_monthly, *,
             regime_weight_overrides=None, bsv_weights=None,
             rebalance_freq="daily", gross_cap=None, deadband_pct=None,
             ideal_nav_eligibility=False, fractional_contracts=False):
    """
    Deterministic simulation against frozen price data.

    Parameters
    ----------
    ideal_nav_eligibility : bool
        If True, use fixed $10M NAV for per-contract cap check (but real NAV for sizing).
        This tests signal quality independent of lot-size pathology.
    fractional_contracts : bool
        If True, skip integer rounding. Measures raw signal effectiveness.
    """
    # Override regime weights (mutate module global, snapshot & restore)
    orig_regime = {k: dict(v) for k, v in bt.REGIME_WEIGHTS.items()}
    if regime_weight_overrides:
        for reg, secs in regime_weight_overrides.items():
            bt.REGIME_WEIGHTS[reg] = {**bt.REGIME_WEIGHTS[reg], **secs}

    # Override composite weights
    if bsv_weights:
        mom_w, carry_w, val_w, rev_w = bsv_weights

        def composite_fn(p):
            return (mom_w * bt.compute_momentum(p) +
                    carry_w * bt.compute_carry_proxy(p) +
                    val_w * bt.compute_value(p) +
                    rev_w * bt.compute_reversal(p))
        composite = composite_fn(prices)
    else:
        composite = bt.compute_bsv_composite(prices)

    config = bt.BacktestConfig(rebalance_freq=rebalance_freq)
    ELIGIBILITY_NAV = 10_000_000.0  # fixed reference NAV for per-contract cap

    sim_start = pd.Timestamp(config.start_date)
    sim_dates = prices.loc[sim_start:].index
    nav = config.initial_nav
    positions = {}
    snapshots = []
    cost_rate = config.transaction_cost_bps / 10000.0

    for i, date in enumerate(sim_dates):
        today_prices = prices.loc[date]

        # Mark-to-market
        if positions and i > 0:
            prev_prices = prices.loc[sim_dates[i - 1]]
            pnl = 0.0
            for a, exp in positions.items():
                if a in today_prices.index and a in prev_prices.index:
                    pt, pp = today_prices[a], prev_prices[a]
                    if pd.notna(pt) and pd.notna(pp) and pp != 0:
                        pnl += exp * (pt / pp - 1)
            nav += pnl

        # Regime
        if len(regime_monthly) > 0:
            vr = regime_monthly[regime_monthly.index <= date]
            current_regime = vr.iloc[-1] if len(vr) > 0 else "Goldilocks"
        else:
            current_regime = "Goldilocks"

        # Rebalance decision
        do_rebalance = False
        if rebalance_freq == "daily":
            do_rebalance = True
        elif rebalance_freq == "weekly" and date.weekday() == 0:
            do_rebalance = True
        elif rebalance_freq == "monthly" and (i == 0 or date.month != sim_dates[i - 1].month):
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

            if gross_cap is not None:
                g = sum(abs(v) for v in target_pcts.values())
                if g > gross_cap:
                    s = gross_cap / g
                    target_pcts = {k: v * s for k, v in target_pcts.items()}

            new_positions = {}
            eligibility_nav = ELIGIBILITY_NAV if ideal_nav_eligibility else nav
            for a, tgt in target_pcts.items():
                if a in today_prices.index and pd.notna(today_prices[a]):
                    price = float(today_prices[a])
                    mult = bt.get_multiplier(a, config.use_micros)
                    one_lot = price * mult
                    if one_lot > config.max_per_contract_pct * eligibility_nav or one_lot <= 0:
                        continue
                    notional_target = tgt * nav
                    if fractional_contracts:
                        exposure = notional_target  # exact, no rounding
                    else:
                        n = round(notional_target / one_lot)
                        exposure = n * one_lot
                    if exposure != 0:
                        new_positions[a] = exposure

            # Dead-band
            if deadband_pct is not None:
                filtered = dict(positions)
                for a, new_exp in new_positions.items():
                    old_exp = positions.get(a, 0.0)
                    if abs(new_exp - old_exp) / max(nav, 1) >= deadband_pct:
                        filtered[a] = new_exp
                for a in list(filtered.keys()):
                    if a not in new_positions and abs(filtered[a]) / max(nav, 1) >= deadband_pct:
                        filtered[a] = 0
                new_positions = {k: v for k, v in filtered.items() if v != 0}

            # Turnover + costs
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

    # Restore
    bt.REGIME_WEIGHTS.clear()
    bt.REGIME_WEIGHTS.update(orig_regime)

    metrics = bt.compute_metrics(snapshots, config)
    navs = np.array([s.nav for s in snapshots])
    rets = np.diff(navs) / navs[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) > 0 and np.std(rets) > 0:
        metrics["sharpe_arithmetic"] = (np.mean(rets) * 252 - 0.03) / (np.std(rets) * np.sqrt(252))
    else:
        metrics["sharpe_arithmetic"] = 0.0
    metrics["avg_n_positions"] = np.mean([s.n_positions for s in snapshots])
    return {"metrics": metrics, "snapshots": snapshots}


def fmt_row(name, m):
    return (f"  {name:<46s}  "
            f"CAGR={m['cagr']*100:+6.2f}%  "
            f"Vol={m['annual_volatility']*100:5.1f}%  "
            f"Sh={m['sharpe_arithmetic']:+.3f}  "
            f"DD={m['max_drawdown']*100:+5.1f}%  "
            f"Pos={m['avg_n_positions']:4.1f}  "
            f"Stag={m['regime_annualized_returns'].get('Stagflation',0)*100:+5.1f}%  "
            f"Gold={m['regime_annualized_returns'].get('Goldilocks',0)*100:+5.1f}%  "
            f"Refl={m['regime_annualized_returns'].get('Reflation',0)*100:+5.1f}%  "
            f"Defl={m['regime_annualized_returns'].get('Deflation',0)*100:+5.1f}%")


def main():
    data = load_cache()
    prices, regime = data["prices"], data["regime_monthly"]
    print(f"Frozen data: {prices.shape[1]} assets × {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"Regime obs: {len(regime)} months\n")

    scenarios = [
        # ─── original scenarios (reproduce v1) ───
        dict(name="0: baseline"),
        dict(name="1: Stag energy 0.8→1.2",
             regime_weight_overrides={"Stagflation": {"energy": 1.2}}),
        dict(name="2: Stag energy→1.4 + livestock→1.0",
             regime_weight_overrides={"Stagflation": {"energy": 1.4, "livestock": 1.0}}),
        dict(name="3: weekly rebal only",
             rebalance_freq="weekly"),
        dict(name="4: regime fix + weekly",
             regime_weight_overrides={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly"),
        dict(name="5: #4 + 100% gross cap",
             regime_weight_overrides={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0),
        dict(name="6: #5 + composite (0.50/0.15/0.20/0.15)",
             regime_weight_overrides={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0,
             bsv_weights=(0.50, 0.15, 0.20, 0.15)),
        dict(name="7: #5 + pure momentum",
             regime_weight_overrides={"Stagflation": {"energy": 1.4, "livestock": 1.0}},
             rebalance_freq="weekly", gross_cap=1.0,
             bsv_weights=(1.0, 0.0, 0.0, 0.0)),
        # ─── NEW: decouple NAV from tradeability ───
        dict(name="8: baseline + ideal-NAV eligibility",
             ideal_nav_eligibility=True),
        dict(name="9: baseline + ideal + fractional contracts",
             ideal_nav_eligibility=True, fractional_contracts=True),
        dict(name="10: pure-momentum + ideal + fractional + weekly",
             ideal_nav_eligibility=True, fractional_contracts=True,
             rebalance_freq="weekly",
             bsv_weights=(1.0, 0.0, 0.0, 0.0)),
        dict(name="11: pure-momentum + ideal + fractional + weekly + 100% cap",
             ideal_nav_eligibility=True, fractional_contracts=True,
             rebalance_freq="weekly", gross_cap=1.0,
             bsv_weights=(1.0, 0.0, 0.0, 0.0)),
        dict(name="12: ideal+fractional+weekly+cap  composite=(0.40/0.25/0.20/0.15)",
             ideal_nav_eligibility=True, fractional_contracts=True,
             rebalance_freq="weekly", gross_cap=1.0),
    ]

    results = []
    for i, s in enumerate(scenarios):
        name = s.pop("name")
        print(f"Running {i}: {name}")
        r = simulate(prices, regime, **s)
        r["name"] = name
        results.append(r)

    print("\n" + "=" * 190)
    print("FROZEN-DATA SCENARIO SWEEP  (15.3y, $1M initial, determinstic against price_cache.pkl)")
    print("=" * 190)
    for r in results:
        print(fmt_row(r["name"], r["metrics"]))
    print("=" * 190)

    out = []
    for r in results:
        m = r["metrics"]
        out.append({
            "name": r["name"],
            "cagr": m["cagr"],
            "vol": m["annual_volatility"],
            "sharpe_arith": m["sharpe_arithmetic"],
            "max_dd": m["max_drawdown"],
            "avg_positions": m["avg_n_positions"],
            "regime_returns": m["regime_annualized_returns"],
            "final_nav": m["final_nav"],
            "total_costs": m["total_costs"],
        })
    with open(Path(__file__).parent / "scenario_results_v2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved scenario_results_v2.json")


if__name__ == "__main__":
    main()
