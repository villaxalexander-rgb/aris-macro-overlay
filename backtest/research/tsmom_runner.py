#!/usr/bin/env python3
"""
tsmom_runner — Time-Series Momentum ensemble (Moskowitz/Ooi/Pedersen 2012)
evaluated on the same frozen price_cache.pkl as the BSV sweep.

TSMOM core mechanics:
  - For each asset INDEPENDENTLY, compute past return over a lookback L
  - Signal = sign(return_L) — long if past return > 0, short if < 0
  - Optionally continuous: signal = tanh(return / k*vol) to avoid full-flip binary
  - Size each leg via volatility targeting:
        weight_i = (target_vol_per_asset / vol_i) * signal_i
    where target_vol_per_asset = portfolio_vol_target / sqrt(N_assets)
  - Ensemble across lookbacks (1M, 3M, 6M, 12M) by averaging signals

This is NOT cross-sectional — it does not require ranks across assets.
Every asset makes its own long/short/flat decision.

Evaluates 6 variants + best BSV reference (scenario 12).
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


# ─── TSMOM signal primitives ─────────────────────────────────

def ts_return(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Past window-day return, no cross-sectional normalization."""
    return prices.pct_change(window)


def ts_sign(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Binary TSMOM — +1 / 0 / -1 based on sign of past return."""
    r = prices.pct_change(window)
    return np.sign(r).fillna(0)


def ts_continuous(prices: pd.DataFrame, window: int,
                  vol_window: int = 63, k: float = 2.0) -> pd.DataFrame:
    """
    Continuous TSMOM — signal in [-1, +1] via tanh(return / k*vol).
    Smoother than binary sign (avoids full flips around zero).
    """
    r = prices.pct_change(window)
    daily = prices.pct_change()
    vol = daily.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(window)
    z = r / (k * vol)
    return np.tanh(z).fillna(0)


def tsmom_ensemble(prices: pd.DataFrame,
                   lookbacks=(21, 63, 126, 252),
                   weights=None,
                   continuous=True,
                   vol_window=63) -> pd.DataFrame:
    """Average TSMOM signal across multiple lookbacks."""
    if weights is None:
        weights = [1.0 / len(lookbacks)] * len(lookbacks)

    components = []
    for L, w in zip(lookbacks, weights):
        if continuous:
            components.append(w * ts_continuous(prices, L, vol_window=vol_window))
        else:
            components.append(w * ts_sign(prices, L))
    return sum(components)


def vol_target_weights(signal: pd.Series, prices_up_to_date: pd.DataFrame,
                       vol_window: int = 63,
                       portfolio_vol_target: float = 0.10) -> dict:
    """
    Convert TSMOM signals to portfolio weights via vol targeting.

    Each asset: weight_i = signal_i * (target_vol_per_asset / vol_i)
    target_vol_per_asset = portfolio_vol_target / sqrt(N_active)

    Returns dict {asset: target_pct_of_NAV}.
    """
    # Per-asset realized vol (annualized)
    daily = prices_up_to_date.pct_change().iloc[-vol_window:]
    per_vol = daily.std() * np.sqrt(252)

    active = signal[signal.abs() > 0.05].index  # small threshold to skip near-zero
    if len(active) == 0:
        return {}

    # Inverse-vol sizing, per-asset
    target_per_asset = portfolio_vol_target / np.sqrt(len(active))
    weights = {}
    for a in active:
        v = per_vol.get(a, np.nan)
        if pd.notna(v) and v > 0:
            weights[a] = float(signal[a]) * (target_per_asset / v)
    return weights


# ─── Simulator (TSMOM mode) ─────────────────────────────────

def simulate_tsmom(prices, regime_monthly, *,
                   lookbacks=(21, 63, 126, 252),
                   continuous=True,
                   portfolio_vol_target=0.10,
                   rebalance_freq="weekly",
                   gross_cap=1.0,
                   ideal_nav_eligibility=True,
                   fractional_contracts=True,
                   apply_regime_weights=False,
                   hybrid_with_bsv=0.0):
    """
    Run backtest using TSMOM ensemble signal instead of BSV.

    hybrid_with_bsv: fraction to blend BSV signal with TSMOM
      0.0 = pure TSMOM, 1.0 = pure BSV, 0.5 = equal blend.
    """
    signal = tsmom_ensemble(prices, lookbacks=lookbacks, continuous=continuous)

    if hybrid_with_bsv > 0:
        bsv = bt.compute_bsv_composite(prices)
        aligned_idx = signal.index.intersection(bsv.index)
        aligned_cols = signal.columns.intersection(bsv.columns)
        signal = signal.loc[aligned_idx, aligned_cols]
        bsv = bsv.loc[aligned_idx, aligned_cols]
        signal = (1 - hybrid_with_bsv) * signal + hybrid_with_bsv * bsv

    config = bt.BacktestConfig(rebalance_freq=rebalance_freq)
    ELIGIBILITY_NAV = 10_000_000.0

    sim_start = pd.Timestamp(config.start_date)
    sim_dates = prices.loc[sim_start:].index
    nav = config.initial_nav
    positions = {}
    snapshots = []
    cost_rate = config.transaction_cost_bps / 10000.0

    for i, date in enumerate(sim_dates):
        today_prices = prices.loc[date]
        if positions and i > 0:
            prev_prices = prices.loc[sim_dates[i - 1]]
            pnl = 0.0
            for a, exp in positions.items():
                if a in today_prices.index and a in prev_prices.index:
                    pt, pp = today_prices[a], prev_prices[a]
                    if pd.notna(pt) and pd.notna(pp) and pp != 0:
                        pnl += exp * (pt / pp - 1)
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
        elif rebalance_freq == "monthly" and (i == 0 or date.month != sim_dates[i - 1].month):
            do_rebalance = True

        turnover = 0.0
        costs = 0.0

        if do_rebalance and date in signal.index:
            sig_today = signal.loc[date].dropna()
            # Volatility-target weights
            prices_to_date = prices.loc[:date]
            target_pcts = vol_target_weights(sig_today, prices_to_date,
                                             portfolio_vol_target=portfolio_vol_target)

            if apply_regime_weights:
                rw = bt.REGIME_WEIGHTS.get(current_regime, bt.REGIME_WEIGHTS["Goldilocks"])
                for a in list(target_pcts.keys()):
                    sector = bt.SECTOR_MAP.get(a, "other")
                    target_pcts[a] *= rw.get(sector, 1.0)

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
                        exposure = notional_target
                    else:
                        n = round(notional_target / one_lot)
                        exposure = n * one_lot
                    if exposure != 0:
                        new_positions[a] = exposure

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
    navs = np.array([s.nav for s in snapshots])
    rets = np.diff(navs) / navs[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) > 0 and np.std(rets) > 0:
        metrics["sharpe_arithmetic"] = (np.mean(rets) * 252 - 0.03) / (np.std(rets) * np.sqrt(252))
    else:
        metrics["sharpe_arithmetic"] = 0.0
    metrics["avg_n_positions"] = np.mean([s.n_positions for s in snapshots])
    metrics["avg_gross_pct_nav"] = np.mean([s.gross_exposure / s.nav if s.nav > 0 else 0 for s in snapshots])
    return {"metrics": metrics, "snapshots": snapshots}


def fmt(name, m):
    return (f"  {name:<56s}  "
            f"CAGR={m['cagr']*100:+6.2f}%  "
            f"Vol={m['annual_volatility']*100:5.1f}%  "
            f"Sh={m['sharpe_arithmetic']:+.3f}  "
            f"DD={m['max_drawdown']*100:+5.1f}%  "
            f"Pos={m['avg_n_positions']:4.1f}  "
            f"Gross={m['avg_gross_pct_nav']*100:5.1f}%  "
            f"Stag={m['regime_annualized_returns'].get('Stagflation',0)*100:+5.1f}%  "
            f"Gold={m['regime_annualized_returns'].get('Goldilocks',0)*100:+5.1f}%  "
            f"Refl={m['regime_annualized_returns'].get('Reflation',0)*100:+5.1f}%  "
            f"Defl={m['regime_annualized_returns'].get('Deflation',0)*100:+5.1f}%")


def main():
    with open(CACHE, "rb") as f:
        data = pickle.load(f)
    prices, regime = data["prices"], data["regime_monthly"]
    print(f"Frozen data: {prices.shape[1]} assets × {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})\n")

    scenarios = [
        # ── TSMOM variants ──
        dict(name="T1: TSMOM sign(1/3/6/12m) vol-target 10%",
             lookbacks=(21, 63, 126, 252), continuous=False, portfolio_vol_target=0.10),
        dict(name="T2: TSMOM continuous(1/3/6/12m) vt=10%",
             lookbacks=(21, 63, 126, 252), continuous=True, portfolio_vol_target=0.10),
        dict(name="T3: TSMOM continuous vt=15% higher risk",
             lookbacks=(21, 63, 126, 252), continuous=True, portfolio_vol_target=0.15),
        dict(name="T4: TSMOM continuous 3/6/12m (no 1m noise)",
             lookbacks=(63, 126, 252), continuous=True, portfolio_vol_target=0.10),
        dict(name="T5: TSMOM + regime weights applied",
             lookbacks=(21, 63, 126, 252), continuous=True, portfolio_vol_target=0.10,
             apply_regime_weights=True),
        dict(name="T6: Hybrid 50% TSMOM / 50% BSV",
             lookbacks=(21, 63, 126, 252), continuous=True, portfolio_vol_target=0.10,
             hybrid_with_bsv=0.5),
        dict(name="T7: Hybrid 70% TSMOM / 30% BSV",
             lookbacks=(21, 63, 126, 252), continuous=True, portfolio_vol_target=0.10,
             hybrid_with_bsv=0.3),
        dict(name="T8: TSMOM continuous 6/12m only (slow only)",
             lookbacks=(126, 252), continuous=True, portfolio_vol_target=0.10),
        dict(name="T9: TSMOM sign 3/6/12m slow-binary",
             lookbacks=(63, 126, 252), continuous=False, portfolio_vol_target=0.10),
    ]

    results = []
    for s in scenarios:
        name = s.pop("name")
        print(f"Running {name}")
        r = simulate_tsmom(prices, regime, **s)
        r["name"] = name
        results.append(r)

    print("\n" + "=" * 215)
    print("TSMOM SWEEP  (frozen data, 15.3y, $1M initial)")
    print("=" * 215)
    for r in results:
        print(fmt(r["name"], r["metrics"]))
    print("=" * 215)

    # For reference, print the best BSV scenario
    print("\nReference — best BSV config from v2 sweep:")
    print("  12: BSV baseline composite + ideal + frac + weekly + cap  "
          "CAGR=+1.12%  Vol=11.6%  Sh=-0.107  DD=-27.4%  Pos=12.0")

    out = []
    for r in results:
        m = r["metrics"]
        out.append({
            "name": r["name"],
            "cagr": m["cagr"], "vol": m["annual_volatility"],
            "sharpe_arith": m["sharpe_arithmetic"],
            "max_dd": m["max_drawdown"],
            "avg_positions": m["avg_n_positions"],
            "avg_gross_pct_nav": m["avg_gross_pct_nav"],
            "regime_returns": m["regime_annualized_returns"],
            "final_nav": m["final_nav"],
            "total_costs": m["total_costs"],
        })
    with open(Path(__file__).parent / "tsmom_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved tsmom_results.json")


if __name__ == "__main__":
    main()
