#!/usr/bin/env python3
"""
A.R.I.S Macro Overlay System — Historical Backtest Engine
=========================================================
Standalone reimplementation of the live A.R.I.S signal chain for backtesting.
Mirrors signal_engine/bsv_signals.py, regime_classifier.py, and daily_signals.py
exactly, using yfinance (free) for price data and FRED for macro data.

Usage:
    python aris_backtest.py                    # Run with defaults
    python aris_backtest.py --start 2011-01-01 # Custom start date
    python aris_backtest.py --no-dashboard     # Skip HTML output

Output:
    backtest_results/aris_backtest_results.json   — full results
    backtest_results/aris_dashboard.html           — interactive dashboard
    backtest_results/equity_curve.png              — static equity curve
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# 1. UNIVERSE & CONFIGURATION
#    Mirrors config/tickers.py — 22 GSCI commodity futures
# ─────────────────────────────────────────────────────────────

TICKERS = {
    # canonical: (yfinance_ticker, sector, multiplier, name)
    "CL": ("CL=F", "energy", 1000, "WTI Crude"),
    "BZ": ("BZ=F", "energy", 1000, "Brent Crude"),
    "NG": ("NG=F", "energy", 10000, "Natural Gas"),
    "HO": ("HO=F", "energy", 42000, "Heating Oil"),
    "RB": ("RB=F", "energy", 42000, "RBOB Gasoline"),
    "GC": ("GC=F", "precious_metals", 100, "Gold"),
    "SI": ("SI=F", "precious_metals", 5000, "Silver"),
    "PL": ("PL=F", "precious_metals", 50, "Platinum"),
    "PA": ("PA=F", "precious_metals", 100, "Palladium"),
    "HG": ("HG=F", "metals", 25000, "Copper"),
    "ZC": ("ZC=F", "agriculture", 5000, "Corn"),
    "ZW": ("ZW=F", "agriculture", 5000, "Wheat"),
    "ZS": ("ZS=F", "agriculture", 5000, "Soybeans"),
    "ZM": ("ZM=F", "agriculture", 100, "Soybean Meal"),
    "ZL": ("ZL=F", "agriculture", 60000, "Soybean Oil"),
    "CT": ("CT=F", "agriculture", 50000, "Cotton"),
    "KC": ("KC=F", "agriculture", 37500, "Coffee"),
    "SB": ("SB=F", "agriculture", 112000, "Sugar"),
    "CC": ("CC=F", "agriculture", 10, "Cocoa"),
    "LE": ("LE=F", "livestock", 40000, "Live Cattle"),
    "GF": ("GF=F", "livestock", 50000, "Feeder Cattle"),
    "HE": ("HE=F", "livestock", 40000, "Lean Hogs"),
}

SECTOR_MAP = {k: v[1] for k, v in TICKERS.items()}

# Micro contract multipliers (used in live for CL/GC/NG/HG)
MICRO_MULTIPLIERS = {
    "CL": 100,   # MCL  (1/10 of CL)
    "GC": 10,    # MGC  (1/10 of GC)
    "NG": 2500,  # QG   (1/4 of NG)
    "HG": 2500,  # MHG  (1/10 of HG)
}

# ─────────────────────────────────────────────────────────────
# 2. REGIME CLASSIFIER
#    Mirrors regime_classifier.py — ISM + CPI trend → quadrant
# ─────────────────────────────────────────────────────────────

REGIME_WEIGHTS = {
    "Goldilocks":  {"energy": 1.0, "metals": 1.0, "agriculture": 1.0, "livestock": 1.0, "precious_metals": 1.0},
    "Reflation":   {"energy": 1.2, "metals": 1.3, "agriculture": 1.0, "livestock": 0.9, "precious_metals": 1.1},
    "Stagflation":  {"energy": 0.8, "metals": 1.4, "agriculture": 1.0, "livestock": 0.7, "precious_metals": 1.5},
    "Deflation":   {"energy": 0.6, "metals": 0.8, "agriculture": 0.9, "livestock": 0.8, "precious_metals": 0.9},
}


def fetch_fred_series(series_id: str, start: str, end: str, api_key: str = "") -> pd.Series:
    """Fetch a FRED series via the public API. Falls back to hardcoded proxy if no key."""
    if api_key:
        import requests
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json().get("observations", [])
        records = [(r["date"], float(r["value"])) for r in data if r["value"] != "."]
        if records:
            s = pd.Series(dict(records))
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)


def classify_regime_series(ism: pd.Series, cpi: pd.Series, sma_window: int = 6) -> pd.Series:
    """
    Classify macro regime at each month based on ISM and CPI trends.
    Returns monthly Series of regime labels, forward-filled to daily.
    """
    # Resample to monthly if needed
    if len(ism) > 0:
        ism_monthly = ism.resample("ME").last().dropna()
    else:
        ism_monthly = pd.Series(dtype=float)

    if len(cpi) > 0:
        cpi_monthly = cpi.resample("ME").last().dropna()
    else:
        cpi_monthly = pd.Series(dtype=float)

    regimes = {}
    all_months = sorted(set(ism_monthly.index) & set(cpi_monthly.index))

    for i, month in enumerate(all_months):
        if i < sma_window:
            regimes[month] = "Goldilocks"  # default until enough data
            continue

        ism_window = ism_monthly.iloc[max(0, i - sma_window):i + 1]
        cpi_window = cpi_monthly.iloc[max(0, i - sma_window):i + 1]

        growth_trend = "up" if len(ism_window) >= 2 and ism_window.iloc[-1] > ism_window.iloc[0] else "down"
        inflation_trend = "up" if len(cpi_window) >= 2 and cpi_window.iloc[-1] > cpi_window.iloc[0] else "down"

        if growth_trend == "up" and inflation_trend == "down":
            regimes[month] = "Goldilocks"
        elif growth_trend == "up" and inflation_trend == "up":
            regimes[month] = "Reflation"
        elif growth_trend == "down" and inflation_trend == "up":
            regimes[month] = "Stagflation"
        else:
            regimes[month] = "Deflation"

    regime_series = pd.Series(regimes)
    regime_series.index = pd.to_datetime(regime_series.index)
    return regime_series


# ─────────────────────────────────────────────────────────────
# 3. BSV SIGNAL COMPUTATION
#    Mirrors bsv_signals.py — momentum, carry, value, reversal
# ─────────────────────────────────────────────────────────────

def rank_normalize(series: pd.Series) -> pd.Series:
    """Rank-normalize to [-1, +1] range. Mirrors bsv_signals._rank_normalize."""
    ranked = series.rank(pct=True)
    return 2 * ranked - 1


def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """12-month price momentum, rank-normalized cross-sectionally."""
    raw = prices.pct_change(window)
    return raw.apply(rank_normalize, axis=1)


def compute_carry_proxy(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Short-term return as carry proxy. In live, real curve carry from LSEG is used.
    For backtest, yfinance doesn't provide futures curves, so proxy is the fallback.
    """
    raw = prices.pct_change(window)
    return raw.apply(rank_normalize, axis=1)


def compute_value(prices: pd.DataFrame, window: int = 1260) -> pd.DataFrame:
    """5-year mean reversion signal, rank-normalized."""
    rolling_mean = prices.rolling(window, min_periods=window // 2).mean()
    deviation = (rolling_mean - prices) / rolling_mean
    return deviation.apply(rank_normalize, axis=1)


def compute_reversal(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """1-month contrarian reversal signal, rank-normalized."""
    raw = -prices.pct_change(window)  # negative = contrarian
    return raw.apply(rank_normalize, axis=1)


def compute_bsv_composite(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BSV composite signal. Mirrors generate_bsv_signals().
    Weights: momentum 0.40, carry 0.25, value 0.20, reversal 0.15
    """
    momentum = compute_momentum(prices)
    carry = compute_carry_proxy(prices)
    value = compute_value(prices)
    reversal = compute_reversal(prices)

    composite = (
        0.40 * momentum +
        0.25 * carry +
        0.20 * value +
        0.15 * reversal
    )
    return composite


# ─────────────────────────────────────────────────────────────
# 4. SECTOR RESCALE
#    Mirrors daily_signals.py — proportional cap at 40%
# ─────────────────────────────────────────────────────────────

SECTOR_GROSS_CAP = 0.40


def apply_sector_rescale(target_positions: dict[str, float]) -> tuple[dict[str, float], list]:
    """Cap any sector whose gross |target_pct| exceeds SECTOR_GROSS_CAP."""
    sector_gross: dict[str, float] = {}
    sector_assets: dict[str, list[str]] = {}

    for asset, tgt in target_positions.items():
        sector = SECTOR_MAP.get(asset, "other")
        sector_gross.setdefault(sector, 0.0)
        sector_gross[sector] += abs(tgt)
        sector_assets.setdefault(sector, []).append(asset)

    rescales = []
    result = dict(target_positions)
    for sector, gross in sector_gross.items():
        if gross > SECTOR_GROSS_CAP:
            scale = SECTOR_GROSS_CAP / gross
            for asset in sector_assets[sector]:
                result[asset] *= scale
            rescales.append((sector, gross, scale))

    return result, rescales


# ─────────────────────────────────────────────────────────────
# 5. PORTFOLIO SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    start_date: str = "2011-01-01"
    end_date: str = "2026-04-11"
    initial_nav: float = 1_000_000.0
    rebalance_freq: str = "daily"  # "daily" or "weekly" or "monthly"
    transaction_cost_bps: float = 2.0  # one-way, in basis points
    use_micros: bool = True  # use micro multipliers where available
    max_per_contract_pct: float = 0.03  # 3% per-contract notional cap
    fred_api_key: str = ""
    warmup_days: int = 1260  # 5yr for value signal


@dataclass
class DailySnapshot:
    date: pd.Timestamp
    nav: float
    gross_exposure: float
    net_exposure: float
    regime: str
    n_positions: int
    turnover: float
    costs: float
    daily_return: float
    positions: dict = field(default_factory=dict)


def get_multiplier(canonical: str, use_micros: bool = True) -> float:
    """Return the contract multiplier, using micros where available."""
    if use_micros and canonical in MICRO_MULTIPLIERS:
        return float(MICRO_MULTIPLIERS[canonical])
    return float(TICKERS[canonical][2])


def run_backtest(config: BacktestConfig) -> dict:
    """
    Run the full A.R.I.S backtest.

    Returns dict with:
        - daily_snapshots: list of DailySnapshot
        - metrics: performance metrics dict
        - regime_history: regime at each date
        - monthly_returns: monthly return series
    """
    print(f"A.R.I.S Macro Overlay Backtest")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial NAV: ${config.initial_nav:,.0f}")
    print(f"Transaction cost: {config.transaction_cost_bps} bps one-way")
    print()

    # ── Fetch price data ──────────────────────────────────
    print("Fetching price data from yfinance...")
    # Need extra lookback for warmup (5yr value signal)
    fetch_start = (pd.Timestamp(config.start_date) - timedelta(days=config.warmup_days + 60)).strftime("%Y-%m-%d")

    yf_tickers = {k: v[0] for k, v in TICKERS.items()}
    ticker_str = " ".join(yf_tickers.values())

    raw = yf.download(ticker_str, start=fetch_start, end=config.end_date,
                      auto_adjust=True, progress=False, threads=True)

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(yf_tickers)]
    else:
        prices = raw

    # Rename yfinance tickers → canonical names
    yf_to_canonical = {v: k for k, v in yf_tickers.items()}
    prices = prices.rename(columns=yf_to_canonical)

    # Keep only canonical columns that exist
    available = [c for c in TICKERS.keys() if c in prices.columns]
    prices = prices[available].sort_index()

    # Forward-fill gaps (weekends, holidays already excluded by yfinance)
    prices = prices.ffill().dropna(how="all")

    print(f"  {len(available)} assets loaded, {len(prices)} trading days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    missing = set(TICKERS.keys()) - set(available)
    if missing:
        print(f"  Missing: {sorted(missing)}")
    print()

    # ── Fetch macro data for regime classification ────────
    print("Fetching macro data (ISM + CPI) from FRED...")
    ism = fetch_fred_series("MANEMP", fetch_start, config.end_date, config.fred_api_key)
    cpi = fetch_fred_series("CPIAUCSL", fetch_start, config.end_date, config.fred_api_key)

    if len(ism) == 0 or len(cpi) == 0:
        print("  FRED API unavailable — using yfinance proxies for regime")
        # Proxy: use SPY (growth) and TIP (inflation expectations)
        macro_raw = yf.download("SPY TIP", start=fetch_start, end=config.end_date,
                                auto_adjust=True, progress=False)
        if isinstance(macro_raw.columns, pd.MultiIndex):
            macro_close = macro_raw["Close"]
        else:
            macro_close = macro_raw

        # Growth proxy: SPY 6-month return
        if "SPY" in macro_close.columns:
            ism = macro_close["SPY"].pct_change(126).dropna() * 100 + 50  # center around 50
        else:
            ism = pd.Series(dtype=float)

        # Inflation proxy: TIP 6-month return (TIPS ETF)
        if "TIP" in macro_close.columns:
            cpi = macro_close["TIP"].pct_change(126).dropna() * 100 + 2  # center around 2%
        else:
            cpi = pd.Series(dtype=float)

    if len(ism) > 0 and len(cpi) > 0:
        regime_monthly = classify_regime_series(ism, cpi)
        print(f"  {len(regime_monthly)} monthly regime observations")
        regime_counts = regime_monthly.value_counts()
        for r, c in regime_counts.items():
            print(f"    {r}: {c} months ({c / len(regime_monthly) * 100:.0f}%)")
    else:
        regime_monthly = pd.Series(dtype=str)
        print("  WARNING: No macro data — defaulting to Goldilocks throughout")
    print()

    # ── Compute BSV composite signals ─────────────────────
    print("Computing BSV composite signals...")
    composite = compute_bsv_composite(prices)
    print(f"  Signal matrix: {composite.shape[0]} days × {composite.shape[1]} assets")
    print()

    # ── Simulation loop ───────────────────────────────────
    print("Running portfolio simulation...")
    sim_start = pd.Timestamp(config.start_date)
    sim_dates = prices.loc[sim_start:].index

    nav = config.initial_nav
    positions: dict[str, float] = {}  # canonical → dollar exposure (signed)
    snapshots: list[DailySnapshot] = []
    prev_weights: dict[str, float] = {}

    cost_rate = config.transaction_cost_bps / 10000.0

    for i, date in enumerate(sim_dates):
        # Get today's prices
        today_prices = prices.loc[date]

        # Mark-to-market existing positions
        if positions and i > 0:
            prev_prices = prices.loc[sim_dates[i - 1]]
            pnl = 0.0
            for asset, exposure in positions.items():
                if asset in today_prices.index and asset in prev_prices.index:
                    p_today = today_prices[asset]
                    p_prev = prev_prices[asset]
                    if pd.notna(p_today) and pd.notna(p_prev) and p_prev != 0:
                        pnl += exposure * (p_today / p_prev - 1)
            nav += pnl

        # Get regime for this date
        if len(regime_monthly) > 0:
            valid_regimes = regime_monthly[regime_monthly.index <= date]
            current_regime = valid_regimes.iloc[-1] if len(valid_regimes) > 0 else "Goldilocks"
        else:
            current_regime = "Goldilocks"

        # Determine if we rebalance today
        do_rebalance = False
        if config.rebalance_freq == "daily":
            do_rebalance = True
        elif config.rebalance_freq == "weekly" and date.weekday() == 0:  # Monday
            do_rebalance = True
        elif config.rebalance_freq == "monthly" and (i == 0 or date.month != sim_dates[i - 1].month):
            do_rebalance = True

        if do_rebalance and date in composite.index:
            # Get BSV signals for today
            signals = composite.loc[date].dropna()

            # Apply regime weights
            target_pcts = {}
            regime_w = REGIME_WEIGHTS.get(current_regime, REGIME_WEIGHTS["Goldilocks"])
            for asset in signals.index:
                sector = SECTOR_MAP.get(asset, "other")
                weight = regime_w.get(sector, 1.0)
                target_pcts[asset] = float(signals[asset]) * weight

            # Apply sector rescale
            target_pcts, _ = apply_sector_rescale(target_pcts)

            # Convert to dollar exposure
            new_positions: dict[str, float] = {}
            for asset, tgt_pct in target_pcts.items():
                if asset in today_prices.index and pd.notna(today_prices[asset]):
                    price = float(today_prices[asset])
                    mult = get_multiplier(asset, config.use_micros)
                    one_lot_notional = price * mult

                    # Target notional
                    notional_target = tgt_pct * nav

                    # Per-contract cap: skip if one lot > max_per_contract_pct * nav
                    if one_lot_notional > config.max_per_contract_pct * nav:
                        continue  # can't trade this at current NAV

                    # Round to integer contracts
                    if one_lot_notional > 0:
                        n_contracts = round(notional_target / one_lot_notional)
                    else:
                        n_contracts = 0

                    if n_contracts != 0:
                        new_positions[asset] = n_contracts * one_lot_notional

            # Compute turnover and costs
            turnover = 0.0
            all_assets = set(list(positions.keys()) + list(new_positions.keys()))
            for asset in all_assets:
                old_exp = positions.get(asset, 0.0)
                new_exp = new_positions.get(asset, 0.0)
                turnover += abs(new_exp - old_exp)

            costs = turnover * cost_rate
            nav -= costs

            positions = new_positions

        # Compute daily metrics
        gross = sum(abs(v) for v in positions.values())
        net = sum(v for v in positions.values())
        daily_return = (nav / (snapshots[-1].nav if snapshots else config.initial_nav)) - 1 if snapshots or i > 0 else 0.0

        snap = DailySnapshot(
            date=date,
            nav=nav,
            gross_exposure=gross,
            net_exposure=net,
            regime=current_regime,
            n_positions=len([v for v in positions.values() if v != 0]),
            turnover=turnover if do_rebalance else 0.0,
            costs=costs if do_rebalance else 0.0,
            daily_return=daily_return,
        )
        snapshots.append(snap)

        turnover = 0.0
        costs = 0.0

    print(f"  Simulation complete: {len(snapshots)} trading days")
    print()

    # ── Compute performance metrics ───────────────────────
    metrics = compute_metrics(snapshots, config)

    # ── Monthly returns ───────────────────────────────────
    nav_series = pd.Series({s.date: s.nav for s in snapshots})
    monthly_nav = nav_series.resample("ME").last()
    monthly_returns = monthly_nav.pct_change().dropna()

    # ── Regime history ────────────────────────────────────
    regime_history = pd.Series({s.date: s.regime for s in snapshots})

    return {
        "snapshots": snapshots,
        "metrics": metrics,
        "monthly_returns": monthly_returns,
        "regime_history": regime_history,
        "nav_series": nav_series,
        "config": config,
    }


def compute_metrics(snapshots: list[DailySnapshot], config: BacktestConfig) -> dict:
    """Compute standard performance metrics."""
    if len(snapshots) < 2:
        return {}

    navs = np.array([s.nav for s in snapshots])
    returns = np.diff(navs) / navs[:-1]

    # Basic
    total_return = (navs[-1] / navs[0]) - 1
    n_years = (snapshots[-1].date - snapshots[0].date).days / 365.25
    cagr = (navs[-1] / navs[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk
    daily_vol = np.std(returns) if len(returns) > 0 else 0
    annual_vol = daily_vol * np.sqrt(252)
    sharpe = (cagr - 0.03) / annual_vol if annual_vol > 0 else 0  # assume 3% risk-free

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (cagr - 0.03) / downside_vol if downside_vol > 0 else 0

    # Drawdown
    peak = np.maximum.accumulate(navs)
    drawdown = (navs - peak) / peak
    max_drawdown = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = snapshots[max_dd_idx].date.strftime("%Y-%m-%d")

    # Find max drawdown duration
    in_drawdown = drawdown < 0
    dd_durations = []
    current_dd_start = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and current_dd_start is None:
            current_dd_start = i
        elif not is_dd and current_dd_start is not None:
            dd_durations.append(i - current_dd_start)
            current_dd_start = None
    if current_dd_start is not None:
        dd_durations.append(len(in_drawdown) - current_dd_start)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    # Win rate
    win_days = np.sum(returns > 0)
    total_days = len(returns)
    win_rate = win_days / total_days if total_days > 0 else 0

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Average exposure
    avg_gross = np.mean([s.gross_exposure for s in snapshots])
    avg_positions = np.mean([s.n_positions for s in snapshots])

    # Total costs
    total_costs = sum(s.costs for s in snapshots)

    # Regime breakdown
    regime_returns = {}
    for regime_name in REGIME_WEIGHTS.keys():
        regime_days = [i for i, s in enumerate(snapshots[1:], 1) if s.regime == regime_name]
        if regime_days:
            r = np.mean([returns[i - 1] for i in regime_days if i - 1 < len(returns)]) * 252
            regime_returns[regime_name] = r

    # Best/worst
    best_day = np.max(returns) if len(returns) > 0 else 0
    worst_day = np.min(returns) if len(returns) > 0 else 0
    best_month_returns = []
    worst_month_returns = []

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_dd_date,
        "max_drawdown_duration_days": max_dd_duration,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "avg_gross_exposure": avg_gross,
        "avg_positions": avg_positions,
        "total_costs": total_costs,
        "n_years": n_years,
        "n_trading_days": len(snapshots),
        "regime_annualized_returns": regime_returns,
        "initial_nav": config.initial_nav,
        "final_nav": navs[-1],
    }


# ─────────────────────────────────────────────────────────────
# 6. HTML DASHBOARD GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_dashboard(results: dict, output_path: str = "aris_dashboard.html"):
    """Generate a self-contained interactive HTML dashboard."""
    snapshots = results["snapshots"]
    metrics = results["metrics"]
    monthly_returns = results["monthly_returns"]
    regime_history = results["regime_history"]
    config = results["config"]

    # Prepare data for charts
    dates = [s.date.strftime("%Y-%m-%d") for s in snapshots]
    navs = [round(s.nav, 2) for s in snapshots]
    regimes = [s.regime for s in snapshots]

    # Drawdown series
    nav_arr = np.array([s.nav for s in snapshots])
    peak = np.maximum.accumulate(nav_arr)
    drawdown = ((nav_arr - peak) / peak * 100).tolist()

    # Gross exposure
    gross = [round(s.gross_exposure / s.nav * 100, 1) if s.nav > 0 else 0 for s in snapshots]

    # Monthly returns heatmap data
    monthly_data = []
    for date, ret in monthly_returns.items():
        monthly_data.append({
            "year": date.year,
            "month": date.month,
            "return": round(ret * 100, 2)
        })

    # Regime colors
    regime_colors = {
        "Goldilocks": "#4CAF50",
        "Reflation": "#FF9800",
        "Stagflation": "#f44336",
        "Deflation": "#2196F3",
    }

    # Regime background data for chart
    regime_bands = []
    current_regime = regimes[0] if regimes else "Goldilocks"
    band_start = dates[0] if dates else ""
    for i in range(1, len(dates)):
        if regimes[i] != current_regime:
            regime_bands.append({"start": band_start, "end": dates[i], "regime": current_regime})
            current_regime = regimes[i]
            band_start = dates[i]
    if dates:
        regime_bands.append({"start": band_start, "end": dates[-1], "regime": current_regime})

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A.R.I.S Macro Overlay — Backtest Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root {{
    --bg: #0a0a0f;
    --card: #12121a;
    --border: #1e1e2e;
    --text: #e0e0e0;
    --text-muted: #888;
    --accent: #6366f1;
    --green: #22c55e;
    --red: #ef4444;
    --gold: #eab308;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
}}
.header {{
    text-align: center;
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}}
.header h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #fff;
}}
.header .subtitle {{
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 4px;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}}
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}}
.metric-card .label {{
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 1px;
}}
.metric-card .value {{
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
}}
.metric-card .value.positive {{ color: var(--green); }}
.metric-card .value.negative {{ color: var(--red); }}
.chart-container {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}}
.chart-container h3 {{
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.chart-container canvas {{
    width: 100% !important;
}}
.monthly-grid {{
    display: grid;
    grid-template-columns: 50px repeat(12, 1fr);
    gap: 2px;
    font-size: 11px;
    text-align: center;
}}
.monthly-grid .header-cell {{
    color: var(--text-muted);
    padding: 4px;
    font-weight: 600;
}}
.monthly-grid .year-cell {{
    color: var(--text-muted);
    padding: 4px;
    font-weight: 600;
    text-align: right;
    padding-right: 8px;
}}
.monthly-grid .cell {{
    padding: 4px;
    border-radius: 3px;
    font-weight: 500;
}}
.regime-legend {{
    display: flex;
    gap: 16px;
    justify-content: center;
    margin: 12px 0;
    font-size: 12px;
}}
.regime-legend span {{
    display: flex;
    align-items: center;
    gap: 4px;
}}
.regime-legend .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}}
@media (max-width: 768px) {{
    .two-col {{ grid-template-columns: 1fr; }}
}}
.footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}}
</style>
</head>
<body>

<div class="header">
    <h1>A.R.I.S MACRO OVERLAY</h1>
    <div class="subtitle">
        Backtest: {config.start_date} to {config.end_date} &nbsp;|&nbsp;
        {config.transaction_cost_bps} bps cost &nbsp;|&nbsp;
        {'Micros' if config.use_micros else 'Full-size'} contracts &nbsp;|&nbsp;
        ${config.initial_nav / 1e6:.1f}M initial
    </div>
</div>

<div class="metrics-grid">
    <div class="metric-card">
        <div class="label">Total Return</div>
        <div class="value {'positive' if metrics.get('total_return',0)>=0 else 'negative'}">{metrics.get('total_return',0)*100:+.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">CAGR</div>
        <div class="value {'positive' if metrics.get('cagr',0)>=0 else 'negative'}">{metrics.get('cagr',0)*100:+.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">Sharpe Ratio</div>
        <div class="value">{metrics.get('sharpe_ratio',0):.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Sortino Ratio</div>
        <div class="value">{metrics.get('sortino_ratio',0):.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Max Drawdown</div>
        <div class="value negative">{metrics.get('max_drawdown',0)*100:.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">Annual Vol</div>
        <div class="value">{metrics.get('annual_volatility',0)*100:.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">Calmar Ratio</div>
        <div class="value">{metrics.get('calmar_ratio',0):.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Win Rate</div>
        <div class="value">{metrics.get('win_rate',0)*100:.1f}%</div>
    </div>
</div>

<div class="chart-container">
    <h3>Equity Curve</h3>
    <div class="regime-legend">
        <span><span class="dot" style="background:#4CAF50"></span> Goldilocks</span>
        <span><span class="dot" style="background:#FF9800"></span> Reflation</span>
        <span><span class="dot" style="background:#f44336"></span> Stagflation</span>
        <span><span class="dot" style="background:#2196F3"></span> Deflation</span>
    </div>
    <canvas id="equityChart" height="100"></canvas>
</div>

<div class="two-col">
    <div class="chart-container">
        <h3>Drawdown</h3>
        <canvas id="drawdownChart" height="80"></canvas>
    </div>
    <div class="chart-container">
        <h3>Gross Exposure (% NAV)</h3>
        <canvas id="exposureChart" height="80"></canvas>
    </div>
</div>

<div class="chart-container">
    <h3>Monthly Returns (%)</h3>
    <div class="monthly-grid" id="monthlyGrid"></div>
</div>

<div class="chart-container">
    <h3>Regime Analysis — Annualized Returns</h3>
    <canvas id="regimeChart" height="60"></canvas>
</div>

<div class="footer">
    A.R.I.S Macro Overlay System &nbsp;|&nbsp; Backtest generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
    &nbsp;|&nbsp; Note: carry signal uses price proxy (no futures curves in yfinance); live system uses real LSEG curve carry.
</div>

<script>
const dates = {json.dumps(dates[::5])};  // downsample for performance
const navs = {json.dumps(navs[::5])};
const drawdowns = {json.dumps([round(d,2) for d in drawdown[::5]])};
const grossExposure = {json.dumps(gross[::5])};
const monthlyData = {json.dumps(monthly_data)};
const regimeBands = {json.dumps(regime_bands)};
const regimeReturns = {json.dumps(metrics.get('regime_annualized_returns', {}))};
const regimeColors = {json.dumps(regime_colors)};

// Regime annotation plugin
const regimePlugin = {{
    id: 'regimeBands',
    beforeDraw(chart) {{
        const {{ctx, chartArea: {{left, right, top, bottom}}, scales: {{x}}}} = chart;
        if (!x) return;
        regimeBands.forEach(band => {{
            const x0 = x.getPixelForValue(band.start);
            const x1 = x.getPixelForValue(band.end);
            if (x1 < left || x0 > right) return;
            ctx.fillStyle = regimeColors[band.regime] + '15';
            ctx.fillRect(Math.max(x0, left), top, Math.min(x1, right) - Math.max(x0, left), bottom - top);
        }});
    }}
}};

// Equity curve
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: dates,
        datasets: [{{
            label: 'NAV',
            data: navs,
            borderColor: '#6366f1',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{
                type: 'time',
                time: {{ unit: 'year' }},
                grid: {{ color: '#1e1e2e' }},
                ticks: {{ color: '#888' }}
            }},
            y: {{
                grid: {{ color: '#1e1e2e' }},
                ticks: {{
                    color: '#888',
                    callback: v => '$' + (v/1e6).toFixed(1) + 'M'
                }}
            }}
        }}
    }},
    plugins: [regimePlugin]
}});

// Drawdown
new Chart(document.getElementById('drawdownChart'), {{
    type: 'line',
    data: {{
        labels: dates,
        datasets: [{{
            label: 'Drawdown %',
            data: drawdowns,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239,68,68,0.1)',
            borderWidth: 1,
            pointRadius: 0,
            fill: true,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ type: 'time', time: {{ unit: 'year' }}, grid: {{ color: '#1e1e2e' }}, ticks: {{ color: '#888' }} }},
            y: {{ grid: {{ color: '#1e1e2e' }}, ticks: {{ color: '#888', callback: v => v + '%' }} }}
        }}
    }}
}});

// Gross exposure
new Chart(document.getElementById('exposureChart'), {{
    type: 'line',
    data: {{
        labels: dates,
        datasets: [{{
            label: 'Gross %',
            data: grossExposure,
            borderColor: '#eab308',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ type: 'time', time: {{ unit: 'year' }}, grid: {{ color: '#1e1e2e' }}, ticks: {{ color: '#888' }} }},
            y: {{ grid: {{ color: '#1e1e2e' }}, ticks: {{ color: '#888', callback: v => v + '%' }} }}
        }}
    }}
}});

// Monthly returns heatmap
const grid = document.getElementById('monthlyGrid');
const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// Header row
grid.innerHTML = '<div class="header-cell"></div>';
months.forEach(m => {{ grid.innerHTML += `<div class="header-cell">${{m}}</div>`; }});

// Group by year
const byYear = {{}};
monthlyData.forEach(d => {{
    if (!byYear[d.year]) byYear[d.year] = {{}};
    byYear[d.year][d.month] = d.return;
}});

Object.keys(byYear).sort().forEach(year => {{
    grid.innerHTML += `<div class="year-cell">${{year}}</div>`;
    for (let m = 1; m <= 12; m++) {{
        const val = byYear[year][m];
        if (val !== undefined) {{
            const color = val >= 0
                ? `rgba(34,197,94,${{Math.min(Math.abs(val)/5, 1) * 0.7 + 0.1}})`
                : `rgba(239,68,68,${{Math.min(Math.abs(val)/5, 1) * 0.7 + 0.1}})`;
            grid.innerHTML += `<div class="cell" style="background:${{color}}">${{val > 0 ? '+' : ''}}${{val.toFixed(1)}}</div>`;
        }} else {{
            grid.innerHTML += `<div class="cell"></div>`;
        }}
    }}
}});

// Regime bar chart
const regimeLabels = Object.keys(regimeReturns);
const regimeVals = regimeLabels.map(r => (regimeReturns[r] * 100).toFixed(1));
const regimeBgColors = regimeLabels.map(r => regimeColors[r] || '#888');

new Chart(document.getElementById('regimeChart'), {{
    type: 'bar',
    data: {{
        labels: regimeLabels,
        datasets: [{{
            label: 'Ann. Return %',
            data: regimeVals,
            backgroundColor: regimeBgColors,
            borderRadius: 4,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
            y: {{ grid: {{ color: '#1e1e2e' }}, ticks: {{ color: '#888', callback: v => v + '%' }} }}
        }}
    }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard saved to: {output_path}")


# ─────────────────────────────────────────────────────────────
# 7. STATIC CHART GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_static_charts(results: dict, output_dir: str = "."):
    """Generate matplotlib charts for the notebook / static output."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    snapshots = results["snapshots"]
    metrics = results["metrics"]

    dates = [s.date for s in snapshots]
    navs = [s.nav for s in snapshots]
    regimes = [s.regime for s in snapshots]

    # Drawdown
    nav_arr = np.array(navs)
    peak = np.maximum.accumulate(nav_arr)
    drawdown = (nav_arr - peak) / peak * 100

    regime_colors = {
        "Goldilocks": "#4CAF50",
        "Reflation": "#FF9800",
        "Stagflation": "#f44336",
        "Deflation": "#2196F3",
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.patch.set_facecolor("#0a0a0f")

    for ax in axes:
        ax.set_facecolor("#12121a")
        ax.tick_params(colors="#888")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333")
        ax.spines["bottom"].set_color("#333")

    # 1. Equity curve with regime shading
    ax1 = axes[0]
    ax1.plot(dates, navs, color="#6366f1", linewidth=1.2, label="A.R.I.S NAV")

    # Shade regimes
    current_regime = regimes[0]
    start_idx = 0
    for i in range(1, len(dates)):
        if regimes[i] != current_regime or i == len(dates) - 1:
            ax1.axvspan(dates[start_idx], dates[i], alpha=0.08,
                       color=regime_colors.get(current_regime, "#888"))
            current_regime = regimes[i]
            start_idx = i

    ax1.set_ylabel("NAV ($)", color="#888")
    ax1.set_title("A.R.I.S Macro Overlay — Equity Curve", color="white", fontsize=14, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
    ax1.legend(facecolor="#12121a", edgecolor="#333", labelcolor="#888")
    ax1.grid(True, alpha=0.1)

    # 2. Drawdown
    ax2 = axes[1]
    ax2.fill_between(dates, drawdown, 0, color="#ef4444", alpha=0.3)
    ax2.plot(dates, drawdown, color="#ef4444", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", color="#888")
    ax2.grid(True, alpha=0.1)

    # 3. Regime timeline
    ax3 = axes[2]
    for i in range(len(dates) - 1):
        ax3.axvspan(dates[i], dates[i + 1], color=regime_colors.get(regimes[i], "#888"), alpha=0.6)
    ax3.set_ylabel("Regime", color="#888")
    ax3.set_yticks([])

    # Legend for regimes
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=r, alpha=0.6) for r, c in regime_colors.items()]
    ax3.legend(handles=legend_patches, loc="upper center", ncol=4,
              facecolor="#12121a", edgecolor="#333", labelcolor="#888", fontsize=9)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(chart_path, dpi=150, facecolor="#0a0a0f", bbox_inches="tight")
    plt.close()
    print(f"Static chart saved to: {chart_path}")


# ─────────────────────────────────────────────────────────────
# 8. RESULTS EXPORT
# ─────────────────────────────────────────────────────────────

def export_results(results: dict, output_dir: str = "backtest_results"):
    """Export results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Metrics JSON
    metrics_path = os.path.join(output_dir, "aris_backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)

    # Daily NAV CSV
    nav_path = os.path.join(output_dir, "daily_nav.csv")
    nav_df = pd.DataFrame([
        {
            "date": s.date.strftime("%Y-%m-%d"),
            "nav": round(s.nav, 2),
            "daily_return": round(s.daily_return, 6),
            "regime": s.regime,
            "n_positions": s.n_positions,
            "gross_exposure": round(s.gross_exposure, 2),
            "costs": round(s.costs, 2),
        }
        for s in results["snapshots"]
    ])
    nav_df.to_csv(nav_path, index=False)

    # Monthly returns CSV
    monthly_path = os.path.join(output_dir, "monthly_returns.csv")
    mr = results["monthly_returns"]
    mr_df = pd.DataFrame({"date": mr.index.strftime("%Y-%m"), "return": mr.values})
    mr_df.to_csv(monthly_path, index=False)

    print(f"Results exported to: {output_dir}/")


# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="A.R.I.S Macro Overlay Backtest")
    parser.add_argument("--start", default="2011-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-11", help="Backtest end date")
    parser.add_argument("--nav", type=float, default=1_000_000, help="Initial NAV")
    parser.add_argument("--cost-bps", type=float, default=2.0, help="Transaction cost (bps, one-way)")
    parser.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--no-micros", action="store_true", help="Use full-size contracts")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip HTML dashboard")
    parser.add_argument("--fred-key", default="", help="FRED API key (optional)")
    parser.add_argument("--output-dir", default="backtest_results", help="Output directory")
    args = parser.parse_args()

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_nav=args.nav,
        transaction_cost_bps=args.cost_bps,
        rebalance_freq=args.rebalance,
        use_micros=not args.no_micros,
        fred_api_key=args.fred_key,
    )

    # Run backtest
    results = run_backtest(config)

    # Print summary
    m = results["metrics"]
    print("\n" + "=" * 60)
    print("  A.R.I.S MACRO OVERLAY — BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:         {config.start_date} to {config.end_date} ({m.get('n_years',0):.1f} years)")
    print(f"  Initial NAV:    ${m.get('initial_nav',0):,.0f}")
    print(f"  Final NAV:      ${m.get('final_nav',0):,.0f}")
    print(f"  Total Return:   {m.get('total_return',0)*100:+.1f}%")
    print(f"  CAGR:           {m.get('cagr',0)*100:+.1f}%")
    print(f"  Annual Vol:     {m.get('annual_volatility',0)*100:.1f}%")
    print(f"  Sharpe Ratio:   {m.get('sharpe_ratio',0):.2f}")
    print(f"  Sortino Ratio:  {m.get('sortino_ratio',0):.2f}")
    print(f"  Max Drawdown:   {m.get('max_drawdown',0)*100:.1f}%")
    print(f"  Calmar Ratio:   {m.get('calmar_ratio',0):.2f}")
    print(f"  Win Rate:       {m.get('win_rate',0)*100:.1f}%")
    print(f"  Avg Positions:  {m.get('avg_positions',0):.1f}")
    print(f"  Total Costs:    ${m.get('total_costs',0):,.0f}")
    print()
    print("  Regime Analysis (annualized returns):")
    for regime, ret in m.get('regime_annualized_returns', {}).items():
        print(f"    {regime:<14} {ret*100:+.1f}%")
    print("=" * 60)

    # Export
    export_results(results, args.output_dir)

    # Dashboard
    if not args.no_dashboard:
        dashboard_path = os.path.join(args.output_dir, "aris_dashboard.html")
        generate_dashboard(results, dashboard_path)

    # Static charts
    generate_static_charts(results, args.output_dir)


if __name__ == "__main__":
    main()
