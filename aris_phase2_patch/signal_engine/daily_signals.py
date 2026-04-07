"""
Module 1 — Daily Signal Pipeline
Orchestrates BSV signals + regime classifier into a daily JSON output.
This is the main entry point for the signal engine.

Phase 2:
  - Uses DualSourceRouter via bsv_signals/regime_classifier (LSEG primary
    for prices+curves, FRED primary for macro).
  - Real curve carry replaces the legacy proxy when LSEG curves are available.
  - HealthRecord captures provider, source, and cross-validation
    disagreements for every input.
"""
import json
import os
from datetime import datetime

import pandas as pd

from signal_engine.bsv_signals import (
    fetch_commodity_prices,
    fetch_commodity_curves,
    generate_bsv_signals,
    _get_router as _get_bsv_router,
)
from signal_engine.regime_classifier import classify_regime, REGIME_WEIGHTS
from signal_engine.resilience import HealthRecord, log
from config.tickers import get_canonical_list, SECTOR_MAP
from config.settings import SIGNAL_OUTPUT_PATH


def run_daily_signals() -> dict:
    """
    Run the full daily signal pipeline with resilient, dual-source data.
    Each input is wrapped: failures retry, then fall back to cached values,
    and the freshness/provider/disagreements of every source are recorded
    in the output JSON.
    """
    health = HealthRecord()
    log.info("=== Daily signal pipeline started (Phase 2) ===")

    canonical = get_canonical_list()

    # 1. Regime ----------------------------------------------------------
    try:
        regime = classify_regime()
        health.record(
            "macro_ism",
            regime.get("ism_source", "fresh"),
            provider=regime.get("ism_provider", "fred"),
        )
        health.record(
            "macro_cpi",
            regime.get("cpi_source", "fresh"),
            provider=regime.get("cpi_provider", "fred"),
        )
        for k, v in (regime.get("router_disagreements") or {}).items():
            health.record_disagreement(f"macro:{k}", v)
        log.info(
            f"Regime: {regime['regime']} "
            f"(Growth {regime['growth_trend']}, Inflation {regime['inflation_trend']})"
        )
    except Exception as e:
        log.critical(f"Regime classification failed completely: {e}")
        health.record("macro_ism", "missing")
        health.record("macro_cpi", "missing")
        regime = {
            "regime": "Unknown",
            "growth_trend": "unknown",
            "inflation_trend": "unknown",
            "growth_value": None,
            "inflation_value": None,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }

    # 2. Prices + curves + BSV signals -----------------------------------
    try:
        prices = fetch_commodity_prices(canonical, lookback_days=365 * 5)
        router = _get_bsv_router()
        health.record(
            "prices",
            prices.attrs.get("source", "fresh"),
            provider=prices.attrs.get("provider", "lseg"),
        )
        health.merge_router_disagreements(router.last_disagreements)

        curves = fetch_commodity_curves(canonical)
        health.record(
            "curves",
            f"{len(curves)}/{len(canonical)}",
            provider="lseg",
        )

        bsv = generate_bsv_signals(prices, curves=curves)
    except Exception as e:
        log.critical(f"Price/signal pipeline failed: {e}")
        health.record("prices", "missing")
        health.record("curves", "missing")
        bsv = pd.DataFrame()

    # 3. Apply regime-based sector weights ------------------------------
    target_positions: dict[str, float] = {}
    if not bsv.empty and "composite" in bsv.columns and regime.get("regime") in REGIME_WEIGHTS:
        weights = REGIME_WEIGHTS[regime["regime"]]
        for asset in bsv.index:
            sector = SECTOR_MAP.get(asset, "metals")
            mult = weights.get(sector, 1.0)
            target_positions[asset] = float(bsv.loc[asset, "composite"]) * mult
    elif not bsv.empty and "composite" in bsv.columns:
        target_positions = {a: float(v) for a, v in bsv["composite"].items()}

    # 4. Build output ---------------------------------------------------
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "signals": (
            bsv.drop(columns=["carry_source"], errors="ignore")
            .to_dict(orient="index")
            if not bsv.empty
            else {}
        ),
        "carry_source_per_asset": (
            bsv["carry_source"].to_dict()
            if "carry_source" in bsv.columns
            else {}
        ),
        "target_positions": target_positions,
        "health": health.to_dict(),
    }

    if not health.is_healthy():
        log.warning(
            f"Pipeline finished with degraded inputs / disagreements: {health.errors}"
        )
    else:
        log.info("Pipeline finished healthy — all inputs fresh, no disagreements")

    return output


def save_daily_signals(output: dict) -> str:
    """Save daily signal output as JSON."""
    os.makedirs(SIGNAL_OUTPUT_PATH, exist_ok=True)
    filename = f"{SIGNAL_OUTPUT_PATH}{output['date']}_signals.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved signals to {filename}")
    return filename


if __name__ == "__main__":
    output = run_daily_signals()
    save_daily_signals(output)
    print(f"\nRegime: {output['regime'].get('regime', 'Unknown')}")
    print(f"Health: {'OK' if output['health']['healthy'] else 'DEGRADED'}")
    if output["target_positions"]:
        top = sorted(
            output["target_positions"].items(), key=lambda x: x[1], reverse=True
        )
        print(f"Top 5 long:  {top[:5]}")
        print(f"Top 5 short: {top[-5:]}")
