"""
Module 1 — Daily Signal Pipeline
Phase 2 orchestrator: DualSourceRouter, real curve carry, regime-weighted targets.
"""
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

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

SECTOR_GROSS_CAP = 0.40


def run_daily_signals() -> dict:
    health = HealthRecord()
    skipped_assets: list[tuple[str, float]] = []
    rescaled_sectors: list[tuple[str, float, float]] = []
    log.info("=== Daily signal pipeline started (Phase 2) ===")

    canonical = get_canonical_list()

    # 1. Regime
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
            "regime": "Unknown", "growth_trend": "unknown", "inflation_trend": "unknown",
            "growth_value": None, "inflation_value": None,
            "timestamp": datetime.now().isoformat(), "error": str(e),
        }


    # 2. Prices + curves + BSV
    try:
        prices = fetch_commodity_prices(canonical, lookback_days=365 * 5)
        router = _get_bsv_router()
        health.record(
            "prices",
            prices.attrs.get("source", "fresh"),
            provider=prices.attrs.get("provider", "lseg"),
        )
        health.merge_router_disagreements(router.last_disagreements)

        # 2b. Drop assets with no current price (NA or 0) — data quality
        #     filter only. Assets with stale/missing prices produce garbage
        #     BSV signals. The executor's risk layer handles sizing.
        valid_prices = prices.dropna(axis=1, how='all').loc[
            :, (prices.iloc[-1].notna()) & (prices.iloc[-1] != 0)
        ]
        if len(valid_prices.columns) < len(prices.columns):
            dropped = sorted(set(prices.columns) - set(valid_prices.columns))
            log.info(f"Dropped {len(dropped)} assets with no current price: {dropped}")
            for sym in dropped:
                skipped_assets.append((sym, 0.0))
            prices = valid_prices

        curves = fetch_commodity_curves(list(prices.columns))
        health.record("curves", f"{len(curves)}/{len(canonical)}", provider="lseg")

        bsv = generate_bsv_signals(prices, curves=curves)
    except Exception as e:
        log.critical(f"Price/signal pipeline failed: {e}")
        health.record("prices", "missing")
        health.record("curves", "missing")
        bsv = pd.DataFrame()

    # 3. Build target positions via vol-target sizing (Phase A) ---------
    from signal_engine.tsmom import compute_asset_volatility
    from config.settings import (
        PORTFOLIO_VOL_TARGET,
        VOL_TARGET_WINDOW,
        SIGNAL_GROSS_CAP,
        TSMOM_LOOKBACKS,
        TSMOM_WEIGHT,
        USE_HYBRID_ALPHA,
    )
    from signal_engine.bsv_signals import generate_hybrid_signals

    target_positions: dict[str, float] = {}

    if USE_HYBRID_ALPHA and not prices.empty:
        # Regenerate using the hybrid engine — supersedes the BSV-only call above
        bsv = generate_hybrid_signals(
            prices,
            curves=curves,
            tsmom_lookbacks=TSMOM_LOOKBACKS,
            tsmom_weight=TSMOM_WEIGHT,
        )

    if not bsv.empty and "composite" in bsv.columns:
        asset_vol = compute_asset_volatility(prices, vol_window=VOL_TARGET_WINDOW)
        active_assets = [
            a for a in bsv.index
            if abs(float(bsv.loc[a, "composite"])) > 0.05
            and a in asset_vol.index
            and asset_vol[a] > 0
        ]
        n_active = max(len(active_assets), 1)
        target_vol_per_asset = PORTFOLIO_VOL_TARGET / np.sqrt(n_active)

        for asset in active_assets:
            signal = float(bsv.loc[asset, "composite"])
            vol = float(asset_vol[asset])
            # weight_i = signal_i * (target_vol_per_asset / vol_i)
            target_positions[asset] = signal * (target_vol_per_asset / vol)

        # Signal-level gross cap (prevents unbounded gross on strong signals)
        gross = sum(abs(v) for v in target_positions.values())
        if gross > SIGNAL_GROSS_CAP:
            scale = SIGNAL_GROSS_CAP / gross
            target_positions = {k: v * scale for k, v in target_positions.items()}
            health.record("gross_cap_scaled", f"{gross:.2%}->{SIGNAL_GROSS_CAP:.2%}")

        # Optional: apply regime overlay on top (kept for A/B testing; default off)
        from config.settings import APPLY_REGIME_TILTS
        if APPLY_REGIME_TILTS and regime.get("regime") in REGIME_WEIGHTS:
            rw = REGIME_WEIGHTS[regime["regime"]]
            for a in list(target_positions.keys()):
                sector = SECTOR_MAP.get(a, "metals")
                target_positions[a] *= rw.get(sector, 1.0)

    # 3b. Proportional sector rescale — cap any sector whose gross weight
    #     exceeds SECTOR_GROSS_CAP. This prevents correlated legs (e.g. all
    #     5 energy futures going long simultaneously) from dominating the book.
    #     The rescale is proportional: all legs in the breaching sector are
    #     scaled by the same factor, preserving relative ordering within sector.
    if target_positions:
        sector_gross: dict[str, float] = {}
        sector_assets: dict[str, list[str]] = {}
        for asset, tgt in target_positions.items():
            sector = SECTOR_MAP.get(asset, "other")
            sector_gross.setdefault(sector, 0.0)
            sector_gross[sector] += abs(tgt)
            sector_assets.setdefault(sector, []).append(asset)

        rescaled_sectors = []
        for sector, gross in sector_gross.items():
            if gross > SECTOR_GROSS_CAP:
                scale = SECTOR_GROSS_CAP / gross
                for asset in sector_assets[sector]:
                    target_positions[asset] *= scale
                rescaled_sectors.append((sector, gross, scale))

        if rescaled_sectors:
            log.info("Sector neutralization applied:")
            for sector, old_gross, scale in rescaled_sectors:
                log.info(
                    f"  {sector}: {old_gross:.1%} -> {SECTOR_GROSS_CAP:.0%} "
                    f"(scale={scale:.3f})"
                )

    # 4. Output
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "signals": (
            bsv.drop(columns=["carry_source"], errors="ignore").to_dict(orient="index")
            if not bsv.empty else {}
        ),
        "carry_source_per_asset": (
            bsv["carry_source"].to_dict() if "carry_source" in bsv.columns else {}
        ),
        "target_positions": target_positions,
        "excluded_assets": [
            {"symbol": s, "one_lot_notional": n} for s, n in skipped_assets
        ],
        "sector_rescales": [
            {"sector": s, "raw_gross": g, "scale": sc} for s, g, sc in rescaled_sectors
        ],
        "health": health.to_dict(),
    }

    if not health.is_healthy():
        log.warning(f"Pipeline finished degraded: {health.errors}")
    else:
        log.info("Pipeline finished healthy — all inputs fresh, no disagreements")

    return output


def save_daily_signals(output: dict) -> str:
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
        top = sorted(output["target_positions"].items(), key=lambda x: x[1], reverse=True)
        print(f"Top 5 long:  {top[:5]}")
        print(f"Top 5 short: {top[-5:]}")
