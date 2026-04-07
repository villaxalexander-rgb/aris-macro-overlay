"""
Dual-source router with cross-validation.

Strategy:
    1. Call both providers (LSEG primary, yfinance/FRED secondary)
    2. If primary succeeds, use its values for the live signal
    3. If primary fails, fall back to secondary (resilience layer already
       handles single-source caching, so this is the second line of defense)
    4. If both succeed, compare the latest closes per asset. Any column
       where the two sources disagree by more than DUAL_SOURCE_DISAGREEMENT_PCT
       gets logged as a data-quality WARNING and recorded in the HealthRecord
       so the daily JSON tells us when something is fishy.

The router exposes the same fetch_prices / fetch_macro signature as the
underlying sources so it can be dropped into the existing pipeline.
"""
from typing import Optional

import pandas as pd

from signal_engine.data_sources import (
    YFinanceSource,
    FREDSource,
    LSEGSource,
    _LSEG_AVAILABLE,
)
from signal_engine.resilience import log
from config.settings import (
    PRIMARY_PRICE_SOURCE,
    PRIMARY_MACRO_SOURCE,
    DUAL_SOURCE_DISAGREEMENT_PCT,
)


class DualSourceRouter:
    """
    Calls both providers in parallel, picks primary, cross-validates.
    Drop-in replacement for any single source.

    The router itself is stateful — it keeps a `last_disagreements` dict
    from the most recent call so the pipeline can stuff it into HealthRecord.
    """

    def __init__(
        self,
        primary_price: str = PRIMARY_PRICE_SOURCE,
        primary_macro: str = PRIMARY_MACRO_SOURCE,
    ):
        self.primary_price = primary_price
        self.primary_macro = primary_macro
        self.yf = YFinanceSource()
        self.fred = FREDSource()
        self.lseg: Optional[LSEGSource] = None
        if _LSEG_AVAILABLE:
            try:
                self.lseg = LSEGSource()
            except Exception as e:
                log.warning(f"LSEG not initialized: {e}. Falling back to yfinance/FRED.")
                self.lseg = None

        # Diagnostic state, refreshed on every call
        self.last_disagreements: dict[str, dict] = {}
        self.last_used_source: dict[str, str] = {}

    # ---------- prices ----------
    def fetch_prices(
        self, canonical: list[str], lookback_days: int = 365 * 5
    ) -> pd.DataFrame:
        """
        Fetch from both providers, cross-validate, return primary.
        On primary failure, fall back to secondary automatically.
        """
        primary_df: Optional[pd.DataFrame] = None
        secondary_df: Optional[pd.DataFrame] = None

        if self.primary_price == "lseg" and self.lseg is not None:
            try:
                primary_df = self.lseg.fetch_prices(canonical, lookback_days)
            except Exception as e:
                log.error(f"LSEG primary prices failed: {e}")
            try:
                secondary_df = self.yf.fetch_prices(canonical, lookback_days)
            except Exception as e:
                log.error(f"yfinance secondary prices failed: {e}")
        else:
            try:
                primary_df = self.yf.fetch_prices(canonical, lookback_days)
            except Exception as e:
                log.error(f"yfinance primary prices failed: {e}")
            if self.lseg is not None:
                try:
                    secondary_df = self.lseg.fetch_prices(canonical, lookback_days)
                except Exception as e:
                    log.error(f"LSEG secondary prices failed: {e}")

        # Pick whichever is available
        chosen = primary_df if primary_df is not None else secondary_df
        if chosen is None:
            raise RuntimeError(
                "Both primary and secondary price sources failed. "
                "Pipeline cannot proceed without prices."
            )

        # Cross-validate latest closes if both succeeded
        if primary_df is not None and secondary_df is not None:
            self._cross_validate_prices(primary_df, secondary_df)
            self.last_used_source["prices"] = "primary+validated"
        elif primary_df is not None:
            log.warning("Prices: primary only (secondary unavailable)")
            self.last_used_source["prices"] = "primary_only"
        else:
            log.warning("Prices: secondary fallback (primary failed)")
            self.last_used_source["prices"] = "secondary_fallback"

        return chosen

    def _cross_validate_prices(
        self, primary: pd.DataFrame, secondary: pd.DataFrame
    ) -> None:
        """
        Compare the latest close per asset between primary and secondary.
        Anything diverging more than DUAL_SOURCE_DISAGREEMENT_PCT gets flagged.
        """
        self.last_disagreements = {}
        common_cols = sorted(set(primary.columns) & set(secondary.columns))
        if not common_cols:
            log.warning("Cross-validation: no overlapping columns between sources")
            return

        try:
            p_last = primary[common_cols].dropna(how="all").iloc[-1]
            s_last = secondary[common_cols].dropna(how="all").iloc[-1]
        except IndexError:
            log.warning("Cross-validation: not enough rows to compare")
            return

        disagreements = 0
        for col in common_cols:
            p_val = p_last.get(col)
            s_val = s_last.get(col)
            if pd.isna(p_val) or pd.isna(s_val) or p_val == 0:
                continue
            pct_diff = abs(p_val - s_val) / abs(p_val)
            if pct_diff > DUAL_SOURCE_DISAGREEMENT_PCT:
                disagreements += 1
                self.last_disagreements[col] = {
                    "primary": float(p_val),
                    "secondary": float(s_val),
                    "pct_diff": float(pct_diff),
                }
                log.warning(
                    f"Cross-validation DISAGREEMENT on {col}: "
                    f"primary={p_val:.4f} secondary={s_val:.4f} "
                    f"({pct_diff:.2%})"
                )

        if disagreements == 0:
            log.info(
                f"Cross-validation OK across {len(common_cols)} assets "
                f"(threshold {DUAL_SOURCE_DISAGREEMENT_PCT:.0%})"
            )
        else:
            log.warning(
                f"Cross-validation: {disagreements}/{len(common_cols)} assets "
                f"diverged beyond threshold"
            )

    # ---------- macro ----------
    def fetch_macro(
        self, series_name: str, lookback_months: int = 24
    ) -> pd.Series:
        """
        Fetch a macro series from primary, fall back to secondary on failure.
        Cross-validation on the most recent observation.
        """
        primary_s: Optional[pd.Series] = None
        secondary_s: Optional[pd.Series] = None

        if self.primary_macro == "lseg" and self.lseg is not None:
            try:
                primary_s = self.lseg.fetch_macro(series_name, lookback_months)
            except Exception as e:
                log.error(f"LSEG primary macro {series_name} failed: {e}")
            try:
                secondary_s = self.fred.fetch_macro(series_name, lookback_months)
            except Exception as e:
                log.warning(f"FRED secondary macro {series_name} failed: {e}")
        else:
            try:
                primary_s = self.fred.fetch_macro(series_name, lookback_months)
            except Exception as e:
                log.error(f"FRED primary macro {series_name} failed: {e}")
            if self.lseg is not None:
                try:
                    secondary_s = self.lseg.fetch_macro(series_name, lookback_months)
                except Exception as e:
                    log.warning(f"LSEG secondary macro {series_name} failed: {e}")

        chosen = primary_s if primary_s is not None else secondary_s
        if chosen is None:
            raise RuntimeError(
                f"Both sources failed for macro series {series_name}"
            )

        # Cross-validate latest value
        if primary_s is not None and secondary_s is not None:
            try:
                p_last = float(primary_s.dropna().iloc[-1])
                s_last = float(secondary_s.dropna().iloc[-1])
                if p_last != 0:
                    pct_diff = abs(p_last - s_last) / abs(p_last)
                    if pct_diff > DUAL_SOURCE_DISAGREEMENT_PCT:
                        log.warning(
                            f"Macro DISAGREEMENT on {series_name}: "
                            f"primary={p_last:.4f} secondary={s_last:.4f} "
                            f"({pct_diff:.2%})"
                        )
                        self.last_disagreements[f"macro:{series_name}"] = {
                            "primary": p_last,
                            "secondary": s_last,
                            "pct_diff": float(pct_diff),
                        }
            except Exception as e:
                log.warning(f"Macro cross-validation failed for {series_name}: {e}")

        return chosen

    # ---------- curves (LSEG only) ----------
    def fetch_curve(self, canonical: str) -> pd.DataFrame:
        """Full futures curve. LSEG-only — yfinance has no curve data."""
        if self.lseg is None:
            raise RuntimeError(
                "Full curve data requires LSEG. Set LSEG_APP_KEY and install lseg-data."
            )
        return self.lseg.fetch_curve(canonical)

    # ---------- VIX shortcut ----------
    def fetch_vix(self) -> float:
        """Latest VIX print, used by the risk halt logic."""
        s = self.fetch_macro("vix", lookback_months=1)
        return float(s.dropna().iloc[-1])
