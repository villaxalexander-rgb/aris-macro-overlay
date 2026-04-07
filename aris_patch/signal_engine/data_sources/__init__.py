"""
A.R.I.S Data Sources Package
============================

One file per upstream provider. Every source implements the same minimal
interface so the dual_source router can substitute one for another, or call
both and reconcile.

Common interface
----------------
    fetch_prices(tickers: list[str], lookback_days: int) -> pd.DataFrame
        Daily close prices, columns = canonical asset names (not provider RICs).
        DataFrame.attrs["source"] is set to the provider name.

    fetch_macro(series_id: str, lookback_months: int) -> pd.Series
        Macro time series (ISM, CPI, etc). attrs["source"] set.

    fetch_curve(asset: str) -> pd.DataFrame    [optional]
        Snapshot of the full futures curve for one asset. Columns:
        contract, expiry, settle, volume, open_interest.

    fetch_vix() -> float                       [optional]
        Latest VIX print. Used by the risk halt logic.

Provider modules
----------------
    yfinance_source — free, fragile, only daily closes via =F tickers
    fred_source     — free FRED API, macro series only
    lseg_source     — Refinitiv/LSEG via lseg-data, real curves + intraday
"""
from .yfinance_source import YFinanceSource
from .fred_source import FREDSource

# LSEG is optional — only import if the library is installed.
try:
    from .lseg_source import LSEGSource
    _LSEG_AVAILABLE = True
except ImportError:
    LSEGSource = None  # type: ignore
    _LSEG_AVAILABLE = False

__all__ = ["YFinanceSource", "FREDSource", "LSEGSource", "_LSEG_AVAILABLE"]
