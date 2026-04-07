"""
FRED data source — wraps fredapi.

Free, official, used as the secondary macro source alongside LSEG.
"""
import os

import pandas as pd
from fredapi import Fred

from signal_engine.resilience import retry, fetch_with_fallback, log
from config.tickers import MACRO_SERIES


class FREDSource:
    """St. Louis FRED source. Free, requires API key."""

    name = "fred"

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY", "")
        if not self.api_key:
            log.warning("FRED_API_KEY not set — FRED source will fail")
        self._client = None

    def _client_lazy(self) -> Fred:
        if self._client is None:
            self._client = Fred(api_key=self.api_key)
        return self._client

    @retry(attempts=3, backoff_seconds=5)
    def _fetch_series_raw(
        self, series_id: str, lookback_months: int
    ) -> pd.Series:
        data = self._client_lazy().get_series(series_id)
        if data is None or len(data) == 0:
            raise ValueError(f"FRED returned empty series for {series_id}")
        return data.tail(lookback_months)

    def fetch_macro(
        self, series_name: str, lookback_months: int = 24
    ) -> pd.Series:
        """Fetch a macro series by canonical name (e.g. 'cpi_yoy', 'vix')."""
        if series_name not in MACRO_SERIES:
            raise KeyError(f"Unknown macro series: {series_name}")
        fred_id = MACRO_SERIES[series_name].get("fred")
        if not fred_id:
            raise ValueError(f"No FRED series id for {series_name}")

        cache_key = f"fred_{series_name}"

        def _fetch():
            return self._fetch_series_raw(fred_id, lookback_months)

        value, source = fetch_with_fallback(_fetch, cache_key)
        if value is None:
            raise RuntimeError(f"FRED unavailable for {series_name}")
        log.info(f"FRED {series_name}: source={source}, n={len(value)}")
        value.attrs["source"] = source
        value.attrs["provider"] = self.name
        value.attrs["series"] = series_name
        return value
