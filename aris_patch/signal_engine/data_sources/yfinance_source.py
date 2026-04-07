"""
yfinance data source — wraps Yahoo Finance via the yfinance library.

Free, fragile, no auth required. Used as the secondary source for prices
and as the historical-backfill source for backtests when LSEG license
restricts deep history.
"""
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from signal_engine.resilience import retry, fetch_with_fallback, log
from config.tickers import TICKERS, yf_to_canonical


class YFinanceSource:
    """Yahoo Finance source. Free but fragile."""

    name = "yfinance"

    @retry(attempts=3, backoff_seconds=10)
    def _fetch_raw(
        self, yf_tickers: list[str], lookback_days: int
    ) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        data = yf.download(
            yf_tickers, start=start, end=end, progress=False, auto_adjust=True
        )["Close"]
        if data is None or data.empty:
            raise ValueError("yfinance returned empty frame")
        nan_pct = float(data.isna().mean().mean())
        if nan_pct > 0.5:
            raise ValueError(f"yfinance returned {nan_pct:.0%} NaN — degraded")
        if len(data) < 30:
            raise ValueError(f"yfinance returned only {len(data)} rows")
        return data

    def fetch_prices(
        self, canonical: list[str], lookback_days: int = 365 * 5
    ) -> pd.DataFrame:
        """
        Daily close prices for the requested canonical assets.
        Returns a DataFrame with columns = canonical names.
        """
        yf_tickers = [TICKERS[c]["yf_ticker"] for c in canonical]
        cache_key = f"yf_prices_{lookback_days}d"

        def _fetch():
            df = self._fetch_raw(yf_tickers, lookback_days)
            # Map yf ticker -> canonical
            rename = {t: yf_to_canonical(t) or t for t in df.columns}
            return df.rename(columns=rename)

        value, source = fetch_with_fallback(_fetch, cache_key)
        if value is None:
            raise RuntimeError("yfinance unavailable and no cache")
        log.info(
            f"yfinance prices: source={source}, shape={value.shape}, "
            f"nan_pct={value.isna().mean().mean():.1%}"
        )
        value.attrs["source"] = source
        value.attrs["provider"] = self.name
        return value

    def fetch_curve(self, canonical: str) -> pd.DataFrame:
        """yfinance does not expose futures curves. Always raises."""
        raise NotImplementedError(
            "yfinance has no futures curve data — use LSEGSource for curves"
        )
