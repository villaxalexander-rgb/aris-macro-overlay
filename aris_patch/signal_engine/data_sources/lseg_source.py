"""
LSEG / Refinitiv data source — wraps the lseg-data Python library.

Supports two session types via env var LSEG_SESSION_TYPE:
    desktop   - uses Workspace desktop app on this machine (default for local dev)
    platform  - uses RDP credentials, headless (for Hetzner VPS)

Capabilities used by A.R.I.S:
    - fetch_prices(canonical_list, lookback_days) -> daily closes for continuous front-month
    - fetch_curve(canonical) -> snapshot of all listed contracts (full curve)
    - fetch_macro(series_name, lookback_months) -> macro time series (VIX, DXY, etc)
    - fetch_vix() -> latest VIX print

Auth env vars (set in .env):
    LSEG_APP_KEY              - Workspace App Key Generator output (required)
    LSEG_SESSION_TYPE         - "desktop" (default) or "platform"
    LSEG_USERNAME             - machine ID, only needed for platform session
    LSEG_PASSWORD             - password,    only needed for platform session

Note: this module imports lseg-data lazily inside __init__ so the rest of the
codebase still loads on machines that don't have the library installed
(e.g. CI, the sandbox, the VPS before provisioning).
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from signal_engine.resilience import retry, fetch_with_fallback, log
from config.tickers import TICKERS, MACRO_SERIES, lseg_to_canonical


class LSEGSource:
    """Refinitiv/LSEG data client built on lseg-data."""

    name = "lseg"

    def __init__(self):
        self.app_key = os.getenv("LSEG_APP_KEY", "")
        self.session_type = os.getenv("LSEG_SESSION_TYPE", "desktop").lower()
        self._opened = False
        self._ld = None  # the lseg-data module, imported lazily

        if not self.app_key:
            raise RuntimeError(
                "LSEG_APP_KEY not set. Generate one in Workspace via the "
                "App Key Generator (APPKEY) tool and add it to .env."
            )

    # ---------- session lifecycle ----------
    def open(self) -> None:
        """Open an LSEG session. Idempotent."""
        if self._opened:
            return

        try:
            import lseg.data as ld
        except ImportError as e:
            raise ImportError(
                "lseg-data not installed. Run: pip install lseg-data"
            ) from e

        self._ld = ld

        if self.session_type == "desktop":
            log.info("Opening LSEG desktop session (Workspace must be running)")
            ld.open_session(
                config_name=None,
                name="workspace",
                app_key=self.app_key,
            )
        elif self.session_type == "platform":
            username = os.getenv("LSEG_USERNAME")
            password = os.getenv("LSEG_PASSWORD")
            if not (username and password):
                raise RuntimeError(
                    "LSEG_SESSION_TYPE=platform requires LSEG_USERNAME and "
                    "LSEG_PASSWORD env vars."
                )
            log.info("Opening LSEG platform session (headless RDP)")
            ld.open_session(
                config_name=None,
                name="rdp",
                app_key=self.app_key,
            )
        else:
            raise ValueError(
                f"Unknown LSEG_SESSION_TYPE={self.session_type!r} "
                f"(expected 'desktop' or 'platform')"
            )

        self._opened = True
        log.info("LSEG session opened successfully")

    def close(self) -> None:
        if self._opened and self._ld is not None:
            try:
                self._ld.close_session()
            except Exception as e:
                log.warning(f"LSEG close_session failed: {e}")
        self._opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ---------- fetchers (raw, retry-decorated) ----------
    @retry(attempts=3, backoff_seconds=5)
    def _get_history_raw(
        self,
        rics: list[str],
        fields: list[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        """Wrapped lseg-data history call. Retries on transient errors."""
        if not self._opened:
            self.open()
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        df = self._ld.get_history(
            universe=rics,
            fields=fields,
            interval="1D",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            raise ValueError(f"LSEG returned empty frame for {rics}")
        return df

    @retry(attempts=3, backoff_seconds=5)
    def _get_data_raw(
        self,
        rics: list[str],
        fields: list[str],
    ) -> pd.DataFrame:
        """Snapshot data (no history). Used for full-curve queries."""
        if not self._opened:
            self.open()
        df = self._ld.get_data(universe=rics, fields=fields)
        if df is None or df.empty:
            raise ValueError(f"LSEG returned empty snapshot for {rics}")
        return df

    # ---------- public interface ----------
    def fetch_prices(
        self,
        canonical: list[str],
        lookback_days: int = 365 * 5,
    ) -> pd.DataFrame:
        """
        Daily close prices for the continuous front-month of each canonical asset.
        Returns a DataFrame with columns = canonical names (NOT RICs).
        """
        rics = [TICKERS[c]["lseg_ric"] for c in canonical if TICKERS[c].get("lseg_ric")]
        cache_key = f"lseg_prices_{lookback_days}d"

        def _fetch():
            df = self._get_history_raw(rics, ["TRDPRC_1"], lookback_days)
            # lseg-data returns a multi-level column frame; flatten and rename
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            # Map RIC -> canonical
            rename = {ric: lseg_to_canonical(ric) or ric for ric in df.columns}
            df = df.rename(columns=rename)
            return df

        value, source = fetch_with_fallback(_fetch, cache_key)
        if value is None:
            raise RuntimeError("LSEG prices unavailable and no cache")
        log.info(f"LSEG prices: source={source}, shape={value.shape}")
        value.attrs["source"] = source
        value.attrs["provider"] = self.name
        return value

    def fetch_curve(self, canonical: str) -> pd.DataFrame:
        """
        Snapshot of the full futures curve for one canonical asset.
        Returns columns: contract, expiry, settle, volume, open_interest.
        """
        chain_ric = TICKERS[canonical].get("lseg_chain")
        if not chain_ric:
            raise ValueError(f"No LSEG chain RIC defined for {canonical}")

        cache_key = f"lseg_curve_{canonical}"

        def _fetch():
            df = self._get_data_raw(
                [chain_ric],
                ["EXPIR_DATE", "SETTLE", "ACVOL_UNS", "OPINT_1"],
            )
            df.columns = [str(c).lower() for c in df.columns]
            df = df.rename(
                columns={
                    "instrument": "contract",
                    "expir_date": "expiry",
                    "acvol_uns": "volume",
                    "opint_1": "open_interest",
                }
            )
            # Drop expired / no-volume contracts
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
                df = df[df["expiry"] >= pd.Timestamp.now()]
            df = df.sort_values("expiry").reset_index(drop=True)
            return df

        value, source = fetch_with_fallback(_fetch, cache_key)
        if value is None:
            raise RuntimeError(f"LSEG curve unavailable for {canonical}")
        value.attrs["source"] = source
        value.attrs["provider"] = self.name
        value.attrs["asset"] = canonical
        return value

    def fetch_macro(
        self, series_name: str, lookback_months: int = 24
    ) -> pd.Series:
        """Fetch a macro time series (VIX, DXY, ISM, CPI, etc)."""
        if series_name not in MACRO_SERIES:
            raise KeyError(f"Unknown macro series: {series_name}")
        ric = MACRO_SERIES[series_name].get("lseg")
        if not ric:
            raise ValueError(f"No LSEG RIC for macro series {series_name}")

        cache_key = f"lseg_macro_{series_name}"

        def _fetch():
            df = self._get_history_raw([ric], ["TRDPRC_1"], lookback_months * 31)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return df.iloc[:, 0]

        value, source = fetch_with_fallback(_fetch, cache_key)
        if value is None:
            raise RuntimeError(f"LSEG macro unavailable for {series_name}")
        value.attrs["source"] = source
        value.attrs["provider"] = self.name
        value.attrs["series"] = series_name
        return value

    def fetch_vix(self) -> float:
        """Latest VIX print, used by the risk halt logic."""
        s = self.fetch_macro("vix", lookback_months=1)
        return float(s.dropna().iloc[-1])


if __name__ == "__main__":
    # Smoke test (requires lseg-data installed and Workspace running)
    print("LSEG smoke test")
    src = LSEGSource()
    src.open()
    try:
        cl = src.fetch_prices(["CL"], lookback_days=30)
        print(f"WTI prices: {cl.shape}, last={cl.iloc[-1].values}")

        curve = src.fetch_curve("CL")
        print(f"WTI curve: {len(curve)} contracts")
        print(curve.head(10))

        vix = src.fetch_vix()
        print(f"VIX: {vix}")
    finally:
        src.close()
