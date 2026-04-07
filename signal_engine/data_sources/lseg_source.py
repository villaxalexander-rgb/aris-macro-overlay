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
            # lseg-data 2.x: passing app_key alone opens a desktop session by default
            ld.open_session(app_key=self.app_key)
        elif self.session_type == "platform":
            username = os.getenv("LSEG_USERNAME")
            password = os.getenv("LSEG_PASSWORD")
            if not (username and password):
                raise RuntimeError(
                    "LSEG_SESSION_TYPE=platform requires LSEG_USERNAME and "
                    "LSEG_PASSWORD env vars."
                )
            log.info("Opening LSEG platform session (headless RDP)")
            # lseg-data 2.x: build a platform session explicitly via Definition
            session = ld.session.platform.Definition(
                app_key=self.app_key,
                grant=ld.session.platform.GrantPassword(
                    username=username, password=password
                ),
            ).get_session()
            session.open()
            ld.session.set_default(session)
        else:
            raise ValueError(
                f"Unknown LSEG_SESSION_TYPE={self.session_type!r} "
                f"(expected 'desktop' or 'platform')"
            )

        # Handshake probe — lseg-data swallows connection errors until the
        # first real request. Do a trivial fetch so we fail loudly here
        # instead of mid-pipeline with a misleading "session opened" log.
        try:
            _probe = ld.get_history(
                universe="GCc1", fields=["TRDPRC_1"], count=1
            )
            if _probe is None or len(_probe) == 0:
                raise RuntimeError("handshake probe returned empty frame")
        except Exception as e:
            raise RuntimeError(
                f"LSEG handshake probe failed ({type(e).__name__}: {e}). "
                f"Is Workspace desktop running and signed in? "
                f"Check http://localhost:9000/api/status"
            ) from e

        self._opened = True
        log.info("LSEG session opened and handshake verified (probe=GCc1)")

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
            # No LSEG RIC (either unmapped or entitlement-excluded).
            # Raise NotImplementedError so DualSourceRouter treats this as
            # "LSEG cannot serve this series" and routes to the secondary
            # provider (FRED / yfinance) instead of triggering the cache
            # fallback + CRITICAL log spam.
            raise NotImplementedError(
                f"LSEG has no RIC for macro series '{series_name}' "
                f"(unmapped or not entitled on this account)"
            )

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
        """Latest VIX print. NOT ENTITLED on this LSEG account — router
        should fetch VIX via FRED (VIXCLS) primary, yfinance (^VIX) secondary.
        Kept here so the interface is symmetrical but raises immediately.
        """
        raise NotImplementedError(
            "VIX is not entitled on this LSEG account (verified 2026-04-06). "
            "Use FRED VIXCLS or yfinance ^VIX via DualSourceRouter."
        )


if __name__ == "__main__":
    # Smoke test (requires lseg-data installed and Workspace running)
    print("LSEG smoke test")
    src = LSEGSource()
    src.open()
    try:
        # 1. Front-month prices for the full commodity universe
        all_canon = [c for c, v in TICKERS.items() if v.get("lseg_ric")]
        prices = src.fetch_prices(all_canon, lookback_days=30)
        print(f"Prices: {prices.shape}  last row:")
        print(prices.tail(1).T)

        # 2. Full WTI curve
        curve = src.fetch_curve("CL")
        print(f"\nWTI curve: {len(curve)} contracts")
        print(curve.head(10))

        # 3. Risk-off via LSEG (MOVE + DXY — both entitled on this account)
        move = src.fetch_macro("move", lookback_months=1)
        print(f"\nMOVE (bond vol): last={move.dropna().iloc[-1]:.2f}")

        dxy = src.fetch_macro("dxy", lookback_months=1)
        print(f"DXY (dollar):    last={dxy.dropna().iloc[-1]:.3f}")

        # 4. VIX is not entitled — expected to raise NotImplementedError
        try:
            src.fetch_vix()
            print("VIX: unexpectedly succeeded?!")
        except NotImplementedError as e:
            print(f"\nVIX correctly rejected -> route via FRED/yfinance")
    finally:
        src.close()
