"""
Module 1.5 — Resilience Layer
Retry-with-backoff + previous-day fallback cache + structured logging
for fragile external data sources (FRED, yfinance, IBKR).

Design principles:
  - Never crash the daily pipeline because of a transient API outage.
  - Always know whether today's signals were built from fresh or stale data.
  - Make every fallback observable in the logs and in the signal JSON.
"""
import functools
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# --- Cache directory ---
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logger setup (structured, written to logs/pipeline_YYYY-MM-DD.log) ---
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "aris") -> logging.Logger:
    """Return a configured logger that writes to both stdout and a daily file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # idempotent
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File
    log_path = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = get_logger()


# --- Retry decorator ---
def retry(
    attempts: int = 3,
    backoff_seconds: float = 5.0,
    exceptions: tuple = (Exception,),
):
    """
    Retry a callable on transient failure with exponential backoff.

    Args:
        attempts: total tries before giving up.
        backoff_seconds: base wait, doubles each attempt (5, 10, 20...).
        exceptions: which exception types to catch.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = backoff_seconds
            last_exc = None
            for i in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    log.warning(
                        f"{func.__name__} failed on attempt {i}/{attempts}: "
                        f"{type(e).__name__}: {e}"
                    )
                    if i < attempts:
                        time.sleep(wait)
                        wait *= 2
            # Out of attempts
            log.error(f"{func.__name__} exhausted {attempts} attempts. Re-raising.")
            raise last_exc
        return wrapper
    return decorator


# --- Pickle-based cache (handles pandas objects natively) ---
def cache_save(key: str, value: Any) -> None:
    """Persist a value to the cache directory under `key`."""
    path = CACHE_DIR / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(
                {"saved_at": datetime.now().isoformat(), "value": value}, f
            )
        log.info(f"cache_save: wrote {key} -> {path}")
    except Exception as e:
        log.warning(f"cache_save failed for {key}: {e}")


def cache_load(key: str) -> tuple[Any, str | None]:
    """
    Load a cached value. Returns (value, saved_at_iso) or (None, None) if absent.
    """
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None, None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload["value"], payload["saved_at"]
    except Exception as e:
        log.warning(f"cache_load failed for {key}: {e}")
        return None, None


# --- High-level wrapper: fetch with cache fallback ---
def fetch_with_fallback(
    fetch_fn: Callable,
    cache_key: str,
    *args,
    **kwargs,
) -> tuple[Any, str]:
    """
    Try to fetch fresh data. On failure, fall back to cached previous value.

    Returns:
        (value, source) where source is "fresh", "cache", or "missing".
    """
    try:
        value = fetch_fn(*args, **kwargs)
        cache_save(cache_key, value)
        return value, "fresh"
    except Exception as e:
        log.error(
            f"fetch_with_fallback: {fetch_fn.__name__} failed ({e}). "
            f"Trying cache for {cache_key}..."
        )
        cached, saved_at = cache_load(cache_key)
        if cached is not None:
            log.warning(
                f"fetch_with_fallback: USING STALE DATA for {cache_key} "
                f"(saved {saved_at})"
            )
            return cached, f"cache:{saved_at}"
        log.critical(
            f"fetch_with_fallback: NO CACHE AVAILABLE for {cache_key}. "
            f"Pipeline must halt or use safe defaults."
        )
        return None, "missing"


# --- Health record for the daily JSON ---
class HealthRecord:
    """Tracks the freshness/source/provider of every data input for one
    pipeline run, plus any cross-validation disagreements between dual sources.
    """

    def __init__(self):
        self.sources: dict[str, str] = {}      # input -> "fresh"|"cache:..."|"missing"
        self.providers: dict[str, str] = {}    # input -> "lseg"|"fred"|"yfinance"|...
        self.disagreements: dict[str, dict] = {}  # input -> {primary, secondary, pct_diff}
        self.errors: list[str] = []
        self.started_at = datetime.now().isoformat()

    def record(
        self,
        name: str,
        source: str,
        provider: str | None = None,
    ) -> None:
        self.sources[name] = source
        if provider is not None:
            self.providers[name] = provider
        if source.startswith("cache") or source == "missing":
            self.errors.append(f"{name}: {source}")

    def record_disagreement(self, name: str, info: dict) -> None:
        """Stash a single cross-validation disagreement."""
        self.disagreements[name] = info
        self.errors.append(
            f"{name}: cross-validation disagreement "
            f"({info.get('pct_diff', 0):.2%})"
        )

    def merge_router_disagreements(self, router_disagreements: dict) -> None:
        """Bulk-merge a DualSourceRouter.last_disagreements dict."""
        for k, v in router_disagreements.items():
            self.record_disagreement(k, v)

    def is_healthy(self) -> bool:
        return (
            all(s == "fresh" for s in self.sources.values())
            and not self.disagreements
        )

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "completed_at": datetime.now().isoformat(),
            "healthy": self.is_healthy(),
            "sources": self.sources,
            "providers": self.providers,
            "disagreements": self.disagreements,
            "degradations": self.errors,
        }


if __name__ == "__main__":
    # Smoke test
    log.info("resilience.py smoke test")

    @retry(attempts=3, backoff_seconds=1)
    def flaky(n=[0]):
        n[0] += 1
        if n[0] < 2:
            raise ConnectionError("simulated transient failure")
        return "success"

    print(flaky())

    cache_save("smoketest", {"hello": "world", "n": 42})
    val, ts = cache_load("smoketest")
    print(f"Loaded from cache: {val} (saved {ts})")

    h = HealthRecord()
    h.record("fred_ism", "fresh")
    h.record("yfinance_prices", "cache:2026-04-05T18:00:00")
    print(json.dumps(h.to_dict(), indent=2))
