"""
Module 1.5 — Resilience Layer
Retry-with-backoff + previous-day fallback cache + structured logging.
Phase 2: HealthRecord extended with providers + cross-validation disagreements.
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

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "aris") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_path = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = get_logger()


def retry(attempts: int = 3, backoff_seconds: float = 5.0, exceptions: tuple = (Exception,)):
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
                    log.warning(f"{func.__name__} failed on attempt {i}/{attempts}: {type(e).__name__}: {e}")
                    if i < attempts:
                        time.sleep(wait)
                        wait *= 2
            log.error(f"{func.__name__} exhausted {attempts} attempts. Re-raising.")
            raise last_exc
        return wrapper
    return decorator


def cache_save(key: str, value: Any) -> None:
    path = CACHE_DIR / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump({"saved_at": datetime.now().isoformat(), "value": value}, f)
        log.info(f"cache_save: wrote {key} -> {path}")
    except Exception as e:
        log.warning(f"cache_save failed for {key}: {e}")


def cache_load(key: str) -> tuple[Any, str | None]:
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


def fetch_with_fallback(fetch_fn: Callable, cache_key: str, *args, **kwargs) -> tuple[Any, str]:
    try:
        value = fetch_fn(*args, **kwargs)
        cache_save(cache_key, value)
        return value, "fresh"
    except Exception as e:
        log.error(f"fetch_with_fallback: {fetch_fn.__name__} failed ({e}). Trying cache for {cache_key}...")
        cached, saved_at = cache_load(cache_key)
        if cached is not None:
            log.warning(f"fetch_with_fallback: USING STALE DATA for {cache_key} (saved {saved_at})")
            return cached, f"cache:{saved_at}"
        log.critical(f"fetch_with_fallback: NO CACHE AVAILABLE for {cache_key}.")
        return None, "missing"


class HealthRecord:
    """Tracks freshness/source/provider of every data input + cross-validation disagreements."""

    def __init__(self):
        self.sources: dict[str, str] = {}
        self.providers: dict[str, str] = {}
        self.disagreements: dict[str, dict] = {}
        self.errors: list[str] = []
        self.started_at = datetime.now().isoformat()

    def record(self, name: str, source: str, provider: str | None = None) -> None:
        self.sources[name] = source
        if provider is not None:
            self.providers[name] = provider
        if source.startswith("cache") or source == "missing":
            self.errors.append(f"{name}: {source}")

    def record_disagreement(self, name: str, info: dict) -> None:
        self.disagreements[name] = info
        self.errors.append(f"{name}: cross-validation disagreement ({info.get('pct_diff', 0):.2%})")

    def merge_router_disagreements(self, router_disagreements: dict) -> None:
        for k, v in router_disagreements.items():
            self.record_disagreement(k, v)

    def is_healthy(self) -> bool:
        return all(s == "fresh" for s in self.sources.values()) and not self.disagreements

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
