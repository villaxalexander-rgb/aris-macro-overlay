#!/usr/bin/env python3
"""
A.R.I.S — Precious Metals LSEG RIC Diagnostic

Tests whether PA (Palladium), PL (Platinum), SI (Silver) RICs are
entitled and returning data on your LSEG session.  Tries the configured
RICs first, then a matrix of known alternatives to find working ones.

Usage (on Hetzner, with .env sourced):
    cd ~/Desktop/aris-macro-overlay
    source venv/bin/activate
    python3 scripts/diagnose_precious_metals.py

Requires: lseg-data, python-dotenv, pandas
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta

# Load .env so LSEG creds are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # ok if not installed — user may have exported env vars manually

import pandas as pd


# ── RIC candidates to test ──────────────────────────────────────────
# For each canonical asset, we try:
#   1. The configured RIC from tickers.py (e.g. PAc1)
#   2. Alternative continuous-contract RICs
#   3. Exchange-specific or cross-currency RICs
#   4. Spot / fix RICs (not futures but useful as fallback indicator)
CANDIDATES = {
    "PA": {
        "description": "Palladium",
        "configured_ric": "PAc1",
        "alternatives": [
            # NYMEX/CME continuous
            "PAc1", "PAc2",
            # Explicit exchange prefix
            "0#PA:",           # chain RIC — should expand to listed contracts
            # Spot / OTC
            "XPD=",            # spot palladium USD
            "XPDUSD=R",        # Reuters spot
            # London fix
            "PALD",            # London PM fix
            "LPMPDSET",        # London Platinum & Palladium Market fix
            # ETF proxy (not ideal but confirms data access)
            "PALL.P",          # Aberdeen Palladium ETF
        ],
    },
    "PL": {
        "description": "Platinum",
        "configured_ric": "PLc1",
        "alternatives": [
            "PLc1", "PLc2",
            "0#PL:",
            "XPT=",            # spot platinum USD
            "XPTUSD=R",
            "PLAT",            # London PM fix
            "LPMPTSET",        # London fix
            "PPLT.P",          # Aberdeen Platinum ETF
        ],
    },
    "SI": {
        "description": "Silver",
        "configured_ric": "SIc1",
        "alternatives": [
            "SIc1", "SIc2",
            "0#SI:",
            "XAG=",            # spot silver USD
            "XAGUSD=R",
            "SILVER",          # London fix
            "LBMASILV",        # LBMA silver fix
            "SLV.P",           # iShares Silver ETF
        ],
    },
}


# ── LSEG session setup ──────────────────────────────────────────────
def open_session():
    """Open an LSEG session using env vars, same logic as lseg_source.py."""
    try:
        import lseg.data as ld
    except ImportError:
        print("ERROR: lseg-data not installed. Run: pip install lseg-data")
        sys.exit(1)

    app_key = os.getenv("LSEG_APP_KEY", "")
    session_type = os.getenv("LSEG_SESSION_TYPE", "desktop").lower()

    if not app_key:
        print("ERROR: LSEG_APP_KEY not set in environment.")
        sys.exit(1)

    print(f"Opening LSEG session (type={session_type})...")
    if session_type == "desktop":
        ld.open_session(app_key=app_key)
    elif session_type == "platform":
        username = os.getenv("LSEG_USERNAME")
        password = os.getenv("LSEG_PASSWORD")
        if not (username and password):
            print("ERROR: LSEG_USERNAME and LSEG_PASSWORD required for platform session.")
            sys.exit(1)
        session = ld.session.platform.Definition(
            app_key=app_key,
            grant=ld.session.platform.GrantPassword(
                username=username, password=password,
            ),
        ).get_session()
        session.open()
        ld.session.set_default(session)
    else:
        print(f"ERROR: Unknown LSEG_SESSION_TYPE={session_type!r}")
        sys.exit(1)

    print("Session opened.\n")
    return ld


# ── Test functions ──────────────────────────────────────────────────
def test_history(ld, ric: str, days: int = 30) -> dict:
    """Try get_history for a single RIC. Returns result dict."""
    result = {"ric": ric, "type": "history", "ok": False, "rows": 0, "error": None, "sample": None}
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = ld.get_history(
            universe=ric,
            fields=["TRDPRC_1", "HIGH_1", "LOW_1", "ACVOL_UNS"],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if df is not None and not df.empty:
            result["ok"] = True
            result["rows"] = len(df)
            # Grab last 3 rows as sample
            tail = df.tail(3)
            result["sample"] = tail.to_string()
        else:
            result["error"] = "Empty DataFrame returned"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def test_snapshot(ld, ric: str) -> dict:
    """Try get_data (real-time snapshot) for a single RIC."""
    result = {"ric": ric, "type": "snapshot", "ok": False, "error": None, "data": None}
    try:
        df = ld.get_data(
            universe=ric,
            fields=["TRDPRC_1", "BID", "ASK", "ACVOL_UNS", "CF_NAME"],
        )
        if df is not None and not df.empty:
            result["ok"] = True
            result["data"] = df.to_string()
        else:
            result["error"] = "Empty DataFrame returned"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def test_chain(ld, chain_ric: str) -> dict:
    """Try to expand a chain RIC (e.g. 0#PA:) to see listed contracts."""
    result = {"ric": chain_ric, "type": "chain", "ok": False, "error": None, "contracts": []}
    try:
        df = ld.get_data(universe=chain_ric, fields=["CF_NAME", "TRDPRC_1", "EXPIR_DATE"])
        if df is not None and not df.empty:
            result["ok"] = True
            result["contracts"] = df.index.tolist() if hasattr(df.index, 'tolist') else list(df.iloc[:, 0])
            result["count"] = len(df)
        else:
            result["error"] = "Empty chain expansion"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


# ── Main diagnostic ────────────────────────────────────────────────
def main():
    ld = open_session()

    # First, verify GC works (known-good precious metal as control)
    print("── Control test: GCc1 (Gold) ──")
    gc_result = test_history(ld, "GCc1", days=10)
    if gc_result["ok"]:
        print(f"  ✓ GCc1 returned {gc_result['rows']} rows — session is healthy")
    else:
        print(f"  ✗ GCc1 FAILED: {gc_result['error']}")
        print("  Session may be broken. Fix this before investigating PA/PL/SI.")
        sys.exit(1)
    print()

    # Test each problem asset
    recommendations = {}

    for canonical, info in CANDIDATES.items():
        print(f"══ {canonical}: {info['description']} ══")
        print(f"   Configured RIC: {info['configured_ric']}")
        print()

        working_history = []
        working_snapshot = []

        for ric in info["alternatives"]:
            # Skip chain RICs for history test
            if ric.startswith("0#"):
                chain = test_chain(ld, ric)
                status = "✓" if chain["ok"] else "✗"
                detail = f"{chain.get('count',0)} contracts" if chain["ok"] else chain["error"]
                print(f"   {status} CHAIN   {ric:<16} {detail}")
                if chain["ok"]:
                    working_snapshot.append(ric)
                time.sleep(0.5)
                continue

            # History test
            hist = test_history(ld, ric, days=30)
            status = "✓" if hist["ok"] else "✗"
            detail = f"{hist['rows']} rows" if hist["ok"] else hist["error"]
            print(f"   {status} HIST    {ric:<16} {detail}")
            if hist["ok"]:
                working_history.append(ric)

            # Snapshot test
            snap = test_snapshot(ld, ric)
            status = "✓" if snap["ok"] else "✗"
            print(f"   {status} SNAP    {ric:<16} {'data ok' if snap['ok'] else snap['error']}")
            if snap["ok"]:
                working_snapshot.append(ric)

            time.sleep(0.5)  # rate-limit courtesy

        # Recommendation
        print()
        if info["configured_ric"] in working_history:
            rec = f"✓ Configured RIC {info['configured_ric']} works — no change needed"
        elif working_history:
            best = working_history[0]
            rec = f"⚠ {info['configured_ric']} broken — RECOMMEND switching to {best}"
        elif working_snapshot:
            best = working_snapshot[0]
            rec = f"⚠ No history RIC works — snapshot-only via {best} (may need spot fallback)"
        else:
            rec = f"✗ NO working RIC found — asset may not be entitled. Contact LSEG support."

        recommendations[canonical] = rec
        print(f"   RECOMMENDATION: {rec}")
        print()

    # Final summary
    print("═══════════════════════════════════════════════════════════════")
    print(" SUMMARY — Precious Metals RIC Diagnostic")
    print("═══════════════════════════════════════════════════════════════")
    for canonical, rec in recommendations.items():
        print(f"  {canonical}: {rec}")
    print()

    # Generate tickers.py patch suggestion if any need changing
    changes_needed = {c: r for c, r in recommendations.items() if "switching to" in r}
    if changes_needed:
        print("── Suggested tickers.py patch ──")
        for canonical, rec in changes_needed.items():
            new_ric = rec.split("switching to ")[1]
            print(f'  "{canonical}": {{"lseg_ric": "{new_ric}", ...}}')
        print()

    # Close session
    try:
        ld.close_session()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
