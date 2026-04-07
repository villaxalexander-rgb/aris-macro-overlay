"""
Ticker mapping table — single source of truth for asset identity across providers.

Each asset has:
    canonical:    short name used internally by A.R.I.S (e.g. "CL")
    name:         human-readable
    sector:       energy / metals / precious_metals / agriculture / livestock
    yf_ticker:    yfinance symbol (with =F suffix for futures)
    lseg_ric:     LSEG continuous front-month RIC (e.g. CLc1)
    lseg_chain:   LSEG chain RIC for the full futures curve (e.g. 0#CL:)

Note on LSEG conventions
------------------------
- "CLc1", "CLc2", etc. = continuous contracts where c1 is the front month,
  c2 the next listed contract, and so on. These are stitched series, no
  rollover gaps.
- "0#CL:" expands to all currently listed individual contracts (e.g.
  CLZ6, CLF7, CLG7...) with their actual settle prices, useful for
  full-curve carry calculations.
- ICE softs (KC, SB, CC, CT) and LME metals use exchange-prefixed RICs:
  e.g. "LCOc1" for Brent, "LCAc3" for LME aluminium 3-month.

If an asset is unavailable from a provider, leave that field as None and the
dual_source router will fall back to the other provider automatically.
"""

# canonical -> mapping dict
TICKERS: dict[str, dict] = {
    # ============ ENERGY ============
    "CL": {
        "name": "WTI Crude Oil",
        "sector": "energy",
        "yf_ticker": "CL=F",
        "lseg_ric": "CLc1",
        "lseg_chain": "0#CL:",
    },
    "BZ": {
        "name": "Brent Crude Oil",
        "sector": "energy",
        "yf_ticker": "BZ=F",
        "lseg_ric": "LCOc1",
        "lseg_chain": "0#LCO:",
    },
    "NG": {
        "name": "Henry Hub Natural Gas",
        "sector": "energy",
        "yf_ticker": "NG=F",
        "lseg_ric": "NGc1",
        "lseg_chain": "0#NG:",
    },
    "HO": {
        "name": "NY Heating Oil",
        "sector": "energy",
        "yf_ticker": "HO=F",
        "lseg_ric": "HOc1",
        "lseg_chain": "0#HO:",
    },
    "RB": {
        "name": "RBOB Gasoline",
        "sector": "energy",
        "yf_ticker": "RB=F",
        "lseg_ric": "RBc1",
        "lseg_chain": "0#RB:",
    },

    # ============ PRECIOUS METALS ============
    "GC": {
        "name": "Gold",
        "sector": "precious_metals",
        "yf_ticker": "GC=F",
        "lseg_ric": "GCc1",
        "lseg_chain": "0#GC:",
    },
    "SI": {
        "name": "Silver",
        "sector": "precious_metals",
        "yf_ticker": "SI=F",
        "lseg_ric": "SIc1",
        "lseg_chain": "0#SI:",
    },
    "PL": {
        "name": "Platinum",
        "sector": "precious_metals",
        "yf_ticker": "PL=F",
        "lseg_ric": "PLc1",
        "lseg_chain": "0#PL:",
    },
    "PA": {
        "name": "Palladium",
        "sector": "precious_metals",
        "yf_ticker": "PA=F",
        "lseg_ric": "PAc1",
        "lseg_chain": "0#PA:",
    },

    # ============ INDUSTRIAL METALS ============
    "HG": {
        "name": "COMEX Copper",
        "sector": "metals",
        "yf_ticker": "HG=F",
        "lseg_ric": "HGc1",
        "lseg_chain": "0#HG:",
    },

    # ============ GRAINS / OILSEEDS ============
    "ZC": {
        "name": "Corn",
        "sector": "agriculture",
        "yf_ticker": "ZC=F",
        "lseg_ric": "Cc1",
        "lseg_chain": "0#C:",
    },
    "ZW": {
        "name": "Chicago Wheat",
        "sector": "agriculture",
        "yf_ticker": "ZW=F",
        "lseg_ric": "Wc1",
        "lseg_chain": "0#W:",
    },
    "ZS": {
        "name": "Soybeans",
        "sector": "agriculture",
        "yf_ticker": "ZS=F",
        "lseg_ric": "Sc1",
        "lseg_chain": "0#S:",
    },
    "ZM": {
        "name": "Soybean Meal",
        "sector": "agriculture",
        "yf_ticker": "ZM=F",
        "lseg_ric": "SMc1",
        "lseg_chain": "0#SM:",
    },
    "ZL": {
        "name": "Soybean Oil",
        "sector": "agriculture",
        "yf_ticker": "ZL=F",
        "lseg_ric": "BOc1",
        "lseg_chain": "0#BO:",
    },

    # ============ SOFTS ============
    "CT": {
        "name": "Cotton",
        "sector": "agriculture",
        "yf_ticker": "CT=F",
        "lseg_ric": "CTc1",
        "lseg_chain": "0#CT:",
    },
    "KC": {
        "name": "Arabica Coffee",
        "sector": "agriculture",
        "yf_ticker": "KC=F",
        "lseg_ric": "KCc1",
        "lseg_chain": "0#KC:",
    },
    "SB": {
        "name": "Sugar #11",
        "sector": "agriculture",
        "yf_ticker": "SB=F",
        "lseg_ric": "SBc1",
        "lseg_chain": "0#SB:",
    },
    "CC": {
        "name": "Cocoa",
        "sector": "agriculture",
        "yf_ticker": "CC=F",
        "lseg_ric": "CCc1",
        "lseg_chain": "0#CC:",
    },

    # ============ LIVESTOCK ============
    "LE": {
        "name": "Live Cattle",
        "sector": "livestock",
        "yf_ticker": "LE=F",
        "lseg_ric": "LCc1",
        "lseg_chain": "0#LC:",
    },
    "GF": {
        "name": "Feeder Cattle",
        "sector": "livestock",
        "yf_ticker": "GF=F",
        "lseg_ric": "FCc1",
        "lseg_chain": "0#FC:",
    },
    "HE": {
        "name": "Lean Hogs",
        "sector": "livestock",
        "yf_ticker": "HE=F",
        "lseg_ric": "LHc1",
        "lseg_chain": "0#LH:",
    },
}

# ============ MACRO SERIES (ISM, CPI, VIX, DXY, HY OAS, MOVE) ============
# Each macro series maps a canonical name to FRED + LSEG identifiers.
MACRO_SERIES: dict[str, dict] = {
    "ism_manufacturing": {
        "name": "ISM Manufacturing PMI (proxy: Manufacturing Employment)",
        "fred": "MANEMP",
        "lseg": "USPMI=ECI",  # ISM Mfg headline
    },
    "cpi_yoy": {
        "name": "US CPI All Items YoY",
        "fred": "CPIAUCSL",
        "lseg": "USCPI=ECI",
    },
    "vix": {
        "name": "CBOE Volatility Index",
        "fred": "VIXCLS",
        "lseg": ".VIX",
    },
    "move": {
        "name": "MOVE Index (rates vol)",
        "fred": None,
        "lseg": ".MOVE",
    },
    "dxy": {
        "name": "US Dollar Index",
        "fred": "DTWEXBGS",
        "lseg": ".DXY",
    },
    "hy_oas": {
        "name": "ICE BofA US High Yield OAS",
        "fred": "BAMLH0A0HYM2",
        "lseg": ".MERH0A0",
    },
}


# ---------- Convenience accessors ----------
def get_canonical_list() -> list[str]:
    """All canonical commodity names."""
    return list(TICKERS.keys())


def get_yf_tickers() -> list[str]:
    """Return all yfinance tickers in canonical order."""
    return [v["yf_ticker"] for v in TICKERS.values()]


def get_lseg_rics(continuous: bool = True) -> list[str]:
    """Return all LSEG continuous-front RICs (or chains if continuous=False)."""
    key = "lseg_ric" if continuous else "lseg_chain"
    return [v[key] for v in TICKERS.values() if v.get(key)]


def yf_to_canonical(yf_ticker: str) -> str | None:
    """Reverse-lookup yfinance ticker -> canonical name."""
    for canon, v in TICKERS.items():
        if v["yf_ticker"] == yf_ticker:
            return canon
    return None


def lseg_to_canonical(ric: str) -> str | None:
    """Reverse-lookup LSEG RIC -> canonical name."""
    for canon, v in TICKERS.items():
        if v.get("lseg_ric") == ric:
            return canon
    return None


def get_sector(canonical: str) -> str:
    """Get the sector for a canonical name."""
    return TICKERS[canonical]["sector"]


SECTOR_MAP: dict[str, str] = {c: v["sector"] for c, v in TICKERS.items()}


if __name__ == "__main__":
    print(f"Total assets: {len(TICKERS)}")
    print(f"Sectors: {sorted(set(SECTOR_MAP.values()))}")
    by_sector: dict[str, list[str]] = {}
    for c, s in SECTOR_MAP.items():
        by_sector.setdefault(s, []).append(c)
    for sector, assets in sorted(by_sector.items()):
        print(f"  {sector:18s} ({len(assets):2d}): {', '.join(assets)}")
    print(f"\nMacro series: {list(MACRO_SERIES.keys())}")
