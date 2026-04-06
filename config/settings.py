"""
A.R.I.S Macro Overlay System — Global Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- IBKR ---
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4001))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 1))

# --- FRED API ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- Anthropic (Jeffrey) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# --- Google Sheets ---
GSHEET_CREDENTIALS = os.getenv("GSHEET_CREDENTIALS", "credentials.json")
GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID", "")

# --- Risk Parameters ---
MAX_POSITION_PCT = 0.02            # Hard cap: 2% NAV per position
VIX_HALT_THRESHOLD = 35
DAILY_LOSS_LIMIT_PCT = 0.03
NO_TRADE_MINUTES_BEFORE_CLOSE = 30

# --- Volatility Targeting ---
# Target annualized vol contribution per position. A score of 1.0 in an
# asset whose realized vol equals VOL_TARGET_PCT produces a position size
# equal to MAX_POSITION_PCT (subject to the cap). Higher-vol assets get
# smaller positions; lower-vol assets get larger positions, both bounded
# by MAX_POSITION_PCT.
VOL_TARGET_PCT = 0.10              # 10% annualized vol target per position
VOL_LOOKBACK_DAYS = 60             # 60-day rolling window for realized vol

# --- GSCI Universe (yfinance front-month futures tickers) ---
GSCI_ASSETS = [
    "CL=F",   # Crude Oil WTI
    "BZ=F",   # Brent Crude
    "NG=F",   # Natural Gas
    "HO=F",   # Heating Oil
    "RB=F",   # RBOB Gasoline
    "GC=F",   # Gold
    "SI=F",   # Silver
    "HG=F",   # Copper
    "PA=F",   # Palladium
    "PL=F",   # Platinum
    "ZC=F",   # Corn
    "ZW=F",   # Wheat
    "ZS=F",   # Soybeans
    "ZM=F",   # Soybean Meal
    "ZL=F",   # Soybean Oil
    "CT=F",   # Cotton
    "KC=F",   # Coffee
    "SB=F",   # Sugar
    "CC=F",   # Cocoa
    "LE=F",   # Live Cattle
    "HE=F",   # Lean Hogs
    "GF=F",   # Feeder Cattle
]

# --- Sector Mapping (ticker -> sector for regime weight application) ---
# Splits base metals (HG) from precious metals (GC, SI, PA, PL) because they
# behave very differently across regimes — precious metals are the inflation
# hedge, base metals are the growth play.
SECTOR_MAP = {
    # Energy
    "CL=F": "energy",
    "BZ=F": "energy",
    "NG=F": "energy",
    "HO=F": "energy",
    "RB=F": "energy",
    # Precious metals
    "GC=F": "precious_metals",
    "SI=F": "precious_metals",
    "PA=F": "precious_metals",
    "PL=F": "precious_metals",
    # Base metals
    "HG=F": "metals",
    # Agriculture
    "ZC=F": "agriculture",
    "ZW=F": "agriculture",
    "ZS=F": "agriculture",
    "ZM=F": "agriculture",
    "ZL=F": "agriculture",
    "CT=F": "agriculture",
    "KC=F": "agriculture",
    "SB=F": "agriculture",
    "CC=F": "agriculture",
    # Livestock
    "LE=F": "livestock",
    "HE=F": "livestock",
    "GF=F": "livestock",
}

# --- FRED Series for Regime Classifier ---
FRED_ISM_SERIES = "MANEMP"
FRED_CPI_SERIES = "CPIAUCSL"

# --- Paths ---
TRADE_LOG_PATH = "logs/trade_log.csv"
SIGNAL_OUTPUT_PATH = "data/signals/"
FUND_NOTE_PATH = "docs/fund_notes/"