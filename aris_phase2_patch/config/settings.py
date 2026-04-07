"""
A.R.I.S Macro Overlay System — Global Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- IBKR ---
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4001))  # 4001 = Gateway live, 4002 = Gateway paper
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 1))

# --- FRED API ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- LSEG / Refinitiv ---
LSEG_APP_KEY = os.getenv("LSEG_APP_KEY", "")
LSEG_SESSION_TYPE = os.getenv("LSEG_SESSION_TYPE", "desktop")  # desktop|platform
LSEG_USERNAME = os.getenv("LSEG_USERNAME", "")  # only needed for platform
LSEG_PASSWORD = os.getenv("LSEG_PASSWORD", "")  # only needed for platform

# --- Data source preferences ---
PRIMARY_PRICE_SOURCE = os.getenv("PRIMARY_PRICE_SOURCE", "lseg")    # lseg|yfinance
PRIMARY_MACRO_SOURCE = os.getenv("PRIMARY_MACRO_SOURCE", "lseg")    # lseg|fred
DUAL_SOURCE_DISAGREEMENT_PCT = 0.02  # 2% — log warning if sources disagree by more

# --- Anthropic (Jeffrey) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# --- Google Sheets ---
GSHEET_CREDENTIALS = os.getenv("GSHEET_CREDENTIALS", "credentials.json")
GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID", "")

# --- Risk Parameters ---
MAX_POSITION_PCT = 0.02        # 2% NAV per contract
VIX_HALT_THRESHOLD = 35        # System halt if VIX > 35 (equity vol)
MOVE_HALT_THRESHOLD = 150      # System halt if MOVE > 150 (rates vol)
DXY_WARN_HIGH = 110            # Log warning if DXY > 110 (USD strength regime)
DXY_WARN_LOW = 90              # Log warning if DXY < 90 (USD weakness regime)
DAILY_LOSS_LIMIT_PCT = 0.03    # 3% daily loss kill switch
NO_TRADE_MINUTES_BEFORE_CLOSE = 30

# --- GSCI Universe (24 assets) ---
GSCI_ASSETS = [
    "CL",  # Crude Oil WTI
    "BZ",  # Brent Crude
    "NG",  # Natural Gas
    "HO",  # Heating Oil
    "RB",  # RBOB Gasoline
    "GC",  # Gold
    "SI",  # Silver
    "HG",  # Copper
    "PA",  # Palladium
    "PL",  # Platinum
    "ZC",  # Corn
    "ZW",  # Wheat
    "ZS",  # Soybeans
    "ZM",  # Soybean Meal
    "ZL",  # Soybean Oil
    "CT",  # Cotton
    "KC",  # Coffee
    "SB",  # Sugar
    "CC",  # Cocoa
    "LC",  # Live Cattle
    "LH",  # Lean Hogs
    "FC",  # Feeder Cattle
    "LE",  # Live Cattle (CME)
    "AL",  # Aluminum
]

# --- FRED Series for Regime Classifier ---
FRED_ISM_SERIES = "MANEMP"     # ISM Manufacturing proxy
FRED_CPI_SERIES = "CPIAUCSL"   # CPI All Urban Consumers

# --- Paths ---
TRADE_LOG_PATH = "logs/trade_log.csv"
SIGNAL_OUTPUT_PATH = "data/signals/"
FUND_NOTE_PATH = "docs/fund_notes/"
