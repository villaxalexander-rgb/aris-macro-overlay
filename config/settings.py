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
MAX_POSITION_PCT = 0.02
VIX_HALT_THRESHOLD = 35
DAILY_LOSS_LIMIT_PCT = 0.03
NO_TRADE_MINUTES_BEFORE_CLOSE = 30

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
    "CC=F",   # Cocoa    "LE=F",   # Live Cattle
    "HE=F",   # Lean Hogs
    "GF=F",   # Feeder Cattle
]

# --- FRED Series for Regime Classifier ---
FRED_ISM_SERIES = "MANEMP"
FRED_CPI_SERIES = "CPIAUCSL"

# --- Paths ---
TRADE_LOG_PATH = "logs/trade_log.csv"
SIGNAL_OUTPUT_PATH = "data/signals/"
FUND_NOTE_PATH = "docs/fund_notes/"