"""
A.R.I.S Macro Overlay System — Global Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- IBKR ---
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4002))  # 4002 = Gateway paper (safe default), 4001 = Gateway live (must opt in via .env)
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 1))

# --- FRED API ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- LSEG / Refinitiv ---
LSEG_APP_KEY = os.getenv("LSEG_APP_KEY", "")
LSEG_SESSION_TYPE = os.getenv("LSEG_SESSION_TYPE", "desktop")  # desktop|platform
LSEG_USERNAME = os.getenv("LSEG_USERNAME", "")
LSEG_PASSWORD = os.getenv("LSEG_PASSWORD", "")

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
    "CL", "BZ", "NG", "HO", "RB",
    "GC", "SI", "HG", "PA", "PL",
    "ZC", "ZW", "ZS", "ZM", "ZL",
    "CT", "KC", "SB", "CC",
    "LC", "LH", "FC", "LE", "AL",
]

# --- FRED Series for Regime Classifier ---
FRED_ISM_SERIES = "MANEMP"
FRED_CPI_SERIES = "CPIAUCSL"

# --- Paths ---
TRADE_LOG_PATH = "logs/trade_log.csv"
SIGNAL_OUTPUT_PATH = "data/signals/"
FUND_NOTE_PATH = "docs/fund_notes/"


# --- Phase A: Alpha engine (T6 Hybrid) ---
USE_HYBRID_ALPHA = os.getenv("USE_HYBRID_ALPHA", "true").lower() == "true"
TSMOM_WEIGHT = float(os.getenv("TSMOM_WEIGHT", "0.50"))  # 0.5 = 50/50 BSV hybrid
TSMOM_LOOKBACKS = (63, 126, 252)  # 3m, 6m, 12m trading days

# --- Phase A: Vol-target sizing ---
PORTFOLIO_VOL_TARGET = float(os.getenv("PORTFOLIO_VOL_TARGET", "0.10"))  # 10% annualized
VOL_TARGET_WINDOW = int(os.getenv("VOL_TARGET_WINDOW", "63"))             # 3m trailing
SIGNAL_GROSS_CAP = float(os.getenv("SIGNAL_GROSS_CAP", "1.00"))           # 100% NAV max gross

# --- Phase A: Regime overlay toggle ---
# Research showed regime tilts did not improve risk-adjusted returns on T6.
# Kept here as a toggle for forward A/B testing; default OFF.
APPLY_REGIME_TILTS = os.getenv("APPLY_REGIME_TILTS", "false").lower() == "true"

# --- Phase A: Rebalance policy ---
REBALANCE_FREQ = os.getenv("REBALANCE_FREQ", "weekly")  # daily|weekly|monthly
REBALANCE_WEEKDAY = int(os.getenv("REBALANCE_WEEKDAY", "0"))  # 0=Monday
TRADE_DEADBAND_PCT = float(os.getenv("TRADE_DEADBAND_PCT", "0.005"))  # 0.5% NAV min leg delta
