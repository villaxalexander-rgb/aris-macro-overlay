# A.R.I.S Macro Overlay System — Full Project Context
# Give this file to any AI assistant or project to share full context.
# Last updated: 2026-04-05

---

## 1. What Is A.R.I.S

A fully autonomous systematic macro trading system running live on IBKR.
Generates daily trading signals from a multi-factor commodity momentum model
(BSV) conditioned on a growth/inflation regime classifier, executes real
trades with hardcoded risk controls, and produces an immutable audit trail
via daily GitHub commits.

### Owner
Alexander Villax (villax.alexander@gmail.com)
GitHub: villaxalexander-rgb
Graduating December 2026, targeting quant/macro roles at Citadel, Brevan
Howard, Goldman Sachs.

### Goal
Build a verifiable 6-month live track record (180 consecutive daily GitHub
commits) before graduation. Walk into interviews with a live repo, real P&L,
and a public audit trail.

### GitHub Repo
https://github.com/villaxalexander-rgb/aris-macro-overlay (public)

---

## 2. Monorepo Structure

```
main.py                      <- Daily orchestration pipeline (steps 1-5)
run_daily.sh                 <- Auto-run script (activates venv, runs main.py, git commit+push)
signal_engine/
  bsv_signals.py             <- 4-factor momentum: momentum, carry, value, reversal
  regime_classifier.py       <- Growth/inflation quadrant via FRED (ISM + CPI)
  vol_target.py              <- Inverse-vol position sizing layer
  daily_signals.py           <- Combines BSV + regime + vol-target -> daily JSON
risk_layer/
  risk_checks.py             <- VIX halt, position limits, loss kill switch, time filter
execution/
  ibkr_executor.py           <- IBKRExecutor class using ib_insync
logging_audit/
  trade_logger.py            <- CSV trade log with pre-trade rationale
jeffrey_briefing/
  fund_note.py               <- Claude API auto-generated fund notes ("Jeffrey")
config/
  settings.py                <- All parameters, API keys, asset universe
tests/
  test_ibkr_connection.py    <- IBKR paper trading connection test
data/signals/                <- Daily signal + risk JSONs (committed to git)
docs/fund_notes/             <- Weekly fund note archive
logs/                        <- Execution logs + trade CSV
```

---

## 3. Module Interfaces (exact function signatures and return types)

### bsv_signals.py

```python
def fetch_commodity_prices(tickers: list[str], lookback_days: int = 365) -> pd.DataFrame:
    """Downloads adjusted close prices from yfinance.
    Returns: DataFrame, shape (N_days, N_assets), index=DatetimeIndex, cols=ticker strings.
    Uses yf.download() — returns the "Close" field only."""

def compute_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """12-month price return, cross-sectionally ranked.
    Returns: Series indexed by ticker. Values in [-1, +1] (pct_rank * 2 - 1).
    Falls back to len(prices)-1 if data shorter than window."""

def compute_carry(prices: pd.DataFrame) -> pd.Series:
    """Roll yield proxy using 21-day return (1 month), cross-sectionally ranked.
    Returns: Series indexed by ticker. Values in [-1, +1]."""

def compute_value(prices: pd.DataFrame, target_window: int = 1200) -> pd.Series:
    """Mean-reversion signal. Iterates per column to handle per-asset NaN.
    Computes (long_run_mean - current) / long_run_mean, then ranks.
    Returns: Series indexed by ticker. Values in [-1, +1].
    Assets with <60 days of data get 0.0 before ranking."""

def compute_reversal(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """1-month contrarian signal (negative of short return), ranked.
    Returns: Series indexed by ticker. Values in [-1, +1]."""

def generate_bsv_signals(prices: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Computes all 4 factors and a weighted composite.
    Default weights: momentum=0.40, carry=0.25, value=0.20, reversal=0.15
    Returns: DataFrame indexed by ticker, columns:
      [momentum, carry, value, reversal, composite]
    All values in [-1, +1] range. Composite is weighted sum of ranked factors."""
```

### regime_classifier.py

```python
def get_fred_data(series_id: str, lookback_months: int = 24) -> pd.Series:
    """Fetches a FRED time series. Returns last `lookback_months` observations."""

def compute_trend(series: pd.Series, window: int = 6) -> str:
    """Returns 'up' or 'down' based on 6-month moving average direction."""

def classify_regime() -> dict:
    """Classifies current macro regime from ISM + CPI.
    Returns dict with keys:
      regime: str          — one of 'Goldilocks', 'Reflation', 'Stagflation', 'Deflation'
      growth_trend: str    — 'up' or 'down'
      inflation_trend: str — 'up' or 'down'
      growth_value: float  — latest ISM reading (raw)
      inflation_value: float — latest CPI YoY % change
      timestamp: str       — ISO format"""

REGIME_WEIGHTS: dict[str, dict[str, float]]
    # Multiplicative sector scalars (1.0 = neutral, >1 = overweight, <1 = underweight)
    # Keys: 'Goldilocks', 'Reflation', 'Stagflation', 'Deflation'
    # Inner keys: 'energy', 'metals', 'precious_metals', 'agriculture', 'livestock'
    # Applied as multipliers to composite scores, not allocation percentages

def get_sector_weight(ticker: str, regime: str) -> float:
    """Look up regime-conditional sector multiplier for a ticker.
    Returns 1.0 if ticker has no sector mapping or regime is unknown."""

def apply_regime_weights(signals: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Apply regime-conditional sector multipliers to BSV composite scores.
    Returns: copy of signals with 3 added columns:
      sector (str), sector_weight (float), regime_adjusted_composite (float)
    Original 'composite' is preserved."""
```

### vol_target.py

```python
def compute_realized_vol(prices: pd.DataFrame, window: int = 60) -> pd.Series:
    """Annualized realized volatility per asset.
    Uses 60-day rolling stdev of daily log returns × sqrt(252).
    Returns: Series indexed by ticker. NaN for assets with insufficient history."""

def compute_vol_target_size(score: float, asset_vol: float,
                             target_vol: float = 0.10,
                             max_pct: float = 0.02) -> float:
    """Vol-targeted position size as fraction of NAV.
    Formula: clip(score * (target_vol / asset_vol), -max_pct, +max_pct)
    Returns 0.0 if asset_vol is missing/zero/negative."""

def apply_vol_targeting(signals: pd.DataFrame, prices: pd.DataFrame,
                         target_vol: float = 0.10,
                         max_pct: float = 0.02) -> pd.DataFrame:
    """Add vol-targeted position sizing to a signals DataFrame.
    Reads regime_adjusted_composite if present, else falls back to composite.
    Adds 3 columns:
      realized_vol_60d (float)
      vol_scalar (float)       — capped at 10x to prevent extreme leverage
      position_pct (float)     — final size as fraction of NAV, capped at max_pct
    This is the FINAL sizing layer — position_pct is what the executor uses."""
```

### daily_signals.py

```python
def run_daily_signals() -> dict:
    """Orchestrates the 5-step daily signal pipeline:
      1. classify_regime() from FRED
      2. fetch_commodity_prices() from yfinance
      3. generate_bsv_signals() — raw 4-factor composite
      4. apply_regime_weights() — sector multipliers → regime_adjusted_composite
      5. apply_vol_targeting() — inverse-vol scaling → position_pct (FINAL)

    Returns dict with keys:
      date: str
      timestamp: str
      regime: dict           — full output of classify_regime()
      signals: dict          — full bsv DataFrame (all factor + sizing columns)
                               as {ticker: {col: value}} (orient='index')
      target_positions: dict — {ticker: position_pct} — vol-targeted, capped, FINAL
      regime_adjusted_composite: dict  — preserved for transparency
      raw_composite: dict    — preserved for transparency
      gross_exposure: float  — sum of |position_pct|
      net_exposure: float    — sum of position_pct"""

def save_daily_signals(output: dict) -> str:
    """Saves output as JSON to data/signals/YYYY-MM-DD_signals.json
    Returns: filepath string."""
```

### risk_checks.py

```python
def check_vix() -> tuple[bool, float]:
    """Checks VIX < 35. Returns (pass: bool, current_vix: float).
    Uses yf.Ticker('^VIX').info['regularMarketPrice']."""

def check_position_size(proposed_notional: float, nav: float) -> tuple[bool, float]:
    """Checks proposed_notional / nav <= 0.02. Returns (pass: bool, pct: float)."""

def check_daily_loss(daily_pnl: float, nav: float) -> tuple[bool, float]:
    """Checks daily_pnl / nav > -0.03. Returns (pass: bool, loss_pct: float)."""

def check_market_hours() -> tuple[bool, int]:
    """Checks >30 min before 5:00 PM close. Returns (pass: bool, minutes_to_close: int).
    NOTE: currently uses 17:00 local time, not ET. Needs timezone fix for production."""

def run_all_checks(proposed_notional: float, nav: float, daily_pnl: float) -> dict:
    """Runs all 4 checks. ALL must pass.
    Returns dict with keys:
      all_pass: bool
      checks: dict with sub-dicts per check (each has 'pass', 'value', 'threshold')
      timestamp: str"""
```

### ibkr_executor.py

```python
class IBKRExecutor:
    def __init__(self) -> None
    def connect(self) -> None
    def disconnect(self) -> None
    def get_nav(self) -> float:
        """Returns NetLiquidationByCurrency (BASE). 0.0 if not found."""
    def get_positions(self) -> list:
        """Returns ib_insync Position objects."""
    def get_daily_pnl(self) -> float:
        """Returns daily P&L from ib.pnl(). 0.0 if unavailable."""
    def create_futures_contract(self, symbol: str, exchange: str = "NYMEX") -> Future:
        """Creates and qualifies a Future contract. NOTE: uses IBKR exchange symbols,
        NOT yfinance tickers. E.g. symbol='CL', exchange='NYMEX' (not 'CL=F')."""
    def place_market_order(self, contract, quantity: int, action: str = "BUY") -> dict:
        """Returns dict: order_id, symbol, action, quantity, status, fill_price, timestamp."""
    def place_limit_order(self, contract, quantity: int, price: float,
                          action: str = "BUY", timeout_seconds: int = 900) -> dict:
        """Limit order with 15-min timeout. Auto-cancels if not filled. Same return as above."""
```

### trade_logger.py

```python
TRADE_LOG_HEADERS: list[str]
    # 13 columns: timestamp, date, symbol, action, quantity, fill_price,
    # order_type, signal_score, regime, pre_trade_thesis,
    # nav_at_trade, daily_pnl, vix_level

def init_trade_log() -> None:
    """Creates logs/trade_log.csv with headers if it doesn't exist."""

def log_trade(trade_data: dict) -> None:
    """Appends one row. Keys match TRADE_LOG_HEADERS. Missing keys → empty string."""

def generate_pre_trade_thesis(symbol: str, signal_score: float, regime: str) -> str:
    """Returns string like 'LONG CL: strong BSV composite (0.65) in Reflation regime'."""
```

### fund_note.py

```python
FUND_NOTE_PROMPT: str
    # Template with placeholders: {date}, {regime}, {growth_trend}, {inflation_trend},
    # {nav}, {wtd_return}, {mtd_return}, {positions}, {signal_summary}
    # Instructs Claude to sound like a macro PM. One-page max.

def generate_fund_note(data: dict) -> str:
    """Calls Anthropic API (claude-sonnet-4-20250514). Returns fund note text.
    Required keys in data: regime, growth_trend, inflation_trend, nav,
    wtd_return, mtd_return, positions_summary, signal_summary."""

def save_fund_note(note: str, output_dir: str = "docs/fund_notes/") -> str:
    """Saves to docs/fund_notes/YYYY-MM-DD_fund_note.md. Returns filepath."""
```

### main.py

```python
def run_daily_pipeline() -> None:
    """5-step daily pipeline:
    Step 1: run_daily_signals() → save JSON (always runs)
    Step 2: run_all_checks() → save risk JSON (always runs)
      If risk blocked → stop here, log signals + risk only
    Step 3: TODO — IBKRExecutor execution
    Step 4: TODO — log_trade() for each fill
    Step 5: TODO — generate_fund_note()
    NOTE: NAV currently hardcoded at 250000 and daily_pnl at 0.
    These must be replaced with IBKRExecutor.get_nav() and get_daily_pnl()."""
```

---

## 4. Data Contracts

### prices DataFrame (primary input throughout bsv_signals.py)
- Shape: `(N_days, N_assets)` — rows = trading days, cols = ticker symbols
- Index: `pd.DatetimeIndex`, timezone-naive, daily frequency
- Values: adjusted close prices (`float64`)
- Column names: yfinance tickers with `=F` suffix: `['CL=F', 'GC=F', 'ZC=F', ...]`
- NaN handling: columns may have different start dates. `compute_value()` handles
  per-column NaN via `dropna()`. Other factor functions use `.iloc[-1]` which may
  return NaN for short-history assets — **this is a known gap**.

### BSV signals DataFrame (output of generate_bsv_signals)
- Shape: `(N_assets, 5)` — one row per ticker
- Index: ticker strings (`'CL=F'`, `'GC=F'`, etc.)
- Columns: `['momentum', 'carry', 'value', 'reversal', 'composite']`
- All values in `[-1, +1]` — cross-sectional percentile rank mapped to `rank(pct=True) * 2 - 1`
- `composite` = weighted sum: `0.40*momentum + 0.25*carry + 0.20*value + 0.15*reversal`

### BSV signals after apply_regime_weights() + apply_vol_targeting()
The full pipeline produces a DataFrame with these columns per ticker:
- `momentum`, `carry`, `value`, `reversal` — raw factor scores in [-1, +1]
- `composite` — weighted sum of factors
- `sector` (str), `sector_weight` (float) — added by apply_regime_weights
- `regime_adjusted_composite` (float) — composite × sector_weight
- `realized_vol_60d` (float) — annualized 60-day vol; NaN if insufficient data
- `vol_scalar` (float) — VOL_TARGET_PCT / realized_vol, capped at 10x
- `position_pct` (float) — final position size as fraction of NAV, in [-2%, +2%]

`position_pct` is what becomes `target_positions` in the daily JSON.
This is the value the executor multiplies by NAV to get dollar notional.

### daily_signals.json (output of save_daily_signals)
```json
{
  "date": "2026-04-05",
  "timestamp": "2026-04-05T06:00:03.123456",
  "regime": {
    "regime": "Deflation",
    "growth_trend": "down",
    "inflation_trend": "down",
    "growth_value": 47.2,
    "inflation_value": 2.4,
    "timestamp": "2026-04-05T06:00:01.234567"
  },
  "signals": {
    "CL=F": {
      "momentum": 0.71, "carry": 1.00, "value": -0.45, "reversal": -0.90,
      "composite": 0.31, "sector": "energy", "sector_weight": 0.6,
      "regime_adjusted_composite": 0.185,
      "realized_vol_60d": 0.66, "vol_scalar": 0.152, "position_pct": 0.0200
    },
    "NG=F": {
      "momentum": -0.04, "carry": -0.18, "value": -0.20, "reversal": 0.10,
      "composite": -0.18, "sector": "energy", "sector_weight": 0.6,
      "regime_adjusted_composite": -0.108,
      "realized_vol_60d": 1.80, "vol_scalar": 0.056, "position_pct": -0.0060
    }
  },
  "target_positions": {
    "CL=F": 0.0200,
    "NG=F": -0.0060
  },
  "regime_adjusted_composite": {"CL=F": 0.185, "NG=F": -0.108},
  "raw_composite": {"CL=F": 0.31, "NG=F": -0.18},
  "gross_exposure": 0.3426,
  "net_exposure": -0.0024
}
```
Note: `target_positions` is the **vol-targeted, capped position_pct** — the final
sizing that the executor uses. `regime_adjusted_composite` and `raw_composite`
are preserved alongside for transparency and debugging the pipeline stages.

### risk_result dict (output of run_all_checks)
```json
{
  "all_pass": true,
  "checks": {
    "vix": {"pass": true, "value": 18.4, "threshold": 35},
    "position_size": {"pass": true, "value": 0.02, "threshold": 0.02},
    "daily_loss": {"pass": true, "value": -0.005, "threshold": -0.03},
    "market_hours": {"pass": true, "minutes_to_close": 247}
  },
  "timestamp": "2026-04-05T06:00:05.123456"
}
```

### trade_log.csv (13 columns)
```
timestamp, date, symbol, action, quantity, fill_price, order_type,
signal_score, regime, pre_trade_thesis, nav_at_trade, daily_pnl, vix_level
```

---

## 5. Risk Logic Specification (exact rules)

### Position sizing
```python
# Currently: proposed_notional is passed into run_all_checks() externally
# check_position_size() enforces: proposed_notional / nav <= 0.02  (2% NAV max)
# This is per-position at entry only — NOT continuously enforced after entry
# Applies to absolute notional (no gross/net distinction yet — KNOWN GAP)
target_size_usd = composite_score * nav * MAX_POSITION_PCT  # planned formula
target_size_usd = clip(target_size_usd, -nav*0.02, nav*0.02)
# No sector-level gross cap exists yet — KNOWN GAP (see §11 Known Weaknesses)
```

### VIX halt
```python
# check_vix() uses yf.Ticker('^VIX').info['regularMarketPrice']
if vix_spot > 35:
    allow_new_entries = False
    # Currently: blocks ALL execution for the day via all_pass=False
    # Does NOT flatten existing positions (hold and wait)
    # No re-entry threshold implemented — KNOWN GAP
    # Planned: re-entry when VIX < 30
```

### Daily loss kill switch
```python
# check_daily_loss() checks: daily_pnl / nav > -0.03
# Currently uses the daily_pnl argument passed in (hardcoded to 0 in main.py)
# Will use IBKRExecutor.get_daily_pnl() once connected
# This checks realized + unrealized combined (IBKR dailyPnL includes both)
# When triggered: blocks new entries via all_pass=False
# Does NOT flatten positions — KNOWN GAP (should flatten, planned for Week 4)
# No re-entry for remainder of session (pipeline runs once daily so moot for now)
```

### Time filter
```python
# check_market_hours() uses local system time, not ET — BUG for production
# Compares against 17:00 (5pm) local — intended for CME futures session close
# Blocks entry if <30 minutes to close
# Does NOT use per-contract session close times — KNOWN SIMPLIFICATION
# For production: must convert to ET and use per-exchange close times
```

### Risk check gating
```python
# run_all_checks() requires ALL 4 checks to pass (boolean AND)
# If any single check fails, all_pass=False and execution is skipped entirely
# Signals and risk JSONs are ALWAYS saved regardless of pass/fail
```

---

## 6. Regime Classifier — Full Specification

### FRED Series
- ISM Manufacturing: series `MANEMP` (manufacturing employment, proxy for ISM PMI)
- CPI: series `CPIAUCSL` (CPI for All Urban Consumers, seasonally adjusted)
- CPI is converted to YoY % change: `cpi.pct_change(12) * 100`

### Trend Detection
- 6-month rolling mean, then: `if ma[-1] > ma[-2]: 'up' else 'down'`
- Binary classification only (no neutral zone currently)

### Quadrant Logic
| Growth ↑ + Inflation ↓ | Goldilocks  |
| Growth ↑ + Inflation ↑ | Reflation   |
| Growth ↓ + Inflation ↑ | Stagflation |
| Growth ↓ + Inflation ↓ | Deflation   |

### Regime → Sector Weight Multipliers (from REGIME_WEIGHTS in regime_classifier.py)
These are **multiplicative scalars** applied to composite scores, NOT allocation percentages.
1.0 = neutral weight, >1.0 = overweight the sector, <1.0 = underweight.
Applied via `apply_regime_weights()` to produce `regime_adjusted_composite`.

| Regime       | Energy | Metals (base) | Precious Metals | Agriculture | Livestock |
|--------------|--------|---------------|-----------------|-------------|-----------|
| Goldilocks   | 1.1    | 1.2           | 0.9             | 1.0         | 1.0       |
| Reflation    | 1.3    | 1.4           | 1.0             | 1.1         | 0.9       |
| Stagflation  | 1.2    | 0.8           | 1.4             | 1.0         | 0.6       |
| Deflation    | 0.6    | 0.7           | 1.1             | 0.8         | 0.7       |

### Sector → Ticker Mapping (config/settings.py SECTOR_MAP)
```
Energy:          CL=F, BZ=F, NG=F, HO=F, RB=F
Metals (base):   HG=F
Precious Metals: GC=F, SI=F, PA=F, PL=F
Agriculture:     ZC=F, ZW=F, ZS=F, ZM=F, ZL=F, CT=F, KC=F, SB=F, CC=F
Livestock:       LE=F, HE=F, GF=F
```

### Update Frequency
- Regime is recalculated on every daily pipeline run (not just monthly)
- However, underlying FRED data is monthly release: ISM = first business day,
  CPI = mid-month. So regime effectively changes at most twice per month.
- Between releases, `classify_regime()` returns the same result (no interpolation)

---

## 7. IBKR Connection Specification

### Connection Parameters (from config/settings.py)
```python
IBKR_HOST = "127.0.0.1"     # localhost
IBKR_PORT = 4001             # .env default; paper=4002, live TWS=7497, live Gateway=4001
IBKR_CLIENT_ID = 1           # must be unique per concurrent connection
```
**IMPORTANT**: The .env should set `IBKR_PORT=4002` for paper trading.
Default in settings.py is 4001 (Gateway live). Override via .env for safety.

### Contract Definitions
```python
# IBKR uses exchange symbols, NOT yfinance tickers
# yfinance: 'CL=F' → IBKR: Future(symbol='CL', exchange='NYMEX')
# The mapping from yfinance ticker to IBKR contract is NOT yet implemented
# ibkr_executor.py uses: Future(symbol=X, exchange='NYMEX') for all — WRONG for non-NYMEX
# Correct exchange mapping needed:
#   NYMEX: CL, NG, HO, RB, PA, PL, HG
#   COMEX: GC, SI (IBKR routes via NYMEX but exchange should be COMEX)
#   CBOT:  ZC, ZW, ZS, ZM, ZL
#   NYBOT/ICE: CT, KC, SB, CC
#   CME:   LE, HE, GF
```

### Order Types Currently Implemented
- Entry: `MarketOrder(action, quantity)` — fills at next available price
- Limit: `LimitOrder(action, quantity, price)` — 15-min timeout, auto-cancel if not filled
- No stop orders implemented yet — KNOWN GAP (planned: -1.5% per-position stop)

### Account Numbers
- Paper account: used for development and testing
- Live account: exists but NEVER submit during development
- Account IDs stored in IBKR, not in code

### IBC (Interactive Brokers Controller) — NOT YET SET UP
- Planned: headless IB Gateway via IBC for Docker/VPS deployment
- Local development: manually launch IB Gateway desktop app
- Production path: IBC + Docker on Hetzner VPS (Week 4-5)

---

## 8. Error Handling Philosophy

### Guiding Principle
**When in doubt, do nothing and log loudly.** A missed trade is far better than
an incorrect trade or an unhandled exception that crashes the pipeline on day 47.

### Hierarchy of Failures
1. **Data failure** (yfinance/FRED unavailable):
   → Use previous day's signals if available. Log WARNING. Do not halt.
   → NOT YET IMPLEMENTED — currently crashes. Needs try/except + fallback logic.

2. **Signal NaN or computation error**:
   → Skip that asset for the day. Log ERROR. Continue with remaining assets.
   → PARTIALLY IMPLEMENTED — `compute_value()` handles NaN, other factors do not.

3. **IBKR connection failure**:
   → Retry 3x with 60s backoff. If still failed, send email alert, skip execution.
   → Do NOT use cached orders from previous day.
   → NOT YET IMPLEMENTED.

4. **Risk check failure** (unexpected exception in risk_checks.py):
   → HALT all execution. This is the one case where we fail loudly and do nothing.
   → PARTIALLY IMPLEMENTED — exceptions will crash, but no alerting.

5. **GitHub commit failure** (in run_daily.sh):
   → Retry 2x. If failed, write to local backup log. Non-blocking.
   → NOT YET IMPLEMENTED — run_daily.sh does single attempt.

### Alert Destinations (planned)
- Email: villax.alexander@gmail.com (via SendGrid or Gmail API)
- Log file: `logs/errors_{date}.log`
- Google Sheets: append to 'Errors' tab

---

## 9. Data Source Fragilities & Workarounds

### yfinance Known Issues
- `LE=F` (Live Cattle) and `HE=F` (Lean Hogs) have frequent data gaps
  → Handle with `ffill(limit=3)` — NOT YET IMPLEMENTED
- `GF=F` (Feeder Cattle) volume is thin; consider capping position size at 0.5x
- Continuous contract prices from yfinance are NOT roll-adjusted
  → Safe for momentum/reversal signals (price change based)
  → Carry signal is a proxy only (21-day return, not true roll yield)
  → Value signal uses price levels — potentially distorted at roll dates
- Download timeout: use `yf.download(..., timeout=30, progress=False)` — NOT SET
- Rate limiting: no sleep between downloads currently; may need 2s delay in batch mode
- Tickers can go "delisted" temporarily — must use `=F` suffix (already done in settings.py)
- Note: settings.py has a formatting issue — `CC=F` and `LE=F` are on the same line
  due to a paste error: `"CC=F",   # Cocoa    "LE=F",   # Live Cattle`
  This means LE=F is a comment, not in the list. **BUG — only 21 tickers, not 22.**

### FRED API Known Issues
- ISM release is typically first business day of month
- CPI release is mid-month
- FRED series `MANEMP` is manufacturing employment, not ISM PMI itself
  → It's a proxy. True ISM PMI series is not on FRED. This is a known approximation.
- If FRED API is down, `fredapi` raises an exception — no fallback logic exists
- Staleness: if >35 days since last reading, should force re-fetch and warn

---

## 10. Expected Daily Run Output

### Console output (main.py successful run)
```
============================================================
A.R.I.S MACRO OVERLAY - Daily Run 2026-04-05 06:00
============================================================

[1/5] Running signal engine...
Regime: Deflation (Growth down, Inflation down)
Signal file: data/signals/2026-04-05_signals.json

[2/5] Running risk checks...
Risk checks: ALL PASS
  vix: PASS (value: 18.4)
  position_size: PASS (value: 0.02)
  daily_loss: PASS (value: 0.0)
  market_hours: PASS (value: 247)
Risk log: data/signals/2026-04-05_risk.json

[3/5] Execution engine...
TODO: Connect IBKR executor (Module 3)

[4/5] Logging trades...
TODO: Log actual trades once execution is live

[5/5] Generating fund note...
TODO: Connect Jeffrey briefing (Module 5)

============================================================
Daily pipeline complete.
============================================================
```

### Files written each day
```
data/signals/YYYY-MM-DD_signals.json   ← full signal output
data/signals/YYYY-MM-DD_risk.json      ← risk check results
```

### Files written each day (planned, not yet implemented)
```
logs/trade_log.csv                     ← append rows per trade
logs/pipeline_YYYY-MM-DD.log           ← full pipeline stdout
data/performance/equity_curve.csv      ← append daily NAV + P&L
docs/fund_notes/YYYY-MM-DD_fund_note.md ← Jeffrey briefing
```

### Git commit (via run_daily.sh)
```bash
# run_daily.sh does: cd project, activate venv, python main.py, then:
git add -A
git commit -m "Daily signal run YYYY-MM-DD"
git push origin main
```

---

## 11. Jeffrey Briefing — Prompt Specification

### System prompt (FUND_NOTE_PROMPT in fund_note.py)
The prompt instructs Claude to act as "Jeffrey, the chief strategist for A.R.I.S."
and write a one-page institutional fund note with these sections:
1. REGIME — current quadrant with growth/inflation values
2. NAV + WTD/MTD returns
3. POSITIONING — current open positions
4. MACRO CONTEXT — 3 sentences on macro environment
5. SIGNAL UPDATE — top signals by composite score
6. WHAT WOULD CHANGE THIS VIEW — 2-3 falsifiable conditions for regime shift

### User prompt (injected from daily pipeline data)
Required keys: `regime`, `growth_trend`, `inflation_trend`, `nav`, `wtd_return`,
`mtd_return`, `positions_summary`, `signal_summary`

### Model
`claude-sonnet-4-20250514` (set in `config/settings.py` as `ANTHROPIC_MODEL`)

### Output
- Saved to: `docs/fund_notes/YYYY-MM-DD_fund_note.md`
- Planned: email via SendGrid, voice via ElevenLabs TTS → MP3 (Week 7)
- Weekly archive: every Friday, `docs/fund_notes/` pushed to GitHub

---

## 12. Backtest Statistics (cite these, do not invent numbers)

```
Period:          Jan 2010 – Dec 2024 (in-sample 2010–2020, OOS 2021–2024)
Ann. Return:     18.3%
Ann. Volatility: 10.1%
Sharpe Ratio:    1.83 (OOS)
Max Drawdown:    13.7%
Win Rate:        58.2% (monthly)
Validation:      Walk-forward OOS, block bootstrap (1000 iterations),
                 permutation test (p<0.01)
Benchmark:       BCOM (Bloomberg Commodity Index)
Benchmark Sharpe: 0.31 over same period
Source:          Booth School coursework, rigorously validated
```

---

## 13. Known Weaknesses & Open Problems

1. **Transaction costs not modeled in backtest** — live results will be degraded by
   futures commissions (~$2.50/contract), bid-ask spread, and slippage. Need to add
   $3/contract estimate for break-even analysis.

2. **Regime classifier lag** — FRED data is monthly. ~3-week lag between real-world
   regime shift and classifier catching it. Signals during transitions are noisy.
   Planned mitigation: blend regime weights across adjacent quadrants during
   transition months (Week 6).

3. **yfinance continuous contracts are not roll-adjusted** — carry signal is a proxy
   (21-day return), not true roll yield. Value signal uses price levels which may be
   distorted at roll dates. Bloomberg or Quandl would fix this but cost money.

4. ~~No volatility-adjusted position sizing~~ **FIXED 2026-04-06**:
   Added `signal_engine/vol_target.py` with `apply_vol_targeting()`. Positions
   are now scaled by `(VOL_TARGET_PCT / realized_vol_60d)` so each contributes
   ~10% annualized vol regardless of underlying asset volatility. Capped at
   ±2% NAV (`MAX_POSITION_PCT`). Live test: NG=F at -0.108 score now sizes
   to -0.6% NAV instead of -2% NAV (saved from oversized exposure to a
   180%-vol asset). Caveat: realized vol estimates are inflated by yfinance
   continuous-contract roll noise (see Known Weakness #3) — this makes the
   sizing slightly conservative, which is the right error direction.

5. **No correlation or portfolio-level risk constraint** — in Stagflation regime,
   energy assets can cluster long simultaneously, creating concentration risk.
   Sector-level gross cap needed.

6. ~~Regime weights not applied in pipeline~~ **FIXED 2026-04-06**:
   `apply_regime_weights()` now multiplies BSV composite scores by sector-specific
   scalars from `REGIME_WEIGHTS`. `daily_signals.py` uses
   `regime_adjusted_composite` as the actual `target_positions`. Raw composite is
   preserved in JSON output for transparency. Sector mapping lives in
   `config/settings.py` (`SECTOR_MAP`), with precious metals split out from base
   metals as a separate sector.

7. ~~settings.py formatting bug~~ **FIXED 2026-04-06**: All 22 GSCI tickers now
   on separate lines.

8. **check_market_hours() uses local time, not ET** — will give wrong results if
   system timezone is not US Eastern. Must convert to ET for production.

9. **No reconnection logic for IBKR** — if IB Gateway drops mid-session, the
   executor has no retry mechanism. Critical for Docker/VPS deployment.

10. **Kill switch doesn't flatten positions** — `run_all_checks()` returns
    `all_pass=False` but doesn't issue close orders. If the daily loss limit is hit
    intraday (and pipeline is running), open positions keep bleeding.

11. **ISM proxy** — FRED series `MANEMP` (manufacturing employment) is not the actual
    ISM PMI. It's correlated but not identical. True ISM PMI is behind a paywall.

12. **No idempotency** — running `main.py` twice in one day overwrites the signal
    JSON. Should check if today's run already exists and skip or version.

---

## 14. Configuration Reference (config/settings.py)

```python
# Risk Parameters
MAX_POSITION_PCT = 0.02           # 2% of NAV per position (hard cap)
VIX_HALT_THRESHOLD = 35           # VIX > 35 → no new entries
DAILY_LOSS_LIMIT_PCT = 0.03       # -3% daily P&L → kill switch
NO_TRADE_MINUTES_BEFORE_CLOSE = 30

# Volatility Targeting
VOL_TARGET_PCT = 0.10             # 10% annualized vol target per position
VOL_LOOKBACK_DAYS = 60            # 60-day rolling window for realized vol

# BSV Factor Weights (default, in bsv_signals.py)
weights = {"momentum": 0.40, "carry": 0.25, "value": 0.20, "reversal": 0.15}

# FRED Series
FRED_ISM_SERIES = "MANEMP"       # Manufacturing employment (ISM proxy)
FRED_CPI_SERIES = "CPIAUCSL"     # CPI All Urban Consumers SA

# File Paths
TRADE_LOG_PATH = "logs/trade_log.csv"
SIGNAL_OUTPUT_PATH = "data/signals/"
FUND_NOTE_PATH = "docs/fund_notes/"

# IBKR
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4001                  # Override in .env: 4002 for paper
IBKR_CLIENT_ID = 1
```

---

## 15. Tech Stack
Python 3.14, pandas, numpy, yfinance, fredapi, ib_insync, python-dotenv,
anthropic SDK, IBC (planned), Docker (planned), Hetzner VPS (planned),
Google Sheets API (planned), SendGrid (planned), ElevenLabs (planned)

## 16. Local Setup
- Project: `~/Desktop/aris-macro-overlay`
- Virtual env: `~/Desktop/aris-macro-overlay/venv`
- Reference repos: `~/Desktop/aris-references` (13 repos)
- Daily scheduler: launchd (`com.aris.daily-signal.plist`) at 6am
- IB Gateway: manual launch, paper trading, port 4002

## 17. Timeline
- Weeks 1-2 (done): Signal engine + regime classifier
- Week 3 (current): Risk layer refinement + IBKR paper trading connection
- Weeks 4-5: IBKR live execution + Docker on Hetzner VPS
- Week 6: Vol-targeting overlay, regime weight application, logging pipeline
- Week 7: Jeffrey briefing (Claude API + ElevenLabs voice)
- Week 8: Public webpage + weekly note archive
- Go-live: End of May 2026
- Track record complete: November 2026

## 18. Current Status (as of April 6, 2026)
- Signal engine: **WORKING** (22 assets scored, no NaN)
- Regime classifier: **WORKING** (currently reading Deflation)
- Regime weights: **APPLIED** ✓ (apply_regime_weights wired into daily pipeline)
- Vol-targeting: **APPLIED** ✓ (apply_vol_targeting equalizes risk per position)
- Risk layer: **WORKING** (correctly blocks outside market hours)
- Full pipeline: **WORKING** (main.py chains steps 1-2, steps 3-5 are TODOs)
- Current book: 22 vol-targeted positions, ~34% gross / ~0% net exposure
- Auto-run script: **WORKING** (run_daily.sh commits + pushes)
- Daily scheduler: **SET UP** (launchd at 6am)
- GitHub commits: 3+ of 180 completed
- IBKR execution: **NOT YET CONNECTED** (connection test written, awaiting Gateway)
- Jeffrey briefing: **NOT YET CONNECTED** (Week 7)
- Error handling layer: **NOT YET BUILT** (FRED 500 errors still crash pipeline)

## 19. The Interview Pitch
"I built and ran a live systematic macro overlay on my IBKR account for
six months. It takes signals from a multi-factor commodity momentum model
conditioned on a growth/inflation regime classifier, executes autonomously
via IBKR, and generates a daily fund note using Claude. Here's the GitHub
repo with 180 consecutive daily commits. Here's the live P&L. Here's where
it worked, where it broke, and what I learned."
