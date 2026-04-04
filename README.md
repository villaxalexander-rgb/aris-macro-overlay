# A.R.I.S Macro Overlay System

A fully autonomous systematic macro trading system running live on Interactive Brokers. Generates daily trading signals from a multi-factor commodity momentum model (BSV) conditioned on a growth/inflation regime classifier, executes real trades with hardcoded risk controls, and produces an immutable audit trail via daily GitHub commits.

## Architecture

```
main.py                      <- Daily orchestration pipeline
├── signal_engine/
│   ├── bsv_signals.py       <- Momentum, carry, value, reversal (24 GSCI assets)
│   ├── regime_classifier.py <- Growth/inflation quadrant via FRED (ISM + CPI)
│   └── daily_signals.py     <- Combines BSV + regime -> daily JSON output
├── risk_layer/
│   └── risk_checks.py       <- Position limits, VIX halt, loss kill switch
├── execution/
│   └── ibkr_executor.py     <- ib_insync order management via IBKR Gateway
├── logging_audit/
│   └── trade_logger.py      <- CSV trade log with pre-trade rationale
├── jeffrey_briefing/
│   └── fund_note.py         <- Claude API auto-generated fund notes
├── config/
│   └── settings.py          <- All parameters, API keys, asset universe
├── data/signals/             <- Daily signal JSONs (committed to git)
├── docs/fund_notes/          <- Weekly fund note archive
└── logs/                     <- Trade execution logs
```
## Signal Model (BSV)

Four-factor cross-sectional commodity momentum model:

| Factor | Weight | Logic |
|--------|--------|-------|
| Momentum | 40% | 12-month price return, ranked |
| Carry | 25% | Roll yield proxy (front vs back month) |
| Value | 20% | Mean reversion vs 5-year average |
| Reversal | 15% | 1-month contrarian signal |

Signals are overlaid with a macro regime classifier that adjusts sector weights based on the current growth/inflation quadrant (Goldilocks, Reflation, Stagflation, Deflation).

## Risk Controls

- Max position size: 2% NAV per contract
- Max portfolio notional cap
- System halt if VIX > 35
- Daily loss limit kill switch (3% NAV)
- No execution in final 30 minutes before close
- All checks must pass before any order is sent

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your API keys
```
## Usage

```bash
# Run the full daily pipeline
python main.py

# Run individual modules
python -m signal_engine.daily_signals
python -m risk_layer.risk_checks
python -m execution.ibkr_executor
```

## Audit Trail

Every trading day produces:
1. A signal JSON committed to this repo (data/signals/)
2. A CSV trade log entry with pre-trade rationale (logs/trade_log.csv)
3. An EOD snapshot to Google Sheets (positions, NAV, P&L vs benchmark)

180 consecutive daily commits = verified live track record.

## License

MIT