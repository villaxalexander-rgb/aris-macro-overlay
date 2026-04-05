"""
Module 4 - Logging and Audit Trail
"""
import csv
import os
from datetime import datetime
from config.settings import TRADE_LOG_PATH

TRADE_LOG_HEADERS = [
    "timestamp", "date", "symbol", "action", "quantity", "fill_price",
    "order_type", "signal_score", "regime", "pre_trade_thesis",
    "nav_at_trade", "daily_pnl", "vix_level",
]


def init_trade_log():
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(TRADE_LOG_HEADERS)


def log_trade(trade_data):
    init_trade_log()
    row = [trade_data.get(h, "") for h in TRADE_LOG_HEADERS]
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


def generate_pre_trade_thesis(symbol, signal_score, regime):
    direction = "LONG" if signal_score > 0 else "SHORT"
    strength = "strong" if abs(signal_score) > 0.5 else "moderate"
    return f"{direction} {symbol}: {strength} BSV composite ({signal_score:.2f}) in {regime} regime"


if __name__ == "__main__":
    init_trade_log()
    log_trade({
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "symbol": "CL", "action": "BUY", "quantity": 1,
        "fill_price": 72.50, "order_type": "MARKET",
        "signal_score": 0.65, "regime": "Reflation",
        "pre_trade_thesis": "LONG CL: strong BSV composite (0.65) in Reflation regime",
        "nav_at_trade": 250000, "daily_pnl": 1200, "vix_level": 18.5,
    })
    print("Test trade logged successfully")
