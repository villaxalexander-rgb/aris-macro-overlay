"""
Module 5 — Jeffrey Daily Briefing

Reads the day's signal JSON, trade log, execution record, and risk state,
then generates an institutional-quality daily fund note via Claude API.

Data sources (all produced by earlier pipeline stages):
  - data/signals/{DATE}_signals.json  → regime, signals, targets, health
  - logs/trade_log.csv                → executed trades with notional/price
  - logs/executed_signals.json        → NAV at execution, intent summary
  - state/nav_at_open.json            → local NAV anchor (if available)

Output:
  - docs/fund_notes/{DATE}_fund_note.md
  - Optionally prints to stdout for cron/log capture
"""
import json
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import anthropic

from config.settings import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    SIGNAL_OUTPUT_PATH,
    FUND_NOTE_PATH,
    TRADE_LOG_PATH,
    VIX_HALT_THRESHOLD,
    MOVE_HALT_THRESHOLD,
    DXY_WARN_HIGH,
    DXY_WARN_LOW,
    DAILY_LOSS_LIMIT_PCT,
)

ROOT = Path(__file__).resolve().parent.parent


# ── Data loaders ──────────────────────────────────────────────

def load_signal_json(date_str: str) -> Optional[dict]:
    """Load the day's signal JSON."""
    path = ROOT / SIGNAL_OUTPUT_PATH / f"{date_str}_signals.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_execution_record() -> dict:
    """Load the execution record (idempotency + NAV at execution)."""
    path = ROOT / "logs" / "executed_signals.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_trade_log(date_str: str) -> list[dict]:
    """Load today's trades from the CSV trade log."""
    path = ROOT / TRADE_LOG_PATH
    if not path.exists():
        return []
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            # Strip whitespace from keys and values
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            if row.get("signal_date", "").startswith(date_str):
                trades.append(row)
    return trades


def load_nav(date_str: str, exec_record: dict) -> Optional[float]:
    """Get NAV — prefer nav_at_open.json, fall back to execution record."""
    nav_path = ROOT / "state" / "nav_at_open.json"
    if nav_path.exists():
        with open(nav_path) as f:
            data = json.load(f)
            return data.get("nav")
    # Fall back to execution record
    day_exec = exec_record.get(date_str, {})
    return day_exec.get("nav")


# ── Data assembly ─────────────────────────────────────────────

def assemble_briefing_context(date_str: str) -> dict:
    """
    Pull together everything Jeffrey needs for the daily note.
    Returns a structured dict ready for prompt formatting.
    """
    signals = load_signal_json(date_str)
    exec_record = load_execution_record()
    trades = load_trade_log(date_str)
    nav = load_nav(date_str, exec_record)

    # ── Regime ──
    regime_data = signals.get("regime", {}) if signals else {}
    regime = regime_data.get("regime", "Unknown")
    growth_trend = regime_data.get("growth_trend", "?")
    inflation_trend = regime_data.get("inflation_trend", "?")
    ism_value = regime_data.get("growth_value")
    cpi_value = regime_data.get("inflation_value")

    # ── Health ──
    health_data = signals.get("health", {}) if signals else {}
    healthy = health_data.get("healthy", False)
    degradations = health_data.get("degradations", [])
    disagreements = health_data.get("disagreements", {})
    sources = health_data.get("sources", {})

    # ── Signals — top longs and shorts ──
    target_positions = signals.get("target_positions", {}) if signals else {}
    signal_details = signals.get("signals", {}) if signals else {}
    sorted_targets = sorted(target_positions.items(), key=lambda x: x[1], reverse=True)
    top_longs = [(k, v) for k, v in sorted_targets if v > 0][:5]
    top_shorts = [(k, v) for k, v in sorted_targets if v < 0][-5:]

    # ── Signal summary string (with component breakdown) ──
    signal_lines = []
    for asset, target in sorted_targets:
        detail = signal_details.get(asset, {})
        mom = detail.get("momentum", 0)
        carry = detail.get("carry", 0)
        val = detail.get("value", 0)
        rev = detail.get("reversal", 0)
        comp = detail.get("composite", target)
        signal_lines.append(
            f"  {asset:>4s}: composite={comp:+.3f} "
            f"(mom={mom:+.2f} carry={carry:+.2f} val={val:+.2f} rev={rev:+.2f}) "
            f"→ target_pct={target:+.3f}"
        )

    # ── Carry source breakdown ──
    carry_sources = signals.get("carry_source_per_asset", {}) if signals else {}
    lseg_carry = sum(1 for v in carry_sources.values() if v == "lseg_curve")
    proxy_carry = sum(1 for v in carry_sources.values() if v == "momentum_proxy")

    # ── Trades executed today ──
    trade_lines = []
    for t in trades:
        trade_lines.append(
            f"  {t.get('action','?')} {t.get('delta','?')} {t.get('local_symbol','?')} "
            f"({t.get('canonical','?')}) @ {t.get('price','?')} "
            f"notional={t.get('notional','?')} micro={t.get('is_micro','?')}"
        )
    if not trade_lines:
        trade_lines = ["  No trades executed (dry run or risk-blocked)"]

    # ── Risk flags ──
    risk_flags = []
    if not healthy:
        risk_flags.append(f"DATA QUALITY: Pipeline unhealthy — {len(degradations)} degradation(s), {len(disagreements)} disagreement(s)")
    for src, status in sources.items():
        if "cache" in str(status) or "missing" in str(status):
            risk_flags.append(f"  {src}: {status}")
    if not risk_flags:
        risk_flags = ["All systems nominal — data fresh, no disagreements"]

    return {
        "date": date_str,
        "regime": regime,
        "growth_trend": growth_trend,
        "inflation_trend": inflation_trend,
        "ism_value": ism_value,
        "cpi_value": cpi_value,
        "nav": nav,
        "healthy": healthy,
        "risk_flags": "\n".join(risk_flags),
        "signal_lines": "\n".join(signal_lines) or "No signals generated",
        "trade_lines": "\n".join(trade_lines),
        "top_longs": top_longs,
        "top_shorts": top_shorts,
        "carry_lseg": lseg_carry,
        "carry_proxy": proxy_carry,
        "n_assets": len(target_positions),
        "degradations": degradations,
        "disagreements": disagreements,
    }


# ── Prompt + generation ───────────────────────────────────────

JEFFREY_SYSTEM = """You are Jeffrey, chief strategist for the A.R.I.S Macro Overlay System — a systematic macro fund that trades 21 commodity futures using a BSV (momentum, carry, value, reversal) signal framework with regime-based overlay.

Your job is to write a concise, institutional-quality daily fund note. You write like a macro PM at a top commodity house — precise, quantitative, no filler. One page maximum.

Rules:
- Lead with the regime and what changed vs yesterday
- Reference specific ISM/CPI levels and what they imply
- When discussing signals, mention which BSV components are driving the composite
- Flag any data quality issues (stale cache, source disagreements) prominently
- "What would change this view" must contain specific, falsifiable conditions
- Do NOT use emojis, bullet-point-only formatting, or chatbot language
- Sound like a Goldman Sachs or Bridgewater daily research note"""


def build_prompt(ctx: dict) -> str:
    """Build the user prompt from assembled context."""
    nav_str = f"€{ctx['nav']:,.0f}" if ctx["nav"] else "NAV unavailable"
    ism_str = f"{ctx['ism_value']:.1f}" if ctx["ism_value"] else "unavailable"
    cpi_str = f"{ctx['cpi_value']:.2f}%" if ctx["cpi_value"] else "unavailable"

    return f"""Generate today's A.R.I.S fund note for {ctx['date']}.

REGIME: {ctx['regime']} (Growth: {ctx['growth_trend']}, Inflation: {ctx['inflation_trend']})
ISM Manufacturing: {ism_str} | CPI YoY: {cpi_str}
NAV: {nav_str}

DATA QUALITY:
{ctx['risk_flags']}

SIGNAL SCORES ({ctx['n_assets']} assets, carry via LSEG: {ctx['carry_lseg']}, proxy: {ctx['carry_proxy']}):
{ctx['signal_lines']}

TRADES EXECUTED:
{ctx['trade_lines']}

Use the template:
A.R.I.S MACRO OVERLAY | {ctx['date']}
REGIME: ...
NAV: ... | DATA: ...
POSITIONING: top longs, top shorts, key trades
MACRO CONTEXT: 3-4 sentences
SIGNAL UPDATE: which BSV components are driving, what's new
WHAT WOULD CHANGE THIS VIEW: 2-3 falsifiable conditions"""


def generate_briefing(date_str: Optional[str] = None) -> str:
    """
    Full pipeline: load data → assemble context → call Claude → return note.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    ctx = assemble_briefing_context(date_str)
    prompt = build_prompt(ctx)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=JEFFREY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def save_briefing(note: str, date_str: Optional[str] = None) -> str:
    """Save to docs/fund_notes/{DATE}_fund_note.md."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = ROOT / FUND_NOTE_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{date_str}_fund_note.md"
    path.write_text(note)
    print(f"Fund note saved to {path}")
    return str(path)


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate A.R.I.S daily fund note")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to generate briefing for (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assemble context and print prompt, but skip Claude call")
    parser.add_argument("--stdout", action="store_true",
                        help="Print note to stdout in addition to saving")
    args = parser.parse_args()

    if args.dry_run:
        ctx = assemble_briefing_context(args.date)
        print("=== ASSEMBLED CONTEXT ===")
        for k, v in ctx.items():
            if isinstance(v, str) and "\n" in v:
                print(f"\n{k}:\n{v}")
            else:
                print(f"{k}: {v}")
        print("\n=== PROMPT ===")
        print(build_prompt(ctx))
    else:
        note = generate_briefing(args.date)
        path = save_briefing(note, args.date)
        if args.stdout:
            print(f"\n{'='*60}")
            print(note)
            print(f"{'='*60}")
