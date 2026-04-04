"""
Module 5 — Jeffrey Morning Briefing
Auto-generates a one-page fund note using Claude API.
"""
import os
import anthropic
from datetime import datetime
from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

FUND_NOTE_PROMPT = """You are Jeffrey, the chief strategist for the A.R.I.S Macro Overlay System.
Write a concise, institutional-quality daily fund note using EXACTLY this template:

A.R.I.S MACRO OVERLAY | {date}

REGIME: {regime} (Growth {growth_trend}, Inflation {inflation_trend})
NAV: EUR {nav:,.0f} | WTD: {wtd_return}% | MTD: {mtd_return}%

POSITIONING:
{positions}

MACRO CONTEXT:
Write exactly 3 sentences summarizing the current macro environment.

SIGNAL UPDATE:
{signal_summary}

WHAT WOULD CHANGE THIS VIEW:
List 2-3 specific, falsifiable conditions that would cause a regime shift.
Rules: Be precise and quantitative. No filler. Sound like a macro PM, not a chatbot. One page max.
"""


def generate_fund_note(data: dict) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = FUND_NOTE_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        regime=data.get("regime", "Unknown"),
        growth_trend=data.get("growth_trend", "?"),
        inflation_trend=data.get("inflation_trend", "?"),
        nav=data.get("nav", 0),
        wtd_return=data.get("wtd_return", 0),
        mtd_return=data.get("mtd_return", 0),
        positions=data.get("positions_summary", "No positions"),
        signal_summary=data.get("signal_summary", "No signals"),
    )
    message = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def save_fund_note(note: str, output_dir: str = "docs/fund_notes/") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}{datetime.now().strftime('%Y-%m-%d')}_fund_note.md"
    with open(filename, "w") as f:
        f.write(note)
    return filename