"""
Module 5 - Jeffrey Morning Briefing
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


def generate_fund_note(data):
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


def save_fund_note(note, output_dir="docs/fund_notes/"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}{datetime.now().strftime('%Y-%m-%d')}_fund_note.md"
    with open(filename, "w") as f:
        f.write(note)
    return filename


if __name__ == "__main__":
    test_data = {
        "regime": "Reflation", "growth_trend": "up", "inflation_trend": "up",
        "nav": 250000, "wtd_return": 1.2, "mtd_return": 3.4,
        "positions_summary": "Long CL (2 lots)\nShort NG (1 lot)",
        "signal_summary": "CL: +0.65 | GC: +0.42 | NG: -0.38 | ZC: +0.21",
    }
    note = generate_fund_note(test_data)
    print(note)
    save_fund_note(note)
