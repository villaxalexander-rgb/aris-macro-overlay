"""
Nav Peak + Max Drawdown Tracker

Maintains state/nav_peak.json with:
  - nav_peak: high-water mark since inception
  - peak_date: date of the peak
  - current_drawdown_pct: (nav_peak - current_nav) / nav_peak
  - max_drawdown_pct: worst drawdown ever observed
  - max_drawdown_date: when it occurred
  - history: last 30 daily snapshots

Called at the start of each pipeline run (after fetching NAV from IBKR).
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "nav_peak.json"


def load_state() -> dict:
    """Load existing drawdown state, or initialize empty."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "nav_peak": 0.0,
        "peak_date": None,
        "current_drawdown_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_date": None,
        "history": [],
    }


def update(current_nav: float, date_str: Optional[str] = None) -> dict:
    """
    Update the drawdown tracker with today's NAV.

    Returns the updated state dict (also persisted to disk).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    state = load_state()

    # Update high-water mark
    if current_nav > state["nav_peak"]:
        state["nav_peak"] = current_nav
        state["peak_date"] = date_str

    # Compute current drawdown
    if state["nav_peak"] > 0:
        dd_pct = (state["nav_peak"] - current_nav) / state["nav_peak"] * 100
    else:
        dd_pct = 0.0
    state["current_drawdown_pct"] = round(dd_pct, 4)

    # Update max drawdown
    if dd_pct > state["max_drawdown_pct"]:
        state["max_drawdown_pct"] = round(dd_pct, 4)
        state["max_drawdown_date"] = date_str

    # Append to rolling history (keep last 30 days)
    state["history"].append({
        "date": date_str,
        "nav": current_nav,
        "drawdown_pct": round(dd_pct, 4),
    })
    state["history"] = state["history"][-30:]

    # Persist
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    return state


def get_summary() -> str:
    """One-line summary for logging / Jeffrey briefing."""
    state = load_state()
    if state["nav_peak"] == 0:
        return "Drawdown tracker: no data yet"
    return (
        f"NAV peak: €{state['nav_peak']:,.0f} ({state['peak_date']}) | "
        f"Current DD: {state['current_drawdown_pct']:.2f}% | "
        f"Max DD: {state['max_drawdown_pct']:.2f}% ({state['max_drawdown_date']})"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nav peak / drawdown tracker")
    parser.add_argument("--nav", type=float, help="Current NAV to record")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--show", action="store_true", help="Show current state")
    args = parser.parse_args()

    if args.show:
        state = load_state()
        print(json.dumps(state, indent=2))
        print(f"\n{get_summary()}")
    elif args.nav:
        state = update(args.nav, args.date)
        print(f"Updated: {get_summary()}")
    else:
        parser.print_help()
