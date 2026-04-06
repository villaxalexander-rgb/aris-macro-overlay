"""
A.R.I.S — IBKR Connection Test
Connects to IB Gateway paper trading and pulls account data.
Run with: python tests/test_ibkr_connection.py
"""
from ib_insync import IB, util
import sys

# Paper trading defaults
HOST = "127.0.0.1"
PORT = 4002
CLIENT_ID = 1


def test_connection():
    ib = IB()

    print("=" * 50)
    print("A.R.I.S — IBKR Connection Test")
    print("=" * 50)
    print(f"\nConnecting to IB Gateway at {HOST}:{PORT} ...")

    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
    except Exception as e:
        print(f"\n✗ CONNECTION FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Is IB Gateway running?")
        print("  2. Edit → Global Configuration → API → Settings:")
        print("     - 'Enable ActiveX and Socket Clients' checked")
        print("     - Socket port = 4002")
        print("     - 'Allow connections from localhost only' checked")
        print("     - 'Read-Only API' unchecked")
        print("  3. Are you logged into your paper trading account?")
        sys.exit(1)

    print("✓ Connected to IB Gateway\n")

    # --- Account summary ---
    print("-" * 50)
    print("ACCOUNT SUMMARY")
    print("-" * 50)

    account_values = ib.accountSummary()
    key_fields = {
        "NetLiquidation": "NAV (Net Liquidation)",
        "TotalCashValue": "Cash",
        "GrossPositionValue": "Gross Position Value",
        "MaintMarginReq": "Maintenance Margin",
        "AvailableFunds": "Available Funds",
        "BuyingPower": "Buying Power",
        "UnrealizedPnL": "Unrealized P&L",
        "RealizedPnL": "Realized P&L",
    }

    for av in account_values:
        if av.tag in key_fields:
            label = key_fields[av.tag]
            try:
                val = float(av.value)
                print(f"  {label:<25} ${val:>15,.2f}  ({av.currency})")
            except ValueError:
                print(f"  {label:<25} {av.value}")

    # --- Positions ---
    print(f"\n{'-' * 50}")
    print("OPEN POSITIONS")
    print("-" * 50)

    positions = ib.positions()
    if not positions:
        print("  No open positions (expected for fresh paper account)")
    else:
        for pos in positions:
            print(f"  {pos.contract.symbol:<8} {pos.position:>8.1f} lots  "
                  f"avg cost: ${pos.avgCost:>10,.2f}")

    # --- Daily P&L ---
    print(f"\n{'-' * 50}")
    print("P&L")
    print("-" * 50)

    pnl = ib.pnl()
    if pnl:
        for p in pnl:
            print(f"  Daily P&L:      ${p.dailyPnL:>12,.2f}" if p.dailyPnL else "  Daily P&L:      N/A")
            print(f"  Unrealized P&L: ${p.unrealizedPnL:>12,.2f}" if p.unrealizedPnL else "  Unrealized P&L: N/A")
            print(f"  Realized P&L:   ${p.realizedPnL:>12,.2f}" if p.realizedPnL else "  Realized P&L:   N/A")
    else:
        print("  P&L data not available yet")

    # --- Connection info ---
    print(f"\n{'-' * 50}")
    print("CONNECTION INFO")
    print("-" * 50)

    managed = ib.managedAccounts()
    print(f"  Managed accounts: {managed}")
    print(f"  Server version:   {ib.client.serverVersion()}")
    print(f"  Connected:        {ib.isConnected()}")

    print(f"\n{'=' * 50}")
    print("✓ All checks passed — A.R.I.S can talk to IBKR")
    print("=" * 50)

    ib.disconnect()
    return True


if __name__ == "__main__":
    test_connection()
