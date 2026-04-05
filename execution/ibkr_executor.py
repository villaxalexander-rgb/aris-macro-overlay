"""
Module 3 - IBKR Execution Engine
"""
from datetime import datetime
from ib_insync import IB, Future, MarketOrder, LimitOrder
from config.settings import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID


class IBKRExecutor:
    def __init__(self):
        self.ib = IB()
        self.connected = False

    def connect(self):
        self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
        self.connected = True
        print(f"Connected to IBKR at {IBKR_HOST}:{IBKR_PORT}")

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False

    def get_nav(self):
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidationByCurrency" and av.currency == "BASE":
                return float(av.value)
        return 0.0

    def get_positions(self):
        return self.ib.positions()

    def get_daily_pnl(self):
        pnl = self.ib.pnl()
        if pnl:
            return pnl[0].dailyPnL or 0.0
        return 0.0

    def create_futures_contract(self, symbol, exchange="NYMEX"):
        contract = Future(symbol=symbol, exchange=exchange)
        self.ib.qualifyContracts(contract)
        return contract

    def place_market_order(self, contract, quantity, action="BUY"):
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        return {
            "order_id": trade.order.orderId,
            "symbol": contract.symbol,
            "action": action,
            "quantity": quantity,
            "status": trade.orderStatus.status,
            "fill_price": trade.orderStatus.avgFillPrice,
            "timestamp": datetime.now().isoformat(),
        }

    def place_limit_order(self, contract, quantity, price, action="BUY", timeout_seconds=900):
        order = LimitOrder(action, quantity, price)
        trade = self.ib.placeOrder(contract, order)
        start = datetime.now()
        while trade.orderStatus.status not in ("Filled", "Cancelled"):
            self.ib.sleep(5)
            if (datetime.now() - start).seconds > timeout_seconds:
                self.ib.cancelOrder(order)
                break
        return {
            "order_id": trade.order.orderId,
            "symbol": contract.symbol,
            "action": action,
            "quantity": quantity,
            "limit_price": price,
            "status": trade.orderStatus.status,
            "fill_price": trade.orderStatus.avgFillPrice,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    executor = IBKRExecutor()
    try:
        executor.connect()
        print(f"NAV: ${executor.get_nav():,.2f}")
        print(f"Open positions: {len(executor.get_positions())}")
    finally:
        executor.disconnect()
