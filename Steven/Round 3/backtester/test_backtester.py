# regression tests
# Ensure the project root is in PATH.
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # All imports of our code are relative to the project root.

from backtester.backtester import Backtester
from backtester.datamodel import TradingState, OrderDepth, Order, Listing

import numpy as np
import pandas as pd
import sys
import unittest


# Tradder with some arbitrary behaviour.
class TestTrader:
    def __init__(self):
        # config
        self.position_limit = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        self.symbols = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]

        # runtime
        self.max_position = {}
        self.min_position = {}
        pass

    def init_runtime_variables(self, state: TradingState):
        for symbol in self.symbols:
            self.max_position[symbol] = state.position[symbol] if symbol in state.position else 0
            self.min_position[symbol] = state.position[symbol] if symbol in state.position else 0

    def run(self, state: TradingState):
        self.init_runtime_variables(state)

        result = {}
        for product in state.order_depths:
            orders: list[Order] = []

            max_buy_amount = self.position_limit[product] - self.max_position[product]
            max_sell_amount = abs(-self.position_limit[product] - self.min_position[product])
            orderbook = state.order_depths[product]

            if len(orderbook.buy_orders) != 0:
                best_bid_price = max(orderbook.buy_orders.keys())
                orders.append(Order(product, best_bid_price, abs(max_buy_amount)))

            if len(orderbook.sell_orders) != 0:
                best_ask_price = max(orderbook.sell_orders.keys())
                orders.append(Order(product, best_ask_price, -abs(max_sell_amount)))

            result[product] = orders
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData


class TestBacktester(unittest.TestCase):

    # Regression test. Ensure the behaviour of the backtester stays the same as I change it.
    def test_regression_pnl_calculation(self):
        # 1. Define the listings.
        listings = {
            "KELP": Listing(symbol="KELP", product="KELP", denomination="SEASHELLS"),
            "RAINFOREST_RESIN": Listing(symbol="RAINFOREST_RESIN", product="RAINFOREST_RESIN", denomination="SEASHELLS"),
            "SQUID_INK": Listing(symbol="SQUID_INK", product="SQUID_INK", denomination="SEASHELLS"),
        }

        # 2. Define the position limits.
        position_limit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
        }

        # 4. Market data and trade history files.
        market_data_day_0 = pd.read_csv(os.path.join("..", "data", "round1", "prices_round_1_day_0.csv"), sep=";")
        trades_day_0 = pd.read_csv(os.path.join("..", "data", "round1", "trades_round_1_day_0.csv"), sep=";")

        # 5. Instantiate trader object
        trader = TestTrader()
        bt = Backtester(trader, listings, position_limit, market_data_day_0, trades_day_0)
        bt.run()

        kelpmetrics = bt.calculate_metrics("KELP")
        resinmetrics = bt.calculate_metrics("RAINFOREST_RESIN")
        squinkmetrics = bt.calculate_metrics("SQUID_INK")

        # Regression tests
        self.assertEqual(kelpmetrics["spreadcrossing_final_pnl"], -21.0)
        self.assertEqual(resinmetrics["spreadcrossing_final_pnl"], 52)
        self.assertEqual(squinkmetrics["spreadcrossing_final_pnl"], -5349)

        self.assertEqual(kelpmetrics["midpoint_final_pnl"], -78.5)
        self.assertEqual(resinmetrics["midpoint_final_pnl"], -72.0)
        self.assertEqual(squinkmetrics["midpoint_final_pnl"], -5390)


if __name__ == "__main__":
    unittest.main()
    # # 1. Define the listings.
    # listings = {
    #     "KELP": Listing(symbol="KELP", product="KELP", denomination="SEASHELLS"),
    #     "RAINFOREST_RESIN": Listing(symbol="RAINFOREST_RESIN", product="RAINFOREST_RESIN", denomination="SEASHELLS"),
    #     "SQUID_INK": Listing(symbol="SQUID_INK", product="SQUID_INK", denomination="SEASHELLS"),
    # }

    # # 2. Define the position limits.
    # position_limit = {
    #     "KELP": 50,
    #     "RAINFOREST_RESIN": 50,
    #     "SQUID_INK": 50,
    # }

    # # 4. Market data and trade history files.
    # market_data_day_0 = pd.read_csv(os.path.join("..", "data", "round1", "prices_round_1_day_0.csv"), sep=";")
    # trades_day_0 = pd.read_csv(os.path.join("..", "data", "round1", "trades_round_1_day_0.csv"), sep=";")

    # # 5. Instantiate trader object
    # trader = TestTrader()
    # bt = Backtester(trader, listings, position_limit, market_data_day_0, trades_day_0)
    # bt.run()

    # print(bt.pnl)
