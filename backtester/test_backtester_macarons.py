# regression tests
# Ensure the project root is in PATH.
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # All imports of our code are relative to the project root.

from backtester.engine import Backtester
from backtester.datamodel import TradingState, OrderDepth, Order, Listing

import numpy as np
import pandas as pd
import sys
import unittest

MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


class TestTrader:
    def __init__(self):
        # config
        pass

    def run(self, state: TradingState):
        result = {}
        orders = []

        position = state.position.get(MAGNIFICENT_MACARONS, 0)
        desired_position = -10

        conversions = 0
        sellorders = state.order_depths[MAGNIFICENT_MACARONS].sell_orders
        buyorders = state.order_depths[MAGNIFICENT_MACARONS].buy_orders

        obs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]

        chef_real_ask = obs.askPrice + obs.importTariff + obs.transportFees
        chef_real_bid = obs.bidPrice - obs.exportTariff - obs.transportFees

        if position > desired_position:
            diff = abs(desired_position - position)
            orders.append(Order(MAGNIFICENT_MACARONS, max(buyorders), -diff))

        for price, quantity in sorted(buyorders.items(), reverse=True):
            if price > chef_real_ask:
                # lets buy from chefs, and sell to local.
                conversions += quantity
                orders.append(Order(MAGNIFICENT_MACARONS, price, -abs(quantity)))

        result[MAGNIFICENT_MACARONS] = orders
        return result, conversions, ""


class TestBacktester(unittest.TestCase):
    def test_macarons(self):
        # 1. Define the listings.
        listings = {
            "MAGNIFICENT_MACARONS": Listing(symbol="MAGNIFICENT_MACARONS", product="MAGNIFICENT_MACARONS", denomination="SEASHELLS"),
        }

        # 2. Define the position limits.
        position_limit = {
            "MAGNIFICENT_MACARONS": 75,
        }

        # 4. Market data and trade history files.
        market_data_round_4_day_3 = pd.read_csv(os.path.join("..", "data", "round4", "prices_round_4_day_3.csv"), sep=";")
        trades_round_4_day_3 = pd.read_csv(os.path.join("..", "data", "round4", "trades_round_4_day_3.csv"), sep=";")
        observations_round_4_day_3 = pd.read_csv(os.path.join("..", "data", "round4", "observations_round_4_day_3.csv"), sep=",")

        # 5. Instantiate trader object
        trader = TestTrader()
        bt = Backtester(trader, listings, position_limit, market_data_round_4_day_3, trades_round_4_day_3, observations_round_4_day_3)
        bt.run()


        metrics = bt.calculate_metrics(MAGNIFICENT_MACARONS)
        print(metrics['spreadcrossing_final_pnl'])


if __name__ == "__main__":
    unittest.main()
