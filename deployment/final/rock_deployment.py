from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import pandas as pd
import numpy as np

VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"


class Trader:
    def __init__(self):
        self.products = [
            "VOLCANIC_ROCK",
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            # "VOLCANIC_ROCK_VOUCHER_10250",
            # "VOLCANIC_ROCK_VOUCHER_10500",
        ]

        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            # "VOLCANIC_ROCK_VOUCHER_10250": 200,
            # "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        # mean‐reversion params
        self.window = {
            "VOLCANIC_ROCK": 150,
            "VOLCANIC_ROCK_VOUCHER_9500": 125,
            "VOLCANIC_ROCK_VOUCHER_9750": 125,
            # "VOLCANIC_ROCK_VOUCHER_10250": 75,
            # "VOLCANIC_ROCK_VOUCHER_10500": 75,
        }

        self.z_threshold = {
            "VOLCANIC_ROCK": 1.7,
            "VOLCANIC_ROCK_VOUCHER_9500": 1.3,
            "VOLCANIC_ROCK_VOUCHER_9750": 1.3,
            # "VOLCANIC_ROCK_VOUCHER_10250": 1.7,
            # "VOLCANIC_ROCK_VOUCHER_10500": 1.7,
        }

        #  runtime
        self.histories = {p: [] for p in self.products}

    def _get_mid_price(self, state: TradingState, product: str) -> float:
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        return (max(buy_orders) + min(sell_orders)) / 2

    def _get_bid_ask(self, state: TradingState, product: str):
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        if len(buy_orders) == 0 or len(sell_orders) == 0:
            return None, None

        return max(buy_orders), min(sell_orders)

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        for prod in self.products:
            best_bid, best_ask = self._get_bid_ask(state, prod)
            if best_bid == None or best_ask == None:
                continue

            midprice = self._get_mid_price(state, prod)
            self.histories[prod].append(midprice)
            hist = self.histories[prod]

            # only compute z‐score once we have enough data
            if len(hist) >= self.window[prod]:
                window = hist[-self.window[prod] :]
                mu = np.mean(window)
                sigma = np.std(window)

                if sigma > 0:
                    z = (midprice - mu) / sigma

                    orders = []
                    position = state.position.get(prod, 0)

                    # sell
                    if z > self.z_threshold[prod] and position > -self.position_limits[prod]:
                        qty = self.position_limits[prod] + position
                        orders.append(Order(prod, best_bid, -qty))

                    # buy!
                    elif z < -self.z_threshold[prod] and position < self.position_limits[prod]:
                        qty = self.position_limits[prod] - position
                        orders.append(Order(prod, best_ask, qty))

                    result[prod] = orders

        return result, conversions, state.traderData
