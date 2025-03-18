from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


class Trader:
    def __init__(self):  # , k):
        self.k = 2

        self.position_limit = {"KELP": 50, "RAINFOREST_RESIN": 50}
        pass

    # Given a tradingstate, and a list of current_orders we already plan to execute, return whether or not we can add an additional order 'new_order'.
    def _can_add_order(self, state: TradingState, current_orders: list[Order], new_order: Order):
        max_possible_pos = state.position.copy()
        min_possible_pos = state.position.copy()

        if "RAINFOREST_RESIN" not in max_possible_pos:
            max_possible_pos["RAINFOREST_RESIN"] = 0
        if "RAINFOREST_RESIN" not in min_possible_pos:
            min_possible_pos["RAINFOREST_RESIN"] = 0

        for order in current_orders:
            if order.quantity < 0:
                min_possible_pos[order.symbol] += order.quantity
            elif order.quantity > 0:
                max_possible_pos[order.symbol] += order.quantity

        if new_order.quantity > 0:
            max_possible_pos[new_order.symbol] += new_order.quantity
        elif new_order.quantity < 0:
            min_possible_pos[new_order.symbol] += new_order.quantity

        for symbol, qty in max_possible_pos.items():
            if abs(qty) > self.position_limit[symbol]:
                return False

        for symbol, qty in min_possible_pos.items():
            if abs(qty) > self.position_limit[symbol]:
                return False

        return True

    def calculate_desired_position(self, state: TradingState):
        order_book: OrderDepth = state.order_depths["RAINFOREST_RESIN"]

        best_ask_price, best_ask_vol = min(order_book.sell_orders, key=lambda item: item[0])
        best_bid_price, best_bid_vol = max(order_book.buy_orders, key=lambda item: item[0])

        mid_price = (best_ask_price + best_bid_price) / 2

        # want long (+50) at 9995
        # want short (-50) at 10005

        desired_position = 100000 - mid_price * 10

        return desired_position

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            orders: list[Order] = []

            if product == "RAINFOREST_RESIN":
                # lets only buy at 9998 and only sell at 10002
                pos = 0
                if product in state.position:
                    pos = state.position[product]

                orders.append(Order(product, 10000 - self.k, 50 - pos))
                orders.append(Order(product, 10000 + self.k, -50 - pos))
                # Some sizing scheme where I do 2 lots at each successive price level?
                # This means I show 2 at 10000 +- 1, 4 at 10000 +- 2, 6 at 10000 +- 3 etc etc.
                # Maybe 25 is too much and the entire order gets cancelled
                # for i in range(1, 5 + 1, 1):

                #     if self._can_add_order(state, orders, Order(product, 10000 + i, -self.k * i)):
                #         orders.append(Order(product, 10000 + i, int(-self.k * i)))
                #     if self._can_add_order(state, orders, Order(product, 10000 - i, self.k * i)):
                #         orders.append(Order(product, 10000 - i, int(self.k * i)))

                # desired_position = self.calculate_desired_position(state)
                # current_position = state.position['RAINFOREST_RESIN']

            result[product] = orders
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
