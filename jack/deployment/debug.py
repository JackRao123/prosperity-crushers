from collections import defaultdict
# # Ensure the project root is in PATH.
# import sys

# sys.path.append("../")
# # All imports of our code are relative to the project root.


# from backtester.backtester import Backtester
# from backtester.datamodel import TradingState, OrderDepth, Order, Listing


from datamodel import OrderDepth, UserId, TradingState, Order


VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"


class Trader:
    def __init__(self):
        pass

    def run(self, state: TradingState):
        result = {}
        if state.position.get(VOLCANIC_ROCK_VOUCHER_10000, 0) == 0:
            if len(state.order_depths[VOLCANIC_ROCK_VOUCHER_10000].sell_orders) != 0:
                best_ask = min(state.order_depths[VOLCANIC_ROCK_VOUCHER_10000].sell_orders)
                result[VOLCANIC_ROCK_VOUCHER_10000] = [Order(VOLCANIC_ROCK_VOUCHER_10000, best_ask, 1)]

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        conversions = 1
        return result, conversions, traderData
