from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


# Position limits for the newly introduced products:

# - `RAINFOREST_RESIN`: 50
# - `KELP`: 50


class Trader:

    # “If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.”

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # This algorithm trades only kelp.
            # Profits from Kelp will be independent from profits from resin
            if product == "KELP":

                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    # Just undercut them by 1.

                    if best_ask - best_bid > 2:
                        orders.append(Order(product, best_ask - 1, -50))
                        orders.append(Order(product, best_bid - 1, 50))

            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
