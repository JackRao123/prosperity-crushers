# # Ensure the project root is in PATH.
# import sys

# sys.path.append("../")
# # All imports of our code are relative to the project root.


# from backtester.backtester import Backtester
# from backtester.datamodel import TradingState, OrderDepth, Order, Listing


from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    def __init__(self):
        self.position_limit = {"KELP": 50, "RAINFOREST_RESIN": 50}
        pass

    def take_best_orders(self, state: TradingState, orderbook: OrderDepth) -> list[Order]:
        orders: list[Order] = []

        symbol = "RAINFOREST_RESIN"
        position = state.position[symbol] if symbol in state.position else 0

        max_buy_amount = self.position_limit[symbol] - position
        max_sell_amount = abs(-self.position_limit[symbol] - position)

        if len(orderbook.buy_orders) != 0:
            best_bid_price = max(orderbook.buy_orders.keys())
            best_bid_volume = orderbook.buy_orders[best_bid_price]

            if best_bid_price > 10000:
                fill_quantity = min(max_sell_amount, best_bid_volume)

                if fill_quantity > 0:
                    orders.append(Order(symbol, best_bid_price, -fill_quantity))
                    del orderbook.buy_orders[best_bid_price]

        if len(orderbook.sell_orders) != 0:
            best_ask_price = min(orderbook.sell_orders.keys())
            best_ask_volume = abs(orderbook.sell_orders[best_ask_price])

            if best_ask_price < 10000:
                fill_quantity = min(max_buy_amount, best_ask_volume)

                if fill_quantity > 0:
                    orders.append(Order(symbol, best_ask_price, fill_quantity))
                    del orderbook.sell_orders[best_ask_price]

        return orders

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            orders: list[Order] = []

            if product == "RAINFOREST_RESIN":
                took = self.take_best_orders(state, state.order_depths[product])

                while len(took) != 0:
                    orders = orders + took
                    took = self.take_best_orders(state, state.order_depths[product])

            result[product] = orders
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
