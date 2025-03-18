from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation
import numpy as np


MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


class Trader:
    def __init__(self):
        # config
        self.position_limit = 75
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

        # handling position limits
        min_pos = position
        max_pos = position

        if position > desired_position:
            diff = abs(desired_position - position)
            orders.append(Order(MAGNIFICENT_MACARONS, max(buyorders), -diff))
            min_pos -= abs(diff)

        # for price, quantity in sorted(buyorders.items(), reverse=True):
        #     if price > chef_real_ask:
        #         # lets buy from chefs, and sell to local.
        #         conversions += quantity
        #         orders.append(Order(MAGNIFICENT_MACARONS, price, -abs(quantity)))

        # market make
        # we buy 10 from the chefs and lets sell a lot, at a price higher than this
        # if we convert, we end up buying at chef_real_ask

        price = np.ceil(chef_real_ask) + 1
        while min_pos > -self.position_limit:
            qty = min(50, abs(-self.position_limit - min_pos))
            if price > chef_real_ask:
                orders.append(Order(MAGNIFICENT_MACARONS, int(price), -abs(qty)))
                min_pos -= abs(qty)
            price += 1

        result[MAGNIFICENT_MACARONS] = orders
        return result, min(10, abs(self.position_limit - max_pos)), ""
