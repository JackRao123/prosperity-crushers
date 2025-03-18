from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation

MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


class Trader:
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
