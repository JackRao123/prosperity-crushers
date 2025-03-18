from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation


MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


class Trader:
    def __init__(self):
        # config
        self.position_limit = 75

        self.bid_price_1 = []
        self.ask_price_1 = []
        self.obs_actual_ask = []
        self.obs_actual_bid = []

        pass

    def run(self, state: TradingState):
        result = {}
        orders = []
        conversions = 0

        for symbol, trades in state.market_trades.items():
            print(trades)

        # controlling position limits
        position = state.position.get(MAGNIFICENT_MACARONS, 0)

        # market data
        sellorders = state.order_depths[MAGNIFICENT_MACARONS].sell_orders
        buyorders = state.order_depths[MAGNIFICENT_MACARONS].buy_orders
        obs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]

        local_best_ask = min(sellorders)
        local_best_bid = max(buyorders)
        chef_real_ask = obs.askPrice + obs.importTariff + obs.transportFees
        chef_real_bid = obs.bidPrice - obs.exportTariff - obs.transportFees

        # market taking - (rare)
        for price, qty in sorted(buyorders.items(), reverse=True):
            if price > chef_real_ask:
                qty_execute = min(qty, abs(-self.position_limit - position))
                orders.append(Order(MAGNIFICENT_MACARONS, price, -qty_execute))
                position -= qty_execute
            else:
                break

        # sell always, instead of just selling when we are net 0
        price_to_quote = local_best_bid + 4
        if position < 0:
            price_to_quote = local_best_bid + 4 + (0.5 + 2 * (-position))

        orders.append(Order(MAGNIFICENT_MACARONS, int(price_to_quote), -abs(-self.position_limit - position)))
        result[MAGNIFICENT_MACARONS] = orders

        if position < 0:
            if len(self.obs_actual_ask) != 0:
                prev_ask = self.obs_actual_ask[-1]
                prev_bid = self.obs_actual_bid[-1]
                prev_spread = prev_ask - prev_bid
                curr_spread = chef_real_ask - chef_real_bid

                # when it spikes a little and is a bit more expensive then dont buy
                # based on the data it stays quite consistent except for little spikes.
                # check spreading.ipynb for spread research
                if curr_spread > prev_spread:
                    conversions = 0
                    print("DONT BUY")
                else:
                    conversions = min(10, -position)
            else:
                conversions = min(10, -position)

        self.bid_price_1.append(local_best_bid)
        self.ask_price_1.append(local_best_ask)
        self.obs_actual_ask.append(chef_real_ask)
        self.obs_actual_bid.append(chef_real_bid)

        return result, conversions, ""
