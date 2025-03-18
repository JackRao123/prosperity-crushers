from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation

import numpy as np


CROISSANTS = "CROISSANTS"
DJEMBES = "DJEMBES"
JAMS = "JAMS"
KELP = "KELP"
PICNIC_BASKET1 = "PICNIC_BASKET2"
PICNIC_BASKET2 = "PICNIC_BASKET2"
RAINFOREST_RESIN = "RAINFOREST_RESIN"
SQUID_INK = "SQUID_INK"
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

CAMILLA = "Camilla"
CAESAR
PABLO = "Pablo"
SUBMISSION = "SUBMISSION"


class Trader:
    def __init__(self):
        # CONSTANT
        self.position_limit = {PICNIC_BASKET2: 100}

        # RUNTIME
        self.last_buysignal_timestamp = -9999999
        self.last_sellsignal_timestamp = -9999999

        # CONFIG
        self.spread_tol = 10000  # how much we are willing to cross the spread buy to execute
        self.signal_timeout = 1300  # time until we stop moving directionally on a previous signal
        self.signal_volume_threshold = 1  # the threshold of difference in camilla's buy, sell volume to take it as a signal
        pass

    def run(self, state: TradingState):
        result = {}
        orders = []
        if PICNIC_BASKET2 in state.order_depths:
            buy_orders = state.order_depths[PICNIC_BASKET2].buy_orders
            sell_orders = state.order_depths[PICNIC_BASKET2].sell_orders

            # From the previous timestamp
            market_trades = state.market_trades.get(PICNIC_BASKET2, [])
            camilla_buy_vol = 0
            camilla_sell_vol = 0
            for trade in market_trades:
                if trade.buyer == CAMILLA and trade.seller == PABLO:
                    camilla_buy_vol += trade.quantity
                elif trade.buyer == PABLO and trade.seller == CAMILLA:
                    camilla_sell_vol += trade.quantity

            # volume threshold for signal
            if camilla_buy_vol - camilla_sell_vol >= self.signal_volume_threshold:
                self.last_buysignal_timestamp = state.timestamp
            elif camilla_sell_vol - camilla_buy_vol >= self.signal_volume_threshold:
                self.last_sellsignal_timestamp = state.timestamp

            time_since_last_buysignal = state.timestamp - self.last_buysignal_timestamp
            time_since_last_sellsignal = state.timestamp - self.last_sellsignal_timestamp

            min_position = state.position.get(PICNIC_BASKET2, 0)
            max_position = state.position.get(PICNIC_BASKET2, 0)
            midprice = (max(buy_orders) + min(sell_orders)) / 2

            if time_since_last_buysignal < time_since_last_sellsignal and time_since_last_buysignal < self.signal_timeout:
                # buy signal
                for price, qty in sorted(sell_orders.items()):
                    qty_executed = min(abs(qty), self.position_limit[PICNIC_BASKET2] - max_position)

                    if qty_executed > 0 and price <= midprice + self.spread_tol:
                        orders.append(Order(PICNIC_BASKET2, price, qty_executed))
                        max_position += qty_executed

                # # limit order
                # qty_remaining = self.position_limit[PICNIC_BASKET2] - max_position
                # if qty_remaining != 0:
                #     orders.append(Order(PICNIC_BASKET2, int(np.floor(midprice)) + 2, qty_remaining))

            elif time_since_last_sellsignal < time_since_last_buysignal and time_since_last_sellsignal < self.signal_timeout:
                # sell signal
                for price, qty in sorted(buy_orders.items(), reverse=True):
                    qty_executed = min(qty, abs(-self.position_limit[PICNIC_BASKET2] - min_position))

                    if qty_executed > 0 and price >= midprice - self.spread_tol:
                        orders.append(Order(PICNIC_BASKET2, price, -qty_executed))
                        min_position -= qty_executed

                # # limit order
                # qty_remaining = abs(-self.position_limit[PICNIC_BASKET2] - min_position)
                # if qty_remaining != 0:
                #     orders.append(Order(PICNIC_BASKET2, int(np.ceil(midprice)) - 2, -qty_remaining))

            result[PICNIC_BASKET2] = orders

        return result, 0, ""
