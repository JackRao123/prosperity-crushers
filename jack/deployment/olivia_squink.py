from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation

import numpy as np


CROISSANTS = "CROISSANTS"
DJEMBES = "DJEMBES"
JAMS = "JAMS"
KELP = "KELP"
PICNIC_BASKET1 = "PICNIC_BASKET1"
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
PABLO = "Pablo"
SUBMISSION = "SUBMISSION"


class OliviaSquink:
    def __init__(self):
        # CONSTANT
        self.position_limit = {SQUID_INK: 50}

        # RUNTIME
        self.last_buysignal_timestamp = -9999999
        self.last_sellsignal_timestamp = -9999999
        self.last_olivia_buyprice = -1
        self.last_olivia_sellprice = -1

        # CONFIG
        self.spread_tol = {SQUID_INK: 5} #doesn't matter
        self.signal_timeout = 2500
        pass

    def _get_mid_price(self, state: TradingState, product: str) -> float:
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        return (max(buy_orders) + max(sell_orders)) / 2

    def _execute_buy(self, state: TradingState, product: str, desired_pos: int):
        orders = []
        midprice = self._get_mid_price(state, product)
        position = state.position.get(product, 0)

        for price, qty in sorted(state.order_depths[product].sell_orders.items()):
            qty_executed = min(abs(qty), desired_pos - position)

            if qty_executed > 0 and price <= midprice + self.spread_tol[product]:
                orders.append(Order(product, price, qty_executed))
                position += qty_executed

        # limit order - this is to reduce spread cost. but we lose so little in spread that this doesn't matter
        # it just helps us reach our desired position faster though. not worth the time tuning this thingy.
        qty_remaining = desired_pos - position
        if qty_remaining != 0:
            orders.append(Order(product, int(np.floor(midprice)), qty_remaining))

        return orders

    def _execute_sell(self, state: TradingState, product: str, desired_pos: int):
        orders = []
        midprice = self._get_mid_price(state, product)
        position = state.position.get(product, 0)

        for price, qty in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            qty_executed = min(qty, abs(desired_pos - position))

            if qty_executed > 0 and price >= midprice - self.spread_tol[product]:
                orders.append(Order(product, price, -qty_executed))
                position -= qty_executed

        # limit order
        qty_remaining = abs(desired_pos - position)
        if qty_remaining != 0:
            orders.append(Order(product, int(np.ceil(midprice)), -qty_remaining))

        return orders

    def run(self, state: TradingState):
        result = {}
        if SQUID_INK in state.order_depths:
            market_trades = state.market_trades.get(SQUID_INK, [])

            olivia_buy_vol = 0
            olivia_sell_vol = 0
            for trade in market_trades:
                if trade.buyer == "Olivia" and trade.seller != "Olivia":
                    olivia_buy_vol += trade.quantity
                elif trade.seller == "Olivia" and trade.buyer != "Olivia":
                    olivia_sell_vol += trade.quantity

            if olivia_buy_vol > olivia_sell_vol:
                self.last_buysignal_timestamp = state.timestamp
                self.last_olivia_buyprice = trade.price
            elif olivia_buy_vol < olivia_sell_vol:
                self.last_sellsignal_timestamp = state.timestamp
                self.last_olivia_sellprice = trade.price

            time_since_last_buysignal = state.timestamp - self.last_buysignal_timestamp
            time_since_last_sellsignal = state.timestamp - self.last_sellsignal_timestamp

            if time_since_last_buysignal < time_since_last_sellsignal and time_since_last_buysignal < self.signal_timeout:
                # limit long
                result[SQUID_INK] = self._execute_buy(state, SQUID_INK, self.position_limit[SQUID_INK])

            elif time_since_last_sellsignal < time_since_last_buysignal and time_since_last_sellsignal < self.signal_timeout:
                # limit short
                result[SQUID_INK] = self._execute_sell(state, SQUID_INK, -self.position_limit[SQUID_INK])

        return result, 0, ""