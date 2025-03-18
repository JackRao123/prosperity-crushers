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


class Trader:
    def __init__(self):
        # CONSTANT
        self.position_limit = {CROISSANTS: 250, DJEMBES: 60, JAMS: 350, PICNIC_BASKET1: 60, PICNIC_BASKET2: 100}

        # we either want to be limit long or limit short.
        # limit long: c=360,j=-350,d=-60,pb1=60,pb2=100
        # net: +1010c, +30j, +0d
        # delta exposure of 30j

        # limit short: c=-360,j=350,d=60,pb1=-60,pb2=-100
        # net: -1010c, -30j, +0d
        # delta exposure of 30j

        # RUNTIME
        self.last_buysignal_timestamp = -9999999
        self.last_sellsignal_timestamp = -9999999
        self.last_olivia_buyprice = -1
        self.last_olivia_sellprice = -1

        # CONFIG
        self.spread_tol = {CROISSANTS: 4, DJEMBES: 4, JAMS: 4, PICNIC_BASKET1: 11, PICNIC_BASKET2: 11}
        self.signal_timeout = 2000

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

        # limit order
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
        if CROISSANTS in state.order_depths:
            market_trades = state.market_trades.get(CROISSANTS, [])

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
                # limit long croissants
                result[CROISSANTS] = self._execute_buy(state, CROISSANTS, 250)
                result[JAMS] = self._execute_sell(state, JAMS, -350)
                result[DJEMBES] = self._execute_sell(state, DJEMBES, -60)
                result[PICNIC_BASKET1] = self._execute_buy(state, PICNIC_BASKET1, 60)
                result[PICNIC_BASKET2] = self._execute_buy(state, PICNIC_BASKET2, 100)

            elif time_since_last_sellsignal < time_since_last_buysignal and time_since_last_sellsignal < self.signal_timeout:
                # limit short croissants
                result[CROISSANTS] = self._execute_sell(state, CROISSANTS, -250)
                result[JAMS] = self._execute_buy(state, JAMS, 350)
                result[DJEMBES] = self._execute_buy(state, DJEMBES, 60)
                result[PICNIC_BASKET1] = self._execute_sell(state, PICNIC_BASKET1, -60)
                result[PICNIC_BASKET2] = self._execute_sell(state, PICNIC_BASKET2, -100)

        return result, 0, ""
