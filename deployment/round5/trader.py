from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import pandas as pd
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


def size_function(z, edge_0, edge_max, max_position=50):
    z = np.array(z)
    direction = np.where(z > 0, -1, 1)
    abs_z = np.abs(z)
    size = np.where(abs_z <= edge_0, 0, np.where(abs_z >= edge_max, max_position, max_position * ((abs_z - edge_0) / (edge_max - edge_0)) ** 2))
    return direction * size


def exit_size_function(z, edge_0, edge_max, max_position=50):
    # Positive quadratic function with points (0, 0) and (-2, 50)
    if z <= 0:
        if z >= -edge_0:
            return 0
        elif z <= -edge_max:
            return max_position

        a = -max_position / (edge_max - edge_0) ** 2
        return a * (z + edge_max) ** 2 + max_position
    else:
        if z <= edge_0:
            return 0
        elif z >= edge_max:
            return -max_position
        a = max_position / (edge_max - edge_0) ** 2
        return a * (z - edge_max) ** 2 - max_position


def volcanic_exit_volcanic_size_function(z, edge_0, edge_max, max_position=50):
    # Positive quadratic function with points (0, 0) and (-2, 50)
    if z <= 0:
        if z >= -edge_0:
            return 0
        elif z <= -edge_max:
            return max_position

        a = -max_position / (edge_max - edge_0) ** 2
        return a * (z + edge_max) ** 2 + max_position
    else:
        if z <= edge_0:
            return 0
        elif z >= edge_max:
            return -max_position
        a = max_position / (edge_max - edge_0) ** 2
        return a * (z - edge_max) ** 2 - max_position


def volcanic_size_function(z, edge_0, edge_max, max_position=50):
    z = np.array(z)
    direction = np.where(z > 0, -1, 1)
    abs_z = np.abs(z)
    size = np.where(abs_z <= edge_0, 0, np.where(abs_z >= edge_max, max_position, max_position * ((abs_z - edge_0) / (edge_max - edge_0)) ** 2))
    return direction * size


# Normal PDF and CDF for Black-Scholes
def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def norm_cdf(x):
    return 0.5 * (1 + np.tanh(np.sqrt(np.pi / 8) * x))  # Approximation


# Black-Scholes price
def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# Vega: sensitivity of option price to volatility
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm_pdf(d1) * np.sqrt(T)


# Delta: rate of change of the option price with respect to the underlying asset price
def delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1


# Implied volatility using Newton-Raphson
def implied_vol_newton(market_price, S, K, T, r, option_type="call", tol=1e-6, max_iter=100):
    if market_price is None:
        return np.nan  # Can't compute implied vol without a valid price

    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        if v == 0:
            return np.nan
        sigma -= (price - market_price) / v
        if abs(price - market_price) < tol:
            return sigma
    return np.nan  # Did not converge


class VolcanicTrader:
    day = 5  # Change to 5 for IMC backtest

    def __init__(self):
        self.retreat_per_lot = 0.1
        self.edge_per_lot = 0.2
        self.edge0 = 0

    volcanic_rock_df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
        ]
    )

    volcanic_rock_9500df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "TTE",
            "underlying_price",
            "strike",
            "implied_vol",
            "delta",
            "vega",
            "m_t",
        ]
    )

    volcanic_rock_9750df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "TTE",
            "underlying_price",
            "strike",
            "implied_vol",
            "delta",
            "vega",
            "m_t",
        ]
    )

    volcanic_rock_10000df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "TTE",
            "underlying_price",
            "strike",
            "implied_vol",
            "delta",
            "vega",
            "m_t",
        ]
    )

    volcanic_rock_10250df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "TTE",
            "underlying_price",
            "strike",
            "implied_vol",
            "delta",
            "vega",
            "m_t",
        ]
    )

    volcanic_rock_10500df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "TTE",
            "underlying_price",
            "strike",
            "implied_vol",
            "delta",
            "vega",
            "m_t",
        ]
    )

    # “If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.”
    def update_df(df, product, state, orders, order_depth):
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

        bid_levels = buy_orders[:3] + [(None, None)] * (3 - len(buy_orders))
        ask_levels = sell_orders[:3] + [(None, None)] * (3 - len(sell_orders))

        if bid_levels[0][0] is not None and ask_levels[0][0] is not None:
            mid_price = (bid_levels[0][0] + ask_levels[0][0]) / 2
        else:
            mid_price = None

        row = {
            "timestamp": state.timestamp,
            "product": product,
            "bid_price_1": bid_levels[0][0],
            "bid_volume_1": bid_levels[0][1] or 0,
            "bid_price_2": bid_levels[1][0],
            "bid_volume_2": bid_levels[1][1] or 0,
            "bid_price_3": bid_levels[2][0],
            "bid_volume_3": bid_levels[2][1] or 0,
            "ask_price_1": ask_levels[0][0],
            "ask_volume_1": ask_levels[0][1] or 0,
            "ask_price_2": ask_levels[1][0],
            "ask_volume_2": ask_levels[1][1] or 0,
            "ask_price_3": ask_levels[2][0],
            "ask_volume_3": ask_levels[2][1] or 0,
            "mid_price": mid_price,
        }

        if "VOUCHER" in product:
            row["TTE"] = (7000000 - VolcanicTrader.day * 1000000 - row["timestamp"]) / 365000000
            buy_orders = sorted(state.order_depths["VOLCANIC_ROCK"].buy_orders.items(), key=lambda x: -x[0])
            sell_orders = sorted(state.order_depths["VOLCANIC_ROCK"].sell_orders.items(), key=lambda x: x[0])

            bid_levels = buy_orders[:3] + [(None, None)] * (3 - len(buy_orders))
            ask_levels = sell_orders[:3] + [(None, None)] * (3 - len(sell_orders))

            if bid_levels[0][0] is not None and ask_levels[0][0] is not None:
                mid_price_vr = (bid_levels[0][0] + ask_levels[0][0]) / 2
            else:
                mid_price_vr = None
            row["underlying_price"] = mid_price_vr
            row["strike"] = int(row["product"][22:])
            row["implied_vol"] = implied_vol_newton(row["mid_price"], row["underlying_price"], row["strike"], row["TTE"], 0, option_type="call")
            row["delta"] = delta(row["underlying_price"], row["strike"], row["TTE"], 0, row["implied_vol"], option_type="call")
            row["vega"] = vega(row["underlying_price"], row["strike"], row["TTE"], 0, row["implied_vol"])
            row["m_t"] = np.log(row["strike"] / row["underlying_price"]) / np.sqrt(row["TTE"])
        row = {k: (np.nan if v is None else v) for k, v in row.items()}
        df.loc[len(df)] = row

    def run(self, state):  # Assuming TradingState is defined elsewhere
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            if product == "VOLCANIC_ROCK":
                volcanic_rock_position = state.position.get(product, 0)
                VolcanicTrader.update_df(VolcanicTrader.volcanic_rock_df, product, state, orders, order_depth)

            elif product == "VOLCANIC_ROCK_VOUCHER_10000":
                volcanic_rock_10000_position = state.position.get(product, 0)
                VolcanicTrader.update_df(VolcanicTrader.volcanic_rock_10000df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders) and len(VolcanicTrader.volcanic_rock_10000df) > 0:
                    z_vals = pd.Series(np.linspace(-0.025, 0.025, 500))
                    current_row = VolcanicTrader.volcanic_rock_10000df.iloc[-1]
                    vol_fair = 1.75119318 * (current_row["m_t"] ** 2) + 0.14342421 * current_row["m_t"] + 0.22059232
                    current_iv = current_row["implied_vol"]
                    vol_mispricing = (
                        current_iv - vol_fair
                    )  # Measures how high above our fair the current IV is. Could later measure edge based on dollars instead of volpoints.
                    # print(f"TTE: {current_row['TTE']}, Implied Vol: {current_row['implied_vol']}, underlying price: {current_row['underlying_price']}, strike: {current_row['strike']}, m_t: {current_row['m_t']}, delta: {current_row['delta']}, vega: {current_row['vega']}, vol_mispricing: {vol_mispricing}")
                    closest_z = z_vals.iloc[(z_vals - vol_mispricing).abs().idxmin()]
                    entry_position = np.round(volcanic_size_function(closest_z, 0.00135, 0.025, 200)).astype(int)
                    exit_position = np.round(volcanic_exit_volcanic_size_function(closest_z, 0.0005, 0.02, 200)).astype(int)
                    if vol_mispricing > 0:  # doing separate cases for when the vol mispricing is above and below 0.
                        if volcanic_rock_10000_position > entry_position:  # we are above leaf, need to sell down to it.
                            orders.append(Order(product, current_row["bid_price_1"], -volcanic_rock_10000_position + entry_position))
                        elif volcanic_rock_10000_position < exit_position:  # we are below leaf, need to buy up to it.
                            orders.append(Order(product, current_row["ask_price_1"], -volcanic_rock_10000_position + exit_position))
                    elif vol_mispricing < 0:
                        if volcanic_rock_10000_position < entry_position:  # We are below leaf, we need to buy up to it.
                            orders.append(Order(product, current_row["ask_price_1"], -volcanic_rock_10000_position + entry_position))
                        elif volcanic_rock_10000_position > exit_position:  # We are above leaf, we need to sell down to it.
                            orders.append(Order(product, current_row["bid_price_1"], -volcanic_rock_10000_position + exit_position))

            result[product] = orders

        return result, 1, "SAMPLE"


class ResinTrader:
    def __init__(self):
        # config
        self.position_limit = {"RAINFOREST_RESIN": 50}
        self.symbol = "RAINFOREST_RESIN"

        # self.sk1 = 0.00
        # self.sk2 = 0.00
        # self.sk3 = 1.0
        # self.sk4 = 0.0
        # self.sk5 = 0.00
        # self.bk1 = 0.00
        # self.bk2 = 0.00
        # self.bk3 = 0.75
        # self.bk4 = 0.25
        # self.bk5 = 0.00

        self.sk1 = 0.00
        self.sk2 = 0.00
        self.sk3 = 0.00
        self.sk4 = 0.00
        self.sk5 = 0.00
        self.sk6 = 0.00
        self.sk7 = 1.00

        self.bk1 = 0.00
        self.bk2 = 0.00
        self.bk3 = 0.00
        self.bk4 = 0.00
        self.bk5 = 0.00
        self.bk6 = 0.00
        self.bk7 = 1.00

        # runtime
        self.max_position = 0
        self.min_position = 0
        pass

    # takes +ev orders from the orderbook.
    def take_best_orders(self, state: TradingState, orderbook: OrderDepth) -> list[Order]:
        orders: list[Order] = []

        max_buy_amount = self.position_limit[self.symbol] - self.max_position
        max_sell_amount = abs(-self.position_limit[self.symbol] - self.min_position)

        if len(orderbook.buy_orders) != 0:
            best_bid_price = max(orderbook.buy_orders.keys())
            best_bid_volume = orderbook.buy_orders[best_bid_price]

            if best_bid_price > 10000:
                fill_quantity = min(max_sell_amount, best_bid_volume)

                if fill_quantity > 0:
                    orders.append(Order(self.symbol, best_bid_price, -fill_quantity))
                    del orderbook.buy_orders[best_bid_price]

        if len(orderbook.sell_orders) != 0:
            best_ask_price = min(orderbook.sell_orders.keys())
            best_ask_volume = abs(orderbook.sell_orders[best_ask_price])

            if best_ask_price < 10000:
                fill_quantity = min(max_buy_amount, best_ask_volume)

                if fill_quantity > 0:
                    orders.append(Order(self.symbol, best_ask_price, fill_quantity))
                    del orderbook.sell_orders[best_ask_price]

        return orders

    # puts in some quoting orders
    def add_mm_orders(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []

        max_buy_amount = self.position_limit[self.symbol] - self.max_position
        max_sell_amount = abs(-self.position_limit[self.symbol] - self.min_position)

        portion = max_sell_amount / 7
        sq1 = self.sk1 * portion
        sq2 = self.sk2 * portion
        sq3 = self.sk3 * portion
        sq4 = self.sk4 * portion
        sq5 = self.sk5 * portion
        sq6 = self.sk6 * portion
        sq7 = self.sk7 * (max_sell_amount - 6 * int(portion))

        portion = max_buy_amount / 7
        bq1 = self.bk1 * portion
        bq2 = self.bk2 * portion
        bq3 = self.bk3 * portion
        bq4 = self.bk4 * portion
        bq5 = self.bk5 * portion
        bq6 = self.bk6 * portion
        bq7 = self.bk7 * (max_buy_amount - 6 * int(portion))

        orders.append(Order(self.symbol, 10001, -int(sq1)))
        orders.append(Order(self.symbol, 10002, -int(sq2)))
        orders.append(Order(self.symbol, 10003, -int(sq3)))
        orders.append(Order(self.symbol, 10004, -int(sq4)))
        orders.append(Order(self.symbol, 10005, -int(sq5)))
        orders.append(Order(self.symbol, 10006, -int(sq6)))
        orders.append(Order(self.symbol, 10007, -int(sq7)))

        orders.append(Order(self.symbol, 9999, int(bq1)))
        orders.append(Order(self.symbol, 9998, int(bq2)))
        orders.append(Order(self.symbol, 9997, int(bq3)))
        orders.append(Order(self.symbol, 9996, int(bq4)))
        orders.append(Order(self.symbol, 9995, int(bq5)))
        orders.append(Order(self.symbol, 9994, int(bq6)))
        orders.append(Order(self.symbol, 9993, int(bq7)))

        return orders

    def init_runtime_variables(self, state: TradingState):
        self.max_position = state.position[self.symbol] if self.symbol in state.position else 0
        self.min_position = state.position[self.symbol] if self.symbol in state.position else 0

    def run(self, state: TradingState):
        self.init_runtime_variables(state)

        result = {}
        for product in state.order_depths:
            orders: list[Order] = []

            if product == "RAINFOREST_RESIN":
                took = self.take_best_orders(state, state.order_depths[product])

                while len(took) != 0:
                    orders = orders + took

                    for order in took:
                        if order.quantity > 0:
                            self.max_position += order.quantity
                        elif order.quantity < 0:
                            self.min_position -= abs(order.quantity)

                    took = self.take_best_orders(state, state.order_depths[product])

                took = self.add_mm_orders(state)
                orders = orders + took

                for order in took:
                    if order.quantity > 0:
                        self.max_position += order.quantity
                    elif order.quantity < 0:
                        self.min_position -= abs(order.quantity)

            result[product] = orders
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData


# Olivia squink trader
class SquinkTrader:
    def __init__(self):
        # CONSTANT
        self.position_limit = {SQUID_INK: 50}

        # RUNTIME
        self.last_buysignal_timestamp = -9999999
        self.last_sellsignal_timestamp = -9999999
        self.last_olivia_buyprice = -1
        self.last_olivia_sellprice = -1

        # CONFIG
        self.spread_tol = {SQUID_INK: 5}
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


class KelpTrader:
    kelp_df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "mmbot_bid",
            "mmbot_ask",
            "mmbot_midprice",
        ]
    )

    resin_df = pd.DataFrame(
        columns=[
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
            "mmbot_bid",
            "mmbot_ask",
            "mmbot_midprice",
        ]
    )
    # “If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.”

    def update_df(df, product, state, orders, order_depth):
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

        bid_levels = buy_orders[:3] + [(None, None)] * (3 - len(buy_orders))
        ask_levels = sell_orders[:3] + [(None, None)] * (3 - len(sell_orders))

        if bid_levels[0][0] is not None and ask_levels[0][0] is not None:
            mid_price = (bid_levels[0][0] + ask_levels[0][0]) / 2
        else:
            mid_price = None

        row = {
            "timestamp": state.timestamp,
            "product": product,
            "bid_price_1": bid_levels[0][0],
            "bid_volume_1": bid_levels[0][1] or 0,  # If bid_volume_1 is None, set it to 0
            "bid_price_2": bid_levels[1][0],
            "bid_volume_2": bid_levels[1][1] or 0,  # If bid_volume_2 is None, set it to 0
            "bid_price_3": bid_levels[2][0],
            "bid_volume_3": bid_levels[2][1] or 0,  # If bid_volume_3 is None, set it to 0
            "ask_price_1": ask_levels[0][0],
            "ask_volume_1": ask_levels[0][1] or 0,  # If ask_volume_1 is None, set it to 0
            "ask_price_2": ask_levels[1][0],
            "ask_volume_2": ask_levels[1][1] or 0,  # If ask_volume_2 is None, set it to 0
            "ask_price_3": ask_levels[2][0],
            "ask_volume_3": ask_levels[2][1] or 0,  # If ask_volume_3 is None, set it to 0
            "mid_price": mid_price,
        }

        if row["bid_volume_1"] >= 15:  # Adverse volume set to 15. #mm_bot_bid will just become the top level if there is no adverse volume.
            mm_bot_bid = row["bid_price_1"]
        elif row["bid_volume_2"] >= 15:
            mm_bot_bid = row["bid_price_2"]
        elif row["bid_volume_3"] >= 15:
            mm_bot_bid = row["bid_price_3"]
        else:
            mm_bot_bid = row["bid_price_1"]

        if row["ask_volume_1"] >= 15:  # Adverse volume set to 15. mm_bot_ask will just become the top level if there is no adverse volume.
            mm_bot_ask = row["ask_price_1"]
        elif row["ask_volume_2"] >= 15:
            mm_bot_ask = row["ask_price_2"]
        elif row["ask_volume_3"] >= 15:
            mm_bot_ask = row["ask_price_3"]
        else:
            mm_bot_ask = row["ask_price_1"]

        row["mmbot_bid"] = mm_bot_bid
        row["mmbot_ask"] = mm_bot_ask
        row["mmbot_midprice"] = (mm_bot_bid + mm_bot_ask) / 2

        df.loc[len(df)] = row

    def __init__(self, retreat_per_lot=0.005, edge_per_lot=0.03, edge0=0):
        self.retreat_per_lot = retreat_per_lot
        self.edge_per_lot = edge_per_lot
        self.edge0 = edge0

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            if product == "KELP":
                kelp_position = state.position.get(product, 0)
                KelpTrader.update_df(KelpTrader.kelp_df, product, state, orders, order_depth)

                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    if len(self.kelp_df) >= 2:
                        current_row = self.kelp_df.iloc[-1]
                        previous_row = self.kelp_df.iloc[-2]
                        current_midprice = (current_row.bid_price_1 + current_row.ask_price_1) / 2
                        previous_midprice = (previous_row.bid_price_1 + previous_row.ask_price_1) / 2
                        current_log_return = np.log(current_midprice) - np.log(previous_midprice)

                        ask_pca = (
                            -0.67802679 * (current_row.ask_volume_1 or 0)
                            + 0.73468115 * (current_row.ask_volume_2 or 0)
                            + 0.02287503 * (current_row.ask_volume_3 or 0)
                        )
                        bid_pca = (
                            -0.69827525 * (current_row.bid_volume_1 or 0)
                            + 0.71532596 * (current_row.bid_volume_2 or 0)
                            + 0.02684134 * (current_row.bid_volume_3 or 0)
                        )

                        lag_1_bidvol_return_interaction = bid_pca * current_log_return
                        lag_1_askvol_return_interaction = ask_pca * current_log_return
                        future_log_return_prediction = (
                            -0.0000035249
                            + 0.0000070160 * ask_pca
                            + -0.0000069054 * bid_pca
                            + -0.2087831028 * current_log_return
                            + -0.0064021782 * lag_1_askvol_return_interaction
                            + -0.0049996728 * lag_1_bidvol_return_interaction
                        )
                        """
                        future_price_prediction = np.exp(np.log(current_midprice) + future_log_return_prediction)
                        """
                        # What if we just set future price prediction to be equal to the mmbot_ midprice?
                        current_mmbot_log_return = np.log(current_row.mmbot_midprice) - np.log(previous_row.mmbot_midprice)
                        future_mmbot_log_return_prediction = -0.2933 * current_mmbot_log_return
                        future_price_prediction = current_row.mmbot_midprice * np.exp(future_mmbot_log_return_prediction)
                        # print(f"current_mmbt_midprice: {current_row.mmbot_midprice}")
                        # print(f"future_mmbot_log_return_prediction: {future_mmbot_log_return_prediction}")
                        # print(f"future_price_prediction: {future_price_prediction}")
                        theo = future_price_prediction - kelp_position * self.retreat_per_lot
                        bid_ask_spread = current_row.ask_price_1 - current_row.bid_price_1
                        # Maybe implement some sort of "dime check" that checks if we are diming others and have QP?
                        # Try a strategy where we go as wide as possible whilst still having QP, and not being in cross with our theo.
                        """
                        if bid_ask_spread <= 2:
                            my_bid = min(int(np.floor(theo)), current_row.bid_price_1)
                            my_ask = max(int(np.ceil(theo)), current_row.ask_price_1) #Try quoting wider - maybe if ba spread is wider we want to quote wider - if market is 4 wide maybe we quote 2 wide, else we quote 1 wide?
                        if bid_ask_spread >= 3:
                            my_bid = int(np.floor(theo - 0.5))
                            my_ask = int(np.ceil(theo + 0.5))
                        """
                        # Quoting as wide as possible whilst still having QP, and not being through our theo.
                        my_bid = min(int(np.floor(theo)), current_row.bid_price_1 + 1)
                        my_ask = max(int(np.ceil(theo)), current_row.ask_price_1 - 1)
                        bid_edge = theo - my_bid
                        ask_edge = my_ask - theo
                        edge0 = self.edge0

                        bid_volume = int(np.floor((bid_edge - edge0) / self.edge_per_lot)) if bid_edge > edge0 else 0
                        ask_volume = -int(np.floor((ask_edge - edge0) / self.edge_per_lot)) if ask_edge > edge0 else 0
                        # Below makes sure that we dont send orders over position limits.
                        bid_volume = min(bid_volume, 50 - kelp_position)
                        ask_volume = max(ask_volume, -50 - kelp_position)

                        orders.append(Order(product, int(my_bid), int(bid_volume)))
                        orders.append(Order(product, int(my_ask), int(ask_volume)))

            elif product == "RAINFOREST_RESIN":
                KelpTrader.update_df(KelpTrader.resin_df, product, state, orders, order_depth)

            result[product] = orders

        return result, 1, "SAMPLE"


class MacaronTrader:
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

        # controlling position limits
        max_position = state.position.get(MAGNIFICENT_MACARONS, 0)
        min_position = state.position.get(MAGNIFICENT_MACARONS, 0)

        # market data
        sellorders = state.order_depths[MAGNIFICENT_MACARONS].sell_orders
        buyorders = state.order_depths[MAGNIFICENT_MACARONS].buy_orders
        obs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]

        local_best_ask = min(sellorders)
        local_best_bid = max(buyorders)
        chef_real_ask = obs.askPrice + obs.importTariff + obs.transportFees
        chef_real_bid = obs.bidPrice - obs.exportTariff - obs.transportFees

        # # market taking - (rare)
        for price, qty in sorted(buyorders.items(), reverse=True):
            # sell to them
            if price > chef_real_ask:
                qty_execute = min(qty, abs(-self.position_limit - min_position))
                orders.append(Order(MAGNIFICENT_MACARONS, price, -qty_execute))
                min_position -= qty_execute
            else:
                break

        # market making (modification-  sell always, instead of just selling when we are net 0)
        price_to_quote = local_best_bid + 3
        if min_position < 0:
            price_to_quote = local_best_bid + 3 + (2 * (-min_position))

        orders.append(Order(MAGNIFICENT_MACARONS, int(price_to_quote), -abs(-self.position_limit - min_position)))
        min_position = -self.position_limit

        result[MAGNIFICENT_MACARONS] = orders

        if max_position < 0:
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
                else:
                    conversions = min(10, -max_position)
            else:
                conversions = min(10, -max_position)

        self.bid_price_1.append(local_best_bid)
        self.ask_price_1.append(local_best_ask)
        self.obs_actual_ask.append(chef_real_ask)
        self.obs_actual_bid.append(chef_real_bid)

        return result, conversions, ""


# ETFTrader = olivia crossaint gambler
class ETFTrader:
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


# punting the underlying
class RockTrader:
    def __init__(self):
        self.products = [
            "VOLCANIC_ROCK",
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            # "VOLCANIC_ROCK_VOUCHER_10250",
            # "VOLCANIC_ROCK_VOUCHER_10500",
        ]

        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            # "VOLCANIC_ROCK_VOUCHER_10250": 200,
            # "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        # mean‐reversion params
        self.window = {
            "VOLCANIC_ROCK": 150,
            "VOLCANIC_ROCK_VOUCHER_9500": 125,
            "VOLCANIC_ROCK_VOUCHER_9750": 125,
            # "VOLCANIC_ROCK_VOUCHER_10250": 75,
            # "VOLCANIC_ROCK_VOUCHER_10500": 75,
        }

        self.z_threshold = {
            "VOLCANIC_ROCK": 1.7,
            "VOLCANIC_ROCK_VOUCHER_9500": 1.3,
            "VOLCANIC_ROCK_VOUCHER_9750": 1.3,
            # "VOLCANIC_ROCK_VOUCHER_10250": 1.7,
            # "VOLCANIC_ROCK_VOUCHER_10500": 1.7,
        }

        #  runtime
        self.histories = {p: [] for p in self.products}

    def _get_mid_price(self, state: TradingState, product: str) -> float:
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        return (max(buy_orders) + min(sell_orders)) / 2

    def _get_bid_ask(self, state: TradingState, product: str):
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        if len(buy_orders) == 0 or len(sell_orders) == 0:
            return None, None

        return max(buy_orders), min(sell_orders)

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        for prod in self.products:
            best_bid, best_ask = self._get_bid_ask(state, prod)
            if best_bid == None or best_ask == None:
                continue

            midprice = self._get_mid_price(state, prod)
            self.histories[prod].append(midprice)
            hist = self.histories[prod]

            # only compute z‐score once we have enough data
            if len(hist) >= self.window[prod]:
                window = hist[-self.window[prod] :]
                mu = np.mean(window)
                sigma = np.std(window)

                if sigma > 0:
                    z = (midprice - mu) / sigma

                    orders = []
                    position = state.position.get(prod, 0)

                    # sell
                    if z > self.z_threshold[prod] and position > -self.position_limits[prod]:
                        qty = self.position_limits[prod] + position
                        orders.append(Order(prod, best_bid, -qty))

                    # buy!
                    elif z < -self.z_threshold[prod] and position < self.position_limits[prod]:
                        qty = self.position_limits[prod] - position
                        orders.append(Order(prod, best_ask, qty))

                    result[prod] = orders

        return result, conversions, state.traderData


# MAIN TRADER
algoresult = dict[str, list[Order]]


class Trader:
    def __init__(self):
        self.resintrader = ResinTrader()
        self.squinktrader = SquinkTrader()
        self.etftrader = ETFTrader()
        self.kelptrader = KelpTrader()
        self.volcanictrader = VolcanicTrader()  # 10000
        self.rocktrader = RockTrader()  # rock underlying.
        self.macarontrader = MacaronTrader()
        pass

    def union(self, results: list[algoresult]) -> algoresult:
        combined: dict[str, list[Order]] = {}

        for result in results:
            for product, orders in result.items():
                if product not in combined:
                    combined[product] = []
                combined[product].extend(orders)

        return combined

    def clean_result(self, result: algoresult):
        for symbol, orders in result.items():
            for order in orders:
                order.price = int(order.price)
                order.quantity = int(order.quantity)

    def run(self, state: TradingState) -> algoresult:
        resinresult, _, _ = self.resintrader.run(state)
        squinkresult, _, _ = self.squinktrader.run(state)
        etfresult, _, _ = self.etftrader.run(state)
        kelpresult, _, _ = self.kelptrader.run(state)
        volcanicresult, _, _ = self.volcanictrader.run(state)
        macaronresult, macaronconversions, _ = self.macarontrader.run(state)

        # new !
        rockresult, _, _ = self.rocktrader.run(state)

        combinedresult = self.union([resinresult, squinkresult, etfresult, kelpresult, volcanicresult, macaronresult, rockresult])

        self.clean_result(combinedresult)
        return combinedresult, macaronconversions, ""
