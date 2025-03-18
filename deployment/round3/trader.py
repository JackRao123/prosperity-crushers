from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import pandas as pd
import numpy as np


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
    day = 3  # Change to 3 for IMC backtest

    def __init__(self):  # Calibrated according to PICNIC_BASKET2.
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

            elif product == "VOLCANIC_ROCK_VOUCHER_9750":
                volcanic_rock_9750_position = state.position.get(product, 0)
                VolcanicTrader.update_df(VolcanicTrader.volcanic_rock_9750df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders) and len(VolcanicTrader.volcanic_rock_9750df) > 0:
                    z_vals = pd.Series(np.linspace(-0.015, 0.015, 500))
                    current_row = VolcanicTrader.volcanic_rock_9750df.iloc[-1]
                    vol_fair = 0.61633621 * (current_row["m_t"] ** 2) - 0.45094566 * current_row["m_t"] + 0.14334967
                    current_iv = current_row["implied_vol"]
                    vol_mispricing = (
                        current_iv - vol_fair
                    )  # Measures how high above our fair the current IV is. Could later measure edge based on dollars instead of volpoints.
                    # print(f"TTE: {current_row['TTE']}, Implied Vol: {current_row['implied_vol']}, underlying price: {current_row['underlying_price']}, strike: {current_row['strike']}, m_t: {current_row['m_t']}, delta: {current_row['delta']}, vega: {current_row['vega']}, vol_mispricing: {vol_mispricing}")
                    closest_z = z_vals.iloc[(z_vals - vol_mispricing).abs().idxmin()]
                    entry_position = np.round(volcanic_size_function(closest_z, 0.005, 0.015, 200)).astype(int)
                    exit_position = np.round(volcanic_exit_volcanic_size_function(closest_z, 0.002, 0.01, 200)).astype(int)
                    if vol_mispricing > 0:  # doing separate cases for when the vol mispricing is above and below 0.
                        if volcanic_rock_9750_position > entry_position:  # we are above leaf, need to sell down to it.
                            orders.append(Order(product, current_row["bid_price_1"], -volcanic_rock_9750_position + entry_position))
                        elif volcanic_rock_9750_position < exit_position:  # we are below leaf, need to buy up to it.
                            orders.append(Order(product, current_row["ask_price_1"], -volcanic_rock_9750_position + exit_position))
                    elif vol_mispricing < 0:
                        if volcanic_rock_9750_position < entry_position:  # We are below leaf, we need to buy up to it.
                            orders.append(Order(product, current_row["ask_price_1"], -volcanic_rock_9750_position + entry_position))
                        elif volcanic_rock_9750_position > exit_position:  # We are above leaf, we need to sell down to it.
                            orders.append(Order(product, current_row["bid_price_1"], -volcanic_rock_9750_position + exit_position))

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


class ETFTrader:
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
        ]
    )

    squink_df = pd.DataFrame(
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
    # “If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.”

    # I think this might make knowing which curve to move on easier, but what if the spread jumps from -50 to +50?
    # We will def want to close out. I guess it depends on current position then
    z_ask = ""
    z_bid = ""

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
            "bid_volume_1": bid_levels[0][1],
            "bid_price_2": bid_levels[1][0],
            "bid_volume_2": bid_levels[1][1],
            "bid_price_3": bid_levels[2][0],
            "bid_volume_3": bid_levels[2][1],
            "ask_price_1": ask_levels[0][0],
            "ask_volume_1": ask_levels[0][1],
            "ask_price_2": ask_levels[1][0],
            "ask_volume_2": ask_levels[1][1],
            "ask_price_3": ask_levels[2][0],
            "ask_volume_3": ask_levels[2][1],
            "mid_price": mid_price,
        }

        df.loc[len(df)] = row

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        # Trading Basket 1 against Basket 2: Djembes and B1 have position limits of 60. This should be the limiting
        # B1 = B2 + 2c + J - D
        result = {}
        synthetic_bid = 0  # What I can sell at
        synthetic_offer = 0  # What I can buy at

        best_bid_croissant, best_ask_croissant, best_ask_amount_croissant, best_bid_amount_croissant = 0, 0, 0, 0
        best_bid_djembes, best_ask_djembes, best_ask_amount_djembes, best_bid_amount_djembes = 0, 0, 0, 0
        best_bid_jams, best_ask_jams, best_ask_amount_jams, best_bid_amount_jams = 0, 0, 0, 0
        best_bid_b2, best_ask_b2, best_ask_amount_b2, best_bid_amount_b2 = 0, 0, 0, 0
        best_bid_b1, best_ask_b1, best_ask_amount_b1, best_bid_amount_b1 = 0, 0, 0, 0

        available_sell = 0
        available_buy = 0
        # print(state.position)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            # Basket 2 is 4 croissants, 2 jams
            if product == "CROISSANTS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_croissant, best_ask_amount_croissant = list(order_depth.sell_orders.items())[0]
                    best_bid_croissant, best_bid_amount_croissant = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_croissant * 2
                    synthetic_offer += best_ask_croissant * 2

            if product == "JAMS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_jams, best_ask_amount_jams = list(order_depth.sell_orders.items())[0]
                    best_bid_jams, best_bid_amount_jams = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_jams * 1
                    synthetic_offer += best_ask_jams * 1

            if product == "DJEMBES":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_djembes, best_ask_amount_djembes = list(order_depth.sell_orders.items())[0]
                    best_bid_djembes, best_bid_amount_djembes = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_ask_djembes * 1
                    synthetic_offer += best_bid_djembes * 1

            if product == "PICNIC_BASKET2":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_b2, best_ask_amount_b2 = list(order_depth.sell_orders.items())[0]
                    best_bid_b2, best_bid_amount_b2 = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_b2 * 1
                    synthetic_offer += best_ask_b2 * 1

            if product == "PICNIC_BASKET1":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_b1, best_ask_amount_b1 = list(order_depth.sell_orders.items())[0]
                    best_bid_b1, best_bid_amount_b1 = list(order_depth.buy_orders.items())[0]

        # print(f"Synthetic bid: {synthetic_bid}, Synthetic offer: {synthetic_offer}, Actual bid: {best_bid_b1}, actual offer: {best_ask_b1}")

        available_sell = min(
            int(best_bid_amount_croissant / 2), int(best_bid_amount_jams), int(abs(best_ask_amount_djembes)), best_bid_amount_b2
        )  # How many we can sell synthetica lly
        available_buy = min(
            int(abs(best_ask_amount_croissant) / 2), int(abs(best_ask_amount_jams)), int(abs(best_bid_amount_djembes)), int(abs(best_ask_amount_b2))
        )  # How many we can buy synthetically

        edge_0 = 50
        edge_max = 200

        edge_max_retreet = 130  # T HIS is higher than edge_0
        edge_0_retreet = -25

        position_max = int(60)

        basket_1_position = state.position.get("PICNIC_BASKET1", 0)

        # Redoing entry and exit scheme:
        z_mid = (best_bid_b1 + best_ask_b1) / 2 - (synthetic_bid + synthetic_offer) / 2

        pos_buy = size_function(z_mid, edge_0, edge_max, position_max)
        pos_sell = exit_size_function(z_mid, edge_0_retreet, edge_max_retreet, position_max)

        if z_mid > 0:
            # We only move on the entry curve. Enter more or hold.
            if pos_buy <= basket_1_position:
                # print("On the entry curve")
                target_position = pos_buy
                trade_needed = int(target_position - basket_1_position)
                trade_multiplier = min(abs(trade_needed), available_buy, abs(best_bid_amount_b1))

                result["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_bid_b1, -trade_multiplier)]

                result["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_ask_b2, trade_multiplier)]
                result["JAMS"] = [Order("JAMS", best_ask_jams, 1 * trade_multiplier)]
                result["CROISSANTS"] = [Order("CROISSANTS", best_ask_croissant, 2 * trade_multiplier)]
                result["DJEMBES"] = [Order("DJEMBES", best_ask_djembes, 1 * trade_multiplier)]

            else:
                # print("Not on entry curve")
                target_position = max(pos_sell, min(basket_1_position, 0))

                trade_needed = int(target_position - basket_1_position)
                trade_multiplier = min(abs(trade_needed), available_sell, abs(best_ask_amount_b1))

                result["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_ask_b1, trade_multiplier)]

                result["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_bid_b2, -1 * trade_multiplier)]
                result["JAMS"] = [Order("JAMS", best_bid_jams, -1 * trade_multiplier)]
                result["CROISSANTS"] = [Order("CROISSANTS", best_bid_croissant, -2 * trade_multiplier)]
                result["DJEMBES"] = [Order("DJEMBES", best_bid_djembes, -1 * trade_multiplier)]

        elif z_mid < 0:
            if pos_buy >= basket_1_position:
                # print("On the entry curve")
                target_position = pos_buy
                trade_needed = int(target_position - basket_1_position)
                trade_multiplier = min(abs(trade_needed), available_sell, abs(best_ask_amount_b1))

                result["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_ask_b1, trade_multiplier)]

                result["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_bid_b2, -1 * trade_multiplier)]
                result["JAMS"] = [Order("JAMS", best_bid_jams, -1 * trade_multiplier)]
                result["CROISSANTS"] = [Order("CROISSANTS", best_bid_croissant, -2 * trade_multiplier)]
                result["DJEMBES"] = [Order("DJEMBES", best_bid_djembes, -1 * trade_multiplier)]

            else:
                # print("Not on entry curve")
                target_position = min(pos_sell, max(basket_1_position, 0))
                trade_needed = int(target_position - basket_1_position)
                trade_multiplier = min(abs(trade_needed), available_buy, abs(best_bid_amount_b1))

                result["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_bid_b1, -trade_multiplier)]

                result["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_ask_b2, trade_multiplier)]
                result["JAMS"] = [Order("JAMS", best_ask_jams, 1 * trade_multiplier)]
                result["CROISSANTS"] = [Order("CROISSANTS", best_ask_croissant, 2 * trade_multiplier)]
                result["DJEMBES"] = [Order("DJEMBES", best_ask_djembes, 1 * trade_multiplier)]

        # print(result)

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData


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


class SquinkTrader:
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
    squid_ink_df = pd.DataFrame(
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

    def __init__(self):
        self.retreat_per_lot = 0.03
        self.edge_per_lot = 0.06
        self.edge0 = 0

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            if product == "SQUID_INK":
                squid_ink_position = state.position.get(product, 0)
                SquinkTrader.update_df(SquinkTrader.squid_ink_df, product, state, orders, order_depth)

                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    if len(self.squid_ink_df) >= 2:
                        current_row = self.squid_ink_df.iloc[-1]
                        previous_row = self.squid_ink_df.iloc[-2]
                        current_midprice = (current_row.bid_price_1 + current_row.ask_price_1) / 2
                        previous_midprice = (previous_row.bid_price_1 + previous_row.ask_price_1) / 2
                        current_log_return = np.log(current_midprice) - np.log(previous_midprice)
                        """
                        future_price_prediction = np.exp(np.log(current_midprice) + future_log_return_prediction)
                        """
                        # What if we just set future price prediction to be equal to the mmbot_ midprice?
                        current_mmbot_log_return = np.log(current_row.mmbot_midprice) - np.log(previous_row.mmbot_midprice)
                        future_mmbot_log_return_prediction = 0  # 0.061 * current_mmbot_log_return
                        future_price_prediction = current_row.mmbot_midprice * np.exp(future_mmbot_log_return_prediction)
                        # #print(f"current_mmbt_midprice: {current_row.mmbot_midprice}")
                        # #print(f"future_mmbot_log_return_prediction: {future_mmbot_log_return_prediction}")
                        # #print(f"future_price_prediction: {future_price_prediction}")
                        theo = future_price_prediction - squid_ink_position * self.retreat_per_lot
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
                        bid_volume = min(bid_volume, 50 - squid_ink_position)
                        ask_volume = max(ask_volume, -50 - squid_ink_position)

                        orders.append(Order(product, int(my_bid), int(bid_volume)))
                        orders.append(Order(product, int(my_ask), int(ask_volume)))

            result[product] = orders

        return result, 1, "SAMPLE"


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


# MAIN TRADER
algoresult = dict[str, list[Order]]


class Trader:
    def __init__(self):
        self.resintrader = ResinTrader()
        self.squinktrader = SquinkTrader()
        self.etftrader = ETFTrader()
        self.kelptrader = KelpTrader()
        self.volcanictrader = VolcanicTrader()
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

        combinedresult = self.union([resinresult, squinkresult, etfresult, kelpresult, volcanicresult])
        
        self.clean_result(combinedresult)
        
        return combinedresult, 1, ""
