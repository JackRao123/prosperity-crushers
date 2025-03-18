from datamodel import TradingState, Trade, OrderDepth, Order, Observation, ConversionObservation, Listing
import numpy as np
import pandas as pd


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


class Trader:

    position_limit = {
        CROISSANTS: 250,
        DJEMBES: 60,
        JAMS: 350,
        KELP: 50,
        PICNIC_BASKET1: 60,
        PICNIC_BASKET2: 100,
        RAINFOREST_RESIN: 50,
        SQUID_INK: 50,
        VOLCANIC_ROCK: 400,
        VOLCANIC_ROCK_VOUCHER_10000: 200,
        VOLCANIC_ROCK_VOUCHER_10250: 200,
        VOLCANIC_ROCK_VOUCHER_10500: 200,
        VOLCANIC_ROCK_VOUCHER_9500: 200,
        VOLCANIC_ROCK_VOUCHER_9750: 200,
    }

    def __init__(self):
        self.basket1_spread_history = []
        self.basket2_spread_history = []
        pass

    def get_mid_price(self, order_depths: dict[str, OrderDepth], symbol: str) -> float:

        return (max(order_depths[symbol].buy_orders) + min(order_depths[symbol].sell_orders)) / 2

    def get_best_bid_best_ask(self, order_depths: dict[str, OrderDepth], symbol: str) -> tuple[int, int]:

        return max(order_depths[symbol].buy_orders), min(order_depths[symbol].sell_orders)

    # Get the required orders to long basket2 ETF and short NAV for 1 unit of ETF.

    # Buys at market

    def get_redeem_basket2_orders(self, order_depths: dict[str, OrderDepth]) -> list[Order]:

        # buy 1x basket2

        # sell 4x croissant

        # sell 2x jam

        orders = []

        croissant_bestbid, croissant_bestask = self.get_best_bid_best_ask(order_depths, CROISSANTS)

        jam_bestbid, jam_bestask = self.get_best_bid_best_ask(order_depths, JAMS)

        basket2_bestbid, basket2_bestask = self.get_best_bid_best_ask(order_depths, PICNIC_BASKET2)

        orders.append(Order(PICNIC_BASKET2, basket2_bestask, 1))

        orders.append(Order(CROISSANTS, croissant_bestbid, -4))

        orders.append(Order(JAMS, jam_bestbid, -2))

        return orders

    # Get the required orders to short basket2 ETF and long NAV for 1 unit of ETF.

    # Buys at market

    def get_create_basket2_orders(self, order_depths: dict[str, OrderDepth]) -> list[Order]:

        # sell 1x basket2

        # buy 4x croissant

        # buy 2x jam

        orders = []

        croissant_bestbid, croissant_bestask = self.get_best_bid_best_ask(order_depths, CROISSANTS)

        jam_bestbid, jam_bestask = self.get_best_bid_best_ask(order_depths, JAMS)

        basket2_bestbid, basket2_bestask = self.get_best_bid_best_ask(order_depths, PICNIC_BASKET2)

        orders.append(Order(PICNIC_BASKET2, basket2_bestbid, -1))

        orders.append(Order(CROISSANTS, croissant_bestask, 4))

        orders.append(Order(JAMS, jam_bestask, 2))

        return orders

    def sort_orders(self, orders: list[Order]) -> dict[str, list[Order]]:

        result = {}

        for order in orders:

            if order.symbol not in result:

                result[order.symbol] = []

            result[order.symbol].append(order)

        return result

    # Calculate the desired amount we want to be long/short the spread.

    # long spread = (long ETF short NAV)

    # short spread = (short ETF long NAV)

    def calculate_desired_net(self, state: TradingState) -> int:

        basket1_price = self.get_mid_price(state.order_depths, PICNIC_BASKET1)

        basket2_price = self.get_mid_price(state.order_depths, PICNIC_BASKET2)

        croissants_price = self.get_mid_price(state.order_depths, CROISSANTS)

        djembes_price = self.get_mid_price(state.order_depths, DJEMBES)

        jams_price = self.get_mid_price(state.order_depths, JAMS)

        synthetic_basket1_price = 6 * croissants_price + 3 * jams_price + 1 * djembes_price

        synthetic_basket2_price = 4 * croissants_price + 2 * jams_price

        # spread = ETF - NAV

        basket1_spread = basket1_price - synthetic_basket1_price

        basket2_spread = basket2_price - synthetic_basket2_price

        self.basket1_spread_history.append(basket1_spread)
        self.basket2_spread_history.append(basket2_spread)

        return int(-(basket2_spread) * 0.8)

        # if basket2_spread > 100:

        #     return -40

        # if basket2_spread > 50:

        #     return -20

        # if basket2_spread < -100:
        #     return 40

        # if basket2_spread < -50:
        #     return 20

        return 0

    # true if violate
    # false if not violate
    def _can_add_orders(self, state: TradingState, existing_orders: list[Order], new_orders: list[Order]) -> bool:
        max_position = state.position
        min_position = state.position

        orders = existing_orders.copy()
        orders.extend(new_orders)

        for order in orders:
            if order.quantity > 0:
                max_position[order.symbol] = order.quantity + max_position.get(order.symbol, 0)
            elif order.quantity < 0:
                min_position[order.symbol] = -abs(order.quantity) + min_position.get(order.symbol, 0)

        for symbol, pos in max_position.items():
            if pos > self.position_limit[symbol]:
                return False

        for symbol, pos in min_position.items():
            if pos < -abs(self.position_limit[symbol]):
                return False
        return True

    def run(self, state: TradingState):

        result = {}

        orders: list[Order] = []

        # Lets start by just trading basket 2 for now.

        # Sizing should be relative to the edge. Just like with our previous algorithms.

        # net picnic baskets
        net = self.calculate_desired_net(state)
        change = net - state.position.get(PICNIC_BASKET2, 0)

        if change > 10:
            # We want to redeem
            new_orders = self.get_redeem_basket2_orders(state.order_depths)
            for _ in range(change):
                if self._can_add_orders(state, orders, new_orders):
                    orders.extend(new_orders)
                else:
                    break
        elif change < -10:
            # We want to create
            new_orders = self.get_create_basket2_orders(state.order_depths)
            for _ in range(abs(change)):
                if self._can_add_orders(state, orders, new_orders):
                    orders.extend(new_orders)
                else:
                    break

        result = self.sort_orders(orders)

        traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData
