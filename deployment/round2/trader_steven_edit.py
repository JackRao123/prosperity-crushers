from datamodel import OrderDepth, UserId, TradingState, Order
import pandas as pd
import numpy as np

def size_function(z, edge_0, edge_max, max_position = 50):
    z = np.array(z)
    direction = np.where(z > 0, -1, 1)
    abs_z = np.abs(z)
    size = np.where(
        abs_z <= edge_0,
        0,
        np.where(
            abs_z >= edge_max,
            max_position,
            max_position * ((abs_z - edge_0) / (edge_max - edge_0)) ** 2
        )
    )
    return direction * size

def exit_size_function(z, edge_0, edge_max, max_position = 50):
    # Positive quadratic function with points (0, 0) and (-2, 50)
    if z <= 0:
        if z >= -edge_0:
            return 0
        elif z <= -edge_max:
            return max_position
            
        a = -max_position/(edge_max - edge_0)**2
        return a * (z + edge_max)**2 + max_position
    else:
        if z <= edge_0:
            return 0
        elif z >= edge_max:
            return -max_position
        a = max_position/(edge_max - edge_0)**2
        return a * (z-edge_max)**2 - max_position


class Trader:
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

    # HARDCODED PARAMETERS
    def __init__(self):
        # config
        self.position_limit = {"RAINFOREST_RESIN": 50}
        self.resinsymbol = "RAINFOREST_RESIN"

        # PARAMETERS - RESIN
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

        # PARAMETERS - KELP
        self.kelp_retreat_per_lot = 0.012
        self.kelp_edge_per_lot = 0.015
        self.kelp_edge0 = 0.02

        # PARAMETERS - SQUINK
        self.squink_retreat_per_lot = 0.005
        self.squink_edge_per_lot = 0.03
        self.squink_edge0 = 0

        # runtime
        self.resin_max_position = 0
        self.resin_min_position = 0
        pass

    def SQUINK_update_df(df, product, state, orders, order_depth):
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

    def KELP_update_df(df, product, state, orders, order_depth):
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

    # takes +ev orders from the orderbook.
    def RESIN_take_best_orders(self, state: TradingState, orderbook: OrderDepth) -> list[Order]:
        orders: list[Order] = []

        max_buy_amount = self.position_limit[self.resinsymbol] - self.resin_max_position
        max_sell_amount = abs(-self.position_limit[self.resinsymbol] - self.resin_min_position)

        if len(orderbook.buy_orders) != 0:
            best_bid_price = max(orderbook.buy_orders.keys())
            best_bid_volume = orderbook.buy_orders[best_bid_price]

            if best_bid_price > 10000:
                fill_quantity = min(max_sell_amount, best_bid_volume)

                if fill_quantity > 0:
                    orders.append(Order(self.resinsymbol, best_bid_price, -fill_quantity))
                    del orderbook.buy_orders[best_bid_price]

        if len(orderbook.sell_orders) != 0:
            best_ask_price = min(orderbook.sell_orders.keys())
            best_ask_volume = abs(orderbook.sell_orders[best_ask_price])

            if best_ask_price < 10000:
                fill_quantity = min(max_buy_amount, best_ask_volume)

                if fill_quantity > 0:
                    orders.append(Order(self.resinsymbol, best_ask_price, fill_quantity))
                    del orderbook.sell_orders[best_ask_price]

        return orders

    # puts in some quoting orders
    def RESIN_add_mm_orders(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []

        max_buy_amount = self.position_limit[self.resinsymbol] - self.resin_max_position
        max_sell_amount = abs(-self.position_limit[self.resinsymbol] - self.resin_min_position)

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

        orders.append(Order(self.resinsymbol, 10001, -int(sq1)))
        orders.append(Order(self.resinsymbol, 10002, -int(sq2)))
        orders.append(Order(self.resinsymbol, 10003, -int(sq3)))
        orders.append(Order(self.resinsymbol, 10004, -int(sq4)))
        orders.append(Order(self.resinsymbol, 10005, -int(sq5)))
        orders.append(Order(self.resinsymbol, 10006, -int(sq6)))
        orders.append(Order(self.resinsymbol, 10007, -int(sq7)))

        orders.append(Order(self.resinsymbol, 9999, int(bq1)))
        orders.append(Order(self.resinsymbol, 9998, int(bq2)))
        orders.append(Order(self.resinsymbol, 9997, int(bq3)))
        orders.append(Order(self.resinsymbol, 9996, int(bq4)))
        orders.append(Order(self.resinsymbol, 9995, int(bq5)))
        orders.append(Order(self.resinsymbol, 9994, int(bq6)))
        orders.append(Order(self.resinsymbol, 9993, int(bq7)))

        return orders

    def RESIN_init_runtime_variables(self, state: TradingState):
        self.resin_max_position = state.position[self.resinsymbol] if self.resinsymbol in state.position else 0
        self.resin_min_position = state.position[self.resinsymbol] if self.resinsymbol in state.position else 0

    def run(self, state: TradingState):
        self.RESIN_init_runtime_variables(state)

        result = {}

        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            if product == "RAINFOREST_RESIN":
                took = self.RESIN_take_best_orders(state, state.order_depths[product])

                while len(took) != 0:
                    orders = orders + took

                    for order in took:
                        if order.quantity > 0:
                            self.resin_max_position += order.quantity
                        elif order.quantity < 0:
                            self.resin_min_position -= abs(order.quantity)

                    took = self.RESIN_take_best_orders(state, state.order_depths[product])

                took = self.RESIN_add_mm_orders(state)
                orders = orders + took

                for order in took:
                    if order.quantity > 0:
                        self.resin_max_position += order.quantity
                    elif order.quantity < 0:
                        self.resin_min_position -= abs(order.quantity)
            elif product == "KELP":
                kelp_position = state.position.get(product, 0)
                Trader.KELP_update_df(Trader.kelp_df, product, state, orders, order_depth)

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
                        theo = future_price_prediction - kelp_position * self.kelp_retreat_per_lot
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
                        edge0 = self.kelp_edge0

                        bid_volume = int(np.floor((bid_edge - edge0) / self.kelp_edge_per_lot)) if bid_edge > edge0 else 0
                        ask_volume = -int(np.floor((ask_edge - edge0) / self.kelp_edge_per_lot)) if ask_edge > edge0 else 0
                        # Below makes sure that we dont send orders over position limits.
                        bid_volume = min(bid_volume, 50 - kelp_position)
                        ask_volume = max(ask_volume, -50 - kelp_position)

                        orders.append(Order(product, int(my_ask), int(ask_volume)))
                        orders.append(Order(product, int(my_bid), int(bid_volume)))
            elif product == "SQUID_INK":
                squid_ink_position = state.position.get(product, 0)
                Trader.SQUINK_update_df(Trader.squid_ink_df, product, state, orders, order_depth)

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
                        # print(f"current_mmbt_midprice: {current_row.mmbot_midprice}")
                        # print(f"future_mmbot_log_return_prediction: {future_mmbot_log_return_prediction}")
                        # print(f"future_price_prediction: {future_price_prediction}")
                        theo = future_price_prediction - squid_ink_position * self.squink_retreat_per_lot
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
                        edge0 = self.squink_edge0

                        bid_volume = int(np.floor((bid_edge - edge0) / self.squink_edge_per_lot)) if bid_edge > edge0 else 0
                        ask_volume = -int(np.floor((ask_edge - edge0) / self.squink_edge_per_lot)) if ask_edge > edge0 else 0
                        # Below makes sure that we dont send orders over position limits.
                        bid_volume = min(bid_volume, 50 - squid_ink_position)
                        ask_volume = max(ask_volume, -50 - squid_ink_position)

                        orders.append(Order(product, my_bid, bid_volume))
                        orders.append(Order(product, my_ask, ask_volume))
            result[product] = orders
            
        synthetic_bid = 0    # What I can sell at
        synthetic_offer = 0    # What I can buy at
        best_ask = 0
        best_bid = 0
        best_ask_jam = 0
        best_bid_jam = 0
        best_bid_croissant = 0
        best_ask_croissant = 0
        available_sell = 0
        available_buy = 0
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            # Basket 2 is 4 croissants, 2 jams
            if product == "CROISSANTS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_croissant, best_ask_amount_croissant = list(order_depth.sell_orders.items())[0]
                    best_bid_croissant, best_bid_amount_croissant = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_croissant * 4
                    synthetic_offer += best_ask_croissant * 4

            if product == "JAMS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_jam, best_ask_amount_jam = list(order_depth.sell_orders.items())[0]
                    best_bid_jam, best_bid_amount_jam = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_jam * 2
                    synthetic_offer += best_ask_jam * 2

            if product == 'PICNIC_BASKET2':
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        available_sell = min(int(best_bid_amount_croissant / 4), int(best_bid_amount_jam / 2))   # How many we can sell synthetica lly
        available_buy = min(int(abs(best_ask_amount_croissant) / 4), int(abs(best_ask_amount_jam) / 2))    # How many we can buy synthetically


        edge_0 = 40
        edge_max = 125
        
        edge_max_retreet = 128
        edge_0_retreet = 33
        position_max = int(250/4)

        basket_2_position = state.position.get("PICNIC_BASKET2", 0)


        z_mid = (best_bid + best_ask)/2 - (synthetic_bid + synthetic_offer)/2
        pos_buy = size_function(z_mid, edge_0, edge_max, position_max)
        pos_sell = exit_size_function(z_mid, edge_0_retreet, edge_max_retreet, position_max)

        if z_mid > 0:           
            if pos_buy <= basket_2_position:
                target_position = pos_buy
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_buy, abs(best_bid_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_bid, -trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_ask_jam, 2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_ask_croissant, 4 * trade_multiplier)]

            else:
                target_position = max(pos_sell, min(basket_2_position, 0))
                
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_sell, abs(best_ask_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_ask, trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_bid_jam, -2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_bid_croissant, -4 * trade_multiplier)]
        
        elif z_mid < 0:
            if pos_buy >= basket_2_position:
                target_position = pos_buy
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_sell,abs(best_ask_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_ask, trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_bid_jam, -2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_bid_croissant, -4 * trade_multiplier)]

            else:
                target_position = min(pos_sell, max(basket_2_position, 0))
                
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_buy,abs(best_bid_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_bid, -trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_ask_jam, 2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_ask_croissant, 4 * trade_multiplier)]


        
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
