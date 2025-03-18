from datamodel import OrderDepth, UserId, TradingState, Order
import pandas as pd
import numpy as np


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

    # not used?
    # resin_df = pd.DataFrame(columns=[
    #     "timestamp", "product",
    #     "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    #     "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    #     "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice'
    # ])
    # “If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.”

    # HARDCODED PARAMETERS
    def __init__(self):
        # config
        self.position_limit = {"RAINFOREST_RESIN": 50}
        self.resinsymbol = "RAINFOREST_RESIN"

        # PARAMETERS - RESIN
        self.sk1 = 0.00
        self.sk2 = 0.00
        self.sk3 = 1.0
        self.sk4 = 0.0
        self.sk5 = 0.00
        self.bk1 = 0.00
        self.bk2 = 0.00
        self.bk3 = 0.75
        self.bk4 = 0.25
        self.bk5 = 0.00

        # PARAMETERS - KELP
        self.retreat_per_lot = 0.012
        self.edge_per_lot = 0.015
        self.edge0 = 0.02

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

        portion = max_sell_amount / 5
        sq1 = self.sk1 * portion
        sq2 = self.sk2 * portion
        sq3 = self.sk3 * portion
        sq4 = self.sk4 * portion
        sq5 = self.sk5 * (max_sell_amount - 4 * int(portion))

        portion = max_buy_amount / 5
        bq1 = self.bk1 * portion
        bq2 = self.bk2 * portion
        bq3 = self.bk3 * portion
        bq4 = self.bk4 * portion
        bq5 = self.bk5 * (max_buy_amount - 4 * int(portion))

        orders.append(Order(self.resinsymbol, 10001, -int(sq1)))
        orders.append(Order(self.resinsymbol, 10002, -int(sq2)))
        orders.append(Order(self.resinsymbol, 10003, -int(sq3)))
        orders.append(Order(self.resinsymbol, 10004, -int(sq4)))
        orders.append(Order(self.resinsymbol, 10005, -int(sq5)))

        orders.append(Order(self.resinsymbol, 9999, int(bq1)))
        orders.append(Order(self.resinsymbol, 9998, int(bq2)))
        orders.append(Order(self.resinsymbol, 9997, int(bq3)))
        orders.append(Order(self.resinsymbol, 9996, int(bq4)))
        orders.append(Order(self.resinsymbol, 9995, int(bq5)))

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

                        orders.append(Order(product, int(my_ask), int(ask_volume)))
                        orders.append(Order(product, int(my_bid), int(bid_volume)))
            elif product == "SQUID_INK":
                # Check the rolling z score. If it is > 4, we sell,
                squink_position = state.position.get(product, 0)
                Trader.SQUINK_update_df(Trader.squink_df, product, state, orders, order_depth)

                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    mid_price = (best_ask + best_bid) / 2
                    rolling_window = 150
                    if len(self.squink_df) > rolling_window:

                        z_score = (mid_price - self.squink_df.mid_price.rolling(rolling_window).mean().iloc[-1]) / self.squink_df.mid_price.rolling(
                            rolling_window
                        ).std().iloc[-1]
                        # Now that I have the Z-Score, I want to implement an edge model for entry

                        edge_0 = 2

                        if abs(z_score) > edge_0:
                            target_position = min(int((np.abs(z_score) - edge_0) * 10), 30)
                        else:
                            target_position = 0

                        if z_score <= -edge_0:
                            # print("A")
                            target_position = max(target_position, squink_position)
                            size_needed = target_position - squink_position
                            if size_needed < 0:
                                pass
                            else:
                                orders.append(Order(product, int(best_ask), int(size_needed)))
                            # orders.append(Order(product, best_bid+1, size_needed))

                        elif z_score >= edge_0:
                            # print("C")
                            target_position = min(-target_position, squink_position)
                            size_needed = target_position - squink_position
                            if size_needed > 0:
                                pass
                            else:
                                orders.append(Order(product, int(best_bid), int(size_needed)))
                                # orders.append(Order(product, best_ask - 1, size_needed))
                        # Now what if it is between?

                        else:
                            if squink_position > 0:
                                if z_score > 0:
                                    target_position = 0
                                    size_needed = target_position - squink_position
                                    orders.append(Order(product, int(best_bid), int(size_needed)))
                                    # orders.append(Order(product, best_ask - 1, size_needed))

                                else:
                                    exit_position = min(squink_position, int(-15 * z_score))
                                    trades_needed = exit_position - squink_position
                                    orders.append(Order(product, int(best_bid), int(trades_needed)))
                                    # orders.append(Order(product, best_ask-1, trades_needed))

                            if squink_position < 0:
                                if z_score < 0:
                                    target_position = 0
                                    size_needed = target_position - squink_position
                                    orders.append(Order(product, int(best_ask), int(size_needed)))
                                    # orders.append(Order(product, best_bid + 1, size_needed))
                                else:
                                    exit_position = min(squink_position, int(15 * z_score))
                                    trades_needed = exit_position - squink_position
                                    orders.append(Order(product, int(best_ask), int(trades_needed)))
                                    # orders.append(Order(product, best_bid + 1, trades_needed))

                        # Entry Attempt to fix sizing scheme above
                        """
                        if z_score < 0:
                            if z_score <= -edge_0:
                                print("A")
                                target_position = max(target_position, squink_position)
                                size_needed = target_position - squink_position
                                if size_needed < 0:
                                    pass
                                else:
                                    orders.append(Order(product, best_ask, size_needed))
                                    
                            elif z_score > -2 and z_score <= 0 and squink_position > 0:
                                print("B")
                                exit_position = min(squink_position, int(-15*z_score))
                                trades_needed = exit_position - squink_position
                                orders.append(Order(product, best_bid, -trades_needed))
                                
                        elif z_score > 0:        
                            if z_score >= edge_0:
                                print("C")
                                target_position = min(-target_position, squink_position)
                                size_needed = target_position - squink_position
                                if size_needed > 0:
                                    pass
                                else:
                                    orders.append(Order(product, best_bid, size_needed))

                            if z_score < 2 and z_score >=0 and squink_position < 0:
                                print("D")
                                exit_position = min(squink_position, int(15*z_score))
                                trades_needed = exit_position - squink_position
                                orders.append(Order(product, best_ask, trades_needed)) 
                            """

            result[product] = orders
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
