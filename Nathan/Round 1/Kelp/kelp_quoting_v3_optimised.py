

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

import pandas as pd
import numpy as np

import string



class Trader:
    retreat_per_lot	= 0.01
    edge_per_lot = 0.02
    edge0 = 0.10
    kelp_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss"
    ])

    resin_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss"
    ])
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
            "bid_price_1": bid_levels[0][0], "bid_volume_1": bid_levels[0][1],
            "bid_price_2": bid_levels[1][0], "bid_volume_2": bid_levels[1][1],
            "bid_price_3": bid_levels[2][0], "bid_volume_3": bid_levels[2][1],
            "ask_price_1": ask_levels[0][0], "ask_volume_1": ask_levels[0][1],
            "ask_price_2": ask_levels[1][0], "ask_volume_2": ask_levels[1][1],
            "ask_price_3": ask_levels[2][0], "ask_volume_3": ask_levels[2][1],
            "mid_price": mid_price,
        }

        df.loc[len(df)] = row

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "KELP":
                kelp_position = state.position.get(product, 0)
                Trader.update_df(Trader.kelp_df, product, state, orders, order_depth)

                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    if len(self.kelp_df) >= 2:
                        current_row = self.kelp_df.iloc[-1]
                        previous_row = self.kelp_df.iloc[-2]
                        current_midprice = (current_row.bid_price_1 + current_row.ask_price_1) / 2
                        previous_midprice = (previous_row.bid_price_1 + previous_row.ask_price_1) / 2
                        current_log_return = np.log(current_midprice) - np.log(previous_midprice)

                        ask_pca = (-0.67802679 * (current_row.ask_volume_1 or 0) + 
                                    0.73468115 * (current_row.ask_volume_2 or 0) + 
                                    0.02287503 * (current_row.ask_volume_3 or 0))
                        bid_pca = (-0.69827525 * (current_row.bid_volume_1 or 0) + 
                                    0.71532596 * (current_row.bid_volume_2 or 0) + 
                                    0.02684134 * (current_row.bid_volume_3 or 0))

                        lag_1_bidvol_return_interaction = bid_pca * current_log_return
                        lag_1_askvol_return_interaction = ask_pca * current_log_return
                        future_log_return_prediction = (
                            -0.0000035249 + 0.0000070160 * ask_pca +
                            -0.0000069054 * bid_pca +
                            -0.2087831028 * current_log_return +
                            -0.0064021782 * lag_1_askvol_return_interaction +
                            -0.0049996728 * lag_1_bidvol_return_interaction
                        )

                        future_price_prediction = np.exp(np.log(current_midprice) + future_log_return_prediction)
                        theo = future_price_prediction - kelp_position * self.retreat_per_lot

                        my_bid = int(np.floor(theo))
                        my_ask = int(np.ceil(theo))
                        bid_edge = theo - my_bid
                        ask_edge = my_ask - theo

                        bid_volume = int(np.floor((bid_edge - self.edge0) / self.edge_per_lot)) if bid_edge > self.edge0 else 0
                        ask_volume = -int(np.floor((ask_edge - self.edge0) / self.edge_per_lot)) if ask_edge > self.edge0 else 0

                        bid_volume = min(bid_volume, 50 - kelp_position)
                        ask_volume = max(ask_volume, -50 - kelp_position)
                        orders.append(Order(product, my_bid, bid_volume))
                        orders.append(Order(product, my_ask, ask_volume))

            elif product == "RAINFOREST_RESIN":
                Trader.update_df(Trader.resin_df, product, state, orders, order_depth)

            result[product] = orders

        return result, 1, "SAMPLE"
