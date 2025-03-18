
import os
import sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import List, Dict, Any, Callable
import matplotlib.pyplot as plt
import os
import sys

from typing import List

import pandas as pd
import numpy as np

import string

#Test out a strategy where we just quote every timestamp.
# Position limits for the newly introduced products:

# - `RAINFOREST_RESIN`: 50
# - `KELP`: 50


class Trader:

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
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # This algorithm trades only kelp.
            # Profits from Kelp will be independent from profits from resin
            if product == "KELP":
                kelp_position = state.position.get(product, 0)
                Trader.update_df(Trader.kelp_df, product, state, orders, order_depth) 

                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    current_row = self.kelp_df.iloc[-1]
                    theo = (current_row.bid_price_1 + current_row.ask_price_1) / 2  - kelp_position * 0.2
                    orders.append(Order(product, int(np.floor(theo), 1)))
                    orders.append(Order(product, int(np.ceil(theo), -1))) 
                    '''
                    if len(self.kelp_df) >= 2:
                        current_row = self.kelp_df.iloc[-1]
                        previous_row = self.kelp_df.iloc[-2]
                        current_midprice = (current_row.bid_price_1 + current_row.ask_price_1) / 2
                        previous_midprice = (previous_row.bid_price_1 + previous_row.ask_price_1) / 2
                        current_log_return = np.log(current_midprice) - np.log(previous_midprice)
                        ask_pca = (-0.681533 * (current_row.ask_volume_1 if current_row.ask_volume_1 is not None else 0) + 
           0.731268 * (current_row.ask_volume_2 if current_row.ask_volume_2 is not None else 0) + 
           0.027555 * (current_row.ask_volume_3 if current_row.ask_volume_3 is not None else 0))

                        bid_pca = (0.707528 * (current_row.bid_volume_1 if current_row.bid_volume_1 is not None else 0) + 
           -0.706056 * (current_row.bid_volume_2 if current_row.bid_volume_2 is not None else 0) + 
           -0.029818 * (current_row.bid_volume_3 if current_row.bid_volume_3 is not None else 0))

                        lag_1_bidvol_return_interaction = bid_pca * current_log_return
                        lag_1_askvol_return_interaction = ask_pca * current_log_return
                        future_log_return_prediction =-0.0000085558 + 0.0000065986 * ask_pca + 0.0000058835 * bid_pca -0.2270522958 * current_log_return -0.0052232755* lag_1_askvol_return_interaction + 0.0038240796 * lag_1_bidvol_return_interaction
                        standardised_log_return_prediction = (future_log_return_prediction)/0.000214
                        future_price_prediction = np.exp(np.log(current_midprice) + future_log_return_prediction)
                        my_bid = int(np.floor(future_price_prediction))
                        my_ask = int(np.ceil(future_price_prediction))
                        desired_absolute_position = int(np.floor(abs(standardised_log_return_prediction)))
                        if standardised_log_return_prediction > 0:
                            desired_position = desired_absolute_position
                            if desired_position > kelp_position:
                                orders.append(Order(product, my_bid, desired_position - kelp_position))
                            elif desired_position < kelp_position:
                                orders.append(Order(product, my_ask, desired_position - kelp_position))
                        elif standardised_log_return_prediction < 0:
                            desired_position = -desired_absolute_position
                            if desired_position < kelp_position:
                                orders.append(Order(product, my_ask, desired_position - kelp_position))
                            elif desired_position > kelp_position:
                                orders.append(Order(product, my_bid, desired_position - kelp_position))
                    '''
            elif product == "RAINFOREST_RESIN":
                Trader.update_df(Trader.resin_df, product, state, orders, order_depth)
                

            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
