#Cumulative Log Returns

import pandas as pd
import numpy as np
class Trader:
    kelp_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', "log_returns", "cumulative_log_returns"
    ])

    resin_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice', "log_returns", "cumulative_log_returns"
    ])

    squid_ink_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice', "log_returns", "cumulative_log_returns"
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

        
        if row["bid_volume_1"] >= 15: #Adverse volume set to 15. #mm_bot_bid will just become the top level if there is no adverse volume.
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
        
        if len(df) == 0:
            row["log_returns"] = np.nan
        else:
            last_mid_price = df.iloc[-1]["mid_price"]
            if last_mid_price is not None and mid_price is not None:
                row["log_returns"] = np.log(mid_price / last_mid_price)
            else:
                row["log_returns"] = np.nan
        row["mmbot_bid"] = mm_bot_bid
        row["mmbot_ask"] = mm_bot_ask
        row["mmbot_midprice"] = (mm_bot_bid + mm_bot_ask) / 2
        
        if len(df) < 200:
            row["cumulative_log_returns"] = np.nan
        else:
            row["cumulative_log_returns"] = df.iloc[-200:]["log_returns"].sum()

        df.loc[len(df)] = row

    def __init__(self):
        self.retreat_per_lot = 0.12
        self.edge_per_lot = 0.15
        self.edge0 = 0.02



    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "KELP":
                Trader.update_df(Trader.kelp_df, product, state, orders, order_depth)

            elif product == "RAINFOREST_RESIN":
                Trader.update_df(Trader.resin_df, product, state, orders, order_depth)
            
            elif product == "SQUID_INK":
                Trader.update_df(Trader.squid_ink_df, product, state, orders, order_depth)
                squid_ink_position = state.position.get(product, 0)
                current_row = Trader.squid_ink_df.iloc[-1]
                current_squid_ink_df = Trader.squid_ink_df
                
                if len(current_squid_ink_df) < 200:
                    continue
                else:
                    vol_200 = current_squid_ink_df["cumulative_log_returns"].iloc[-200:].std() #Volatility of the last 200 rows.
                    z_score = current_row["cumulative_log_returns"] / vol_200  #Z-score of the last row.
                    z_score = current_row["cumulative_log_returns"]
                    if z_score > 0.01: #As signal is above a threshold we want to go to a short position. 
                        if squid_ink_position != -50: #Ensuring we have enough edge to cross spread.
                            orders.append(Order(product, current_row["bid_price_1"], -50 - squid_ink_position))
                    elif z_score < -0.01: #As signal is below a threshold we want to go to a long position.
                        if squid_ink_position != 50:
                            orders.append(Order(product, current_row["ask_price_1"], 50 - squid_ink_position))
                    if z_score > 0 and squid_ink_position > 0: #As signal is above 0 we want to close our our long position.
                        orders.append(Order(product, current_row["bid_price_1"], -squid_ink_position))
                    elif z_score < 0 and squid_ink_position < 0: #As signal is below 0 we want to close our our short position.
                        orders.append(Order(product, current_row["ask_price_1"], -squid_ink_position))
                            


            result[product] = orders
                

        return result, 1, "SAMPLE"
