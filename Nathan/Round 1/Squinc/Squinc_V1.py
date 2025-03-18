import pandas as pd
import numpy as np
class Trader:
    kelp_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice'
    ])

    resin_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice'
    ])

    squid_ink_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice'
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
        
        row["mmbot_bid"] = mm_bot_bid
        row["mmbot_ask"] = mm_bot_ask
        row["mmbot_midprice"] = (mm_bot_bid + mm_bot_ask) / 2

                

        df.loc[len(df)] = row

    def __init__(self, retreat_per_lot, edge_per_lot, edge0):
        self.retreat_per_lot = retreat_per_lot
        self.edge_per_lot = edge_per_lot
        self.edge0 = edge0

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
                if len(self.squid_ink_df) >= 201:
                    current_row = self.squid_ink_df.iloc[-1]
                    previous_200_row = self.squid_ink_df.iloc[-201]
                    current_midprice = (current_row["ask_price_1"] + current_row["bid_price_1"]) / 2
                    previous_200_midprice = (previous_200_row["ask_price_1"] + previous_200_row["bid_price_1"]) / 2
                    log_return_200 = np.log(current_midprice / previous_200_midprice)
                    predicted_log_return_200 = log_return_200 * -0.2773
                    predicted_200_midprice = previous_200_midprice * np.exp(predicted_log_return_200)
                    print(f"predicted_200_midprice: {predicted_200_midprice}")
                    if predicted_200_midprice > current_row["ask_price_1"]:
                        orders.append(Order(product, current_row["ask_price_1"], current_row["ask_volume_1"]))
                        print(f"Trading {product} at {current_row['ask_price_1']} for volume {current_row['ask_volume_1']}")
                    elif predicted_200_midprice < current_row["bid_price_1"]:
                        orders.append(Order(product, current_row["bid_price_1"], -current_row["bid_volume_1"]))
                        print(f"Trading {product} at {current_row['bid_price_1']} for volume {-current_row['bid_volume_1']}")

            result[product] = orders

        return result, 1, "SAMPLE"
