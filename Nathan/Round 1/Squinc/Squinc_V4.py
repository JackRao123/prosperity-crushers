#Quoting Multiple Layers
class Trader:
    kelp_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
    ])

    resin_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice', 'log_returns'
    ])
    squid_ink_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask', 'mmbot_midprice', 'log_returns'
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

        if len(df) >= 1:
            previous_row = df.iloc[-1]
            if previous_row["mid_price"] is not None and row["mid_price"] is not None:
                log_return = np.log(row["mid_price"]) - np.log(previous_row["mid_price"])
                row["log_returns"] = log_return
            else:
                row["log_returns"] = 0

        df.loc[len(df)] = row

    def __init__(self):
        self.retreat_per_lot = 0.05
        self.edge_per_lot = 0.1
        self.edge0 = 0

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "SQUID_INK":
                squid_ink_position = state.position.get(product, 0)
                Trader.update_df(Trader.squid_ink_df, product, state, orders, order_depth)

                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    if len(self.squid_ink_df) >= 2:
                        retreat_per_lot = self.retreat_per_lot 
                        edge_per_lot = self.edge_per_lot
                        current_row = self.squid_ink_df.iloc[-1]
                        previous_row = self.squid_ink_df.iloc[-2]
                        #What if we just set future price prediction to be equal to the mmbot_ midprice?
                        '''
                        current_mmbot_log_return = np.log(current_row.mmbot_midprice) - np.log(previous_row.mmbot_midprice)
                        future_mmbot_log_return_prediction = 0 #0.061 * current_mmbot_log_return 
                        future_price_prediction = current_row.mmbot_midprice * np.exp(future_mmbot_log_return_prediction)
                        '''
                        future_price_prediction = current_row.mmbot_midprice
                        #print(f"current_mmbt_midprice: {current_row.mmbot_midprice}")
                        #print(f"future_mmbot_log_return_prediction: {future_mmbot_log_return_prediction}")
                        #print(f"future_price_prediction: {future_price_prediction}")
                        theo = future_price_prediction - squid_ink_position * retreat_per_lot
                        bid_ask_spread = current_row.ask_price_1 - current_row.bid_price_1
                        #Maybe implement some sort of "dime check" that checks if we are diming others and have QP?
                        #Try a strategy where we go as wide as possible whilst still having QP, and not being in cross with our theo.

                        #Quoting as wide as possible whilst still having QP, and not being through our theo.
                        '''
                        my_bid = min(int(np.floor(theo)), current_row.bid_price_1+1)
                        my_ask = max(int(np.ceil(theo)), current_row.ask_price_1-1)
                        '''
                        #Quoting 1 wide to encourage them to hit / lift both sides.
                        
                        my_bid_1 = int(np.floor(theo))
                        my_ask_1 = int(np.ceil(theo))
                        
                        my_bid_2 = int(np.floor(theo)) - 1
                        my_ask_2 = int(np.ceil(theo)) + 1

                        my_bid_3 = int(np.floor(theo)) - 2
                        my_ask_3 = int(np.ceil(theo)) + 2

                        bid_edge_1 = theo - my_bid_1
                        ask_edge_1 = my_ask_1 - theo

                        bid_edge_2 = theo - my_bid_2
                        ask_edge_2 = my_ask_2 - theo


                        edge0 = self.edge0

                        bid_volume_1 = int(np.floor((bid_edge_1 - edge0) / edge_per_lot)) if bid_edge_1 > edge0 else 0
                        ask_volume_1 = -int(np.floor((ask_edge_1 - edge0) / edge_per_lot)) if ask_edge_1 > edge0 else 0

                        bid_volume_2 = int(np.floor((bid_edge_2 - edge0) / edge_per_lot)) if bid_edge_2 > edge0 else 0
                        ask_volume_2 = -int(np.floor((ask_edge_2 - edge0) / edge_per_lot)) if ask_edge_2 > edge0 else 0

                        #lets just quote the remaining leftover volume at volume 3.
                        #Below makes sure that we dont send orders over position limits.
                        if bid_volume_1 + bid_volume_2 > 50 - squid_ink_position:
                            bid_volume_2 = 50 - squid_ink_position - bid_volume_1
                            bid_volume_3 = 0
                        else:
                            bid_volume_3 = 50 - squid_ink_position - bid_volume_1 - bid_volume_2
                        if ask_volume_1 + ask_volume_2 < -50 - squid_ink_position:
                            ask_volume_2 = -50 - squid_ink_position - ask_volume_1
                            ask_volume_3 = 0
                        else:
                            ask_volume_3 = -50 - squid_ink_position - ask_volume_1 - ask_volume_2
                        
                        orders.append(Order(product, my_bid_1, bid_volume_1))
                        orders.append(Order(product, my_ask_1, ask_volume_1))
                        orders.append(Order(product, my_bid_2, bid_volume_2))
                        orders.append(Order(product, my_ask_2, ask_volume_2))
                        orders.append(Order(product, my_bid_3, bid_volume_3))
                        orders.append(Order(product, my_ask_3, ask_volume_3))

                        

            elif product == "RAINFOREST_RESIN":
                Trader.update_df(Trader.resin_df, product, state, orders, order_depth)

            result[product] = orders

        return result, 1, "SAMPLE"
