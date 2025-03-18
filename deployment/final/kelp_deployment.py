
class Trader:
    
    def __init__(self, retreat_per_lot=0.005, edge_per_lot=0.03, edge0=0):
        self.retreat_per_lot = retreat_per_lot
        self.edge_per_lot = edge_per_lot
        self.edge0 = edge0
        self.kelp_df = pd.DataFrame(columns=[
            "timestamp", "product",
            "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
            "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
            "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice'
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


    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "KELP":
                kelp_position = state.position.get(product, 0)
                self.update_df(self.kelp_df, product, state, orders, order_depth)

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
                        '''
                        future_price_prediction = np.exp(np.log(current_midprice) + future_log_return_prediction)
                        '''
                        #What if we just set future price prediction to be equal to the mmbot_ midprice?
                        current_mmbot_log_return = np.log(current_row.mmbot_midprice) - np.log(previous_row.mmbot_midprice)
                        future_mmbot_log_return_prediction = -0.2933 * current_mmbot_log_return
                        future_price_prediction = current_row.mmbot_midprice * np.exp(future_mmbot_log_return_prediction)
                        print(f"current_mmbt_midprice: {current_row.mmbot_midprice}")
                        print(f"future_mmbot_log_return_prediction: {future_mmbot_log_return_prediction}")
                        print(f"future_price_prediction: {future_price_prediction}")
                        theo = future_price_prediction - kelp_position * self.retreat_per_lot
                        bid_ask_spread = current_row.ask_price_1 - current_row.bid_price_1
                        #Maybe implement some sort of "dime check" that checks if we are diming others and have QP?
                        #Try a strategy where we go as wide as possible whilst still having QP, and not being in cross with our theo.
                        '''
                        if bid_ask_spread <= 2:
                            my_bid = min(int(np.floor(theo)), current_row.bid_price_1)
                            my_ask = max(int(np.ceil(theo)), current_row.ask_price_1) #Try quoting wider - maybe if ba spread is wider we want to quote wider - if market is 4 wide maybe we quote 2 wide, else we quote 1 wide?
                        if bid_ask_spread >= 3:
                            my_bid = int(np.floor(theo - 0.5))
                            my_ask = int(np.ceil(theo + 0.5))
                        '''
                        #Quoting as wide as possible whilst still having QP, and not being through our theo.
                        my_bid = min(int(np.floor(theo)), current_row.bid_price_1+1)
                        my_ask = max(int(np.ceil(theo)), current_row.ask_price_1-1)
                        bid_edge = theo - my_bid
                        ask_edge = my_ask - theo
                        edge0 = self.edge0

                        bid_volume = int(np.floor((bid_edge - edge0) / self.edge_per_lot)) if bid_edge > edge0 else 0
                        ask_volume = -int(np.floor((ask_edge - edge0) / self.edge_per_lot)) if ask_edge > edge0 else 0
                        #Below makes sure that we dont send orders over position limits.
                        bid_volume = min(bid_volume, 50 - kelp_position)
                        ask_volume = max(ask_volume, -50 - kelp_position)
                        
                        orders.append(Order(product, my_bid, bid_volume))
                        orders.append(Order(product, my_ask, ask_volume))

                        

            elif product == "RAINFOREST_RESIN":
                self.update_df(self.resin_df, product, state, orders, order_depth)

            result[product] = orders

        return result, 1, "SAMPLE"
