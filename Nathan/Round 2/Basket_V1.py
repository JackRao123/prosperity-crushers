
class Trader:
    picnic_basket1_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
    ])

    picnic_basket2_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
    ])

    djembes_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
    ])

    croissants_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
    ])

    jams_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", 'mmbot_bid', 'mmbot_ask','mmbot_midprice', 'log_returns'
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
        if product == "CROISSANTS":
            adverse_volume = 49
        elif product == "JAMS":
            adverse_volume = 99
        elif product == "PICNIC_BASKET2":
            adverse_volume = 14
        elif product == "PICNIC_BASKET1":
            adverse_volume = 9
        elif product == "DJEMBES":
            adverse_volume = 29
        
        if row["bid_volume_1"] >= adverse_volume: #Adverse volume set to 15. #mm_bot_bid will just become the top level if there is no adverse volume.
            mm_bot_bid = row["bid_price_1"]
        elif row["bid_volume_2"] >= adverse_volume:
            mm_bot_bid = row["bid_price_2"]
        elif row["bid_volume_3"] >= adverse_volume:
            mm_bot_bid = row["bid_price_3"]
        else:
            mm_bot_bid = row["bid_price_1"]

        if row["ask_volume_1"] >= adverse_volume:  # Adverse volume set to 15. mm_bot_ask will just become the top level if there is no adverse volume.
            mm_bot_ask = row["ask_price_1"]
        elif row["ask_volume_2"] >= adverse_volume:
            mm_bot_ask = row["ask_price_2"]
        elif row["ask_volume_3"] >= adverse_volume:
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

    def __init__(self): #Calibrated according to PICNIC_BASKET2.
        self.retreat_per_lot = 0.1
        self.edge_per_lot = 0.2
        self.edge0 = 0

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == "PICNIC_BASKET1":
                picnic_basket1_position = state.position.get(product, 0)
                Trader.update_df(Trader.picnic_basket1_df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    current_row = self.picnic_basket1_df.iloc[-1]
                    mmbot_midprice = current_row['mmbot_midprice']
                    theo = mmbot_midprice - picnic_basket1_position * self.retreat_per_lot * 2
                    #Quoting as wide as possible whilst still having QP, and not being through our theo.
                    my_bid = min(int(np.floor(theo)), current_row.bid_price_1+1)
                    my_ask = max(int(np.ceil(theo)), current_row.ask_price_1-1)
                    bid_edge = theo - my_bid
                    ask_edge = my_ask - theo
                    edge0 = self.edge0
                    bid_volume = int(np.floor((bid_edge - edge0) / (self.edge_per_lot * 2))) if bid_edge > edge0 else 0
                    ask_volume = -int(np.floor((ask_edge - edge0) / (self.edge_per_lot * 2))) if ask_edge > edge0 else 0
                    #Below makes sure that we dont send orders over position limits.
                    bid_volume = min(bid_volume, 60 - picnic_basket1_position)
                    ask_volume = max(ask_volume, -60 - picnic_basket1_position)
                    
                    orders.append(Order(product, my_bid, bid_volume))
                    orders.append(Order(product, my_ask, ask_volume))
                    print(f"Inserted order for {product}: Bid: {my_bid}, Volume: {bid_volume}, Ask: {my_ask}, Volume: {ask_volume}")
            if product == "PICNIC_BASKET2":
                picnic_basket2_position = state.position.get(product, 0)
                Trader.update_df(Trader.picnic_basket2_df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    current_row = self.picnic_basket2_df.iloc[-1]
                    mmbot_midprice = current_row['mmbot_midprice']
                    theo = mmbot_midprice - picnic_basket2_position * self.retreat_per_lot
                    #Quoting as wide as possible whilst still having QP, and not being through our theo.
                    my_bid = min(int(np.floor(theo)), current_row.bid_price_1+1)
                    my_ask = max(int(np.ceil(theo)), current_row.ask_price_1-1)
                    bid_edge = theo - my_bid
                    ask_edge = my_ask - theo
                    edge0 = self.edge0
                    bid_volume = int(np.floor((bid_edge - edge0) / self.edge_per_lot)) if bid_edge > edge0 else 0
                    ask_volume = -int(np.floor((ask_edge - edge0) / self.edge_per_lot)) if ask_edge > edge0 else 0
                    #Below makes sure that we dont send orders over position limits.
                    bid_volume = min(bid_volume, 100 - picnic_basket2_position)
                    ask_volume = max(ask_volume, -100 - picnic_basket2_position)
                    
                    orders.append(Order(product, my_bid, bid_volume))
                    orders.append(Order(product, my_ask, ask_volume))
                    print(f"Inserted order for {product}: Bid: {my_bid}, Volume: {bid_volume}, Ask: {my_ask}, Volume: {ask_volume}")
            if product == "DJEMBES":
                djembes_position = state.position.get(product, 0)
                Trader.update_df(Trader.djembes_df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    current_row = self.djembes_df.iloc[-1]
                    mmbot_midprice = current_row['mmbot_midprice']
                    theo = mmbot_midprice - djembes_position * self.retreat_per_lot * 2
                    # Quoting as wide as possible whilst still having QP, and not being through our theo.
                    my_bid = min(int(np.floor(theo)), current_row.bid_price_1 + 1)
                    my_ask = max(int(np.floor(theo) + 1), current_row.ask_price_1 - 1)
                    bid_edge = theo - my_bid
                    ask_edge = my_ask - theo
                    edge0 = self.edge0
                    bid_volume = int(np.floor((bid_edge - edge0) / (self.edge_per_lot * 2))) if bid_edge > edge0 else 0
                    ask_volume = -int(np.floor((ask_edge - edge0) / (self.edge_per_lot * 2))) if ask_edge > edge0 else 0
                    # Below makes sure that we don’t send orders over position limits.
                    bid_volume = min(bid_volume, 60 - djembes_position)
                    ask_volume = max(ask_volume, -60 - djembes_position)

                    orders.append(Order(product, my_bid, bid_volume))
                    orders.append(Order(product, my_ask, ask_volume))
                    print(f"Inserted order for {product}: Bid: {my_bid}, Volume: {bid_volume}, Ask: {my_ask}, Volume: {ask_volume}")
            if product == "CROISSANTS":
                croissants_position = state.position.get(product, 0)
                Trader.update_df(Trader.croissants_df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    current_row = self.croissants_df.iloc[-1]
                    mmbot_midprice = current_row['mmbot_midprice']
                    theo = mmbot_midprice - croissants_position * (self.retreat_per_lot / 2)
                    # Quoting as wide as possible whilst still having QP, and not being through our theo.
                    my_bid = min(int(np.floor(theo)), current_row.bid_price_1 + 1)
                    my_ask = max(int(np.floor(theo) + 1), current_row.ask_price_1 - 1)
                    bid_edge = theo - my_bid
                    ask_edge = my_ask - theo
                    edge0 = self.edge0
                    bid_volume = int(np.floor((bid_edge - edge0) / (self.edge_per_lot / 2))) if bid_edge > edge0 else 0
                    ask_volume = -int(np.floor((ask_edge - edge0) / (self.edge_per_lot / 2))) if ask_edge > edge0 else 0
                    # Below makes sure that we don’t send orders over position limits.
                    bid_volume = min(bid_volume, 250 - croissants_position)
                    ask_volume = max(ask_volume, -250 - croissants_position)

                    orders.append(Order(product, my_bid, bid_volume))
                    orders.append(Order(product, my_ask, ask_volume))
                    print(f"Inserted order for {product}: Bid: {my_bid}, Volume: {bid_volume}, Ask: {my_ask}, Volume: {ask_volume}")
            if product == "JAMS":
                jams_position = state.position.get(product, 0)
                Trader.update_df(Trader.jams_df, product, state, orders, order_depth)
                if len(order_depth.sell_orders) and len(order_depth.buy_orders):
                    current_row = self.jams_df.iloc[-1]
                    mmbot_midprice = current_row['mmbot_midprice']
                    theo = mmbot_midprice - jams_position * (self.retreat_per_lot /3)
                    # Quoting as wide as possible whilst still having QP, and not being through our theo.
                    my_bid = min(int(np.floor(theo)), current_row.bid_price_1 + 1)
                    my_ask = max(int(np.floor(theo) + 1), current_row.ask_price_1 - 1)
                    bid_edge = theo - my_bid
                    ask_edge = my_ask - theo
                    edge0 = self.edge0
                    bid_volume = int(np.floor((bid_edge - edge0) / (self.edge_per_lot /3))) if bid_edge > edge0 else 0
                    ask_volume = -int(np.floor((ask_edge - edge0) / (self.edge_per_lot /3))) if ask_edge > edge0 else 0
                    # Below makes sure that we don’t send orders over position limits.
                    bid_volume = min(bid_volume, 350 - jams_position)
                    ask_volume = max(ask_volume, -350 - jams_position)

                    orders.append(Order(product, my_bid, bid_volume))
                    orders.append(Order(product, my_ask, ask_volume))
                    print(f"Inserted order for {product}: Bid: {my_bid}, Volume: {bid_volume}, Ask: {my_ask}, Volume: {ask_volume}")




            result[product] = orders

        return result, 1, "SAMPLE"
