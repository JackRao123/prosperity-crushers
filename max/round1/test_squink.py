from datamodel import OrderDepth, UserId, TradingState, Order
import pandas as pd
import numpy as np

class Trader:

    squink_df = pd.DataFrame(columns=[
        "timestamp", "product",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss"
    ])

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

        if product == 'SQUID_INK':
            rolling_window = 200
            Trader.squink_df['rolling_mean'] = Trader.squink_df['mid_price'].rolling(
                window=rolling_window,
                min_periods=1
            ).mean()

            Trader.squink_df['rolling_std'] = Trader.squink_df['mid_price'].rolling(
                window=rolling_window,
                min_periods=1
            ).std()

    def exp_leaf(x, k):
        y = np.where(x < 0, -(np.exp(-k * x) - 1), np.exp(k * x) - 1)
        return y / (np.exp(k) - 1)

    def exp_leaf_inverse(y, k):
        y = np.clip(y, -1, 1)
        denom = np.exp(k) - 1
        return np.where(y < 0,
                        -1 / k * np.log(-y * denom + 1),
                        1 / k * np.log(y * denom + 1))

    # Buy curve: inverse on left, exp on right
    def buy_leaf(x, k):
        x = np.clip(x, -1, 1)
        y = np.zeros_like(x)
        y[x < 0] = Trader.exp_leaf_inverse(x[x < 0], k)
        y[x >= 0] = Trader.exp_leaf(x[x >= 0], k)
        return y

    # Sell curve: original on left, inverse on right
    def sell_leaf(x, k):
        x = np.clip(x, -1, 1)
        y = np.zeros_like(x)
        y[x < 0] = Trader.exp_leaf(x[x < 0], k)
        y[x >= 0] = Trader.exp_leaf_inverse(x[x >= 0], k)
        return y
    
    def execute_edge_trade(self, state: TradingState):
        orders: list[Order] = []

        product = "SQUID_INK"
        df = Trader.squink_df

        order_depth: OrderDepth = state.order_depths[product]

        if df.empty or 'rolling_mean' not in df.columns or pd.isna(df.iloc[-1]['rolling_std']):
            return
        
        latest = df.iloc[-1]
        mid_price = latest['mid_price']
        mean = latest['rolling_mean']
        std = latest['rolling_std']

        if std == 0 or std is None or np.isnan(std):
            return
        
        z = (mid_price - mean) / std
        max_z = 6
        if z > max_z:
            z = max_z
        elif z < -max_z:
            z = -max_z

        # z_clipped = np.clip(z, -4, 4)

        # k = 4
        # buy_pos = int(round(Trader.buy_leaf(np.array([z_clipped / 4]), k)[0] * 50))
        # sell_pos = int(round(Trader.sell_leaf(np.array([z_clipped / 4]), k)[0] * 50))
        
        k = 4
        buy_pos = int(round(Trader.buy_leaf(z / max_z, k) * 50))
        sell_pos = int(round(Trader.sell_leaf(z / max_z, k) * 50))

        current_position = state.position.get(product, 0)
        buy_volume = max(0, buy_pos - current_position)
        sell_volume = max(0, current_position - sell_pos)

        if buy_volume > 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            available_ask_volume = order_depth.sell_orders[best_ask]
            volume = min(buy_volume, available_ask_volume)
            if volume > 0:
                orders.append(Order(product, best_ask, volume))

        if sell_volume > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            available_bid_volume = order_depth.buy_orders[best_bid]
            volume = min(sell_volume, available_bid_volume)
            if volume > 0:
                orders.append(Order(product, best_bid, -volume))

        return orders


        
    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            orders: list[Order] = []
            order_depth: OrderDepth = state.order_depths[product]

            if product == "SQUID_INK":
                Trader.update_df(Trader.squink_df, product, state, orders, order_depth) 
                trades = Trader.execute_edge_trade(self, state)
                orders = orders + trades

            result[product] = orders
            
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        conversions = 1
        return result, conversions, traderData
