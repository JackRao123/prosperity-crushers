from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pandas as pd
import numpy as np


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

    squink_df = pd.DataFrame(columns=[
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
            if product == "SQUID_INK":
                # Check the rolling z score. If it is > 4, we sell,
                squink_position = state.position.get(product, 0)
                Trader.update_df(Trader.squink_df, product, state, orders, order_depth) 
                
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    mid_price = (best_ask + best_bid)/2
                    rolling_window = 150
                    if len(self.squink_df) > rolling_window:
                        
                        z_score = (mid_price - self.squink_df.mid_price.rolling(rolling_window).mean().iloc[-1])/self.squink_df.mid_price.rolling(rolling_window).std().iloc[-1]
                        # Now that I have the Z-Score, I want to implement an edge model for entry
                        
                        edge_0 = 2.5
                        if abs(z_score) > edge_0:
                            target_position = min(int((np.abs(z_score) - edge_0) * 20), 30)
                        else:
                            target_position = 0
                        # Trying simple linear scheme right now to enter, next is to consider exit
                        print(f"Squink_position: {squink_position}")
                        print(f"z_score: {z_score}, target_position: {target_position}")
                        edge_0 = 3
                        if abs(z_score) > edge_0:
                            target_position = min(int((np.abs(z_score) - edge_0) * 10), 30)
                        else:
                            target_position = 0
                        # Trying simple linear scheme right now to enter, next is to consider exit
                        print(f"Squink_position: {squink_position}")
                        print(f"z_score: {z_score}, target_position: {target_position}")
                        # Entry
                        
                        if z_score <= -edge_0:
                            print("A")
                            target_position = max(target_position, squink_position)
                            size_needed = target_position - squink_position
                            if size_needed < 0:
                                pass
                            else:
                                orders.append(Order(product, best_ask, size_needed))
                                #orders.append(Order(product, best_bid+1, size_needed))
                         
                        elif z_score >= edge_0:
                            print("C")
                            target_position = min(-target_position, squink_position)
                            size_needed = target_position - squink_position
                            if size_needed > 0:
                                pass
                            else:
                                orders.append(Order(product, best_bid, size_needed))
                                #orders.append(Order(product, best_ask - 1, size_needed))
                        # Now what if it is between?

                        else:
                            if squink_position > 0:
                                if z_score > 0:
                                    target_position = 0
                                    size_needed = target_position - squink_position
                                    orders.append(Order(product, best_bid, size_needed))
                                    #orders.append(Order(product, best_ask - 1, size_needed))

                                else:
                                    exit_position = min(squink_position, int(-15*z_score))
                                    trades_needed = exit_position - squink_position
                                    orders.append(Order(product, best_bid, trades_needed))
                                    #orders.append(Order(product, best_ask-1, trades_needed))

                            if squink_position < 0:
                                if z_score < 0:
                                    target_position = 0
                                    size_needed = target_position - squink_position
                                    orders.append(Order(product, best_ask, size_needed))
                                    #orders.append(Order(product, best_bid + 1, size_needed))
                                else:
                                    exit_position = min(squink_position, int(15*z_score))
                                    trades_needed = exit_position - squink_position
                                    orders.append(Order(product, best_ask, trades_needed)) 
                                    #orders.append(Order(product, best_bid + 1, trades_needed))
                        
                        # Entry Attempt to fix sizing scheme above
                        '''
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
                            '''


                        
    
                        

                result[product] = orders
                print(orders)

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
