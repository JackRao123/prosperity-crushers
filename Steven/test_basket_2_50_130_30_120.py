from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pandas as pd
import numpy as np
def size_function(z, edge_0, edge_max, max_position = 50):
    z = np.array(z)
    direction = np.where(z > 0, -1, 1)
    abs_z = np.abs(z)
    size = np.where(
        abs_z <= edge_0,
        0,
        np.where(
            abs_z >= edge_max,
            max_position,
            max_position * ((abs_z - edge_0) / (edge_max - edge_0)) ** 2
        )
    )
    return direction * size

def exit_size_function(z, edge_0, edge_max, max_position = 50):
    # Positive quadratic function with points (0, 0) and (-2, 50)
    if z <= 0:
        if z >= -edge_0:
            return 0
        elif z <= -edge_max:
            return max_position
            
        a = -max_position/(edge_max - edge_0)**2
        return a * (z + edge_max)**2 + max_position
    else:
        if z <= edge_0:
            return 0
        elif z >= edge_max:
            return -max_position
        a = max_position/(edge_max - edge_0)**2
        return a * (z-edge_max)**2 - max_position
    
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

    # I think this might make knowing which curve to move on easier, but what if the spread jumps from -50 to +50? 
    # We will def want to close out. I guess it depends on current position then
    z_ask = ""
    z_bid = ""

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
        synthetic_bid = 0    # What I can sell at
        synthetic_offer = 0    # What I can buy at
        best_ask = 0
        best_bid = 0
        best_ask_jam = 0
        best_bid_jam = 0
        best_bid_croissant = 0
        best_ask_croissant = 0
        available_sell = 0
        available_buy = 0
        print(state.position)
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            # Basket 2 is 4 croissants, 2 jams
            if product == "CROISSANTS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_croissant, best_ask_amount_croissant = list(order_depth.sell_orders.items())[0]
                    best_bid_croissant, best_bid_amount_croissant = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_croissant * 4
                    synthetic_offer += best_ask_croissant * 4

            if product == "JAMS":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask_jam, best_ask_amount_jam = list(order_depth.sell_orders.items())[0]
                    best_bid_jam, best_bid_amount_jam = list(order_depth.buy_orders.items())[0]
                    synthetic_bid += best_bid_jam * 2
                    synthetic_offer += best_ask_jam * 2

            if product == 'PICNIC_BASKET2':
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        print(best_ask_amount_croissant)
        available_sell = min(int(best_bid_amount_croissant / 4), int(best_bid_amount_jam / 2))   # How many we can sell synthetica lly
        available_buy = min(int(abs(best_ask_amount_croissant) / 4), int(abs(best_ask_amount_jam) / 2))    # How many we can buy synthetically

        edge_0 = 50
        edge_max = 130

        edge_max_retreet = 120
        edge_0_retreet = 30

        position_max = int(250/4)

        basket_2_position = state.position.get("PICNIC_BASKET2", 0)
        sell = False
        buy = False
        print(f"BEST ASK: {best_ask}, BEST BID: {best_bid}, Synthetic Bid: {synthetic_bid}, Synthetic Offer: {synthetic_offer}")

        # Redoing entry and exit scheme:
        z_mid = (best_bid + best_ask)/2 - (synthetic_bid + synthetic_offer)/2
        pos_buy = size_function(z_mid, edge_0, edge_max, position_max)
        pos_sell = exit_size_function(z_mid, edge_0_retreet, edge_max_retreet, position_max)

        if z_mid > 0:
            # We only move on the entry curve. Enter more or hold.
            
            if pos_buy <= basket_2_position:
                print("On the entry curve")
                target_position = pos_buy
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_buy, abs(best_bid_amount))
                print(available_buy, abs(trade_needed))
                print(trade_multiplier)
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_bid, -trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_ask_jam, 2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_ask_croissant, 4 * trade_multiplier)]

                print(f"In z_mid > 0, and entering. position: {basket_2_position}, target_position: {target_position}, trade_needed: {trade_needed}")

            else:
                print("Not on entry curve")
                target_position = max(pos_sell, min(basket_2_position, 0))
                
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_sell, abs(best_ask_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_ask, trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_bid_jam, -2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_bid_croissant, -4 * trade_multiplier)]
                print(f"In z_mid > 0, and exiting. position: {basket_2_position}, target_position: {target_position}, trade_needed: {trade_needed}")

            #print(f"z_mid: {z_mid}, target_position: {target_position}, self_position: {basket_2_position}, trade_needed: {trade_needed}, order: {result}")

        
        elif z_mid < 0:
            if pos_buy >= basket_2_position:
                print("On the entry curve")
                target_position = pos_buy
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_sell,abs(best_ask_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_ask, trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_bid_jam, -2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_bid_croissant, -4 * trade_multiplier)]
                print(f"In z_mid < 0, and entering. position: {basket_2_position}, target_position: {target_position}, trade_needed: {trade_needed}")

            else:
                print("Not on entry curve")
                target_position = min(pos_sell, max(basket_2_position, 0))
                
                trade_needed = int(target_position - basket_2_position)
                trade_multiplier = min(abs(trade_needed), available_buy,abs(best_bid_amount))
                
                result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_bid, -trade_multiplier)]
                result['JAMS'] = [Order('JAMS', best_ask_jam, 2 * trade_multiplier)]
                result['CROISSANTS'] = [Order('CROISSANTS', best_ask_croissant, 4 * trade_multiplier)]
                print(f"In z_mid < 0, and exiting. position: {basket_2_position}, target_position: {target_position}, trade_needed: {trade_needed}")

            #print(f"z_mid: {z_mid}, target_position: {target_position}, self_position: {basket_2_position}, trade_needed: {trade_needed}, order: {result}")

        '''


        
        # Below should be the sizing scheme. Need to know when to move on which leaf. If z > 0: we are selling, if z < 0: we are buying.
        if best_ask < synthetic_bid:
            z = best_ask - synthetic_bid
            pos_buy = size_function(z, edge_0, edge_max, position_max)
            pos_sell = exit_size_function(z, edge_0_retreet, edge_max_retreet, position_max)
            need_trade = 0
            if pos_buy > basket_2_position:
                target_position = pos_buy
                need_trade = int(target_position - basket_2_position)
            else:
                target_position = min(basket_2_position, pos_sell)
                need_trade = int(target_position - basket_2_position)
            print(f"In A: z = {z}, current_position = {basket_2_position}, targetposition = {target_position}, needed_trade = {need_trade}")
            trade_multiplier = min(abs(need_trade), available_sell)

            
            result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_ask, trade_multiplier)]
            result['JAMS'] = [Order('JAMS', best_bid_jam, -2 * trade_multiplier)]
            result['CROISSANTS'] = [Order('CROISSANTS', best_bid_croissant, -4 * trade_multiplier)]
            buy = True

        elif best_bid > synthetic_offer:
            z = best_bid - synthetic_offer
            pos_buy = size_function(z, edge_0, edge_max, position_max)
            pos_sell = exit_size_function(z, edge_0_retreet, edge_max_retreet, position_max)
            need_trade = 0
            if pos_buy < basket_2_position:
                target_position = pos_buy
                need_trade = int(target_position - basket_2_position)
            else:
                target_position = max(basket_2_position, pos_buy)
                need_trade = int(target_position - basket_2_position)
            print(f"In B: z = {z}, current_position = {basket_2_position}, targetposition = {target_position}, needed_trade = {need_trade}")

            trade_multiplier = min(abs(need_trade), available_buy)

            
            result['PICNIC_BASKET2'] = [Order('PICNIC_BASKET2', best_bid, -trade_multiplier)]
            result['JAMS'] = [Order('JAMS', best_ask_jam, 2 * trade_multiplier)]
            result['CROISSANTS'] = [Order('CROISSANTS', best_ask_croissant, 4 * trade_multiplier)]
            sell = True


        '''
        print(result)

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData



