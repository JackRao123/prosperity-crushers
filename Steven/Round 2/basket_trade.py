from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            #acceptable_price = 0;  # Participant should calculate this value
            #This only trades if amethysts, and sets fair to be 10000
            if product == "SQUID_INK":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    curr_ask, curr_ask_amount = list(order_depth.sell_orders.items())[0]
                    curr_bid, curr_bid_amount = list(order_depth.buy_orders.items())[0]

                    mid_price = int(round((curr_ask + curr_bid)/2))

                    orders.append(Order(product, mid_price-1, 1))
                    orders.append(Order(product, mid_price+1, -1))

            result[product] = orders
            print(orders)
           
                
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
