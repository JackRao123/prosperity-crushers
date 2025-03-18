

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

                    min_spread = 2

                    if curr_ask - curr_bid > min_spread:     #Spreads are at least 3 wide so we can improve both by 1 and not hack. We then quote 2 lots
                        orders.append(Order(product, list(order_depth.buy_orders.items())[0][0]+1, 1))
                        orders.append(Order(product, list(order_depth.sell_orders.items())[0][0]-1, -1))

                result[product] = orders
                print(orders)

           
                
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
