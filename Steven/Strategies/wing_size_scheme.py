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
            if product == "RAINFOREST_RESIN":
                # Some sizing scheme where I do 2 lots at each successive price level?
                # This means I show 2 at 10000 +- 1, 4 at 10000 +- 2, 6 at 10000 +- 3 etc etc. 
                # Maybe 25 is too much and the entire order gets cancelled 
                for i in range(2, 5):
                    for j in range(2):
                        orders.append(Order(product, 10000 + i, -i))
                        orders.append(Order(product, 10000-i, i))
                

            result[product] = orders
            print(orders)
           
                
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
