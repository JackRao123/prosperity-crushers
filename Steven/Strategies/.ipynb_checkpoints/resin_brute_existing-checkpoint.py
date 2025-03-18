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
                acceptable_price = 10000

                print("Acceptable price : " + str(acceptable_price))
                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
        
                if len(order_depth.sell_orders) != 0:
                    i = 0
                    while i < len(list(order_depth.sell_orders.items())):
                        curr_ask, curr_ask_amount = list(order_depth.sell_orders.items())[i]
                        if curr_ask < acceptable_price:
                            orders.append(Order(product, curr_ask, -1*curr_ask_amount))
                        i += 1

                if len(order_depth.buy_orders) != 0:
                    k = 0
                    while k < len(list(order_depth.buy_orders.items())):
                        curr_bid, curr_bid_amount = list(order_depth.buy_orders.items())[k]
                        if curr_bid > acceptable_price:
                            orders.append(Order(product, curr_bid, -1*curr_bid_amount))
                        k += 1

                orders.append(Order(product, 10005, -5))
                orders.append(Order(product, 11000, -10))

                orders.append(Order(product, 9995, 5))
                orders.append(Order(product, 9000, 10))

            result[product] = orders
            print(orders)
           
                
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
