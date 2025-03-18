# If there are a lot of buyers, ie imbalance is -ve (since midprice – bid), then we improve the offer.
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
            if product == 'KELP':
                position = 0
                if product in state.position:
                    position = state.position[product]
                else:
                    pass

                imbal = 0
                midprice = 0
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    curr_ask, curr_ask_amount = list(order_depth.sell_orders.items())[0]
                    midprice += curr_ask
                    curr_bid, curr_bid_amount = list(order_depth.buy_orders.items())[0]
                    midprice += curr_bid
                    midprice = midprice / 2

                    if len(order_depth.sell_orders) != 0:
                        i = 0
                        while i < len(list(order_depth.sell_orders.items())):
                            curr_ask, curr_ask_amount = list(order_depth.sell_orders.items())[i]
                            imbal += (curr_ask - midprice) * curr_ask_amount
                            i += 1

                    if len(order_depth.buy_orders) != 0:
                        k = 0
                        while k < len(list(order_depth.buy_orders.items())):
                            curr_bid, curr_bid_amount = list(order_depth.buy_orders.items())[k]
                            imbal += (curr_bid - midprice) * curr_bid_amount
                            k += 1
                    #if position > 0:
                     #   orders.append(Order(product, list(order_depth.buy_orders.items())[0][0], -position))
                    #elif position < 0:
                    #    orders.append(Order(product, list(order_depth.sell_orders.items())[0][0], position))

                    # test if we passivelt enter orders
                
                    if imbal >  50:
                        orders.append(Order(product, list(order_depth.sell_orders.items())[0][0]-1, -1))
                    elif imbal < -50:
                        orders.append(Order(product, list(order_depth.buy_orders.items())[0][0]+1, 1))

                else:
                    break
                
            
            result[product] = orders
            print(orders)
           
                
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
