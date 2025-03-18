import math
from datamodel import OrderDepth, UserId, TradingState, Order
import pandas as pd
import numpy as np

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

import numpy as np

# Bisection method for implied volatility
def bisection_implied_vol(S, K, T, r, market_price, tol=1e-6, max_iter=100):
    # Initial bounds for volatility
    lower_bound = 1e-6
    upper_bound = 5.0
    
    for i in range(max_iter):
        # Midpoint for the current bounds
        sigma_mid = (lower_bound + upper_bound) / 2
        
        # Calculate the Black-Scholes price for this sigma
        option_price = black_scholes_call_price(S, K, T, r, sigma_mid)
        
        # If the price is close enough to the market price, return this sigma
        if abs(option_price - market_price) < tol:
            return sigma_mid
        
        # If the option price is higher than the market price, adjust the upper bound
        if option_price > market_price:
            upper_bound = sigma_mid
        else:
            # Otherwise, adjust the lower bound
            lower_bound = sigma_mid
    
    # Return the midpoint as the implied volatility
    return (lower_bound + upper_bound) / 2

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
    def __init__(self):
        self.position_limit = {"KELP": 50, "RAINFOREST_RESIN": 50}
        pass     

    def run(self, state: TradingState):
        result = {}
        volcano_s0 = 0
        option_mid = 0
        for product in state.order_depths:
            orders: list[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            if product == "VOLCANIC_ROCK":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask1, best_ask_amount1 = list(order_depth.sell_orders.items())[0]
                    best_bid1, best_bid_amount1 = list(order_depth.buy_orders.items())[0]
                    volcano_s0 = (best_ask1 + best_bid1)/2
                
            if product == "VOLCANIC_ROCK_VOUCHER_10000":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    option_mid = (best_ask + best_bid)/2
            #result[product] = orders

        if volcano_s0 == 0 or option_mid == 0:
            pass
        else:
            # Day 5 to begin with
            T = (5 - state.timestamp/1000000)/252        
            iv = bisection_implied_vol(volcano_s0, 10000, T, 0, option_mid) * 100
            z_mid = iv - 14.2
            edge_0 = 0.5
            edge_max = 1.5
            
            edge_max_retreet = 1.5
            edge_0_retreet = 0.5
            position_max = 200
            position_10000 = state.position.get("VOLCANIC_ROCK_VOUCHER_10000", 0)

            
            pos_buy = size_function(z_mid, edge_0, edge_max, position_max)
            pos_sell = exit_size_function(z_mid, edge_0_retreet, edge_max_retreet, position_max)

            available_sell = abs(best_bid_amount)
            available_buy = abs(best_ask_amount)


            print(f"Position: {position_10000}")
            
            if z_mid > 0:               
                if pos_buy <= position_10000:
                    target_position = pos_buy
                    trade_needed = int(target_position - position_10000)
                    trade_multiplier = min(abs(trade_needed), available_buy, abs(best_bid_amount))
                    result['VOLCANIC_ROCK_VOUCHER_10000'] = [Order('VOLCANIC_ROCK_VOUCHER_10000', best_bid, -trade_multiplier)]
                else:
                    target_position = max(pos_sell, min(position_10000, 0))
                    trade_needed = int(target_position - position_10000)
                    trade_multiplier = min(abs(trade_needed), available_sell, abs(best_ask_amount))
                    result['VOLCANIC_ROCK_VOUCHER_10000'] = [Order('VOLCANIC_ROCK_VOUCHER_10000', best_ask, trade_multiplier)]
      
            elif z_mid < 0:
                if pos_buy >= position_10000:
                    target_position = pos_buy
                    trade_needed = int(target_position - position_10000)
                    trade_multiplier = min(abs(trade_needed), available_sell,abs(best_ask_amount))
                    result['VOLCANIC_ROCK_VOUCHER_10000'] = [Order('VOLCANIC_ROCK_VOUCHER_10000', best_ask, trade_multiplier)]
    
                else:
                    target_position = min(pos_sell, max(position_10000, 0))
                    trade_needed = int(target_position - position_10000)
                    trade_multiplier = min(abs(trade_needed), available_buy,abs(best_bid_amount))
                    result['VOLCANIC_ROCK_VOUCHER_10000'] = [Order('VOLCANIC_ROCK_VOUCHER_10000', best_bid, -trade_multiplier)]

        print(result)
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
