import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import List, Dict, Any, Callable
import matplotlib.pyplot as plt

from backtester.datamodel import TradingState, Listing, OrderDepth, Trade, Order, Observation
from backtester.log import Log
from backtester.algo import Trader
import copy


# Dataframe mimics the behaviour of a pandas dataframe except uses dicts and lists to store data
# This is because a real pd.Dataframe has very high overhead
# (This is a dataclass)
class Dataframe:
    # Fields
    cash: defaultdict
    position: defaultdict
    midpoint_pnl: defaultdict
    spreadcrossing_pnl: defaultdict

    def __init__(self):
        self.position = defaultdict(int)

        self.midpoint_cash = defaultdict(float)
        self.spreadcrossing_cash = defaultdict(float)

        self.midpoint_pnl = defaultdict(list[int])
        self.spreadcrossing_pnl = defaultdict(list[int])


class Backtester:
    def __init__(
        self,
        trader,
        listings: Dict[str, Listing],
        position_limit: Dict[str, int],
        market_data: pd.DataFrame,
        trade_history: pd.DataFrame,
        output_log_filename: str = None,
    ):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data.copy()
        self.position_limit = position_limit
        self.trade_history = trade_history.copy()
        self.output_log_filename = output_log_filename

        # For marking PNL, and tracking cash/position
        self.data = Dataframe()

        # Contains a log of the trades done over time
        self.new_trade_history: list[Trade] = []

        # Sandbox logs. For the output log file.
        self.sandbox_logs = []

        # optimisation - filter data to only the products we want to test on
        symbols = self.listings.keys()

        self.market_data = self.market_data[self.market_data["product"].isin(symbols)]
        self.trade_history = self.trade_history[self.trade_history["symbol"].isin(symbols)]

    def run(self):
        traderData = ""

        timestamp_group_md = self.market_data.groupby("timestamp")
        timestamp_group_th = self.trade_history.groupby("timestamp")

        own_trades = defaultdict(list)
        market_trades = defaultdict(list)

        trade_history_dict: Dict[int, List] = {}

        for timestamp, group in timestamp_group_th:
            trades = []
            for _, row in group.iterrows():
                symbol = row["symbol"]
                price = row["price"]
                quantity = row["quantity"]
                buyer = row["buyer"]
                seller = row["seller"]

                trade = Trade(symbol, int(price), int(quantity), buyer, seller, timestamp)

                trades.append(trade)
            trade_history_dict[timestamp] = trades

        # store midprices for later use
        # price = self.midprices[timestamp][product]
        self.midprices = defaultdict(dict)

        # trade_history timestamps are a subset of market_data timestamps.
        # (market_data timestamps include all existing timestamps)
        for timestamp, group in timestamp_group_md:
            for _, row in group.iterrows():
                self.midprices[timestamp][row["product"]] = row["mid_price"]

            order_depths_trader = self._construct_order_depths(group)  # passed to the trader
            order_depths_matching = self._construct_order_depths(group)  # used for matching
            order_depths_calc_pnl = self._construct_order_depths(group)  # shouldn't get modified, used to calculate pnl
            state = self._construct_trading_state(
                traderData,
                timestamp,
                self.listings.copy(),
                order_depths_trader,
                dict(own_trades.copy()),
                dict(market_trades.copy()),
                self.data.position.copy(),
            )
            orders, conversions, traderData = self.trader.run(state)
            sandbox_log = ""

            violated, sandbox_log = self._check_violate_position_limits(orders, sandbox_log)
            if violated:
                orders = {}
                print(f"Violating position limits. Cancelling all orders.")

                # i'm not sure if the prosperity website clears all orders, or all orders for this symbol
                # but, just ensure you don't violate position limits and you should be fine loll

            products = group["product"].tolist()
            trades_at_timestamp = trade_history_dict.get(timestamp, [])

            for product in products:
                trades_executed = []
                for order in orders.get(product, []):
                    new_trades, sandbox_log = self._match_order(timestamp, order, order_depths_matching, trade_history_dict, sandbox_log)
                    trades_executed.extend(new_trades)
                own_trades[product] = trades_executed
            self.sandbox_logs.append({"sandbox_log": sandbox_log, "lambdaLog": "", "timestamp": timestamp})

            trades_at_timestamp = trade_history_dict.get(timestamp, [])
            if trades_at_timestamp:
                for trade in trades_at_timestamp:
                    market_trades[trade.symbol].append(trade)
            else:
                for product in products:
                    market_trades[product] = []

            for product in products:
                # PNL of all products should be independent.
                self._update_cash_position_pnl(timestamp, product, order_depths_calc_pnl[product], own_trades[product])
            self._add_trades(own_trades, market_trades)

        # sort self.new_trade_history in time order
        self.new_trade_history = sorted(self.new_trade_history, key=lambda x: x.timestamp)

        # log results
        if self.output_log_filename != None:
            self._create_log_file(self.output_log_filename)

        return

    # Checks if it is possible to violate position limits with a selection of orders.
    # Returns true if possible
    # Returns false if not
    def _check_violate_position_limits(self, orderbook: dict[str, list[Order]], sandbox_log: str):
        max_possible_pos = self.data.position.copy()
        min_possible_pos = self.data.position.copy()

        violated = False
        for symbol, orders in orderbook.items():
            for order in orders:
                if order.quantity < 0:
                    min_possible_pos[symbol] += order.quantity
                elif order.quantity > 0:
                    max_possible_pos[symbol] += order.quantity

        for symbol, _ in max_possible_pos.items():
            if abs(max_possible_pos[symbol]) > self.position_limit[symbol] or abs(min_possible_pos[symbol]) > self.position_limit[symbol]:
                violated = True
                sandbox_log += f"\nOrders for product {symbol} exceeded limit of {self.position_limit[symbol]} set"

        return violated, sandbox_log

    # This function does not check that orders are valid.
    # Ensure that position limits are respected before calling this function.
    def _match_order(self, timestamp, order, order_depths, trade_history_dict, sandbox_log):
        if order.quantity == 0:
            return [], sandbox_log

        if order.quantity > 0:
            return self._match_buy_order(timestamp, order, order_depths, trade_history_dict, sandbox_log)
        else:
            return self._match_sell_order(timestamp, order, order_depths, trade_history_dict, sandbox_log)

    def _match_buy_order(self, timestamp, order, order_depths, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            trades.append(Trade(order.symbol, price, trade_volume, "SUBMISSION", "", timestamp))
            order_depth.sell_orders[price] += trade_volume
            order.quantity -= trade_volume

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]

        # we try to match our orders here to the trades_at_timestamp. any that aren't matched go back to trade_history_dict
        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        updated_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price < order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))

                    if trade_volume != 0:
                        trades.append(Trade(order.symbol, order.price, trade_volume, "SUBMISSION", "", timestamp))
                        order.quantity -= trade_volume

                    new_quantity = trade.quantity - trade_volume
                    if new_quantity != 0:
                        updated_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            updated_trades_at_timestamp.append(trade)

        trade_history_dict[timestamp] = updated_trades_at_timestamp

        return trades, sandboxLog

    def _match_sell_order(self, timestamp, order, order_depths, trade_history_dict, sandbox_log):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            trades.append(Trade(order.symbol, price, trade_volume, "", "SUBMISSION", timestamp))
            order_depth.buy_orders[price] -= abs(trade_volume)
            order.quantity += trade_volume

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        updated_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                # has to be strict inequality
                # this is because of price-time priority.
                # any trade that occured in history meant it got matched to an existing trade in the orderbook
                # this means, that orderbook quote will already be there, and that will get priority over us.
                # so we won't be able to match anything that is equal.
                if trade.price > order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))

                    if trade_volume != 0:

                        trades.append(Trade(order.symbol, order.price, trade_volume, "", "SUBMISSION", timestamp))
                        order.quantity += trade_volume

                        new_quantity = trade.quantity - trade_volume

                        if new_quantity != 0:
                            updated_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            updated_trades_at_timestamp.append(trade)

        trade_history_dict[timestamp] = updated_trades_at_timestamp

        return trades, sandbox_log

    def _construct_trading_state(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position):
        state = TradingState(traderData, timestamp, listings, order_depths, own_trades, market_trades, position, None)
        return state

    def _construct_order_depths(self, group):
        order_depths = {}
        for idx, row in group.iterrows():
            product = row["product"]
            order_depth = OrderDepth()
            for i in range(1, 4):
                if f"bid_price_{i}" in row and f"bid_volume_{i}" in row:
                    bid_price = row[f"bid_price_{i}"]
                    bid_volume = row[f"bid_volume_{i}"]
                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                        order_depth.buy_orders[int(bid_price)] = int(bid_volume)
                if f"ask_price_{i}" in row and f"ask_volume_{i}" in row:
                    ask_price = row[f"ask_price_{i}"]
                    ask_volume = row[f"ask_volume_{i}"]
                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                        order_depth.sell_orders[int(ask_price)] = -int(ask_volume)
            order_depths[product] = order_depth
        return order_depths

    def _add_trades(self, own_trades: Dict[str, List[Trade]], market_trades: Dict[str, List[Trade]]):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.new_trade_history.extend(own_trades.get(product, []))
        for product in products:
            self.new_trade_history.extend(market_trades.get(product, []))

    def _create_log_file(self, filename: str):
        # spreadcrossing pnl is like the 'real' pnl.
        self.market_data["profit_and_loss"] = self.data.spreadcrossing_pnl

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(index=False, sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        new_trade_history_jsons = [self._trade_to_dict(trade) for trade in self.new_trade_history]
        output += json.dumps(new_trade_history_jsons, indent=2)

        with open(filename, "w") as file:
            file.write(output)

    # Update the cash, position, and pnl at timestamp.
    # The repeated calling of _match_order function matches all possible orders, but does not modify position nor cash.
    # Here, we calculate the position, and also the spreadcrossing and the midpoint PNL.
    # We do this for just one timestamp and one product.
    # order_depth and executed_trades correspond to exactly this timestamp and product.
    def _update_cash_position_pnl(self, timestamp: int, product: str, order_depth: OrderDepth, executed_trades: list[Trade]):
        # to evaluate PNL at a particular point we need cash, and product position * fair price
        # the default method of evaluating fair price is to use the midprice.
        # do this regardless of whether we are calculating midpoint or spreadcrossing pnl.

        # market data history handles cases where there is no buy or there is no sell orders.
        midprice = self.midprices[timestamp][product]

        # MIDPOINT - assume trade is executed at midpoint
        # Spreadcrossing - this is the price the trade actually gets executed at.

        for trade in executed_trades:
            qty = trade.quantity
            price = trade.price

            if trade.buyer == "SUBMISSION":
                self.data.position[product] += qty

                self.data.midpoint_cash[product] -= qty * midprice
                self.data.spreadcrossing_cash[product] -= qty * price

            elif trade.seller == "SUBMISSION":
                self.data.position[product] -= qty

                self.data.midpoint_cash[product] += qty * midprice
                self.data.spreadcrossing_cash[product] += qty * price

        midpoint_pnl = self.data.midpoint_cash[product] + self.data.position[product] * midprice
        spreadcrossing_pnl = self.data.spreadcrossing_cash[product] + self.data.position[product] * midprice

        self.data.midpoint_pnl[product].append(midpoint_pnl)
        self.data.spreadcrossing_pnl[product].append(spreadcrossing_pnl)

    def calculate_metrics(self, product) -> dict[str, float]:
        midpoint_pnl = self.data.midpoint_pnl[product]
        spreadcrossing_pnl = self.data.spreadcrossing_pnl[product]

        # CALCULATE SHARPE
        midpoint_returns = np.diff(midpoint_pnl)
        spreadcrossing_returns = np.diff(spreadcrossing_pnl)

        midpoint_sharpe = np.mean(midpoint_returns) / np.std(midpoint_returns) * np.sqrt(252)  # annualize it
        spreadcrossing_sharpe = np.mean(spreadcrossing_returns) / np.std(spreadcrossing_returns) * np.sqrt(252)  # annualize it

        # CALCULATE PNL BPS
        total_notional = 0
        for trade in self.new_trade_history:
            if trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION":
                total_notional += trade.price * abs(trade.quantity)

        # Use final PnL (e.g. midpoint)
        final_midpoint_pnl = midpoint_pnl[-1]
        final_spreadcrossing_pnl = spreadcrossing_pnl[-1]

        res = {
            "timestamp": np.unique(self.market_data["timestamp"]),  # for convenicence.
            "spreadcrossing_pnl": spreadcrossing_pnl,
            "spreadcrossing_final_pnl": spreadcrossing_pnl[-1],
            "spreadcrossing_sharpe": spreadcrossing_sharpe,
            "spreadcrossing_pnl_bps": (final_spreadcrossing_pnl / total_notional) * 1e4,
            "midpoint_pnl": midpoint_pnl,
            "midpoint_final_pnl": midpoint_pnl[-1],
            "midpoint_sharpe": midpoint_sharpe,
            "midpoint_pnl_bps": (final_midpoint_pnl / total_notional) * 1e4,
        }

        return res

    # PNL returns the pnl for all instruments
    # both spreadcrossing and midpoint
    def pnl(self) -> dict[str, float]:
        res = {
            "spreadcrossing": {sym: pnl[-1] for sym, pnl in self.data.spreadcrossing_pnl.items()},
            "midpoint": {sym: pnl[-1] for sym, pnl in self.data.midpoint_pnl.items()},
        }

        res["spreadcrossing"]["total"] = np.sum([pnl for sym, pnl in res["spreadcrossing"].items()])
        res["midpoint"]["total"] = np.sum([pnl for sym, pnl in res["midpoint"].items()])

        return res

    def _trade_to_dict(self, trade: Trade) -> dict[str, Any]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "price": float(trade.price),
            "quantity": float(trade.quantity),
        }
