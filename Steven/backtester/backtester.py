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


class Backtester:
    def __init__(
        self,
        trader,
        listings: Dict[str, Listing],
        position_limit: Dict[str, int],
        fair_value_evaluator: Dict[str, Callable[[OrderDepth], float]],
        market_data: pd.DataFrame,
        trade_history: pd.DataFrame,
        output_log_filename: str = None,
        # Debug flags
        spread_crossing_warning: bool = False,
    ):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data
        self.position_limit = position_limit
        self.fair_value_evaluator = fair_value_evaluator
        self.trade_history = trade_history
        self.output_log_filename = output_log_filename

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}

        # for the market_data['profit_and_loss'] column
        # will the pnl be always counted differently for different symbols? or together?
        self.pnl_history = []

        self.trades = []
        self.sandbox_logs = []

        # Metrics
        # format: f"{metric}_{symbol}"
        # like "spreadcrossing_pnl_KELP"

        self.metrics = defaultdict(lambda: defaultdict(dict))
        metrics = ["spreadcrossing_pnl", "midpoint_pnl"]

        self.spread_crossing_warning = spread_crossing_warning

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

        for timestamp, group in timestamp_group_md:
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
                self.current_position.copy(),
            )
            orders, conversions, traderData = self.trader.run(state)
            sandboxLog = ""

            violated, sandboxLog = self._check_violate_position_limits(orders, sandboxLog)
            if violated:
                orders = {}
                print(f"Violating position limits. Cancelling all orders.")

                # i'm not sure if the prosperity website clears all orders, or all orders for this symbol
                # but, just ensure you don't violate position limits and you should be fine loll

            products = group["product"].tolist()
            trades_at_timestamp = trade_history_dict.get(timestamp, [])

            for product in products:
                new_trades = []
                for order in orders.get(product, []):
                    trades_done, sandboxLog = self._execute_order(
                        timestamp, order, order_depths_matching, self.current_position, trade_history_dict, sandboxLog
                    )
                    new_trades.extend(trades_done)
                own_trades[product] = new_trades
            self.sandbox_logs.append({"sandboxLog": sandboxLog, "lambdaLog": "", "timestamp": timestamp})

            trades_at_timestamp = trade_history_dict.get(timestamp, [])
            if trades_at_timestamp:
                for trade in trades_at_timestamp:
                    market_trades[trade.symbol].append(trade)
            else:
                for product in products:
                    market_trades[product] = []

            for product in products:
                self._mark_pnl(timestamp, self.cash, self.current_position, order_depths_calc_pnl, self.pnl, product)
                self.pnl_history.append(self.pnl[product])
            self._add_trades(own_trades, market_trades)

        # sort self.trades
        self.trades = sorted(self.trades, key=lambda x: x["timestamp"])
        self._log_trades(self.output_log_filename)
        return

    # Checks if it is possible to violate position limits with a selection of orders.
    # Returns true if possible
    # Returns false if not
    def _check_violate_position_limits(self, orderbook: dict[str, list[Order]], sandboxLog: str):
        max_possible_pos = self.current_position.copy()
        min_possible_pos = self.current_position.copy()

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
                sandboxLog += f"\nOrders for product {symbol} exceeded limit of {self.position_limit[symbol]} set"

        return violated, sandboxLog

    def _log_trades(self, filename: str = None):
        if filename is None:
            return

        self.market_data["profit_and_loss"] = self.pnl_history

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(index=False, sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)

        with open(filename, "w") as file:
            file.write(output)

    def _add_trades(self, own_trades: Dict[str, List[Trade]], market_trades: Dict[str, List[Trade]]):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in own_trades.get(product, [])])
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in market_trades.get(product, [])])

    def _trade_to_dict(self, trade: Trade) -> dict[str, Any]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "price": trade.price,
            "quantity": trade.quantity,
        }

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

    def _execute_buy_order(self, timestamp, order, order_depths, position, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if not isinstance(price, int):
                raise Exception("Order price must be int.")
            if not isinstance(volume, int):
                raise Exception("Order volume must be int.")

            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + position[order.symbol]) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "SUBMISSION", "", timestamp))
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                raise Exception(
                    f"Orders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set, but this should already have been checked."
                )
                # sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        updated_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price < order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))

                    if trade_volume != 0:
                        trades.append(Trade(order.symbol, order.price, trade_volume, "SUBMISSION", "", timestamp))
                        order.quantity -= trade_volume
                        position[order.symbol] += trade_volume
                        self.cash[order.symbol] -= order.price * trade_volume

                    new_quantity = trade.quantity - trade_volume
                    if new_quantity != 0:
                        updated_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            updated_trades_at_timestamp.append(trade)

        trade_history_dict[timestamp] = updated_trades_at_timestamp

        return trades, sandboxLog

    def _execute_sell_order(self, timestamp, order, order_depths, position, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if not isinstance(price, int):
                raise Exception("Order price must be int.")
            if not isinstance(volume, int):
                raise Exception("Order volume must be int.")

            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(position[order.symbol] - trade_volume) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "", "SUBMISSION", timestamp))
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                raise Exception(
                    f"Orders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set, but this should already have been checked."
                )
                # sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        updated_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price > order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))

                    if trade_volume != 0:

                        trades.append(Trade(order.symbol, order.price, trade_volume, "", "SUBMISSION", timestamp))
                        order.quantity += trade_volume
                        position[order.symbol] -= trade_volume
                        self.cash[order.symbol] += order.price * trade_volume

                        new_quantity = trade.quantity - trade_volume

                        if new_quantity != 0:
                            updated_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            updated_trades_at_timestamp.append(trade)

        trade_history_dict[timestamp] = updated_trades_at_timestamp

        return trades, sandboxLog

    def _execute_order(self, timestamp, order, order_depths, position, trades_at_timestamp, sandboxLog):
        if order.quantity == 0:
            return [], sandboxLog

        if order.quantity > 0:
            return self._execute_buy_order(timestamp, order, order_depths, position, trades_at_timestamp, sandboxLog)
        else:
            return self._execute_sell_order(timestamp, order, order_depths, position, trades_at_timestamp, sandboxLog)

    def _mark_pnl(self, timestamp, cash, position, order_depths, pnl, product):
        # to evaluate PNL at a particular point we need cash, and product position * fair price
        # the default method of evaluating fair price is to average the best_ask and best_bid, but you can override this with your own function.

        order_depth = order_depths[product]
        if product in self.fair_value_evaluator:
            evaluator = self.fair_value_evaluator[product]
            fair_price = evaluator(order_depth)
        else:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid = (best_ask + best_bid) / 2
            fair_price = mid

        pnl[product] = cash[product] + fair_price * position[product]
        midpoint_pnl = pnl[product]

        # now calculate the spread crossing pnl
        liquidated_position_value = 0
        if position[product] > 0:
            # we are long, must sell
            pos = position[product]
            last_executed_level = 0
            for level, volume in sorted(order_depth.buy_orders.items(), key=lambda x: -x[0]):
                liquidated_position_value += min(pos, volume) * level
                pos = pos - min(pos, volume)
                last_executed_level = level
                if pos == 0:
                    break

            if pos > 0:
                if self.spread_crossing_warning:
                    print(
                        f"Warning: in calculating spread-crossing pnl, not possible to liquidate entire position. Assuming liquidation of long {pos} more units at {last_executed_level}."
                    )
                liquidated_position_value += last_executed_level * pos

        elif position[product] < 0:
            # we are short, must buy
            pos = position[product]
            last_executed_level = np.inf
            for level, volume in sorted(order_depth.sell_orders.items(), key=lambda x: x[0]):  # for sell orders, volume is negative
                liquidated_position_value -= min(np.abs(pos), np.abs(volume)) * level
                pos = pos + min(np.abs(pos), np.abs(volume))
                last_executed_level = level
                if pos == 0:
                    break

            if pos < 0:
                if self.spread_crossing_warning:
                    print(
                        f"Warning: in calculating spread-crossing pnl, not possible to liquidate entire position. Assuming liquidation of short {pos} more units at {last_executed_level}."
                    )
                liquidated_position_value += last_executed_level * pos

        spreadcrossing_pnl = cash[product] + liquidated_position_value

        self.metrics[timestamp][f"midpoint_pnl_{product}"] = midpoint_pnl
        self.metrics[timestamp][f"spreadcrossing_pnl_{product}"] = spreadcrossing_pnl

    def calculate_metrics(self, product) -> dict[str, float]:
        # if not self.metrics_enabled:
        #     raise Exception("Error: Metrics are not enabled. Enable metrics to call this function")

        midpoint_pnl = [d[f"midpoint_pnl_{product}"] for _, d in self.metrics.items()]
        spreadcrossing_pnl = [d[f"spreadcrossing_pnl_{product}"] for _, d in self.metrics.items()]

        # CALCULATE SHARPE
        midpoint_returns = np.diff(midpoint_pnl)
        spreadcrossing_returns = np.diff(spreadcrossing_pnl)

        midpoint_sharpe = np.mean(midpoint_returns) / np.std(midpoint_returns) * np.sqrt(252)  # annualize it
        spreadcrossing_sharpe = np.mean(spreadcrossing_returns) / np.std(spreadcrossing_returns) * np.sqrt(252)  # annualize it

        # CALCULATE PNL BPS
        total_notional = 0
        for trade in self.trades:
            total_notional += trade["price"] * abs(trade["quantity"])

        # Use final PnL (e.g. midpoint)
        final_midpoint_pnl = midpoint_pnl[-1]
        final_spreadcrossing_pnl = spreadcrossing_pnl[-1]

        res = {
            "midpoint_sharpe": midpoint_sharpe,
            "spreadcrossing_sharpe": spreadcrossing_sharpe,
            "midpoint_pnl_bps": (final_midpoint_pnl / total_notional) * 1e4,
            "spreadcrossing_pnl_bps": (final_spreadcrossing_pnl / total_notional) * 1e4,
        }
        return res

    # retrieves a metric as a list, in time order
    def get_metric(self, metric, product) -> list:
        return [d[f"{metric}_{product}"] for _, d in self.metrics.items()]


if __name__ == "__main__":
    trader = Trader()

    listings = {
        "KELP": Listing(symbol="KELP", product="KELP", denomination="SEASHELLS"),
        "RAINFOREST_RESIN": Listing(symbol="RAINFOREST_RESIN", product="RAINFOREST_RESIN", denomination="SEASHELLS"),
    }

    position_limit = {
        "KELP": 50,
        "RAINFOREST_RESIN": 50,
    }

    def calc_rainforest_resin_fair(order_depth: OrderDepth) -> float:
        return 10000

    fair_value_evaluator = {
        # omit dictionary entry for kelp, so that it uses default behaviour (best_bid+best_ask)/2
        "RAINFOREST_RESIN": calc_rainforest_resin_fair,
    }

    market_data = pd.read_csv(os.path.join(__file__, "..", "..", "data", "tutorial", "market_data.csv"), sep=";")
    trade_history = pd.read_csv(os.path.join(__file__, "..", "..", "data", "tutorial", "trade_history.csv"), sep=";")

    bt = Backtester(
        trader=trader,
        listings=listings,
        position_limit=position_limit,
        fair_value_evaluator=fair_value_evaluator,
        market_data=market_data,
        trade_history=trade_history,
        output_log_filename="backtest1.log",
        spread_crossing_warning=False,
    )

    bt.run()
    product = "RAINFOREST_RESIN"  # or "KELP"

    resin_metrics = bt.calculate_metrics(product)
    print(f"PNL: {bt.pnl}")

    print(f"Midpoint Sharpe: {resin_metrics['midpoint_sharpe']:.4f}")
    print(f"Spreadcrossing Sharpe: {resin_metrics['spreadcrossing_sharpe']:.4f}")
    print(f"Midpoint PnL (bps): {resin_metrics['midpoint_pnl_bps']:.2f}")
    print(f"Spreadcrossing PnL (bps): {resin_metrics['spreadcrossing_pnl_bps']:.2f}")

    # THIS PART PLOTS SPREADCROSSING_PNL AND MIDPOINT_PNL
    spreadcrossing_pnl_history = bt.get_metric("spreadcrossing_pnl", product)
    midpoint_pnl_history = bt.get_metric("midpoint_pnl", product)
    timestamps = np.unique(bt.market_data["timestamp"])

    plt.plot(timestamps, spreadcrossing_pnl_history, label="Spreadcrossing PnL", color="blue")
    plt.plot(timestamps, midpoint_pnl_history, label="Midpoint PnL", color="orange")
    plt.xlabel("Timestamp")
    plt.ylabel("PnL")
    plt.title("Spread Crossing vs Midpoint PnL Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
