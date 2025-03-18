How to use the backtester.


The backtester needs three main things.
1. Market data (to backtest on)
2. Trade history (to backtest on)
3. Trading algo (to backtest)

 

How to use the backtester?

bt = Backtester(trader, listings, position_limit, fair_value_evaluator, market_data, trade_history, output_log_filename):

- Trader is your trader algo.
- Listings is the symbols (stocks) that are listed.
- Position_limit is the position limit for each.
- Fair_value_evaluator are functions that evaluate the value of each symbol (used for approximating PnL as the backtester runs).
- Market_data is a pandas dataframe of the market data.
- Trade_history is a pandas dataframe of the trade history.
- Output_log_filename is a path to the desired output logfile after backtesting. (The purpose of this is for the visualiser - not built yet).


 
Check example.ipynb for an example of how to use.

 