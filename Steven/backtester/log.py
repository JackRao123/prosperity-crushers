import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import pandas as pd
import json

from backtester.datamodel import Trade


# Log represents the data with a .log file downloaded from the prosperity dashboard
# There are 3 main parts: sandbox log, activities log, and trade history
class Log:
    sandbox_logs: list[dict]
    activities_log: pd.DataFrame
    trade_history: list[Trade]

    def __init__(self, content: str):
        self.sandbox_logs, self.activities_log, self.trade_history = self._parse_file_contents(content)

    @classmethod
    def from_file(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return cls(content)

    @classmethod
    def from_content(cls, content: str):
        return cls(content)

    # _parse_file_contents parses a string representation of the file.
    # returns sandbox_logs, activites_log, trade_history
    def _parse_file_contents(self, content: str):
        sandbox_start = content.find("Sandbox logs:")
        activities_start = content.find("Activities log:")
        trade_history_start = content.find("Trade History:")

        sandbox_text = content[sandbox_start + len("Sandbox logs:") : activities_start].strip()
        activities_text = content[activities_start + len("Activities log:") : trade_history_start].strip()
        trade_history_text = content[trade_history_start + len("Trade History:") :].strip()

        return self._parse_sandbox_logs(sandbox_text), self._parse_activities_log(activities_text), self._parse_trade_history(trade_history_text)

    def _parse_sandbox_logs(self, text: str) -> list[dict]:
        logs = []
        current_json = ""
        depth = 0

        for char in text:
            if char == "{":
                if depth == 0:
                    current_json = ""
                depth += 1
            if depth > 0:
                current_json += char
            if char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        logs.append(json.loads(current_json))
                    except json.JSONDecodeError:
                        pass
        return logs

    def _parse_activities_log(self, text: str) -> pd.DataFrame:
        from io import StringIO

        return pd.read_csv(StringIO(text), sep=";")

    def _parse_trade_history(self, text: str) -> list[Trade]:
        trades_dicts: list[dict] = json.loads(text)

        trades: list[Trade] = []
        for trade in trades_dicts:
            trades.append(
                Trade(
                    symbol=trade["symbol"],
                    price=trade["price"],
                    quantity=trade["quantity"],
                    buyer=trade["buyer"],
                    seller=trade["seller"],
                    timestamp=trade["timestamp"],
                )
            )

        return trades
