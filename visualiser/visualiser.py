import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import dash
import numpy as np
import pandas as pd
from dash import dcc, html
import plotly.express as px
from backtester.log import Log
from dash.dependencies import Input, Output

from backtester.datamodel import Trade

import base64
import io

# Create a figure
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


class Visualiser:
    market_data: pd.DataFrame  # log.activities_log
    trade_history: list[Trade]  # log.trade_history

    def __init__(self, log: Log = None):
        if log is None:
            # Initial layout with only file upload, a container for graphs, and a hidden Store.
            app.layout = html.Div(
                [
                    html.H1("Visualiser", id="page-title"),
                    self._construct_file_upload_component(),
                    html.Div(id="graph-container"),  # Initially empty
                    dcc.Store(id="log-loaded", data=False),  # Hidden store to trigger graph display
                ]
            )
        else:
            self.market_data = log.activities_log
            self.trade_history = log.trade_history
            self.display()

    # Sets the visualiser to visualise a new log
    def load_log(self, log: Log):
        self.market_data = log.activities_log
        self.trade_history = log.trade_history

    # _construct_price_section constructs:
    #   Dropdown to select which symbol
    #   A graph of price for symbol
    # Returns: html.Div, wrapping these elements
    def _construct_price_section(self) -> html.Div:
        # Combine all symbols into one DataFrame
        df = self.market_data[["timestamp", "product", "mid_price"]]

        # Create a single line plot, colored by product
        figure_midprice = px.line(
            df,
            x="timestamp",
            y="mid_price",
            color="product",
            labels={"timestamp": "Time", "mid_price": "MidPrice", "product": "Symbol"},
            title="MidPrice Over Time for All Symbols",
        )

        return html.Div(dcc.Graph(figure=figure_midprice))

    # _construct_pnl_section constructs:
    #   Dropdown to select which symbol
    #   A graph of PNL for symbol
    # Returns: html.Div, wrapping these elements
    def _construct_pnl_section(self) -> html.Div:
        # Combine all symbols into one DataFrame
        df = self.market_data[["timestamp", "product", "profit_and_loss"]]

        # Compute ETF net PNL from individual product PNLs
        pivot_df = df.pivot(index="timestamp", columns="product", values="profit_and_loss")
        pivot_df["ETF_NET_PNL"] = (
            pivot_df.get("PICNIC_BASKET1", 0)
            + pivot_df.get("PICNIC_BASKET2", 0)
            + pivot_df.get("CROISSANTS", 0)
            + pivot_df.get("DJEMBES", 0)
            + pivot_df.get("JAMS", 0)
        )

        etf_pnl_df = pivot_df[["ETF_NET_PNL"]].reset_index()
        etf_pnl_df["product"] = "ETF_NET_PNL"
        etf_pnl_df = etf_pnl_df.rename(columns={"ETF_NET_PNL": "profit_and_loss"})

        df = pd.concat([df, etf_pnl_df], ignore_index=True)
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        # Create a single line plot, colored by product
        figure_pnl = px.line(
            df,
            x="timestamp",
            y="profit_and_loss",
            color="product",
            labels={"timestamp": "Time", "profit_and_loss": "PNL", "product": "Symbol"},
            title="PNL Over Time for All Symbols",
        )

        return html.Div(dcc.Graph(figure=figure_pnl))

    # _construct_fills_graph_section constructs:
    #   Dropdown to select which symbol
    #   A graph of trades filled for symbol
    # Returns: html.Div, wrapping these elements
    def _construct_fills_graph_section(self) -> html.Div:
        # Build full dataframe including all trades
        self.fill_df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "price": t.price,
                    "quantity": abs(t.quantity),
                    "side": ("Buy" if t.buyer == "SUBMISSION" else "Sell" if t.seller == "SUBMISSION" else "Bot Trade"),
                    "symbol": t.symbol,
                    "buyer": t.buyer,
                    "seller": t.seller,
                    "is_bot_trade": not (t.buyer == "SUBMISSION" or t.seller == "SUBMISSION"),
                }
                for t in self.trade_history
            ]
        )

        symbols = self.fill_df["symbol"].unique()

        return html.Div(
            [
                html.Label("Select Symbol:"),
                dcc.Dropdown(
                    id="symbol-dropdown",
                    options=[{"label": s, "value": s} for s in symbols],
                    value=symbols[0],
                    clearable=False,
                ),
                html.Br(),
                html.Label("Show bot-to-bot trades?"),
                dcc.Checklist(
                    id="bot-toggle",
                    options=[{"label": "Include Bot Trades", "value": "show"}],
                    value=[],  # Default is unchecked
                    inputStyle={"margin-right": "5px", "margin-left": "10px"},
                ),
                html.Br(),
                dcc.Graph(id="fills-graph"),
            ]
        )

    # _calculate_position_over_time calculates position over time
    # Parameters: a symbol like "KELP".
    # Returns: two lists, the net position, and timestamp
    def _calculate_position_over_time(self, symbol: str) -> tuple[list[int], list[int]]:
        positions = [0]
        timestamps = [0]

        current_pos = 0

        for trade in self.trade_history:
            if trade.symbol != symbol:
                continue

            if trade.buyer == "SUBMISSION":
                current_pos += abs(trade.quantity)
            elif trade.seller == "SUBMISSION":
                current_pos -= abs(trade.quantity)
            else:
                continue

            positions.append(current_pos)
            timestamps.append(trade.timestamp)

        # Edge case so it displays constant 0 position instead of just 1 dot.
        if len(positions) == 1:
            # We made no trades
            max_timestamp = self.market_data.iloc[len(self.market_data) - 1]["timestamp"]
            positions.append(0)
            timestamps.append(max_timestamp)

        ## TEMPORRARY

        # df = pd.DataFrame(columns=["timestamp", f"position_{symbol}"])
        # df["timestamp"] = timestamps
        # df[f"position_{symbol}"] = positions
        # df.to_csv(f"position_over_time_{symbol}.csv")

        ##TEMPORARY

        return timestamps, positions

    # _construct_net_position_section constructs:
    #   Dropdown to select which symbol
    #   A graph of net position for symbol
    # Returns: html.Div, wrapping these elements
    def _construct_net_position_section(self) -> html.Div:
        symbols = np.unique(self.market_data["product"])

        return html.Div(
            [
                html.Label("Select Symbol for Position:"),
                dcc.Dropdown(
                    id="position-symbol-dropdown",
                    options=[{"label": s, "value": s} for s in symbols],
                    value=symbols[0],
                    clearable=False,
                ),
                html.Br(),
                dcc.Graph(id="position-graph"),
            ]
        )

    # _construct_file_upload_component constructs:
    #   Component to select log file to visualise
    # Returns: html.Div, wrapping this element
    def _construct_file_upload_component(self) -> html.Div:
        return html.Div(
            [
                html.H2("Upload a new log file"),
                dcc.Upload(
                    id="upload-log",
                    children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=False,
                ),
            ]
        )

    # When log data is ready, build and return the graph components.
    def get_graph_components_to_display(self) -> list[html.Div]:
        pnl_graph_component = self._construct_pnl_section()
        fills_graph_component = self._construct_fills_graph_section()
        net_position_graph_component = self._construct_net_position_section()
        midprice_graph_component = self._construct_price_section()
        return [midprice_graph_component, pnl_graph_component, fills_graph_component, net_position_graph_component]

    # For when a log is provided at startup, immediately display everything.
    def display(self):
        app.layout = html.Div([html.H1("Visualiser")] + self.get_graph_components_to_display())


# VISUALISER
# logpath = os.path.join(__file__, "..", "..", "jack", "trader1.log")
# logpath = os.path.join(__file__, "..", "no_mm_test.log")
vis = Visualiser()
# END VISUALISER


# CALLBACKS
@app.callback(Output("fills-graph", "figure"), Input("symbol-dropdown", "value"), Input("bot-toggle", "value"))
def update_fills_plot(selected_symbol, toggle_value):
    df = vis.fill_df[vis.fill_df["symbol"] == selected_symbol]

    if "show" not in toggle_value:
        df = df[df["side"] != "Bot Trade"]

    fig = px.scatter(
        df,
        x="timestamp",
        y="price",
        size="quantity",
        color="side",
        color_discrete_map={"Buy": "green", "Sell": "red", "Bot Trade": "gray"},
        labels={"timestamp": "Time", "price": "Price", "quantity": "Quantity", "side": "Side", "buyer": "Buyer", "seller": "Seller"},
        title=f"Filled Trades for {selected_symbol}",
        hover_data=["buyer", "seller"]
    )
    return fig


@app.callback(Output("position-graph", "figure"), Input("position-symbol-dropdown", "value"))
def update_position_plot(symbol):
    timestamps, positions = vis._calculate_position_over_time(symbol)

    df = pd.DataFrame({"timestamp": timestamps, "position": positions})

    fig = px.line(df, x="timestamp", y="position", labels={"timestamp": "Time", "position": "Net Position"}, title=f"Position Over Time for {symbol}")

    return fig


@app.callback(
    [Output("log-loaded", "data"), Output("page-title", "children")],
    Input("upload-log", "contents"),
    Input("upload-log", "filename"),
    prevent_initial_call=True,
)
def update_log_from_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    # Decode and save the file
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    file_data = io.BytesIO(decoded)
    text_content = file_data.read().decode("utf-8")

    # # Load the log and update the visualiser
    new_log = Log(content=text_content)
    vis.load_log(new_log)

    # Return True to signal the file is loaded and update the title with the filename
    return True, f"Visualiser: {filename}"


# Callback to update the graph container once a file is loaded.
@app.callback(
    Output("graph-container", "children"),
    Input("log-loaded", "data"),
    prevent_initial_call=True,
)
def display_graphs_when_ready(log_loaded):
    if not log_loaded:
        raise dash.exceptions.PreventUpdate
    return vis.get_graph_components_to_display()


# Run the server
if __name__ == "__main__":

    app.run(debug=True)
