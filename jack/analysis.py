import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns


HISTORICAL_DATA_FILE = "historical_data.csv"
historical_data = pd.read_csv(HISTORICAL_DATA_FILE, sep=";")

# Plot KELP
kelp = historical_data[historical_data["product"] == "KELP"]


# Kelp, indexed by timestamp
kelp_t = kelp.copy(deep=True)
kelp_t.set_index("timestamp", inplace=True)


def plot():
    df = kelp.copy(deep=True)
    # Drop any rows with NaN in these columns.
    df = df.dropna(subset=["bid_price_1", "ask_price_1"])

    plt.figure(figsize=(10, 5))

    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    plt.plot(df["timestamp"], df["spread"], label="Spread")
    # plt.plot(df["timestamp"], df["bid_price_1"], label="Bid Price 1")
    # plt.plot(df["timestamp"], df["ask_price_1"], label="Ask Price 1")

    plt.xlabel("Timestamp")
    plt.ylabel("Spread")
    plt.title("Spread Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()


# "distance" is meant to be a representation of imbalance in the orderbook
# distance = sum(for all levels)((1/(midpoint-level.price))*level.volume)
# trying to find relationship between distance and returns
def calculate_distance(timestamp):
    total_dist = 0.0

    if pd.isna(kelp_t.loc[timestamp]["bid_price_1"]):
        raise Exception(f"At timestamp {timestamp} we have NaN bid_price_1")

    if pd.isna(kelp_t.loc[timestamp]["ask_price_1"]):
        raise Exception(f"At timestamp {timestamp} we have NaN ask_price_1")

    midpoint = (kelp_t.loc[timestamp]["ask_price_1"] + kelp_t.loc[timestamp]["bid_price_1"]) / 2

    # Bids
    for i in range(1, 4, 1):
        for side in ["bid", "ask"]:
            price = kelp_t.loc[timestamp][f"{side}_price_{i}"]

            if not pd.isna(price):
                volume = kelp_t.loc[timestamp][f"{side}_volume_{i}"]
                dist = midpoint - price
                total_dist += (1 / dist) * volume  # "Top level more significant"

    return total_dist


if __name__ == "__main__":
    df = kelp.copy(deep=True).reset_index(drop=True)

    distance_list = []
    delta_prices: dict[str, list] = {}

    price_intervals = [1, 3, 10, 30]

    for pi in price_intervals:
        delta_prices[f"delta_price_{pi}_list"] = []
        # represents p_t - p_{t-pi}

    for i, row in df.iterrows():
        distance_list.append(calculate_distance(row["timestamp"]))

        for pi in price_intervals:
            if i - pi >= 0:
                midpoint_t = row["ask_price_1"] - row["bid_price_1"]
                prev_row = df.loc[i - pi]
                midpoint_tminuspi = prev_row["ask_price_1"] - prev_row["bid_price_1"]
                delta_prices[f"delta_price_{pi}_list"].append(midpoint_t - midpoint_tminuspi)
            else:
                delta_prices[f"delta_price_{pi}_list"].append(0)

    # Build indicators DataFrame
    indicators = pd.DataFrame(
        {
            "distance": distance_list,
        }
    )

    for pi in price_intervals:
        indicators[f"delta_price_{pi}_list"] = delta_prices[f"delta_price_{pi}_list"]

    # Generate pairwise scatterplot grid
    sns.pairplot(indicators)
    plt.suptitle("Pairwise Scatter Plots of Distance and Δ Prices", y=1.02)
    plt.show()
