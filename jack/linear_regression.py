import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import seaborn as sns

from analysis import calculate_distance
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.preprocessing import PolynomialFeatures


# For KELP
class ModelCreator:
    def __init__(self):
        pass

    def preprocess(self):
        self.data = pd.read_csv("kelp.csv")
        self.data.columns = self.data.columns.str.strip()  # Clean up any whitespace

        self.data["mid_price"] = (self.data["bid_price_1"] + self.data["ask_price_1"]) / 2

        # steven imbalance metric
        imbalance = []  # imbalance = sum(for all levels)((1/(midpoint-level.price))*level.volume)
        for index, row in self.data.iterrows():
            total_imbalance = 0.0
            for i in range(1, 4, 1):
                for side in ["bid", "ask"]:
                    price = row[f"{side}_price_{i}"]

                    if not pd.isna(price):
                        volume = row[f"{side}_volume_{i}"]
                        dist = row["mid_price"] - price
                        total_imbalance += (1 / dist) * volume  # 'Top level more significant'

            imbalance.append(total_imbalance)

        self.data["imbalance"] = imbalance

        # log return
        # shift(1) moves the previous row's value forward. so therefore here logreturn[i] = logreturn from row i-1 to row i
        for dt in range(1, self.lookback + 1, 1):
            self.data[f"log_return_{dt}"] = np.log(self.data["mid_price"] / self.data["mid_price"].shift(dt)).fillna(0)

        # volatility (is this even relevant)
        self.data["volatility_10"] = self.data["log_return_1"].rolling(10).std()

        # momentum
        self.data["momentum_5"] = self.data["mid_price"].rolling(5).mean()

    def run_regression(self, lookahead=5, lookback=5, train_ratio=0.8, debug=False):
        self.lookback = lookback  # uses log returns from t-1, t-2,... t-lookback
        self.preprocess()

        # Target: future log return (t+lookahead)
        # shift(-lookahead) means target[i] gives the logreturn from t to t+lookahead
        self.data["target"] = np.log(self.data["mid_price"].shift(-lookahead) / self.data["mid_price"])
        self.data.dropna(subset=["target"], inplace=True)  # dropna for y

        # Features
        features = ["imbalance", "volatility_10", "momentum_5"] + [f"log_return_{dt}" for dt in range(1, self.lookback + 1)]
        self.data.dropna(subset=features, inplace=True)  # dropna for X
        X = self.data[features]
        y = self.data["target"]

        # attempt: to use polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Train/test split to prevent lookahead bias
        split_idx = int(len(self.data) * train_ratio)
        X_train, X_test = X_poly[:split_idx], X_poly[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        if debug:
            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            print(f"Lookahead: {lookahead}")

            train_r2 = r2_score(y_train, model.predict(X_train))
            train_mse = mean_squared_error(y_train, model.predict(X_train))
            print(f"Training r2: {train_r2:.4f}")
            print(f"Training MSE: {train_mse:.8f}")  # these will be very small, (not because its accurate) but because logreturns are just very small

            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)

            print(f"Test r2: {test_r2:.4f}")
            print(f"MSE: {test_mse:.8f}")  # these will be very small, (not because its accurate) but because logreturns are just very small

            print("Intercept:", model.intercept_)
            print("Coefficients: ")

            # for debugging, i wanna see what features are important (after poly transformation)
            coeffs = list(zip(poly.get_feature_names_out(input_features=X.columns), model.coef_))

            # Sort by absolute value (importance)
            sorted_coeffs = sorted(coeffs, key=lambda x: abs(x[1]), reverse=True)
            print("Top polynomial features by importance:")
            for name, value in sorted_coeffs[:100]:
                print(f"{name}: {value:.6f}")

            # Plot predicted vs actual
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label="Actual", alpha=0.7)
            plt.plot(y_pred, label="Predicted", alpha=0.7)
            plt.legend()
            plt.title(f"Log Return Prediction (lookahead={lookahead})")
            plt.xlabel("Time Index")
            plt.ylabel("Log Return")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return model.predict(X_poly)


if __name__ == "__main__":
    # model = creator.run_regression(lookahead=2, lookback=5, train_ratio=0.8)

    mid_prices = None

    predictions: dict[int, LinearRegression] = {}
    min_lookahead = 1
    max_lookahead = 10
    for lookahead in range(min_lookahead, max_lookahead + 1, 1):
        creator = ModelCreator()
        predictions[lookahead] = creator.run_regression(lookahead=lookahead, lookback=5, train_ratio=0.8)
        print("done")

        mid_prices = creator.data["mid_price"].to_numpy()

    # Start figure
    plt.figure(figsize=(12, 6))
    plt.plot(mid_prices, label="Actual Price", color="black", linewidth=1.5)

    for i in range(len(mid_prices) - max_lookahead):
        base_price = mid_prices[i]

        # Predicted future prices using log return
        predicted_prices = [base_price * np.exp(predictions[k][i]) for k in range(1, max_lookahead + 1)]

        # X-values for predicted points
        x_values = list(range(i, i + max_lookahead + 1))
        y_values = [base_price] + predicted_prices

        # Plot the predicted line from point i
        plt.plot(x_values, y_values, color="blue", alpha=0.3)

    plt.title("Multi-Step Forecasts from Each Time Point")
    plt.xlabel("Time Index")
    plt.ylabel("Mid Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # bro this model is so shit R^2 is like 0.25
    # i think its because its not linearly correlated or smth idk
    # or maybe im just not using enough indicators
    # im using like last 10 steps of log return but those are just the exact same thing
    # maybe i need some other indicators
