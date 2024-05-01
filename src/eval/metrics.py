import matplotlib.pyplot as plt
import pandas as pd


class ForecastEval:
    def __init__(self, orginal_data: pd.DataFrame, predicted_data: pd.DataFrame):
        self.orginal_data = orginal_data
        self.predicted_data = predicted_data
        self.eps = 1e-4

    def evaluate_position_accuracy(self):
        high_bid_between = (
            self.predicted_data["high_bid"] >= self.orginal_data["open_bid"]
        ) & (self.predicted_data["high_bid"] <= self.orginal_data["high_bid"])
        low_ask_between = (
            self.predicted_data["low_ask"] <= self.orginal_data["open_ask"]
        ) & (self.orginal_data["low_ask"] <= self.predicted_data["low_ask"])
        both_in_middle = high_bid_between & low_ask_between
        accuracy = both_in_middle.mean() * 100.0
        print("Position Accuracy: {:.2f}%".format(accuracy))
        return accuracy

    def get_absolute_error(self):
        bid_absolute_error = 100 * abs(
            (self.predicted_data["high_bid"] - self.orginal_data["high_bid"])
            / (
                (self.orginal_data["open_bid"] - self.orginal_data["high_bid"])
                + self.eps
            )
        )

        ask_absolute_error = 100 * (
            abs(self.predicted_data["low_ask"] - self.orginal_data["low_ask"])
            / (
                (self.orginal_data["open_ask"] - self.orginal_data["low_ask"])
                + self.eps
            )
        )
        return (bid_absolute_error + ask_absolute_error) / 2

    def plot_absolute_error(self):
        absolute_error = self.get_absolute_error()
        print("ABS error: ", absolute_error.mean())
        plt.figure(figsize=(10, 6))
        plt.plot(
            absolute_error.index, absolute_error, marker="o", linestyle="-", color="b"
        )
        plt.xlabel("Data Points")
        plt.ylabel("Absolute Error (%)")
        plt.title("Absolute Error at Each Data Point")
        plt.grid(True)
        plt.show()

    def run_evaluation(self):
        self.evaluate_position_accuracy()
        self.plot_absolute_error()
