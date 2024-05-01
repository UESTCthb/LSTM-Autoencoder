import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as stats
import mplfinance as mpf
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

plt.style.use("ggplot")

COLS = ["open", "close", "high", "low", "volume"]

color = {
    "open": "red",
    "close": "blue",
    "high": "green",
    "low": "orange",
    "volume": "purple",
}


def plot_price_volume(data: pd.DataFrame, kind: str = "bid", cols_to_plot: list = COLS):
    fig, axes = plt.subplots(nrows=len(COLS), ncols=1, figsize=(10, 20))
    cols_kind = [col + "_" + kind for col in COLS]
    for idx, col in enumerate(cols_kind):
        axes[idx].plot(data.index, data[col], color=color[cols_to_plot[idx]])
        axes[idx].set_ylabel(col)
    plt.xlabel("open_time")
    plt.tight_layout()
    plt.show()


PRICE_COLS = ["open", "high", "low", "close"]


def plot_candlestick(data, kind="bid", cols_to_plot=PRICE_COLS):
    cols_kind = [col + "_" + kind for col in PRICE_COLS]
    candlestick_data = data[cols_kind]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data[cols_kind[0]],
                high=data[cols_kind[1]],
                low=data[cols_kind[2]],
                close=data[cols_kind[3]],
            )
        ]
    )

    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        dragmode="select",
    )

    init_notebook_mode(connected=True)

    iplot(fig)


class Stationary:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check(self, feature: str, threshold: float = 0.05) -> bool:
        print(f"checking stationarity for {feature}")
        result = adfuller(self.data[feature])
        p_value = result[1]
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {p_value}")
        print("Is stationary:", p_value < threshold)
        return p_value < threshold

    def plot(self, feature: str, window: int = 6) -> None:
        plt.figure(figsize=(22, 10))
        ts = self.data[feature]
        rolmean = ts.rolling(window).apply(np.mean)
        rolstd = ts.rolling(window).apply(np.std)

        plt.plot(ts, color="red", label="Original")
        plt.plot(rolmean, color="black", label="Rolling Mean")
        plt.plot(rolstd, color="green", label="Rolling Std")
        plt.xlabel("Date")
        plt.ylabel(f"Mean {feature}")
        plt.title("Rolling Mean & Standard Deviation")
        plt.legend()
        plt.show()


class RelationCheck:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _plot(self, lag_values, title: str):
        plt.figure(figsize=(22, 10))
        plt.plot(lag_values)
        plt.axhline(y=0, linestyle="--", color="gray")
        plt.axhline(y=-1.96 / np.sqrt(len(self.data)), linestyle="--", color="gray")
        plt.axhline(y=1.96 / np.sqrt(len(self.data)), linestyle="--", color="gray")
        plt.title(title, pad=20)
        plt.show()

    def acf_plot(self, feature: str, nlags: int = 20):
        lag_acf = acf(self.data[feature], nlags=nlags)
        self._plot(lag_acf, title="Autocorrelation Function")
        return lag_acf

    def pacf_plot(self, feature: str, nlags: int = 20, method: str = "ols"):
        lag_pacf = pacf(self.data[feature], nlags=nlags, method=method)
        self._plot(lag_pacf, title="Partial Autocorrelation Function")
        return lag_pacf


class Normality:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check(self, feature: str, threshold: float = 0.05) -> bool:
        _, p_value = stats.normaltest(self.data[feature].dropna())
        if p_value < threshold:
            return False
        return True

    def _get_annotation(self, is_normal: bool) -> str:
        if not is_normal:
            return "Not Normally Distributed"
        return "Normally Distributed"

    def plot(self, feature: str, threshold: float = 0.05) -> str:
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[feature], bins=30, color="purple")
        plt.xlabel("Values")
        plt.ylabel("Frequency")

        normality_status = self.check(feature, threshold)
        normality_status = self._get_annotation(normality_status)

        plt.annotate(
            normality_status,
            xy=(threshold, 1 - threshold),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )
        plt.show()
