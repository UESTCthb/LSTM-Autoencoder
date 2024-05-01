from dataclasses import dataclass, field
from functools import partial
from typing import List

import pandas as pd


from src.data.feature_engineering import Transformer


def _post_processing(x, side: str):
    forecast_value = x[0]
    open_value = x[1]

    if side == "high":
        if forecast_value < open_value:
            return open_value
        return forecast_value

    if forecast_value > open_value:
        return open_value
    return forecast_value


@dataclass
class ForecastsAggergator(Transformer):
    instrument: str = "bid_ask"
    timestep: int = 1
    sides: list = field(default_factory=lambda: ["high", "low"])
    quantiles: List[float] = None
    boundary_level: List[float] = None
    add_pips_diff: bool = True
    add_pct: bool = True
    drop_original_cols: bool = True
    use_close_price: bool = True
    post_processing: bool = False
    online: bool = False

    def _add_quantiles(self, dataset):
        models_cols = [col for col in dataset.columns if "." in col]
        if self.boundary_level is not None:
            models_cols = [col for col in models_cols if float(col.split("_")[-1]) in self.boundary_level]

        for side in self.sides:
            side_cols = [col for col in models_cols if side in col]
            for quantile in self.quantiles:
                dataset[f"{side}_q{int(100*quantile)}"] = dataset[side_cols].quantile(quantile, axis=1)
        if self.post_processing:
            dataset = self._post_processing(dataset)
        return dataset

    
    def _add_percentage_of_change(self, dataset: pd.DataFrame):
        for side in self.sides:
            kind = "bid" if side == "high" else "ask"
            opposite_side = "bid" if side == "low" else "ask"
            sign = 1 if kind == "bid" else -1

            col = f"open_{opposite_side}" if self.instrument == "bid_ask" else "open"
            open_prices = dataset[col]
            if self.use_close_price:
                col = f"close_{opposite_side}" if self.instrument == "bid_ask" else "close"
                open_prices = dataset[col].shift(self.timestep)

            for quantile in self.quantiles:
                dataset[f"{side}_q{int(100*quantile)}_pct"] = (
                    sign * (dataset[f"{side}_q{int(100*quantile)}"] - open_prices) / open_prices
                ) * 100
            if not self.online:
                col = side + "_" + kind if self.instrument == "bid_ask" else side
                dataset[f"{side}_pct"] = (sign * (dataset[col] - open_prices) / open_prices) * 100
        return dataset

    def _post_processing(self, dataset: pd.DataFrame):
        for side in self.sides:
            func = partial(_post_processing, side=side)
            open_col = "open"
            if self.instrument == "bid_ask":
                open_col = "open_bid" if side == "high" else "open_ask"
            for quantile in self.quantiles:
                col = f"{side}_q{int(100*quantile)}"
                if side == "high":
                    valid_rows = dataset[col] >= dataset[open_col]
                else:
                    valid_rows = dataset[open_col] >= dataset[col]
                dataset[col] = dataset[col] * valid_rows + dataset[open_col] * (1 - valid_rows)
        return dataset

    def transform(self, dataset: pd.DataFrame):
        dataset = self._add_quantiles(dataset.copy())
        
        if self.add_pct:
            dataset = self._add_percentage_of_change(dataset)
        if self.drop_original_cols:
            cols = [col for col in dataset.columns if "." in col]
            dataset.drop(labels=cols, axis=1, inplace=True)
        # dataset = filter_data_based_on_schedluer(dataset, scheduler=self.instrument.symbol_metadata.scheduler)
        return dataset


@dataclass
class ForecastMetricsAdder(Transformer):
    """
    ForecastMetricsCalculator class:
    Generates metrics for forecast performance based on forecasts and market data
    """

    instrument: str = "bid_ask"
    timestep: int = 1
    sides: list = field(default_factory=lambda: ["high", "low"])
    quantiles: List[float] = None
    add_pips_error: bool = True
    add_pct_error: bool = True
    accuracy_sensitivity: float = 0.01

    def _get_pred_col(self, side: str, quantile: float):
        quantile = "q" + str(int(100 * quantile))
        return f"{side}_{quantile}"

    def _get_actual_col(self, side: str):
        if self.instrument != "bid_ask":
            return side
        kind = "bid" if side == "high" else "ask"
        return f"{side}_{kind}"
    
    def _add_metrics(self, dataset: pd.DataFrame):
        metric_data = dataset[["open_time"]].copy()
        for side in self.sides:
            actual_col = self._get_actual_col(side)
            sign = 1 if side == "high" else -1
            for quantile in self.quantiles:
                pred_col = self._get_pred_col(side, quantile)

                metric_data[f"{pred_col}_error"] = sign * (dataset[pred_col] - dataset[actual_col])
                metric_data[f"{pred_col}_abs_error"] = abs(metric_data[f"{pred_col}_error"])
                metric_data[f"{pred_col}_pct_error"] = dataset[pred_col + "_pct"] - dataset[side + "_pct"]
                metric_data[f"{pred_col}_abs_pct_error"] = abs(metric_data[f"{pred_col}_pct_error"])

                metric_data[f"{pred_col}_accuracy"] = (
                    self.accuracy_sensitivity >= metric_data[f"{pred_col}_pct_error"]
                ).astype(int)

                
        return metric_data

    def transform(self, dataset: pd.DataFrame):
        return self._add_metrics(dataset)