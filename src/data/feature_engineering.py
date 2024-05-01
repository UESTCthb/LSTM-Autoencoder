from abc import ABC, abstractstaticmethod
import pywt
import numpy as np
import pandas as pd


class Transformer(ABC):
    @abstractstaticmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """transform data"""


class BollingerBandAdder(Transformer):
    def __init__(
        self,
        small_window: int = 5,
        large_window: int = 30,
        kind: str = "bid",
        dropna: bool = False,
    ):
        self.small_window = small_window
        self.large_window = large_window
        self.kind = kind
        self.dropna = dropna

    def transform(self, data: pd.DataFrame):
        data = data.copy()

        data[f"{self.kind}_ma{self.small_window}"] = (
            data[f"high_{self.kind}"].rolling(window=self.small_window).mean()
        )
        data[f"{self.kind}_ma{self.large_window}"] = (
            data[f"high_{self.kind}"].rolling(window=self.large_window).mean()
        )

        data[f"{self.kind}_golden/death"] = (
            data[f"{self.kind}_ma{self.small_window}"]
            > data[f"{self.kind}_ma{self.large_window}"]
        ).astype(int)
        data[f"{self.kind}_standard_deviation"] = (
            data[f"high_{self.kind}"].rolling(window=self.large_window).std()
        )
        data[f"{self.kind}_upper_band"] = (
            data[f"{self.kind}_ma{self.large_window}"]
            + 2 * data[f"{self.kind}_standard_deviation"]
        )
        data[f"{self.kind}_lower_band"] = (
            data[f"{self.kind}_ma{self.large_window}"]
            - 2 * data[f"{self.kind}_standard_deviation"]
        )
        data[f"{self.kind}_middle_band"] = data[f"{self.kind}_ma{self.large_window}"]
        data[f"{self.kind}_bollinger_touch"] = 0

        touch_ratio = (data[f"high_{self.kind}"] - data[f"{self.kind}_middle_band"]) / (
            data[f"{self.kind}_upper_band"] - data[f"{self.kind}_middle_band"]
        )
        data.loc[
            data[f"high_{self.kind}"] > data[f"{self.kind}_upper_band"],
            f"{self.kind}_bollinger_touch",
        ] = 1
        data.loc[
            data[f"high_{self.kind}"] < data[f"{self.kind}_lower_band"],
            f"{self.kind}_bollinger_touch",
        ] = -1
        data.loc[
            (data[f"high_{self.kind}"] > data[f"{self.kind}_lower_band"])
            & (data[f"high_{self.kind}"] < data[f"{self.kind}_upper_band"]),
            f"{self.kind}_bollinger_touch",
        ] = touch_ratio

        data.drop(
            [
                f"{self.kind}_ma{self.small_window}",
                f"{self.kind}_ma{self.large_window}",
            ],
            axis=1,
            inplace=True,
        )
        if self.dropna:
            data.dropna(inplace=True)
        return data


class PercentageFeatureAdder(Transformer):
    """
    Adds features that were determined as baseline. All features are
    calculated in a percentage-of-change manner. Features are calculated
    with Open_Time in mind, so data may be upsampled prior to this transformer.

    Params:
        rows_to_skip - the number of rows that represent time differences between observations in minutes
        remove_original - whether to remove rows incoming into this transformer

    @creator: Donatas Tamosauskas
    """

    def __init__(
        self,
        rows_to_skip: int,
        remove_original: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rows_to_skip = rows_to_skip
        self.remove_original = remove_original
        self.normalize = normalize

    def transform(self, dataset):
        dataset = dataset.copy()
        instruments = [
            "bid",
            "ask",
        ]  # set(col.split("_")[1] for col in dataset.columns)
        features = []

        for instr in instruments:
            hhpc_1, llpc_1 = self._generate_change_feat(dataset, instr, skip=1)
            hhpc_3, llpc_3 = self._generate_change_feat(dataset, instr, skip=3)

            hplpcp_1 = self._generate_volatility_feat(dataset, instr, skip=1)
            hplpcp_3 = self._generate_volatility_feat(dataset, instr, skip=3)
            hplpcp_4 = self._generate_volatility_feat(dataset, instr, skip=4)

            vvpavg3, vvp3avg3 = self._generate_volume_feat(dataset, instr)
            hc, ho, co = self._generate_momentum_feat(dataset, instr)
            features += [
                hhpc_1,
                llpc_1,
                hhpc_3,
                llpc_3,
                hplpcp_1,
                hplpcp_3,
                hplpcp_4,
                vvpavg3,
                vvp3avg3,
                hc,
                ho,
                co,
            ]

        try:
            if self.normalize:
                features = [100 * x for x in features]
        except attributeerror:
            print(f"version mismatch, update the pipeline!")
            features = [100 * x for x in features]

        original = [] if self.remove_original else [dataset]
        dataset = pd.concat(original + features, axis=1)
        dataset = dataset.dropna()
        return dataset

    def _generate_momentum_feat(self, data, instrument):
        hc = (data[f"high_{instrument}"] - data[f"close_{instrument}"]) / data[
            f"high_{instrument}"
        ]
        hc.name = f"hc_{instrument}"

        co = (data[f"close_{instrument}"] - data[f"open_{instrument}"]) / data[
            f"open_{instrument}"
        ]
        co.name = f"co_{instrument}"

        ho = (data[f"high_{instrument}"] - data[f"open_{instrument}"]) / data[
            f"high_{instrument}"
        ]
        ho.name = f"ho_{instrument}"
        return hc, ho, co

    def _generate_change_feat(self, data, instrument, skip=1):
        prev_high = data[f"high_{instrument}"].shift(skip * self.rows_to_skip)
        prev_low = data[f"low_{instrument}"].shift(skip * self.rows_to_skip)

        hhpc = (data[f"high_{instrument}"] - prev_high) / data[f"close_{instrument}"]
        hhpc.name = f"hhpc_{instrument}_{skip}"
        llpc = (data[f"low_{instrument}"] - prev_low) / data[f"close_{instrument}"]
        llpc.name = f"llpc_{instrument}_{skip}"

        return hhpc, llpc

    def _generate_volatility_feat(self, data, instrument, skip=1):
        prev_high = data[f"high_{instrument}"].shift(skip * self.rows_to_skip)
        prev_low = data[f"low_{instrument}"].shift(skip * self.rows_to_skip)

        prev_close = data[f"close_{instrument}"].shift(1 * self.rows_to_skip)

        hplpcp = (prev_high - prev_low) / prev_close
        hplpcp.name = f"hplpcp_{instrument}_{skip}"
        return hplpcp

    def _generate_volume_feat(self, data, instrument):
        vol_prev_1 = data[f"volume_{instrument}"].shift(1 * self.rows_to_skip)
        vol_prev_2 = data[f"volume_{instrument}"].shift(2 * self.rows_to_skip)
        vol_prev_3 = data[f"volume_{instrument}"].shift(3 * self.rows_to_skip)

        vol_mean = (data[f"volume_{instrument}"] + vol_prev_1 + vol_prev_2) / 3

        vvpavg3 = (data[f"volume_{instrument}"] - vol_prev_1) / vol_mean
        vvpavg3.name = f"vvpavg3_{instrument}"
        vvp3avg3 = (data[f"volume_{instrument}"] - vol_prev_3) / vol_mean
        vvp3avg3.name = f"vvp3avg3_{instrument}"

        return vvpavg3, vvp3avg3


class TargetAdder(Transformer):
    """
    Creates the target to be predicted by the model

    The transformation is for statistical purposes: Ensure Stationarity and Normality of the target.

    Methods:
        transform_dataframe(dataset) applies a predefind formula to generate the target columns
    """

    def __init__(self, step_number, look_back_steps, data_type="bid_ask"):
        self.step_number = step_number
        self.look_back_steps = look_back_steps
        self.data_type = data_type

        self.open_high_col = "open_ask" if self.data_type == "bid_ask" else "open"
        self.open_low_col = "open_bid" if self.data_type == "bid_ask" else "open"
        self.high_col = "high_bid" if self.data_type == "bid_ask" else "high"
        self.low_col = "low_ask" if self.data_type == "bid_ask" else "low"

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[f"{self.high_col}_change_rate_target"] = (
            100
            * (
                dataset[self.high_col]
                - dataset[self.open_high_col].shift(
                    periods=self.look_back_steps * self.step_number
                )
            )
            / dataset[self.open_high_col].shift(
                periods=self.look_back_steps * self.step_number
            )
        )

        dataset[f"{self.low_col}_change_rate_target"] = (
            100
            * (
                dataset[self.open_low_col].shift(
                    periods=self.look_back_steps * self.step_number
                )
                - dataset[self.low_col]
            )
            / dataset[self.open_low_col].shift(
                periods=self.look_back_steps * self.step_number
            )
        )

        dataset[f"last_{self.open_high_col}"] = dataset[self.open_high_col].shift(
            periods=self.look_back_steps * self.step_number
        )
        dataset[f"last_{self.open_low_col}"] = dataset[self.open_low_col].shift(
            periods=self.look_back_steps * self.step_number
        )
        return dataset

    def retransform(self, dataset: pd.DataFrame) -> pd.DataFrame:

        dataset[f"{self.high_col}"] = dataset[f"last_{self.open_high_col}"] + (
            dataset[f"{self.high_col}_change_rate_target"]
            * dataset[f"last_{self.open_high_col}"]
            / 100
        )

        dataset[f"{self.low_col}"] = dataset[f"last_{self.open_low_col}"] - (
            dataset[f"{self.low_col}_change_rate_target"]
            * dataset[f"last_{self.open_low_col}"]
            / 100
        )

        return dataset


class Wavelet:
    def __init__(self, window_size: int, level: int, wavelet_type="db4"):
        self.window_size = window_size
        self.level = level
        self.wavelet_type = wavelet_type

    def wavelet_smooth(self, data):
        coeff = pywt.wavedec(data, self.wavelet_type, self.level)
        for i in range(1, len(coeff)):
            detail_coeff = coeff[i]
            threshold_value = np.std(detail_coeff) * 0.5  # threshold multiply

            for j in range(len(detail_coeff)):
                magnitude = np.abs(detail_coeff[j])

                if magnitude != 0:
                    detail_coeff[j] = (
                        1 - detail_coeff[j] / magnitude
                    ) * threshold_value
                else:
                    detail_coeff[j] = 0

            coeff[i] = detail_coeff
        smoothed_data = pywt.waverec(coeff, self.wavelet_type)
        return smoothed_data[-1]

    def transform(self, data: pd.DataFrame):
        smoothed_data = data.copy()

        for column in data.columns:
            smoothed_data[column] = (
                data[column].rolling(self.window_size).apply(self.wavelet_smooth)
            )

        return smoothed_data


class Scalar:
    def __init__(self, scalar_type="minmax"):
        self.scalar_type = scalar_type

        if self.scalar_type not in ["minmax", "standard", "robust"]:
            raise ValueError(
                "Invalid scalar type. Supported scalar types are 'minmax', 'standard', and 'robust'."
            )

    def fit(self, data):

        if self.scalar_type == "minmax":
            self.min_val = np.min(data)
            self.max_val = np.max(data)

        elif self.scalar_type == "standard":
            self.mean = np.mean(data)
            self.std = np.std(data)

        elif self.scalar_type == "robust":
            self.median = np.median(data)
            self.q1 = np.percentile(data, 25)
            self.q3 = np.percentile(data, 75)

    def transform(self, data):
        if self.scalar_type == "minmax":
            return (data - self.min_val) / (self.max_val - self.min_val)
        elif self.scalar_type == "standard":
            return (data - self.mean) / self.std
        elif self.scalar_type == "robust":
            return (data - self.median) / (self.q3 - self.q1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def retransform(self, transformed_data):
        if self.scalar_type == "minmax":
            return transformed_data * (self.max_val - self.min_val) + self.min_val
        elif self.scalar_type == "standard":
            return transformed_data * self.std + self.mean
        elif self.scalar_type == "robust":
            return transformed_data * (self.q3 - self.q1) + self.median
