import torch.nn as nn


def adding_bollinger_band_vol(data):
    data = data.copy()
    data["bid_ma5"] = data["high_bid"].rolling(window=6).mean()
    data["bid_ma30"] = data["high_bid"].rolling(window=30).mean()
    data["bid_golden/death"] = (data["bid_ma5"] > data["bid_ma30"]).astype(int)

    data["ask_ma5"] = data["low_bid"].rolling(window=6).mean()
    data["ask_ma30"] = data["low_bid"].rolling(window=30).mean()
    data["ask_golden/death"] = (data["ask_ma5"] > data["ask_ma30"]).astype(int)

    data["bid_standard_deviation"] = data["high_bid"].rolling(window=30).std()

    data["bid_upper_band"] = data["bid_ma30"] + 2 * data["bid_standard_deviation"]
    data["bid_lower_band"] = data["bid_ma30"] - 2 * data["bid_standard_deviation"]

    data["bid_middle_band"] = data["bid_ma30"]

    data["bid_bollinger_touch"] = 0

    touch_ratio = (data["high_bid"] - data["bid_middle_band"]) / (
        data["bid_upper_band"] - data["bid_middle_band"]
    )

    data.loc[data["high_bid"] > data["bid_upper_band"], "bid_bollinger_touch"] = 1
    data.loc[data["high_bid"] < data["bid_lower_band"], "bid_bollinger_touch"] = -1
    data.loc[
        (data["high_bid"] > data["bid_lower_band"])
        & (data["high_bid"] < data["bid_upper_band"]),
        "bid_bollinger_touch",
    ] = touch_ratio

    data["ask_standard_deviation"] = data["low_ask"].rolling(window=30).std()

    data["ask_upper_band"] = data["ask_ma30"] + 2 * data["ask_standard_deviation"]
    data["ask_lower_band"] = data["ask_ma30"] - 2 * data["ask_standard_deviation"]

    data["ask_middle_band"] = data["ask_ma30"]

    data["ask_bollinger_touch"] = 0

    touch_ratio = (data["high_ask"] - data["ask_middle_band"]) / (
        data["ask_upper_band"] - data["ask_middle_band"]
    )

    data.loc[data["low_ask"] > data["ask_upper_band"], "ask_bollinger_touch"] = 1
    data.loc[data["low_ask"] < data["ask_lower_band"], "ask_bollinger_touch"] = -1
    data.loc[
        (data["low_ask"] > data["ask_lower_band"])
        & (data["low_ask"] < data["ask_upper_band"]),
        "ask_bollinger_touch",
    ] = touch_ratio
    return data


def change_to_pct(feat, data):
    for i in range(len(feat)):
        feature = feat[i]
        if (
            "high" in feature
            or "low" in feature
            or "open" in feature
            or "close" in feature
        ):
            data[feature + "_pct"] = data[feature].pct_change() * 100
            feat[i] = feature + "_pct"
    return feat, data
