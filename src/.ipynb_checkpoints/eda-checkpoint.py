import torch.nn as nn
import pywt
import numpy as np
from statsmodels.tsa.stattools import adfuller


def check_stationarity(data):
    result = adfuller(data)
    p_value = result[1]
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {p_value}")
    print("Is stationary:", p_value < 0.05)


class MyDataset:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        window = self.data[index : index + self.window_size, :-2]
        target = self.data[index + self.window_size, 2]
        open = self.data[index + self.window_size, -2]
        # more items
        return window, target, open


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


def dataset_split(data, train_ratio, valid_ratio):
    train_size = int(train_ratio * len(data))
    valid_size = int(valid_ratio * len(data))
    test_size = data - train_size - valid_size

    train_data = data[:train_size].dropna()
    valid_data = data[train_size : train_size + valid_size].dropna()
    test_data = data[train_size + valid_size :].dropna()
    return train_data, valid_data, test_data


def wavelet_smooth_dataset(dataset, feat, wavelet="db4", level=1, window_size=60):
    num_features = dataset.shape[1]
    num_rows = dataset.shape[0]

    for feat_name in feat:
        dataset.loc[:window_size, feat_name] = wavelet_smooth(
            dataset.loc[:window_size, feat_name], wavelet, level
        )

    for j in range(window_size, num_rows):
        window_data = dataset.iloc[j - window_size + 1 : j + 1, :]
        for feat_name in feat:
            window_feature = window_data.loc[:, feat_name]
            dataset.loc[j, feat_name] = wavelet_smooth(window_feature, wavelet, level)[
                -1
            ]

    return dataset


def wavelet_smooth(data, wavelet="db4", level=1):
    coeff = pywt.wavedec(data, wavelet, level=level)
    for i in range(1, len(coeff)):
        detail_coeff = coeff[i]
        threshold_value = np.std(detail_coeff) * 0.5  # threshold multiply

        for j in range(len(detail_coeff)):
            magnitude = np.abs(detail_coeff[j])

            # Check if magnitude is zero before division
            if magnitude != 0:
                detail_coeff[j] = (1 - detail_coeff[j] / magnitude) * threshold_value
            else:
                detail_coeff[j] = 0

        coeff[i] = detail_coeff

    smoothed_data = pywt.waverec(coeff, wavelet)
    return smoothed_data


class LinearAutoencoder(nn.Module):  # honrizontal
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, 1, batch_first=True)
        self.latent_dim = latent_dim

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

    def get_latent_features(self, x):
        _, (h_n, _) = self.encoder(x)
        return h_n[-1, :, : self.latent_dim]
