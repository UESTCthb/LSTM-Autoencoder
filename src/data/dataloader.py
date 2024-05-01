import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class StockDataset:
    def __init__(
        self, features_data: pd.DataFrame, target_data: pd.DataFrame, window_size: int
    ):
        assert len(features_data) == len(target_data), "SIZE DOES NOT MATCH"
        self.idx = np.array(target_data.index)
        self.features_data = features_data.values
        self.target_data = target_data.values
        self.window_size = window_size

    def __len__(self):
        return len(self.target_data) - self.window_size

    def __getitem__(self, item: int):
        out = dict()
        item += self.window_size
        out["idx"] = torch.tensor(self.idx[item], dtype=torch.long)
        out["features"] = torch.tensor(
            self.features_data[item - self.window_size : item], dtype=torch.float
        )
        out["target"] = torch.tensor(self.target_data[item], dtype=torch.float)
        return out


class CreateDataset:
    def __init__(
        self, data: pd.DataFrame, window_size: int, feat: list, target: list
    ) -> None:
        self.data = data
        self.window_size = window_size
        self.feat = feat
        self.target = target

    def dataset_split(self, train_ratio: int, valid_ratio: int):
        train_size = int(train_ratio * len(self.data))
        valid_size = int(valid_ratio * len(self.data))

        train_data = self.data[:train_size].dropna()
        valid_data = self.data[train_size : train_size + valid_size].dropna()
        test_data = self.data[train_size + valid_size :].dropna()
        return train_data, valid_data, test_data

    def create_dataset(self, tensor_data):
        dataset = StockDataset(
            features_data=tensor_data[self.feat],
            target_data=tensor_data[self.target],
            window_size=self.window_size,
        )
        return dataset

    def create_loader(
        self, dataset: StockDataset, batch_size: int, shuffle: bool = False
    ):
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle
        )
        return dataloader
