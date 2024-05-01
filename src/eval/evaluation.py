import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization library
import matplotlib.pyplot as plt  # visualization library
import torch
import torch.nn as nn


class Evaluation:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def evaluate(self, loss_fn):  # Do evaluation without plot
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(device)
                target = target.to(device)

                output = self.model(data)
                criterion = loss_fn
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)

                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        average_loss = total_loss / len(self.dataloader.dataset)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return average_loss, predictions, targets

    def plot_prediction(self, loss_fn):  # Do evaluation and plot
        average_loss, predictions, targets = self.evaluate(loss_fn)
        plt.plot(targets, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def accuracy(self, test_data):
        high_bid_within_range = (
            (test_data["high_bid"] >= test_data["pred_high_bid"])
            & (test_data["pred_high_bid"] >= test_data["open_bid"])
        ).float()
        low_ask_within_range = (
            (test_data["low_risk"] <= test_data["pred_low_ask"])
            & (test_data["pred_low_ask"] <= test_data["open_ask"])
        ).float()

        high_bid_accuracy = high_bid_within_range / len(test_data)
        low_ask_accuracy = low_ask_within_range / len(test_data)
        return high_bid_accuracy, low_ask_accuracy

    def absolute_errors(self, true_data, pre_data):
        absolute_errors = abs(true_data - pre_data) / true_data
        mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
        return mean_absolute_error


def merge_predictions_with_original(predictions, orginal_test_data):
    
    predictions = predictions.rename(columns={"high_bid": "predicted.high_bid", "low_ask": "predicted.low_ask"})
    
    
    selected_cols_prediction = predictions[["predicted.high_bid", "predicted.low_ask"]]
    selected_cols_original_test = orginal_test_data[["high_bid", "low_ask", "open_bid", "open_ask", "close_bid", "close_ask"]]
    
    
    merged_data = pd.concat([selected_cols_prediction, selected_cols_original_test], axis=1)
    
    return merged_data

