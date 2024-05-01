import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization library
import matplotlib.pyplot as plt  # visualization library
import torch
import torch.nn as nn


def evaluate(model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            criterion = nn.MSELoss()
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    average_loss = total_loss / len(dataloader.dataset)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return average_loss, predictions, targets


def plot_prediction(predictions, targets):
    plt.plot(targets, label="Actual")

    plt.plot(predictions, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def directinal_accuracy(test_data):
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


def absolute_errors(test_data):
    absolute_errors = (
        test_data["pred_high_bid", "pred_low_ask"] - test_data["high_bid", "low_ask"]
    ) / (test_data["open_bid", "open_ask"] - test_data["high_bid", "low_ask"])

    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    return mean_absolute_error
