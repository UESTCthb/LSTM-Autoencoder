from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        EPOCHS,
        MAX_WAIT,
        device,
        loss_fn,
        scheduler,
        
    ):
        self.model = model
        self.model.to(device)

        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.EPOCHS = EPOCHS
        self.MAX_WAIT = MAX_WAIT
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train_step(self):
        self.model.train()
        total_loss = 0
        for data in tqdm(self.train_dataloader):
            x = data["features"].to(self.device)
            y = data["target"].to(self.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)

    def valid_step(self):
        self.model.eval()
        total_loss = 0
        for data in tqdm(self.valid_dataloader):
            x = data["features"].to(self.device)
            y = data["target"].to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
            total_loss += loss.item()
        return total_loss / len(self.valid_dataloader)

    def fit(self):
        train_losses = []
        valid_losses = []
        min_valid_loss = np.inf
        patience = 0
        for epoch in range(self.EPOCHS):
            train_loss = self.train_step()
            train_losses.append(train_loss)

            valid_loss = self.valid_step()
            valid_losses.append(valid_loss)
            print(f"EPOCH = {epoch}")
            print(f"train_loss = {train_loss}")
            print(f"valid_loss = {valid_loss}")
            if valid_loss > min_valid_loss:
                patience += 1
            else:
                min_valid_loss = valid_loss
                patience = 0

            self.scheduler.step()

            if patience > self.MAX_WAIT:
                print(f"EARLY STOPPING AT EPOCH = {epoch}")
                break
        return train_losses, valid_losses

    def predict(self, dataloader):
        indexes = []
        high_bid_predictions = []
        low_ask_predictions = []
        with torch.no_grad():
            for data in dataloader:
                batch_X = data["features"].to(self.device)

                preds = self.model(batch_X)
                high_bid_predictions.append(preds[:, 0].cpu().numpy())
                low_ask_predictions.append(preds[:, 1].cpu().numpy())
                indexes.append(data["idx"].numpy())
        indexes = np.concatenate(indexes)
        high_bid_predictions = np.concatenate(high_bid_predictions)
        low_ask_predictions = np.concatenate(low_ask_predictions)
        return indexes, high_bid_predictions, low_ask_predictions
