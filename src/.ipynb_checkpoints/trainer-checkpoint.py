from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import numpy as np

# whether inject the concept of within the range into loss functions helps the training
class CustomLoss(nn.Module):
    def forward(self, predicted_value, target, open_bid):
        distance = torch.abs(predicted_value - open_bid) + torch.abs(
            predicted_value - target
        )
        loss = torch.mean(distance)
        return loss


def train_and_validate(
    model, optimizer, train_loader, valid_loader, max_wait=5, epochs=10
):
    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf
    patience = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for i, (batch_data, batch_target, batch_open) in enumerate(train_loader):
            predicted_output = model(batch_data)
            loss = loss_fn(predicted_output, batch_target, batch_open)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # valid
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (batch_data, batch_target, batch_open) in enumerate(valid_loader):
                predicted_output = model(batch_data)

                loss = loss_fn(predicted_output, batch_target, batch_open)

                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        print(f"EPOCH = {epoch}")
        print(f"train_loss = {train_loss}")
        print(f"valid_loss = {valid_loss}")

        if valid_loss > min_valid_loss:
            patience += 1
        else:
            min_valid_loss = valid_loss
            patience = 0

        if patience > max_wait:
            print(f"EARLY STOPPING AT EPOCH = {epoch}")
            break

    return train_losses, valid_losses
