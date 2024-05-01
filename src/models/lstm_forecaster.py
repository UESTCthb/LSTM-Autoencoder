import torch
import torch.nn as nn

# LSTM
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out
    
#LSTM-Autoencoder
class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=300, hidden_layer=100, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        # decoder
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm,
                                                 (torch.zeros(1, self.batch_size, self.input_layer),
                                                  torch.zeros(1, self.batch_size, self.input_layer)))
        return decoder_lstm.squeeze()

