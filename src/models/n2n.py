import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(
    x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True
):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size))
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size))


class AttnEncoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.window_size
        self.add_noise = True  # config.denoising
        self.directions = 1  # config.directions
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len, out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.
        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.
        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (
            init_hidden(input_data, self.hidden_size, num_dir=self.directions).to(
                input_data.device
            ),
            init_hidden(input_data, self.hidden_size, num_dir=self.directions).to(
                input_data.device
            ),
        )

        attentions, input_encoded = (
            Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)).to(
                input_data.device
            ),
            Variable(
                torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)
            ).to(input_data.device),
        )

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(input_data.device)

        for t in range(self.seq_len):
            x = torch.cat(
                (
                    h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    input_data.permute(0, 2, 1),
                ),
                dim=2,
            )

            e_t = self.attn(
                x.view(-1, self.hidden_size * 2 + self.seq_len)
            )  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size))

            weighted_input = torch.mul(a_t, input_data[:, t, :])
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


class AttnDecoder(nn.Module):
    def __init__(self, config, output_size):
        """
        Initialize the network.
        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = config.window_size
        self.encoder_hidden_size = config.hidden_size
        self.decoder_hidden_size = config.hidden_size
        self.out_feats = output_size

        self.attn = nn.Sequential(
            nn.Linear(
                2 * self.decoder_hidden_size + self.encoder_hidden_size,
                self.encoder_hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1),
        )
        self.lstm = nn.LSTM(
            input_size=self.out_feats, hidden_size=self.decoder_hidden_size
        )
        self.fc = nn.Linear(self.encoder_hidden_size, self.out_feats)
        self.fc_out = nn.Linear(
            self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats
        )
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor):
        """
        Perform forward computation.
        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
        """
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size).to(
                input_encoded.device
            ),
            init_hidden(input_encoded, self.decoder_hidden_size).to(
                input_encoded.device
            ),
        )
        context = Variable(
            torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(
                input_encoded.device
            )
        )

        y = []
        for t in range(self.seq_len):
            x = torch.cat(
                (
                    h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                    c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                    input_encoded,
                ),
                dim=2,
            )
            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1,
            )
            context = torch.bmm(x.unsqueeze(1), input_encoded)

            y_tilde = self.fc(context)  # (batch_size, out_size)
            y.append(y_tilde)
            # self.lstm.flatten_parameters()
            # _, (h_t, c_t) = self.lstm(y_tilde, (h_t, c_t))

        return torch.cat(
            y, dim=1
        )  # self.fc_out(torch.cat((h_t[0], context), dim=1))  # predicting value at t=self.seq_length+1


class TimeSeriesAutoEnc(nn.Module):
    def __init__(self, config, input_size):
        """
        Initialize the network.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(TimeSeriesAutoEnc, self).__init__()
        self.encoder = AttnEncoder(config, input_size)
        self.decoder = AttnDecoder(config, input_size)

    def forward(self, encoder_input):
        """
        Forward computation. encoder_input_inputs.
        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether or not to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output)
        return outputs
