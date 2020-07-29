import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils
from asteroid.filterbanks import make_enc_dec
from asteroid.filterbanks.transforms import take_mag, apply_mag_mask


class Generator(nn.Module):
    """ Generator also mentioned as G"""

    def __init__(self, encoder, decoder, negative_slope=0.3):
        super().__init__()
        self.encoder = encoder
        self.LSTM = nn.LSTM(input_size=257, hidden_size=200, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(400, 300),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.05),
            nn.Linear(300, 257),
            nn.Sigmoid(),
        )
        self.decoder = decoder

    def forward(self, x):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
        """

        # Encode
        spec = self.encoder(x)
        mag = take_mag(spec)
        # x = nn.utils.spectral_norm(x)
        mag = torch.transpose(mag, 1, 2)
        # Compute mask
        self.LSTM.flatten_parameters()
        mask, _ = self.LSTM(mag)
        mask = self.model(mask)
        mask = torch.transpose(mask, 1, 2)
        y = apply_mag_mask(spec, mask)
        # Decode
        y = self.decoder(y)
        return torch_utils.pad_x_to_y(y, x)


class GeneratorLoss(_Loss):

    def __init__(self, s=1):
        super().__init__()
        self.s = s

    def forward(self, est_labels):
        loss = torch.mean((est_labels - self.s) ** 2)
        return loss


def make_generator_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    encoder, decoder = make_enc_dec(**conf['filterbank'])
    model = Generator(encoder, decoder)
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    g_loss = GeneratorLoss(conf['g_loss']['s'])
    return model, optimizer, g_loss
