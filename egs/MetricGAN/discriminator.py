import torch
import torch.nn as nn
from asteroid.engine.optimizers import make_optimizer
from torch.nn.modules.loss import _Loss
from asteroid.metrics import get_metrics
from asteroid.filterbanks import make_enc_dec
from asteroid.filterbanks.transforms import take_mag


class Discriminator(nn.Module):
    """D"""

    def __init__(self, encoder, decoder, negative_slope=0.3):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.conv = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.utils.spectral_norm(nn.Conv2d(2, 15, 5, 1)),
            nn.LeakyReLU(negative_slope),
            nn.utils.spectral_norm(nn.Conv2d(15, 25, 7, 1)),
            nn.LeakyReLU(negative_slope),
            nn.utils.spectral_norm(nn.Conv2d(25, 40, 9, 1)),
            nn.LeakyReLU(negative_slope),
            nn.utils.spectral_norm(nn.Conv2d(40, 50, 11, 1)),
            nn.LeakyReLU(negative_slope))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(50, 50)),
            nn.LeakyReLU(negative_slope),
            nn.utils.spectral_norm(nn.Linear(50, 10)),
            nn.LeakyReLU(negative_slope),
            nn.utils.spectral_norm(nn.Linear(10, 1)),
            )

    def forward(self, x, z, y):
        """
        Forward pass of discriminator.
        Args:
            x: inputs
            y: noisy
            z: clean
        """
        # Encode
        x = self.encoder(x)
        x = take_mag(x)
        x = x.unsqueeze(1)

        # Encode
        y = self.encoder(y)
        y = take_mag(y)
        y = y.unsqueeze(1)

        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        x = self.pool(x).squeeze()
        x = self.linear(x)
        return x


class DiscriminatorLoss(_Loss):

    # Least Square loss function

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def forward(self, noisy, clean, estimates, est_labels, labels):
        # Behaves differently if estimates come from  the generated data or not
        #
        if labels:
            loss = torch.mean((est_labels - torch.ones_like(est_labels)) ** 2)
        else:
            loss = torch.mean((est_labels - get_metric(self.metric, noisy,
                                                       clean, estimates))**2)
        return loss


def get_metric(metric, noisy, clean, estimates):
    noisy_np = noisy.cpu().data.numpy()
    clean_np = clean.cpu().data.numpy()
    estimates_np = estimates.cpu().data.numpy()
    metrics = []
    for i in range(noisy_np.shape[0]):
        metrics.append(get_metrics(noisy_np[i], clean_np[i], estimates_np[i],
                                   metrics_list=[metric])[metric])
    metrics = torch.Tensor(metrics)
    if metric == 'pesq':
        metrics = (metrics + 0.5)/5.0
    return metrics.to(noisy.device)


def make_discriminator_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.

    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    encoder, decoder = make_enc_dec(**conf['filterbank'])
    model = Discriminator(encoder, decoder)
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    d_loss = DiscriminatorLoss(conf['metric_to_opt']['metric'])
    return model, optimizer, d_loss
