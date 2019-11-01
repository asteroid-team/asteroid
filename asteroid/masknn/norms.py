"""
Normalization classes.
@author : Manuel Pariente, Inria-Nancy
"""

import torch
from torch import nn
EPS = 1e-8


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1),
                                 requires_grad=True)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""
    def forward(self, x):
        """
        Args:
            x: [batch, chan, spec_len]
        Returns:
            gLN_x: [batch, chan, spec_len]
        """
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=[1, 2], keepdim=True)
        return self.gamma * (x - mean) / (var + EPS).sqrt() + self.beta


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""
    def forward(self, x):
        """
        Args:
            x: [batch, chan, spec_len]
        Returns:
            chanLN_x: [batch, chan, spec_len]
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + EPS).sqrt() + self.beta


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""
    def forward(self, x):
        """
        Args:
            x: [batch, channels, length]
        Returns:
            cumLN_x: [batch, channels, length]
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan*(spec_len+1),
                           step=chan, dtype=x.dtype).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.gamma * (x - cum_mean) / (cum_var + EPS).sqrt() + self.beta


# Aliases.
gLN = GlobLN
cLN = ChanLN
cgLN = CumLN


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Could not interpret normalization identifier: ' +
                             str(identifier))
        return cls
    else:
        raise ValueError('Could not interpret normalization identifier: ' +
                         str(identifier))
