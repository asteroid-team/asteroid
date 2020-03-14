import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

EPS = 1e-8


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""
    def forward(self, x):
        """ Applies forward pass.
        
        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""
    def forward(self, x):
        """ Applies forward pass.
        
        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, *]`
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""
    def forward(self, x):
        """

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             :class:`torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan*(spec_len+1),
                           step=chan, dtype=x.dtype).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""
    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError('expected 4D or 3D input (got {}D input)'
                             .format(input.dim()))

# Aliases.
gLN = GlobLN
cLN = ChanLN
cgLN = CumLN
bN = BatchNorm


def get(identifier):
    """ Returns a norm class from a string. Returns its input if it
    is callable (already a :class:`._LayerNorm` for example).

    Args:
        identifier (str or Callable or None): the norm identifier.

    Returns:
        :class:`._LayerNorm` or None
    """
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
