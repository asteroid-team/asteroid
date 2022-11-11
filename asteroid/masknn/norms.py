from functools import partial
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing import List

from .. import complex_nn
from ..utils.torch_utils import script_if_tracing

EPS = 1e-8


def z_norm(x, dims: List[int], eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value


@script_if_tracing
def _glob_norm(x, eps: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


@script_if_tracing
def _feat_glob_norm(x, eps: float = 1e-8):
    dims: List[int] = torch.arange(2, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x, EPS: float = 1e-8):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""

    def forward(self, x, EPS: float = 1e-8):
        """Applies forward pass.

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

    def forward(self, x, EPS: float = 1e-8):
        """

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             :class:`torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(
            start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class FeatsGlobLN(_LayerNorm):
    """Feature-wise global Layer Normalization (FeatsGlobLN).
    Applies normalization over frames for each channel."""

    def forward(self, x, EPS: float = 1e-8):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, time]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, time]`
        """
        value = _feat_glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError("expected 4D or 3D input (got {}D input)".format(input.dim()))


# Aliases.
gLN = GlobLN
fgLN = FeatsGlobLN
cLN = ChanLN
cgLN = CumLN
bN = BatchNorm


def register_norm(custom_norm):
    """Register a custom norm, gettable with `norms.get`.

    Args:
        custom_norm: Custom norm to register.

    """
    if custom_norm.__name__ in globals().keys() or custom_norm.__name__.lower() in globals().keys():
        raise ValueError(f"Norm {custom_norm.__name__} already exists. Choose another name.")
    globals().update({custom_norm.__name__: custom_norm})


def get(identifier):
    """Returns a norm class from a string. Returns its input if it
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
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))


def get_complex(identifier):
    """Like `.get` but returns a complex norm created with `asteroid.complex_nn.OnReIm`."""
    norm = get(identifier)
    if norm is None:
        return None
    else:
        return partial(complex_nn.OnReIm, norm)
