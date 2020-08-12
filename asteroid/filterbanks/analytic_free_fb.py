import torch
import torch.nn as nn
import numpy as np
from .enc_dec import Filterbank


class AnalyticFreeFB(Filterbank):
    """ Free analytic (fully learned with analycity constraints) filterbank.
    For more details, see [1].

    Args:
        n_filters (int): Number of filters. Half of `n_filters` will
            have parameters, the other half will be the hilbert transforms.
            `n_filters` should be even.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.

    Attributes:
        n_feats_out (int): Number of output filters.

    References:
        [1] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, stride=None, **kwargs):
        super(AnalyticFreeFB, self).__init__(n_filters, kernel_size, stride=stride)
        self.cutoff = int(n_filters // 2)
        self.n_feats_out = 2 * self.cutoff
        if n_filters % 2 != 0:
            print(
                "If the number of filters `n_filters` is odd, the "
                "output size of the layer will be `n_filters - 1`."
            )

        self._filters = nn.Parameter(torch.ones(n_filters // 2, 1, kernel_size), requires_grad=True)
        for p in self.parameters():
            nn.init.xavier_normal_(p, gain=1.0 / np.sqrt(2.0))

    @property
    def filters(self):
        ft_f = torch.rfft(self._filters, 1, normalized=True)
        hft_f = torch.stack([ft_f[:, :, :, 1], -ft_f[:, :, :, 0]], dim=-1)
        hft_f = torch.irfft(hft_f, 1, normalized=True, signal_sizes=(self.kernel_size,))
        return torch.cat([self._filters, hft_f], dim=0)
