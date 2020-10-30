import torch
import torch.nn as nn
from .enc_dec import Filterbank


class FreeFB(Filterbank):
    """Free filterbank without any constraints. Equivalent to
    :class:`nn.Conv1d`.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.

    Attributes:
        n_feats_out (int): Number of output filters.

    References:
        [1] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, stride=None, sample_rate=8000.0, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def filters(self):
        return self._filters
