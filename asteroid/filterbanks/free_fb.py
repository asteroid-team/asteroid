"""
Free filterbank.
@author : Manuel Pariente, Inria-Nancy
"""

import torch
import torch.nn as nn
from .enc_dec import EncoderDecoder


class FreeFB(EncoderDecoder):
    """Free filterbank without any constraints. Equivalent to nn.Conv1d.

    # Args
        n_filters: Positive int. Number of filters.
        kernel_size: Positive int. Length of the filters.
        stride: Positive int. Stride of the convolution.
        enc_or_dec: String. `enc` or `dec`. Controls if filterbank is used as
            an encoder or a decoder.

    # References
        [1] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent.
    """
    def __init__(self, n_filters, kernel_size, stride, enc_or_dec='encoder',
                 **kwargs):
        super(FreeFB, self).__init__(stride, enc_or_dec=enc_or_dec)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    @property
    def filters(self):
        return self._filters

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_or_dec': self.enc_or_dec
        }
        return config
