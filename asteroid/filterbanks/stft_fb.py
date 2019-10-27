"""
Free filterbank.
@author : Manuel Pariente, Inria-Nancy
"""
import torch
import numpy as np
from .enc_dec import EncoderDecoder

"""
Notes:
- From DC to Nyquist, the number of filters is `n_filters / 2 + 1`. In the 
current implementation, the output shape is not consistent with the other 
filterbanks. There are several solutions to this:
    - Change `n_filters` to be odd and have the same number of output 
    filters than the other filterbanks. Handle the window size accordingly.
    - Handle the STFT differently than the other ones. Maybe add an attribute
    `N_out` that can be used by the separator.
    - Add the DC component in the other analytic filterbanks.
- We need to add the flexibility of choosing the window. 
- For large number of filters (overcomplete STFT), the filters have a lot of 
zeros in them. This is not efficient, the filters should be truncated.
"""


class STFTFB(EncoderDecoder):
    def __init__(self, n_filters, kernel_size, stride, enc_or_dec='enc',
                 **kwargs):
        """ STFT filterbank.
        # Arguments
            n_filters: Positive int. Number of filters. Determines the length
                of the STFT filters before windowing.
            kernel_size: Positive int. Length of the filters (i.e the window).
            stride: Positive int. Stride of the convolution (hop size).
            enc_or_dec: String. `enc` or `dec`. Controls if filterbank is used
                as an encoder or a decoder.
        """
        super(STFTFB, self).__init__(stride, enc_or_dec=enc_or_dec)
        assert n_filters >= kernel_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.oneside = int(n_filters/2 + 1)  # was int(n_filters/2 + 1)

        filters = np.fft.fft(np.eye(n_filters))
        filters /= (.5 * kernel_size / np.sqrt(stride))
        filters = np.vstack([np.real(filters[:self.oneside, :]),
                             np.imag(filters[:self.oneside, :])])
        filters[0, :] /= np.sqrt(2)
        filters[-1, :] /= np.sqrt(2)

        win = np.hanning(kernel_size + 1)[:-1]**.5
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        self.window = np.concatenate(np.zeros((lpad,)), win, np.zeros((rpad,)))

        filters = torch.Tensor(filters * self.window).unsqueeze(1)
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters
