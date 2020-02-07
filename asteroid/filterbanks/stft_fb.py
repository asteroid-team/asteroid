import torch
import numpy as np
from .enc_dec import Filterbank


class STFTFB(Filterbank):
    """STFT filterbank.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.
    """
    def __init__(self, n_filters, kernel_size, stride=None, window=None,
                 **kwargs):
        super(STFTFB, self).__init__(n_filters, kernel_size, stride=stride)
        assert n_filters >= kernel_size
        self.cutoff = int(n_filters/2 + 1)
        self.n_feats_out = 2 * self.cutoff

        if window is None:
            self.window = np.hanning(kernel_size + 1)[:-1]**.5
        else:
            ws = window.size
            if not (ws == kernel_size):
                raise AssertionError('Expected window of size {}.'
                                     'Received window of size {} instead.'
                                     ''.format(kernel_size, ws))
            self.window = window
        # Create and normalize DFT filters (can be overcomplete)
        filters = np.fft.fft(np.eye(n_filters))
        filters /= (.5 * np.sqrt(kernel_size * n_filters / self.stride))
        # Keep only the windowed centered part to save computation.
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        indexes = list(range(lpad, n_filters - rpad))
        filters = np.vstack([np.real(filters[:self.cutoff, indexes]),
                             np.imag(filters[:self.cutoff, indexes])])
        filters[0, :] /= np.sqrt(2)
        filters[-1, :] /= np.sqrt(2)
        filters = torch.from_numpy(filters * self.window).unsqueeze(1).float()
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters
