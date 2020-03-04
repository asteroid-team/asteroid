import torch
import numpy as np
from .enc_dec import Filterbank


class STFTFB(Filterbank):
    """ STFT filterbank.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.

    Attributes:
        n_feats_out (int): Number of output filters.
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
        filters /= (0.5 * np.sqrt(kernel_size * n_filters / self.stride))

        # Keep only the windowed centered part to save computation.
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        indexes = list(range(lpad, n_filters - rpad))
        filters = np.vstack([np.real(filters[:self.cutoff, indexes]),
                             np.imag(filters[:self.cutoff, indexes])])

        filters[0, :] /= np.sqrt(2)
        filters[n_filters // 2, :] /= np.sqrt(2)
        filters = torch.from_numpy(filters * self.window).unsqueeze(1).float()
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters


def perfect_synthesis_window(analysis_window, hop_size):
    """ Computes a window for perfect synthesis given an analysis window and
        a hop size.

    Args:
        analysis_window (np.array): Analysis window of the transform.
        hop_size (int): Hop size in number of samples.

    Returns:
        np.array : the synthesis window to use for perfectly inverting the STFT.
    """
    win_size = len(analysis_window)
    den = np.zeros_like(analysis_window)

    loop_on = (win_size - 1) // hop_size
    for win_idx in range(-loop_on, loop_on + 1):
        shifted = np.roll(analysis_window ** 2, win_idx * hop_size)
        if win_idx < 0:
            shifted[win_idx * hop_size:] = 0
        elif win_idx > 0:
            shifted[:win_idx * hop_size] = 0
        den += shifted
    den = np.where(den != 0., den, np.finfo(den.dtype).tiny)
    correction = int(0.5 * len(analysis_window) / hop_size)
    return correction * analysis_window / den
