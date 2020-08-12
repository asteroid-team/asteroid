import numpy as np
import torch
import torch.nn as nn
from .enc_dec import Filterbank


class ParamSincFB(Filterbank):
    """Extension of the parameterized filterbank from [1] proposed in [2].
    Modified and extended from from `<https://github.com/mravanelli/SincNet>`__

    Args:
        n_filters (int): Number of filters. Half of `n_filters` (the real
            parts) will have parameters, the other half will correspond to the
            imaginary parts. `n_filters` should be even.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.
        sample_rate (int, optional): The sample rate (used for initialization).
        min_low_hz (int, optional): Lowest low frequency allowed (Hz).
        min_band_hz (int, optional): Lowest band frequency allowed (Hz).

    Attributes:
        n_feats_out (int): Number of output filters.

    References:
        [1] : "Speaker Recognition from raw waveform with SincNet". SLT 2018.
        Mirco Ravanelli, Yoshua Bengio.  https://arxiv.org/abs/1808.00158

        [2] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent. https://arxiv.org/abs/1910.10400
    """

    def __init__(
        self, n_filters, kernel_size, stride=None, sample_rate=16000, min_low_hz=50, min_band_hz=50
    ):
        if kernel_size % 2 == 0:
            print(
                "Received kernel_size={}, force ".format(kernel_size)
                + "kernel_size={} so filters are odd".format(kernel_size + 1)
            )
            kernel_size += 1
        super(ParamSincFB, self).__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz

        self.half_kernel = self.kernel_size // 2
        self.cutoff = int(n_filters // 2)
        self.n_feats_out = 2 * self.cutoff
        self._initialize_filters()
        if n_filters % 2 != 0:
            print(
                "If the number of filters `n_filters` is odd, the "
                "output size of the layer will be `n_filters - 1`."
            )

        window_ = np.hamming(self.kernel_size)[: self.half_kernel]  # Half window
        n_ = (
            2 * np.pi * (torch.arange(-self.half_kernel, 0.0).view(1, -1) / self.sample_rate)
        )  # Half time vector
        self.register_buffer("window_", torch.from_numpy(window_).float())
        self.register_buffer("n_", n_)

    def _initialize_filters(self):
        """ Filter Initialization along the Mel scale"""
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.n_filters // 2 + 1, dtype="float32"
        )
        hz = self.to_hz(mel)
        # filters parameters (out_channels // 2, 1)
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).view(-1, 1))

    @property
    def filters(self):
        """ Compute filters from parameters """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2
        )
        cos_filters = self.make_filters(low, high, filt_type="cos")
        sin_filters = self.make_filters(low, high, filt_type="sin")
        return torch.cat([cos_filters, sin_filters], dim=0)

    def make_filters(self, low, high, filt_type="cos"):
        band = (high - low)[:, 0]
        ft_low = torch.matmul(low, self.n_)
        ft_high = torch.matmul(high, self.n_)
        if filt_type == "cos":  # Even filters from the SincNet paper.
            bp_left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2)) * self.window_
            bp_center = 2 * band.view(-1, 1)
            bp_right = torch.flip(bp_left, dims=[1])
        elif filt_type == "sin":  # Extension including odd filters
            bp_left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2)) * self.window_
            bp_center = torch.zeros_like(band.view(-1, 1))
            bp_right = -torch.flip(bp_left, dims=[1])
        else:
            raise ValueError("Invalid filter type {}".format(filt_type))
        band_pass = torch.cat([bp_left, bp_center, bp_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        return band_pass.view(self.n_filters // 2, 1, self.kernel_size)

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {
            "sample_rate": self.sample_rate,
            "min_low_hz": self.min_low_hz,
            "min_band_hz": self.min_band_hz,
        }
        base_config = super(ParamSincFB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
