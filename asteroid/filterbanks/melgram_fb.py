import torch
from . import STFTFB
from . import transforms


class MelGramFB(STFTFB):
    """Mel magnitude spectrogram filterbank.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.
        n_mels (int): Number of mel bands.
        fmin (float): Minimum frequency of the mel filters.
        fmax (float): Maximum frequency of the mel filters.
        **kwargs:
    """

    def __init__(
        self,
        n_filters,
        kernel_size,
        stride=None,
        window=None,
        sample_rate=8000.0,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        **kwargs,
    ):
        from librosa.filters import mel

        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        super().__init__(
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            window=window,
            sample_rate=sample_rate,
            **kwargs,
        )
        fb_mat = mel(sr=sample_rate, n_fft=n_filters, fmin=fmin, fmax=fmax, n_mels=n_mels)
        self.register_buffer("fb_mat", torch.from_numpy(fb_mat).unsqueeze(0))
        self.n_feats_out = n_mels

    def post_encode(self, spec: torch.Tensor):
        mag_spec = transforms.take_mag(spec, dim=-2)
        mel_spec = torch.matmul(self.fb_mat, mag_spec)
        return mel_spec

    def get_config(self):
        conf = dict(fmin=self.fmin, fmax=self.fmax, n_mels=self.n_mels)
        return {**super().get_config(), **conf}
