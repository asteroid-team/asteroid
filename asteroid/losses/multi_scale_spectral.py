import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from asteroid_filterbanks import STFTFB, Encoder
from asteroid_filterbanks.transforms import mag


class SingleSrcMultiScaleSpectral(_Loss):
    r"""Measure multi-scale spectral loss as described in [1]

    Args:
        n_filters (list): list containing the number of filter desired for
            each STFT
        windows_size (list): list containing the size of the window desired for
            each STFT
        hops_size (list): list containing the size of the hop desired for
            each STFT

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples
        >>> import torch
        >>> targets = torch.randn(10, 32000)
        >>> est_targets = torch.randn(10, 32000)
        >>> # Using it by itself on a pair of source/estimate
        >>> loss_func = SingleSrcMultiScaleSpectral()
        >>> loss = loss_func(est_targets, targets)

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # Using it with PITLossWrapper with sets of source/estimates
        >>> loss_func = PITLossWrapper(SingleSrcMultiScaleSpectral(),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and
        Adam Roberts "DDSP: Differentiable Digital Signal Processing" ICLR 2020.
    """

    def __init__(self, n_filters=None, windows_size=None, hops_size=None, alpha=1.0):
        super().__init__()

        if windows_size is None:
            windows_size = [2048, 1024, 512, 256, 128, 64, 32]
        if n_filters is None:
            n_filters = [2048, 1024, 512, 256, 128, 64, 32]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128, 64, 32, 16]

        self.windows_size = windows_size
        self.n_filters = n_filters
        self.hops_size = hops_size
        self.alpha = alpha

        self.encoders = nn.ModuleList(
            Encoder(STFTFB(n_filters[i], windows_size[i], hops_size[i]))
            for i in range(len(self.n_filters))
        )

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)

        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)
        return loss

    def compute_spectral_loss(self, encoder, est_target, target, EPS=1e-8):
        batch_size = est_target.shape[0]
        spect_est_target = mag(encoder(est_target)).view(batch_size, -1)
        spect_target = mag(encoder(target)).view(batch_size, -1)
        linear_loss = self.norm1(spect_est_target - spect_target)
        log_loss = self.norm1(torch.log(spect_est_target + EPS) - torch.log(spect_target + EPS))
        return linear_loss + self.alpha * log_loss

    @staticmethod
    def norm1(a):
        return torch.norm(a, p=1, dim=1)
