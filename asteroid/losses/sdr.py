"""
| Negative Scale invariant Signal to Distorsion Ratio (SI-SDR) losses.
| @author : Manuel Pariente, Inria-Nancy & Samuele Cornell, UNIVPM-Ancona
"""

import torch
from torch.nn.modules.loss import _Loss
EPS = 1e-8


class PairwiseNegSDR(_Loss):
    """ Base class for pair-wise negative SI-SDR, SD-SDR and SNR on a batch.

        Args:
            type (string): choose between "snr" for plain SNR, "sisdr" for SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch, n_src, n_src].
            Pair-wise losses.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper, PairwiseNegSDR
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"), mode='pairwise')
            >>> loss = loss_func(targets, est_targets)
            >>> print(loss.size())
            torch.Size([10, 2, 2])

        References:
            [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?."
            ICASSP 2019-2019 IEEE International Conference on Acoustics,
            Speech and Signal Processing (ICASSP). IEEE, 2019.
        """

    __constants__ = ['type', 'zero_mean', 'take_log', 'reduction']

    def __init__(self, type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()

        assert type in ["snr", "sisdr", "sdsdr"]
        self.type  = type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, targets, est_targets):

        assert targets.size() == est_targets.size()
        # if scale_invariant:
        # Step 1. Zero-mean norm
        if self.zero_mean == True:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
                torch.sum(e_noise ** 2, dim=3) + EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return - pair_wise_sdr


class NoSrcSDR(_Loss):
    """ Base class for single-source negative SI-SDR, SD-SDR and SNR.

        Args:
            type (string): choose between "snr" for plain SNR, "sisdr" for SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
            reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper, NoSrcSDR
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(NoSrcSDR("sisdr"), mode='wo_src')
            >>> loss = loss_func(targets, est_targets)
            >>> print(loss.size())
            torch.Size([10, 2])

        References:
            [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?."
            ICASSP 2019-2019 IEEE International Conference on Acoustics,
            Speech and Signal Processing (ICASSP). IEEE, 2019.
        """

    __constants__ = ['type', 'zero_mean', 'take_log', 'reduction']

    def __init__(self, type, zero_mean=True, take_log=True, reduction='none'):
        assert reduction != 'sum', NotImplementedError
        super(NoSrcSDR, self).__init__(None, None, reduction)

        assert type in ["snr", "sisdr", "sdsdr"]
        self.type  = type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, target, est_target):

        assert target.size() == est_target.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        if self.type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target ** 2, dim=1,
                                        keepdim=True) + EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
                torch.sum(e_noise ** 2, dim=1) + EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == 'mean' else losses
        return - losses


class NonPitSDR(_Loss):
    """ Base class for computing negative SI-SDR, SD-SDR and SNR for a given permutation of source and
        their estimates.

        Args:
            type (string): choose between "snr" for plain SNR, "sisdr" for SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper, NonPitSDR
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(NonPitSDR("sisdr"), mode='w_src')
            >>> loss = loss_func(targets, est_targets)
            >>> print(loss.size())
            torch.Size([10])

        References:
            [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?."
            ICASSP 2019-2019 IEEE International Conference on Acoustics,
            Speech and Signal Processing (ICASSP). IEEE, 2019.
        """

    __constants__ = ['type', 'zero_mean', 'take_log', 'reduction']

    def __init__(self, type, zero_mean=True, take_log=True):
        super(NonPitSDR, self).__init__()

        assert type in ["snr", "sisdr", "sdsdr"]
        self.type  = type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, targets, est_targets):

        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        if self.type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2,
                                      keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2,
                                        keepdim=True) + EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.type in ["sdsdr", "snr"]:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
                torch.sum(e_noise ** 2, dim=2) + EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return - torch.mean(pair_wise_sdr, dim=-1)


# aliases
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
nosrc_neg_sisdr = NoSrcSDR("sisdr")
nosrc_neg_sdsdr = NoSrcSDR("sdsdr")
nosrc_neg_snr = NoSrcSDR("snr")
nonpit_neg_sisdr = NonPitSDR("sisdr")
nonpit_neg_sdsdr = NonPitSDR("sdsdr")
nonpit_neg_snr = NonPitSDR("snr")







