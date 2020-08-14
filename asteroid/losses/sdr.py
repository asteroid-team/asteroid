import torch
from torch.nn.modules.loss import _Loss
from ..utils.deprecation_utils import DeprecationMixin

EPS = 1e-8


class PairwiseNegSDR(_Loss):
    """ Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

        Args:
            sdr_type (str): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch, n_src, n_src].
            Pairwise losses.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
            >>>                            pit_from='pairwise')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return -pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    """ Base class for single-source negative SI-SDR, SD-SDR and SNR.

        Args:
            sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and
                estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
            reduction (string, optional): Specifies the reduction to apply to
                the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else
                [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
            >>>                            pit_from='pw_pt')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none"):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_target, target):
        assert target.size() == est_target.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    """ Base class for computing negative SI-SDR, SD-SDR and SNR for a given
        permutation of source and their estimates.

        Args:
            sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else
                [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
            >>>                            pit_from='perm_avg')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.

        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)


# aliases
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")


# Legacy
class NonPitSDR(MultiSrcNegSDR, DeprecationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


class NoSrcSDR(SingleSrcNegSDR, DeprecationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


nosrc_neg_sisdr = singlesrc_neg_sisdr
nosrc_neg_sdsdr = singlesrc_neg_sdsdr
nosrc_neg_snr = singlesrc_neg_snr
nonpit_neg_sisdr = multisrc_neg_sisdr
nonpit_neg_sdsdr = multisrc_neg_sdsdr
nonpit_neg_snr = multisrc_neg_snr
