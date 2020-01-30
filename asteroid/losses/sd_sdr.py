"""
| Negative Scale Dependent to Distorsion Ratio (SD-SDR) losses.
| @author : Samuele Cornell, Università Politecnica delle Marche, Ancona
"""

import torch
EPS = 1e-8


def pairwise_neg_sdsdr(targets, est_targets, zero_mean=True):
    """ Measure pair-wise negative SI-SDR on a batch.

    Args:
        targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
            Batch of training targets.
        est_targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
            Batch of target estimates.
        zero_mean (bool): Whether to zero-mean the estimates and the targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch, n_src, n_src].
        Pair-wise losses.

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(pairwise_neg_sdsdr, mode='pairwise')
        >>> loss = loss_func(targets, est_targets)
        >>> print(loss.size())
        torch.Size([])

    References:

    """
    assert targets.size() == est_targets.size()
    # if scale_invariant:
    # Step 1. Zero-mean norm
    if zero_mean:
        mean_source = torch.mean(targets, dim=2, keepdim=True)
        mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
        targets = targets - mean_source
        est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    s_target = torch.unsqueeze(targets, dim=1)
    s_estimate = torch.unsqueeze(est_targets, dim=2)

    # [batch, n_src, n_src, 1]
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
    # [batch, 1, n_src, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
    # [batch, n_src, n_src, time]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy

    e_noise = s_estimate - s_target
    # [batch, n_src, n_src]
    pair_wise_si_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
                torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_losses = - 10 * torch.log10(pair_wise_si_sdr + EPS)
    return pair_wise_losses


def nosrc_neg_sdsdr(target, est_target, zero_mean=True):
    """ Measure negative SI-SDR between a single source and its estimate.

    Args:
        target (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of training target.
        est_target (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of target estimate.
        zero_mean (bool): Whether to zero-mean the estimates and the targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch]. Batch losses.

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(nosrc_neg_sdsdr, mode='wo_src')
        >>> loss = loss_func(targets, est_targets)
        >>> print(loss.size())
        torch.Size([10])

    References:

    """
    assert target.size() == est_target.size()
    # Step 1. Zero-mean norm
    if zero_mean:
        mean_source = torch.mean(target, dim=1, keepdim=True)
        mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
        target = target - mean_source
        est_target = est_target - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    # [batch, 1]
    dot = torch.sum(est_target * target, dim=1, keepdim=True)
    # [batch, 1]
    s_target_energy = torch.sum(target ** 2, dim=1,
                                    keepdim=True) + EPS
    # [batch, time]
    scaled_target = dot * target / s_target_energy

    e_noise = est_target - target
    # [batch]
    si_sdr = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + EPS)
    losses = - 10 * torch.log10(si_sdr + EPS)
    return losses

def nonpit_neg_sdsdr(targets, est_targets, zero_mean=True):
    """ Measure mean negative SI-SDR for a given permutation of source and
        their estimates.

    Args:
        targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of training targets.
        est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of target estimates.
        zero_mean (bool): Whether to zero-mean the estimates and the targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch].
            Batch losses for this permutation.

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(nonpit_neg_sdsdr, mode='w_src')
        >>> loss = loss_func(targets, est_targets)
        >>> print(loss.size())
        torch.Size([10])

    References:

    """
    assert targets.size() == est_targets.size()
    # Step 1. Zero-mean norm
    if zero_mean:
        mean_source = torch.mean(targets, dim=2, keepdim=True)
        mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
        targets = targets - mean_source
        est_targets = est_targets - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)

    # [batch, n_src]
    pair_wise_dot = torch.sum(est_targets * targets, dim=2,
                                  keepdim=True)
    # [batch, n_src]
    s_target_energy = torch.sum(targets ** 2, dim=2,
                                    keepdim=True) + EPS
    # [batch, n_src, time]
    scaled_targets = pair_wise_dot * targets / s_target_energy
    e_noise = est_targets - targets
    # [batch, n_src]
    pair_wise_si_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_losses = - 10 * torch.log10(pair_wise_si_sdr + EPS)
    return torch.mean(pair_wise_losses, dim=-1)



