import torch
EPS = 1e-8


def pairwise_neg_sisdr(targets, est_targets, scale_invariant=True):
    """Calculate pair-wise negative SI-SDR.

    Args:
        targets: torch.Tensor of shape [batch, n_src, time]. The target sources.
        est_targets: torch.Tensor of shape [batch, n_src, time]. Estimates
            of the target sources.
        scale_invariant: Boolean. Whether to rescale the estimated sources to
            the targets.

    Returns:
        torch.Tensor of shape [batch, n_src, n_src]. Pair-wise losses.
    """
    assert targets.size() == est_targets.size()
    # if scale_invariant:
    # Step 1. Zero-mean norm
    mean_source = torch.mean(targets, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
    targets = targets - mean_source
    est_targets = est_targets - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    s_target = torch.unsqueeze(targets, dim=1)
    s_estimate = torch.unsqueeze(est_targets, dim=2)
    if scale_invariant:
        # [batch, n_src, n_src, 1]
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
        # [batch, 1, n_src, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
        # [batch, n_src, n_src, time]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy
    else:
        # [batch, n_src, n_src, time]
        pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
    e_noise = s_estimate - pair_wise_proj
    # [batch, n_src, n_src]
    pair_wise_si_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
                torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_losses = - 10 * torch.log10(pair_wise_si_sdr + EPS)
    return pair_wise_losses


def nosrc_neg_sisdr(target, est_target, scale_invariant=True):
    """ Calculate negative SI-SDR for tensors without source axis.

    Args:
        target: torch.Tensor of shape [batch, time]. The target source.
        est_target: torch.Tensor of shape [batch, time]. Estimates
            of the target source.
        scale_invariant: Boolean. Whether to rescale the estimated source to
            the target.

    Returns:
        torch.Tensor of shape [batch]. Batch losses for this source/est pair.
    """
    assert target.size() == est_target.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(target, dim=1, keepdim=True)
    mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
    target = target - mean_source
    est_target = est_target - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    if scale_invariant:
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
    e_noise = est_target - scaled_target
    # [batch]
    si_sdr = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + EPS)
    losses = - 10 * torch.log10(si_sdr + EPS)
    return losses


def nonpit_neg_sisdr(targets, est_targets, scale_invariant=True):
    """ Calculate mean negative SI-SDR for tensors with source axis.
    Args:
        targets: torch.Tensor of shape [batch, n_src, time]. The target sources.
        est_targets: torch.Tensor of shape [batch, n_src, time]. Estimates
            of the target sources.
        scale_invariant: Boolean. Whether to rescale the estimated sources to
            the targets.

    Returns:
        torch.Tensor of shape [batch]. Batch losses for this permutation.
    """
    assert targets.size() == est_targets.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(targets, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
    targets = targets - mean_source
    est_targets = est_targets - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    if scale_invariant:
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
    e_noise = est_targets - scaled_targets
    # [batch, n_src]
    pair_wise_si_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_losses = - 10 * torch.log10(pair_wise_si_sdr + EPS)
    return torch.mean(pair_wise_losses, dim=-1)
