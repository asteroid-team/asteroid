import torch


def compute_delta(feats: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute delta coefficients of a tensor.

    Args:
        feats: Input features to compute deltas with.
        dim: feature dimension in the feats tensor.

    Returns:
        Tensor: Tensor of deltas.

    Examples
        >>> import torch
        >>> phase = torch.randn(2, 257, 100)
        >>> # Compute instantaneous frequency
        >>> inst_freq = compute_delta(phase, dim=-1)
        >>> # Or group delay
        >>> group_delay = compute_delta(phase, dim=-2)
    """
    if dim != -1:
        return compute_delta(feats.transpose(-1, dim), dim=-1).transpose(-1, dim)
    # First frame has nothing. Then each frame is the diff with the previous one.
    delta = feats.new_zeros(feats.shape)
    delta[..., 1:] = feats[..., 1:] - feats[..., :-1]
    return delta


def concat_deltas(feats: torch.Tensor, order: int = 1, dim: int = -1) -> torch.Tensor:
    """Concatenate delta coefficients of a tensor to itself.

    Args:
        feats: Input features to compute deltas with.
        order: Order of the delta e.g with order==2, compute delta of delta
            as well.
        dim: feature dimension in the feats tensor.

    Returns:
        Tensor: Concatenation of the features, the deltas and subsequent deltas.

    Examples
        >>> import torch
        >>> phase = torch.randn(2, 257, 100)
        >>> # Compute second order instantaneous frequency
        >>> phase_and_inst_freq = concat_deltas(phase, order=2, dim=-1)
        >>> # Or group delay
        >>> phase_and_group_delay = concat_deltas(phase, order=2, dim=-2)
    """
    all_feats = [feats]
    for _ in range(order):
        all_feats.append(compute_delta(all_feats[-1], dim=dim))
    return torch.cat(all_feats, dim=dim)
