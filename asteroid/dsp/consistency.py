import torch
from typing import Optional, List


def mixture_consistency(
    mixture: torch.Tensor,
    est_sources: torch.Tensor,
    src_weights: Optional[torch.Tensor] = None,
    dim: int = 1,
) -> torch.Tensor:
    """Applies mixture consistency to a tensor of estimated sources.

    Args:
        mixture (torch.Tensor): Mixture waveform or TF representation.
        est_sources (torch.Tensor): Estimated sources waveforms or TF representations.
        src_weights (torch.Tensor): Consistency weight for each source.
            Shape needs to be broadcastable to `est_source`.
            We make sure that the weights sum up to 1 along dim `dim`.
            If `src_weights` is None, compute them based on relative power.
        dim (int): Axis which contains the sources in `est_sources`.

    Returns
        torch.Tensor with same shape as `est_sources`, after applying mixture
        consistency.

    Examples
        >>> # Works on waveforms
        >>> mix = torch.randn(10, 16000)
        >>> est_sources = torch.randn(10, 2, 16000)
        >>> new_est_sources = mixture_consistency(mix, est_sources, dim=1)
        >>> # Also works on spectrograms
        >>> mix = torch.randn(10, 514, 400)
        >>> est_sources = torch.randn(10, 2, 514, 400)
        >>> new_est_sources = mixture_consistency(mix, est_sources, dim=1)

    .. note::
        This method can be used only in 'complete' separation tasks, otherwise
        the residual error will contain unwanted sources. For example, this
        won't work with the task `"sep_noisy"` from WHAM.

    References
        Scott Wisdom et al. "Differentiable consistency constraints for improved
        deep speech enhancement", ICASSP 2019.
    """
    # If the source weights are not specified, the weights are the relative
    # power of each source to the sum. w_i = P_i / (P_all), P for power.
    if src_weights is None:
        all_dims: List[int] = torch.arange(est_sources.ndim).tolist()
        all_dims.pop(dim)  # Remove source axis
        all_dims.pop(0)  # Remove batch axis
        src_weights = torch.mean(est_sources**2, dim=all_dims, keepdim=True)
    # Make sure that the weights sum up to 1
    norm_weights = torch.sum(src_weights, dim=dim, keepdim=True) + 1e-8
    src_weights = src_weights / norm_weights

    # Compute residual mix - sum(est_sources)
    if mixture.ndim == est_sources.ndim - 1:
        # mixture (batch, *), est_sources (batch, n_src, *)
        residual = (mixture - est_sources.sum(dim=dim)).unsqueeze(dim)
    elif mixture.ndim == est_sources.ndim:
        # mixture (batch, 1, *), est_sources (batch, n_src, *)
        residual = mixture - est_sources.sum(dim=dim, keepdim=True)
    else:
        n, m = est_sources.ndim, mixture.ndim
        raise RuntimeError(
            f"The size of the mixture tensor should match the "
            f"size of the est_sources tensor. Expected mixture"
            f"tensor to have {n} or {n-1} dimension, found {m}."
        )
    # Compute remove
    new_sources = est_sources + src_weights * residual
    return new_sources
