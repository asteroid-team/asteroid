"""
| Mean squared error loss functions.
| @author : Manuel Pariente, Inria-Nancy
"""


def pairwise_mse(est_targets, targets):
    """ Measure pair-wise mean square error on a batch.

    Args:
        est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
            The batch of target estimates.
        targets: torch.Tensor. Expected shape [batch, nsrc, *].
            The batch of training targets

    Returns:
        :class:`torch.Tensor`: with shape [batch, nsrc, nsrc]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(pairwise_mse, mode='pairwise')
        >>> loss = loss_func(est_targets, targets)
        >>> print(loss.size())
        torch.Size([10, 2, 2])
    """
    targets = targets.unsqueeze(1)
    est_targets = est_targets.unsqueeze(2)
    pw_loss = (targets - est_targets)**2
    # Need to return [batch, nsrc, nsrc]
    mean_over = list(range(3, pw_loss.ndim))
    return pw_loss.mean(dim=mean_over)


def nosrc_mse(est_targets, targets):
    """ Measure mean square error on a batch.

    Supports both tensors with and without source axis.

    Args:
        est_targets: torch.Tensor. Expected shape [batch, *].
            The batch of target estimates.
        targets: torch.Tensor. Expected shape [batch, *].
            The batch of training targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # nosrc_mse / nonpit_mse support both 'wo_src' and 'w_src'.
        >>> loss_func = PITLossWrapper(nosrc_mse, mode='wo_src')
        >>> loss = loss_func(est_targets, targets)
        >>> print(loss.size())
        torch.Size([10])
    """
    loss = (targets - est_targets)**2
    mean_over = list(range(1, loss.ndim))
    return loss.mean(dim=mean_over)


nonpit_mse = nosrc_mse
