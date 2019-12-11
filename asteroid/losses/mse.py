"""
Mean squared error loss functions.
@author : Manuel Pariente, Inria-Nancy
"""


def pairwise_mse(targets, est_targets):
    """ Pair-wise mean square error
    Args:
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
    Returns:
        torch.tensor [batch, nsrc, nsrc]
    Usage:
        >>> from asteroid.losses import PITLossWrapper
        >>> loss_class = PITLossWrapper(pairwise_mse)
        >>> loss = loss_class(targets, est_targets)
    """
    targets = targets.unsqueeze(1)
    est_targets = est_targets.unsqueeze(2)
    pw_loss = (targets - est_targets)**2
    # Need to return [batch, nsrc, nsrc]
    mean_over = list(range(3, pw_loss.ndim))
    return pw_loss.mean(dim=mean_over)


def nosrc_mse(targets, est_targets):
    """ Mean square error over batch .

    Supports both tensors with and without source axis.

    Args:
            targets: torch.Tensor. Expected shape [batch, *].
                The batch of training targets (one source)
            est_targets: torch.Tensor. Expected shape [batch, *].
                The batch of target estimates (one source)
    Returns:
        torch.tensor [batch]
    Usage:
        >>> from asteroid.losses import PITLossWrapper
        >>> loss_class = PITLossWrapper(nosrc_mse)
        >>> loss = loss_class(targets, est_targets)
    """
    loss = (targets - est_targets)**2
    # Need to return [batch]
    mean_over = list(range(1, loss.ndim))
    return loss.mean(dim=mean_over)


nonpit_mse = nosrc_mse
