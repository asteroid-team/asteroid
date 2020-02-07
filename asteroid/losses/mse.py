from torch.nn.modules.loss import _Loss


class PairwiseMSE(_Loss):
    """ Measure pairwise mean square error on a batch.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of training targets

    Returns:
        :class:`torch.Tensor`: with shape [batch, nsrc, nsrc]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseMSE(), mode='pairwise')
        >>> loss = loss_func(est_targets, targets)
        >>> print(loss.size())
        torch.Size([10, 2, 2])
    """
    def forward(self, est_targets, targets):
        targets = targets.unsqueeze(1)
        est_targets = est_targets.unsqueeze(2)
        pw_loss = (targets - est_targets)**2
        # Need to return [batch, nsrc, nsrc]
        mean_over = list(range(3, pw_loss.ndim))
        return pw_loss.mean(dim=mean_over)


class NoSrcMSE(_Loss):
    """ Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, *].
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
    def forward(self, est_targets, targets):
        loss = (targets - est_targets)**2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)


NonPitMSE = NoSrcMSE

# aliases
pairwise_mse = PairwiseMSE()
nosrc_mse = NoSrcMSE()
nonpit_mse = NonPitMSE()

