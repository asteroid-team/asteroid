import torch
from torch import nn

from . import PITLossWrapper


class SinkPITLossWrapper(nn.Module):
    r"""Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        n_iter (int): number of the Sinkhorn iteration (default = 200).
            Supposed to be an even number.
        hungarian_validation (boolean) : Whether to use the Hungarian algorithm
            for the validation. (default = True)

    ``loss_func`` computes pairwise losses and returns a torch.Tensor of shape
    :math:`(batch, n\_src, n\_src)`. Each element :math:`(batch, i, j)` corresponds to
    the loss between :math:`targets[:, i]` and :math:`est\_targets[:, j]`
    It evaluates an approximate value of the PIT loss using Sinkhorn's iterative algorithm.
    See :meth:`~PITLossWrapper.best_softperm_sinkhorn` and http://arxiv.org/abs/2010.11871

    Examples
        >>> import torch
        >>> import pytorch_lightning as pl
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute SinkPIT loss based on pairwise losses
        >>> loss_func = SinkPITLossWrapper(pairwise_neg_sisdr)
        >>> loss_val = loss_func(est_sources, sources)
        >>> # A fixed temperature parameter `beta` (=10) is used
        >>> # unless a cooling callback is set. The value can be
        >>> # dynamically changed using a cooling callback module as follows.
        >>> model = NeuralNetworkModel()
        >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
        >>> dataset = YourDataset()
        >>> loader = data.DataLoader(dataset, batch_size=16)
        >>> system = System(
        >>>     model,
        >>>     optimizer,
        >>>     loss_func=SinkPITLossWrapper(pairwise_neg_sisdr),
        >>>     train_loader=loader,
        >>>     val_loader=loader,
        >>>     )
        >>>
        >>> trainer = pl.Trainer(
        >>>     max_epochs=100,
        >>>     callbacks=[SinkPITBetaScheduler(lambda epoch : 1.02 ** epoch)],
        >>>     )
        >>>
        >>> trainer.fit(system)
    """

    def __init__(self, loss_func, n_iter=200, hungarian_validation=True):
        super().__init__()
        self.loss_func = loss_func
        self._beta = 10
        self.n_iter = n_iter
        self.hungarian_validation = hungarian_validation

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        assert beta > 0
        self._beta = beta

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """Evaluate the loss using Sinkhorn's algorithm.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
                the batch. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape :math:`(batch, nsrc, ...)`.
        """
        n_src = targets.shape[1]
        assert n_src < 100, f"Expected source axis along dim 1, found {n_src}"

        # Evaluate the loss using Sinkhorn's iterative algorithm
        pw_losses = self.loss_func(est_targets, targets, **kwargs)

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        if not return_est:
            if self.training or not self.hungarian_validation:
                # Train or sinkhorn validation
                min_loss, soft_perm = self.best_softperm_sinkhorn(
                    pw_losses, self._beta, self.n_iter
                )
                mean_loss = torch.mean(min_loss)
                return mean_loss
            else:
                # Reorder the output by using the Hungarian algorithm below
                min_loss, batch_indices = PITLossWrapper.find_best_perm(pw_losses)
                mean_loss = torch.mean(min_loss)
                return mean_loss
        else:
            # Test -> reorder the output by using the Hungarian algorithm below
            min_loss, batch_indices = PITLossWrapper.find_best_perm(pw_losses)
            mean_loss = torch.mean(min_loss)
            reordered = PITLossWrapper.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered

    @staticmethod
    def best_softperm_sinkhorn(pair_wise_losses, beta=10, n_iter=200):
        r"""Compute an approximate PIT loss using Sinkhorn's algorithm.
        See http://arxiv.org/abs/2010.11871

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n_src, n_src)`. Pairwise losses.
            beta (float) : Inverse temperature parameter. (default = 10)
            n_iter (int) : Number of iteration. Even number. (default = 200)

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.Tensor`:
              A soft permutation matrix.
        """
        C = pair_wise_losses.transpose(-1, -2)
        n_src = C.shape[-1]
        # initial values
        Z = -beta * C
        for it in range(n_iter // 2):
            Z = Z - torch.logsumexp(Z, axis=1, keepdim=True)
            Z = Z - torch.logsumexp(Z, axis=2, keepdim=True)
        min_loss = torch.einsum("bij,bij->b", C + Z / beta, torch.exp(Z))
        min_loss = min_loss / n_src
        return min_loss, torch.exp(Z)
