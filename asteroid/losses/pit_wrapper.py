from itertools import permutations
import torch
from torch import nn


class PITLossWrapper(nn.Module):
    """ Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        pit_from (str): Determines how PIT is applied.

            * ``'pw_mtx'`` (pairwise matrix): `loss_func` computes pairwise
              losses and returns a torch.Tensor of shape
              :math:`(batch, n\_src, n\_src)`. Each element
              :math:`[batch, i, j]` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\_targets[:, j]`
            * ``'pw_pt'`` (pairwise point): `loss_func` computes the loss for
              a batch of single source and single estimates (tensors won't
              have the source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'perm_avg'``(permutation average): `loss_func` computes the
              average loss for a given permutations of the sources and
              estimates. Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_perm_avg_loss`.

            In terms of efficiency, ``'perm_avg'`` is the least efficicient.

        perm_reduce (Callable): torch function to reduce permutation losses.
            Defaults to None (equivalent to mean). Signature of the func
            (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!).
            `perm_reduce` can receive **kwargs during forward using the
            `reduce_kwargs` argument (dict). If those argument are static,
            consider defining a small function or using `functools.partial`.
            Only used in `'pw_mtx'` and `'pw_pt'` `pit_from` modes.

    For each of these modes, the best permutation and reordering will be
    automatically computed.

    Examples:
        >>> import torch
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute PIT loss based on pairwise losses
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        >>> loss_val = loss_func(est_sources, sources)
        >>>
        >>> # Using reduce
        >>> def reduce(perm_loss, src):
        >>>     weighted = perm_loss * src.norm(dim=-1, keepdim=True)
        >>>     return torch.mean(weighted, dim=-1)
        >>>
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx',
        >>>                            perm_reduce=reduce)
        >>> reduce_kwargs = {'src': sources}
        >>> loss_val = loss_func(est_sources, sources,
        >>>                      reduce_kwargs=reduce_kwargs)
    """

    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        if self.pit_from not in ["pw_mtx", "pw_pt", "perm_avg"]:
            raise ValueError(
                "Unsupported loss function type for now. Expected"
                "one of [`pw_mtx`, `pw_pt`, `perm_avg`]"
            )

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        """ Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
                the batch. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape [batch, nsrc, *].
        """
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, min_loss_idx = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, min_loss_idx = self.find_best_perm(
            pw_losses, n_src, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        """ Get pair-wise losses between the training targets and its estimate
        for a given loss function.

        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function to get pair-wise losses from.
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            torch.Tensor or size [batch, nsrc, nsrc], losses computed for
            all permutations of the targets and est_targets.

        This function can be called on a loss function which returns a tensor
        of size [batch]. There are more efficient ways to compute pair-wise
        losses using broadcasting.
        """
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_perm_avg_loss(loss_func, est_targets, targets, **kwargs):
        """ Find best permutation from loss function with source axis.

        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function batch losses from.
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            tuple:
                :class:`torch.Tensor`: The loss corresponding to the best
                permutation of size (batch,).

                :class:`torch.LongTensor`: The indexes of the best permutations.
        """
        n_src = targets.shape[1]
        perms = list(permutations(range(n_src)))
        loss_set = torch.stack(
            [loss_func(est_targets[:, perm], targets, **kwargs) for perm in perms], dim=1
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx[:, 0]

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src, perm_reduce=None, **kwargs):
        """Find the best permutation, given the pair-wise losses.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape [batch, n_src, n_src]. Pairwise losses.
            n_src (int): Number of sources.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!)
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            tuple:
                :class:`torch.Tensor`: The loss corresponding to the best
                permutation of size (batch,).

                :class:`torch.LongTensor`: The indexes of the best permutations.

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        # Column permutation indices
        idx = torch.unsqueeze(perms, 2)
        # Loss mean of each permutation
        if perm_reduce is None:
            # one-hot, [n_src!, n_src, n_src]
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum("bij,pij->bp", [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            # batch = pwl.shape[0]; n_perm = idx.shape[0]
            # [batch, n_src!, n_src] : Pairwise losses for each permutation.
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            # Apply reduce [batch, n_src!, n_src] --> [batch, n_src!]
            loss_set = perm_reduce(pwl_set, **kwargs)
        # Indexes and values of min losses for each batch element
        min_loss_idx = torch.argmin(loss_set, dim=1)
        min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx

    @staticmethod
    def reorder_source(source, n_src, min_loss_idx):
        """ Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape [batch, n_src, time]
            n_src (int): Number of sources.
            min_loss_idx (torch.LongTensor): Tensor of shape [batch],
                each item is in [0, n_src!).

        Returns:
            :class:`torch.Tensor`:
                Reordered sources of shape [batch, n_src, time].

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        perms = source.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        # Reorder estimate targets according the best permutation
        min_loss_perm = torch.index_select(perms, dim=0, index=min_loss_idx)
        # maybe use torch.gather()/index_select()/scatter() to impl this?
        reordered_sources = torch.zeros_like(source)
        for b in range(source.shape[0]):
            for c in range(n_src):
                reordered_sources[b, c] = source[b, min_loss_perm[b][c]]
        return reordered_sources
