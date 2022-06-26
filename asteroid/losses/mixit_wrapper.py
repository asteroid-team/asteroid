import warnings
from itertools import combinations
import torch
from torch import nn


class MixITLossWrapper(nn.Module):
    r"""Mixture invariant loss wrapper.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        generalized (bool): Determines how MixIT is applied. If False ,
            apply MixIT for any number of mixtures as soon as they contain
            the same number of sources (:meth:`~MixITLossWrapper.best_part_mixit`.)
            If True (default), apply MixIT for two mixtures, but those mixtures do not
            necessarly have to contain the same number of sources.
            See :meth:`~MixITLossWrapper.best_part_mixit_generalized`.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    For each of these modes, the best partition and reordering will be
    automatically computed.

    Examples:
        >>> import torch
        >>> from asteroid.losses import multisrc_mse
        >>> mixtures = torch.randn(10, 2, 16000)
        >>> est_sources = torch.randn(10, 4, 16000)
        >>> # Compute MixIT loss based on pairwise losses
        >>> loss_func = MixITLossWrapper(multisrc_mse)
        >>> loss_val = loss_func(est_sources, mixtures)

    References
        [1] Scott Wisdom et al. "Unsupervised sound separation using
        mixtures of mixtures." arXiv:2006.12701 (2020)
    """

    def __init__(self, loss_func, generalized=True, reduction="mean"):
        super().__init__()
        self.loss_func = loss_func
        self.generalized = generalized
        self.reduction = reduction

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        r"""Find the best partition and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, *)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets
            return_est: Boolean. Whether to return the estimated mixtures
                estimates (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best partition loss for each batch sample, average over
              the batch. torch.Tensor(loss_value)
            - The estimated mixtures (estimated sources summed according to the partition)
              if return_est is True. torch.Tensor of shape :math:`(batch, nmix, ...)`.
        """
        # Check input dimensions
        assert est_targets.shape[0] == targets.shape[0]
        assert est_targets.shape[2] == targets.shape[2]

        if not self.generalized:
            min_loss, min_loss_idx, parts = self.best_part_mixit(
                self.loss_func, est_targets, targets, **kwargs
            )
        else:
            min_loss, min_loss_idx, parts = self.best_part_mixit_generalized(
                self.loss_func, est_targets, targets, **kwargs
            )

        # Apply any reductions over the batch axis
        returned_loss = min_loss.mean() if self.reduction == "mean" else min_loss
        if not return_est:
            return returned_loss

        # Order and sum on the best partition to get the estimated mixtures
        reordered = self.reorder_source(est_targets, targets, min_loss_idx, parts)
        return returned_loss, reordered

    @staticmethod
    def best_part_mixit(loss_func, est_targets, targets, **kwargs):
        r"""Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid for any number of
        mixtures as soon as they contain the same number of sources.

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indices of the best partition.

            - :class:`list`:
              list of the possible partitions of the sources.

        """
        nmix = targets.shape[1]
        nsrc = est_targets.shape[1]
        if nsrc % nmix != 0:
            raise ValueError("The mixtures are assumed to contain the same number of sources")
        nsrcmix = nsrc // nmix

        # Generate all unique partitions of size k from a list lst of
        # length n, where l = n // k is the number of parts. The total
        # number of such partitions is: NPK(n,k) = n! / ((k!)^l * l!)
        # Algorithm recursively distributes items over parts
        def parts_mixit(lst, k, l):
            if l == 0:
                yield []
            else:
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    for r in parts_mixit(rest, k, l - 1):
                        yield [list(c), *r]

        # Generate all the possible partitions
        parts = list(parts_mixit(range(nsrc), nsrcmix, nmix))
        # Compute the loss corresponding to each partition
        loss_set = MixITLossWrapper.loss_set_from_parts(
            loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def best_part_mixit_generalized(loss_func, est_targets, targets, **kwargs):
        r"""Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid only for two mixtures,
        but those mixtures do not necessarly have to contain the same number of
        sources e.g the case where one mixture is silent is allowed..

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indexes of the best permutations.

            - :class:`list`:
              list of the possible partitions of the sources.
        """
        nmix = targets.shape[1]  # number of mixtures
        nsrc = est_targets.shape[1]  # number of estimated sources
        if nmix != 2:
            raise ValueError("Works only with two mixtures")

        # Generate all unique partitions of any size from a list lst of
        # length n. Algorithm recursively distributes items over parts
        def parts_mixit_gen(lst):
            partitions = []
            for k in range(len(lst) + 1):
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    partitions.append([list(c), rest])
            return partitions

        # Generate all the possible partitions
        parts = parts_mixit_gen(range(nsrc))
        # Compute the loss corresponding to each partition
        loss_set = MixITLossWrapper.loss_set_from_parts(
            loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def loss_set_from_parts(loss_func, est_targets, targets, parts, **kwargs):
        """Common loop between both best_part_mixit"""
        loss_set = []
        for partition in parts:
            # sum the sources according to the given partition
            est_mixes = torch.stack([est_targets[:, idx, :].sum(1) for idx in partition], dim=1)
            # get loss for the given partition
            loss_partition = loss_func(est_mixes, targets, **kwargs)
            if loss_partition.ndim != 1:
                raise ValueError("Loss function return value should be of size (batch,).")
            loss_set.append(loss_partition[:, None])
        loss_set = torch.cat(loss_set, dim=1)
        return loss_set

    @staticmethod
    def reorder_source(est_targets, targets, min_loss_idx, parts):
        """Reorder sources according to the best partition.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets.
            min_loss_idx: torch.LongTensor. The indexes of the best permutations.
            parts: list of the possible partitions of the sources.

        Returns:
            :class:`torch.Tensor`: Reordered sources of shape :math:`(batch, nmix, time)`.

        """
        # For each batch there is a different min_loss_idx
        ordered = torch.zeros_like(targets)
        for b, idx in enumerate(min_loss_idx):
            right_partition = parts[idx]
            # Sum the estimated sources to get the estimated mixtures
            ordered[b, :, :] = torch.stack(
                [est_targets[b, idx, :][None, :, :].sum(1) for idx in right_partition], dim=1
            )

        return ordered
