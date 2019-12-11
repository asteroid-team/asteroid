"""
Package-level utils.
@author : Manuel Pariente, Inria-Nancy

Modified and extended from github.com/kaituoxu/Conv-TasNet/
MIT Copyright (c) 2018 Kaituo XU
See also github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE
"""

from itertools import permutations
import torch


class PITLossWrapper(object):
    """ Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
    Two different behaviors are accepted for loss_func. For targets and
    est_targets of shape [batch, nsrc, *], it can return :
            - A tensor of shape [batch, nsrc, nsrc]. This corresponds to
            pair-wise losses. Each element [batch, i, j] corresponds to the
            loss between target[:, i] and est_targets[:, j]. The best
            permutation and reordering is automatically computed from this
            tensor.
            - A tensor of shape [batch]. This corresponds to a loss computed
            between two sources (i.e there is no source dimension in the
            expected tensors). `PITLossWrapper` will loop to compute pair-wise
            losses from `loss_func` and find the best permutations.
            See :meth:`~PITLossWrapper.get_pw_losses`.
        Note that the secong way is easier but a bit slower because the of the
        loop over the pairs of sources. The expensive part of the computation is
        actually the backward so the difference should be marginal during
        training.
        mode (str): ``'pairwise'``, ``'wo_src'``, ``'w_src'``
    """

    def __init__(self, loss_func, mode='pairwise'):
        self.loss_func = loss_func
        self.mode = mode
        if self.mode not in ["pairwise", "wo_src", "w_src"]:
            raise ValueError('Unsupported loss function type for now. Expected'
                             'one of [`pairwise`, `wo_src`, `w_src`]')

    def __call__(self, targets, est_targets, return_est=False, **kwargs):
        """ Find the best permutation and return the loss.
        Args:
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
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
        if self.mode == 'pairwise':
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(targets, est_targets, **kwargs)
        elif self.mode == 'wo_src':
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, targets,
                                           est_targets, **kwargs)
        elif self.mode == 'w_src':
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, min_loss_idx = self.best_perm_from_wsrc_loss(
                self.loss_func, targets, est_targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, ("Something went wrong with the loss "
                                     "function, please read the docs.")
        assert (pw_losses.shape[0] ==
                targets.shape[0]), "PIT loss needs same batch dim as input"

        min_loss, min_loss_idx = self.find_best_perm(pw_losses, n_src)
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, targets, est_targets, **kwargs):
        """ Get pair-wise losses between the training targets and its estimate
        for a given loss function.
        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function to get pair-wise losses from
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
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
        pair_wise_losses = torch.empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    target_src, est_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_wsrc_loss(loss_func, targets, est_targets, **kwargs):
        """

        Args:
            loss_func:
            targets:
            est_targets:
            **kwargs:

        Returns:

        """
        n_src = targets.shape[1]
        perms = list(permutations(range(n_src)))
        loss_set = torch.stack([loss_func(targets[:, perm],
                                          est_targets,
                                          **kwargs) for perm in perms],
                               dim=1)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx[:, 0]

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src):
        """Find the best permutation, given the pair wise losses.

        Args:
            pair_wise_losses: torch.Tensor. Expected shape [batch, n_src, n_src]
                Pair-wise losses.
            n_src: int > 0. Number of sources.

        Returns:
            - torch.Tensor (batch,). The best permutation loss.
            - torch.LongTensor. The indexes of the best permutations.
        """
        pwl = pair_wise_losses
        perms = pwl.new_tensor(list(permutations(range(n_src))),
                               dtype=torch.long)
        # one-hot, [n_src!, n_src, n_src]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, index,
                                                                       1)
        # Loss sum of each permutation
        loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
        # Indexes and values of min losses for each batch element
        min_loss_idx = torch.argmin(loss_set, dim=1)
        min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
        min_loss /= n_src
        return min_loss, min_loss_idx

    @staticmethod
    def reorder_source(source, n_src, min_loss_idx):
        """ Reorder sources according to the best permutation.

        Args:
            source: torch.Tensor. shape [batch, n_src, time]
            n_src: int > 0. Number of sources.
            min_loss_idx: torch.LongTensor. shape [batch], each item is in [0, C!)

        Returns:
            reordered_sources: [batch, n_src, time]
        """
        perms = source.new_tensor(list(permutations(range(n_src))),
                                  dtype=torch.long)
        # Reorder estimate targets according the best permutation
        min_loss_perm = torch.index_select(perms, dim=0, index=min_loss_idx)
        # maybe use torch.gather()/index_select()/scatter() to impl this?
        reordered_sources = torch.zeros_like(source)
        for b in range(source.shape[0]):
            for c in range(n_src):
                reordered_sources[b, c] = source[b, min_loss_perm[b][c]]
        return reordered_sources
