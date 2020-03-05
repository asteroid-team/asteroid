from itertools import permutations
import torch
from torch import nn


class PITLossWrapper(nn.Module):
    """ Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        mode (str): Determines how PIT is applied.

            * ``'pairwise'``: `loss_func` computes pairwise losses and returns
              a torch.Tensor of shape :math:`(batch, n\_src, n\_src)`.
              Each element :math:`[batch, i, j]` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\_targets[:, j]`
            * ``'wo_src'``: `loss_func` computes the loss for a batch of single
              source and single estimates (tensors won't have the
              source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'w_src'``: `loss_func` computes the loss for a given
              permutations of the sources and estimates. Output shape :
              :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_wsrc_loss`.

            In terms of efficiency, ``'w_src'`` is the least efficicient
            however easier for the user.

    For each of these modes, the best permutation and reordering will be
    automatically computed.

    """

    def __init__(self, loss_func, mode='pairwise'):
        super().__init__()
        self.loss_func = loss_func
        self.mode = mode
        if self.mode not in ["pairwise", "wo_src", "w_src"]:
            raise ValueError('Unsupported loss function type for now. Expected'
                             'one of [`pairwise`, `wo_src`, `w_src`]')

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """ Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
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
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.mode == 'wo_src':
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets,
                                           targets, **kwargs)
        elif self.mode == 'w_src':
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, min_loss_idx = self.best_perm_from_wsrc_loss(
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
        pair_wise_losses = torch.empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_wsrc_loss(loss_func, est_targets, targets, **kwargs):
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
        loss_set = torch.stack([loss_func(est_targets[:, perm],
                                          targets,
                                          **kwargs) for perm in perms],
                               dim=1)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx[:, 0]

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src):
        """Find the best permutation, given the pair-wise losses.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape [batch, n_src, n_src]. Pairwise losses.
            n_src (int): Number of sources.

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
        pwl = pair_wise_losses.transpose(-1, -2)
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
