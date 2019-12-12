"""
| Package-level utils.
| @author : Manuel Pariente, Inria-Nancy

| Modified and extended from `<https://github.com/kaituoxu/Conv-TasNet/>`__
| MIT Copyright (c) 2018 Kaituo XU
| See also `<https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
"""

from itertools import permutations
import torch
from ..utils import has_arg
EPS = 1e-8


class PITLossContainer(object):
    """Permutation invariant loss container. The loss function

    Args:
        loss_func (callable): The loss function to compute after the forward.
            The loss function is expected to output the pairwise losses
            (a :class:`torch.Tensor` of shape [batch, n_src, n_src]).
        n_src (int): Number of output sources in the network.

    Attributes:
        loss_func (callable)
        n_src (int)
    """
    def __init__(self, loss_func, n_src):
        self.loss_func = loss_func  # set up retrieving from string as well
        self.n_src = n_src
        self.loss_kwargs = None

    def compute(self, targets, est_targets, infos=None, return_est=False):
        """Compute the minimum loss to be back-propagated.

        Return reordered estimate when `return_est` option is set to ``True``.

        Args:
            targets (:class:`torch.Tensor`): Training targets.
            est_targets (:class:`torch.Tensor`): Targets estimated by the
                network.
            infos (dict): Optional dictionary containing keyword arguments for
                the loss computation.
            return_est (bool): Whether to return the reordered estimates.

        Returns:
            - The loss to be back-propagated.
            - The reordered estimates (torch.Tensor [batch, n_src, time]).
        """
        pair_wise_losses = self.loss_func(targets, est_targets,
                                          **self.get_infos_subdict(infos))
        min_loss, min_loss_idx = find_best_perm(pair_wise_losses, self.n_src)
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered_est_sources = reorder_source(est_targets, self.n_src,
                                               min_loss_idx)
        return mean_loss, reordered_est_sources

    def get_infos_subdict(self, infos):
        """Get a sub-dictionary from `infos`.

        The retrieved sub-dictionaty contains only the key-value pairs
        accepted by the loss function :attr:`loss_func`.

        Args:
            infos (dict): Additional information for loss computation.

        Returns:
            dict: Sub-dict of infos accepted by :attr:`loss_func`.
        """
        self.get_loss_func_args(infos)
        return {key: infos[key] for key in self.loss_kwargs}

    def get_loss_func_args(self, infos):
        """Get the list of accepted kwargs by `loss_func` from `infos`.

        Assumes that `infos` will have the same keys from one call to another.

        Args:
            infos (dict): Additional information for loss computation.

        Returns:
            list[str]: Keywords from `infos` accepted by :attr:`loss_func`.
        """
        # First pass
        if self.loss_kwargs is None:
            if infos is None:
                self.loss_kwargs = []
            else:
                self.loss_kwargs = [key for key in infos.keys() if
                                    has_arg(self.loss_func, key)]
                print('First pass key filtering:')
                print('\tInput keys in infos : ', list(infos.keys()))
                print('\tKeys which will be passed to the loss : ',
                      self.loss_kwargs)


def pairwise_neg_scaleawaresdr(source, est_source):

    """Calculate pair-wise negative Scale Aware SDR as proposed in [1].

        Args:
            source (:class:`torch.Tensor`): Tensor of shape [batch, n_src, time].
                The target sources.
            est_source (:class:`torch.Tensor`): Tensor of shape
                [batch, n_src, time]. Estimates of the target sources.

        Returns:
            :class:`torch.Tensor`:
                Tensor of shape [batch, n_src, n_src]. Pair-wise losses.

    [1] Roux, Jonathan Le, et al. "SDR-half-baked or well done?." arXiv preprint arXiv:1811.02508 (2018).
    """

    assert source.size() == est_source.size()

    # zero mean
    mean_source = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_source, dim=2, keepdim=True)
    source = source - mean_source
    est_source = est_source - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    s_target = torch.unsqueeze(source, dim=1)
    s_estimate = torch.unsqueeze(est_source, dim=2)

    # compute alpha
    scale_factor =  torch.sum(s_estimate * s_target, dim=3, keepdim=True)
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
    alpha = (scale_factor / (s_target_energy + EPS))

    # compute snr
    s_target = s_target.repeat(1, s_target.shape[2], 1, 1)
    e_noise = s_estimate - s_target


    # [batch, n_src, n_src]
    pair_wise_snr = torch.sum(s_target ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_snr = 10 * torch.log10(pair_wise_snr + EPS)

    sdsdr = pair_wise_snr + 10*torch.log10(alpha**2).squeeze(-1)

    return -torch.min(torch.stack((pair_wise_snr, sdsdr)), dim=0)[0]


def pairwise_neg_sdsdr(source, est_source):
    """Calculate pair-wise negative SD-SDR as proposed in [1].

            Args:
                source (:class:`torch.Tensor`): Tensor of shape [batch, n_src, time].
                    The target sources.
                est_source (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_src, time]. Estimates of the target sources.

            Returns:
                :class:`torch.Tensor`:
                    Tensor of shape [batch, n_src, n_src]. Pair-wise losses.

    [1] Roux, Jonathan Le, et al. "SDR-half-baked or well done?." arXiv preprint arXiv:1811.02508 (2018).
    """
    assert source.size() == est_source.size()

    # zero mean
    mean_source = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_source, dim=2, keepdim=True)
    source = source - mean_source
    est_source = est_source - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    s_target = torch.unsqueeze(source, dim=1)
    s_estimate = torch.unsqueeze(est_source, dim=2)

    # compute alpha
    scale_factor = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
    pair_wise_proj = scale_factor * s_target / s_target_energy

    # compute snr
    e_noise = s_estimate - s_target.repeat(1, s_target.shape[2], 1, 1)

    # [batch, n_src, n_src]
    pair_wise_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + EPS)

    return -10 * torch.log10(pair_wise_snr + EPS)


def pairwise_neg_sisdr(source, est_source, scale_invariant=True):
    """Calculate pair-wise negative SI-SDR.

    Args:
        source (:class:`torch.Tensor`): Tensor of shape [batch, n_src, time].
            The target sources.
        est_source (:class:`torch.Tensor`): Tensor of shape
            [batch, n_src, time]. Estimates of the target sources.
        scale_invariant (bool): Whether to rescale the estimated sources to
            the targets. If False this loss function will be the plain SNR.

    Returns:
        :class:`torch.Tensor`:
            Tensor of shape [batch, n_src, n_src]. Pair-wise losses.
    """
    assert source.size() == est_source.size()
    # if scale_invariant:
    # Step 1. Zero-mean norm
    mean_source = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_source, dim=2, keepdim=True)
    source = source - mean_source
    est_source = est_source - mean_estimate
    # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
    s_target = torch.unsqueeze(source, dim=1)
    s_estimate = torch.unsqueeze(est_source, dim=2)
    if scale_invariant:
        # [batch, n_src, n_src, 1]
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
        # [batch, 1, n_src, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
        # [batch, n_src, n_src, time]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy
    else:
        # [batch, n_src, n_src, time]
        pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
    e_noise = s_estimate - pair_wise_proj
    # [batch, n_src, n_src]
    pair_wise_si_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
        torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_losses = - 10 * torch.log10(pair_wise_si_sdr + EPS)
    return pair_wise_losses


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
    """
    pwl = pair_wise_losses
    perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
    # one-hot, [n_src!, n_src, n_src]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, index, 1)
    # Loss sum of each permutation
    loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
    # Indexes and values of min losses for each batch element
    min_loss_idx = torch.argmin(loss_set, dim=1)
    min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
    min_loss /= n_src
    return min_loss, min_loss_idx


def reorder_source(source, n_src, min_loss_idx):
    """ Reorder sources according to the best permutation.

    Args:
        source torch.Tensor): Tensor of shape [batch, n_src, time]
        n_src (int): Number of sources.
        min_loss_idx (torch.LongTensor): Tensor of shape [batch],
            each item is in [0, n_src!).

    Returns:
        :class:`torch.Tensor`:
            Reordered sources of shape [batch, n_src, time].
    """
    perms = source.new_tensor(list(permutations(range(n_src))),
                              dtype=torch.long)
    # Reorder estimate source according the best permutation
    min_loss_perm = torch.index_select(perms, dim=0, index=min_loss_idx)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reordered_sources = torch.zeros_like(source)
    for b in range(source.shape[0]):
        for c in range(n_src):
            reordered_sources[b, c] = source[b, min_loss_perm[b][c]]
    return reordered_sources
