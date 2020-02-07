import pytest
import itertools
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper, pairwise_mse


def bad_loss_func_ndim0(y_pred, y_true):
    return torch.randn(1).mean()


def bad_loss_func_ndim1(y_pred, y_true):
    return torch.randn(1)


def good_batch_loss_func(y_pred, y_true):
    batch, *_ = y_true.shape
    return torch.randn(batch)


def good_pairwise_loss_func(y_pred, y_true):
    batch, n_src, *_ = y_true.shape
    return torch.randn(batch, n_src, n_src)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("time", [16000, 32000, 1221])
def test_wrapper(batch_size, n_src, time):
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)
    for bad_loss_func in [bad_loss_func_ndim0, bad_loss_func_ndim1]:
        loss = PITLossWrapper(bad_loss_func)
        with pytest.raises(AssertionError):
            loss(est_targets, targets)
    # wo_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, mode='wo_src')
    loss_value_no_return = loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # pairwise loss function / With and without return estimates
    loss = PITLossWrapper(good_pairwise_loss_func, mode='pairwise')
    loss_value_no_return = loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # w_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, mode='w_src')
    loss_value_no_return = loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape


@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2])) +
                                 list(itertools.permutations([0, 1, 2, 3])) +
                                 list(itertools.permutations([0, 1, 2, 3, 4])))
def test_permutation(perm):
    """ Construct fake target/estimates pair. Check the value and reordering."""
    n_src = len(perm)
    perm_tensor = torch.Tensor(perm)
    source_base = torch.ones(1, n_src, 10)
    sources = torch.arange(n_src).unsqueeze(-1) * source_base
    est_sources = perm_tensor.unsqueeze(-1) * source_base

    loss_func = PITLossWrapper(pairwise_mse)
    loss_value, reordered = loss_func(est_sources, sources, return_est=True)

    assert loss_value.item() == 0
    assert_allclose(sources, reordered)


